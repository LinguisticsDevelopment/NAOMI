"""Clauses as the model's unit — token-free TPR assembly + cross-clause memory.

The unit of thought is the **clause** (predicate + arguments), not the token. A
clause is assembled into a single Tensor-Product matrix with **no token
embeddings anywhere**: the only atomic vectors are the NSM **primes** (via each
content word's explication → :func:`nsm_ct.tpr.TPRCodec.encode_matrix`) and
**entity-variable** atoms (a fresh vector per referent — NSM's "someone X", NOT
decomposed). Clauses are correlated across a discourse through shared
entity-variables in an order-3 ``entity⊗relation⊗value`` memory (the TPR-RNN
shape; Schlag & Schmidhuber 2018).

Prototype: numpy, deterministic, not wired into the model. See
``scripts/probe_clause_tpr.py`` and RESEARCH_NOTES §0g.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_structures import ParseNode, ParseTree
from .episode import _NAMES
from .tpr import TPRCodec

# Entities are *variables* (referents), not meanings: explicit names + pronouns.
_ENTITY_NAMES = {n.lower() for n in _NAMES}
_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they",
             "him", "her", "them", "us", "me"}
_PUNCT = set(".?!,;:")
# A person's place: treat both locative ("in") and directional ("to") as PLACE,
# so "is in the kitchen" then "went to the office" update the same slot.
_PREP_RELATION = {"in": "PLACE", "on": "PLACE", "at": "PLACE", "inside": "PLACE",
                  "to": "PLACE", "into": "PLACE", "from": "SOURCE"}


@dataclass
class Clause:
    """A predicate plus its (relation, argument-node) pairs."""

    predicate: str
    args: List[Tuple[str, ParseNode]]   # (relation, arg_node)
    head: ParseNode


def _clause_from_node(node: ParseNode) -> Clause:
    """Build one :class:`Clause` from a predicate-bearing parse node.

    The per-node argument logic, factored out of :func:`extract_clauses` so the
    discourse path can reuse it. Output is byte-identical to the old inline loop.
    """
    args: List[Tuple[str, ParseNode]] = []
    for ch in node.children:
        tok = (ch.token or "").lower()
        if ch.relation in ("SUBJECT", "OBJECT", "INDIRECT_OBJECT") and tok and tok not in _PUNCT:
            args.append((ch.relation, ch))
        elif ch.label.startswith("PP") or ch.relation == "MODIFICATION":
            # preposition node: its PREPOSITION child is the object.
            obj = next((g for g in ch.children
                        if g.relation == "PREPOSITION" and g.token), None)
            if obj is not None:
                rel = _PREP_RELATION.get((ch.token or "").lower(),
                                         (ch.token or "PREP").upper())
                args.append((rel, obj))
    return Clause(predicate=node.token, args=args, head=node)


def extract_clauses(tree: ParseTree) -> List[Clause]:
    """Pull clauses out of a parse tree (curriculum-scoped: SUBJECT + PP place)."""
    clauses: List[Clause] = []
    for node in tree.iter_preorder():
        if node.label not in ("CLAUSE", "PREDICATE", "VERBAL"):
            continue
        if not node.token or node.token in _PUNCT:
            continue
        clauses.append(_clause_from_node(node))
    return clauses


def is_entity(word: str) -> bool:
    """True for referents (names, pronouns) — these become variables, not meanings."""
    w = (word or "").lower()
    return w in _ENTITY_NAMES or w in _PRONOUNS


class EntityTracker:
    """Minimal coreference by recency: names → themselves; pronouns → last entity.

    The honest stand-in for real coreference (the genuine hard part); it is enough
    to demonstrate cross-clause correlation on explicit-entity discourses.
    """

    def __init__(self) -> None:
        self._recent: List[str] = []

    def resolve(self, word: str) -> str:
        w = (word or "").lower()
        if w in _ENTITY_NAMES:
            if w in self._recent:
                self._recent.remove(w)
            self._recent.append(w)
            return w
        if w in _PRONOUNS and self._recent:
            return self._recent[-1]          # nearest antecedent
        return w


def _content_vec(codec: TPRCodec, resolver, word: str) -> np.ndarray:
    """A content word's meaning as a fixed vector: contract(TPR(explication))."""
    tree = resolver.resolve(word)
    return codec.contract(codec.encode_matrix(tree.root))


def clause_tpr(
    clause: Clause, codec: TPRCodec, resolver, tracker: Optional[EntityTracker] = None
) -> Tuple[np.ndarray, List[Tuple[str, str, np.ndarray]]]:
    """Assemble a clause into one d×d TPR matrix — token-free.

    Returns ``(matrix, triples)`` where ``triples`` are
    ``(entity_name, relation, value_vec)`` for the cross-clause memory. Fillers are
    ONLY prime-composed content vectors or entity-variable atoms — no token vectors.
    """
    tracker = tracker or EntityTracker()
    # predicate's own meaning (e.g. "is"/"went") on the reserved self-role.
    m = codec.bind(codec.self_role, _content_vec(codec, resolver, clause.predicate))

    subject: Optional[str] = None
    triples: List[Tuple[str, str, np.ndarray]] = []
    for pos, (relation, arg) in enumerate(clause.args):
        word = arg.token
        if is_entity(word):
            name = tracker.resolve(word)
            filler = codec.filler_vec("var:" + name)   # a VARIABLE atom (not decomposed)
            if relation == "SUBJECT":
                subject = name
        else:
            filler = _content_vec(codec, resolver, word)
        m = m + codec.bind(codec.role_vec(pos, relation), filler)
        if relation not in ("SUBJECT",):
            triples.append((None, relation, filler))   # entity filled in below

    # bind the discourse triples to the clause's subject entity
    triples = [(subject, rel, val) for (_, rel, val) in triples if subject is not None]
    return m, triples


def decode_clause(matrix: np.ndarray, clause: Clause, codec: TPRCodec, resolver) -> Dict[str, object]:
    """Recover each argument from the clause matrix (fidelity readout)."""
    entity_book = {n: codec.filler_vec("var:" + n) for n in _ENTITY_NAMES}
    out: Dict[str, object] = {}
    for pos, (relation, arg) in enumerate(clause.args):
        u = codec.unbind(matrix, codec.role_vec(pos, relation))
        if is_entity(arg.token):
            # cleanup against the entity codebook
            best, score = None, -1.0
            un = u / (np.linalg.norm(u) + 1e-8)
            for name, vec in entity_book.items():
                s = float(un @ vec)
                if s > score:
                    best, score = name, s
            out[relation] = (best, round(score, 2))
        else:
            # content: lift the contracted vector back to a matrix, guided-decode
            mat = codec.lift(u)
            tree = resolver.resolve(arg.token)
            correct, total = codec.decode_guided(mat, tree.root)
            out[relation] = (arg.token, f"{correct}/{total} primes")
    return out


class EntityMemory:
    """Order-3 ``entity⊗relation⊗value`` TPR memory — the cross-clause substrate.

    A later write to the same (entity, relation) **updates** the binding (recency),
    so a discourse's most recent fact about an entity dominates the query.
    """

    def __init__(self, codec: TPRCodec) -> None:
        self.codec = codec
        self.M = np.zeros((codec.dim, codec.dim, codec.dim), dtype=np.float32)
        self._last: Dict[Tuple[str, str], np.ndarray] = {}

    def _ekey(self, entity: str) -> np.ndarray:
        return self.codec.filler_vec("var:" + entity)

    def _rkey(self, relation: str) -> np.ndarray:
        return self.codec.filler_vec("rel:" + relation)

    def write(self, entity: str, relation: str, value: np.ndarray) -> None:
        key = (entity, relation)
        if key in self._last:                    # overwrite, not accumulate
            self.M -= self.codec.bind3(self._ekey(entity), self._rkey(relation), self._last[key])
        self.M += self.codec.bind3(self._ekey(entity), self._rkey(relation), value)
        self._last[key] = value

    def query(self, entity: str, relation: str) -> np.ndarray:
        return self.codec.unbind3(self.M, self._ekey(entity), self._rkey(relation))


# ===========================================================================
# Discourse: coordinators RELATE lossless clauses; "store as OR, decide truth"
# ===========================================================================
#
# The parser emits the logical structure as typed edges (COORDINATION between
# coordinated elements and their coordinator; MODIFIER 'not'; SUBORDINATION), but
# the *tree* view (quantum_adapter.hypothesis_to_tree) drops it — coordinated
# elements point UP to their coordinator, so they are unreachable from the root.
# We therefore read discourse from the flat HypGraph (quantum_adapter.HypGraph).
# A coordinator becomes a *relation between clauses*: each disjunct is kept as its
# OWN lossless clause matrix (nothing summed across them), related through the
# coordinator's NSM atom. NSM has no AND/OR exponent — disjunction *is* MAYBE.

# coordinator surface token -> structural label
_COORDINATOR_LABEL = {"or": "OR", "but": "BUT", "and": "AND"}
# structural label -> the NSM logical prime that grounds it (None = no atom)
_COORD_PRIME = {"OR": "MAYBE", "BECAUSE": "BECAUSE", "IF": "IF", "WHEN": "WHEN",
                "NOT": "NOT", "AND": None, "BUT": None}

# Truth adjectives an "overwrite but don't forget" tag puts on a clause. Genuine
# NSM atoms (TRUE; MAYBE). A FALSE-tagged disjunct stays losslessly recoverable.
_TRUTH_VALUES = ("TRUE", "FALSE", "MAYBE")
# Reserved role positions (OTHER family) for the truth / connective tags. Clause
# arguments occupy OTHER position 1 (the place); 0 and 2 are free, so unbinding
# the tag stays exact (orthonormal Q columns).
_TRUTH_POS = 0
_CONNECTIVE_POS = 2


@dataclass
class DiscourseLink:
    """A coordinator relating two clauses (clause ``i`` ⟷ clause ``j``)."""

    coordinator: str            # OR / AND / BUT / BECAUSE / IF / NOT
    prime: Optional[str]        # the NSM atom (MAYBE for OR, …) or None
    i: int
    j: int


def _syn(label: str, token: Optional[str], relation: Optional[str] = None) -> ParseNode:
    """A synthetic parse node (the graph carries indices/tokens, not ParseNodes)."""
    return ParseNode(label=label, token=token, relation=relation)


def _relation_for(graph, idx: int) -> str:
    """Relation under which node ``idx`` attaches (preposition → PLACE, else OBJECT…)."""
    for p, c in graph.edges_of("PREPOSITION"):
        if c == idx:
            return _PREP_RELATION.get((graph.token(p) or "").lower(), "PLACE")
    for t, _p, c in graph.edges:
        if c == idx and t in ("OBJECT", "INDIRECT_OBJECT", "SUBJECT"):
            return t
    return "PLACE"


def _subject_predicate(graph):
    """The first clause's (subject token, predicate token, clause index)."""
    subs = graph.edges_of("SUBJECT")
    if not subs:
        return None, None, None
    clause_idx, subj_idx = subs[0]
    return graph.token(subj_idx), graph.token(clause_idx), clause_idx


def _fact_clause(graph, subj: str, pred: Optional[str], rel: str, val_idx: int) -> Clause:
    val = _syn(graph.label(val_idx) or "NOUN", graph.token(val_idx), rel)
    return Clause(predicate=pred or "is",
                  args=[("SUBJECT", _syn("NOMINAL", subj, "SUBJECT")), (rel, val)],
                  head=_syn("CLAUSE", pred))


def extract_discourse(graph) -> Tuple[List[Clause], List[DiscourseLink]]:
    """Pull clauses + their coordinating links from a flat hypothesis graph.

    Handles the three shapes the curriculum needs: **coordination** (``A or B``:
    one clause per coordinated value, related by an OR/MAYBE link), **negation**
    (``not in A``: one clause tagged with a NOT link), and a plain single fact
    (degrades to one clause, no links). Curriculum-scoped, like
    :func:`extract_clauses`; conditionals (SUBORDINATION) are not handled yet.
    """
    if graph is None or not getattr(graph, "nodes", None):
        return [], []
    subj, pred, _clause_idx = _subject_predicate(graph)

    # -- coordination (A or B): one lossless clause per coordinated value --------
    coord_edges = graph.edges_of("COORDINATION")
    if coord_edges and subj is not None:
        by_coord: Dict[int, List[int]] = {}
        for parent, child in coord_edges:
            by_coord.setdefault(child, []).append(parent)
        coord_idx, elements = max(by_coord.items(), key=lambda kv: len(kv[1]))
        coordinator = _COORDINATOR_LABEL.get((graph.token(coord_idx) or "").lower(), "AND")
        rel = _relation_for(graph, coord_idx)
        elements = sorted(set(elements))
        clauses = [_fact_clause(graph, subj, pred, rel, el) for el in elements]
        prime = _COORD_PRIME.get(coordinator)
        links = [DiscourseLink(coordinator, prime, 0, j) for j in range(1, len(clauses))]
        return clauses, links

    # -- negation (not in A): one clause, NOT link -------------------------------
    neg = any(lab == "MODIFIER" and (tok or "").lower() in ("not", "never")
              for _i, lab, tok in graph.nodes)
    prep_edges = graph.edges_of("PREPOSITION")
    if neg and subj is not None and prep_edges:
        prep_node, val_idx = prep_edges[0]
        rel = _PREP_RELATION.get((graph.token(prep_node) or "").lower(), "PLACE")
        return [_fact_clause(graph, subj, pred, rel, val_idx)], [DiscourseLink("NOT", "NOT", 0, 0)]

    # -- plain single fact -------------------------------------------------------
    if subj is not None and prep_edges:
        prep_node, val_idx = prep_edges[0]
        rel = _PREP_RELATION.get((graph.token(prep_node) or "").lower(),
                                 (graph.token(prep_node) or "PREP").upper())
        return [_fact_clause(graph, subj, pred, rel, val_idx)], []
    return [], []


# -- truth-tagging (non-destructive "overwrite but don't forget") ---------------
def truth_book(codec: TPRCodec) -> Dict[str, np.ndarray]:
    """The 3-atom truth codebook {TRUE, FALSE, MAYBE} (local; not tpr's codebook)."""
    return {v: codec.filler_vec(v) for v in _TRUTH_VALUES}


def tag_truth(matrix: np.ndarray, value: str, codec: TPRCodec) -> np.ndarray:
    """Bind a TRUE/FALSE/MAYBE adjective onto a clause matrix (additive, lossless)."""
    return matrix + codec.bind(codec.role_vec(_TRUTH_POS, "TRUTH"), codec.filler_vec(value))


def read_truth(matrix: np.ndarray, codec: TPRCodec) -> Tuple[Optional[str], float]:
    """Recover the truth adjective from a (possibly tagged) clause matrix."""
    u = codec.unbind(matrix, codec.role_vec(_TRUTH_POS, "TRUTH"))
    n = float(np.linalg.norm(u))
    if n < 1e-8:
        return None, 0.0
    un = u / n
    book = truth_book(codec)
    best, score = None, -1.0
    for name, vec in book.items():
        s = float(un @ vec)
        if s > score:
            best, score = name, s
    return best, score


@dataclass
class DiscourseTPR:
    """Lossless coordination: distinct clause matrices + a relating edges tensor."""

    clauses: List[np.ndarray]                         # lossless d×d, one per disjunct
    clause_triples: List[List[Tuple[str, str, np.ndarray]]]
    edges: np.ndarray                                 # Σ role(i,COORD)⊗contract(clauses[j]) + connective
    link_index: List[Tuple[int, int]]
    coordinator: str
    prime: Optional[str]

    def recover_link(self, codec: TPRCodec, i: int) -> int:
        """Recover the clause index j related to clause i (unbind + nearest)."""
        key = codec.unbind(self.edges, codec.role_vec(i, "COORDINATION"))
        cands = [codec.contract(m) for m in self.clauses]
        kn = key / (np.linalg.norm(key) + 1e-8)
        sims = [float(kn @ (c / (np.linalg.norm(c) + 1e-8))) for c in cands]
        return int(np.argmax(sims))


def build_discourse_tpr(
    clauses: List[Clause], links: List[DiscourseLink], codec: TPRCodec, resolver,
    tracker: Optional[EntityTracker] = None,
) -> DiscourseTPR:
    """Assemble lossless per-clause matrices and the coordinator edges tensor."""
    mats: List[np.ndarray] = []
    triples: List[List[Tuple[str, str, np.ndarray]]] = []
    for cl in clauses:
        m, tr = clause_tpr(cl, codec, resolver, tracker)
        mats.append(m)
        triples.append(tr)
    edges = np.zeros((codec.dim, codec.dim), dtype=np.float32)
    link_index: List[Tuple[int, int]] = []
    for link in links:
        if 0 <= link.j < len(mats):
            edges = edges + codec.bind(codec.role_vec(link.i, "COORDINATION"),
                                       codec.contract(mats[link.j]))
            link_index.append((link.i, link.j))
    coordinator = links[0].coordinator if links else "NONE"
    prime = links[0].prime if links else None
    if prime:  # the connective atom (e.g. MAYBE) is itself readable on the edges
        edges = edges + codec.bind(codec.role_vec(_CONNECTIVE_POS, "CONNECTIVE"),
                                   codec.filler_vec(prime))
    return DiscourseTPR(mats, triples, edges, link_index, coordinator, prime)


def read_connective(dtpr: DiscourseTPR, codec: TPRCodec) -> Tuple[Optional[str], float]:
    """Recover the connective atom (e.g. MAYBE) bound on the edges tensor."""
    u = codec.unbind(dtpr.edges, codec.role_vec(_CONNECTIVE_POS, "CONNECTIVE"))
    return codec.cleanup(u)


class DisjunctionBuffer:
    """Stores a disjunction as several lossless, truth-tagged clause matrices.

    "Store as OR, then decide truth": every disjunct starts tagged ``MAYBE``;
    evidence (e.g. a negation) re-tags the refuted disjunct ``FALSE`` and, when one
    candidate remains, the survivor ``TRUE`` — **without deleting any matrix** (the
    losing disjunct stays losslessly recoverable, now carrying a FALSE adjective).
    A query returns the first-class ``MAYBE`` answer until exactly one disjunct is
    proven TRUE.
    """

    def __init__(self, codec: TPRCodec) -> None:
        self.codec = codec
        self.group: List[Dict[str, object]] = []   # one dict per disjunct

    def store_disjunction(self, dtpr: DiscourseTPR) -> None:
        self.group = []
        for base, tr in zip(dtpr.clauses, dtpr.clause_triples):
            subj, rel, val = (tr[0] if tr else (None, "PLACE", None))
            self.group.append({
                "base": base,                                   # untagged clause matrix
                "matrix": tag_truth(base, "MAYBE", self.codec),  # tagged copy
                "subject": subj, "relation": rel, "value": val,  # value = content vec
                "truth": "MAYBE",
            })

    def _retag(self, entry: Dict[str, object], value: str) -> None:
        entry["truth"] = value
        entry["matrix"] = tag_truth(entry["base"], value, self.codec)  # fresh tag, no pile-up

    def decide_truth(self, refuted_value: np.ndarray) -> None:
        """Evidence that ``refuted_value`` is FALSE — re-tag it, survivor TRUE."""
        if not self.group:
            return
        def cos(a, b):
            return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))
        # the disjunct whose value best matches the refuted place
        idx = max(range(len(self.group)),
                  key=lambda k: cos(self.group[k]["value"], refuted_value))
        self._retag(self.group[idx], "FALSE")
        survivors = [e for e in self.group if e["truth"] != "FALSE"]
        if len(survivors) == 1:
            self._retag(survivors[0], "TRUE")

    def query(self) -> Tuple[str, np.ndarray]:
        """('RESOLVED', value_vec) once one disjunct is TRUE; else ('MAYBE', MAYBE atom)."""
        trues = [e for e in self.group if e["truth"] == "TRUE"]
        if len(trues) == 1:
            return "RESOLVED", trues[0]["value"]
        return "MAYBE", self.codec.filler_vec("MAYBE")
