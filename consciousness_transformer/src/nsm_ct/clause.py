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


def extract_clauses(tree: ParseTree) -> List[Clause]:
    """Pull clauses out of a parse tree (curriculum-scoped: SUBJECT + PP place)."""
    clauses: List[Clause] = []
    for node in tree.iter_preorder():
        if node.label not in ("CLAUSE", "PREDICATE", "VERBAL"):
            continue
        if not node.token or node.token in _PUNCT:
            continue
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
        clauses.append(Clause(predicate=node.token, args=args, head=node))
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
