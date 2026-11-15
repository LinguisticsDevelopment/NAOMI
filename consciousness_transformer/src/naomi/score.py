"""Tree scorers: grade a tree as a percentage WITH REASONS (the reasons are the training feedback).

Nothing here encodes a language: no word lists, no literals. Meaning is read from USVS (the
sense vectors over the named axes); names (kind == "name", numerals included) have no meaning point
and match by identity of the token (each distinct name is its own orthogonal unit dimension; a word
with no sense is a unit dimension keyed by its lemma).

(1) :func:`score_coverage` -- reference-free. How much of the tree's own sentence is accounted for.
    percent = 100 * (0.6 * attached + 0.2 * grounded + 0.2 * structure)
      attached  share of non-punctuation tokens that are attached (rel != UNATTACHED and not under an
                unattached component), also reported per part-of-speech tag
      grounded  share of ``word`` nodes with at least one candidate (or a committed) sense; 1.0 if none
      structure 1.0 when there is exactly one ROOT node and every token index occurs once, else 0.0

(2) :func:`score_pair` -- reference vs candidate. Two numbers and their combination (owner, 2026-11-15):
    scoring only the flattened meaning would let bad tree-building go unnoticed; raw shape alone is
    unfair to "not happy" vs "unhappy" (an operator plus a descriptor vs one word).

    OPERATOR FOLDING (applied to both trees first). A node is a foldable operator when (a) its chosen
    sense is a closed-class lexicon sense (id prefix ``fn.``), (b) it is a leaf, and (c) it hangs on a
    real head by one of FOLD_RELATIONS (the grammar's own modifier edge types; argument and linking
    relations are excluded). Folding removes it from the skeleton and adds its USVS vector to its
    head's vector. Function words WITH dependents (a preposition and its object, a connector and its
    clause) stay as nodes: they are structure. FOLD_ONTO_ROOT decides whether the head may be the
    tree's ROOT node (an operator on the main verb).

    ALIGNMENT (one-to-one, deterministic, on the folded skeletons). The two ROOT nodes are matched.
    The reference is then taken level by level (shallowest first). At each level every (reference
    node, unmatched candidate node) pair gets  0.6*meaning + 0.2*[same relation to head]
    + 0.2*[heads already matched to each other]; pairs are taken greedily by descending score (ties:
    higher meaning, then pre-order position in reference, then in candidate) and only if the score is
    at least MATCH_MIN (meaning here is the base-sense similarity, without folded operators). Sibling order and repeated relations therefore do not matter. Token positions
    are never used.

    MEANING % = flattened similarity. A tree flattens to  sum over nodes of (role vector of the node's
    relation path from the root) (x) (the node's USVS vector, folded operators included); two trees
    compare by the cosine of the flattened forms. With orthogonal role vectors per path this is
        sum over aligned pairs on the SAME path (heads aligned and equal relation) of dot(v_ref, v_cand)
        / ( sqrt(sum |v_ref|^2) * sqrt(sum |v_cand|^2) )
    over all skeleton nodes (nodes with no partner count only in the norms). A recorded antonym pair
    (sense-level WordNet pointers, tiers ``antonym_tiers``; word store optional) has its dot set to 0 and gives reason opposite_meaning.
    SHAPE % = attachment F1 of the folded skeletons: 2C / (N_ref + N_cand), C = aligned nodes whose
    heads are aligned to each other AND whose relation label is equal.
    OVERALL = W_MEANING * meaning + W_SHAPE * shape (simple mean).

    Node meaning: a ``word`` node is the vector of its committed ``sense`` if set, else of its FIRST
    candidate sense (``best_candidate=True`` instead takes the best-matching candidate pair).
    Node-level figures (recall / precision / F1 / attachment) are kept for the skeleton, overall and
    for function-word and content-word nodes separately (None when neither tree has such a node).
    Reasons (``PairScore.reasons``, a list of dicts): node_dropped (function_node_dropped=True and the
    named dimensions when the node is a lexicon-sense node), node_extra, meaning_shift (top differing
    dimensions of the base senses, signed candidate minus reference), opposite_meaning (with tier),
    operator_folded_differs (differing folded-operator dimensions), name_mismatch, relation_changed,
    head_changed, subtree_moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .reader import _PUNCT_POS, FN_PREFIX
from .tree import walk
from .types import KIND_NAME, REL_ROOT, REL_UNATTACHED, Node, Tree

try:
    from nsm_ct.ground.usvs import ANTONYM_TIERS, DEFAULT_ANTONYM_TIERS
except Exception:  # pragma: no cover
    ANTONYM_TIERS = DEFAULT_ANTONYM_TIERS = ()

W_COV_ATTACHED, W_COV_GROUNDED, W_COV_STRUCTURE = 0.6, 0.2, 0.2
W_ALIGN_MEANING, W_ALIGN_RELATION, W_ALIGN_HEAD = 0.6, 0.2, 0.2
W_OVERALL_NODE_F1, W_OVERALL_ATTACHMENT = 0.5, 0.5
W_MEANING, W_SHAPE = 0.5, 0.5                      # overall = W_MEANING * meaning% + W_SHAPE * shape%
# The grammar's own modifier edge types: a leaf function word hanging on a head by one of these folds
# into the head. Argument relations (SUBJECT, OBJECT, INDIRECT_OBJECT, COMPLEMENT) and linking relations
# (PREPOSITION, SUBORDINATION, COORDINATION) never fold.
FOLD_RELATIONS = ("SPECIFICATION", "MODIFICATION", "DESCRIPTION")
FOLD_ONTO_ROOT = True                              # an operator on the ROOT node (main verb) folds too
MATCH_MIN = 0.3
TOP_DIMS = 5
_EPS = 1e-6


# ----------------------------------------------------------------------------- coverage
@dataclass
class Coverage:
    percent: float
    attached: float                                   # fraction in [0, 1]
    attached_by_pos: Dict[str, Tuple[int, int]]       # pos -> (attached, total)
    grounded: float                                   # fraction of word nodes with a sense candidate
    structure_ok: bool
    structure_problems: List[str] = field(default_factory=list)
    diagnostics: List[dict] = field(default_factory=list)


def _real_nodes(tree: Tree):
    """(node, attached?) for every non-punctuation token node; attached = no UNATTACHED on the path."""
    out = []
    stack = [(tree.root, True)]
    while stack:
        n, ok = stack.pop()
        ok = ok and n.rel != REL_UNATTACHED
        if n.token_index is not None and n.pos != _PUNCT_POS:
            out.append((n, ok))
        for c in n.children:
            stack.append((c, ok))
    return out


def score_coverage(tree: Tree) -> Coverage:
    real = _real_nodes(tree)
    by_pos: Dict[str, List[int]] = {}
    for n, ok in real:
        a = by_pos.setdefault(n.pos, [0, 0])
        a[0] += int(ok)
        a[1] += 1
    total = len(real)
    attached = sum(int(ok) for _, ok in real) / total if total else 1.0
    words = [n for n, _ in real if n.kind != KIND_NAME]
    diags = list(tree.diagnostics)
    n_ground = 0
    for n in words:
        if n.sense or n.senses:
            n_ground += 1
        else:
            diags.append(dict(kind="ungrounded_word", token=n.token, index=n.token_index, pos=n.pos))
    grounded = n_ground / len(words) if words else 1.0
    problems: List[str] = []
    roots = [n for n, _, _ in walk(tree.root) if n.rel == REL_ROOT]
    if len(roots) != 1 or roots[0] is not tree.root:
        problems.append(f"{len(roots)} ROOT nodes")
    idx = [n.token_index for n, _, _ in walk(tree.root) if n.token_index is not None]
    if len(idx) != len(set(idx)):
        problems.append("a token index occurs more than once")
    ok_struct = not problems
    pct = 100.0 * (W_COV_ATTACHED * attached + W_COV_GROUNDED * grounded + W_COV_STRUCTURE * float(ok_struct))
    return Coverage(percent=pct, attached=attached, attached_by_pos={p: (a, t) for p, (a, t) in by_pos.items()},
                    grounded=grounded, structure_ok=ok_struct, structure_problems=problems, diagnostics=diags)


# ----------------------------------------------------------------------------- pair score
@dataclass
class Figures:
    node_recall: float
    node_precision: float
    node_f1: float
    attachment_f1: float
    overall: float


@dataclass
class PairScore:
    overall: float                 # W_MEANING * meaning + W_SHAPE * shape
    meaning: float                 # flattened, antonym-aware, operators folded (percent)
    shape: float                   # folded-skeleton attachment F1 (percent)
    figures: Figures               # node-by-node figures on the folded skeleton
    function_words: Optional[Figures]
    content_words: Optional[Figures]
    matches: List[Tuple[Node, Node, float]] = field(default_factory=list)   # (reference, candidate, meaning)
    reasons: List[dict] = field(default_factory=list)


@dataclass
class _M:
    """Meaning of an aligned pair of nodes."""
    sim: float = 0.0               # cosine of the (folded) node vectors, 0 for an antonym pair
    sa: Optional[str] = None
    sb: Optional[str] = None
    tier: Optional[str] = None     # antonym tier when the pair is a recorded opposite
    base: float = 0.0              # similarity of the base senses without folded operators
    dot: float = 0.0               # contribution to the flattened dot product


_ANT_CACHE: Dict = {}
_IDF_CACHE: Dict = {}


def axis_weights(usvs) -> np.ndarray:
    """Per-axis IDF weight from the sense table (closed-class lexicon senses included, so operator axes
    carried by few senses weigh a lot):  w = log((N_senses + 1) / (df + 1)), df = senses with a
    non-zero value on the axis. (No importable implementation exists: the earlier ones are archived.)"""
    k = id(usvs)
    if k not in _IDF_CACHE:
        n = len(usvs.sense_indptr) - 1
        df = np.bincount(usvs.sense_axis_idx[usvs.sense_axis_val != 0], minlength=len(usvs.axes))
        _IDF_CACHE[k] = np.log((n + 1.0) / (df + 1.0))
    return _IDF_CACHE[k]


def _wn_synset(sid: str):
    try:
        from nltk.corpus import wordnet as wn
        return wn.synset(sid)
    except Exception:
        return None


def _heads(syn):
    """WordNet's own head relation: a satellite adjective's heads are its similar-to synsets."""
    return list(syn.similar_tos()) if syn.pos() == "s" else [syn]


def sense_opposite(sa: Optional[str], sb: Optional[str], tiers) -> Optional[str]:
    """Sense-level antonymy from WordNet's own pointers. ``direct``: a lemma of one synset has an antonym
    lemma in the other. ``satellite_head``: one is an adjective satellite and the heads of the two are
    direct antonyms."""
    if not sa or not sb or sa == sb:
        return None
    a, b = _wn_synset(sa), _wn_synset(sb)
    if a is None or b is None:
        return None
    def direct(x, y):
        return any(an.synset() == y for lem in x.lemmas() for an in lem.antonyms())
    if tiers and ANTONYM_TIERS[0] in tiers and (direct(a, b) or direct(b, a)):
        return ANTONYM_TIERS[0]
    if tiers and ANTONYM_TIERS[1] in tiers and "s" in (a.pos(), b.pos()):
        if any(direct(x, y) or direct(y, x) for x in _heads(a) for y in _heads(b)):
            return ANTONYM_TIERS[1]
    return None


def _antonym_pairs(usvs, tiers) -> Dict:
    """frozenset({word, word}) -> strongest allowed tier, from the USVS word-level antonym store."""
    key = (id(usvs), tuple(tiers))
    if key not in _ANT_CACHE:
        rank = {t: i for i, t in enumerate(ANTONYM_TIERS)}
        out: Dict = {}
        for a, b, t in usvs.antonyms:
            if t in tiers:
                k = frozenset((a, b))
                if k not in out or rank.get(t, 99) < rank.get(out[k], 99):
                    out[k] = t
        _ANT_CACHE[key] = out
    return _ANT_CACHE[key]


class _Space:
    """Sense vectors over the named axes (plus a unit dimension per name / unsensed lemma), the folded
    operators of each skeleton node, and the antonym store."""

    def __init__(self, usvs, best_candidate: bool, tiers=None, word_tiers=(), idf: bool = True):
        self.u, self.best, self._v, self._r = usvs, best_candidate, {}, {}
        self.ops: Dict[int, List[Node]] = {}
        self.tiers = DEFAULT_ANTONYM_TIERS if tiers is None else tuple(tiers)
        self.ant = _antonym_pairs(usvs, tuple(word_tiers)) if word_tiers else {}
        # IDF weighting enters the dot product as w * x * y (vectors are scaled by sqrt(w));
        # a unit dimension (name / unsensed lemma) is as specific as it gets: the largest weight.
        self.w = axis_weights(usvs) if idf else np.ones(len(usvs.axes))
        self.sw = np.sqrt(self.w)
        self.unit = float(self.w.max()) ** 0.5

    def vec(self, sid: str) -> Optional[np.ndarray]:
        if sid not in self._v:
            v = self.u.sense_dense(sid)
            self._v[sid] = None if v is None else v.astype(np.float64) * self.sw
        return self._v[sid]

    def options(self, n: Node) -> List[str]:
        if n.sense:
            return [n.sense]
        return list(n.senses) if self.best else list(n.senses[:1])

    def chosen(self, n: Node) -> Optional[str]:
        return n.sense or (n.senses[0] if n.senses else None)

    def is_function(self, n: Node) -> bool:
        s = self.chosen(n)
        return bool(s) and s.startswith(FN_PREFIX)

    def rep(self, n: Node, sid: Optional[str], fold: bool = True):
        """(dense vector or None, {unit key: weight}) of a node with the given sense."""
        k = (id(n), sid, fold)
        if k not in self._r:
            base = None if (n.kind == KIND_NAME or not sid) else self.vec(sid)
            if base is None:
                self._r[k] = (None, {(n.kind[:1], n.token if n.kind == KIND_NAME else n.lemma): self.unit})
            else:
                v = base
                for o in (self.ops.get(id(n), []) if fold else []):
                    ov = self.vec(self.chosen(o))
                    if ov is not None:
                        v = v + ov
                self._r[k] = (v, {})
        return self._r[k]

    @staticmethod
    def dot(ra, rb) -> float:
        d = float(np.dot(ra[0], rb[0])) if ra[0] is not None and rb[0] is not None else 0.0
        return d + sum(w * rb[1].get(key, 0.0) for key, w in ra[1].items())

    def norm2(self, n: Node, sid: Optional[str]) -> float:
        r = self.rep(n, sid)
        return (float(np.dot(r[0], r[0])) if r[0] is not None else 0.0) + sum(w * w for w in r[1].values())

    def _cos(self, ra, rb) -> float:
        na, nb = self.dot(ra, ra), self.dot(rb, rb)
        return min(1.0, self.dot(ra, rb) / (na * nb) ** 0.5) if na > 0 and nb > 0 else 0.0

    def opposite(self, a: Node, b: Node) -> Optional[str]:
        if a.kind == KIND_NAME or b.kind == KIND_NAME:
            return None
        return self.opposite_senses(a, b)

    def opposite_senses(self, a: Node, b: Node, sa=None, sb=None) -> Optional[str]:
        sa, sb = sa or self.chosen(a), sb or self.chosen(b)
        if not (sa or sb) or a.kind == KIND_NAME or b.kind == KIND_NAME:
            return None
        t = sense_opposite(sa, sb, self.tiers)
        if t:
            return t
        for x in {a.lemma, a.token}:
            for y in {b.lemma, b.token}:
                t = self.ant.get(frozenset((x, y)))
                if t:
                    return t
        return None

    def meaning(self, a: Node, b: Node) -> _M:
        oa, ob = self.options(a), self.options(b)
        if a.kind == KIND_NAME or b.kind == KIND_NAME or not oa or not ob:
            oa, ob = [None], [None]
        best = None
        for sa in oa:
            for sb in ob:
                ra, rb = self.rep(a, sa), self.rep(b, sb)
                s = self._cos(ra, rb)
                if best is None or s > best.sim + _EPS:
                    ba, bb = self.rep(a, sa, False), self.rep(b, sb, False)
                    best = _M(sim=s, sa=sa, sb=sb, base=1.0 if sa == sb else self._cos(ba, bb), dot=self.dot(ra, rb))
        best.tier = None if a.kind == KIND_NAME or b.kind == KIND_NAME else self.opposite_senses(a, b, best.sa, best.sb)
        if best.tier:
            best.sim = best.dot = 0.0
        return best

    def dims(self, sid: Optional[str]) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self.u.sense_signature(sid).items()} if sid else {}

    def diff(self, sa: Optional[str], sb: Optional[str]) -> List[Tuple[str, float]]:
        return self._top(self.u.sense_signature(sa) if sa else {}, self.u.sense_signature(sb) if sb else {})

    def op_signature(self, n: Node) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for o in self.ops.get(id(n), []):
            for k, v in self.u.sense_signature(self.chosen(o)).items():
                out[k] = out.get(k, 0.0) + v
        return out

    def _top(self, da, db) -> List[Tuple[str, float]]:
        """Top differing dimensions, ranked by IDF-weighted absolute difference; the raw signed
        difference (candidate minus reference) is reported."""
        wt = {a: float(w) for a, w in zip(self.u.axes, self.w)}
        out = [(k, round(db.get(k, 0.0) - da.get(k, 0.0), 4)) for k in set(da) | set(db)]
        out = [x for x in out if abs(x[1]) > _EPS]
        return sorted(out, key=lambda x: (-abs(x[1]) * wt.get(x[0], 1.0), x[0]))[:TOP_DIMS]


def fold_operators(tree: Tree, sp: _Space) -> Tree:
    """Copy of the tree without its foldable operators (see the module docstring); the operators of
    each remaining node are recorded in ``sp.ops`` under the id of the copied node."""

    def rec(n: Node) -> Node:
        new = replace(n, children=[])
        for c in n.children:
            if (c.rel in FOLD_RELATIONS and not c.children and sp.is_function(c) and n.token_index is not None
                    and (FOLD_ONTO_ROOT or n.rel != REL_ROOT)):
                sp.ops.setdefault(id(new), []).append(c)
            else:
                new.children.append(rec(c))
        return new

    return replace(tree, root=rec(tree.root))


def _index(tree: Tree):
    """Real nodes in pre-order with parent and depth; the synthetic root is kept as the anchor."""
    order, parent, depth = [], {}, {}
    for n, d, p in walk(tree.root):
        order.append(n)
        parent[id(n)], depth[id(n)] = p, d
    return order, parent, depth


def _virtual(n: Optional[Node]) -> bool:
    """No real token: the synthetic anchor of a multi-root tree, or 'no head'."""
    return n is None or n.token_index is None


def _heads_aligned(r: Node, c: Node, rp, cp, match) -> bool:
    pr, pc = rp[id(r)], cp[id(c)]
    if _virtual(pr):
        return _virtual(pc)
    return (not _virtual(pc)) and match.get(id(pr)) is pc


def _subtree(n: Node) -> List[Node]:
    return [m for m, _, _ in walk(n)]


def _align(ref: Tree, cand: Tree, sp: _Space):
    ro, rp, rd = _index(ref)
    co, cp, _cd = _index(cand)
    rpos = {id(n): i for i, n in enumerate(ro)}
    cpos = {id(n): i for i, n in enumerate(co)}
    match: Dict[int, Node] = {}                              # ref id -> cand node
    taken: set = set()
    meaning: Dict[int, _M] = {}
    levels: Dict[int, List[Node]] = {}
    for n in ro:
        if not _virtual(n):
            levels.setdefault(rd[id(n)], []).append(n)
    for d in sorted(levels):
        scored = []
        for r in levels[d]:
            for c in co:
                if id(c) in taken or _virtual(c):
                    continue
                m = sp.meaning(r, c)
                am = 0.0 if m.tier else m.base      # align on the base senses: folded operators must not steer it
                s = (W_ALIGN_MEANING * am + W_ALIGN_RELATION * (r.rel == c.rel)
                     + W_ALIGN_HEAD * _heads_aligned(r, c, rp, cp, match))
                if s >= MATCH_MIN - _EPS:
                    scored.append((-s, -am, rpos[id(r)], cpos[id(c)], r, c, m))
        for _s, _m, _i, _j, r, c, m in sorted(scored, key=lambda t: t[:4]):
            if id(r) in match or id(c) in taken:
                continue
            match[id(r)], meaning[id(r)] = c, m
            taken.add(id(c))
    return ro, rp, co, cp, match, meaning


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def _figures(refs: List[Node], cands: List[Node], credit: Dict, good: set) -> Optional[Figures]:
    if not refs and not cands:
        return None
    rec = sum(credit.get(id(r), 0.0) for r in refs) / len(refs) if refs else 1.0
    prec = sum(credit.get(("c", id(c)), 0.0) for c in cands) / len(cands) if cands else 1.0
    f1 = _f1(prec, rec)
    corr_r = sum(1 for r in refs if id(r) in good)
    corr_c = sum(1 for c in cands if ("c", id(c)) in good)
    denom = len(refs) + len(cands)
    att = (corr_r + corr_c) / denom if denom else 1.0
    return Figures(100 * rec, 100 * prec, 100 * f1, 100 * att, 100 * (W_OVERALL_NODE_F1 * f1 + W_OVERALL_ATTACHMENT * att))


def score_pair(reference: Tree, candidate: Tree, usvs, best_candidate: bool = False, antonym_tiers=None,
               word_store_tiers=(), idf: bool = True) -> PairScore:
    """``antonym_tiers``: sense-level (WordNet pointer) tiers, default direct + satellite_head.
    ``word_store_tiers``: optional extra tiers from the word-level USVS store (default off: conflated).
    ``idf``: weight the axes by IDF in every vector comparison (default on)."""
    sp = _Space(usvs, best_candidate, antonym_tiers, word_store_tiers, idf)
    reference, candidate = fold_operators(reference, sp), fold_operators(candidate, sp)
    ro, rp, co, cp, match, meaning = _align(reference, candidate, sp)
    rn = [n for n in ro if not _virtual(n)]
    cn = [n for n in co if not _virtual(n)]
    cmatch = {id(c): r for r in rn if id(r) in match for c in [match[id(r)]]}
    credit: Dict = {}
    good: set = set()
    dot = n2r = n2c = 0.0
    for r in rn:
        c = match.get(id(r))
        if c is None:
            n2r += sp.norm2(r, sp.chosen(r))
            continue
        m = meaning[id(r)]
        n2r += sp.norm2(r, m.sa)
        credit[id(r)] = credit[("c", id(c))] = m.sim
        if _heads_aligned(r, c, rp, cp, match) and r.rel == c.rel:
            good.add(id(r))
            good.add(("c", id(c)))
            dot += m.dot
    for c in cn:
        r = cmatch.get(id(c))
        n2c += sp.norm2(c, meaning[id(r)].sb if r is not None else sp.chosen(c))
    if not rn and not cn:
        meaning_pct = shape_pct = 100.0
    else:
        meaning_pct = 100.0 * max(0.0, min(1.0, dot / (n2r * n2c) ** 0.5)) if n2r > 0 and n2c > 0 else 0.0
        shape_pct = 100.0 * (len(good)) / (len(rn) + len(cn))
    reasons: List[dict] = []

    def who(n: Node, side: str) -> dict:
        return dict(side=side, token=n.token, lemma=n.lemma, pos=n.pos, rel=n.rel,
                    folded_operators=[o.token for o in sp.ops.get(id(n), [])])

    for r in rn:
        c = match.get(id(r))
        if c is None:
            s = sp.chosen(r)
            rs = who(r, "reference")
            rs.update(kind="node_dropped", function_node_dropped=sp.is_function(r), dimensions=sp.dims(s) if sp.is_function(r) else {})
            reasons.append(rs)
            continue
        m = meaning[id(r)]
        if m.tier:
            reasons.append(dict(who(r, "reference"), kind="opposite_meaning", candidate_token=c.token, tier=m.tier))
        elif r.kind == KIND_NAME or c.kind == KIND_NAME:
            if m.sim < 1 - _EPS:
                reasons.append(dict(who(r, "reference"), kind="name_mismatch", reference_token=r.token, candidate_token=c.token))
        elif m.base < 1 - _EPS:
            reasons.append(dict(who(r, "reference"), kind="meaning_shift", candidate_token=c.token, similarity=round(m.base, 4),
                                reference_sense=m.sa, candidate_sense=m.sb,
                                dimensions=[dict(dimension=k, difference=v) for k, v in sp.diff(m.sa, m.sb)]))
        od = sp._top(sp.op_signature(r), sp.op_signature(c))
        if od:
            reasons.append(dict(who(r, "reference"), kind="operator_folded_differs", candidate_token=c.token,
                                reference_operators=[o.token for o in sp.ops.get(id(r), [])],
                                candidate_operators=[o.token for o in sp.ops.get(id(c), [])],
                                dimensions=[dict(dimension=k, difference=v) for k, v in od]))
        if _heads_aligned(r, c, rp, cp, match):
            if r.rel != c.rel:
                reasons.append(dict(who(r, "reference"), kind="relation_changed", reference_rel=r.rel, candidate_rel=c.rel))
        else:
            sub_r = _subtree(r)
            sub_c = {id(x) for x in _subtree(c)}
            whole = all(id(x) in match and id(match[id(x)]) in sub_c for x in sub_r) and len(sub_r) == len(sub_c)
            ph, ch = rp[id(r)], cp[id(c)]
            reasons.append(dict(who(r, "reference"), kind="subtree_moved" if whole else "head_changed",
                                reference_head=None if _virtual(ph) else ph.token, candidate_head=None if _virtual(ch) else ch.token,
                                reference_rel=r.rel, candidate_rel=c.rel))
    for c in cn:
        if id(c) not in cmatch:
            s = sp.chosen(c)
            rs = who(c, "candidate")
            rs.update(kind="node_extra", function_node_extra=sp.is_function(c), dimensions=sp.dims(s) if sp.is_function(c) else {})
            reasons.append(rs)

    allf = _figures(rn, cn, credit, good)
    fr = [n for n in rn if sp.is_function(n)]
    fc = [n for n in cn if sp.is_function(n)]
    cr = [n for n in rn if not sp.is_function(n)]
    cc = [n for n in cn if not sp.is_function(n)]
    if allf is None:
        allf = Figures(100.0, 100.0, 100.0, 100.0, 100.0)
    matches = [(r, match[id(r)], meaning[id(r)].sim) for r in rn if id(r) in match]
    return PairScore(overall=W_MEANING * meaning_pct + W_SHAPE * shape_pct, meaning=meaning_pct, shape=shape_pct,
                     figures=allf, function_words=_figures(fr, fc, credit, good),
                     content_words=_figures(cr, cc, credit, good), matches=matches, reasons=reasons)


def tree_similarity(a: Tree, b: Tree, usvs, best_candidate: bool = False, antonym_tiers=None,
                    word_store_tiers=(), idf: bool = True) -> float:
    """Symmetric: the overall percentage with a as reference averaged with b as reference."""
    return 0.5 * (score_pair(a, b, usvs, best_candidate, antonym_tiers, word_store_tiers, idf).overall
                  + score_pair(b, a, usvs, best_candidate, antonym_tiers, word_store_tiers, idf).overall)
