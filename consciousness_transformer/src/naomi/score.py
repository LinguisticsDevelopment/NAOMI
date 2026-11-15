"""Tree scorers: grade a tree as a percentage WITH REASONS (the reasons are the training feedback).

Nothing here encodes a language: no word lists, no literals. Meaning is read from USVS (the
sense vectors over the named axes); names (kind == "name", numerals included) have no meaning point
and match by identity of the token.

(1) :func:`score_coverage` -- reference-free. How much of the tree's own sentence is accounted for.
    percent = 100 * (0.6 * attached + 0.2 * grounded + 0.2 * structure)
      attached  share of non-punctuation tokens that are attached (rel != UNATTACHED and not under an
                unattached component), also reported per part-of-speech tag
      grounded  share of ``word`` nodes with at least one candidate (or a committed) sense; 1.0 if none
      structure 1.0 when there is exactly one ROOT node and every token index occurs once, else 0.0

(2) :func:`score_pair` -- reference vs candidate, node by node, by closeness in USVS (a synonym or a
    reordering still counts; token positions are never used; no composition of attached nodes).
    Node meaning: a ``word`` node is the vector of its committed ``sense`` if set, else of its FIRST
    candidate sense (``best_candidate=True`` instead takes the best-matching candidate pair). Cosine
    over all named axes. A name is 1 if the same token, else 0; word vs name is 0. A word with no
    sense at all falls back to token identity.
    Alignment (one-to-one, deterministic, five lines): the two ROOT nodes are matched. The reference
    tree is then taken level by level (shallowest first). At each level every (reference node,
    unmatched candidate node) pair gets  0.6*meaning + 0.2*[same relation to head]
    + 0.2*[heads already matched to each other]; pairs are taken greedily by descending score
    (ties: higher meaning, then pre-order position in reference, then in candidate) and only if the
    score is at least MATCH_MIN (so a to->from swap in place still matches, with meaning 0).
    A matched pair's credit is its meaning similarity (structure is graded separately).
    Scores, in percent:
      node_recall     sum of credits / reference nodes;  node_precision  sum of credits / candidate nodes
      node_f1         harmonic mean of the two
      attachment_f1   2 * C / (N_ref + N_cand), C = matched nodes whose heads are matched to each
                      other AND whose relation label is equal (a dropped/extra node counts against it)
      overall         0.5 * node_f1 + 0.5 * attachment_f1
    The same figures are given for function-word nodes (chosen sense id has the lexicon prefix) and
    content-word nodes separately (None when neither tree has any node of that class).
    Reasons (``PairScore.reasons``, a list of dicts): node_dropped (function_node_dropped=True and the
    named dimensions when the node is a lexicon-sense node), node_extra, meaning_shift (top differing
    dimensions, signed candidate minus reference), name_mismatch, relation_changed, head_changed,
    subtree_moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .reader import _PUNCT_POS, FN_PREFIX
from .tree import walk
from .types import KIND_NAME, REL_ROOT, REL_UNATTACHED, Node, Tree

W_COV_ATTACHED, W_COV_GROUNDED, W_COV_STRUCTURE = 0.6, 0.2, 0.2
W_ALIGN_MEANING, W_ALIGN_RELATION, W_ALIGN_HEAD = 0.6, 0.2, 0.2
W_OVERALL_NODE_F1, W_OVERALL_ATTACHMENT = 0.5, 0.5
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
    overall: float
    figures: Figures
    function_words: Optional[Figures]
    content_words: Optional[Figures]
    matches: List[Tuple[Node, Node, float]] = field(default_factory=list)   # (reference, candidate, meaning)
    reasons: List[dict] = field(default_factory=list)


class _Space:
    """Sense vectors over the named axes, cached; the meaning of a node."""

    def __init__(self, usvs, best_candidate: bool):
        self.u, self.best, self._v = usvs, best_candidate, {}

    def vec(self, sid: str) -> Optional[np.ndarray]:
        if sid not in self._v:
            self._v[sid] = self.u.sense_dense(sid)
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

    def meaning(self, a: Node, b: Node) -> Tuple[float, Optional[str], Optional[str]]:
        """(similarity in [0, 1], reference sense used, candidate sense used)."""
        if a.kind == KIND_NAME or b.kind == KIND_NAME:
            return (1.0 if (a.kind == b.kind and a.token == b.token) else 0.0), None, None
        oa, ob = self.options(a), self.options(b)
        if not oa or not ob:
            return (1.0 if a.token == b.token else 0.0), None, None
        best = (-1.0, None, None)
        for sa in oa:
            for sb in ob:
                s = 1.0 if sa == sb else self._cos(sa, sb)
                if s > best[0] + _EPS:
                    best = (s, sa, sb)
        return best

    def _cos(self, sa: str, sb: str) -> float:
        va, vb = self.vec(sa), self.vec(sb)
        if va is None or vb is None:
            return 0.0
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        return float(np.dot(va, vb) / (na * nb)) if na > 0 and nb > 0 else 0.0

    def dims(self, sid: Optional[str]) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self.u.sense_signature(sid).items()} if sid else {}

    def diff(self, sa: Optional[str], sb: Optional[str]) -> List[Tuple[str, float]]:
        da, db = self.u.sense_signature(sa) if sa else {}, self.u.sense_signature(sb) if sb else {}
        out = [(k, round(db.get(k, 0.0) - da.get(k, 0.0), 4)) for k in set(da) | set(db)]
        out = [x for x in out if abs(x[1]) > _EPS]
        return sorted(out, key=lambda x: (-abs(x[1]), x[0]))[:TOP_DIMS]


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
    meaning: Dict[int, Tuple[float, Optional[str], Optional[str]]] = {}
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
                s = (W_ALIGN_MEANING * m[0] + W_ALIGN_RELATION * (r.rel == c.rel)
                     + W_ALIGN_HEAD * _heads_aligned(r, c, rp, cp, match))
                if s >= MATCH_MIN - _EPS:
                    scored.append((-s, -m[0], rpos[id(r)], cpos[id(c)], r, c, m))
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


def score_pair(reference: Tree, candidate: Tree, usvs, best_candidate: bool = False) -> PairScore:
    sp = _Space(usvs, best_candidate)
    ro, rp, co, cp, match, meaning = _align(reference, candidate, sp)
    rn = [n for n in ro if not _virtual(n)]
    cn = [n for n in co if not _virtual(n)]
    cmatch = {id(c): r for r in rn if id(r) in match for c in [match[id(r)]]}
    credit: Dict = {}
    good: set = set()
    for r in rn:
        c = match.get(id(r))
        if c is None:
            continue
        credit[id(r)] = credit[("c", id(c))] = meaning[id(r)][0]
        heads_ok = _heads_aligned(r, c, rp, cp, match)
        if heads_ok and r.rel == c.rel:
            good.add(id(r))
            good.add(("c", id(c)))
    reasons: List[dict] = []

    def who(n: Node, side: str) -> dict:
        return dict(side=side, token=n.token, lemma=n.lemma, pos=n.pos, rel=n.rel)

    for r in rn:
        c = match.get(id(r))
        if c is None:
            s = sp.chosen(r)
            rs = who(r, "reference")
            rs.update(kind="node_dropped", function_node_dropped=sp.is_function(r), dimensions=sp.dims(s) if sp.is_function(r) else {})
            reasons.append(rs)
            continue
        sim, sa, sb = meaning[id(r)]
        if r.kind == KIND_NAME or c.kind == KIND_NAME:
            if sim < 1 - _EPS:
                reasons.append(dict(who(r, "reference"), kind="name_mismatch", reference_token=r.token, candidate_token=c.token))
        elif sim < 1 - _EPS:
            reasons.append(dict(who(r, "reference"), kind="meaning_shift", candidate_token=c.token, similarity=round(sim, 4),
                                reference_sense=sa, candidate_sense=sb,
                                dimensions=[dict(dimension=k, difference=v) for k, v in sp.diff(sa, sb)]))
        if _heads_aligned(r, c, rp, cp, match):
            if r.rel != c.rel:
                reasons.append(dict(who(r, "reference"), kind="relation_changed", reference_rel=r.rel, candidate_rel=c.rel))
        else:
            sub_r = _subtree(r)
            sub_c = {id(m) for m in _subtree(c)}
            whole = all(id(m) in match and id(match[id(m)]) in sub_c for m in sub_r) and len(sub_r) == len(sub_c)
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
    matches = [(r, match[id(r)], meaning[id(r)][0]) for r in rn if id(r) in match]
    return PairScore(overall=allf.overall, figures=allf, function_words=_figures(fr, fc, credit, good),
                     content_words=_figures(cr, cc, credit, good), matches=matches, reasons=reasons)


def tree_similarity(a: Tree, b: Tree, usvs, best_candidate: bool = False) -> float:
    """Symmetric: the overall percentage with a as reference averaged with b as reference."""
    return 0.5 * (score_pair(a, b, usvs, best_candidate).overall + score_pair(b, a, usvs, best_candidate).overall)
