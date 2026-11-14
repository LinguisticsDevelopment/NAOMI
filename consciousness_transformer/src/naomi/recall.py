"""Answer a question ClauseTree from Memory, or abstain ("not stated").

Pipeline: (1) tensor candidates via ``memory.similar``; (2) structural
confirmation on the lossless trees with the explicit score below; (3) negation
handling; (4) optional multiple-choice projection. Deterministic, no learning.

score = content + hole + conflict + W_TENSOR * cosine
  content : per non-hole question filler, best match weight against the
            candidate (predicate-predicate, same-role arg, or cross-role arg)
  hole    : how well the candidate can fill the wh-slot (same role > fallback)
  conflict: penalty per non-hole question argument whose role the candidate
            has but filled with something that does not match
A candidate needs content > 0 and a hole filler, and score >= MIN_SCORE.
"""

from __future__ import annotations

from typing import List, Optional

from .memory import MODIFIER_ROLES, Memory, StoredClause
from .types import Answer, ClauseTree, Filler

# --- named weights ---------------------------------------------------------
W_PRED = 3.0            # question predicate matches candidate predicate
W_SAME_ROLE = 2.0       # non-hole argument matches candidate filler in same role
W_CROSS_ROLE = 1.0      # ... matches a filler in a different role
MODIFIER_FACTOR = 0.5   # content matches on modifier roles count half
W_HOLE_SAME = 2.0       # candidate has a filler in the hole's role
W_HOLE_PRED = 2.0       # hole on PREDICATE: candidate predicate returned
W_CONFLICT = -3.0       # same-role argument present in candidate but different
W_TENSOR = 0.5          # weight of handle cosine (tie-breaker / soft evidence)
W_OPT_HOLE = 1.0        # MC: option overlaps the answer filler
W_OPT_CLAUSE = 0.4      # MC: option overlaps another filler of the support
MIN_SCORE = 3.0         # default abstain threshold
K_CANDIDATES = 64

HOLE_FALLBACK = {       # asked role -> {available role: weight}
    "SUBJECT": {"AGENT": 1.5},
    "AGENT": {"SUBJECT": 1.5},
    "OBJECT": {"PLACE": 0.5, "RECIPIENT": 0.5, "GOAL": 0.5, "INSTRUMENT": 0.5},
    "PLACE": {"GOAL": 1.0, "SOURCE": 0.5},
    "RECIPIENT": {"OBJECT": 0.5, "GOAL": 1.0},
}
LIGHT = frozenset("be is are was were am been being do does did done have has had".split())
STOP = frozenset("a an the of to in on at by for with and or is was were are be".split())


def _senses(f: Filler) -> set:
    return set(f.senses) | ({f.sense} if f.sense else set())


def _same(a: Filler, b: Filler) -> bool:
    if a.lemma == b.lemma or a.token == b.token:
        return True
    if a.sense and b.sense:
        return a.sense == b.sense
    return bool(_senses(a) & _senses(b))


def _content_fillers(q: ClauseTree) -> List[Filler]:
    out = []
    for f in [q.predicate] + list(q.args):
        if f.is_hole:
            continue
        if f.role == "PREDICATE" and f.lemma in LIGHT:
            continue
        out.append(f)
    return out


def _score(q: ClauseTree, hole: Filler, c: StoredClause):
    """Return (score_without_tensor, answer_filler, notes) or None."""
    ct = c.tree
    notes, content, used = [], 0.0, set()
    for qf in _content_fillers(q):
        best, bw, bf = None, 0.0, None
        if qf.role == "PREDICATE":
            if _same(qf, ct.predicate):
                best, bw, bf = "pred", W_PRED, ct.predicate
        else:
            for cf in ct.args:
                if not _same(qf, cf) or id(cf) in used:
                    continue
                w = W_SAME_ROLE if cf.role == qf.role else W_CROSS_ROLE
                if w > bw:
                    best, bw, bf = cf.role, w, cf
            if bf is not None and qf.role in MODIFIER_ROLES:
                bw *= MODIFIER_FACTOR
        if bf is not None:
            content += bw
            used.add(id(bf))
            notes.append(f"{qf.lemma}~{bf.lemma}:{bw:g}")
        elif qf.role not in MODIFIER_ROLES and qf.role != "PREDICATE":
            if ct.arg(qf.role) is not None:
                content += W_CONFLICT
                notes.append(f"{qf.lemma}!={ct.arg(qf.role).lemma}:{W_CONFLICT:g}")
    if not used:
        return None
    # hole filler
    ans, hw = None, 0.0
    if hole.role == "PREDICATE":
        ans, hw = ct.predicate, W_HOLE_PRED
    else:
        same = [a for a in ct.args if a.role == hole.role and id(a) not in used]
        if same:
            ans, hw = same[0], W_HOLE_SAME
        else:
            for role, w in HOLE_FALLBACK.get(hole.role, {}).items():
                alt = [a for a in ct.args if a.role == role and id(a) not in used]
                if alt:
                    ans, hw = alt[0], w
                    break
    if ans is None:
        return None
    notes.append(f"hole[{hole.role}]={ans.lemma}({ans.role}):{hw:g}")
    return content + hw, ans, notes


def _words(s: str) -> set:
    return {w for w in s.lower().replace("-", " ").split() if w not in STOP}


def _opt_score(opt: str, ans: Filler, support: ClauseTree) -> float:
    ow = _words(opt)
    if not ow:
        return 0.0
    af = {ans.lemma, ans.token}
    s = W_OPT_HOLE * len(ow & af) / len(ow)
    others = set()
    for f in [support.predicate] + support.args:
        if f is not ans and not f.is_hole:
            others |= {f.lemma, f.token}
    return s + W_OPT_CLAUSE * len(ow & others) / len(ow)


def recall(memory: Memory, question: ClauseTree, layers=None,
           options: Optional[List[str]] = None, min_score: float = MIN_SCORE) -> Answer:
    trace: List[str] = []
    hole = question.hole()
    if hole is None:
        return Answer(None, trace=["question has no hole -> abstain"])
    cands = memory.similar(memory.handle_of(question), k=K_CANDIDATES, layers=layers)
    trace.append(f"{len(cands)} tensor candidates (layers={layers})")
    scored = []
    for c, cos in cands:
        r = _score(question, hole, c)
        if r is None:
            continue
        s, ans, notes = r
        scored.append((s + W_TENSOR * cos, not c.negated, -c.nid, c, ans, notes, cos))
    scored.sort(key=lambda t: (round(t[0], 6), t[1], t[2]), reverse=True)
    for s, pos, _, c, ans, notes, cos in scored[:3]:
        trace.append(f"cand #{c.nid} {'NOT ' if c.negated else ''}"
                     f"{c.tree.predicate.lemma} [{c.layer}] score={s:.2f} cos={cos:.2f} "
                     f"{' '.join(notes)}")
    if not scored or scored[0][0] < min_score:
        best = f"{scored[0][0]:.2f}" if scored else "none"
        trace.append(f"abstain: best structural score {best} < min_score {min_score}")
        return Answer(None, trace=trace)
    s, pos, _, c, ans, notes, cos = scored[0]
    text = ans.token
    if options:
        sc = [_opt_score(o, ans, c.tree) for o in options]
        i = max(range(len(options)), key=lambda j: (sc[j], -j))
        trace.append(f"option scores={[round(x, 2) for x in sc]} raw_filler={ans.token!r}")
        if sc[i] <= 0:
            trace.append("abstain: no option supported by the support clause")
            return Answer(None, trace=trace)
        text = options[i]
        trace.append(f"option_index={i}")
    if c.negated:
        trace.append(f"chosen #{c.nid} is NEGATED -> stated, negated")
        return Answer(text, stated=True, negated=True, support=c.tree, score=s, trace=trace)
    trace.append(f"chosen #{c.nid} score={s:.2f}")
    return Answer(text, stated=True, support=c.tree, score=s, trace=trace)


def option_index(answer: Answer) -> Optional[int]:
    for line in reversed(answer.trace):
        if line.startswith("option_index="):
            return int(line.split("=", 1)[1])
    return None
