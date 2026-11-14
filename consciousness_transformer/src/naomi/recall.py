"""Answer a question ClauseTree from Memory, or abstain ("not stated").

Pipeline: (1) tensor candidates via ``memory.similar``; (2) structural
confirmation on the lossless trees with the explicit score below; (3) negation /
polarity handling; (4) optional multiple-choice projection. Deterministic, no learning.

score = content + hole + conflict + polarity + W_TENSOR * cosine
  content : per non-hole question filler, best match weight against the
            candidate (predicate-predicate, same-role arg, or cross-role arg).
            Parser role labels are noisy, so a cross-role match is nearly as good.
  hole    : how well the candidate can fill the wh-slot: same role > fallback
            role > "the one unmatched core argument left" (role-noise tolerance);
            QUANTITY holes take the numeral nearest the matched noun;
            copular questions ("who was X's Y") read the clause symmetrically.
  conflict: penalty per non-hole SUBJECT/OBJECT question argument whose role the
            candidate has but filled with something that does not match
A candidate is structurally *backed* only if its predicate matches the question's
predicate (or an XCOMP verb of it) or at least two non-modifier question fillers
match it (or it is a naming clause for a "what was the name of X" question);
unbacked candidates never answer. Candidates also need score >= MIN_SCORE.
Optional light coreference: a pronoun subject carries the provenance field
``antecedent`` (set by ``Mind.teach``); it matches / answers at reduced weight.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional

from .memory import MODIFIER_ROLES, Memory, StoredClause
from .types import Answer, ClauseTree, Filler

# --- named weights ---------------------------------------------------------
W_PRED = 3.0            # question predicate matches candidate predicate
W_XCOMP = 2.5           # a question XCOMP verb equals the candidate predicate
W_SAME_ROLE = 2.0       # non-hole argument matches candidate filler in same role
W_CROSS_ROLE = 1.5      # ... matches a filler in a different role (role-label noise)
W_ANTECEDENT = 1.2      # ... matches the antecedent of a pronoun subject
MODIFIER_FACTOR = 0.5   # content matches on modifier roles count half
W_CONTEXT = 0.5         # subordinate-clause words: optional evidence, never a veto
W_NAMING = 2.0          # naming clause for a "name of X" question
W_HOLE_SAME = 2.0       # candidate has a filler in the hole's role
W_HOLE_PRED = 2.0       # hole on PREDICATE: candidate predicate returned
W_HOLE_LEFT = 1.5       # the single unmatched core argument fills the hole
W_HOLE_LEFT_MANY = 0.8  # several unmatched core arguments: best role guess
W_CONFLICT = -3.0       # same-role argument present in candidate but different
W_POLARITY = -4.0       # question negated, candidate not
W_TENSOR = 0.5          # weight of handle cosine (tie-breaker / soft evidence)
W_OPT_HOLE = 1.0        # MC: option overlaps the answer filler
W_OPT_CLAUSE = 0.4      # MC: option overlaps another filler of the support
W_OPT_SENT = 0.12       # MC: option overlaps other words of the support sentence
W_OPT_ECHO = 1.0        # MC: penalty for an option that just restates the question
MIN_SCORE = 3.0         # default abstain threshold
K_CANDIDATES = 64
N_SUPPORTS = 3          # MC: top-k support clauses considered
SUPPORT_BAND = 1.5      # ... within this score of the best one

HOLE_FALLBACK = {       # asked role -> {available role: weight}
    "SUBJECT": {"AGENT": 1.5},
    "AGENT": {"SUBJECT": 1.5},
    "OBJECT": {"PLACE": 0.5, "RECIPIENT": 0.5, "GOAL": 0.5, "INSTRUMENT": 0.5,
               "WITH": 0.5, "FOR": 0.5, "ABOUT": 0.5, "TO": 0.5},
    "PLACE": {"GOAL": 1.0, "SOURCE": 0.5},
    "RECIPIENT": {"OBJECT": 0.5, "GOAL": 1.0},
    "ATTRIBUTE": {"COMPLEMENT": 2.0, "OBJECT": 1.5},
}
COPULAS = frozenset("be become seem look feel appear remain name call".split())
LIGHT = frozenset("be is are was were am been being do does did done have has had".split())
STOP = frozenset("a an the of to in on at by for with and or is was were are be".split())
SPECIAL_ROLES = frozenset({"KIND", "XCOMP", "CONTEXT"})
# roles whose unmatched filler may stand in for a who/what hole (role-noise tolerance)
LEFTOVER_ROLES = ("OBJECT", "SUBJECT", "ATTRIBUTE", "INDIRECT_OBJECT", "AGENT", "RECIPIENT",
                  "BENEFICIARY", "INSTRUMENT", "WITH", "FOR", "ABOUT", "OF", "TO", "FROM", "BY")
LEFTOVER_HOLES = frozenset(LEFTOVER_ROLES) | {"PLACE"}
PRONOUNS = frozenset("i you he she it we they me him her us them".split())
NAMING_WORDS = frozenset({"named", "called", "name", "names"})
_NUM = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
_NUM.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
             "eighty": "80", "ninety": "90", "hundred": "100", "dozen": "12"})
_NUM_REV = {v: k for k, v in _NUM.items()}


def _senses(f: Filler) -> set:
    return set(f.senses) | ({f.sense} if f.sense else set())


def _same(a: Filler, b: Filler) -> bool:
    if a.lemma == b.lemma or a.token == b.token:
        return True
    if a.sense and b.sense:
        return a.sense == b.sense
    return bool(_senses(a) & _senses(b))


def _role_eq(a: str, b: str) -> bool:
    return a == b or {a, b} <= {"POSSESSOR", "OF"} or {a, b} <= {"ATTRIBUTE", "COMPLEMENT"}


def _is_pron(f: Filler) -> bool:
    return f.lemma in PRONOUNS or f.token in PRONOUNS


def _antecedent(c: StoredClause) -> Optional[dict]:
    a = c.provenance.get("antecedent") if c.provenance else None
    return a if isinstance(a, dict) else None


def _content_fillers(q: ClauseTree) -> List[Filler]:
    out = []
    for f in [q.predicate] + list(q.args):
        if f.is_hole or f.role in SPECIAL_ROLES:
            continue
        if f.role == "PREDICATE" and f.lemma in LIGHT:
            continue
        out.append(f)
    return out


_COLOR = ("chromatic_color.n.01", "achromatic_color.n.01", "color.n.01")


@lru_cache(maxsize=20000)
def _isa(lemma: str, kind: str) -> bool:
    """WordNet typing: is some noun/adjective sense of ``lemma`` a kind of ``kind``?
    ("beaver" is an animal, "brown" is a colour). General English lexical knowledge."""
    try:
        from nltk.corpus import wordnet as wn
        if kind in ("color", "colour"):
            targets = {wn.synset(n) for n in _COLOR}
        else:
            targets = set(wn.synsets(kind, pos="n")[:2])
        if not targets:
            return False
        for s in wn.synsets(lemma)[:8]:
            if s.pos() in ("s", "a"):   # adjective "brown": go through its noun forms
                for l in s.lemmas():
                    for d in l.derivationally_related_forms():
                        if d.synset() in targets:
                            return True
                continue
            for path in s.hypernym_paths():
                if targets & set(path):
                    return True
    except Exception:
        return False
    return False


def _unit(ct: ClauseTree, q: Filler) -> Optional[Filler]:
    """The counted noun right after a QUANTITY filler (found by token index)."""
    if q.token_index is None:
        return None
    best = None
    for a in ct.args:
        if a.token_index is not None and q.token_index < a.token_index <= q.token_index + 3 and a.role != "QUANTITY":
            if best is None or a.token_index < best.token_index:
                best = a
    return best


def _score(q: ClauseTree, hole: Filler, c: StoredClause):
    """Return (score_without_tensor, answer_filler, notes, used_fillers) or None."""
    ct = c.tree
    notes, content, used = [], 0.0, {}
    n_core, backed = 0, False
    copular_q = q.predicate.lemma == "be" and not q.predicate.is_hole
    cand_copular = ct.predicate.lemma in COPULAS
    kinds = {f.lemma for f in q.args if f.role == "KIND"}
    if copular_q and not cand_copular and not (kinds - {"name"}):
        return None
    ant = _antecedent(c)
    # --- predicate / XCOMP verbs
    if not q.predicate.is_hole and q.predicate.lemma not in LIGHT:
        if _same(q.predicate, ct.predicate):
            content += W_PRED
            backed = True
            notes.append(f"pred:{W_PRED:g}")
        elif ct.predicate.lemma == "be" and any(
                a.role == "OBJECT" and a.lemma == q.predicate.lemma for a in ct.args):
            vb = next(a for a in ct.args if a.role == "OBJECT" and a.lemma == q.predicate.lemma)
            used[id(vb)] = vb
            content += W_XCOMP      # progressive / passive parsed as be(OBJECT=verb)
            backed = True
            notes.append(f"be+verb:{W_XCOMP:g}")
        elif hole.role == "ATTRIBUTE" and q.predicate.lemma in COPULAS and cand_copular:
            content += W_XCOMP      # "how did X feel/look" ~ "X was/felt ADJ"
            backed = True
            notes.append(f"copular-pred:{W_XCOMP:g}")
        else:
            for x in q.args:
                if x.role == "XCOMP" and _same(x, ct.predicate):
                    content += W_XCOMP
                    backed = True
                    notes.append(f"xcomp:{W_XCOMP:g}")
                    break
    elif q.predicate.is_hole:
        pass
    # --- non-hole arguments
    unmatched, pred_used, subj_same, pairs = [], False, False, set()
    for qf in _content_fillers(q):
        if qf.role == "PREDICATE":
            continue
        best, bw = None, 0.0
        for cf in ct.args:
            if id(cf) in used or not _same(qf, cf):
                continue
            w = W_SAME_ROLE if _role_eq(cf.role, qf.role) else W_CROSS_ROLE
            if w > bw:
                best, bw = cf, w
        if best is None and not pred_used and ct.predicate.lemma not in LIGHT and _same(qf, ct.predicate):
            pred_used = True        # noun parsed as the clause head ("puppy(DESCRIPTION=brown)")
            content += W_CROSS_ROLE * (MODIFIER_FACTOR if qf.role in MODIFIER_ROLES else 1.0)
            n_core += qf.role not in MODIFIER_ROLES
            notes.append(f"{qf.lemma}~pred:{W_CROSS_ROLE:g}")
            continue
        if best is None and ant is not None and qf.role in ("SUBJECT", "OBJECT"):
            sj = ct.arg("SUBJECT")
            if sj is not None and _is_pron(sj) and ant.get("lemma") in (qf.lemma, qf.token) and id(sj) not in used:
                best, bw = sj, W_ANTECEDENT
        if best is not None:
            subj_same = subj_same or (qf.role == "SUBJECT" and best.role == "SUBJECT")
            pairs.add((qf.role, best.role if best.role in ("SUBJECT", "OBJECT", "INDIRECT_OBJECT") else "-"))
            if qf.role in MODIFIER_ROLES:
                bw *= MODIFIER_FACTOR
            else:
                n_core += 1
            content += bw
            used[id(best)] = best
            notes.append(f"{qf.lemma}~{best.lemma}:{bw:g}")
        else:
            unmatched.append(qf)
    if not copular_q and (("SUBJECT", "OBJECT") in pairs or ("SUBJECT", "INDIRECT_OBJECT") in pairs) \
            and ("OBJECT", "SUBJECT") in pairs:
        content += W_CONFLICT       # subject and object of the question appear swapped in the clause
        notes.append(f"role-swap:{W_CONFLICT:g}")
    for qf in unmatched:
        if qf.role == "POSSESSOR":      # "Anna's father" vs a clause about Jim's father
            other = [a for a in ct.args if a.role == "POSSESSOR" and id(a) not in used]
            if other:
                content += W_CONFLICT
                notes.append(f"{qf.lemma}!={other[0].lemma}:{W_CONFLICT:g}")
        if qf.role in ("SUBJECT", "OBJECT"):
            cf = ct.arg(qf.role)
            known = cf is not None and (not _is_pron(cf) or (ant is not None and cf.role == "SUBJECT"))
            if known and id(cf) not in used and cf.lemma not in NAMING_WORDS:
                content += W_CONFLICT
                notes.append(f"{qf.lemma}!={cf.lemma}:{W_CONFLICT:g}")
    # optional subordinate-clause context
    for qf in q.args:
        if qf.role == "CONTEXT":
            for cf in [ct.predicate] + ct.args:
                if id(cf) not in used and _same(qf, cf):
                    content += W_CONTEXT
                    notes.append(f"ctx:{qf.lemma}")
                    break
    # naming questions: "what was the name of X"
    if "name" in kinds:
        sent = set(ct.sentence.lower().replace(",", " ").split()) if ct.sentence else set()
        if ct.predicate.lemma in ("name", "call") or sent & NAMING_WORDS:
            content += W_NAMING
            n_core += 1
            notes.append(f"naming:{W_NAMING:g}")
        else:
            return None
    kind_typed = sorted(kinds - {"name", "kind", "type", "sort", "size", "shape", "names"})
    typed = None
    if kind_typed and hole.role not in ("QUANTITY", "PREDICATE"):
        # restriction noun ("which ANIMAL ...", "what COLOR ..."): the answer must be of that type
        for a in [a for a in ct.args if id(a) not in used and a.role not in SPECIAL_ROLES]:
            if _isa(a.lemma, kind_typed[0]):
                typed = a
                break
        if typed is not None and n_core >= 1:
            backed = True
            notes.append(f"typed:{typed.lemma}")
    if copular_q and cand_copular and n_core >= 1 and hole.role in ("SUBJECT", "OBJECT", "ATTRIBUTE"):
        backed = True       # "who was X?": X found in a copular clause, the other side answers
    if hole.role == "PREDICATE" and subj_same:
        backed = True       # "what did X do?": X is this clause's subject
    if not (backed or n_core >= 2):
        return None
    # --- polarity
    if q.negated and not c.negated:
        content += W_POLARITY
        notes.append(f"polarity:{W_POLARITY:g}")
    # --- hole filler
    avail = [a for a in ct.args if id(a) not in used]
    ans, hw = None, 0.0
    if typed is not None:
        ans, hw = typed, W_HOLE_SAME
    elif hole.role == "PREDICATE":
        ans, hw = ct.predicate, W_HOLE_PRED
    elif hole.role == "QUANTITY":
        qs = [a for a in avail if a.role == "QUANTITY"]
        if qs:
            anchors = [u.token_index for u in used.values() if u.token_index is not None]
            if anchors:
                qs.sort(key=lambda a: min(abs((a.token_index or 0) - x) for x in anchors))
            ans, hw = qs[0], W_HOLE_SAME
    else:
        same = [a for a in avail if _role_eq(a.role, hole.role)]
        if same:
            ans, hw = same[0], W_HOLE_SAME
        else:
            for role, w in HOLE_FALLBACK.get(hole.role, {}).items():
                alt = [a for a in avail if a.role == role]
                if alt:
                    ans, hw = alt[0], w
                    break
        if ans is not None and ant is not None and ans.role == "SUBJECT" and _is_pron(ans):
            ans, hw = Filler(role="SUBJECT", token=ant["token"], lemma=ant["lemma"]), hw - 0.3
        if ans is None and ant is not None and hole.role in ("SUBJECT", "AGENT"):
            sj = ct.arg("SUBJECT")
            if sj is not None and _is_pron(sj):
                ans, hw = Filler(role="SUBJECT", token=ant["token"], lemma=ant["lemma"]), W_ANTECEDENT + 0.5
        if ans is None and hole.role in LEFTOVER_HOLES:
            left = [a for a in avail if a.role in LEFTOVER_ROLES and a.lemma not in LIGHT
                    and a.lemma not in {"name", "color", "colour", "kind", "type", "sort"}]
            if left:
                left.sort(key=lambda a: LEFTOVER_ROLES.index(a.role))
                ans = left[0]
                hw = W_HOLE_LEFT if len(left) == 1 else W_HOLE_LEFT_MANY
    if ans is None:
        return None
    notes.append(f"hole[{hole.role}]={ans.lemma}({ans.role}):{hw:g}")
    return content + hw, ans, notes, list(used.values())


# --------------------------------------------------------------------------- MC
def _lem(w: str) -> str:
    try:
        from .reader import _wn_lemma
        return _wn_lemma(w, None)
    except Exception:
        return w


def _variants(w: str) -> set:
    v = {w, _lem(w)}
    if w in _NUM:
        v.add(_NUM[w])
    if w in _NUM_REV:
        v.add(_NUM_REV[w])
    return v


def _words(s: str) -> List[str]:
    out = []
    for w in re.sub(r"[^\w\s'-]", " ", s.lower()).replace("-", " ").split():
        w = w.strip("'")
        if w and w not in STOP:
            out.append(w)
    return out


def _wordset(words) -> set:
    out = set()
    for w in words:
        out |= _variants(w)
    return out


def _answer_words(ans: Filler, ct: ClauseTree, kinds: set) -> List[str]:
    ws = [ans.token, ans.lemma]
    if ans.role == "QUANTITY":
        u = _unit(ct, ans)
        if u is not None:
            ws += [u.token, u.lemma]
    elif ans.role == "PREDICATE":
        ws += [a.token for a in ct.args if a.role in ("OBJECT", "ATTRIBUTE", "PLACE")]
    if "name" in kinds and ct.sentence:
        toks = re.sub(r"[^\w\s'-]", " ", ct.sentence.lower()).split()
        for i, t in enumerate(toks):
            if t in ("named", "called"):
                ws += toks[i + 1:i + 4]
    return [w for w in ws if w not in STOP]


def _opt_value(opt: str, ans: Filler, ct: ClauseTree, qwords: set, kinds: set):
    """-> (value, structural_hit)"""
    ow = _words(opt)
    if not ow:
        return 0.0, 0.0
    A = _wordset(_answer_words(ans, ct, kinds))
    B = set()
    for f in [ct.predicate] + ct.args:
        if f is not ans and not f.is_hole and f.role not in SPECIAL_ROLES:
            B |= _variants(f.token) | _variants(f.lemma)
    C = _wordset(_words(ct.sentence))
    n = len(ow)
    a = sum(1 for w in ow if _variants(w) & A) / n
    b = sum(1 for w in ow if not (_variants(w) & A) and (_variants(w) & B)) / n
    c = sum(1 for w in ow if not (_variants(w) & (A | B)) and (_variants(w) & C)) / n
    echo = sum(1 for w in ow if _variants(w) & qwords) / n
    val = W_OPT_HOLE * a + W_OPT_CLAUSE * b + W_OPT_SENT * c - W_OPT_ECHO * echo
    return val, a + b


def _render(ans: Filler, ct: ClauseTree) -> str:
    if ans.role == "QUANTITY":
        u = _unit(ct, ans)
        return f"{ans.token} {u.token}" if u is not None else ans.token
    if ans.role == "PREDICATE":
        objs = [a.token for a in ct.args if a.role in ("OBJECT", "PLACE")]
        return " ".join([ans.token] + objs)
    return ans.token


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
        s, ans, notes, used = r
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
    text = _render(ans, c.tree)
    if options:
        qwords = _wordset(w for f in [question.predicate] + question.args
                          if not f.is_hole and f.role not in ("XCOMP", "CONTEXT", "KIND")
                          for w in (f.token, f.lemma) if w not in LIGHT)
        kinds = {f.lemma for f in question.args if f.role == "KIND"}
        n_sup = 8 if hole.role == "PREDICATE" else N_SUPPORTS
        sups = [t for t in scored[:n_sup] if t[0] >= min_score and t[0] >= scored[0][0] - SUPPORT_BAND
                and (t[3].negated == question.negated)]
        if not sups:
            trace.append("abstain: support polarity differs from the question")
            return Answer(None, trace=trace)
        best_val = [(-9.0, 0.0, None)] * len(options)
        for t in sups:
            for j, o in enumerate(options):
                val, hit = _opt_value(o, t[4], t[3].tree, qwords, kinds)
                if hit > 0 and val > best_val[j][0] and (hole.role != "PREDICATE" or val > W_OPT_CLAUSE):
                    best_val[j] = (val, hit, t)
        vals = [b[0] for b in best_val]
        order = sorted(range(len(options)), key=lambda j: (-vals[j], j))
        i = order[0]
        second = vals[order[1]] if len(order) > 1 else -9.0
        trace.append(f"option scores={[round(x, 2) for x in vals]} raw_filler={ans.token!r}")
        if best_val[i][2] is None or vals[i] <= 0:
            trace.append("abstain: no option supported by a structurally matched clause")
            return Answer(None, trace=trace)
        if vals[i] - second < 1e-6:
            trace.append("abstain: top options tie")
            return Answer(None, trace=trace)
        s, pos, _, c, ans, notes, cos = best_val[i][2]
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
