"""The language membrane — native, owned `text ↔ meaning objects` (M6).

Peripheral plumbing: the system does **not** learn language; this membrane renders
meaning objects to text and parses text back to meaning objects, over the
curriculum's controlled grammar. **No LLM.** It operates on the *existing* meaning
types — `reasoning_oracle` ``Triple`` ``(subject, relation, value)`` plus
truth/operator tags — so it plugs straight into M1–M4.

Meaning objects are tagged tuples (cleanly comparable for cycle-consistency):
  * ``("fact", s, r, v, negate)``        — a (possibly negated) clause
  * ``("disj", s, r, (v1, v2))``         — a disjunction (… or …)
  * ``("rule", (ant_triple,), cons_triple)`` — a conditional (if … , …)
  * ``("query", s, r)``                  — a question

The renderer reproduces the **exact** curriculum sentences (replacing the
leaf-concat seed in :mod:`nsm_ct.reverse_parser`); the parser is the deterministic
template-inverse (the reliable, tested path). ``quantum_parser`` +
``clause.extract_discourse`` are the documented open-domain encoder upgrade; the
``meaning`` resolver is the word-grounding lexicon a future learned decoder needs.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# relation -> (statement template, optional negation template, question template,
# the "if/then" clause phrase). Mirrors episode.py's level templates + _REL_PHRASE.
RELATION_TEMPLATES = {
    "PLACE":     ("{s} is in the {v} .",   "{s} is not in the {v} .", "where is {s} ?",      "is in"),
    "IS_A":      ("a {s} is a {v} .",       None,                      None,                  "is a"),
    "CAN":       ("a {s} can {v} .",        None,                      "what can a {s} do ?", "can"),
    "CAN_SEE":   ("{s} can see the {v} .",  None,                      "what can {s} see ?",  "can see"),
    "CAN_HOLD":  ("{s} can hold the {v} .", None,                      "what can {s} hold ?", "can hold"),
    "CAN_OPEN":  ("{s} can open the {v} .", None,                      "what can {s} open ?", "can open"),
    "CAN_REACH": ("{s} can reach the {v} .", None,                     "what can {s} reach ?", "can reach"),
}
_VERB_REL = {"see": "CAN_SEE", "hold": "CAN_HOLD", "open": "CAN_OPEN", "reach": "CAN_REACH"}

# relation -> polar (yes/no) question form, the inverse of the statement template.
# Used by the cooperative ASK move (M14): turn a missing premise into a question.
_POLAR = {
    "PLACE":     "is {s} in the {v} ?",
    "IS_A":      "is {s} a {v} ?",
    "CAN":       "can {s} {v} ?",
    "CAN_SEE":   "can {s} see the {v} ?",
    "CAN_HOLD":  "can {s} hold the {v} ?",
    "CAN_OPEN":  "can {s} open the {v} ?",
    "CAN_REACH": "can {s} reach the {v} ?",
}


# --------------------------------------------------------------------- render ---
def render_fact(s: str, r: str, v: str, *, negate: bool = False) -> str:
    """A (possibly negated) clause as its curriculum sentence."""
    stmt, neg, _q, _p = RELATION_TEMPLATES[r]
    if negate and neg is not None:
        return neg.format(s=s, v=v)
    return stmt.format(s=s, v=v)


def render_clause(s: str, r: str, v: str) -> str:
    """The clause without the trailing ' .' (for conditionals + verbalization)."""
    return render_fact(s, r, v)[:-2]


def render_disjunction(s: str, r: str, values) -> str:
    a, b = values
    return f"{s} is in the {a} or the {b} ."


def render_rule(antecedents, consequent) -> str:
    """A conditional ``if <ant> , <cons> .`` from clause forms."""
    (sa, ra, va), = antecedents
    sc, rc, vc = consequent
    return f"if {render_clause(sa, ra, va)} , {render_clause(sc, rc, vc)} ."


def render_query(s: str, r: str) -> str:
    _stmt, _neg, q, _p = RELATION_TEMPLATES[r]
    if q is None:
        raise ValueError(f"relation {r} has no question form")
    return q.format(s=s)


def render_polar_question(s: str, r: str, v: str) -> str:
    """A literal ``(s, r, v)`` as a yes/no question (the inverse of its statement)."""
    return _POLAR[r].format(s=s, v=v)


def render(obj) -> str:
    """Render any tagged meaning object to its curriculum sentence."""
    tag = obj[0]
    if tag == "fact":
        _, s, r, v, neg = obj
        return render_fact(s, r, v, negate=neg)
    if tag == "disj":
        _, s, r, vs = obj
        return render_disjunction(s, r, vs)
    if tag == "rule":
        _, ants, cons = obj
        return render_rule(ants, cons)
    if tag == "query":
        _, s, r = obj
        return render_query(s, r)
    raise ValueError(f"unknown meaning object: {obj!r}")


# ---------------------------------------------------------------------- parse ---
def _parse_clause(text: str) -> Optional[Tuple[str, str, str]]:
    """Parse an if/then clause (no trailing period) → ``(s, r, v)``."""
    m = re.match(r"^(\w+) is in the (\w+)$", text)
    if m:
        return (m.group(1), "PLACE", m.group(2))
    m = re.match(r"^(\w+) can (see|hold|open|reach) the (\w+)$", text)
    if m:
        return (m.group(1), _VERB_REL[m.group(2)], m.group(3))
    return None


def parse(text: str):
    """Parse a curriculum sentence to a tagged meaning object (``None`` if unknown)."""
    t = text.strip()
    # questions
    m = re.match(r"^where is (\w+) \?$", t)
    if m:
        return ("query", m.group(1), "PLACE")
    m = re.match(r"^what can a (\w+) do \?$", t)
    if m:
        return ("query", m.group(1), "CAN")
    m = re.match(r"^what can (\w+) (see|hold|open|reach) \?$", t)
    if m:
        return ("query", m.group(1), _VERB_REL[m.group(2)])
    # conditional
    m = re.match(r"^if (.+?) , (.+) \.$", t)
    if m:
        ant, cons = _parse_clause(m.group(1)), _parse_clause(m.group(2))
        if ant and cons:
            return ("rule", (ant,), cons)
        return None
    # statements
    m = re.match(r"^(\w+) is not in the (\w+) \.$", t)
    if m:
        return ("fact", m.group(1), "PLACE", m.group(2), True)
    m = re.match(r"^(\w+) is in the (\w+) or the (\w+) \.$", t)
    if m:
        return ("disj", m.group(1), "PLACE", (m.group(2), m.group(3)))
    m = re.match(r"^(\w+) is in the (\w+) \.$", t)
    if m:
        return ("fact", m.group(1), "PLACE", m.group(2), False)
    # "moved to X" is the same PLACE meaning as "is in X" (the recency cue is the
    # controller's overwrite gate, not a meaning difference) — canonicalize it.
    m = re.match(r"^(\w+) moved to the (\w+) \.$", t)
    if m:
        return ("fact", m.group(1), "PLACE", m.group(2), False)
    m = re.match(r"^a (\w+) is a (\w+) \.$", t)
    if m:
        return ("fact", m.group(1), "IS_A", m.group(2), False)
    # name-subject IS_A ("sandra is a person") — no article on the subject.
    m = re.match(r"^(\w+) is a (\w+) \.$", t)
    if m:
        return ("fact", m.group(1), "IS_A", m.group(2), False)
    m = re.match(r"^a (\w+) can (\w+) \.$", t)
    if m:
        return ("fact", m.group(1), "CAN", m.group(2), False)
    m = re.match(r"^(\w+) can (see|hold|open|reach) the (\w+) \.$", t)
    if m:
        return ("fact", m.group(1), _VERB_REL[m.group(2)], m.group(3), False)
    return None


__all__ = [
    "RELATION_TEMPLATES", "render", "render_fact", "render_clause",
    "render_disjunction", "render_rule", "render_query", "render_polar_question", "parse",
]
