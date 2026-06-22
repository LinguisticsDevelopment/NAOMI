"""The controlled-language grammar (M13) — an owned, deterministic, RECURSIVE parser.

Supersedes the flat regex in :func:`membrane.parse` with a small recursive-descent
parser over a controlled English grammar, so a clause can *contain* a clause
(subordination: conditionals, relative clauses, quantified descriptions). It emits the
**same tagged meaning objects** the rest of the system already consumes
(``fact``/``disj``/``rule``/``query`` — see :mod:`membrane`), so it drops straight into
``ConsciousLoop.consume``. Deterministic and round-trip-tested; never an LLM. Parsing is
**structural** (driven by articles/prepositions/verbs, not a vocabulary list), so it
generalizes across the controlled lexicon. Any unparsable input returns ``None`` (the
caller abstains) — no silent garbage.

Grammar (informally)::

    sentence  := conditional | quantified | question | statement
    conditional := 'if' clause ',' clause '.'
    quantified  := ('everyone'|'everything'|'someone'|'something') relclause predicate '.'
    statement   := np predicate '.'
    question    := 'where is' np '?' | 'what can' np ['do'|verb] '?' | 'is' np predicate '?'
    np          := name | 'a' noun | 'the' noun [relclause]
    relclause   := ('that'|'who'|'which') predicate
    predicate   := 'is' ['not'] 'in' 'the' noun ['or' 'the' noun]   (PLACE / neg / disj)
                 | 'is' 'a' noun                                     (IS_A)
                 | 'can' verb 'the' noun                             (CAN_SEE/HOLD/OPEN/REACH)
                 | 'can' word                                       (CAN)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from . import membrane

# verb → CAN_* relation (mirrors membrane._VERB_REL)
_VERB_REL = {"see": "CAN_SEE", "hold": "CAN_HOLD", "open": "CAN_OPEN", "reach": "CAN_REACH"}
_QUANT = {"everyone", "everything", "someone", "something", "anyone", "anything"}
_RELPRON = {"that", "who", "which"}


def _tokens(text: str) -> List[str]:
    # a '?x' variable is one token; a lone '?' stays question punctuation.
    return re.findall(r"\?[a-z]+|[A-Za-z]+|[?.,]", text.strip().lower())


class _Parser:
    """Recursive-descent over the controlled grammar. Methods consume tokens and
    return a partial meaning, or raise ``_Fail`` to abort the whole parse cleanly."""

    def __init__(self, tokens: List[str]) -> None:
        self.t = tokens
        self.i = 0

    # -- token helpers -------------------------------------------------------
    def _peek(self, k: int = 0) -> Optional[str]:
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def _eat(self, *expected: str) -> str:
        tok = self._peek()
        if tok is None or (expected and tok not in expected):
            raise _Fail()
        self.i += 1
        return tok

    def _at(self, *toks: str) -> bool:
        return self._peek() in toks

    def _noun(self) -> str:
        """A single content word in a noun slot (not a function word)."""
        tok = self._eat()
        if tok in {"the", "a", "is", "can", "not", "in", "or", ",", ".", "?"} | _RELPRON:
            raise _Fail()
        return tok

    # -- noun phrase: returns (subject_token, [relative_constraint or None]) --
    def _np(self) -> Tuple[str, Optional[Tuple[str, str, str, bool]]]:
        if self._at("a"):
            self._eat("a")
            return self._noun(), None
        if self._at("the"):
            self._eat("the")
            head = self._noun()
            rel = None
            if self._at(*_RELPRON):                       # 'the N that <predicate>'
                self._eat()
                rel = self._predicate(head)               # constraint clause on the head
            return head, rel
        return self._noun(), None                          # bare name

    # -- predicate over a given subject → (s, relation, value, negate) -------
    def _predicate(self, subj: str):
        if self._at("is"):
            self._eat("is")
            if self._at("a"):                              # IS_A
                self._eat("a")
                return (subj, "IS_A", self._noun(), False)
            negate = False
            if self._at("not"):
                self._eat("not"); negate = True
            self._eat("in"); self._eat("the")
            v1 = self._noun()
            if self._at("or"):                             # disjunction
                self._eat("or"); self._eat("the")
                return ("disj", subj, "PLACE", (v1, self._noun()))
            return (subj, "PLACE", v1, negate)
        if self._at("can"):
            self._eat("can")
            verb = self._eat()
            if verb in _VERB_REL and self._at("the"):      # CAN_SEE the window
                self._eat("the")
                return (subj, _VERB_REL[verb], self._noun(), False)
            return (subj, "CAN", verb, False)              # a bird can fly
        if self._at("moved"):                              # surface variant of 'is in'
            self._eat("moved"); self._eat("to"); self._eat("the")
            return (subj, "PLACE", self._noun(), False)
        raise _Fail()

    def _end(self, *toks: str) -> None:
        if self._peek() not in toks or self.i + 1 != len(self.t):
            raise _Fail()


class _Fail(Exception):
    pass


def _clause_to_obj(c):
    """A predicate tuple → a tagged meaning object (fact/disj)."""
    if c[0] == "disj":
        return c
    s, r, v, neg = c
    return ("fact", s, r, v, neg)


def _shared_var(ant, cons):
    """Repeated-subject → shared variable ``?p`` so a conditional generalizes.

    ``ant``/``cons`` are 4-tuples ``(s, r, v, neg)`` (no disj in rules). If the two
    name the same subject, rename it to ``?p`` in both (coreference inside a rule)."""
    a_s, a_r, a_v, _ = ant
    c_s, c_r, c_v, _ = cons
    if a_s == c_s:
        return (("?p", a_r, a_v), ("?p", c_r, c_v))
    return ((a_s, a_r, a_v), (c_s, c_r, c_v))


def parse(text: str):
    """Parse a controlled-English sentence → a tagged meaning object, or ``None``."""
    toks = _tokens(text)
    if not toks:
        return None
    try:
        return _parse_sentence(_Parser(toks))
    except _Fail:
        return None


def _parse_sentence(p: _Parser):
    head = p._peek()

    # -- conditional: 'if' clause ',' clause '.' → GROUNDED rule -----------------
    # A named conditional ("if mary …, mary …") is about mary specifically — it does
    # NOT license inferring about anyone else, so the subject stays a constant.
    if head == "if":
        p._eat("if")
        subj, _rel = p._np()
        ant = p._predicate(subj)
        p._eat(",")
        subj2, _rel2 = p._np()
        cons = p._predicate(subj2)
        p._end(".")
        if ant[0] == "disj" or cons[0] == "disj":
            raise _Fail()
        return ("rule", ((ant[0], ant[1], ant[2]),), (cons[0], cons[1], cons[2]))

    # -- quantified: 'everyone that <pred> <pred> .' → rule ----------------------
    if head in _QUANT:
        p._eat()
        if not p._at(*_RELPRON):
            raise _Fail()
        p._eat()
        ant = p._predicate("?p")                           # relative clause (the restriction)
        cons = p._predicate("?p")                          # the main predicate
        p._end(".")
        if ant[0] == "disj" or cons[0] == "disj":
            raise _Fail()
        return ("rule", ((ant[0], ant[1], ant[2]),), (cons[0], cons[1], cons[2]))

    # -- questions ---------------------------------------------------------------
    if head == "where":
        p._eat("where"); p._eat("is")
        subj, _r = p._np()
        p._end("?")
        return ("query", subj, "PLACE")
    if head == "what":
        p._eat("what"); p._eat("can")
        subj, _r = p._np()
        if p._at("do"):                                    # 'what can a robin do ?'
            p._eat("do"); p._end("?")
            return ("query", subj, "CAN")
        verb = p._eat()                                    # 'what can mary see ?'
        if verb in _VERB_REL:
            p._end("?")
            return ("query", subj, _VERB_REL[verb])
        raise _Fail()
    if head == "is" and p.t[-1] == "?":                    # yes/no: 'is alice smart ?'
        p._eat("is")
        subj, _r = p._np()
        # reuse predicate machinery by re-injecting an 'is'
        c = _predicate_after_is(p, subj)
        p._end("?")
        s, r, v, neg = c
        return ("query", s, r, v, "-" if neg else "+")

    # -- statement: np predicate '.' --------------------------------------------
    subj, rel = p._np()
    c = p._predicate(subj)
    p._end(".")
    obj = _clause_to_obj(c)
    if rel is not None:                                    # 'the N that <rel> <pred>' (restrictive)
        # the relative clause is a constraint; emit it as a rule: rel ⇒ pred
        if obj[0] == "disj" or rel[0] == "disj":
            raise _Fail()
        a, cc = _shared_var(rel, c)
        return ("rule", (a,), cc)
    return obj


def _predicate_after_is(p: _Parser, subj: str):
    """Predicate for a yes/no question whose leading 'is' was already eaten."""
    if p._at("a"):
        p._eat("a")
        return (subj, "IS_A", p._noun(), False)
    negate = False
    if p._at("not"):
        p._eat("not"); negate = True
    if p._at("in"):
        p._eat("in"); p._eat("the")
        return (subj, "PLACE", p._noun(), negate)
    # bare adjective/value: 'is alice smart ?'  → relation 'is'
    return (subj, "is", p._noun(), negate)


__all__ = ["parse"]
