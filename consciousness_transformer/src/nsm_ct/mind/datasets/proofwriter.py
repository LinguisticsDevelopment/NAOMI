"""ProofWriter ingestion — a real, broad deductive-reasoning dataset (M8).

ProofWriter (Allen AI; the RuleTaker successor) is exactly this architecture's
task at scale: **facts + Horn rules + a query → True / False / Unknown**, where the
open-world (OWA) **Unknown is derive-or-abstain**, graded by reasoning depth 0–5,
with negation and arbitrary predicates/entities — far broader than the toy
curriculum's 7 relations.

Each literal is a **4-tuple** ``(subject, predicate, object, polarity)`` with
``polarity ∈ {"+","-"}``. Because :func:`reasoning_oracle.forward_chain` /
``_unify`` are arity-generic (they ``zip`` pattern and fact), we reuse the existing
engine **unchanged** with polarity as the 4th element — no change to the shared
reasoner. The universally-quantified ProofWriter variable ("someone"/"they")
maps to the oracle's ``?x``.

This module gives (1) a representation parser, (2) a loader over the OWA JSONL,
and (3) :func:`verify` — True if the positive literal is in the forward-chain
closure, False if the negative one is, else Unknown (abstain).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from ...reasoning_oracle import Rule, forward_chain

Literal = Tuple[str, str, str, str]  # (subject, predicate, object, polarity "+"/"-")

# ProofWriter's universal variable exponents → the oracle's variable form.
_VARS = {"someone", "something", "they", "it"}
_LIT_RE = re.compile(r'\("([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\s+"([+\-])"\)')

TRUE, FALSE, UNKNOWN = "true", "false", "Unknown"


def _tok(s: str) -> str:
    s = s.strip().lower()
    return "?x" if s in _VARS else s


def parse_literal(rep: str) -> Literal:
    """``("Gary" "is" "kind" "+")`` → ``("gary","is","kind","+")`` (vars → ?x)."""
    m = _LIT_RE.search(rep)
    if not m:
        raise ValueError(f"bad literal representation: {rep!r}")
    s, p, o, pol = m.groups()
    return (_tok(s), _tok(p), _tok(o), pol)


def parse_rule(rep: str) -> Rule:
    """``(((A1)(A2)) -> (C))`` → a :class:`Rule` of 4-tuple literals."""
    left, right = rep.split("->", 1)
    ants = tuple(( _tok(a), _tok(b), _tok(c), pol)
                 for (a, b, c, pol) in _LIT_RE.findall(left))
    cm = _LIT_RE.search(right)
    a, b, c, pol = cm.groups()
    return Rule(antecedents=ants, consequent=(_tok(a), _tok(b), _tok(c), pol), name="pw")


@dataclass
class PWExample:
    """One ProofWriter theory + its questions."""
    facts: List[Literal]
    rules: List[Rule]
    questions: List[Tuple[Literal, str, int]]  # (query literal, gold answer, depth)


def parse_record(rec: dict) -> PWExample:
    facts = [parse_literal(t["representation"]) for t in rec.get("triples", {}).values()]
    rules = [parse_rule(r["representation"]) for r in rec.get("rules", {}).values()]
    questions = []
    for q in rec.get("questions", {}).values():
        lit = parse_literal(q["representation"])
        ans = q["answer"]
        gold = TRUE if ans is True else FALSE if ans is False else UNKNOWN
        questions.append((lit, gold, int(q.get("QDep") or 0)))
    return PWExample(facts=facts, rules=rules, questions=questions)


def load_records(path: str, limit: Optional[int] = None) -> Iterator[dict]:
    """Yield raw ProofWriter JSONL records from ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def verify(facts: List[Literal], rules: List[Rule], query: Literal) -> str:
    """Forward-chain (OWA) and label the query ``TRUE`` / ``FALSE`` / ``UNKNOWN``.

    The query carries its own polarity (ProofWriter asks both "X is kind" and
    "X is not kind"). The asked literal is **True** if derivable, **False** if its
    opposite polarity is derivable, else **Unknown** (the derive-or-abstain case).
    """
    known, _chain = forward_chain(list(facts), list(rules))
    s, p, o, qpol = query
    opp = "-" if qpol == "+" else "+"
    if (s, p, o, qpol) in known:
        return TRUE
    if (s, p, o, opp) in known:
        return FALSE
    return UNKNOWN


def default_data_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "..", "..", "data", "proofwriter")


__all__ = [
    "Literal", "PWExample", "parse_literal", "parse_rule", "parse_record",
    "load_records", "verify", "TRUE", "FALSE", "UNKNOWN", "default_data_dir",
]
