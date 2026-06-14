"""Reasoning oracle — a tiny forward-chainer used ONLY to grade/generate, never at
inference.

The model must learn to reason *emergently* (in the ClausePsyche loop). This
oracle exists so the curriculum has ground truth: given in-context facts + rules,
it derives the gold answer, the gold derivation **chain**, and whether a query is
**answerable** at all (for the abstain target). It is a minimal Datalog-style
unifier over ``(entity, relation, value)`` triples — enough for the two reasoning
forms we train:

* **Conditionals (modus ponens):** ``rule (?p PLACE A) ⇒ (?p CAN_SEE X)`` fired by
  a fact ``(n PLACE A)`` derives ``(n CAN_SEE X)``.
* **Transitivity / inheritance:** ``rule (?x IS_A ?y) ∧ (?y CAN ?z) ⇒ (?x CAN ?z)``
  derives ``(robin CAN fly)`` from ``(robin IS_A bird) ∧ (bird CAN fly)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Triple = Tuple[str, str, str]  # (entity, relation, value); a leading '?' marks a variable


@dataclass(frozen=True)
class Rule:
    """A Horn rule: all antecedent patterns ⇒ the consequent (shared variables)."""

    antecedents: Tuple[Triple, ...]
    consequent: Triple
    name: str = ""


@dataclass
class DerivStep:
    """One derivation: ``derived`` produced by ``rule`` from ``support`` facts."""

    derived: Triple
    rule: str
    support: Tuple[Triple, ...]


def _is_var(x: str) -> bool:
    return isinstance(x, str) and x.startswith("?")


def _unify(pattern: Triple, fact: Triple, subst: Dict[str, str]) -> Optional[Dict[str, str]]:
    s = dict(subst)
    for p, f in zip(pattern, fact):
        if _is_var(p):
            if p in s and s[p] != f:
                return None
            s[p] = f
        elif p != f:
            return None
    return s


def _ground(pattern: Triple, s: Dict[str, str]) -> Triple:
    return tuple(s.get(x, x) if _is_var(x) else x for x in pattern)  # type: ignore[return-value]


def forward_chain(
    facts: List[Triple], rules: List[Rule], max_iters: int = 16,
) -> Tuple[set, List[DerivStep]]:
    """Saturate ``facts`` under ``rules`` to a fixpoint; return (all facts, chain)."""
    known = set(facts)
    chain: List[DerivStep] = []
    for _ in range(max_iters):
        added = False
        for rule in rules:
            substs: List[Dict[str, str]] = [{}]
            for ant in rule.antecedents:
                nxt: List[Dict[str, str]] = []
                for s in substs:
                    for f in known:
                        u = _unify(ant, f, s)
                        if u is not None:
                            nxt.append(u)
                substs = nxt
            for s in substs:
                cons = _ground(rule.consequent, s)
                if cons not in known:
                    known.add(cons)
                    chain.append(DerivStep(cons, rule.name,
                                           tuple(_ground(a, s) for a in rule.antecedents)))
                    added = True
        if not added:
            break
    return known, chain


def derive(
    facts: List[Triple], rules: List[Rule], query: Tuple[str, str],
) -> Tuple[Optional[str], List[DerivStep]]:
    """Answer ``query = (entity, relation)`` → ``(value | None, chain)``.

    ``value is None`` means **unanswerable** (nothing derives it) — the abstain case.
    """
    known, chain = forward_chain(facts, rules)
    e, r = query
    value = next((v for (qe, qr, v) in known if qe == e and qr == r), None)
    return value, chain


# -- the two rule schemas the curriculum uses -------------------------------------
def conditional_rule(place: str, obj: str) -> Rule:
    """``if (?p in PLACE) then (?p CAN_SEE obj)`` — grounded modus ponens."""
    return Rule((("?p", "PLACE", place),), ("?p", "CAN_SEE", obj), name="modus_ponens")


# generic property-inheritance over an is-a edge (transitivity in one rule)
INHERITANCE = Rule((("?x", "IS_A", "?y"), ("?y", "CAN", "?z")), ("?x", "CAN", "?z"),
                   name="inheritance")

# is-a is transitive: lets is-a chains compose to any depth (robin->bird->animal)
IS_A_TRANS = Rule((("?x", "IS_A", "?y"), ("?y", "IS_A", "?z")), ("?x", "IS_A", "?z"),
                  name="is_a_trans")
