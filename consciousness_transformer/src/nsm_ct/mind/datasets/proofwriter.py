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


# --------------------------------------------------------------------------- #
# Verification-mode training (M8 step 2): facts + rules + query -> {true,false,idk}
# Reuse the controller's contrastive MC head by making the options the three
# answer atoms; polarity is folded into the value atom so it is first-class.
# --------------------------------------------------------------------------- #
_ANS = (TRUE, FALSE, UNKNOWN)
_ANS_IDX = {a: i for i, a in enumerate(_ANS)}


def flatten(examples: List[PWExample]) -> List[Tuple[List[Literal], List[Rule], Literal, int, int]]:
    """Flatten to ``(facts, rules, query, answer_idx, depth)`` per question."""
    items = []
    for ex in examples:
        for (lit, gold, qd) in ex.questions:
            items.append((ex.facts, ex.rules, lit, _ANS_IDX[gold], qd))
    return items


def build_pw_batch(items, codec):
    """Build a verification :class:`ClauseBatch` from flattened ProofWriter items.

    Each item streams its facts (statement steps) + rules (antecedents+consequent
    on the IF coord channel, variables as their own atom) + the query triple (the
    ``is_q`` step). Terms are atomic ``filler_vec`` s; polarity is folded into the
    value atom (``v:+:kind`` vs ``v:-:kind``) so true/false reasoning is first-class.
    Options are the three answer atoms; the answer is the gold label index.
    """
    import numpy as np
    import torch
    from ...clause_reactor import ClauseBatch

    d = codec.dim
    E = lambda x: codec.filler_vec("e:" + x)
    R = lambda x: codec.filler_vec("r:" + x)
    SV = lambda o, pol: codec.filler_vec(f"v:{pol}:{o}")
    pred_is, pred_if, pred_q = codec.filler_vec("p:is"), codec.filler_vec("p:if"), codec.filler_vec("p:?")
    IFv = codec.filler_vec("IF")
    z = np.zeros(d, np.float32)
    opts = [codec.filler_vec("ans:" + a) for a in _ANS]

    rows = []
    for (facts, rules, query, ans_idx, _qd) in items:
        steps = []
        for (s, p, o, pol) in facts:
            steps.append((E(s), R(p), SV(o, pol), pred_is, z, 0))
        for rule in rules:
            for (s, p, o, pol) in rule.antecedents:
                steps.append((E(s), R(p), SV(o, pol), pred_if, IFv, 0))
            cs, cp, co, cpol = rule.consequent
            steps.append((E(cs), R(cp), SV(co, cpol), pred_if, IFv, 0))
        qs, qp, qo, qpol = query
        steps.append((E(qs), R(qp), SV(qo, qpol), pred_q, z, 1))
        rows.append((steps, ans_idx))

    b = len(rows)
    T = max(len(s) for s, _ in rows)
    ent = torch.zeros(b, T, d); rel = torch.zeros(b, T, d); val = torch.zeros(b, T, d)
    prd = torch.zeros(b, T, d); crd = torch.zeros(b, T, d)
    is_q = torch.zeros(b, T); mask = torch.zeros(b, T)
    options = torch.zeros(b, 3, d); answer = torch.zeros(b, dtype=torch.long)
    answerable = torch.ones(b)                       # Unknown is an option, not abstain
    for i, (steps, a) in enumerate(rows):
        for t, (e, r, v, p, c, q) in enumerate(steps):
            ent[i, t] = torch.from_numpy(e); rel[i, t] = torch.from_numpy(r)
            val[i, t] = torch.from_numpy(v); prd[i, t] = torch.from_numpy(p)
            crd[i, t] = torch.from_numpy(c); is_q[i, t] = q; mask[i, t] = 1.0
        for k, ov in enumerate(opts):
            options[i, k] = torch.from_numpy(ov)
        answer[i] = a
    return ClauseBatch(ent, rel, val, prd, is_q, mask, options, answer, crd, answerable)


# --------------------------------------------------------------------------- #
# M9 — proof-chain teacher supervision. Our forward_chain reproduces ProofWriter
# gold at 0.989 AND returns the derivation chain, so it is the proof teacher (no
# need to parse ProofWriter's allProofs). ProofWriter is attribute rule-chaining
# (furry→kind→smart): the discriminative per-hop signal is the derived VALUE.
# --------------------------------------------------------------------------- #
def _value_atom(lit: Literal) -> str:
    """The value atom build_pw_batch encodes for a literal — ``v:{pol}:{obj}``."""
    _s, _p, o, pol = lit
    return f"v:{pol}:{o}"


def proof_path(facts, rules, query):
    """The ordered derived literals on the proof path facts→query (excluding base
    facts), plus the label. For TRUE the path proves the query; for FALSE it proves
    the opposite polarity; for UNKNOWN there is no path (``[]``)."""
    known, chain = forward_chain(list(facts), list(rules))
    s, p, o, qpol = query
    opp = "-" if qpol == "+" else "+"
    if (s, p, o, qpol) in known:
        target, label = (s, p, o, qpol), TRUE
    elif (s, p, o, opp) in known:
        target, label = (s, p, o, opp), FALSE
    else:
        return [], UNKNOWN
    derived_by = {st.derived: st for st in chain}
    order = {st.derived: i for i, st in enumerate(chain)}
    needed, seen, frontier = [], set(), [target]
    while frontier:                                   # backward trace from the query
        lit = frontier.pop()
        if lit in seen:
            continue
        seen.add(lit)
        st = derived_by.get(lit)
        if st is None:
            continue                                  # a base fact, not a derivation
        needed.append(lit)
        frontier.extend(st.support)
    needed.sort(key=lambda l: order.get(l, 0))        # derivation order = facts→query
    return needed, label


def proof_rule_steps(facts, rules, query):
    """The proof as an ordered sequence of ``(current_facts, gold_rule_index)`` steps
    — the supervision for *learned navigation* (which rule to apply next). Rules are
    uniquely indexed so the firing rule is identifiable; ``current_facts`` is the
    closure just before that rule fires (teacher-forced state). Returns
    ``(steps, label)``; ``steps == []`` for Unknown (no proof to navigate)."""
    idx_rules = [Rule(r.antecedents, r.consequent, name=f"r{i}") for i, r in enumerate(rules)]
    known, chain = forward_chain(list(facts), idx_rules)
    s, p, o, qpol = query
    opp = "-" if qpol == "+" else "+"
    if (s, p, o, qpol) in known:
        target, label = (s, p, o, qpol), TRUE
    elif (s, p, o, opp) in known:
        target, label = (s, p, o, opp), FALSE
    else:
        return [], UNKNOWN
    derived_by = {st.derived: st for st in chain}
    order = {st.derived: i for i, st in enumerate(chain)}
    needed, seen, frontier = [], set(), [target]
    while frontier:                                   # which derivations the proof needs
        lit = frontier.pop()
        if lit in seen:
            continue
        seen.add(lit)
        st = derived_by.get(lit)
        if st is None:
            continue
        needed.append(lit)
        frontier.extend(st.support)
    needed.sort(key=lambda l: order.get(l, 0))        # derivation order = facts→query
    base = set(facts)
    steps = []
    for lit in needed:
        rule_idx = int(derived_by[lit].rule[1:])      # "r{i}" → i
        steps.append((sorted(base), rule_idx))        # state BEFORE this rule fires
        base = base | {lit}
    return steps, label


def value_codebook(items, codec):
    """The dataset's value-atom codebook ``([V,d] float32, {atom: idx})`` — every
    value atom appearing in facts, rule literals, and queries (covers derived
    values, which are rule consequents). Replaces M3's fixed relation codebook."""
    import numpy as np
    atoms = set()
    for (facts, rules, query, _a, _d) in items:
        for lit in facts:
            atoms.add(_value_atom(lit))
        for r in rules:
            for a in r.antecedents:
                atoms.add(_value_atom(a))
            atoms.add(_value_atom(r.consequent))
        atoms.add(_value_atom(query))
    ordered = sorted(atoms)
    cb = np.stack([codec.filler_vec(a) for a in ordered]).astype(np.float32)
    return cb, {a: i for i, a in enumerate(ordered)}


def _lit_vec(codec, lit):
    """A literal's content vector: subject + relation + signed-value atoms."""
    s, p, o, pol = lit
    sv = "e:?x" if s == "?x" else "e:" + s
    return codec.filler_vec(sv) + codec.filler_vec("r:" + p) + codec.filler_vec(f"v:{pol}:{o}")


def encode_rule(codec, rule):
    """A rule → a unit content vector (consequent emphasized): what it NEEDS
    (antecedents) and what it GIVES (consequent) — the signature the navigation
    policy matches against current facts + goal to pick the next move."""
    import numpy as np
    v = 2.0 * _lit_vec(codec, rule.consequent)
    for a in rule.antecedents:
        v = v + _lit_vec(codec, a)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


def build_proofsearch_batch(examples, codec):
    """A rule-SELECTION :class:`ClauseBatch`: stream = current facts + the goal
    (``is_q``); options = the theory's candidate rules (encoded); answer = the gold
    rule index for this step. Reuses the controller's contrastive head as the
    next-rule navigation policy. ``examples`` = ``(current_facts, goal, rules, gold_idx)``."""
    import numpy as np
    import torch
    from ...clause_reactor import ClauseBatch

    d = codec.dim
    E = lambda x: codec.filler_vec("e:" + x)
    R = lambda x: codec.filler_vec("r:" + x)
    SV = lambda o, pol: codec.filler_vec(f"v:{pol}:{o}")
    pred_is, pred_q = codec.filler_vec("p:is"), codec.filler_vec("p:?")
    z = np.zeros(d, np.float32)

    rows = []
    K = max(len(rules) for (_f, _g, rules, _i) in examples)
    for (facts, goal, rules, gold) in examples:
        steps = [(E(s), R(p), SV(o, pol), pred_is, z, 0) for (s, p, o, pol) in facts]
        gs, gp, go, gpol = goal
        steps.append((E(gs), R(gp), SV(go, gpol), pred_q, z, 1))
        opts = [encode_rule(codec, r) for r in rules] + [z] * (K - len(rules))
        rows.append((steps, opts, gold))

    b = len(rows)
    T = max(len(s) for s, _o, _g in rows)
    ent = torch.zeros(b, T, d); rel = torch.zeros(b, T, d); val = torch.zeros(b, T, d)
    prd = torch.zeros(b, T, d); crd = torch.zeros(b, T, d)
    is_q = torch.zeros(b, T); mask = torch.zeros(b, T)
    options = torch.zeros(b, K, d); answer = torch.zeros(b, dtype=torch.long)
    for i, (steps, opts, gold) in enumerate(rows):
        for t, (e, r, v, p, c, q) in enumerate(steps):
            ent[i, t] = torch.from_numpy(e); rel[i, t] = torch.from_numpy(r)
            val[i, t] = torch.from_numpy(v); prd[i, t] = torch.from_numpy(p)
            crd[i, t] = torch.from_numpy(c); is_q[i, t] = q; mask[i, t] = 1.0
        for k, ov in enumerate(opts):
            options[i, k] = torch.from_numpy(ov)
        answer[i] = gold
    return ClauseBatch(ent, rel, val, prd, is_q, mask, options, answer, crd, torch.ones(b))


def proof_supervision(items, hops: int, atom2idx):
    """Per-hop derived-value targets ``[B,hops]`` (value-codebook index, ``-1`` pad),
    ``depth [B]`` (proof steps; ``hops`` for Unknown so it runs the full budget), and
    ``answer [B]`` — the M9 analog of :func:`teacher.build_supervision` (value, not
    relation). Unknown items carry no value targets (the answer head learns ``idk``)."""
    import numpy as np
    b = len(items)
    vt = np.full((b, hops), -1, np.int64)
    depth = np.zeros(b, np.int64)
    answer = np.zeros(b, np.int64)
    for i, (facts, rules, query, a, _d) in enumerate(items):
        answer[i] = a
        needed, _label = proof_path(facts, rules, query)
        if needed:
            depth[i] = min(len(needed), hops)
            for k, lit in enumerate(needed[:hops]):
                vt[i, k] = atom2idx.get(_value_atom(lit), -1)
        else:
            depth[i] = hops                           # Unknown: spend the full budget
    return {"value_targets": vt, "depth": depth, "answer": answer}


def navigation_examples(items):
    """Expand ProofWriter items into per-step rule-selection training examples
    ``(current_facts, goal, rules, gold_rule_idx)`` — only provable (TRUE/FALSE)
    items yield steps (Unknown has no proof to navigate; the rollout abstains via
    the step budget)."""
    out = []
    for (facts, rules, query, _a, _d) in items:
        steps, _label = proof_rule_steps(facts, rules, query)
        for (current_facts, gold_idx) in steps:
            out.append((current_facts, query, rules, gold_idx))
    return out


__all__ = [
    "Literal", "PWExample", "parse_literal", "parse_rule", "parse_record",
    "load_records", "verify", "TRUE", "FALSE", "UNKNOWN", "default_data_dir",
    "flatten", "build_pw_batch", "proof_path", "value_codebook", "proof_supervision",
    "proof_rule_steps", "encode_rule", "build_proofsearch_batch", "navigation_examples",
]
