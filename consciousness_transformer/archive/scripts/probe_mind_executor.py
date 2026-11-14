"""Probe: the deterministic mind executor over generated reasoning episodes.

Generates the L9-L13 reasoning curriculum (the levels whose gold answer comes from
``reasoning_oracle.derive`` over explicit triples), builds a gold op-trace for each
episode directly from its ``meta`` (PERCEIVE the facts → load the rules into LTM →
INFER → RESPOND the query), runs it through :class:`nsm_ct.mind.executor.Executor`,
and reports per-level solve / abstain accuracy against the oracle's gold answer.

This is the M2 "generalizes across generated episodes" check (the unit test
``tests/test_mind_executor.py`` is the hard gate). Mirrors ``scripts/probe_reasoning.py``.

Usage:  python scripts/probe_mind_executor.py [--n 400] [--seed 0]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from nsm_ct.episode import CurriculumGenerator
from nsm_ct.mind import ops
from nsm_ct.mind.executor import Executor
from nsm_ct.mind.knowledge import KnowledgeGraph
from nsm_ct import reasoning_oracle as ro


def _rules_for(ep):
    """The rules an episode reasons over, reconstructed from its level + meta."""
    if ep.level in (9, 11):                      # conditional: rule about place a → object x
        (_, _, a), (_, _, x) = ep.meta["rule"]
        return [ro.conditional_rule(a, x)]
    if ep.level == 10:
        return [ro.INHERITANCE]
    if ep.level == 12:
        return [ro.IS_A_TRANS, ro.INHERITANCE]
    if ep.level == 13:
        return [ro.Rule((a,), c, name="mp") for a, c in ep.meta["rules"]]
    return []


def _trace_for(ep):
    """A gold op-trace: perceive every fact, infer, respond the query."""
    trace = [ops.Op(ops.PERCEIVE, {"subject": s, "relation": r, "value": v})
             for (s, r, v) in ep.meta["facts"]]
    trace.append(ops.Op(ops.INFER, {}))
    qs, qr = ep.meta["query"]
    trace.append(ops.Op(ops.RESPOND, {"subject": qs, "relation": qr}))
    return trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gen = CurriculumGenerator(max_level=13, seed=args.seed)
    episodes = [e for e in gen.generate(args.n) if e.level in (9, 10, 11, 12, 13)]

    correct = defaultdict(int)
    total = defaultdict(int)
    for ep in episodes:
        ltm = KnowledgeGraph(dim=64)
        ltm.add_rules(_rules_for(ep))
        ex = Executor(ltm)
        out = ex.run_trace(_trace_for(ep))
        got = out["answer"]
        total[ep.level] += 1
        if got == ep.answer_text:
            correct[ep.level] += 1

    print(f"mind executor over {len(episodes)} reasoning episodes (oracle-graded):")
    overall_c = overall_t = 0
    for lvl in sorted(total):
        c, t = correct[lvl], total[lvl]
        overall_c += c
        overall_t += t
        tag = {9: "modus ponens", 10: "inheritance", 11: "abstain",
               12: "deep is-a chains", 13: "chained modus ponens"}[lvl]
        print(f"  L{lvl:<2} {tag:<22} {c}/{t}  ({c / max(t, 1):.2f})")
    print(f"  {'OVERALL':<26} {overall_c}/{overall_t}  ({overall_c / max(overall_t, 1):.2f})")


if __name__ == "__main__":
    main()
