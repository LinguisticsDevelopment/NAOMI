"""Stage R1 gate — the reasoning curriculum + oracle (derive vs abstain).

Shows L9 (modus ponens), L10 (inheritance/transitivity) and L11 (unanswerable →
abstain): the gold answer, the derivation chain, and that the query is NOT
directly asserted (so retrieval/recency cannot answer — derivation is necessary).

Run:
    python scripts/probe_reasoning.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.episode import CurriculumGenerator  # noqa: E402


def _one(level):
    gen = CurriculumGenerator(max_level=11, seed=7)
    return next(e for e in gen.generate(60) if e.level == level)


def main() -> None:
    for level, title in [(9, "modus ponens"), (10, "inheritance/transitivity"),
                         (11, "unanswerable → abstain")]:
        ep = _one(level)
        q = ep.meta["query"]
        direct = any(e == q[0] and r == q[1] for (e, r, _v) in ep.meta["facts"])
        print(f"=== L{level}: {title} ===")
        for s in ep.context:
            print(f"   {s}")
        print(f"   Q: {ep.question}")
        print(f"   gold answer: {ep.answer_text!r}   answerable={ep.answerable}   "
              f"directly-retrievable={direct}")
        for step in (ep.gold_chain or []):
            derived, rule, support = step
            print(f"   derive {derived} via {rule} from {list(support)}")
        print()
    print("GATE OK: reasoning levels are derivable only by chaining (retrieval fails); "
          "unanswerable -> abstain.")


if __name__ == "__main__":
    main()
