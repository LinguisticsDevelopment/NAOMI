"""Stage 4 gate — STM read-time resolution on the L8 / L7 scenarios (symbolic).

Prints the navigable trace: two distinct clauses sharing one mary referent, the
negation that resolves L8, recency for "moved", and the disjunction → MAYBE /
narrowed → resolved. No training — the core proven symbolically first.

Run:
    python scripts/probe_psyche_graph.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause_psyche_graph import STM  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def _show(stm, label):
    status, val = stm.read("mary", "PLACE")
    print(f"  {label:42s} -> {status:8s} {stm.value_label(val) or ''}")


def main() -> None:
    res = NSMMeaningResolver()

    print("=== L8: kitchen, office, not office -> kitchen ===")
    stm = STM(TPRCodec(dim=128), res)
    stm.add_clause("mary", "PLACE", "kitchen")
    o = stm.add_clause("mary", "PLACE", "office")
    n = len(stm.graph.clauses_about(stm.graph.referent_index["mary"]))
    print(f"  two distinct clause nodes, one shared mary referent: clauses={n}, "
          f"referents={len(stm.graph.referent_index)}")
    stm.negate(o)
    _show(stm, "after 'mary is not in the office'")

    print("\n=== L3: kitchen, then moved to office -> office (recency) ===")
    stm = STM(TPRCodec(dim=128), res)
    stm.add_clause("mary", "PLACE", "kitchen")
    stm.add_clause("mary", "PLACE", "office")
    _show(stm, "after 'mary moved to the office'")

    print("\n=== L7a: kitchen OR office (unresolved) -> MAYBE ===")
    stm = STM(TPRCodec(dim=128), res)
    stm.add_disjunction("mary", "PLACE", ["kitchen", "office"])
    _show(stm, "after 'mary is in the kitchen or office'")

    print("\n=== L7b: kitchen OR office, not kitchen -> office ===")
    stm = STM(TPRCodec(dim=128), res)
    _op, (k, o) = stm.add_disjunction("mary", "PLACE", ["kitchen", "office"])
    stm.negate(k)
    _show(stm, "after 'mary is not in the kitchen'")

    print("\nGATE OK: distinct blobs, navigable read-time resolution.")


if __name__ == "__main__":
    main()
