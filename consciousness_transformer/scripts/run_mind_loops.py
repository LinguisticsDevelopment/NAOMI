"""Run the M4 two-loop system — and show that it MATURES M3 over rounds.

The subconscious loop self-trains the M3 controller on replayed + freshly generated
episodes (accumulating iterations — the scale M3 lacked) and pre-derives multi-hop
conclusions into LTM (offline inference). Reports per-round held-out decode and
op-trace match: the "M4 helps M3" demonstration. Mirrors scripts/lifelong.py.

Run:
    python scripts/run_mind_loops.py [--rounds 12] [--steps 30]
        [--episodes-per-round 160] [--dim 48] [--hops 5] [--halting]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.mind.controller import MindController  # noqa: E402
from nsm_ct.mind.knowledge import KnowledgeGraph  # noqa: E402
from nsm_ct.mind.subconscious_loop import SubconsciousLoop  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

_REASONING = (9, 10, 11, 12, 13)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--episodes-per-round", type=int, default=160)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hops", type=int, default=5)
    ap.add_argument("--halting", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)

    codec = TPRCodec(dim=args.dim)
    # A fixed held-out val set from a disjoint seed (never trained on).
    val = [e for e in CurriculumGenerator(max_level=13, seed=999).generate(200)
           if e.level in _REASONING]

    ltm = KnowledgeGraph(codec=codec)
    controller = MindController(codec, hidden=96, hops=args.hops, halting=args.halting)
    sub = SubconsciousLoop(ltm, controller, codec=codec, seed=0, total_rounds=args.rounds)

    print(f"M4 two-loop run: {args.rounds} rounds x {args.steps} steps "
          f"(hops={args.hops}, halting={args.halting}); {len(val)} held-out val episodes")
    hist = sub.run(args.rounds, episodes_per_round=args.episodes_per_round,
                   steps=args.steps, val=val, verbose=True)

    first, last = hist[0], hist[-1]
    print("\n--- M4 matures M3 (held-out, across rounds) ---")
    print(f"  val decode      : {first.get('val_decode', 0):.2f} -> {last.get('val_decode', 0):.2f}")
    print(f"  val op-trace    : {first.get('val_optrace_match', 0):.2f} -> "
          f"{last.get('val_optrace_match', 0):.2f}")
    print(f"  LTM facts grown : {first['ltm_facts']} -> {last['ltm_facts']} "
          f"(offline-inferred accumulates)")


if __name__ == "__main__":
    main()
