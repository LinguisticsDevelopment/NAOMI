"""Run the lifelong loop: learn more and more over successive tests.

Run:
    python scripts/lifelong.py [--config configs/default.yaml]
                               [--rounds 12] [--episodes-per-round 32]
                               [--out runs/ltm.pt]

Each round trains on fresh episodes (parametric learning) and consolidates what
it absorbed into a persistent long-term memory (non-parametric growth). Watch the
LTM entries / connections grow round over round while accuracy improves.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.config import load_config  # noqa: E402
from nsm_ct.lifelong import run_lifelong  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Lifelong learning loop")
    ap.add_argument("--config", default=None)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--episodes-per-round", type=int, default=32)
    ap.add_argument("--out", default="runs/ltm.pt", help="Where to save the long-term repo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    _stack, history = run_lifelong(
        cfg,
        num_rounds=args.rounds,
        episodes_per_round=args.episodes_per_round,
        ltm_path=args.out,
    )

    first, last = history[0], history[-1]
    print(
        f"\nOver {len(history)} rounds: ans_acc {first['ans_acc']:.3f} -> {last['ans_acc']:.3f} "
        f"| long-term memory grew to {last['ltm_entries']} entries, "
        f"{last['ltm_connections']} connections."
    )


if __name__ == "__main__":
    main()
