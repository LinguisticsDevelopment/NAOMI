"""Probe: the honest baseline for word-meaning understanding (M17.0).

Prints the three lookup-free baseline metrics over a kind-spanning vocabulary:
- convergence       — does decomposition reach a stable prime fixpoint?
- prime_grounding   — what fraction of leaves actually reach a prime?
- deepnsm_agreement — independent external check vs the DeepNSM/gold dictionary.

M17.3 extends this to compare the derived-basis generator against this baseline.

Usage:
    python scripts/probe_ground_consistency.py [--depth N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground import clause_self_consistency as csc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    r = csc.report(depth=args.depth)

    print(f"=== M17.0 baseline (depth={r['depth']}, n={r['n_words']}) ===")
    print(f"mean convergence      : {r['mean_convergence']:.3f}")
    print(f"mean prime grounding  : {r['mean_prime_grounding']:.3f}")
    agree = r["mean_deepnsm_agreement"]
    agree_s = "n/a" if agree is None else f"{agree:.3f}"
    print(f"mean deepnsm agreement: {agree_s}  (covered {r['deepnsm_covered']}/{r['n_words']})")
    print()
    print(f"{'word':12s} {'conv':>5s} {'grnd':>5s} {'dnsm':>5s}  active_axes")
    for w, m in r["per_word"].items():
        conv = m["convergence"]
        grnd = m["prime_grounding"]
        dnsm = m["deepnsm_agreement"]
        grnd_s = " n/a " if grnd is None else f"{grnd:5.2f}"
        dnsm_s = " n/a " if dnsm is None else f"{dnsm:5.2f}"
        axes = ",".join(m["active_axes"][:6])
        print(f"{w:12s} {conv:5.2f} {grnd_s} {dnsm_s}  {axes}")


if __name__ == "__main__":
    main()
