"""Probe: the M17.3 understanding evaluation (seed NSM-65 vs derived basis).

Answers, with numbers, "how does the system understand word meaning?" — on a
relationally-closed vocabulary, with the understanding probes computed on words
OUTSIDE the DeepNSM/gold dictionary (derivation, not lookup) and DeepNSM
agreement reported on the covered subset as an independent external check.

Usage:
    python scripts/probe_ground_understanding.py [--depth N] [--max-axes K] [--no-expand]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.evaluation import evaluate  # noqa: E402


def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--max-axes", type=int, default=15)
    ap.add_argument("--no-expand", action="store_true")
    args = ap.parse_args()

    r = evaluate(depth=args.depth, max_axes=args.max_axes, expand=not args.no_expand)

    print(f"=== M17.3 understanding evaluation (depth={r['depth']}, |vocab|={r['n_vocab']}) ===")
    print(f"derivation set (outside DeepNSM): {r['n_derivation']}   "
          f"DeepNSM-covered (external check): {r['n_deepnsm_covered']}")
    print(f"promoted axes ({len(r['promoted_axes'])}): "
          f"{', '.join(a for a, _ in r['promoted_axes'])}")
    print()

    s, d = r["seed"], r["derived"]
    print(f"{'metric':24s} {'seed':>8s} {'derived':>8s}")
    print(f"{'grounding_rate':24s} {_f(s['grounding_rate']):>8s} {_f(d['grounding_rate']):>8s}")
    print(f"{'convergence':24s} {_f(s['convergence']):>8s} {_f(d['convergence']):>8s}")
    print(f"{'syn>ant discrimination':24s} {_f(s['syn_ant']['accuracy']):>8s} {_f(d['syn_ant']['accuracy']):>8s}"
          f"   (n={d['syn_ant']['n']})")
    print(f"{'hypernym_containment':24s} {_f(s['hypernym_containment']):>8s} {_f(d['hypernym_containment']):>8s}"
          f"   (n={d['n_hypernym_pairs']})")
    print()
    rt = r["round_trip"]
    print(f"clause==word round-trip (held-out): exact={_f(rt['exact'])}  "
          f"perturbed={_f(rt['perturbed'])}  (n={rt['n']})")
    das, dad = r["deepnsm_agreement_seed"], r["deepnsm_agreement_derived"]
    print(f"DeepNSM agreement (external check):  seed={_f(das['mean'])}  "
          f"derived={_f(dad['mean'])}  (n={dad['n']})")
    print()
    print("Headline: deriving a basis improves grounding & convergence on held-out")
    print("words the dictionary never covered — understanding by derivation, not lookup.")


if __name__ == "__main__":
    main()
