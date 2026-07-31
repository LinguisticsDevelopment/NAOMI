"""Probe: the minimum named axes that reproduce the relations (M19.3).

Reports the fidelity-vs-#axes curve (held-out synonym-vs-antonym discrimination as
axes are added back by importance) and the minimal named axis subset reaching most
of the full fidelity — the empirical "minimum axes of meaning", every axis named.

Usage:
    python scripts/probe_minimality.py [--n 3000] [--keep-frac 0.95]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.axes import MeaningAxes  # noqa: E402
from nsm_ct.ground.cache import DecompCache  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.ground.minimality import minimal_axes  # noqa: E402
from nsm_ct.ground.relations import RelationGraph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--keep-frac", type=float, default=0.95)
    args = ap.parse_args()

    vocab = gloss_vocabulary(args.n)
    t = time.time()
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    r = minimal_axes(g.words(), g, ax, cache=cache, depth=3, alpha=0.7, keep_frac=args.keep_frac)
    dt = time.time() - t

    print(f"=== M19.3 minimum axes ({dt:.0f}s) ===")
    print(f"candidate named axes: {r['n_axes']}   full held-out fidelity: {r['full_discrimination']:.3f}")
    print(f"minimal axes for {int(args.keep_frac*100)}% fidelity: {r['minimal_k']}")
    print()
    print("fidelity vs #axes:")
    for k, d in r["curve"]:
        print(f"  K={k:4d}  disc={d:.3f}")
    print()
    print(f"the {r['minimal_k']} kept axes (all named/interpretable):")
    print("  " + ", ".join(r["kept_axes"]))


if __name__ == "__main__":
    main()
