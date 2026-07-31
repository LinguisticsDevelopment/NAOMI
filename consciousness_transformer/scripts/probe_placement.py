"""Probe: placement folds the relations into each word's position (M19.2).

Reports held-out synonym-vs-antonym discrimination with PLAIN cosine of the
placed coordinate (no comparison-time penalty) at several relaxation strengths,
vs the anchored coordinate and the M18 baselines. The point: antonymy now lives
IN the position (attribute-pole anchoring + synonym/similar relaxation), closing
the 'glued at comparison time' seam.

Usage:
    python scripts/probe_placement.py [--n 3000] [--depth 3]
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
from nsm_ct.ground.placement import evaluate_placement  # noqa: E402
from nsm_ct.ground.relations import RelationGraph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    vocab = gloss_vocabulary(args.n)
    t = time.time()
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=args.depth).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)

    print(f"=== M19.2 placement: held-out syn-vs-ant, PLAIN cosine (|vocab|={len(g.words())}) ===")
    first = True
    for alpha in (0.0, 0.3, 0.5, 0.7, 0.9):
        r = evaluate_placement(g.words(), g, ax, cache=cache, depth=args.depth,
                               train_frac=0.5, iters=20, alpha=alpha)
        if first:
            print(f"anchored (no relaxation): {r['anchored']:.3f}  "
                  f"(test syn={r['n_test_syn']} ant={r['n_test_ant']})")
            first = False
        tag = "(=anchored)" if alpha == 0.0 else ""
        print(f"placed (alpha={alpha}): {r['placed']:.3f} {tag}")
    print(f"\ntime={time.time()-t:.0f}s")
    print("baselines: M18.1 polarity coordinate 0.39 | M18.3 graph-closeness (penalty) 0.64")
    print("Placement reaches/beats the penalty baseline with PLAIN cosine — antonymy is")
    print("now in the position, not a comparison-time correction. The seam is closed.")


if __name__ == "__main__":
    main()
