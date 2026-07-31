"""Probe: does the derived basis CONVERGE as the corpus grows? (M18.0)

Runs MDL basis discovery over the gloss-vocabulary corpus at increasing sizes and
reports the promoted axes + grounding at each size, plus how stable the top axes
are across sizes (Jaccard overlap of the promoted sets). A stable basis is
evidence we are finding a real defining vocabulary, not corpus-size artefacts.

Usage:
    python scripts/probe_basis_convergence.py [--sizes 1000,5000,10000] [--max-axes 25] [--depth 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.basis_search import search  # noqa: E402
from nsm_ct.ground.cache import DecompCache  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.ground.definition_graph import DefinitionGraph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="1000,5000,10000")
    ap.add_argument("--max-axes", type=int, default=25)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    promoted_sets = {}
    for n in sizes:
        vocab = gloss_vocabulary(n)
        t = time.time()
        cache = DecompCache(depth=args.depth).warm(vocab)
        graph = DefinitionGraph.build(vocab)
        res = search(vocab, depth=args.depth, max_axes=args.max_axes, graph=graph, cache=cache)
        dt = time.time() - t
        promoted = [a for a, _ in res.registry.summary()]
        promoted_sets[n] = set(promoted)
        sm, fm = res.seed_metrics, res.final_metrics
        print(f"n={n:6d}  {dt:5.1f}s  grounding {sm['grounding_rate']:.3f}->{fm['grounding_rate']:.3f}  "
              f"axes={len(promoted)}")
        print(f"        {', '.join(promoted[:15])}")

    print("\nBasis stability (Jaccard overlap of promoted sets across sizes):")
    for i in range(len(sizes) - 1):
        a, b = promoted_sets[sizes[i]], promoted_sets[sizes[i + 1]]
        j = len(a & b) / len(a | b) if (a | b) else 1.0
        print(f"  {sizes[i]} vs {sizes[i+1]}: {j:.2f}")


if __name__ == "__main__":
    main()
