"""Probe: the unified relation store coverage at scale (M19.0).

Builds the RelationGraph over the gloss corpus and reports per-relation coverage
plus the candidate NAMED axes the relations hand us: the attribute dimensions and
the lexname categories. These become the interpretable axis set in M19.1.

Usage:
    python scripts/probe_relations.py [--n 3000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.ground.relations import RelationGraph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    vocab = gloss_vocabulary(args.n)
    t = time.time()
    g = RelationGraph.build(vocab)
    dt = time.time() - t

    print(f"=== M19.0 relation store ({len(g.words())} words, {dt:.0f}s) ===")
    print("per-relation coverage (fraction of words carrying it):")
    for r, c in g.coverage().items():
        print(f"  {r:14s} {c:.3f}")
    print()
    axes = g.attribute_axes()
    print(f"attribute axes (named dimensions): {len(axes)}")
    print(f"  top: {', '.join(axes[:20])}")
    lx = g.lexnames()
    print(f"lexname categories: {len(lx)}")
    print(f"  {', '.join(lx)}")
    print()
    print("in-vocab pair counts per word-word relation:")
    for r in ("synonym", "antonym", "similar", "is_a", "meronym", "derivational", "verb_group"):
        print(f"  {r:14s} {len(g.typed_pairs(r))}")


if __name__ == "__main__":
    main()
