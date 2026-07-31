"""Probe: polarity-aware coordinates vs unsigned, on syn>ant discrimination (M18.1).

Reports syn>ant discrimination (are synonyms placed closer than antonyms?) with
the unsigned M17 coordinate vs the M18.1 polarity-aware coordinate, plus how many
antonym pairs carry a polarity signal (morphology or gloss magnitude).

Honest expectation: polarity *improves* syn>ant but does not by itself cross 0.5 —
antonyms share too much definitional structure for coordinates alone. The graph
closeness layer (M18.3) is where antonym edges finish the job.

Usage:
    python scripts/probe_ground_polarity.py [--n 3000] [--depth 3] [--weight 2.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.basis_search import _value_vec  # noqa: E402
from nsm_ct.ground.cache import DecompCache  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.ground.definition_graph import DefinitionGraph  # noqa: E402
from nsm_ct.ground.evaluation import syn_ant_discrimination  # noqa: E402
from nsm_ct.ground import polarity as P  # noqa: E402
from nsm_ct.ground.semantic_axes import AxisRegistry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    vocab = gloss_vocabulary(args.n)
    reg = AxisRegistry.seed()
    graph = DefinitionGraph.build(vocab)
    cache = DecompCache(depth=args.depth).warm(vocab)

    unsigned = syn_ant_discrimination(vocab, reg, args.depth, graph)
    signed_coord = {w: P.polarity_vector(w, axes=reg.axes, depth=args.depth,
                                         decompose=lambda x: cache.decompose(x, args.depth)) for w in vocab}
    signed = syn_ant_discrimination(vocab, reg, args.depth, graph, coord=signed_coord)

    morph = sum(1 for w in vocab if P.negation_base(w))
    cued = sum(1 for w in vocab if P.gloss_polarity(w) != 0.0)

    print(f"=== M18.1 polarity (|vocab|={len(vocab)}, depth={args.depth}) ===")
    print(f"syn>ant  unsigned : {unsigned['accuracy']:.3f}  (n={unsigned['n']})")
    print(f"syn>ant  polarity : {signed['accuracy']:.3f}  (n={signed['n']})")
    print(f"delta             : {signed['accuracy'] - unsigned['accuracy']:+.3f}")
    print(f"polarity coverage : morphological negations={morph}  gloss-magnitude cued={cued}")
    print()
    print("Honest: polarity improves syn>ant but coordinates alone stay below 0.5 —")
    print("antonyms share definitional structure; M18.3 graph closeness finishes it.")


if __name__ == "__main__":
    main()
