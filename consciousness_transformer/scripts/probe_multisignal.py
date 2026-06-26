"""Probe: multi-signal vs MDL-only basis on held-out relatedness (M18.2).

Grows a basis steering by synonym/hypernym relatedness on a TRAIN split, and
compares it against the MDL-only basis (M17.2) on a HELD-OUT split of relation
pairs. Honest expectation (consistent with M17.2): basis-axis selection has weak
control over relatedness, so the gain is small/mixed — the levers for relatedness
are the coordinate (M18.1 polarity) and the edges (M18.3 graph closeness).

Usage:
    python scripts/probe_multisignal.py [--n 2000] [--max-axes 15] [--depth 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.cache import DecompCache  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.ground.definition_graph import DefinitionGraph  # noqa: E402
from nsm_ct.ground.multisignal import multisignal_search  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--max-axes", type=int, default=15)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    vocab = gloss_vocabulary(args.n)
    graph = DefinitionGraph.build(vocab)
    cache = DecompCache(depth=args.depth).warm(vocab)

    t = time.time()
    res = multisignal_search(vocab, depth=args.depth, max_axes=args.max_axes,
                             graph=graph, cache=cache, train_frac=0.5)
    dt = time.time() - t

    mo, ms = res.mdl_only_metrics, res.multisignal_metrics
    print(f"=== M18.2 multi-signal vs MDL-only (|vocab|={len(vocab)}, {dt:.0f}s) ===")
    print(f"multisignal axes: {', '.join(a for a, _ in res.added)}")
    print()
    print(f"{'held-out metric':22s} {'MDL-only':>9s} {'multisignal':>12s}")
    print(f"{'synonym_cos':22s} {mo['synonym_cos']:9.3f} {ms['synonym_cos']:12.3f}   (n={mo['n_test_syn']})")
    print(f"{'hypernym_containment':22s} {mo['hypernym_containment']:9.3f} {ms['hypernym_containment']:12.3f}"
          f"   (n={mo['n_test_hyp']})")
    print()
    print("Honest: basis-axis selection only weakly controls relatedness; the gain")
    print("is small/mixed. Relatedness levers are the coordinate (M18.1) and edges (M18.3).")


if __name__ == "__main__":
    main()
