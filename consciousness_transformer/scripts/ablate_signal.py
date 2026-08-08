"""Run the Step B ablation for one signal module (dev/SEMANTIC_MAPPING_PLAN.md).

A signal lives at ``nsm_ct.ground.signal_<name>`` and exposes
``extras(vocab, graph) -> dict`` with any of: close_extra (list of word pairs),
antonym_extra (list of word pairs), feature_extra (word -> [category]).

Usage:
    python scripts/ablate_signal.py --signal genus [--n 3000] [--depth 3]
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.ablation import ablate, format_report  # noqa: E402
from nsm_ct.ground.axes import MeaningAxes  # noqa: E402
from nsm_ct.ground.cache import DecompCache  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.ground.relations import RelationGraph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    mod = importlib.import_module(f"nsm_ct.ground.signal_{args.signal}")

    t0 = time.time()
    vocab = gloss_vocabulary(args.n)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=args.depth).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)

    extras = mod.extras(vocab, g)
    res = ablate(vocab, g, ax, cache=cache, name=args.signal,
                 close_extra=extras.get("close_extra", ()),
                 antonym_extra=extras.get("antonym_extra", ()),
                 feature_extra=extras.get("feature_extra"),
                 depth=args.depth)
    print(format_report(res))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
