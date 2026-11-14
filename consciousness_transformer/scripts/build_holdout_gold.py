"""Build gold records for an EXPLICIT sentence list (one sentence per line,
``scripts/make_holdout.py``'s own output format) rather than a corpus glob.

Reuses ``scripts/build_encoder_gold_v2.py``'s own ``load_usvs``/``_new_parser``/
``process_sentences`` (same per-sentence code path, same JSONL record shape)
so a holdout rebuild is byte-identical in format to the real
``GOLD_CORPUS_GLOB``-driven pipeline -- this script only swaps where the
sentence list comes from. Single-process only (holdout lists are small;
no ``--workers`` sharding needed).

Usage:
    python scripts/build_holdout_gold.py \
        --sentences runs/holdout_sentences.txt \
        --out runs/holdout_gold_v6.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_encoder_gold_v2 import (  # noqa: E402
    FOREST_MARGIN_DEFAULT, FOREST_MODE_DEFAULT, USVS_DIR, _new_parser, process_sentences,
)
from nsm_ct.ground.usvs import load_usvs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentences", required=True, help="one sentence per line (e.g. runs/holdout_sentences.txt)")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--forest", default=FOREST_MODE_DEFAULT, choices=("top1", "margin", "all"))
    ap.add_argument("--forest-margin", type=float, default=FOREST_MARGIN_DEFAULT)
    args = ap.parse_args()

    sentences = [s.strip() for s in Path(args.sentences).read_text(encoding="utf-8").splitlines() if s.strip()]
    print(f"{len(sentences)} sentences loaded from {args.sentences}", flush=True)

    usvs = load_usvs(USVS_DIR)
    parser = _new_parser()
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable; aborting")
        return 1

    out_path = Path(args.out)
    stats = process_sentences(usvs, parser, sentences, args.forest, args.forest_margin, out_path)
    n_ok = len(stats["n_trees"])
    print(f"wrote {n_ok}/{len(sentences)} records -> {out_path}", flush=True)
    print(f"outcomes: {stats['outcomes']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
