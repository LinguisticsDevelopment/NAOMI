"""Build the DeepNSM explication store from HuggingFace dataset.

Downloads ``baartmar/nsm_dataset`` (cc-by-nc-sa-4.0; AI-GENERATED explications
from Gemini-2.0-Flash per arxiv 2505.11764) and writes a compact JSONL store to
``consciousness_transformer/data/deepnsm_explications.jsonl``.

Each line is a JSON object with fields: word, syn, explication, score, split.
The ``examples`` and ``ambig_examples`` columns are dropped to keep the file
small.

Attribution: This script uses the DeepNSM dataset (baartmar/nsm_dataset) by
Martinc et al., licensed CC BY-NC-SA 4.0. See consciousness_transformer/NOTICE.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_PATH = _REPO_ROOT / "data" / "deepnsm_explications.jsonl"


def main() -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading baartmar/nsm_dataset from HuggingFace…")
    splits = ("train", "validation", "test")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    counts: dict[str, int] = {}

    with open(_OUT_PATH, "w", encoding="utf-8") as fh:
        for split in splits:
            ds = load_dataset("baartmar/nsm_dataset", split=split)
            n = 0
            for row in ds:
                obj = {
                    "word": row["word"],
                    "syn": row.get("syn", ""),
                    "explication": row["explication"],
                    "score": row.get("score", None),
                    "split": split,
                }
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
            counts[split] = n
            total += n
            print(f"  {split:12s}: {n:6d} rows")

    print(f"\nTotal rows written: {total}")
    print(f"Output: {_OUT_PATH}")


if __name__ == "__main__":
    main()
