"""Builds the FairytaleQA fetch manifest + a small committed sample.

Reads ``data/k12/fairytaleqa/`` (populated by ``scripts/fetch_k12.py
--source fairytaleqa --allow-download``, gitignored -- see dev/
K12_CORPUS_SCOUT.md) and writes:

  1. ``data/k12_samples/fairytaleqa_full_sample/MANIFEST.json`` -- one row
     per story (its ``story_meta.csv`` fields + sha256 of both CSVs), plus
     corpus totals and the license string. Committed (small, no raw text).
  2. ``data/k12_samples/fairytaleqa_full_sample/*.csv`` -- the first
     ``--sample-n`` stories' story+questions CSVs, copied verbatim, so the
     repo carries a real, runnable slice of the full fetch (distinct from
     the pre-existing hand-picked ``data/k12_samples/fairytaleqa/``).

Usage:
    python scripts/build_fairytaleqa_manifest.py [--sample-n 10]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = _ROOT / "data" / "k12" / "fairytaleqa"
SAMPLE_DIR = _ROOT / "data" / "k12_samples" / "fairytaleqa_full_sample"

LICENSE = ("FairytaleQA (uci-soe/FairytaleQAData) -- Apache License 2.0 "
           "(https://github.com/uci-soe/FairytaleQAData/blob/main/LICENSE). "
           "Stories are 278 Project Gutenberg fairy tales (public domain); "
           "the QA annotations are the Apache-2.0-licensed original "
           "contribution of that repo.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(sample_n: int) -> None:
    meta_path = SRC_DIR / "story_meta.csv"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found -- run scripts/fetch_k12.py "
                          "--source fairytaleqa --allow-download first")

    rows = list(csv.DictReader(meta_path.open(encoding="utf-8")))
    stories = []
    total_sections = 0
    total_questions = 0
    fetched = 0
    for row in rows:
        fname = row["filename"]
        story_csv = SRC_DIR / f"{fname}-story.csv"
        q_csv = SRC_DIR / f"{fname}-questions.csv"
        entry = {
            "story_id": fname,
            "origin": row["origin"],
            "split": row["split"],
            "sections": int(row["sections"]),
            "words": int(row["words"]),
            "questions": int(row["questions"]),
            "fetched": story_csv.exists() and q_csv.exists(),
        }
        if entry["fetched"]:
            entry["story_sha256"] = _sha256(story_csv)
            entry["questions_sha256"] = _sha256(q_csv)
            total_sections += entry["sections"]
            total_questions += entry["questions"]
            fetched += 1
        stories.append(entry)

    manifest = {
        "source": "FairytaleQA (github.com/uci-soe/FairytaleQAData)",
        "license": LICENSE,
        "stories_indexed": len(rows),
        "stories_fetched": fetched,
        "total_sections": total_sections,
        "total_questions": total_questions,
        "stories": stories,
    }

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLE_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SAMPLE_DIR / 'MANIFEST.json'}: {fetched}/{len(rows)} stories, "
          f"{total_sections} sections, {total_questions} questions")

    # Deterministic 10-story sample: first N fetched stories in story_meta.csv order.
    sample_written = 0
    sample_meta_rows = []
    meta_fieldnames = list(rows[0].keys()) if rows else []
    for row, entry in zip(rows, stories):
        if sample_written >= sample_n:
            break
        if not entry["fetched"]:
            continue
        fname = entry["story_id"]
        for suffix in ("story", "questions"):
            src = SRC_DIR / f"{fname}-{suffix}.csv"
            shutil.copy(src, SAMPLE_DIR / src.name)
        sample_meta_rows.append(row)
        sample_written += 1
    print(f"copied {sample_written} sample stories -> {SAMPLE_DIR}")

    # A slim story_meta.csv scoped to just the sample -- keeps
    # scripts/convert_fairytaleqa.py's --in-dir contract usable directly
    # against this committed sample directory (tests/test_fairytaleqa.py).
    with (SAMPLE_DIR / "story_meta.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=meta_fieldnames)
        writer.writeheader()
        writer.writerows(sample_meta_rows)

    readme = SAMPLE_DIR / "SOURCE.md"
    readme.write_text(
        "# FairytaleQA -- full-fetch manifest + sample\n\n"
        f"MANIFEST.json indexes all {len(rows)} stories from the full "
        f"`scripts/fetch_k12.py --source fairytaleqa --allow-download` fetch "
        f"({fetched} fetched successfully -- see dev/FAIRYTALEQA_STATS.md for "
        "why any are missing) with per-story split/section/question counts "
        "and sha256 of both CSVs. The raw fetch itself lives in "
        "`data/k12/fairytaleqa/` (gitignored -- re-fetch with the command "
        "above). The CSVs alongside this file are a deterministic "
        f"{sample_written}-story slice (first fetched stories in "
        "story_meta.csv order), copied verbatim, distinct from the "
        "pre-existing hand-picked `data/k12_samples/fairytaleqa/`.\n\n"
        f"- **License**: {LICENSE}\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-n", type=int, default=10)
    args = ap.parse_args()
    build(args.sample_n)


if __name__ == "__main__":
    main()
