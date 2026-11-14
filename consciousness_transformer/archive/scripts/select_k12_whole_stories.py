"""Pick a seeded (seed 0 default) random subset of WHOLE K-12 FairytaleQA
train-split stories (never a partial story) that fits a sentence budget --
the same "keep all Gutenberg, restrict K-12 to a story subset" policy
``dev/ENCODER_GOLD_V5_K12_STATS.md``'s 81-story pilot used, generalized into
a reusable, reproducible script (that run's subset was hand-picked once and
not committed as a script).

Reads ``runs/k12_fairytaleqa_manifest.jsonl`` (sentence -> story_id/
section_idx/split trace, already committed) rather than re-parsing the
corpus text file, so story boundaries are exact. Writes the chosen stories'
sentences, one per line (ready for ``build_encoder_gold_v2.py --extra-glob``),
and prints the chosen story-id list for the record.

Usage:
    python scripts/select_k12_whole_stories.py \
        --manifest runs/k12_fairytaleqa_manifest.jsonl --split train \
        --exclude-file runs/holdout_sentences.txt --exclude-file runs/holdout_dev_sentences.txt \
        --budget 14000 --seed 0 --out /tmp/k12_train_subset.txt
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--exclude-file", action="append", default=[],
                     help="newline-delimited sentence file to exclude; may be given multiple times.")
    ap.add_argument("--budget", type=int, required=True, help="target sentence count (soft cap; stops at the "
                     "first story whose inclusion would meet or exceed it).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output: chosen stories' sentences, one per line.")
    args = ap.parse_args()

    exclude = set()
    for path in args.exclude_file:
        exclude |= {ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()}

    by_story: "defaultdict[str, List[str]]" = defaultdict(list)
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] != args.split:
                continue
            s = rec["sentence"].strip()
            if not s or s in exclude:
                continue
            by_story[rec["story_id"]].append(s)

    story_ids = sorted(by_story.keys())
    rng = random.Random(args.seed)
    rng.shuffle(story_ids)

    chosen: List[str] = []
    total = 0
    for sid in story_ids:
        sents = by_story[sid]
        if chosen and total + len(sents) > args.budget:
            continue
        chosen.append(sid)
        total += len(sents)
        if total >= args.budget:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for sid in chosen:
        lines.extend(by_story[sid])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(story_ids)} candidate {args.split}-split stories (post-exclude)")
    print(f"chosen: {len(chosen)} whole stories, {total} sentences (budget {args.budget}, seed {args.seed})")
    print(f"wrote -> {out_path}")
    print("story ids:", ", ".join(sorted(chosen)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
