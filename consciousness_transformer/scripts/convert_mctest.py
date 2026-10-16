"""MCTest -> NAOMI Episode converter (dev/reports/K12_CORPUS_SCOUT.md's #2 pick,
dev/reports/K12_BENCHMARK_FIX.md item 1c).

Unlike FairytaleQA, MCTest ships its OWN in-story 4-way MC distractors --
no distractor-synthesis policy is needed or applied here; this script is a
pure format conversion into the same :class:`~nsm_ct.episode.Episode`
schema ``scripts/convert_fairytaleqa.py`` emits, so
``scripts/cheat_baselines_k12.py`` and ``scripts/eval_prose.py`` both work
on it unchanged.

Input: ``data/k12_samples/mctest/`` (or any dir fetched by
``scripts/fetch_k12.py --source mctest --allow-download``) -- ``mc160.{train,
dev}.tsv`` + matching ``.ans`` files. Each ``.tsv`` row is one story:
``id \\t author/worktime metadata \\t story text (literal "\\newline" line
breaks) \\t (tag: question \\t opt_A \\t opt_B \\t opt_C \\t opt_D) x4``.
Each ``.ans`` row is 4 gold letters (A-D), one per question, in the same
tab-separated order as its ``.tsv`` row. ``tag`` is ``"one"`` (answerable
from one sentence) or ``"multiple"`` (needs combining several) -- kept
verbatim in ``meta["sentence_tag"]``.

Usage:
    python scripts/convert_mctest.py --in-dir data/k12_samples/mctest \\
        --variant mc160 --splits train dev --out runs/mctest_mc160_episodes.jsonl
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.corpus import iter_sentences  # noqa: E402
from nsm_ct.episode import Episode  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN_DIR = _ROOT / "data" / "k12_samples" / "mctest"
_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
_N_QUESTIONS_PER_STORY = 4
_N_OPTIONS = 4


def _load_tsv_rows(path: Path) -> List[List[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f, delimiter="\t") if row]


def _load_ans_rows(path: Path) -> List[List[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f, delimiter="\t") if row]


def convert_split(in_dir: Path, variant: str, split: str) -> List[Episode]:
    """One ``(variant, split)`` pair, e.g. ``("mc160", "train")`` ->
    one Episode per (story, question) pair -- 4 per story, fixed schema."""
    tsv_path = in_dir / f"{variant}.{split}.tsv"
    ans_path = in_dir / f"{variant}.{split}.ans"
    if not tsv_path.exists() or not ans_path.exists():
        print(f"[convert_mctest] SKIP {variant}.{split}: missing {tsv_path.name} or {ans_path.name}")
        return []

    tsv_rows = _load_tsv_rows(tsv_path)
    ans_rows = _load_ans_rows(ans_path)
    if len(tsv_rows) != len(ans_rows):
        raise ValueError(f"{tsv_path.name}: {len(tsv_rows)} stories but {ans_path.name} has "
                          f"{len(ans_rows)} answer rows")

    episodes: List[Episode] = []
    for row, ans in zip(tsv_rows, ans_rows):
        story_id, _meta, story_text = row[0], row[1], row[2]
        rest = row[3:]
        if len(rest) != _N_QUESTIONS_PER_STORY * (1 + _N_OPTIONS):
            raise ValueError(f"{story_id}: expected "
                              f"{_N_QUESTIONS_PER_STORY * (1 + _N_OPTIONS)} question/option "
                              f"columns, got {len(rest)}")
        if len(ans) != _N_QUESTIONS_PER_STORY:
            raise ValueError(f"{story_id}: expected {_N_QUESTIONS_PER_STORY} gold letters, got {len(ans)}")

        context = iter_sentences(story_text.replace("\\newline", " "))
        for qi in range(_N_QUESTIONS_PER_STORY):
            base = qi * (1 + _N_OPTIONS)
            tagged_question = rest[base]
            options = rest[base + 1: base + 1 + _N_OPTIONS]
            sentence_tag, _, question_text = tagged_question.partition(":")
            sentence_tag = sentence_tag.strip().lower()
            question_text = question_text.strip()

            gold_letter = ans[qi].strip().upper()
            if gold_letter not in _LETTER_TO_IDX:
                raise ValueError(f"{story_id} q{qi}: unknown gold letter {gold_letter!r}")
            answer_idx = _LETTER_TO_IDX[gold_letter]
            answer_text = options[answer_idx]

            episodes.append(Episode(
                context=context,
                question=question_text,
                answer_text=answer_text,
                options=list(options),
                answer_idx=answer_idx,
                level=0,
                meta={
                    "kind": "prose",
                    "dataset": "mctest",
                    "source_doc": f"mctest/{variant}/{story_id}",
                    "story_id": story_id,
                    "variant": variant,
                    "split": split,
                    "sentence_tag": sentence_tag,  # "one" | "multiple"
                    "question_index": qi,
                },
            ))
    print(f"[convert_mctest] {variant}.{split}: {len(tsv_rows)} stories -> "
          f"{len(episodes)} episodes", flush=True)
    return episodes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--variant", type=str, default="mc160", choices=["mc160", "mc500"])
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "dev"],
                     choices=["train", "dev", "test"])
    ap.add_argument("--out", type=Path, default=_ROOT / "runs" / "mctest_mc160_episodes.jsonl")
    args = ap.parse_args()

    if not args.in_dir.exists():
        raise SystemExit(f"{args.in_dir} not found -- run scripts/fetch_k12.py "
                          "--source mctest --allow-download first, or point --in-dir at "
                          "data/k12_samples/mctest")

    episodes: List[Episode] = []
    for split in args.splits:
        episodes.extend(convert_split(args.in_dir, args.variant, split))

    tag_counts = {}
    for ep in episodes:
        tag_counts[ep.meta["sentence_tag"]] = tag_counts.get(ep.meta["sentence_tag"], 0) + 1
    print(f"[convert_mctest] total: {len(episodes)} episodes across splits {args.splits}; "
          f"sentence_tag breakdown: {tag_counts}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(dataclasses.asdict(ep)) + "\n")
    print(f"[convert_mctest] wrote {len(episodes)} episodes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
