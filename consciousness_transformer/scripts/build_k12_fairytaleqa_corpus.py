"""Build the FairytaleQA sentence corpus for encoder gold v5 (branch
encoder-gold-v5-k12).

Reads the fetched FairytaleQA story sections (``data/k12/fairytaleqa/*-
story.csv``, via ``scripts/fetch_k12.py --source fairytaleqa
--allow-download``) and writes ``data/corpus/k12_fairytaleqa_{train,val,
test}.txt`` in the SAME convention ``scripts/fetch_corpus.py`` uses for
``data/corpus/real_gutenberg_*.txt`` (a ``#``-comment header line, then
blank-line-separated blocks = passages, raw untokenized prose --
``nsm_ct.corpus.load_corpus``/``iter_sentences`` do the tokenizing and
sentence-splitting at gold-build time, not this script). The split
(train/val/test) is FairytaleQA's own ``story_meta.csv`` split, so the
later comprehension phase can hold out stories, not just sentences.

One block per story SECTION (not per-story), in story_meta order, each
section's own text re-wrapped to a single whitespace-normalized line.
Within a section, sentences are split with the same deterministic
splitter the gold builder uses
(:func:`nsm_ct.corpus._split_raw_sentences`), so this script can dedupe
and holdout-filter at the EXACT sentence granularity the gold builder
will later see (its own dedup key is the tokenized/lowercased sentence
string, so dedup/holdout comparisons here use that same tokenized form,
not the raw surface text) -- a dropped sentence does not leave a gap in
its block; the remaining kept sentences of that section are rejoined.
A section that loses every sentence this way is dropped entirely (no
empty block emitted).

Dedup is GLOBAL across the whole fetch (first-story-in-story_meta-order,
first-section-in-file-order wins), mirroring ``load_corpus``'s own
first-file-wins convention. Sentences in ``runs/holdout_sentences.txt``
(the v4b-16k test split) or ``runs/holdout_dev_sentences.txt`` (regenerated
by this run against the v4b-16k gold, see RESEARCH_NOTES) are excluded
from every split so no later encoder arm can leak them.

A sidecar manifest (``runs/k12_fairytaleqa_manifest.jsonl``, one row per
KEPT sentence: ``{"sentence": <tokenized text>, "story_id": ..., "section_idx":
<0-based row index within that story's own *-story.csv>, "split": ...}``)
lets any gold record be traced back to its story/section.

Usage:
    python scripts/build_k12_fairytaleqa_corpus.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nsm_ct.corpus import _split_raw_sentences, _tokenize_words  # noqa: E402

IN_DIR = ROOT / "data" / "k12" / "fairytaleqa"
OUT_CORPUS_DIR = ROOT / "data" / "corpus"
OUT_MANIFEST = ROOT / "runs" / "k12_fairytaleqa_manifest.jsonl"
HOLDOUT_FILES = [ROOT / "runs" / "holdout_sentences.txt", ROOT / "runs" / "holdout_dev_sentences.txt"]


def tokenized(raw_sentence: str) -> str:
    return " ".join(_tokenize_words(raw_sentence))


def load_holdout() -> set:
    out = set()
    for p in HOLDOUT_FILES:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    out.add(line)
    return out


def main() -> None:
    story_meta_path = IN_DIR / "story_meta.csv"
    stories = list(csv.DictReader(story_meta_path.open(encoding="utf-8")))

    holdout = load_holdout()
    print(f"holdout sentences loaded: {len(holdout)}")

    seen_global: set = set()
    n_sections_total = 0
    n_sections_kept = 0
    n_sentences_seen = 0
    n_sentences_kept = 0
    n_dropped_dup = 0
    n_dropped_holdout = 0

    split_blocks = {"train": [], "val": [], "test": []}
    manifest_rows = []

    for story in stories:
        story_id = story["filename"]
        split = story["split"]
        if split not in split_blocks:
            continue
        story_csv = IN_DIR / f"{story_id}-story.csv"
        if not story_csv.exists():
            continue  # the one 404'd story (snow-man)
        rows = list(csv.DictReader(story_csv.open(encoding="utf-8")))
        for section_idx, row in enumerate(rows):
            n_sections_total += 1
            raw_text = " ".join(row["text"].split())
            raw_sentences = _split_raw_sentences(raw_text)
            kept_raw = []
            for raw_sent in raw_sentences:
                n_sentences_seen += 1
                tok = tokenized(raw_sent)
                if not tok:
                    continue
                if tok in holdout:
                    n_dropped_holdout += 1
                    continue
                if tok in seen_global:
                    n_dropped_dup += 1
                    continue
                seen_global.add(tok)
                kept_raw.append(raw_sent)
                n_sentences_kept += 1
                manifest_rows.append({
                    "sentence": tok,
                    "story_id": story_id,
                    "section_idx": section_idx,
                    "split": split,
                })
            if kept_raw:
                n_sections_kept += 1
                split_blocks[split].append(" ".join(kept_raw))

    OUT_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for split, blocks in split_blocks.items():
        n_sent = sum(len(_split_raw_sentences(b)) for b in blocks)
        header = (
            f"# K-12 FAIRYTALEQA (gold v5 corpus, encoder-gold-v5-k12) -- {split} split, "
            f"story-level (FairytaleQA's own story_meta.csv split, not re-derived), from "
            f"scripts/build_k12_fairytaleqa_corpus.py over data/k12/fairytaleqa/ "
            f"(scripts/fetch_k12.py --source fairytaleqa --allow-download). One block = one "
            f"story section (*-story.csv row); sentences are globally deduped across all "
            f"three split files (first-occurrence-in-story_meta-order wins, mirroring "
            f"nsm_ct.corpus.load_corpus's own first-file-wins convention) and sentences in "
            f"runs/holdout_sentences.txt / runs/holdout_dev_sentences.txt are excluded. "
            f"Traceability: runs/k12_fairytaleqa_manifest.jsonl maps every kept sentence back "
            f"to its story_id/section_idx/split. {n_sent} sentences, {len(blocks)} passages."
        )
        content = header + "\n\n" + "\n\n".join(blocks) + "\n"
        out_path = OUT_CORPUS_DIR / f"k12_fairytaleqa_{split}.txt"
        out_path.write_text(content, encoding="utf-8")
        print(f"wrote {out_path} ({n_sent} sentences, {len(blocks)} passages)")

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {OUT_MANIFEST} ({len(manifest_rows)} rows)")

    print(f"sections: {n_sections_total} total, {n_sections_kept} kept (>=1 surviving sentence)")
    print(f"sentences: {n_sentences_seen} seen, {n_sentences_kept} kept, "
          f"{n_dropped_dup} dropped (dup), {n_dropped_holdout} dropped (holdout)")


if __name__ == "__main__":
    main()
