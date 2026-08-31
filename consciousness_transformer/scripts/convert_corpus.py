"""M58a corpus->episode converter CLI (see dev/AURORA_SPRINT.md M58,
dev/NEXT_ARC_PLAN.md M57 "the corpus campaign").

Reads plain-text files (each blank-line-separated block = one passage),
runs every passage through :mod:`nsm_ct.corpus` (the SAME parser path
``nsm_ct.clause_reactor._context_steps`` uses), and:

  1. writes converted :class:`~nsm_ct.episode.Episode` objects (self-
     generated comprehension questions, no hand labeling) to a JSONL file;
  2. prints the taxonomy stats block (sentences total, parsed-ok %, failure
     histogram, episodes produced, questions per relation) -- split out
     synthetic vs. real text by filename prefix (``synthetic_*`` vs
     everything else);
  3. writes dev/PROSE_FAILURE_TAXONOMY.md: the histogram + up to 3 verbatim
     example sentences per failure class -- THE real failure taxonomy the
     week-2 worklist gets built from.

Usage:
    python scripts/convert_corpus.py --in "data/corpus/*.txt" \\
        --out runs/prose_episodes.jsonl --stats
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.corpus import (  # noqa: E402
    FAILURE_REASONS,
    ParsedClause,
    ParseFailure,
    iter_sentences,
    make_episodes,
    parse_passage,
    taxonomy_counts,
)
from nsm_ct.episode import Episode  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402


def load_passages(path: str) -> List[str]:
    """A file's blank-line-separated blocks, ``#``-comment lines stripped."""
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln for ln in text.split("\n") if not ln.strip().startswith("#")]
    text = "\n".join(lines)
    blocks = [b.strip() for b in text.split("\n\n")]
    return [b for b in blocks if b]


def corpus_group(filename: str) -> str:
    base = os.path.basename(filename)
    return "synthetic" if base.startswith("synthetic_") else "real"


def convert(in_glob: str, seed: int = 0):
    """Returns (episodes, per_group_taxonomy, per_group_totals, failure_examples,
    per_relation_questions, all_passage_results)."""
    files = sorted(glob.glob(in_glob))
    if not files:
        raise SystemExit(f"no files matched {in_glob!r}")

    all_sentences: List[str] = []
    passages: List[dict] = []  # {doc_id, group, sentences}
    for fpath in files:
        for pi, block in enumerate(load_passages(fpath)):
            sents = iter_sentences(block)
            if not sents:
                continue
            doc_id = f"{Path(fpath).stem}#{pi}"
            passages.append({"doc_id": doc_id, "group": corpus_group(fpath), "sentences": sents})
            all_sentences.extend(sents)

    tok = SimpleTokenizer.build(all_sentences, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="en")
    if getattr(parser, "_parser", None) is None:
        raise SystemExit("quantum_parser unavailable in this environment")

    # Pass 1: parse every passage.
    for p in passages:
        p["results"] = parse_passage(p["sentences"], parser)

    # Global distractor pool per relation (for the "or corpus" fallback),
    # excluding each passage's own clauses when it's used for that passage.
    all_clauses: List[ParsedClause] = [
        r for p in passages for r in p["results"] if isinstance(r, ParsedClause)
    ]

    # Pass 2: make_episodes per passage, seeded deterministically per doc_id.
    episodes: List[Episode] = []
    reject_counts: Dict[str, Counter] = defaultdict(Counter)  # M58d: episode-quality-filter rejections, per group
    for i, p in enumerate(passages):
        own_ids = {id(r) for r in p["results"]}
        pool = [c for c in all_clauses if id(c) not in own_ids]
        group_stats = Counter()
        eps = make_episodes(p["results"], holdout="last", seed=seed + i,
                             doc_id=p["doc_id"], distractor_pool=pool, reject_stats=group_stats)
        episodes.extend(eps)
        reject_counts[p["group"]] += group_stats
        reject_counts["all"] += group_stats

    # Stats, split by group.
    group_taxonomy: Dict[str, Counter] = defaultdict(Counter)
    group_sentence_total: Dict[str, int] = defaultdict(int)
    failure_examples: Dict[str, List[tuple]] = defaultdict(list)  # reason -> [(doc_id, sentence, detail)]
    for p in passages:
        counts = taxonomy_counts(p["results"])
        group_taxonomy[p["group"]] += counts
        group_taxonomy["all"] += counts
        group_sentence_total[p["group"]] += sum(counts.values())
        group_sentence_total["all"] += sum(counts.values())
        for r in p["results"]:
            if isinstance(r, ParseFailure) and len(failure_examples[r.reason]) < 3:
                failure_examples[r.reason].append((p["doc_id"], r.sentence, r.detail))

    relation_counts: Dict[str, Counter] = defaultdict(Counter)
    for ep in episodes:
        grp = "synthetic" if ep.meta.get("source_doc", "").startswith("synthetic_") else "real"
        relation_counts[grp][ep.meta["relation"]] += 1
        relation_counts["all"][ep.meta["relation"]] += 1

    return episodes, group_taxonomy, group_sentence_total, failure_examples, relation_counts, passages, reject_counts


def print_stats(group_taxonomy, group_sentence_total, relation_counts, episodes, reject_counts=None):
    reject_counts = reject_counts or {}
    for grp in ("all", "synthetic", "real"):
        if group_sentence_total.get(grp, 0) == 0:
            continue
        total = group_sentence_total[grp]
        counts = group_taxonomy[grp]
        ok = counts.get("ok", 0)
        print(f"\n=== {grp} ===")
        print(f"sentences total: {total}")
        print(f"parsed ok: {ok}/{total} ({ok/total:.1%})")
        print("failure histogram:")
        for reason in FAILURE_REASONS:
            c = counts.get(reason, 0)
            if c:
                print(f"  {reason:<28} {c:>4}  ({c/total:.1%})")
        n_eps = sum(1 for ep in episodes
                    if (grp == "all")
                    or (grp == "synthetic" and ep.meta.get("source_doc", "").startswith("synthetic_"))
                    or (grp == "real" and not ep.meta.get("source_doc", "").startswith("synthetic_")))
        print(f"episodes produced: {n_eps}")
        rejected = reject_counts.get(grp, Counter()).get("episode-rejected-quality", 0)
        if rejected:
            print(f"episode-rejected-quality: {rejected}  (candidate clauses rejected by the M58d quality filter)")
        print("questions per relation:")
        for rel, c in sorted(relation_counts[grp].items(), key=lambda kv: -kv[1]):
            print(f"  {rel:<12} {c}")


def write_taxonomy_doc(path: str, group_taxonomy, group_sentence_total, failure_examples):
    lines = [
        "# Prose failure taxonomy (M58a)",
        "",
        "Generated by `scripts/convert_corpus.py --stats` from the corpus in",
        "`data/corpus/`. Every sentence in the corpus gets EXACTLY one outcome:",
        "`ok` (parsed_passage produced at least one (entity, relation, value)",
        "triple) or one of the six failure reasons below. This is the REAL",
        "failure map week 2's parser/extraction round is driven by (see",
        "dev/AURORA_SPRINT.md \"Week 2\").",
        "",
        "## Histogram",
        "",
    ]
    for grp in ("all", "synthetic", "real"):
        total = group_sentence_total.get(grp, 0)
        if not total:
            continue
        counts = group_taxonomy[grp]
        lines.append(f"### {grp} ({total} sentences)")
        lines.append("")
        lines.append(f"- ok: {counts.get('ok', 0)} ({counts.get('ok', 0)/total:.1%})")
        for reason in FAILURE_REASONS:
            c = counts.get(reason, 0)
            lines.append(f"- {reason}: {c} ({c/total:.1%})")
        lines.append("")

    lines.append("## Examples (up to 3 verbatim per failure class, corpus-wide)")
    lines.append("")
    for reason in FAILURE_REASONS:
        examples = failure_examples.get(reason, [])
        lines.append(f"### {reason}")
        lines.append("")
        if not examples:
            lines.append("_(no occurrences in this corpus)_")
        for doc_id, sent, detail in examples:
            detail_s = f" -- {detail}" if detail else ""
            lines.append(f"- `{doc_id}`: \"{sent}\"{detail_s}")
        lines.append("")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_glob", default="data/corpus/*.txt")
    ap.add_argument("--out", default="runs/prose_episodes.jsonl")
    ap.add_argument("--taxonomy-out", default="dev/PROSE_FAILURE_TAXONOMY.md")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    episodes, group_taxonomy, group_sentence_total, failure_examples, relation_counts, _, reject_counts = \
        convert(args.in_glob, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(dataclasses.asdict(ep)) + "\n")
    print(f"Wrote {len(episodes)} episodes to {out_path}")

    write_taxonomy_doc(args.taxonomy_out, group_taxonomy, group_sentence_total, failure_examples)
    print(f"Wrote failure taxonomy to {args.taxonomy_out}")

    if args.stats:
        print_stats(group_taxonomy, group_sentence_total, relation_counts, episodes, reject_counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
