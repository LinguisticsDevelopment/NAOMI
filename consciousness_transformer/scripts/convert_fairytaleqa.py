"""FairytaleQA -> NAOMI Episode converter (K12 comprehension-QA campaign,
dev/K12_CORPUS_SCOUT.md's #1 pick).

Reads the full fetch (``data/k12/fairytaleqa/`` -- ``scripts/fetch_k12.py
--source fairytaleqa --allow-download``) and produces ONE
:class:`~nsm_ct.episode.Episode` per (story section(s), question) pair:

* ``context``  -- the tokenized sentences of the question's own section(s)
  (:func:`nsm_ct.corpus.iter_sentences`, the same tokenization the M58a
  Gutenberg-prose converter uses). A ``local`` question's context is its
  ONE ``cor_section``; a ``summary`` question's context is the CONCATENATION
  of all sections it cites, in story order.
* ``question`` -- the dataset's own question text, verbatim.
* ``answer_text`` -- the first annotator's first answer (``answer1``,
  falling back to the second annotator's ``answer4`` when the first is
  blank -- a handful of rows have only one annotator filled in). NEVER
  invented, synthesized, or paraphrased.
* ``options``/``answer_idx`` -- set ONLY for episodes whose gold answer is
  classified ``entity`` (see :func:`classify_answer_type`): gold + up to 3
  same-``attribute`` distractor answers drawn from elsewhere in the corpus
  (mirrors ``nsm_ct.corpus.make_episodes``' own same-relation distractor
  convention). Every other episode carries ``options=None`` (open-ended;
  NAOMI's current MC-only QA head can't score these yet -- see
  dev/FAIRYTALEQA_STATS.md's answer-type breakdown for why).
* ``meta`` -- ``story_id``, ``section_id`` (the cor_section list, as given),
  ``ex_or_im`` (annotator 1's ``ex-or-im1``; ``ex_or_im2`` also carried),
  ``attribute`` (normalized -- see :data:`_ATTRIBUTE_MAP``), ``attribute2``
  (raw, when a second attribute is marked), ``local_or_sum``, ``split``
  (the dataset's own train/val/test, from ``story_meta.csv`` -- NOT
  re-derived), ``answer_type``, ``all_gold_answers`` (every non-empty
  ``answer1..6``, deduped), ``parse_stats`` (per-episode taxonomy over its
  own passage's sentences -- see below), ``fully_parseable``.

**Parsing**: every SECTION in the corpus is parsed exactly once (through
``nsm_ct.corpus.parse_passage`` -- the real parser path, default
``CORPUS_MAX_PARSE_SECONDS``/``CORPUS_MAX_HYPOTHESES`` caps, unmodified) and
cached by ``(story_id, section_id)``; a summary question's passage stats
are the UNION of its cited sections' cached per-sentence outcomes (the
pronoun registry resets at each section boundary either way -- sections are
treated as the atomic parse unit, not the multi-section summary passage,
so this union is exact, not an approximation). This is what makes parsing
the FULL corpus tractable at all: 10,556 questions cite only ~4,082
distinct sections, a ~2.6x reduction, and every section is parsed once
regardless of how many questions cite it. Parsing itself is CPU-bound and
slow (quantum_parser's real grammar, not a toy tagger) -- this script
shards sections across ``--workers`` processes (default: all CPU cores).

Usage:
    python scripts/convert_fairytaleqa.py \\
        --in-dir data/k12/fairytaleqa --out runs/fairytaleqa_episodes.jsonl \\
        --workers 4 --parse sample

``--parse`` (``none``/``sample``/default ``sample``/``all``) makes per-sentence
parsing OPTIONAL, independent of episode construction (context/question/
answer/metadata are always built from every episode regardless of this
flag): ``none`` parses nothing at all; ``sample`` additionally parses a
stratified random ~2,000-sentence sample (seeded, by passage-length bin,
:func:`select_sample_sections`) so the corpus parse-yield table stays
meaningful without the ~31K-sentence full-corpus cost; ``all`` is the
original full-corpus behavior. See dev/FAIRYTALEQA_STATS.md.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.corpus import _PassageRegistry, _parse_one_sentence, iter_sentences  # noqa: E402
from nsm_ct.episode import Episode  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN_DIR = _ROOT / "data" / "k12" / "fairytaleqa"

# "usable" (a real fact was extracted, ambiguous or pronoun-resolved
# included) vs. "strict full-tree" (unambiguous, no pronoun fallback) --
# see module docstring / dev/FAIRYTALEQA_STATS.md's own 4-bucket table.
_USABLE_OUTCOMES = {"ok", "parsed-ambiguous", "parsed-pronoun-resolved"}
_STRICT_OUTCOME = "ok"
_CAP_OUTCOME = "parse-resource-capped"

_ATTRIBUTE_MAP = {
    "character": "character",
    "setting": "setting",
    "action": "action",
    "causal relationship": "causal",
    "outcome resolution": "outcome",
    "outcome solution": "outcome",  # rare typo variant in the source data
    "prediction": "prediction",
    "feeling": "feeling",
    "problem": "other",
    "": "",
}


def normalize_attribute(raw: str) -> str:
    return _ATTRIBUTE_MAP.get((raw or "").strip().lower(), "other")


# ---------------------------------------------------------------------------
# answer-type classification (item 3's (a)-(d) breakdown)
# ---------------------------------------------------------------------------

_COMMON_VERB_TOKENS = {
    "is", "was", "were", "are", "am", "be", "been", "went", "said", "gave",
    "took", "had", "have", "has", "made", "did", "saw", "came", "ran",
    "walked", "asked", "told", "found", "got", "put", "let", "began",
    "looked", "wanted", "knew", "thought", "felt", "cried", "called",
    "gave", "helped", "loved", "killed", "died", "married", "lived",
    "turned", "fell", "flew", "brought", "sent", "left", "stayed",
    "wept", "laughed", "cut", "broke", "opened", "closed",
}


def classify_answer_type(answer: str, passage: str) -> str:
    """(a) entity, (b) verb_phrase, (c) free_text, (d) not_substring.

    A heuristic, not a parser: ``entity`` = a short (<=4 token), verbless
    noun phrase that appears verbatim in the passage; ``verb_phrase`` = a
    short (<=6 token) span headed by a finite/participial verb; ``free_text``
    = a longer span or one ending in sentence punctuation; ``not_substring``
    = the gold text (normalized) does not appear in the passage at all
    (implicit-inference answers routinely fall here -- the point of the
    ``explicit``/``implicit`` distinction FairytaleQA itself already marks).
    """
    a = (answer or "").strip()
    if not a:
        return "not_substring"
    norm_passage = " ".join(passage.lower().split())
    norm_answer = " ".join(a.lower().rstrip(".").split())
    if not norm_answer or norm_answer not in norm_passage:
        return "not_substring"
    tokens = norm_answer.split()
    n = len(tokens)
    ends_sentence = a.rstrip().endswith((".", "!", "?"))
    has_verb_tok = any(t in _COMMON_VERB_TOKENS or t.endswith(("ed", "ing")) for t in tokens)
    starts_verby = tokens[0] in _COMMON_VERB_TOKENS or tokens[0].endswith(("ed", "ing"))
    if n <= 4 and not has_verb_tok and not ends_sentence:
        return "entity"
    if n <= 6 and starts_verby and not ends_sentence:
        return "verb_phrase"
    if ends_sentence or n > 6:
        return "free_text"
    if has_verb_tok:
        return "verb_phrase"
    return "free_text"


# ---------------------------------------------------------------------------
# corpus loading
# ---------------------------------------------------------------------------

def load_split_map(in_dir: Path) -> Dict[str, str]:
    meta_path = in_dir / "story_meta.csv"
    out = {}
    with meta_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["filename"]] = row["split"]
    return out


def load_story_sections(in_dir: Path, story_id: str) -> Dict[str, List[str]]:
    """``{section_id: [tokenized sentences]}`` for one story, in file order."""
    path = in_dir / f"{story_id}-story.csv"
    out: Dict[str, List[str]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["section"]] = iter_sentences(row["text"])
    return out


def load_story_questions(in_dir: Path, story_id: str) -> List[dict]:
    path = in_dir / f"{story_id}-questions.csv"
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def discover_story_ids(in_dir: Path) -> List[str]:
    ids = sorted(p.name[: -len("-story.csv")] for p in in_dir.glob("*-story.csv"))
    return [i for i in ids if (in_dir / f"{i}-questions.csv").exists()]


# ---------------------------------------------------------------------------
# parallel per-section parsing
# ---------------------------------------------------------------------------

# Stratified-sample defaults (--parse sample): a fixed seed keeps the sample
# (and therefore the corpus parse-yield table) reproducible run to run.
_SAMPLE_SIZE_SENTENCES = 2000
_SAMPLE_SEED = 0


def _section_length_bin(n_sent: int) -> str:
    if n_sent <= 3:
        return "short (<=3 sent)"
    if n_sent <= 7:
        return "medium (4-7 sent)"
    if n_sent <= 15:
        return "long (8-15 sent)"
    return "very_long (16+ sent)"


def select_sample_sections(in_dir: Path, story_ids: List[str], sample_size: int,
                            seed: int) -> Set[Tuple[str, str]]:
    """``--parse sample``: a stratified-by-passage-length-bin random sample
    targeting ``sample_size`` SENTENCES total, sampled at whole-SECTION
    granularity -- a section is the atomic parse unit (module docstring),
    so sampling whole sections reproduces exactly what parsing just those
    sections in isolation would produce. Each length bin contributes
    sections (seed-shuffled) proportional to its own share of the corpus's
    total sentences, until that bin's target is hit or the bin runs out.
    """
    import random

    all_sections: List[Tuple[str, str, int]] = []
    for sid in story_ids:
        for sec_id, sents in load_story_sections(in_dir, sid).items():
            if sents:
                all_sections.append((sid, sec_id, len(sents)))

    total_sents = sum(n for _, _, n in all_sections)
    if total_sents <= sample_size:
        return {(sid, sec) for sid, sec, _ in all_sections}

    by_bin: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    for row in all_sections:
        by_bin[_section_length_bin(row[2])].append(row)

    rng = random.Random(seed)
    selected: Set[Tuple[str, str]] = set()
    for rows in by_bin.values():
        bin_total = sum(n for _, _, n in rows)
        target = round(sample_size * bin_total / total_sents)
        rng.shuffle(rows)
        got = 0
        for sid, sec, n in rows:
            if got >= target:
                break
            selected.add((sid, sec))
            got += n
    return selected


_MAX_SENTENCE_LENGTH = 100  # quantum_parser's own config.max_sentence_length default


def _parse_section_safe(sents: List[str], parser) -> List[str]:
    """One outcome tag per sentence of ONE section -- calls
    ``nsm_ct.corpus._parse_one_sentence`` per sentence (rather than
    ``parse_passage``'s single call over the whole section) so a
    pathological sentence can't crash the whole shard/worker.

    Two failure modes need special handling here, both seen on this
    corpus's real prose:

    * ``quantum_parser``'s own ``ValueError("Sentence too long (...)"``
      guard (``config.max_sentence_length``, default 100 tokens) is
      already caught INSIDE ``ParserInputEncoder._parse_topk_one`` (a
      broad ``except Exception``) and silently downgraded to "no
      hypotheses" -- it never reaches this function as a raised
      exception at all, and would otherwise land in the generic
      ``"no-parse"`` bucket, indistinguishable from every other
      no-hypotheses cause. Checked directly here (same threshold, same
      whitespace tokenization ``_parse_one_sentence_uncapped`` itself
      uses) BEFORE calling the parser at all, so it gets its own
      ``"sentence-too-long"`` tag and skips a doomed parse attempt.
    * A ``RecursionError`` from deep tree-walking in
      ``extract_discourse``/``_extract_triples`` (downstream of the
      parser's own broad catch, so NOT swallowed there) -- caught here
      and tagged ``"max-recursion-depth"`` instead of crashing the
      worker process/pool.
    """
    registry = _PassageRegistry()
    tags: List[str] = []
    for idx, sent in enumerate(sents):
        if len(sent.split()) > _MAX_SENTENCE_LENGTH:
            tags.append("sentence-too-long")
            continue
        try:
            results = _parse_one_sentence(idx, sent, parser, registry=registry)
        except RecursionError:
            tags.append("max-recursion-depth")
            continue
        except Exception:  # noqa: BLE001 -- deliberate catch-all safety net
            tags.append("unexpected-error")
            continue
        if not results:
            tags.append("no-relation-extracted")
            continue
        # A single _parse_one_sentence call is always homogeneous: either a
        # list of ParsedClause (all sharing the same ambiguous flag, set
        # uniformly across `triples`) or a single-element ParseFailure list
        # -- so the first result's tag applies to the whole sentence.
        first = results[0]
        if first.__class__.__name__ == "ParsedClause":
            if first.hypotheses is not None:
                tags.append("parsed-ambiguous")
            elif first.pronoun_candidates is not None:
                tags.append("parsed-pronoun-resolved")
            else:
                tags.append("ok")
        else:
            tags.append(first.reason)
    return tags


def _parse_shard(args: Tuple[Path, List[str], Optional[Set[Tuple[str, str]]]]
                  ) -> Dict[Tuple[str, str], List[str]]:
    """Worker: parses the requested sections of its assigned stories --
    ``only`` is ``None`` for every section of every assigned story
    (``--parse all``), or a specific ``{(story_id, section_id)}`` subset
    (``--parse sample``), letting the same shard machinery serve both.
    Returns ``{(story_id, section_id): [outcome_tag_per_sentence]}``."""
    in_dir, story_ids, only = args
    all_sents: List[str] = []
    per_story_sections: List[Tuple[str, Dict[str, List[str]]]] = []
    for sid in story_ids:
        sections = load_story_sections(in_dir, sid)
        if only is not None:
            sections = {sec: sents for sec, sents in sections.items() if (sid, sec) in only}
        per_story_sections.append((sid, sections))
        for sents in sections.values():
            all_sents.extend(sents)

    tok = SimpleTokenizer.build(all_sents, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="en")

    out: Dict[Tuple[str, str], List[str]] = {}
    for sid, sections in per_story_sections:
        for sec_id, sents in sections.items():
            if not sents:
                out[(sid, sec_id)] = []
                continue
            tags = _parse_section_safe(sents, parser)
            assert len(tags) == len(sents)
            out[(sid, sec_id)] = tags
    return out


def parse_all_sections(in_dir: Path, story_ids: List[str], workers: int,
                        mode: str = "all", sample_size: int = _SAMPLE_SIZE_SENTENCES,
                        seed: int = _SAMPLE_SEED) -> Dict[Tuple[str, str], List[str]]:
    """``mode``: ``"none"`` -- no parsing at all (returns ``{}``); ``"sample"``
    -- parses a stratified ``sample_size``-sentence sample only (see
    :func:`select_sample_sections`); ``"all"`` (default, matches the
    pre-``--parse``-flag behavior every caller not passing ``mode`` still
    gets) -- parses every section of every given story."""
    if mode not in ("none", "sample", "all"):
        raise ValueError(f"unknown parse mode {mode!r}")
    if mode == "none":
        return {}

    only: Optional[Set[Tuple[str, str]]] = None
    shard_story_ids = story_ids
    if mode == "sample":
        only = select_sample_sections(in_dir, story_ids, sample_size, seed)
        only_story_ids = {sid for sid, _ in only}
        shard_story_ids = [sid for sid in story_ids if sid in only_story_ids]

    if workers <= 1:
        return _parse_shard((in_dir, shard_story_ids, only))
    shards = [shard_story_ids[i::workers] for i in range(workers)]
    shards = [s for s in shards if s]
    t0 = time.time()
    with mp.get_context("spawn").Pool(processes=len(shards)) as pool:
        results = pool.map(_parse_shard, [(in_dir, s, only) for s in shards])
    cache: Dict[Tuple[str, str], List[str]] = {}
    for r in results:
        cache.update(r)
    print(f"[convert_fairytaleqa] parsed {sum(len(v) for v in cache.values())} sentences "
          f"across {len(cache)} sections in {time.time() - t0:.1f}s "
          f"({workers} workers, --parse {mode})", flush=True)
    return cache


# ---------------------------------------------------------------------------
# episode construction
# ---------------------------------------------------------------------------

def _gather_answers(row: dict) -> Tuple[str, List[str]]:
    a1 = (row.get("answer1") or "").strip()
    a4 = (row.get("answer4") or "").strip()
    primary = a1 or a4
    all_answers: List[str] = []
    for key in ("answer1", "answer2", "answer3", "answer4", "answer5", "answer6"):
        v = (row.get(key) or "").strip()
        if v and v not in all_answers:
            all_answers.append(v)
    return primary, all_answers


def build_episodes(in_dir: Path, story_ids: List[str], split_map: Dict[str, str],
                    section_cache: Dict[Tuple[str, str], List[str]]) -> Tuple[List[Episode], Counter]:
    episodes: List[Episode] = []
    answer_type_counts: Counter = Counter()

    for sid in story_ids:
        sections = load_story_sections(in_dir, sid)
        questions = load_story_questions(in_dir, sid)
        split = split_map.get(sid, "unknown")
        for row in questions:
            cor_secs = [p.strip() for p in row["cor_section"].split(",") if p.strip()]
            cor_secs = sorted(set(cor_secs), key=lambda s: int(s))
            context: List[str] = []
            for sec in cor_secs:
                context.extend(sections.get(sec, []))
            if not context:
                continue

            tags: List[str] = []
            for sec in cor_secs:
                tags.extend(section_cache.get((sid, sec), []))
            n_sent = len(tags)
            outcome_counts = Counter(tags)
            fully_parseable = n_sent > 0 and all(t in _USABLE_OUTCOMES for t in tags)
            parse_stats = {
                "n_sentences": n_sent,
                "strict_full_tree": outcome_counts.get(_STRICT_OUTCOME, 0),
                "usable": sum(outcome_counts.get(k, 0) for k in _USABLE_OUTCOMES),
                "cap_hit": outcome_counts.get(_CAP_OUTCOME, 0),
                "histogram": dict(outcome_counts),
            }

            primary_answer, all_answers = _gather_answers(row)
            passage_text = " ".join(context)
            answer_type = classify_answer_type(primary_answer, passage_text)
            answer_type_counts[answer_type] += 1

            attribute = normalize_attribute(row.get("attribute1", ""))
            meta = {
                # "prose" (not a bespoke "fairytaleqa" kind): routes through
                # nsm_ct.clause_reactor.build_clause_batch's _prose_steps --
                # the SAME real-prose pronoun-candidate-set grounding
                # nsm_ct.corpus's own Gutenberg-prose episodes get (see that
                # function's docstring), not the plainer default/else path.
                "kind": "prose",
                "dataset": "fairytaleqa",
                "source_doc": f"fairytaleqa/{sid}",
                "story_id": sid,
                "section_id": cor_secs,
                "local_or_sum": row["local-or-sum"],
                "ex_or_im": row.get("ex-or-im1", ""),
                "ex_or_im2": row.get("ex-or-im2", ""),
                "attribute": attribute,
                "attribute_raw": row.get("attribute1", ""),
                "attribute2": row.get("attribute2", ""),
                "split": split,
                "answer_type": answer_type,
                "all_gold_answers": all_answers,
                "parse_stats": parse_stats,
                "fully_parseable": fully_parseable,
                "question_id": row.get("question_id", ""),
            }
            episodes.append(Episode(
                context=context,
                question=row["question"],
                answer_text=primary_answer,
                options=None,
                answer_idx=None,
                level=0,
                meta=meta,
            ))

    return episodes, answer_type_counts


def attach_entity_options(episodes: List[Episode], seed: int = 0, n_distractors: int = 3) -> int:
    """M58d-style MC options for the ``entity``-answer-type subset only
    (the ONLY subset NAOMI's current single-entity-answer QA head can score
    -- see module docstring / dev/FAIRYTALEQA_STATS.md). Distractors: other
    entity-type gold answers of the SAME normalized attribute, corpus-wide,
    deduped, deterministically shuffled per episode. An episode with fewer
    than ``n_distractors`` available same-attribute distractors keeps
    ``options=None`` (open-ended; not MC-scoreable). Returns the count of
    episodes that got options attached.
    """
    import random

    by_attribute: Dict[str, List[str]] = defaultdict(list)
    for ep in episodes:
        if ep.meta.get("answer_type") == "entity" and ep.answer_text:
            by_attribute[ep.meta["attribute"]].append(ep.answer_text)

    n_attached = 0
    for i, ep in enumerate(episodes):
        if ep.meta.get("answer_type") != "entity" or not ep.answer_text:
            continue
        gold = ep.answer_text
        pool = [v for v in by_attribute.get(ep.meta["attribute"], []) if v.lower() != gold.lower()]
        seen = {gold.lower()}
        distractors: List[str] = []
        rng = random.Random(seed + i)
        rng.shuffle(pool)
        for v in pool:
            if v.lower() in seen:
                continue
            seen.add(v.lower())
            distractors.append(v)
            if len(distractors) >= n_distractors:
                break
        if len(distractors) < 2:
            continue
        options = distractors + [gold]
        rng.shuffle(options)
        ep.options = options
        ep.answer_idx = options.index(gold)
        n_attached += 1
    return n_attached


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out", type=Path, default=_ROOT / "runs" / "fairytaleqa_episodes.jsonl")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit-stories", type=int, default=0, help="0 = all stories.")
    ap.add_argument("--parse", type=str, default="sample", choices=["none", "sample", "all"],
                     help="none = every episode still gets its full passage/question/answer/"
                          "metadata, just no parse_stats outcomes (no parsing at all -- fast). "
                          "sample (default) = additionally parses a stratified random "
                          f"~{_SAMPLE_SIZE_SENTENCES}-sentence sample (seeded, by passage-length "
                          "bin) for the corpus parse-yield table -- only THOSE sentences get an "
                          "outcome. all = parse every sentence of every section (slow: ~31K "
                          "sentences corpus-wide -- the original, pre-flag behavior).")
    args = ap.parse_args()

    if not args.in_dir.exists():
        raise SystemExit(f"{args.in_dir} not found -- run scripts/fetch_k12.py "
                          "--source fairytaleqa --allow-download first")

    story_ids = discover_story_ids(args.in_dir)
    if args.limit_stories:
        story_ids = story_ids[: args.limit_stories]
    split_map = load_split_map(args.in_dir)
    print(f"[convert_fairytaleqa] {len(story_ids)} stories", flush=True)

    section_cache = parse_all_sections(args.in_dir, story_ids, args.workers, mode=args.parse)
    episodes, answer_type_counts = build_episodes(args.in_dir, story_ids, split_map, section_cache)
    n_attached = attach_entity_options(episodes)
    print(f"[convert_fairytaleqa] {len(episodes)} episodes, "
          f"{n_attached} with MC options (answer_type=entity, >=2 distractors)", flush=True)
    print(f"[convert_fairytaleqa] answer_type: {dict(answer_type_counts)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(dataclasses.asdict(ep)) + "\n")
    print(f"[convert_fairytaleqa] wrote {len(episodes)} episodes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
