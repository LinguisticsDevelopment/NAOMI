"""Tests for the FairytaleQA -> Episode converter (scripts/convert_fairytaleqa.py).

Runs entirely against the small, committed sample corpus (10 stories) under
``data/k12_samples/fairytaleqa_full_sample/`` -- no network access, no full
``data/k12/fairytaleqa/`` fetch required. Covers: conversion determinism,
every episode carrying non-empty passage/question/answer + the expected
metadata fields, split labels matching the sample's own ``story_meta.csv``,
and the answer-type classifier's basic (a)/(d) cases.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from convert_fairytaleqa import (  # noqa: E402
    build_episodes,
    classify_answer_type,
    discover_story_ids,
    load_split_map,
    normalize_attribute,
    parse_all_sections,
)

_SAMPLE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "k12_samples", "fairytaleqa_full_sample")


def _require_sample():
    if not os.path.isdir(_SAMPLE_DIR) or not os.path.exists(os.path.join(_SAMPLE_DIR, "story_meta.csv")):
        pytest.skip(f"{_SAMPLE_DIR} not present")


def _convert(workers=1):
    from pathlib import Path
    in_dir = Path(_SAMPLE_DIR)
    story_ids = discover_story_ids(in_dir)
    split_map = load_split_map(in_dir)
    section_cache = parse_all_sections(in_dir, story_ids, workers)
    episodes, answer_type_counts, drop_reasons = build_episodes(in_dir, story_ids, split_map, section_cache)
    return episodes, answer_type_counts, story_ids, split_map, drop_reasons


# Session-scoped: the real quantum_parser conversion over even 10 stories
# takes minutes (CORPUS_MAX_PARSE_SECONDS-capped sentences dominate) -- one
# shared conversion for every non-determinism test, rather than one per test.
@pytest.fixture(scope="module")
def converted():
    _require_sample()
    return _convert()


# ---------------------------------------------------------------------------
# 1. conversion determinism
# ---------------------------------------------------------------------------

def test_conversion_is_deterministic(converted):
    eps_a, _counts_a, _ids_a, _split_a, _drops_a = converted
    eps_b, _counts_b, _ids_b, _split_b, _drops_b = _convert()
    assert len(eps_a) == len(eps_b)
    assert len(eps_a) > 0
    for a, b in zip(eps_a, eps_b):
        assert a.context == b.context
        assert a.question == b.question
        assert a.answer_text == b.answer_text
        assert a.options == b.options
        assert a.answer_idx == b.answer_idx
        assert a.meta == b.meta


# ---------------------------------------------------------------------------
# 2. every episode: non-empty fields + metadata contract
# ---------------------------------------------------------------------------

_REQUIRED_META_KEYS = {
    "story_id", "section_id", "local_or_sum", "ex_or_im", "ex_or_im2",
    "attribute", "split", "answer_type", "all_gold_answers", "parse_stats",
    "fully_parseable",
}


def test_every_episode_has_required_fields(converted):
    episodes, _counts, _ids, _split_map, _drops = converted
    assert episodes
    for ep in episodes:
        assert ep.context and all(s.strip() for s in ep.context)
        assert ep.question.strip()
        assert ep.answer_text.strip()
        assert _REQUIRED_META_KEYS <= set(ep.meta.keys())
        assert ep.meta["answer_type"] in {"entity", "verb_phrase", "free_text", "not_substring"}
        assert isinstance(ep.meta["section_id"], list) and ep.meta["section_id"]
        assert isinstance(ep.meta["fully_parseable"], bool)
        if ep.options is not None:
            assert ep.answer_idx is not None
            assert 0 <= ep.answer_idx < len(ep.options)
            assert ep.options[ep.answer_idx] == ep.answer_text
            assert ep.meta["answer_type"] == "entity"


def test_one_episode_per_story_section_question_pair(converted):
    import csv
    episodes, _counts, story_ids, _split_map, _drops = converted
    expected = 0
    for sid in story_ids:
        with open(os.path.join(_SAMPLE_DIR, f"{sid}-questions.csv"), encoding="utf-8") as f:
            expected += sum(1 for _ in csv.DictReader(f))
    assert len(episodes) == expected


# ---------------------------------------------------------------------------
# 3. split labels match the dataset's own story_meta.csv
# ---------------------------------------------------------------------------

def test_split_labels_match_dataset(converted):
    episodes, _counts, _ids, split_map, _drops = converted
    assert set(split_map.values()) <= {"train", "val", "test"}
    for ep in episodes:
        assert ep.meta["split"] == split_map[ep.meta["story_id"]]


# ---------------------------------------------------------------------------
# 4. answer-type classifier
# ---------------------------------------------------------------------------

def test_classify_answer_type_entity():
    passage = "dullhead met a little grey man in the forest ."
    assert classify_answer_type("a little grey man", passage) == "entity"


def test_classify_answer_type_not_substring():
    passage = "dullhead met a little grey man in the forest ."
    assert classify_answer_type("he was hungry and afraid", passage) == "not_substring"


def test_classify_answer_type_empty_answer():
    assert classify_answer_type("", "some passage text .") == "not_substring"


def test_normalize_attribute_maps_known_and_unknown():
    assert normalize_attribute("causal relationship") == "causal"
    assert normalize_attribute("outcome resolution") == "outcome"
    assert normalize_attribute("Character") == "character"
    assert normalize_attribute("something-weird") == "other"
