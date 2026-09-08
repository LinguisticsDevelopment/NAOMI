"""Row-accounting regression test for scripts/convert_fairytaleqa.py::build_episodes.

Guards against the bug this audit found: an earlier fetch-time default
(``scripts/fetch_k12.py``'s ``--fairytaleqa-limit``, previously 100 of 278
stories) silently truncated the corpus long before conversion ever ran,
and nothing checked that every QA row was accounted for. ``build_episodes``
now labels every row it reads ``"emitted"`` or a named drop reason (see
dev/FAIRYTALEQA_STATS.md's "Row accounting" section) -- this test runs it
over the small, committed 10-story sample (no network, no full
``data/k12/fairytaleqa/`` fetch required) and checks the count is exact
and every reason is a real, named one (never the catch-all ``"other"``).
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from convert_fairytaleqa import (  # noqa: E402
    build_episodes,
    discover_story_ids,
    load_split_map,
)

_SAMPLE_DIR = Path(os.path.join(
    os.path.dirname(__file__), "..", "data", "k12_samples", "fairytaleqa_full_sample"))


def _require_sample():
    if not _SAMPLE_DIR.is_dir() or not (_SAMPLE_DIR / "story_meta.csv").exists():
        pytest.skip(f"{_SAMPLE_DIR} not present")


def test_emitted_plus_dropped_equals_questions_csv_rows():
    """Every QA row in the sample's *-questions.csv files must land in
    exactly one bucket (emitted or a drop reason), and the buckets must
    sum to the actual row count -- not a self-reported/manifest count."""
    _require_sample()
    story_ids = discover_story_ids(_SAMPLE_DIR)
    assert story_ids, "sample corpus is empty"
    split_map = load_split_map(_SAMPLE_DIR)

    expected_rows = 0
    for sid in story_ids:
        with open(_SAMPLE_DIR / f"{sid}-questions.csv", encoding="utf-8") as f:
            expected_rows += sum(1 for _ in csv.DictReader(f))

    # --parse none: row accounting doesn't depend on parse outcomes at all.
    episodes, _answer_type_counts, drop_reasons = build_episodes(
        _SAMPLE_DIR, story_ids, split_map, section_cache={})

    assert sum(drop_reasons.values()) == expected_rows
    assert drop_reasons.get("emitted", 0) == len(episodes)
    assert len(episodes) + sum(n for r, n in drop_reasons.items() if r != "emitted") == expected_rows


def test_no_unnamed_drop_reason():
    """Every drop must have a specific, named reason -- never a catch-all
    'other' bucket that would hide an unaudited code path."""
    _require_sample()
    story_ids = discover_story_ids(_SAMPLE_DIR)
    split_map = load_split_map(_SAMPLE_DIR)

    _episodes, _answer_type_counts, drop_reasons = build_episodes(
        _SAMPLE_DIR, story_ids, split_map, section_cache={})

    for reason in drop_reasons:
        assert reason == "emitted" or reason != "other", (
            f"unnamed drop reason {reason!r} -- give it a specific name")
