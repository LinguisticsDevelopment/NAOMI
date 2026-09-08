"""Tests for the shared held-out sentence workflow (dev/AUDIT_2026-09-08.md
finding 7 + recommendation (c)): scripts/make_holdout.py and the
nsm_ct.encoder_train_util helpers scripts/train_encoder.py /
scripts/colab_train_encoder.py use for --holdout-file / --n-train
--subset-seed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nsm_ct import encoder_train_util as etu

GOLD = ROOT / "runs" / "encoder_gold_v2.jsonl"


def _skip_if_no_gold():
    if not GOLD.exists():
        pytest.skip("runs/encoder_gold_v2.jsonl not present in this checkout")


def test_make_holdout_matches_seeded_test_split(tmp_path):
    """scripts/make_holdout.py's default output must be byte-for-byte the
    sentence texts of train_encoder.py's own stratified_split test split at
    the same (seed, n_train, n_dev, n_test) -- it is claimed to be the EXACT
    set the existing 0.70/0.68 numbers were measured on."""
    _skip_if_no_gold()
    records = etu.load_gold(str(GOLD))
    _train, dev_recs, test_recs = etu.stratified_split(records, seed=0, n_train=788, n_dev=98, n_test=98)
    assert 90 <= len(test_recs) <= 100  # "n~98"

    out_test = tmp_path / "holdout_sentences.txt"
    out_dev = tmp_path / "holdout_dev_sentences.txt"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_holdout.py"),
         "--gold", str(GOLD), "--out-test", str(out_test), "--out-dev", str(out_dev)],
        check=True, cwd=str(ROOT), capture_output=True, text=True,
    )

    written_test = etu.load_holdout_sentences(str(out_test))
    written_dev = etu.load_holdout_sentences(str(out_dev))
    assert written_test == [r["text"] for r in test_recs]
    assert written_dev == [r["text"] for r in dev_recs]


def test_exclude_by_text_removes_exactly_the_holdout_sentences():
    _skip_if_no_gold()
    records = etu.load_gold(str(GOLD))
    _train, _dev, test_recs = etu.stratified_split(records, seed=0, n_train=788, n_dev=98, n_test=98)
    holdout = [r["text"] for r in test_recs]

    pool = etu.exclude_by_text(records, holdout)
    assert len(pool) == len(records) - len(test_recs)
    pool_texts = {r["text"] for r in pool}
    assert pool_texts.isdisjoint(set(holdout))


def test_records_for_sentences_counts_missing_without_crashing():
    gold_records = [{"text": "a cat sleeps"}, {"text": "a dog runs"}]
    sentences = ["a cat sleeps", "a sentence nowhere in gold"]
    matched, n_missing = etu.records_for_sentences(gold_records, sentences)
    assert n_missing == 1
    assert len(matched) == 1
    assert matched[0]["text"] == "a cat sleeps"


def test_seeded_subset_is_deterministic_and_bounded():
    _skip_if_no_gold()
    records = etu.load_gold(str(GOLD))
    a = etu.seeded_subset(records, 200, seed=7)
    b = etu.seeded_subset(records, 200, seed=7)
    c = etu.seeded_subset(records, 200, seed=8)
    assert len(a) == 200
    assert [r["text"] for r in a] == [r["text"] for r in b]
    assert [r["text"] for r in a] != [r["text"] for r in c]
