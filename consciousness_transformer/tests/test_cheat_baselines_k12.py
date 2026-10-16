"""Tests for scripts/cheat_baselines_k12.py (dev/reports/K12_BENCHMARK_FIX.md item
1b) -- a 5-episode fixture with hand-computed expected values for all four
baselines (random, majority, passage-substring, word-overlap), including
the tie-break ("ties -> random among tied", expressed as expected credit
1/|tie| when gold is among the tied options) and the "miss" case (zero
substring/overlap matches at all).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cheat_baselines_k12 import (  # noqa: E402
    majority_baseline,
    passage_substring_baseline,
    random_baseline,
    word_overlap_baseline,
)
from nsm_ct.episode import Episode  # noqa: E402

import random as _random  # noqa: E402


def _ep(context, answer_text, options, answer_idx):
    return Episode(context=context, question="?", answer_text=answer_text,
                    options=options, answer_idx=answer_idx)


def _fixture():
    return [
        _ep(["the cat sat on the mat ."], "the cat",
            ["the cat", "the mat", "a dog", "a bird"], 0),
        _ep(["a dog ran in the park ."], "a dog",
            ["a dog", "a cat", "a park", "a tree"], 0),
        _ep(["the bird flew over the house ."], "a bird",
            ["a bird", "a house", "a car", "a boat"], 0),
        _ep(["mice like cheese ."], "mice",
            ["mice", "cheese", "bread", "milk"], 0),
        _ep(["dogs bark loudly at night ."], "cats",
            ["cats", "dogs", "night", "day"], 0),
    ]


def test_random_baseline():
    assert random_baseline(_fixture()) == 0.25


def test_majority_baseline():
    acc, _text = majority_baseline(_fixture())
    assert acc == 0.2  # every answer_text is distinct -> modal count 1 / 5


def test_passage_substring_baseline_ties_and_misses():
    rng = _random.Random(0)
    acc, n_unique, n_ties, n_misses = passage_substring_baseline(_fixture(), rng)
    # ep1: {"the cat","the mat"} tied, gold tied -> 0.5
    # ep2: {"a dog"} unique correct -> 1.0
    # ep3: no option is a passage substring -> miss -> 0.0
    # ep4: {"mice","cheese"} tied, gold tied -> 0.5
    # ep5: {"dogs","night"} tied, gold ("cats") NOT among them -> 0.0
    assert acc == 0.4
    assert n_unique == 1
    assert n_ties == 3
    assert n_misses == 1
    assert n_unique + n_ties + n_misses == len(_fixture())


def test_word_overlap_baseline():
    rng = _random.Random(0)
    acc = word_overlap_baseline(_fixture(), rng)
    # ep1: {"the cat","the mat"} tied at overlap=2, gold tied -> 0.5
    # ep2: {"a dog","a park"} tied at overlap=2, gold tied -> 0.5
    # ep3: {"a bird","a house"} tied at overlap=1, gold tied -> 0.5
    # ep4: {"mice","cheese"} tied at overlap=1, gold tied -> 0.5
    # ep5: {"dogs","night"} tied at overlap=1, gold ("cats") NOT tied -> 0.0
    assert acc == 0.4


def test_gate_logic_matches_random_plus_point_one():
    """Sanity on the gate threshold arithmetic itself (scripts/
    cheat_baselines_k12.py's CLI only prints the verdict, never asserts
    it): this fixture deliberately keeps some cross-story-style passage
    leakage (ep1/ep2/ep4 all have a non-gold option that is ALSO a passage
    substring), so its substring cheat (0.4) exceeds random+0.10 (0.35) --
    the gate correctly flags it as FAIL. The real same-passage policy's
    gate-passing behavior is covered by test_distractor_policy.py, which
    locks the actual invariant (every option is a passage substring) that
    makes the gate moot rather than merely satisfied."""
    r = random_baseline(_fixture())
    rng = _random.Random(0)
    sub_acc, *_ = passage_substring_baseline(_fixture(), rng)
    assert sub_acc > r + 0.10
