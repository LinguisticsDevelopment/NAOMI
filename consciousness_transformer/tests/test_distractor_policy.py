"""Tests for the same-passage MC distractor policy (dev/reports/K12_BENCHMARK_FIX.md
item 1a) -- scripts/convert_fairytaleqa.py's fix for the broken
cross-story distractor policy dev/reports/AUDIT_2026-10-15.md's audit flagged
(passage-presence alone identified the gold answer at ~97%).

No network, no full corpus fetch: builds small in-memory Episode fixtures
directly (the same shape scripts/convert_fairytaleqa.py's build_episodes
produces) so this runs fast and deterministically.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from convert_fairytaleqa import (  # noqa: E402
    attach_entity_options_cross_story,
    attach_entity_options_same_passage,
)
from nsm_ct.episode import Episode  # noqa: E402


def _entity_episode(context, answer_text, attribute="character"):
    return Episode(
        context=context,
        question="who?",
        answer_text=answer_text,
        options=None,
        answer_idx=None,
        meta={"answer_type": "entity", "attribute": attribute},
    )


def _norm(s):
    return " ".join(s.lower().split())


def _rich_fixture():
    """Passages with >=2 extractable same-passage NP candidates each, so
    the gate population (>=2 distractors) is non-trivial."""
    return [
        _entity_episode(
            ["the old man lived in the small village .",
             "the old man found a golden key ."],
            "the old man", attribute="character",
        ),
        _entity_episode(
            ["a brave knight rode to the dark castle .",
             "the brave knight met a wise wizard there ."],
            "a wise wizard", attribute="character",
        ),
        _entity_episode(
            ["the fox hid behind the tall tree .",
             "a small rabbit watched from the green bush ."],
            "the tall tree", attribute="setting",
        ),
    ]


def test_same_passage_options_are_passage_substrings():
    """The actual fix: every attached option (gold AND distractors) must
    itself be a verbatim substring of the episode's own passage -- "no
    distractor may be a substring of the passage unless the gold is also"
    collapses to this for entity-type episodes, where the gold is ALWAYS a
    passage substring by construction (classify_answer_type's own
    definition of "entity")."""
    episodes = _rich_fixture()
    n_attached = attach_entity_options_same_passage(episodes, seed=0, n_distractors=3)
    assert n_attached >= 1

    for ep in episodes:
        if ep.options is None:
            continue
        passage = _norm(" ".join(ep.context))
        for opt in ep.options:
            assert _norm(opt) in passage, f"option {opt!r} is not a passage substring"
        # the gold itself must also be a substring (sanity on the fixture)
        assert _norm(ep.answer_text) in passage


def test_same_passage_yields_at_least_two_distractors_when_available():
    """The gate population invariant: an entity episode whose passage has
    enough distinct NP material gets >=2 distractors (3 options total +
    gold), not left at options=None."""
    episodes = _rich_fixture()
    attach_entity_options_same_passage(episodes, seed=0, n_distractors=3)
    with_options = [ep for ep in episodes if ep.options is not None]
    assert len(with_options) >= 1
    for ep in with_options:
        n_distractors = len(ep.options) - 1
        assert n_distractors >= 2
        assert ep.answer_idx is not None
        assert ep.options[ep.answer_idx] == ep.answer_text


def test_same_passage_degrades_gracefully_with_no_candidates():
    """A passage with no extractable NP other than the gold itself must
    leave the episode open-ended (options=None), never crash and never
    attach fewer than 2 distractors."""
    episodes = [_entity_episode(["ok ."], "ok")]
    n_attached = attach_entity_options_same_passage(episodes, seed=0, n_distractors=3)
    assert n_attached == 0
    assert episodes[0].options is None
    assert episodes[0].answer_idx is None


def test_cross_story_policy_distractors_need_not_be_passage_substrings():
    """Characterizes the OLD policy this fix replaces: a cross-story
    distractor pool is shared corpus-wide, so (with enough episodes of the
    same attribute) at least one attached distractor is typically NOT a
    substring of ITS episode's own passage -- the exact property that let
    "pick the option in the passage" cheat. Kept only so the two policies'
    behavior stays documented and distinguishable in code, not just prose."""
    episodes = [
        _entity_episode(["the king ruled the land ."], "the king", attribute="character"),
        _entity_episode(["a queen lived in a tower ."], "a queen", attribute="character"),
        _entity_episode(["the prince rode a horse ."], "the prince", attribute="character"),
        _entity_episode(["a princess sang a song ."], "a princess", attribute="character"),
    ]
    attach_entity_options_cross_story(episodes, seed=0, n_distractors=3)
    attached = [ep for ep in episodes if ep.options is not None]
    assert attached, "fixture should have enough same-attribute gold answers to cross-pollinate"
    any_non_substring_distractor = False
    for ep in attached:
        passage = _norm(" ".join(ep.context))
        for i, opt in enumerate(ep.options):
            if i == ep.answer_idx:
                continue
            if _norm(opt) not in passage:
                any_non_substring_distractor = True
    assert any_non_substring_distractor
