"""Tests for episodes and their sources."""

from nsm_ct.episode import (
    BabiSource,
    CurriculumGenerator,
    Episode,
    TextbookSource,
    make_source,
)


def test_episode_mc_consistency():
    ep = Episode(
        context=["mary is in the kitchen ."],
        question="where is mary ?",
        answer_text="kitchen",
        options=["garden", "kitchen", "office", "bedroom"],
        answer_idx=1,
    )
    assert ep.is_multiple_choice
    assert ep.options[ep.answer_idx] == ep.answer_text


def test_episode_rejects_bad_index():
    import pytest

    with pytest.raises(ValueError):
        Episode(context=["a"], question="q?", answer_text="x", options=["x", "y"], answer_idx=5)


def test_curriculum_generator_levels():
    gen = CurriculumGenerator(max_level=3, seed=0)
    eps = gen.generate(30)
    assert len(eps) == 30
    # all levels 1..3 represented
    assert {e.level for e in eps} == {1, 2, 3}
    for e in eps:
        assert e.is_multiple_choice
        assert len(e.options) == 4
        assert e.answer_text in e.options
        assert e.context and e.question.endswith("?")


def test_level3_answer_is_most_recent():
    # Find a level-3 (movement) episode and check recency semantics.
    gen = CurriculumGenerator(max_level=3, seed=1)
    eps = [e for e in gen.generate(30) if e.level == 3]
    assert eps
    e = eps[0]
    # The second statement ("moved to the X") determines the answer.
    assert e.answer_text in e.context[1]


def test_babi_falls_back_offline():
    # Force an unreachable path so the download/lookup fails -> fallback.
    src = BabiSource(task=1, path="/nonexistent/path/for/test", seed=0)
    # Avoid real network: monkeypatch the loader to report no data.
    src._load_lines = lambda: None  # type: ignore
    eps = src.generate(12)
    assert len(eps) == 12
    assert all(isinstance(e, Episode) for e in eps)


def test_babi_parse_format():
    lines = [
        "1 Mary went to the kitchen.",
        "2 John moved to the garden.",
        "3 Where is Mary?\tkitchen\t1",
    ]
    eps = BabiSource.parse_babi(lines)
    assert len(eps) == 1
    assert eps[0].answer_text == "kitchen"
    assert len(eps[0].context) == 2


def test_make_source_factory():
    assert isinstance(make_source("curriculum"), CurriculumGenerator)
    assert isinstance(make_source("babi"), BabiSource)
    assert isinstance(make_source("textbook"), TextbookSource)
