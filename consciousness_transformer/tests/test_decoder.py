"""Tests for the Phase-1 decoder (dev/DECODER_DESIGN.md): rule-grounded
short-answer realization, and the no-confabulation ablation gate (design §4.2)
that makes M65's invariant executable — sever the memory->decoder link and the
output MUST collapse to abstention, never fabricated content.
"""

from __future__ import annotations

import copy
import re

import pytest

from nsm_ct import decoder
from nsm_ct.ground.usvs import load_usvs


@pytest.fixture(scope="module")
def usvs():
    return load_usvs("data/usvs")


ENTITY_BOOK = {"h_mary": "Mary", "h_sue": "Sue"}


def _where_is_mary():
    """design §7.1: memory holds mary -PLACE-> garden (garden.n.01, h_garden)."""
    return {
        "answer_kind": "place",
        "answer_clause": {
            "predicate": "is",
            "predicate_grounding": {"type": "prime", "prime": "BE_SOMEWHERE", "handle": None},
            "is_question": False, "utterance_kind": "proposition",
            "roles": [
                {"relation": "SUBJECT", "grounding": {"type": "entity", "handle": "h_mary"}},
                {"relation": "PLACE",
                 "grounding": {"type": "sense", "sense_id": "garden.n.01", "handle": "h_garden"}},
            ],
        },
        "focus": {"slot": "role", "role_index": 1},
        "provenance": ["h_mary", "h_garden"],
    }


def _who_saw_sue():
    """"Who saw Sue?" -> "Mary." — entity/name short answer, focus on SUBJECT."""
    return {
        "answer_kind": "entity",
        "answer_clause": {
            "predicate": "saw",
            "predicate_grounding": {"type": "sense", "sense_id": "saw.v.01", "handle": "h_see"},
            "is_question": False, "utterance_kind": "proposition",
            "roles": [
                {"relation": "SUBJECT", "grounding": {"type": "entity", "handle": "h_mary"}},
                {"relation": "OBJECT", "grounding": {"type": "entity", "handle": "h_sue"}},
            ],
        },
        "focus": {"slot": "role", "role_index": 0},
        "provenance": ["h_mary", "h_sue", "h_see"],
    }


def _is_the_ball_happy():
    """attribute short answer, focus on the ATTRIBUTE role -> "Happy."."""
    return {
        "answer_kind": "attribute",
        "answer_clause": {
            "predicate": "is",
            "predicate_grounding": {"type": "prime", "prime": "BE_SOMEONE_SOMETHING", "handle": None},
            "is_question": False, "utterance_kind": "proposition",
            "roles": [
                {"relation": "SUBJECT",
                 "grounding": {"type": "sense", "sense_id": "ball.n.01", "handle": "h_ball"}},
                {"relation": "ATTRIBUTE",
                 "grounding": {"type": "sense", "sense_id": "happy.a.01", "handle": "h_happy"}},
            ],
        },
        "focus": {"slot": "role", "role_index": 1},
        "provenance": ["h_ball", "h_happy"],
    }


def _is_the_ball_red_verdict():
    """design §7.2: yes/no verdict answer."""
    return {
        "answer_kind": "verdict",
        "verdict": "yes",
        "answer_clause": {
            "predicate": "is",
            "predicate_grounding": {"type": "prime", "prime": "BE_SOMEONE_SOMETHING", "handle": None},
            "is_question": False, "utterance_kind": "proposition",
            "roles": [
                {"relation": "SUBJECT",
                 "grounding": {"type": "sense", "sense_id": "ball.n.01", "handle": "h_ball"}},
                {"relation": "ATTRIBUTE",
                 "grounding": {"type": "sense", "sense_id": "happy.a.01", "handle": "h_happy"}},
            ],
        },
        "focus": None,
        "provenance": ["h_ball", "h_happy"],
    }


# --------------------------------------------------------------------- (a) ---
def test_who_short_answer(usvs):
    assert decoder.realize(_who_saw_sue(), usvs=usvs, entity_book=ENTITY_BOOK) == "Mary."


def test_where_short_answer(usvs):
    assert decoder.realize(_where_is_mary(), usvs=usvs, entity_book=ENTITY_BOOK) == "The garden."


def test_where_full_clause(usvs):
    assert (decoder.realize(_where_is_mary(), usvs=usvs, entity_book=ENTITY_BOOK, form="full")
            == "Mary is in the garden.")


def test_attribute_short_answer(usvs):
    assert decoder.realize(_is_the_ball_happy(), usvs=usvs, entity_book=ENTITY_BOOK) == "Happy."


def test_attribute_full_clause(usvs):
    assert (decoder.realize(_is_the_ball_happy(), usvs=usvs, entity_book=ENTITY_BOOK, form="full")
            == "The ball is happy.")


def test_yes_no_verdict(usvs):
    yes = _is_the_ball_red_verdict()
    assert decoder.realize(yes, usvs=usvs, entity_book=ENTITY_BOOK) == "Yes."
    no = copy.deepcopy(yes)
    no["verdict"] = "no"
    assert decoder.realize(no, usvs=usvs, entity_book=ENTITY_BOOK) == "No."


# --------------------------------------------------------------------- (b) ---
def test_abstention_is_first_class(usvs):
    """design §7.3: comprehension found nothing grounded -> the honest answer."""
    abstain = {"answer_kind": "abstain", "focus": None, "provenance": []}
    assert decoder.realize(abstain, usvs=usvs, entity_book=ENTITY_BOOK) == decoder.ABSTAIN_TEXT
    assert decoder.ABSTAIN_TEXT == "I don't know."


def test_verdict_with_no_verdict_value_abstains(usvs):
    unresolved = {"answer_kind": "verdict", "verdict": None, "focus": None, "provenance": []}
    assert decoder.realize(unresolved, usvs=usvs, entity_book=ENTITY_BOOK) == decoder.ABSTAIN_TEXT


# --------------------------------------------------------------------- (c) ---
def _sever_memory(answer):
    """The design §4.2 ablation: null every grounding's content bindings — the
    exact structure the decoder would be handed off a zeroed memory. Deep-copies
    so the with-memory fixtures above are untouched."""
    severed = copy.deepcopy(answer)
    clause = severed.get("answer_clause")
    if clause:
        for role in clause.get("roles", []):
            g = role.get("grounding")
            if g:
                g["sense_id"] = None
                g["handle"] = None
                g["prime"] = None
    if "verdict" in severed:
        severed["verdict"] = None
    return severed


ALL_CONTENT_WORDS = {"mary", "sue", "garden", "ball", "happy", "yes", "no"}


@pytest.mark.parametrize("build,form", [
    (_where_is_mary, "short"), (_where_is_mary, "full"),
    (_who_saw_sue, "short"),
    (_is_the_ball_happy, "short"), (_is_the_ball_happy, "full"),
    (_is_the_ball_red_verdict, "short"),
])
def test_no_confabulation_ablation(usvs, build, form):
    """Sever the memory->decoder path (every grounding nulled). The realized
    output MUST collapse to abstention/empty and MUST NOT contain any of the
    with-memory content words — the no-confabulation gate (design §4.2)."""
    intact = decoder.realize(build(), usvs=usvs, entity_book=ENTITY_BOOK, form=form)
    severed_answer = _sever_memory(build())
    severed_output = decoder.realize(severed_answer, usvs=usvs, entity_book=ENTITY_BOOK, form=form)

    assert severed_output in (decoder.ABSTAIN_TEXT, "")
    lowered_words = set(re.findall(r"[a-z']+", severed_output.lower()))
    leaked = lowered_words & ALL_CONTENT_WORDS
    assert not leaked, (
        f"ablation leaked grounded content {leaked!r} into {severed_output!r} "
        f"(with-memory output was {intact!r})"
    )


def test_ablation_on_fully_empty_memory_answer(usvs):
    """Equivalent framing (design §4.2): the answer structure built from an
    empty/zeroed memory in the first place — no provenance, no clause."""
    empty = {"answer_kind": "abstain", "focus": None, "provenance": []}
    assert decoder.realize(empty, usvs=usvs, entity_book=ENTITY_BOOK) == decoder.ABSTAIN_TEXT


def test_unresolved_candidate_set_is_rejected_not_realized(usvs):
    """§2.1: a grounding still carrying `candidates` is an unresolved input (a
    comprehension bug) — the decoder rejects it and abstains, never realizes it."""
    unresolved = _where_is_mary()
    unresolved["answer_clause"]["roles"][1]["grounding"]["candidates"] = ["garden.n.01", "yard.n.01"]
    assert decoder.realize(unresolved, usvs=usvs, entity_book=ENTITY_BOOK) == decoder.ABSTAIN_TEXT
