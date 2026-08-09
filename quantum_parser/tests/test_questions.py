"""Regression tests for M43's subject-initial question fix (the ``question1``
ruleset + the ``position_constraints`` DSL feature it introduced).

Root cause (see grammars/english.json's ``question1`` ruleset docstring and
the ``verb1``/``rel2`` collision documented for the wh-cleft, RESEARCH_NOTES
M38): a copula with no NOMINAL to its left (subject-aux inversion, "is the
ball ...?", or wh-fronting, "where is the ball ?") has no way to get a
SUBJECT edge through the ordinary ``clause1`` rule (which requires the
subject BEFORE the predicate) -- so ``predicate1``'s generic transitive-
object rule (VERBAL + NOMINAL(after) -> OBJECT) or ``noun3``'s NP-internal
PP attachment (NOUN + adjacent PP_NOUN -> MODIFICATION) grab the postposed
subject first, and no clause with a SUBJECT ever comes out.

These tests pin the mechanism at the engine level (end-to-end extraction is
covered by consciousness_transformer's probe_parser_stress battery and
tests/test_questions.py there).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parser import ConnectionType, QuantumParser
from parser.dsl import DSLParseError, parse_rule
from parser.matcher import _resolve_position_ref
from parser.pos_tagger import tag_sentence

_GRAMMAR = str(Path(__file__).resolve().parent.parent / "grammars" / "english.json")


def _find(hyp, text: str) -> int:
    return next(i for i, n in enumerate(hyp.nodes) if n.value and n.value.text == text)


def _has_edge(hyp, etype: ConnectionType, parent_idx: int, child_idx: int) -> bool:
    return any(e.type == etype and e.parent == parent_idx and e.child == child_idx for e in hyp.edges)


# -- 1. yes/no subject-aux inversion --------------------------------------

def test_inverted_yes_no_question_gets_subject_edge():
    parser = QuantumParser(_GRAMMAR)
    words = tag_sentence("is the ball in the garden ?")
    best = parser.parse(words).best_hypothesis()
    assert best is not None
    is_idx = _find(best, "is")
    ball_idx = _find(best, "ball")
    assert _has_edge(best, ConnectionType.SUBJECT, is_idx, ball_idx)
    # the landmine this fix avoids: the postposed NOMINAL must NOT also be
    # mistaken for a direct OBJECT of the copula.
    assert not _has_edge(best, ConnectionType.OBJECT, is_idx, ball_idx)


def test_midsentence_copula_is_not_touched_by_inversion_rule():
    """Regression guard: an ordinary declarative copula (not sentence-
    initial) must keep going through the normal clause1 SUBJECT rule, not
    the new position_constraints-gated rule."""
    parser = QuantumParser(_GRAMMAR)
    words = tag_sentence("mary is in the garden .")
    best = parser.parse(words).best_hypothesis()
    assert best is not None
    is_idx = _find(best, "is")
    mary_idx = _find(best, "mary")
    garden_idx = _find(best, "garden")
    assert _has_edge(best, ConnectionType.SUBJECT, is_idx, mary_idx)
    in_idx = _find(best, "in")
    assert _has_edge(best, ConnectionType.PREPOSITION, in_idx, garden_idx)


# -- 2. wh-fronted question -------------------------------------------------

def test_wh_question_gets_subject_edge():
    parser = QuantumParser(_GRAMMAR)
    words = tag_sentence("where is the ball ?")
    best = parser.parse(words).best_hypothesis()
    assert best is not None
    is_idx = _find(best, "is")
    ball_idx = _find(best, "ball")
    assert _has_edge(best, ConnectionType.SUBJECT, is_idx, ball_idx)
    assert not _has_edge(best, ConnectionType.OBJECT, is_idx, ball_idx)


def test_verb1_rel2_collision_is_unaffected_by_the_new_rule():
    """The wh-cleft's known verb1/rel2 collision (RESEARCH_NOTES M38): verb1's
    generic SPECIFIER-adjacency rules ('quickly runs') still eat a RELATIVE
    'where' mid-sentence -- question1 only claims sentence-initial wh, so it
    must not change this pre-existing (documented, out-of-scope) behavior."""
    parser = QuantumParser(_GRAMMAR)
    words = tag_sentence("the garden is where mary is .")
    best = parser.parse(words).best_hypothesis()
    assert best is not None
    # "where" is still mid-sentence (index != 0), so question1's wh rule
    # (position_constraints {"before[0]": 0}) must never have fired on it.
    # ``_find`` returns the FIRST match, which is the first "is" -- the one
    # immediately after "garden", the only one a misfire could touch.
    is1_idx = _find(best, "is")
    where_idx = _find(best, "where")
    assert not _has_edge(best, ConnectionType.SUBJECT, is1_idx, where_idx)


# -- 3. the position_constraints DSL feature itself -------------------------

def test_position_constraints_parses_and_defaults_empty():
    rule_no_constraint = parse_rule(
        {
            "pattern": {"anchor": {"type": "NOUN"}},
            "connections": [],
            "consume": [],
        },
        default_result=__import__("parser.enums", fromlist=["NodeType"]).NodeType.NOUN,
    )
    assert rule_no_constraint.position_constraints == {}

    rule_with_constraint = parse_rule(
        {
            "pattern": {"anchor": {"type": "NOUN"}},
            "connections": [],
            "consume": [],
            "position_constraints": {"anchor": 0},
        },
        default_result=__import__("parser.enums", fromlist=["NodeType"]).NodeType.NOUN,
    )
    assert rule_with_constraint.position_constraints == {"anchor": 0}


def test_position_constraints_rejects_bad_reference():
    import pytest

    with pytest.raises(DSLParseError):
        parse_rule(
            {
                "pattern": {"anchor": {"type": "NOUN"}},
                "connections": [],
                "consume": [],
                "position_constraints": {"bogus": 0},
            },
            default_result=__import__("parser.enums", fromlist=["NodeType"]).NodeType.NOUN,
        )


def test_resolve_position_ref():
    assert _resolve_position_ref("anchor", 5, [1, 2], [7, 8]) == 5
    assert _resolve_position_ref("before[0]", 5, [1, 2], [7, 8]) == 1
    assert _resolve_position_ref("before[1]", 5, [1, 2], [7, 8]) == 2
    assert _resolve_position_ref("after[0]", 5, [1, 2], [7, 8]) == 7
    assert _resolve_position_ref("before[5]", 5, [1, 2], [7, 8]) is None
