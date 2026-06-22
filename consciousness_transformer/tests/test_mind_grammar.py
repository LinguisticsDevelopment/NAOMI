"""M13 gates: the owned controlled-language grammar (recursive parse) + the converse door.

Deterministic, no training. The recursive parser is a drop-in SUPERSET of the flat
``membrane.parse`` (full parity on the curriculum) that additionally handles
subordination (conditionals, relatives, quantified descriptions) and yes/no questions.
"""

from __future__ import annotations

from nsm_ct.episode import CurriculumGenerator
from nsm_ct.mind import grammar, membrane
from nsm_ct.mind.conscious_loop import ConsciousLoop
from nsm_ct.mind.knowledge import KnowledgeGraph


def _curriculum_sentences(n=150, seed=0):
    sents = []
    for ep in CurriculumGenerator(max_level=13, seed=seed).generate(n):
        sents += list(ep.context) + [ep.question] + list(getattr(ep, "post_context", []) or [])
    return sents


def test_grammar_is_superset_of_membrane_on_curriculum():
    """Every curriculum sentence parses to EXACTLY the membrane's meaning object."""
    sents = _curriculum_sentences()
    assert sents
    for s in sents:
        assert grammar.parse(s) == membrane.parse(s), s


def test_meaning_side_round_trip_for_new_forms():
    """parse(render(m)) == m for the rule/query forms (render is membrane's)."""
    for m in [
        ("rule", (("?p", "PLACE", "kitchen"),), ("?p", "CAN_SEE", "window")),
        ("rule", (("bill", "PLACE", "bathroom"),), ("bill", "CAN_SEE", "window")),
        ("query", "mary", "PLACE"),
        ("query", "robin", "CAN"),
    ]:
        assert grammar.parse(membrane.render(m)) == m


def test_quantified_description_is_a_universal_rule():
    """'everyone who is in the kitchen can see the window' → a ?p-variable rule."""
    assert grammar.parse("everyone who is in the kitchen can see the window .") == \
        ("rule", (("?p", "PLACE", "kitchen"),), ("?p", "CAN_SEE", "window"))
    # a restrictive relative on a noun head is likewise a restriction → ?p
    assert grammar.parse("the dog that is in the garden can see the stove .") == \
        ("rule", (("?p", "PLACE", "garden"),), ("?p", "CAN_SEE", "stove"))


def test_named_conditional_stays_grounded():
    """'if mary …, mary …' is about mary specifically — the subject stays a constant
    (it must NOT generalize to ?p)."""
    assert grammar.parse("if mary is in the kitchen , mary can see the window .") == \
        ("rule", (("mary", "PLACE", "kitchen"),), ("mary", "CAN_SEE", "window"))


def test_yes_no_question_carries_polarity():
    assert grammar.parse("is alice in the kitchen ?") == ("query", "alice", "PLACE", "kitchen", "+")


def test_unparsable_returns_none():
    assert grammar.parse("colorless green ideas sleep furiously .") is None
    assert grammar.parse("") is None


def test_converse_teach_then_ask_universal():
    """The end-to-end gate: teach a universal + a fact in prose, then ask — in English."""
    loop = ConsciousLoop(KnowledgeGraph(dim=32))     # wh + forward-chain: no controller needed
    replies = loop.converse([
        "everyone who is in the kitchen can see the window .",
        "mary is in the kitchen .",
        "what can mary see ?",
    ])
    assert replies == ["Mary can see the window."]


def test_converse_abstains_and_survives_garbage():
    loop = ConsciousLoop(KnowledgeGraph(dim=32))
    assert loop.converse(["where is mary ?"]) == ["I don't know."]   # nothing taught
    assert loop.converse(["@#$ not english", "a b c d e ."]) == []    # no crash, no reply
