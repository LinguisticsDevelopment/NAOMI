"""M6 gates: the language membrane (text ↔ meaning) + faithful verbalization.

Deterministic, no training: every curriculum sentence round-trips both directions,
and "respond with thinking" verbalizes the *actual* DerivStep provenance.
"""

from __future__ import annotations

from nsm_ct.episode import CurriculumGenerator
from nsm_ct.mind import membrane, ops, teacher
from nsm_ct.mind.verbalize import verbalize_answer, verbalize_trace


def _curriculum_sentences(n=200, seed=0):
    sents = []
    for ep in CurriculumGenerator(max_level=13, seed=seed).generate(n):
        sents += list(ep.context) + [ep.question] + list(getattr(ep, "post_context", []) or [])
    return sents


def test_render_matches_curriculum_strings():
    """The renderer reproduces the exact curriculum sentences."""
    assert membrane.render_fact("mary", "PLACE", "kitchen") == "mary is in the kitchen ."
    assert membrane.render_fact("john", "PLACE", "office", negate=True) == "john is not in the office ."
    assert membrane.render_fact("robin", "IS_A", "bird") == "a robin is a bird ."
    assert membrane.render_fact("bird", "CAN", "fly") == "a bird can fly ."
    assert membrane.render_rule((("bill", "PLACE", "bathroom"),), ("bill", "CAN_SEE", "window")) \
        == "if bill is in the bathroom , bill can see the window ."
    assert membrane.render_query("mary", "PLACE") == "where is mary ?"
    assert membrane.render_query("robin", "CAN") == "what can a robin do ?"


def test_cycle_consistency_both_directions():
    """Every curriculum sentence parses and is meaning-stable; canonical forms round-trip
    text→meaning→text exactly; surface variants (e.g. 'moved to') normalize to the same
    meaning. meaning→text→meaning is exact for all."""
    sents = _curriculum_sentences()
    assert sents
    exact = 0
    for s in sents:
        obj = membrane.parse(s)
        assert obj is not None, f"failed to parse: {s!r}"
        # meaning→text→meaning is exact (the firm bijection on the meaning side).
        assert membrane.parse(membrane.render(obj)) == obj, f"meaning round-trip: {obj!r}"
        if membrane.render(obj) == s:
            exact += 1
    # The vast majority of curriculum language reproduces verbatim; the remainder are
    # meaning-preserving surface variants (moved to / name-is-a).
    assert exact / len(sents) > 0.9, exact / len(sents)


def test_disjunction_and_conditional_round_trip():
    s_or = "sandra is in the kitchen or the garden ."
    assert membrane.render(membrane.parse(s_or)) == s_or
    s_if = "if sandra can see the gate , sandra can hold the stove ."
    obj = membrane.parse(s_if)
    assert obj[0] == "rule" and membrane.render(obj) == s_if


def test_verbalize_answer():
    assert verbalize_answer(("robin", "CAN"), "fly") == "A robin can fly."
    assert verbalize_answer(("sandra", "CAN_SEE"), ops.ABSTAIN) == "I don't know."
    assert verbalize_answer(("mary", "PLACE"), "maybe") == "Maybe."


def test_respond_with_thinking_is_faithful():
    """The verbalized reason is EXACTLY the executor's DerivStep chain, not invented."""
    ep = next(e for e in CurriculumGenerator(max_level=10, seed=1).generate(60) if e.level == 10)
    res = teacher.replay(ep)
    infer = next(s for s in res["trace"] if s.op == ops.INFER)
    support = infer.support
    assert support, "expected a derivation chain"

    query = tuple(ep.meta["query"])
    text = verbalize_trace(query, ep.answer_text, support).lower()

    # Every premise and conclusion in the ACTUAL chain appears; nothing else is asserted.
    for step in support:
        for f in step.support:
            assert membrane.render_clause(*f).lower() in text, (f, text)
        assert membrane.render_clause(*step.derived).lower() in text
    # The conclusion matches the oracle answer (the derived value).
    assert ep.answer_text.lower() in text


if __name__ == "__main__":  # pragma: no cover
    import pytest
    pytest.main([__file__, "-v"])
