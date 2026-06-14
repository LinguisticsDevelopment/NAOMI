"""Stage R1 gate — the reasoning oracle derives L9/L10, flags L11; retrieval fails."""

from nsm_ct.episode import CurriculumGenerator
from nsm_ct.reasoning_oracle import (
    INHERITANCE,
    IS_A_TRANS,
    conditional_rule,
    derive,
    forward_chain,
)


def test_modus_ponens_derives_consequent():
    facts = [("mary", "PLACE", "kitchen")]
    rules = [conditional_rule("kitchen", "stove")]
    known, chain = forward_chain(facts, rules)
    assert ("mary", "CAN_SEE", "stove") in known
    val, _ = derive(facts, rules, ("mary", "CAN_SEE"))
    assert val == "stove"
    assert any(s.rule == "modus_ponens" for s in chain)


def test_inheritance_is_two_hop_transitive():
    facts = [("robin", "IS_A", "bird"), ("bird", "CAN", "fly")]
    val, chain = derive(facts, [INHERITANCE], ("robin", "CAN"))
    assert val == "fly"
    assert chain and chain[-1].derived == ("robin", "CAN", "fly")


def test_unanswerable_returns_none():
    # rule about the kitchen, but mary is in the bedroom -> antecedent never fires
    facts = [("mary", "PLACE", "bedroom")]
    val, _ = derive(facts, [conditional_rule("kitchen", "stove")], ("mary", "CAN_SEE"))
    assert val is None


def _gen_level(level, n=12):
    gen = CurriculumGenerator(max_level=13, seed=3)
    return [e for e in gen.generate(13 * n) if e.level == level]


def test_curriculum_levels_are_oracle_consistent():
    for ep in _gen_level(9):
        assert ep.answerable and ep.gold_chain          # derivable, has a chain
        # the answer is NOT directly asserted -> retrieval/recency cannot get it
        q = ep.meta["query"]
        assert not any(e == q[0] and r == q[1] for (e, r, _v) in ep.meta["facts"])
    for ep in _gen_level(10):
        assert ep.answerable and ep.gold_chain
        q = ep.meta["query"]
        assert not any(e == q[0] and r == q[1] for (e, r, _v) in ep.meta["facts"])
    for ep in _gen_level(11):
        assert not ep.answerable
        assert ep.answer_text == "idk"
        assert "idk" in ep.options


def test_is_a_transitivity_composes_a_deep_chain():
    # robin -> bird -> animal -> creature, creature can move => robin can move (3 is-a hops)
    facts = [("robin", "IS_A", "bird"), ("bird", "IS_A", "animal"),
             ("animal", "IS_A", "creature"), ("creature", "CAN", "move")]
    val, _ = derive(facts, [IS_A_TRANS, INHERITANCE], ("robin", "CAN"))
    assert val == "move"


def test_deep_levels_are_oracle_consistent():
    for ep in _gen_level(12) + _gen_level(13):
        assert 2 <= ep.meta["chain_len"] <= 4
        if ep.answerable:
            assert ep.gold_chain                      # a real derivation exists
        else:
            assert ep.answer_text == "idk"            # broken chain -> abstain


def test_retrieval_baseline_cannot_answer_reasoning_levels():
    # No base fact directly answers the query for any answerable reasoning episode:
    # only forward-chaining reaches it (this is what makes the level a reasoning task).
    for level in (9, 10):
        for ep in _gen_level(level):
            q = ep.meta["query"]
            direct = [(e, r, v) for (e, r, v) in ep.meta["facts"] if e == q[0] and r == q[1]]
            assert direct == []
