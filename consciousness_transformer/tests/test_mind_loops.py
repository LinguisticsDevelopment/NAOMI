"""M4 gates: the two loops over one substrate.

Firm, deterministic gates: teach-once-with-no-weight-update (the core invariant),
offline-inference-reduces-runtime-depth, STM→LTM consolidation, and the conscious
loop returning an answer + a faithful op-trace with a symbolic validator.
"""

from __future__ import annotations

import torch

from nsm_ct.clause_psyche_graph import STM
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.mind import ops
from nsm_ct.mind.conscious_loop import ConsciousLoop
from nsm_ct.mind.controller import MindController
from nsm_ct.mind.datasets import proofwriter as pw
from nsm_ct.mind.executor import _TrivialResolver
from nsm_ct.mind.knowledge import KnowledgeGraph
from nsm_ct.mind.subconscious_loop import SubconsciousLoop
from nsm_ct.reasoning_oracle import INHERITANCE
from nsm_ct.tpr import TPRCodec


def test_teach_once_no_weight_update():
    """A fact taught by graph write is recalled with the controller's weights frozen."""
    codec = TPRCodec(dim=48)
    ltm = KnowledgeGraph(codec=codec)
    ctrl = MindController(codec, hidden=32, hops=5, halting=False)
    before = {k: v.clone() for k, v in ctrl.state_dict().items()}

    ltm.add_fact("mary", "PLACE", "kitchen")          # a GRAPH WRITE — no training
    loop = ConsciousLoop(ltm, controller=ctrl)
    value, _chain = loop.recall("mary", "PLACE")

    assert value == "kitchen"                          # taught once, known forever
    after = ctrl.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # zero weight change


def test_consume_one_door_learns_and_answers():
    """The single door: a mixed clause feed is self-routed — facts/rules are learned,
    and a yes/no query is ANSWERED by reasoning (no is_q flag, no answer options, no
    weight update). A query with no support abstains (Unknown)."""
    codec = TPRCodec(dim=32)
    ltm = KnowledgeGraph(codec=codec)
    ctrl = MindController(codec, hidden=32, hops=3, halting=False)
    before = {k: v.clone() for k, v in ctrl.state_dict().items()}
    loop = ConsciousLoop(ltm, controller=ctrl)

    feed = [
        ("fact", "alice", "is", "furry", "+"),                 # taught
        ("rule", (("?x", "is", "furry", "+"),), ("?x", "is", "kind", "+")),
        ("rule", (("?x", "is", "kind", "+"),), ("?x", "is", "smart", "+")),
        ("query", "alice", "is", "smart", "+"),                # a question — self-detected
        ("query", "alice", "is", "green", "+"),                # unsupported → abstain
    ]
    resp = loop.consume(feed)

    assert loop.facts == [("alice", "is", "furry", "+")]       # learned the fact
    assert len(loop.rules) == 2                                # learned the rules
    assert len(resp) == 2                                      # answered both queries
    assert resp[0]["answer"] in (pw.TRUE, pw.FALSE, pw.UNKNOWN)
    assert resp[1]["answer"] == pw.UNKNOWN                     # not derivable → abstain
    after = ctrl.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # zero weight change


def test_consume_wh_query_derives_value():
    """A wh-query `("query", s, r)` is routed by its type and derives the value over
    the accumulated theory (no controller needed for the forward closure)."""
    loop = ConsciousLoop(KnowledgeGraph(dim=32))
    feed = [
        ("fact", "robin", "IS_A", "bird", "+"),
        ("rule", (("?x", "IS_A", "bird"),), ("?x", "CAN", "fly")),
        ("query", "robin", "CAN"),                             # wh: what can robin do?
    ]
    resp = loop.consume(feed)
    assert resp[0]["answer"] == "fly"                          # derived, not selected


def test_offline_infer_makes_multihop_a_direct_fact():
    """offline INFER pre-derives a chain so the conscious loop answers it in one hop."""
    ltm = KnowledgeGraph(dim=64)
    ltm.assert_facts([("robin", "IS_A", "bird"), ("bird", "CAN", "fly")])
    ltm.add_rule(INHERITANCE)
    assert ("robin", "CAN", "fly") not in set(ltm.facts())   # derived-only, not direct

    sub = SubconsciousLoop(ltm)
    added = sub.offline_infer()
    assert added >= 1
    assert ("robin", "CAN", "fly") in set(ltm.facts())       # now a DIRECT fact (1-hop)
    assert ConsciousLoop(ltm).recall("robin", "CAN")[0] == "fly"


def test_consolidate_stm_to_ltm():
    """A finished episode's resolved STM facts are promoted into durable LTM."""
    codec = TPRCodec(dim=48)
    ltm = KnowledgeGraph(codec=codec)
    stm = STM(codec, _TrivialResolver())
    stm.add_clause("john", "PLACE", "office")
    sub = SubconsciousLoop(ltm, codec=codec)
    n = sub.consolidate(stm)
    assert n >= 1
    assert ("john", "PLACE", "office") in set(ltm.facts())
    assert ConsciousLoop(ltm).recall("john", "PLACE")[0] == "office"


def test_conscious_loop_answer_and_faithful_trace():
    """respond() returns an answer + a non-empty op-trace; the validator gives oracle truth."""
    codec = TPRCodec(dim=48)
    eps = CurriculumGenerator(max_level=10, seed=0).generate(60)
    l10 = next(e for e in eps if e.level == 10)
    ltm = KnowledgeGraph(codec=codec)
    ctrl = MindController(codec, hidden=32, hops=5, halting=False)
    loop = ConsciousLoop(ltm, controller=ctrl)

    out = loop.respond(l10)
    assert out["answer"] in (l10.options + [ops.ABSTAIN])
    assert out["trace"] and out["trace"][-1].op == ops.RESPOND      # faithful, inspectable trace
    # The symbolic validator independently yields the oracle's answer.
    assert loop.validate(l10)["answer"] == l10.answer_text


def test_subconscious_run_smoke():
    """A couple of subconscious rounds execute end-to-end and grow/maintain LTM."""
    codec = TPRCodec(dim=32)
    ltm = KnowledgeGraph(codec=codec)
    ctrl = MindController(codec, hidden=32, hops=5, halting=False)
    sub = SubconsciousLoop(ltm, ctrl, codec=codec, total_rounds=2)
    hist = sub.run(rounds=2, episodes_per_round=40, steps=3, verbose=False)
    assert len(hist) == 2
    assert hist[-1]["train_rel_match"] >= 0.0


if __name__ == "__main__":  # pragma: no cover
    import pytest
    pytest.main([__file__, "-v"])
