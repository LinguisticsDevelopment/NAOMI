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
from nsm_ct.mind import routing
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


def _train_router(ctrl, codec, clauses, steps=160):
    """Overfit the controller's act-head on a handful of clauses (the M12 router)."""
    opt = torch.optim.Adam(ctrl.parameters(), lr=0.03)
    tgt = routing.act_targets(clauses)
    batch = routing.build_routing_batch(clauses, codec)
    ctrl.train()
    for _ in range(steps):
        out = ctrl(batch); loss = routing.act_routing_loss(out, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return ctrl


def test_consume_one_door_learns_and_answers():
    """The single door with LEARNED routing: the controller decides absorb-vs-answer
    per clause (no is_q flag, no options). Facts/rules are learned; a yes/no query is
    answered by reasoning; an unsupported query abstains. Weights frozen during consume."""
    codec = TPRCodec(dim=32)
    ltm = KnowledgeGraph(codec=codec)
    ctrl = MindController(codec, hidden=32, hops=3, halting=False)

    feed = [
        ("fact", "alice", "is", "furry", "+"),                 # taught
        ("rule", (("?x", "is", "furry", "+"),), ("?x", "is", "kind", "+")),
        ("rule", (("?x", "is", "kind", "+"),), ("?x", "is", "smart", "+")),
        ("query", "alice", "is", "smart", "+"),                # a question — model-detected
        ("query", "alice", "is", "green", "+"),                # unsupported → abstain
    ]
    _train_router(ctrl, codec, feed)                           # learn the act-routing
    before = {k: v.clone() for k, v in ctrl.state_dict().items()}
    loop_facts = ConsciousLoop(ltm, controller=ctrl)
    resp = loop_facts.consume(feed)
    assert loop_facts.facts == [("alice", "is", "furry", "+")]  # routed to memory
    assert len(loop_facts.rules) == 2                           # rules routed to memory
    assert len(resp) == 2                                       # both questions answered
    assert resp[0]["answer"] in (pw.TRUE, pw.FALSE, pw.UNKNOWN)
    assert resp[1]["answer"] == pw.UNKNOWN                      # not derivable → abstain
    after = ctrl.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # zero weight change in consume
    after = ctrl.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # zero weight change


def test_learned_router_overfits_and_generalizes():
    """The act-head learns absorb-vs-answer from the clause encoding (≈1.0 on its train
    set) and GENERALIZES to unseen entities/relations — it learned 'interrogative mood →
    answer', not a memorized lookup."""
    codec = TPRCodec(dim=32)
    ctrl = MindController(codec, hidden=32, hops=2, halting=False)
    train = [
        ("fact", "bob", "is", "tall", "+"),
        ("rule", (("?x", "is", "tall", "+"),), ("?x", "is", "big", "+")),
        ("query", "bob", "is", "big", "+"),
        ("query", "bob", "CAN"),
    ]
    _train_router(ctrl, codec, train)
    assert routing.predict_acts(ctrl, train, codec) == [routing.gold_act(c) for c in train]

    unseen = [                                                 # different entities/relations
        ("fact", "zara", "lives_in", "rome", "+"),
        ("query", "zara", "lives_in", "rome", "+"),
        ("rule", (("?x", "lives_in", "rome", "+"),), ("?x", "is", "roman", "+")),
        ("query", "quinn", "PLACE"),
    ]
    got = routing.predict_acts(ctrl, unseen, codec)
    want = [routing.gold_act(c) for c in unseen]
    # the decisive split (answer vs absorb) generalizes from the mood marker
    assert [a in routing.ANSWER_ACTS for a in got] == [w in routing.ANSWER_ACTS for w in want]


def test_consume_routes_by_model_not_tag_mood_flip():
    """consume's decision is the model's: the SAME content routes to memory when
    declarative and to reasoning when interrogative (mood-flip)."""
    codec = TPRCodec(dim=32)
    ctrl = MindController(codec, hidden=32, hops=2, halting=False)
    _train_router(ctrl, codec, [
        ("fact", "amy", "is", "warm", "+"),
        ("query", "amy", "is", "warm", "+"),
    ])
    loop = ConsciousLoop(KnowledgeGraph(codec=codec), controller=ctrl)

    # declarative content → absorbed (answers nothing); interrogative content → answered
    assert loop.consume([("fact", "amy", "is", "warm", "+")]) == []
    assert loop.facts == [("amy", "is", "warm", "+")]
    resp = loop.consume([("fact", "amy", "is", "warm", "+"),
                         ("query", "amy", "is", "warm", "+")])
    assert len(resp) == 1 and resp[0]["answer"] == pw.TRUE     # routed to reasoning → proven


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
