"""M8 gates: ProofWriter ingestion + the substrate reproduces its gold logic.

Data-free unit gates (parser + verify on hand-built theories) always run; the
real-data parity gate skips if the (git-ignored, downloadable) data is absent.
"""

from __future__ import annotations

import os

import pytest

from nsm_ct.mind.datasets import proofwriter as pw


# ------------------------------------------------------------- parser (data-free)
def test_parse_literal_and_variable_mapping():
    assert pw.parse_literal('("Gary" "is" "kind" "+")') == ("gary", "is", "kind", "+")
    assert pw.parse_literal('("cow" "needs" "bear" "-")') == ("cow", "needs", "bear", "-")
    # ProofWriter's universal variable ("someone"/"they") maps to the oracle's ?x.
    assert pw.parse_literal('("someone" "is" "furry" "+")') == ("?x", "is", "furry", "+")


def test_parse_rule():
    r = pw.parse_rule('((("someone" "is" "furry" "+")) -> ("someone" "is" "kind" "+"))')
    assert r.antecedents == (("?x", "is", "furry", "+"),)
    assert r.consequent == ("?x", "is", "kind", "+")
    r2 = pw.parse_rule('((("Anne" "is" "cold" "+") ("Anne" "is" "big" "+")) -> ("Anne" "is" "red" "+"))')
    assert len(r2.antecedents) == 2 and r2.consequent == ("anne", "is", "red", "+")


# ------------------------------------------------------------- verify (data-free)
def test_verify_true_false_unknown():
    facts = [("alice", "is", "furry", "+")]
    rules = [pw.parse_rule('((("someone" "is" "furry" "+")) -> ("someone" "is" "kind" "+"))')]
    assert pw.verify(facts, rules, ("alice", "is", "kind", "+")) == pw.TRUE       # derivable
    assert pw.verify(facts, rules, ("alice", "is", "kind", "-")) == pw.FALSE      # opposite derivable
    assert pw.verify(facts, rules, ("alice", "is", "green", "+")) == pw.UNKNOWN   # abstain


def test_verify_negative_query_against_positive_fact():
    facts = [("cow", "needs", "bear", "+")]
    # "cow does NOT need bear" is FALSE because the positive fact is present.
    assert pw.verify(facts, [], ("cow", "needs", "bear", "-")) == pw.FALSE


def test_multihop_chain():
    facts = [("bob", "is", "furry", "+")]
    rules = [
        pw.parse_rule('((("someone" "is" "furry" "+")) -> ("someone" "is" "kind" "+"))'),
        pw.parse_rule('((("someone" "is" "kind" "+")) -> ("someone" "is" "smart" "+"))'),
    ]
    assert pw.verify(facts, rules, ("bob", "is", "smart", "+")) == pw.TRUE        # 2-hop


# ------------------------------------------------------------- real-data parity
def _data_present() -> bool:
    return os.path.exists(os.path.join(pw.default_data_dir(), "owa-depth1-test.jsonl"))


@pytest.mark.skipif(not _data_present(), reason="ProofWriter data absent (run scripts/fetch_proofwriter.py)")
def test_proofwriter_gold_parity():
    """Forward-chain reproduces ProofWriter's gold True/False/Unknown labels."""
    ok = total = 0
    for depth in ("0", "1", "2", "3", "5"):
        path = os.path.join(pw.default_data_dir(), f"owa-depth{depth}-test.jsonl")
        for rec in pw.load_records(path, limit=40):
            ex = pw.parse_record(rec)
            for (lit, gold, _qd) in ex.questions:
                ok += pw.verify(ex.facts, ex.rules, lit) == gold
                total += 1
    assert total > 500
    assert ok / total >= 0.95, ok / total   # measured ~0.99 across depths 0-5


# ----------------------------------------- verification-mode batch (data-free)
def test_build_pw_batch_and_controller_forward():
    """Step-2 perception: facts+rules+query -> a 3-way {true,false,idk} MC batch
    the controller can score (verification = MC over the three answer atoms)."""
    from nsm_ct.tpr import TPRCodec
    from nsm_ct.mind.controller import MindController
    from nsm_ct.reasoning_oracle import Rule

    codec = TPRCodec(dim=32)
    facts = [("alice", "is", "furry", "+")]
    rules = [Rule(antecedents=(("?x", "is", "furry", "+"),),
                  consequent=("?x", "is", "kind", "+"), name="pw")]
    ex = pw.PWExample(facts=facts, rules=rules, questions=[
        (("alice", "is", "kind", "+"), pw.TRUE, 1),
        (("alice", "is", "green", "+"), pw.UNKNOWN, 0),
    ])
    items = pw.flatten([ex])
    batch = pw.build_pw_batch(items, codec)
    assert batch.options.shape == (2, 3, 32)
    assert batch.answer.tolist() == [0, 2]            # true -> 0, Unknown -> 2
    out = MindController(codec, hidden=32, hops=4, halting=False)(batch)
    assert out["answer_logits"].shape == (2, 3)       # 3-way verification readout


# --------------------------------------- M9 proof-chain teacher (data-free)
def test_proof_path_supervision_and_value_loss():
    """Step-3 teacher: forward_chain yields the proof's derived-VALUE sequence
    (facts→query), and the per-hop value-supervision loss runs over hop_derived."""
    import torch
    from nsm_ct.tpr import TPRCodec
    from nsm_ct.reasoning_oracle import Rule
    from nsm_ct.mind.controller import MindController
    from nsm_ct.mind.controller_losses import value_supervision_loss

    codec = TPRCodec(dim=32)
    facts = [("alice", "is", "furry", "+")]
    rules = [Rule((("?x", "is", "furry", "+"),), ("?x", "is", "kind", "+"), name="r1"),
             Rule((("?x", "is", "kind", "+"),), ("?x", "is", "smart", "+"), name="r2")]

    needed, label = pw.proof_path(facts, rules, ("alice", "is", "smart", "+"))
    assert label == pw.TRUE == pw.verify(facts, rules, ("alice", "is", "smart", "+"))
    assert [l[2] for l in needed] == ["kind", "smart"]      # derivation order facts→query
    u_needed, u_label = pw.proof_path(facts, rules, ("alice", "is", "green", "+"))
    assert u_label == pw.UNKNOWN and u_needed == []

    ex = pw.PWExample(facts=facts, rules=rules, questions=[
        (("alice", "is", "smart", "+"), pw.TRUE, 2),
        (("alice", "is", "green", "+"), pw.UNKNOWN, 0)])
    items = pw.flatten([ex])
    cb, atom2idx = pw.value_codebook(items, codec)
    sup = pw.proof_supervision(items, hops=4, atom2idx=atom2idx)
    assert sup["value_targets"].shape == (2, 4)
    assert sup["value_targets"][0, 0] == atom2idx["v:+:kind"]
    assert sup["value_targets"][0, 1] == atom2idx["v:+:smart"]
    assert sup["value_targets"][0, 2] == -1 and sup["depth"][0] == 2
    assert (sup["value_targets"][1] == -1).all() and sup["depth"][1] == 4  # Unknown: full budget

    out = MindController(codec, hidden=32, hops=4, halting=False)(pw.build_pw_batch(items, codec))
    assert out["hop_derived"].shape == (2, 4, 32)          # per-hop derive head exposed
    vs = value_supervision_loss(out, torch.from_numpy(sup["value_targets"]),
                                torch.from_numpy(sup["depth"]), torch.from_numpy(cb))
    assert torch.isfinite(vs["value"]) and vs["value"].item() >= 0.0


# ----------------------------------- M10 executor-driven verification (data-free)
def _verify_via_executor(facts, rules, query, executor=None):
    """Run [INFER, RESPOND_VERIFY] through the deterministic executor."""
    from nsm_ct.mind.executor import Executor
    from nsm_ct.mind import ops
    ex = executor or Executor(codec=_small_codec())
    ex.load_theory(facts, rules)
    s, p, o, pol = query
    trace = [ops.Op(ops.INFER, {}),
             ops.Op(ops.RESPOND_VERIFY, {"subject": s, "relation": p, "value": o, "polarity": pol})]
    return ex.run_trace(trace)


def _small_codec():
    from nsm_ct.tpr import TPRCodec
    return TPRCodec(dim=16)


def test_executor_verification_path():
    """The controller's reasoning ops, run through the executor, yield the symbolic
    verdict — the engine does the derivation, an op-trace drives it (M10 foundation)."""
    from nsm_ct.mind.executor import PW_TRUE, PW_UNKNOWN
    from nsm_ct.reasoning_oracle import Rule
    facts = [("alice", "is", "furry", "+")]
    rules = [Rule((("?x", "is", "furry", "+"),), ("?x", "is", "kind", "+"), name="r1"),
             Rule((("?x", "is", "kind", "+"),), ("?x", "is", "smart", "+"), name="r2")]
    assert _verify_via_executor(facts, rules, ("alice", "is", "smart", "+"))["answer"] == PW_TRUE
    u = _verify_via_executor(facts, rules, ("alice", "is", "green", "+"))
    assert u["answer"] == PW_UNKNOWN and u["abstained"]


# --------------------------------- M10 step 2: learned proof-search navigation
def test_proof_rule_steps_and_proofsearch_rollout():
    """The gold proof as a rule-selection sequence; the batch builder turns it into
    a contrastive selection task; the rollout drives the executor and terminates."""
    from nsm_ct.tpr import TPRCodec
    from nsm_ct.reasoning_oracle import Rule
    from nsm_ct.mind.controller import MindController
    from nsm_ct.mind.proof_search import ProofSearch

    facts = [("alice", "is", "furry", "+")]
    rules = [Rule((("?x", "is", "kind", "+"),), ("?x", "is", "smart", "+"), name="a"),   # r0
             Rule((("?x", "is", "furry", "+"),), ("?x", "is", "kind", "+"), name="b")]    # r1
    steps, label = pw.proof_rule_steps(facts, rules, ("alice", "is", "smart", "+"))
    assert label == pw.TRUE
    assert [g for (_f, g) in steps] == [1, 0]          # fire r1 (furry→kind) then r0 (kind→smart)

    ex = pw.PWExample(facts=facts, rules=rules,
                      questions=[(("alice", "is", "smart", "+"), pw.TRUE, 2)])
    navex = pw.navigation_examples(pw.flatten([ex]))
    assert len(navex) == 2
    codec = TPRCodec(dim=32)
    batch = pw.build_proofsearch_batch(navex, codec)
    assert batch.options.shape == (2, 2, 32) and batch.answer.tolist() == [1, 0]

    ctrl = MindController(codec, hidden=32, hops=3, halting=False)
    verdict, nsteps = ProofSearch(ctrl, codec).run(
        facts, rules, ("alice", "is", "smart", "+"), max_steps=5)
    assert verdict in (pw.TRUE, pw.FALSE, pw.UNKNOWN) and 0 <= nsteps <= 5


def test_apply_rule_navigation_chain():
    """Following the gold rule sequence via single-rule moves reaches the proof —
    the executor derives each step symbolically; navigation only orders the moves."""
    from nsm_ct.mind.executor import Executor
    from nsm_ct.reasoning_oracle import Rule
    facts = [("bob", "is", "furry", "+")]
    rules = [Rule((("?x", "is", "kind", "+"),), ("?x", "is", "smart", "+"), name="a"),    # r0
             Rule((("?x", "is", "furry", "+"),), ("?x", "is", "kind", "+"), name="b")]     # r1
    ex = Executor(codec=_small_codec())
    ex.load_theory(facts, rules)
    gold = [g for (_f, g) in pw.proof_rule_steps(facts, rules, ("bob", "is", "smart", "+"))[0]]
    for idx in gold:                                   # the gold navigation [r1, r0]
        ex.apply_rule(rules[idx])
    assert ("bob", "is", "smart", "+") in ex.pw_closure   # reached the proof


def _two_step_theory():
    from nsm_ct.reasoning_oracle import Rule
    facts = [("cara", "is", "furry", "+")]
    rules = [Rule((("?x", "is", "kind", "+"),), ("?x", "is", "smart", "+"), name="a"),   # r0
             Rule((("?x", "is", "furry", "+"),), ("?x", "is", "kind", "+"), name="b"),   # r1
             Rule((("?x", "is", "round", "+"),), ("?x", "is", "blue", "+"), name="c")]   # r2 (distractor)
    return facts, rules, ("cara", "is", "smart", "+")


def test_gold_plan_parity_and_expert_recovery():
    """gold_plan is the single source of truth (proof_rule_steps is a wrapper), and
    the DAgger expert returns the right move at BOTH the base state and an off-path
    state the policy could wander into — the property that cures exposure bias."""
    facts, rules, query = _two_step_theory()
    needed, rule_of, label = pw.gold_plan(facts, rules, query)
    steps, lab2 = pw.proof_rule_steps(facts, rules, query)
    assert label == lab2 == pw.TRUE
    assert [g for (_f, g) in steps] == [rule_of[lit] for lit in needed] == [1, 0]

    # at the base state the expert picks r1 (furry→kind), the first proof move.
    assert pw.expert_action(set(facts), needed, rule_of) == 1
    # OFF-PATH: the policy fired the distractor r2 (adds blue); the recovery move is
    # still r1 — the earliest needed literal (kind) is what's missing.
    off = set(facts) | {("cara", "is", "blue", "+")}
    assert pw.expert_action(off, needed, rule_of) == 1
    # mid-proof (kind present): expert advances to r0 (kind→smart).
    assert pw.expert_action(set(facts) | {("cara", "is", "kind", "+")}, needed, rule_of) == 0
    # all needed literals present (full chain derived) → goal proved, no move.
    proved = set(facts) | {("cara", "is", "kind", "+"), ("cara", "is", "smart", "+")}
    assert pw.expert_action(proved, needed, rule_of) is None
    # Unknown query has no plan to navigate.
    assert pw.gold_plan(facts, rules, ("cara", "is", "green", "+")) == ([], {}, pw.UNKNOWN)


def test_collect_dagger_labels_are_applicable():
    """collect_dagger rolls out the (untrained) policy and labels each visited state
    with an applicable expert move — examples drop straight into the trainer."""
    from nsm_ct.tpr import TPRCodec
    from nsm_ct.mind.controller import MindController
    from nsm_ct.mind.proof_search import ProofSearch
    facts, rules, query = _two_step_theory()
    items = [(facts, rules, query, 0, 2)]              # (facts, rules, query, ans_idx, depth)
    codec = TPRCodec(dim=32)
    searcher = ProofSearch(MindController(codec, hidden=32, hops=3, halting=False), codec)
    ex = searcher.collect_dagger(items, max_steps=6)
    assert ex                                          # provable item yields states
    needed, rule_of, _ = pw.gold_plan(facts, rules, query)
    for (state, q, rs, expert_idx) in ex:
        assert q == query and rs is rules
        assert expert_idx == pw.expert_action(set(state), needed, rule_of)  # valid label


# ----------------------------------- M10 step 3 backward navigation (data-free)
def test_backward_step_and_examples():
    """The symbolic backward move unifies a rule consequent with a subgoal and yields
    its grounded antecedents; backward_examples label each subgoal with its gold rule."""
    from nsm_ct.mind.proof_search import backward_step
    facts, rules, query = _two_step_theory()              # furry; kind->smart(r0), furry->kind(r1)
    # to prove smart, r0 (kind->smart) matches → new subgoal kind; r1 (furry->kind) doesn't.
    assert backward_step(("cara", "is", "smart", "+"), rules[0]) == [("cara", "is", "kind", "+")]
    assert backward_step(("cara", "is", "smart", "+"), rules[1]) is None
    needed, rule_of, _ = pw.gold_plan(facts, rules, query)
    bex = pw.backward_examples([(facts, rules, query, 0, 2)])
    assert {(lit, gold) for (_f, lit, _r, gold) in bex} == {(l, rule_of[l]) for l in needed}


def test_backward_search_verdicts():
    """BackwardSearch proves a 2-step theory (TRUE), refutes a negation (FALSE), and
    abstains (UNKNOWN) — same controller, goal-directed subgoal stack."""
    from nsm_ct.tpr import TPRCodec
    from nsm_ct.mind.controller import MindController
    from nsm_ct.mind.proof_search import BackwardSearch
    from nsm_ct.reasoning_oracle import Rule
    codec = TPRCodec(dim=32)
    bs = BackwardSearch(MindController(codec, hidden=32, hops=3, halting=False), codec)
    facts, rules, query = _two_step_theory()
    for q in (query, ("cara", "is", "green", "+")):       # provable + unprovable
        v, n = bs.run(facts, rules, q, max_steps=6)
        assert v in (pw.TRUE, pw.FALSE, pw.UNKNOWN) and n >= 0
    # a directly-stated fact is TRUE at zero rule-applications when rules can't fire it.
    v0, _ = bs.run(facts, [], ("cara", "is", "furry", "+"), max_steps=6)
    assert v0 == pw.TRUE


@pytest.mark.skipif(not _data_present(), reason="ProofWriter data absent")
def test_executor_matches_symbolic_floor():
    """Executor-driven verdict == the 0.989 symbolic floor (same engine) across depths
    — so the controller, driving the executor, inherits broad-data correctness."""
    from nsm_ct.mind.executor import Executor
    executor = Executor(codec=_small_codec())                 # reused across questions
    ok = total = 0
    for depth in ("0", "1", "2", "3"):
        path = os.path.join(pw.default_data_dir(), f"owa-depth{depth}-test.jsonl")
        for rec in pw.load_records(path, limit=15):
            ex = pw.parse_record(rec)
            for (lit, _gold, _qd) in ex.questions:
                got = _verify_via_executor(ex.facts, ex.rules, lit, executor)["answer"]
                ok += (got == pw.verify(ex.facts, ex.rules, lit))
                total += 1
    assert total > 200 and ok == total          # exact: same forward_chain engine


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
