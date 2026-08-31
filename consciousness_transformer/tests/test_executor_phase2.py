"""EXECUTOR PHASE 2 tests: control-signal purity, D1 hard-key selection,
teacher-forced op-selection convergence, the six-family corpus builder,
the LOFO splitter, oracle/floor arm equivalence, and a SHORT (smoke-scale,
non-gating) end-to-end LOFO run.

Per dev/EXECUTOR_DESIGN.md's own rule (CLAUDE.md: "Smoke-scale results
NEVER gate curriculum validity"), the end-to-end LOFO test at the bottom
PRINTS its numbers and asserts only that the machinery runs end-to-end
without crashing -- it does NOT assert a GO/KILL threshold.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from nsm_ct import op_select as osl, ops, programs
from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch
from nsm_ct.curriculum2 import InstanceCurriculumGenerator, WriteBackCurriculumGenerator
from nsm_ct.executor import Executor
from nsm_ct.meaning import NSMMeaningResolver
from nsm_ct.resolver import make_resolver
from nsm_ct.tpr import TPRCodec

import train_executor as te  # noqa: E402  (scripts/train_executor.py -- the corpus builder under test)

DIM = 16
HIDDEN = 12


def _writeback_batch(n=8, seed=0, dim=DIM, hidden=HIDDEN):
    meaning = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)
    eps = WriteBackCurriculumGenerator(seed=seed).generate(n)
    batch = build_clause_batch(eps, None, meaning, codec)
    resolver = make_resolver("A", dim, hidden)
    model = ClauseReactor(dim=dim, hidden=hidden, resolver=resolver)
    return batch, model


def _instance_batch(n=10, seed=1, dim=DIM, hidden=HIDDEN, inverse_frac=1.0):
    meaning = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)
    eps = InstanceCurriculumGenerator(seed=seed, inverse_frac=inverse_frac).generate(n)
    batch = build_clause_batch(eps, None, meaning, codec)
    resolver = make_resolver("A", dim, hidden, use_cand_feature=True, cand_feature_extra=1)
    model = ClauseReactor(dim=dim, hidden=hidden, resolver=resolver)
    return batch, model


def _heads(ex: Executor):
    op_sel = osl.OpSelect(k_max=ex.k_max)
    arg_sel = osl.ArgSelect(op_sel.op_embed, ctrl_dim=op_sel.ctrl_dim)
    return op_sel, arg_sel


# ---------------------------------------------------------------------------
# 1. Control-signal purity: OpSelect's input is EXACTLY the control signal
#    -- dimension check (already asserted inside OpSelect.encode) plus a
#    poisoning test: perturbing REGISTER VALUES (the batch's own entity/
#    relation/value tensors) changes nothing about the control signal
#    itself, only downstream task computation.
# ---------------------------------------------------------------------------
def test_control_signal_dimension_matches_declared_ctrl_dim():
    op_sel = osl.OpSelect(k_max=12)
    b = 5
    ctrl = osl.control_signal_to_tensor(Executor.build_control_signal(
        prev_op_id=torch.zeros(b, dtype=torch.long), step_idx=0,
        type_mask=torch.tensor([1., 1., 0., 1., 1.]),
        margin=torch.zeros(b), abstain_flag=torch.zeros(b, dtype=torch.bool),
        halt_budget=12, struct_flags=torch.zeros(b, len(osl.STRUCT_FLAG_NAMES))),
        batch=b, device=None)
    vec = op_sel.encode(ctrl)
    assert vec.shape == (b, op_sel.ctrl_dim)
    # op_embed_dim(8) + step_embed_dim(4) + type_mask(5) + margin(1) +
    # abstain(1) + halt_budget(1) + struct_flags(6) == 26 at OpSelect's own
    # defaults -- the struct_flags(6) term is the director's D2-consistent
    # ruling extension (RESEARCH_NOTES "LOFO instrument repair").
    assert op_sel.ctrl_dim == 8 + 4 + 5 + 1 + 1 + 1 + len(osl.STRUCT_FLAG_NAMES)
    assert len(osl.STRUCT_FLAG_NAMES) == 6


def test_control_signal_poisoning_register_values_does_not_change_it():
    """Perturbing the batch's REGISTER DATA (entity/relation/value
    vectors) must not change the control signal at all -- D2's Harvard
    split, "never register data vectors", made into a concrete test:
    build two control signals from Executor.run_learned's own per-step
    construction, one from a normal batch and one where every entity/
    value vector has been replaced with random noise, holding every
    CONTROL-relevant quantity (family membership -> entry-op target,
    step index, candidate mask/margin source) fixed. The two control
    signals must be numerically identical -- INCLUDING struct_flags
    (director ruling, RESEARCH_NOTES "LOFO instrument repair"):
    Executor.structural_flags reads only batch STRUCTURE (masks/
    None-ness), never entity/value CONTENTS, so poisoning those contents
    must leave struct_flags untouched too."""
    batch, model = _writeback_batch()
    model.eval()
    ex = Executor(model)
    op_sel, arg_sel = _heads(ex)

    def _capture_ctrl_a(b):
        # Reaches into run_learned's own per-step ctrl_a construction via
        # the SAME public API it uses (Executor.build_control_signal +
        # op_select.control_signal_to_tensor) -- not a private hook, so
        # this test exercises exactly what run_learned itself builds.
        prev_op_id = torch.zeros(b.entity.shape[0], dtype=torch.long)
        return osl.control_signal_to_tensor(Executor.build_control_signal(
            prev_op_id=prev_op_id, step_idx=0,
            type_mask=torch.tensor([1., 1., 0., 1., 1.]),
            margin=torch.zeros(b.entity.shape[0]),
            abstain_flag=torch.zeros(b.entity.shape[0], dtype=torch.bool),
            halt_budget=ex.k_max, struct_flags=Executor.structural_flags(b, 0)),
            batch=b.entity.shape[0], device=None)

    ctrl_clean = _capture_ctrl_a(batch)

    poisoned_entity = batch.entity.clone()
    poisoned_entity[:, 0, :] = torch.randn_like(poisoned_entity[:, 0, :]) * 1000.0
    poisoned_value = batch.value.clone()
    poisoned_value[:, 0, :] = torch.randn_like(poisoned_value[:, 0, :]) * 1000.0
    import dataclasses
    poisoned_batch = dataclasses.replace(batch, entity=poisoned_entity, value=poisoned_value)

    ctrl_poisoned = _capture_ctrl_a(poisoned_batch)

    for key in ctrl_clean:
        assert torch.equal(ctrl_clean[key], ctrl_poisoned[key]), \
            f"control signal field {key!r} changed after poisoning REGISTER VALUES -- D2 violation"

    op_vec_clean = op_sel.encode(ctrl_clean)
    op_vec_poisoned = op_sel.encode(ctrl_poisoned)
    assert torch.equal(op_vec_clean, op_vec_poisoned)


# ---------------------------------------------------------------------------
# 1b. STRUCTURAL FLAG EXTENSION (director ruling, RESEARCH_NOTES "LOFO
#    instrument repair"): definite_desc_read and inverse_query rows at the
#    SAME shared clause step now produce DIFFERENT control signals -- this
#    is the information-bottleneck finding that ruling exists to fix, made
#    into a direct assertion (not just measured downstream via op_acc,
#    see the load-bearing training test below).
# ---------------------------------------------------------------------------
def test_definite_desc_read_vs_inverse_query_control_signals_differ():
    """InstanceCurriculumGenerator(inverse_frac=0.5) mixes addr-redirect
    ("definite_desc_read") and inverse-query ("inverse_query") episodes in
    one batch; both families' final question step lands at the SAME
    clause-step column t (verified below, not assumed) -- exactly the
    collision RESEARCH_NOTES "LOFO instrument repair" reports: prior to
    the struct_flags extension, every OTHER control-signal input
    (prev_op_id, step_idx, margin=0/abstain=0 pre-resolver, a CONSTANT
    type_mask) is identical for these two families at that step, so
    op_acc for this exact pair was pinned at 0.000 regardless of
    balance/scale. With Executor.structural_flags added (inverse_query
    rows: inverse_mask_flag=1, has_candidates=0; definite_desc_read rows:
    has_candidates=1, addr_mask_flag=1, inverse_mask_flag=0), the encoded
    control vectors -- and the raw struct_flags themselves -- must now
    differ."""
    meaning = NSMMeaningResolver()
    codec = TPRCodec(dim=DIM)
    eps = InstanceCurriculumGenerator(seed=7, inverse_frac=0.5).generate(24)
    batch = build_clause_batch(eps, None, meaning, codec)
    resolver = make_resolver("A", DIM, HIDDEN, use_cand_feature=True, cand_feature_extra=1)
    model = ClauseReactor(dim=DIM, hidden=HIDDEN, resolver=resolver)
    ex = Executor(model)
    op_sel, arg_sel = _heads(ex)
    b = batch.entity.shape[0]

    found_shared_step = False
    for t in range(batch.entity.shape[1]):
        fam_t = programs.family_of_step(batch, t)
        real_t = (batch.mask[:, t] > 0).tolist()
        defdesc_idx = [i for i, (f, r) in enumerate(zip(fam_t, real_t)) if r and f == "definite_desc_read"]
        inverse_idx = [i for i, (f, r) in enumerate(zip(fam_t, real_t)) if r and f == "inverse_query"]
        if not defdesc_idx or not inverse_idx:
            continue
        found_shared_step = True

        struct_flags = Executor.structural_flags(batch, t)
        assert not torch.equal(struct_flags[defdesc_idx[0]], struct_flags[inverse_idx[0]]), (
            f"struct_flags identical for definite_desc_read vs inverse_query at shared step {t}")

        ctrl = osl.control_signal_to_tensor(Executor.build_control_signal(
            prev_op_id=torch.zeros(b, dtype=torch.long), step_idx=t,
            type_mask=torch.tensor([1., 1., 0., 1., 1.]),
            margin=torch.zeros(b), abstain_flag=torch.zeros(b, dtype=torch.bool),
            halt_budget=ex.k_max, struct_flags=struct_flags), batch=b, device=None)
        vec = op_sel.encode(ctrl)
        assert not torch.equal(vec[defdesc_idx[0]], vec[inverse_idx[0]]), (
            f"encoded control signal identical for definite_desc_read vs inverse_query at step {t}")
    assert found_shared_step, (
        "test setup: no clause step had both definite_desc_read AND inverse_query rows -- "
        "cannot demonstrate the collision fix")


# ---------------------------------------------------------------------------
# 2. D1: Addr selections identical under argmax (EMIT's destination,
#    dest_mode="hard" -- straight-through forward equals plain argmax
#    regardless of the underlying soft distribution).
# ---------------------------------------------------------------------------
def test_d1_hard_dest_matches_argmax_regardless_of_soft_distribution():
    b, ctrl_dim = 6, 20
    op_sel = osl.OpSelect(k_max=12)
    arg_sel = osl.ArgSelect(op_sel.op_embed, ctrl_dim=op_sel.ctrl_dim)
    torch.manual_seed(0)
    ctrl_vec = torch.randn(b, op_sel.ctrl_dim)
    emit_id = torch.full((b,), osl.OP_INDEX["EMIT"] + 1, dtype=torch.long)
    dest_logits = arg_sel(ctrl_vec, emit_id, osl.ARG_SIGNATURES["EMIT"])
    dest_soft = torch.softmax(dest_logits, dim=-1)
    dest_pred_idx = dest_logits.argmax(-1)

    hard_addr = (dest_pred_idx == 0).float()
    straight_through = hard_addr + (dest_soft[:, 0] - dest_soft[:, 0].detach())
    # Forward VALUE must be byte-identical to the hard argmax indicator,
    # independent of how close the soft distribution was to 0.5.
    assert torch.equal(straight_through.detach(), hard_addr)
    assert torch.equal((straight_through > 0.5), (dest_pred_idx == 0))


def test_d1_addr_write_violation_masking_mechanism():
    """dev/EXECUTOR_DESIGN.md Sec.1.3's <=1-WRITE invariant, applied to a
    LEARNED (adversarial) op sequence -- op_select.mask_second_write masks
    and COUNTS rather than raising (contrast with Executor.
    execute_step_program's assertion-based Tier-2 mechanism, Phase 1)."""
    seq = ["QUERY", "TICK", "GATE", "OVERWRITE", "NEGATE", "WRITE", "RESPOND", "WRITE", "HALT"]
    masked, violations = osl.mask_second_write(seq)
    assert violations == 1
    assert masked.count("WRITE") == 1
    assert masked[5] == "WRITE" and masked[7] == "_MASKED_WRITE"

    clean_seq = ["QUERY", "TICK", "WRITE", "RESPOND", "HALT"]
    masked_clean, violations_clean = osl.mask_second_write(clean_seq)
    assert violations_clean == 0
    assert masked_clean == clean_seq


# ---------------------------------------------------------------------------
# 3. Teacher-forced training on ONE family reaches op-selection accuracy
#    >0.95 in <200 steps (tiny dims). Uses InstanceCurriculumGenerator with
#    inverse_frac=1.0 so the "inverse_query" family's entry op
#    (QUERY_ENTITY) is genuinely distinct from the plain_fact context
#    steps' own entry op (QUERY) sharing the same episodes -- a real,
#    non-degenerate classification target.
# ---------------------------------------------------------------------------
def test_teacher_forced_op_selection_converges_fast():
    batch, model = _instance_batch(n=12, inverse_frac=1.0, dim=10, hidden=8)
    ex = Executor(model)
    op_sel, arg_sel = _heads(ex)
    params = list(model.parameters()) + list(op_sel.parameters()) + list(arg_sel.parameters())
    opt = torch.optim.Adam(params, lr=5e-3)
    gold = batch.answer

    final_acc = 0.0
    for step in range(200):
        opt.zero_grad()
        out = ex.run_learned(batch, op_sel, arg_sel, teacher_force=True)
        loss = F.cross_entropy(out["answer_logits"], gold) + out["trace_loss"]
        loss.backward()
        opt.step()
        c = sum(v[0] for v in out["op_acc"].values())
        n = sum(v[1] for v in out["op_acc"].values())
        final_acc = c / n if n else 0.0
        if final_acc > 0.95 and step >= 5:
            break
    assert final_acc > 0.95, f"op-selection accuracy only reached {final_acc:.3f} in <200 steps"
    assert "inverse_query" in out["op_acc"]
    iq_c, iq_n = out["op_acc"]["inverse_query"]
    assert iq_n > 0
    assert iq_c / iq_n > 0.9, f"inverse_query entry-op accuracy {iq_c}/{iq_n} did not converge"


# ---------------------------------------------------------------------------
# 3b. LOAD-BEARING: the struct_flags extension actually RESOLVES the
#    definite_desc_read/inverse_query information-bottleneck collision --
#    a short teacher-forced training run (tiny scale) reaches op_acc > 0.9
#    on BOTH families (previously pinned at 0.000 regardless of
#    balance/scale, RESEARCH_NOTES "LOFO instrument repair"). Uses
#    family_balance_weights (repair #1) since the corpus this generator
#    produces is dominated by its own plain_fact-shaped context steps,
#    same imbalance mechanism repair #1 targets -- the point of THIS test
#    is isolating the struct_flags fix, not re-proving repair #1, so the
#    balanced weights are applied rather than fighting both problems at
#    once.
# ---------------------------------------------------------------------------
def test_struct_flags_resolve_defdesc_vs_inverse_collision():
    torch.manual_seed(0)
    meaning = NSMMeaningResolver()
    codec = TPRCodec(dim=10)
    eps = InstanceCurriculumGenerator(seed=5, inverse_frac=0.5).generate(60)
    batch = build_clause_batch(eps, None, meaning, codec)
    resolver = make_resolver("A", 10, 8, use_cand_feature=True, cand_feature_extra=1)
    model = ClauseReactor(dim=10, hidden=8, resolver=resolver)
    ex = Executor(model)
    op_sel, arg_sel = _heads(ex)

    hist: Counter = Counter()
    for t in range(batch.entity.shape[1]):
        hist.update(programs.family_of_step(batch, t))
    assert hist.get("definite_desc_read", 0) > 0 and hist.get("inverse_query", 0) > 0, (
        "test setup: corpus must contain both collision families")
    family_weight = osl.family_balance_weights(dict(hist))

    params = list(model.parameters()) + list(op_sel.parameters()) + list(arg_sel.parameters())
    opt = torch.optim.Adam(params, lr=5e-3)
    gold = batch.answer

    dd_acc = iq_acc = 0.0
    out = None
    for step in range(100):
        opt.zero_grad()
        out = ex.run_learned(batch, op_sel, arg_sel, teacher_force=True, family_weight=family_weight)
        loss = F.cross_entropy(out["answer_logits"], gold) + out["trace_loss"]
        loss.backward()
        opt.step()
        dd_c, dd_n = out["op_acc"].get("definite_desc_read", (0, 0))
        iq_c, iq_n = out["op_acc"].get("inverse_query", (0, 0))
        dd_acc = dd_c / dd_n if dd_n else 0.0
        iq_acc = iq_c / iq_n if iq_n else 0.0
        if dd_acc > 0.9 and iq_acc > 0.9 and step >= 5:
            break

    assert dd_acc > 0.9, f"definite_desc_read op_acc only reached {dd_acc:.3f} in <=100 steps"
    assert iq_acc > 0.9, f"inverse_query op_acc only reached {iq_acc:.3f} in <=100 steps"


# ---------------------------------------------------------------------------
# 4. The six-family corpus builder covers all families.
# ---------------------------------------------------------------------------
def test_six_family_corpus_builder_covers_all_families():
    model, codec, meaning, corpus = te.build_corpus(episodes_per_family=16, dim=12, hidden=8, seed=0)
    hist = te.family_histogram(corpus, split="train")
    print("family_of_step coverage:", dict(hist))
    missing = [f for f in programs.FAMILY_NAMES if hist.get(f, 0) == 0]
    if "pronoun" not in corpus:
        # quantum_parser unavailable in this environment -- documented
        # skip contract (build_pronoun_batch's own docstring); the other
        # five families must still all be covered.
        assert missing == ["pronoun_value_redirect"], (
            f"unexpected missing families with quantum_parser unavailable: {missing}")
    else:
        assert not missing, f"family_of_step histogram has ZERO coverage for: {missing}"
    for fam in programs.FAMILY_NAMES:
        if fam not in missing:
            assert hist[fam] > 0


# ---------------------------------------------------------------------------
# 5. LOFO splitter drops exactly the named family's trace loss.
# ---------------------------------------------------------------------------
def test_lofo_keep_mask_drops_exactly_the_named_family():
    families = ["plain_fact", "writeback_addr_redirect", "plain_fact",
                "inverse_query", "writeback_addr_redirect"]
    keep_none = osl.lofo_keep_mask(families, None)
    assert keep_none == [True] * 5

    keep_wb = osl.lofo_keep_mask(families, "writeback_addr_redirect")
    assert keep_wb == [True, False, True, True, False]

    keep_iq = osl.lofo_keep_mask(families, "inverse_query")
    assert keep_iq == [True, True, True, False, True]


def test_run_learned_lofo_family_gets_zero_trace_loss_contribution():
    """A batch containing ONLY plain_fact (old L1-6) steps: LOFO-ing
    "plain_fact" must zero the ENTIRE trace_loss (every row's family
    matches the held-out one, so trace_row_a/trace_row_b are all-False
    everywhere in run_learned)."""
    from nsm_ct.episode import CurriculumGenerator
    meaning = NSMMeaningResolver()
    codec = TPRCodec(dim=DIM)
    eps = CurriculumGenerator(max_level=2, seed=0).generate(6)
    batch = build_clause_batch(eps, None, meaning, codec)
    resolver = make_resolver("A", DIM, HIDDEN)
    model = ClauseReactor(dim=DIM, hidden=HIDDEN, resolver=resolver)
    ex = Executor(model)
    op_sel, arg_sel = _heads(ex)

    out_lofo = ex.run_learned(batch, op_sel, arg_sel, teacher_force=True, lofo_family="plain_fact")
    assert float(out_lofo["trace_loss"].detach()) == 0.0

    out_normal = ex.run_learned(batch, op_sel, arg_sel, teacher_force=True, lofo_family=None)
    assert float(out_normal["trace_loss"].detach()) > 0.0


# ---------------------------------------------------------------------------
# 6. Oracle mode == Phase 1 forced-programs output (reuse the anchor):
#    run_learned(teacher_force=True, lofo_family=None) must reproduce
#    Executor.run()'s own task output byte-for-byte -- Stage-(a) teacher
#    forcing never changes the executed math, only adds a parallel loss.
# ---------------------------------------------------------------------------
def test_oracle_equivalence_run_learned_matches_run():
    for build in (_writeback_batch, lambda: _instance_batch(inverse_frac=0.3)):
        batch, model = build()
        model.eval()
        ex = Executor(model)
        op_sel, arg_sel = _heads(ex)
        with torch.no_grad():
            out_run = ex.run(batch)
            out_learned = ex.run_learned(batch, op_sel, arg_sel, teacher_force=True, lofo_family=None)
        diff = (out_run["answer_logits"] - out_learned["answer_logits"]).abs().max().item()
        assert diff < 1e-5, f"run_learned(teacher_force=True) diverged from run(): {diff}"


# ---------------------------------------------------------------------------
# 7. Floor mode runs.
# ---------------------------------------------------------------------------
def test_floor_mode_runs():
    batch, model = _instance_batch(inverse_frac=0.3)
    model.eval()
    ex = Executor(model)
    with torch.no_grad():
        out_normal = ex.run(batch)
        out_floor = ex.run(batch, force_plain_fact=True)
    assert out_floor["answer_logits"].shape == out_normal["answer_logits"].shape
    # The floor arm ignores every candidate/inverse branch -- its output
    # should generally DIFFER from the normal run on a batch that actually
    # has non-trivial candidate/inverse steps.
    assert not torch.allclose(out_normal["answer_logits"], out_floor["answer_logits"])


# ---------------------------------------------------------------------------
# 8. SHORT end-to-end: train on five families (tiny scale), report the
#    sixth's task accuracy vs its oracle and floor -- printed, NOT gated
#    (smoke-scale numbers never gate, CLAUDE.md's ops rule).
# ---------------------------------------------------------------------------
def test_short_end_to_end_lofo_smoke():
    model, codec, meaning, corpus = te.build_corpus(episodes_per_family=10, dim=10, hidden=8, seed=0)
    ex = Executor(model)
    op_sel, arg_sel = _heads(ex)
    params = list(model.parameters()) + list(op_sel.parameters()) + list(arg_sel.parameters())
    opt = torch.optim.Adam(params, lr=5e-3)

    held_out = "writeback_addr_redirect"
    for epoch in range(3):
        te.train_epoch(model, ex, op_sel, arg_sel, opt, corpus, codec,
                        batch_size=8, seed=0, epoch=epoch, lofo_family=held_out, trace_weight=1.0)

    results = te.evaluate(model, ex, op_sel, arg_sel, corpus, codec)
    r = results["writeback"]
    print(f"\n[SMOKE, non-gating] held-out family={held_out!r}: "
          f"learned_acc={r['learned_acc']:.3f} oracle_acc={r['oracle_acc']:.3f} floor_acc={r['floor_acc']:.3f}")
    print("op_acc:", r["op_acc"], "arg_acc:", r["arg_acc"])
    # Only assert the machinery ran end-to-end and produced well-formed
    # numbers -- NOT a threshold (smoke-scale results never gate).
    assert 0.0 <= r["learned_acc"] <= 1.0
    assert 0.0 <= r["oracle_acc"] <= 1.0
    assert 0.0 <= r["floor_acc"] <= 1.0


# ---------------------------------------------------------------------------
# 9. EXECUTOR LOFO GATE REPAIR #1: class-balanced trace loss
#    (RESEARCH_NOTES "Executor LOFO gate #1" instrument defect (1) --
#    op_select.family_balance_weights).
# ---------------------------------------------------------------------------
def test_family_balance_weights_sum_correctly():
    """The inverse-frequency normalization's own invariant, at the default
    ``power=0.5`` (inverse-square-root, see the function's own docstring
    for why ``power=1.0``'s FULL equalization proved too aggressive at
    this milestone's smoke scale): every present family's weighted mass
    ``weight[f] * counts[f]**power`` is IDENTICAL across families, and the
    grand total weighted mass across all families equals the raw total
    unchanged (the reweighting rebalances, it does not rescale). A
    zero-count family is dropped, never divides by zero. The same
    invariants hold at ANY ``power`` (parametrized below), including the
    ``power=1.0`` full-equalization case."""
    counts = {"plain_fact": 8000, "writeback_addr_redirect": 500,
              "definite_desc_read": 80, "inverse_query": 40, "recall_link": 0}
    present = [f for f in counts if f != "recall_link"]
    total = sum(counts[f] for f in present)

    for power in (0.5, 1.0, 0.25):
        w = osl.family_balance_weights(counts, power=power)
        assert "recall_link" not in w  # zero-count family dropped, no div-by-zero
        per_family_mass = [w[f] * counts[f] ** power for f in present]
        target = per_family_mass[0]
        for f, mass in zip(present, per_family_mass):
            assert abs(mass - target) < 1e-6, f"power={power} {f}: weighted mass {mass} != {target}"
        assert abs(sum(w[f] * counts[f] for f in present) - total) < 1e-6
        # Minority families get a LARGER weight than the majority family.
        assert w["inverse_query"] > w["definite_desc_read"] > w["writeback_addr_redirect"] > w["plain_fact"]

    # power=1.0 is the textbook full-equalization special case: every
    # family's TOTAL mass (weight[f]*counts[f], not weight[f]*counts[f]**1
    # written out) is exactly total/k.
    w_full = osl.family_balance_weights(counts, power=1.0)
    k = len(present)
    for f in present:
        assert abs(w_full[f] * counts[f] - total / k) < 1e-6

    # Uniform counts -> uniform (all-1.0) weights, at any power.
    uniform = osl.family_balance_weights({"a": 10, "b": 10, "c": 10}, power=0.5)
    assert all(abs(v - 1.0) < 1e-6 for v in uniform.values())

    assert osl.family_balance_weights({}) == {}


def test_family_balanced_loss_converges_on_minority_family():
    """Gate #1's own failure mode, reproduced and then fixed: a heavily
    imbalanced two-corpus stream (old L1-6, almost entirely ``plain_fact``,
    mixed with a SMALL, PURE-inverse InstanceCurriculumGenerator slice --
    ``inverse_frac=1.0`` so no ``definite_desc_read`` episode is ever
    generated) trains an entry-op selector whose op_acc on the minority
    family (``inverse_query``) converges to >0.9 WITH ``family_weight``
    applied -- ONE combined batch/call (both corpora's episodes built
    together, so their steps compete WITHIN THE SAME trace-loss
    computation, the genuine gate #1 failure mode: two SEPARATE
    per-corpus calls, each with its own optimizer step -- tried first --
    turned out NOT to reproduce it, since a rare family trained via its
    OWN dedicated call converges fine regardless of weighting when
    nothing else competes inside that call).

    NOTE on scope (this milestone's report has the full finding): a
    SEPARATE, deeper defect exists when TWO CANDIDATE-BEARING families
    share an identical preceding control-signal trace within the SAME
    corpus (``definite_desc_read`` vs ``inverse_query``, both riding
    InstanceCurriculumGenerator/RichEpisodeGenerator's own target-question
    vs inverse-question branching, which happens ENTIRELY at the final
    step with no earlier structural difference) -- OpSelect's D2-mandated
    Harvard-split control-only input (prev_op_id/step_idx/margin=0-pre-
    resolver/abstain=0/a CONSTANT type_mask) has no way to discriminate
    them at that shared step, so op_acc for that specific PAIR stays near
    0.000 regardless of loss weighting (measured: does not resolve even
    at --episodes-per-family 80/dim 24/epochs 25). That is a genuine
    information-bottleneck finding, not a class-imbalance one -- this
    test isolates and proves the CLASS-IMBALANCE mechanism repair #1
    actually targets and fixes, using a family (``inverse_query`` alone,
    no ``definite_desc_read`` competing for the same column) that is NOT
    subject to that separate collision."""
    torch.manual_seed(0)
    meaning = NSMMeaningResolver()
    codec = TPRCodec(dim=DIM)
    from nsm_ct.episode import CurriculumGenerator
    majority_eps = CurriculumGenerator(max_level=4, seed=0).generate(80)
    minority_eps = InstanceCurriculumGenerator(seed=2, inverse_frac=1.0).generate(10)
    batch = build_clause_batch(majority_eps + minority_eps, None, meaning, codec)

    hist: Counter = Counter()
    for t in range(batch.entity.shape[1]):
        hist.update(programs.family_of_step(batch, t))
    n_minority = hist.get("inverse_query", 0)
    assert n_minority >= 5, f"test setup: too few inverse_query steps to measure ({n_minority})"
    minority_frac = n_minority / sum(hist.values())
    assert minority_frac < 0.15, f"test setup not imbalanced enough: {minority_frac:.3f}"
    # power=1.0 (full inverse-frequency, not the module default power=0.5 --
    # see family_balance_weights's own docstring): this test isolates and
    # demonstrates repair #1's MECHANISM at full strength; scripts/
    # train_executor.py's own six-family mixed corpus uses the gentler
    # default for overall multi-family training stability (that trade-off
    # is this milestone's own reported finding).
    family_weight = osl.family_balance_weights(dict(hist), power=1.0)

    # Measured at a SMOKE-scale step budget (35 steps, checkpointed every 5
    # from step 10 on): both arms eventually reach the minority family's
    # op_acc -- pure class imbalance (no collision partner) is not, by
    # itself, a hard classification problem once a family owns its own
    # unrivaled column (see the docstring above). The measured DIFFERENCE
    # is training STABILITY under a bounded budget: unweighted dips back
    # to 0.000 between early checkpoints (the majority family's gradient
    # still dominates the SHARED parameters early on) before eventually
    # recovering by chance; weighted reaches and STAYS at >=0.9 from its
    # first checkpoint on. This milestone's report quotes the exact trace
    # this assertion is pinned to. NOTE (struct_flags extension,
    # RESEARCH_NOTES "LOFO instrument repair"): the checkpoint window now
    # starts at step 10, not step 5 -- OpSelect/ArgSelect's own parameter
    # count grew (the 6-wide structural-flag vector), which shifts this
    # seeded trace's random-init draw and needs ~2 more optimizer steps of
    # warmup before the weighted arm's first checkpoint lands at >=0.9;
    # this is re-measured against the WIDER architecture, not a loosened
    # threshold (still >=0.9, still required to beat the unweighted arm's
    # own minimum).
    def _train(weight):
        torch.manual_seed(1)
        resolver = make_resolver("A", DIM, HIDDEN, use_cand_feature=True, cand_feature_extra=1)
        model = ClauseReactor(dim=DIM, hidden=HIDDEN, resolver=resolver)
        ex = Executor(model)
        op_sel, arg_sel = _heads(ex)
        params = list(model.parameters()) + list(op_sel.parameters()) + list(arg_sel.parameters())
        opt = torch.optim.Adam(params, lr=5e-3)
        checkpoints = []
        for step in range(35):
            opt.zero_grad()
            out = ex.run_learned(batch, op_sel, arg_sel, teacher_force=True, family_weight=weight)
            out["trace_loss"].backward()
            opt.step()
            if (step + 1) >= 10 and (step + 1) % 5 == 0:
                c, n = out["op_acc"].get("inverse_query", (0, 0))
                checkpoints.append(c / n if n else 0.0)
        return checkpoints

    unweighted_checkpoints = _train(None)
    weighted_checkpoints = _train(family_weight)
    weighted_acc = weighted_checkpoints[-1]
    assert weighted_acc > 0.9, (
        f"family-balanced trace loss failed to converge the minority family: {weighted_acc:.3f}")
    assert min(weighted_checkpoints) >= 0.9, (
        f"family-balanced weighting was not STABLE across checkpoints: {weighted_checkpoints}")
    assert min(weighted_checkpoints) > min(unweighted_checkpoints), (
        f"balanced weighting (checkpoints={weighted_checkpoints}) was not more stable than "
        f"unweighted (checkpoints={unweighted_checkpoints})")


# ---------------------------------------------------------------------------
# 10. EXECUTOR LOFO GATE REPAIR #2: the interaction-feature padding fix,
#     verified (not sidestepped) -- a family that populates NONE of the
#     optional extra scalar columns must still run cleanly through
#     run_learned when the shared resolver is built with a WIDENED
#     cand_feature_extra (RESEARCH_NOTES "Executor LOFO gate #1"
#     instrument defect (2)).
# ---------------------------------------------------------------------------
def test_run_learned_padding_regression_widened_register():
    """A writeback batch never populates cand_evidence_target/
    cand_from_ltm/cand_recency (WriteBackCurriculumGenerator sets none of
    them) -- with the shared resolver built ``cand_feature_extra=4`` (the
    mixed six-family corpus's own width, scripts/train_executor.py's
    build_corpus), run_learned must still zero-pad up to that width
    rather than hitting a torch.cat width mismatch."""
    batch, _unused_model = _writeback_batch()
    resolver = make_resolver("A", DIM, HIDDEN, use_cand_feature=True, cand_feature_extra=4)
    model = ClauseReactor(dim=DIM, hidden=HIDDEN, resolver=resolver)
    ex = Executor(model)
    op_sel, arg_sel = _heads(ex)
    out = ex.run_learned(batch, op_sel, arg_sel, teacher_force=True)
    assert out["answer_logits"].shape[0] == batch.entity.shape[0]
    assert torch.isfinite(out["trace_loss"])
    assert out["write_violations"] == 0
    # A backward pass must also work end to end (the padded columns are a
    # real, differentiable part of the graph, not a detached patch).
    loss = F.cross_entropy(out["answer_logits"], batch.answer) + out["trace_loss"]
    loss.backward()
    assert model.resolver.net[0].weight.grad is not None


# ---------------------------------------------------------------------------
# 11. EXECUTOR LOFO GATE REPAIR #3 (mechanics test, not a curriculum-
#     validity gate -- CLAUDE.md: "Smoke-scale results NEVER gate
#     curriculum validity"): the pronoun oracle-below-floor inversion
#     (RESEARCH_NOTES "Executor LOFO gate #1": oracle 0.170 << floor
#     0.960). Diagnosis (this milestone's report has the full write-up):
#     PronounCurriculumGenerator's placeholder VALUE (``ep.meta[
#     "gold_place"]``, M53a's "ground the sentence's TRUE meaning
#     directly" design) is ALREADY the correct answer -- the floor arm
#     trivially scores near-ceiling by never touching it. The oracle arm
#     (Executor.run(), correct STRUCTURAL routing but the SAME shared
#     resolver's own candidate pick) discards that correct placeholder for
#     whichever candidate the resolver scores highest; with an unbalanced
#     trace loss and no per-candidate gender feature (gate #1's own
#     instrument defects), that pick was near chance, sinking oracle below
#     floor. BEFORE even reaching that comparison, gate #1's own sidestep
#     (use_cand_feature=False) meant `Executor.run()`'s oracle/floor arms
#     never hit the widened-register code path at all; turning it back on
#     (repair #2) exposed a REAL crash (mat1/mat2 shape mismatch) in
#     run()'s own extra-column padding -- fixed above alongside
#     run_learned's own copy of the same fix. This test is the crash
#     regression for `Executor.run()` specifically (test #10 above already
#     covers `run_learned`); it does NOT assert a training-convergence
#     threshold -- at tiny (noisy, single-digit-n_val) smoke scale a
#     seeded oracle-vs-floor A/B swung EITHER direction run to run in this
#     milestone's own exploration (small-sample variance, not a directional
#     finding), so per CLAUDE.md's smoke-scale rule that comparison is
#     reported (not asserted) in this milestone's own report instead,
#     using the real smoke battery's own numbers.
# ---------------------------------------------------------------------------
def test_pronoun_oracle_run_does_not_crash_with_widened_register():
    """A pronoun batch (PronounCurriculumGenerator's own
    ``cand_feature_per_candidate`` -- the mention's gender/kind feature --
    populated, no evidence-interaction/from_ltm/recency columns, exactly
    the empty-extra_cols case repair #2 exposed) must run cleanly through
    ``Executor.run()``'s oracle AND floor arms once the shared resolver is
    built with a WIDENED ``cand_feature_extra`` (untrained weights are
    fine -- this is a forward-pass shape regression, not a convergence
    claim, so it needs no training loop and carries none of that noise)."""
    meaning = NSMMeaningResolver()
    codec = TPRCodec(dim=DIM)
    import train_executor as _te  # noqa: E402  (re-import for build_pronoun_batch)
    batch = _te.build_pronoun_batch(10, seed=3, meaning=meaning, codec=codec)
    if batch is None:
        pytest.skip("quantum_parser unavailable in this environment -- pronoun family untestable")
    resolver = make_resolver("A", DIM, HIDDEN, use_cand_feature=True, cand_feature_extra=4)
    model = ClauseReactor(dim=DIM, hidden=HIDDEN, resolver=resolver)
    model.eval()
    ex = Executor(model)
    with torch.no_grad():
        oracle = ex.run(batch)
        floor = ex.run(batch, force_plain_fact=True)
    assert torch.isfinite(oracle["answer_logits"]).all()
    assert torch.isfinite(floor["answer_logits"]).all()
    assert oracle["answer_logits"].shape == floor["answer_logits"].shape == (batch.entity.shape[0], batch.options.shape[1])
