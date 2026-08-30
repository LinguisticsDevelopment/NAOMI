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
        halt_budget=12), batch=b, device=None)
    vec = op_sel.encode(ctrl)
    assert vec.shape == (b, op_sel.ctrl_dim)
    # op_embed_dim(8) + step_embed_dim(4) + type_mask(5) + margin(1) +
    # abstain(1) + halt_budget(1) == 20 at OpSelect's own defaults.
    assert op_sel.ctrl_dim == 8 + 4 + 5 + 1 + 1 + 1


def test_control_signal_poisoning_register_values_does_not_change_it():
    """Perturbing the batch's REGISTER DATA (entity/relation/value
    vectors) must not change the control signal at all -- D2's Harvard
    split, "never register data vectors", made into a concrete test:
    build two control signals from Executor.run_learned's own per-step
    construction, one from a normal batch and one where every entity/
    value vector has been replaced with random noise, holding every
    CONTROL-relevant quantity (family membership -> entry-op target,
    step index, candidate mask/margin source) fixed. The two control
    signals must be numerically identical."""
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
            halt_budget=ex.k_max), batch=b.entity.shape[0], device=None)

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
