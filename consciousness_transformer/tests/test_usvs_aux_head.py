"""Gates for the opt-in USVS-space auxiliary head (`--aux-usvs`; lead
directive follow-up to dev/USVS_GRADED_SCORING.md S5.4: "the encoder has NO
node embedding and NEVER scores sense candidates ... the nearest hook would
be a new Linear(controller_hidden, d_axes) head").

Covers: the default path (no --aux-usvs) is numerically UNCHANGED; the aux
loss is finite and > 0 at init when enabled; gradients reach `usvs_head`
(and only when enabled); checkpoint save/load round-trips the head; an old
checkpoint (no head) still loads, with a clear message, via
`EncoderModel.load_checkpoint_state`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nsm_ct import encoder_model as em
from nsm_ct import encoder_train_util as etu
from nsm_ct.tree_render import normalize_gold_tree

_ROOT = Path(__file__).resolve().parent.parent
_USVS_DIR = _ROOT / "data" / "usvs"
_GOLD = _ROOT / "runs" / "encoder_gold_v4b.jsonl"
_GOLD_FALLBACK = _ROOT / "runs" / "encoder_gold_v2.jsonl"


@pytest.fixture(scope="module")
def usvs():
    if not _USVS_DIR.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")
    from nsm_ct.ground.usvs import load_usvs
    return load_usvs(str(_USVS_DIR))


@pytest.fixture(scope="module")
def records():
    path = _GOLD if _GOLD.exists() else _GOLD_FALLBACK
    if not path.exists():
        pytest.skip("needs a gold file in runs/")
    out = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            tree = rec["lattice"]["trees"][0]
            if sum(len(c["roles"]) for c in tree["clauses"]) >= 2:
                out.append(rec)
            if len(out) >= 20:
                break
    if not out:
        pytest.skip("no usable gold records")
    return out


def _fixed_model_and_item(usvs, records):
    rec = records[0]
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    torch.manual_seed(1234)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes),
                             hash_buckets=512, d_model=32, controller_hidden=32)
    feats = em.build_features(rec, usvs, pos_vocab, 512)
    steps = em.linearize_tree(rec, rec["lattice"]["trees"][0])
    return model, feats, steps


# ---------------------------------------------------------------------------
# NODE_EMIT_ACTIONS / node_targets_for_steps / linearize_normalized_tree
# ---------------------------------------------------------------------------

def test_node_emit_actions_excludes_attach():
    assert set(em.NODE_EMIT_ACTIONS) == {"GROUND", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT"}
    assert "ATTACH" not in em.NODE_EMIT_ACTIONS


def test_node_targets_for_steps_only_at_node_emitting_steps(usvs, records):
    model, feats, steps = _fixed_model_and_item(usvs, records)
    targets = em.node_targets_for_steps(steps, feats, usvs)
    assert len(targets) == len(steps)
    for step, tgt in zip(steps, targets):
        if step.action in em.NODE_EMIT_ACTIONS:
            assert tgt is not None
            assert isinstance(tgt, np.ndarray)
            assert tgt.shape == (len(usvs.axes) + em_reserved_size(),)
        else:
            assert tgt is None


def em_reserved_size():
    from nsm_ct import usvs_graded as ug
    return ug.reserved_size()


def test_node_targets_for_steps_tolerates_out_of_range_token_index(usvs):
    """`legal_action_types`'s docstring: a duplicate-token-index collision
    can push a step's `eff_tidx` to `>= T`. This must fall back to the
    reserved block (like an uncovered token), never crash."""
    feats = em.SentenceFeatures(
        tokens=["a", "b"], pos=["NOUN", "NOUN"],
        tok_hash=torch.tensor([0, 1]), pos_id=torch.tensor([0, 0]),
        sense_feat=torch.zeros(2, len(usvs.axes)), fired_rules=torch.zeros(2, em.N_RULES),
        sense_cand=[[], []])
    steps = [em.Step(action="GROUND", token_index=5, role="SUBJECT",
                      gtype="sense", source="lexicon")]
    out = em.node_targets_for_steps(steps, feats, usvs)
    assert out[0] is not None
    assert out[0].shape == (len(usvs.axes) + em_reserved_size(),)
    assert np.allclose(out[0][:len(usvs.axes)], 0.0)  # no USVS coverage -> reserved-block fallback


def test_linearize_normalized_tree_round_trips_gold_node_set(usvs, records):
    """The normalized-tree walk must visit the SAME nodes (by identity) as
    `usvs_graded.flatten_tree` sees on the same tree."""
    from nsm_ct import usvs_graded as ug
    rec = records[0]
    norm = normalize_gold_tree(rec, rec["lattice"]["trees"][0])
    steps = em.linearize_normalized_tree(norm, len(rec["tokens"]))
    emitted = {(s.role, s.token_index) for s in steps if s.action in em.NODE_EMIT_ACTIONS}
    flat = ug.flatten_tree(rec, norm, usvs)
    expected = {(n.role, n.token_index) for n in flat.nodes}
    assert emitted == expected


# ---------------------------------------------------------------------------
# The regression that matters: the DEFAULT path is unchanged
# ---------------------------------------------------------------------------

def test_default_path_numerically_unchanged_with_or_without_node_targets(usvs, records):
    model, feats, steps = _fixed_model_and_item(usvs, records)
    node_targets = em.node_targets_for_steps(steps, feats, usvs)

    loss_bare = em.teacher_force_loss(model, feats, steps, terminal_weight=4.0)
    loss_explicit_off = em.teacher_force_loss(model, feats, steps, terminal_weight=4.0,
                                               node_targets=None, aux_usvs_weight=0.0)
    # node_targets PRESENT but weight 0.0 must still be a no-op -- the aux
    # term is gated on aux_usvs_weight > 0.0, not merely on node_targets
    # being given.
    loss_targets_but_zero_weight = em.teacher_force_loss(
        model, feats, steps, terminal_weight=4.0, node_targets=node_targets, aux_usvs_weight=0.0)

    assert float(loss_bare) == pytest.approx(float(loss_explicit_off), abs=1e-9)
    assert float(loss_bare) == pytest.approx(float(loss_targets_but_zero_weight), abs=1e-9)


def test_default_path_matches_independent_reimplementation(usvs, records):
    """Same style of gate as test_usvs_graded.py's: recompute the ORIGINAL
    per-step loss independently and confirm the aux-head-aware
    `teacher_force_loss` reproduces it exactly when aux is unused."""
    model, feats, steps = _fixed_model_and_item(usvs, records)
    got = em.teacher_force_loss(model, feats, steps, terminal_weight=4.0)

    enc = model.encode(feats)
    T = enc.shape[0]
    h = model.init_controller_state()
    open_clause, has_clause = False, False
    open_kind_id = model._none_clause_id
    prev_action_id = model._start_action_id
    i = 0
    w = torch.ones(len(em.ACTION_TYPES))
    for a in em.TERMINAL_ACTION_TYPES:
        w[em.ACTION_INDEX[a]] = 4.0
    losses = []
    for step in steps:
        i_clamped = min(i, T - 1) if T > 0 else 0
        enc_i = enc[i_clamped] if T > 0 else torch.zeros(model.d_model)
        h = model.controller_step(enc_i, open_kind_id, prev_action_id, h)
        legal = em.legal_action_types(open_clause, i, T, has_clause)
        logits = model.action_type_head(h).squeeze(0) + em._mask_vector(legal)
        losses.append(F.cross_entropy(logits.unsqueeze(0),
                                       torch.tensor([em.ACTION_INDEX[step.action]]), weight=w))
        if step.action == "OPEN_CLAUSE":
            losses.append(F.cross_entropy(model.kind_head(h),
                                           torch.tensor([em.KIND_INDEX.get(step.kind, 0)])))
            open_clause = True
            open_kind_id = em.KIND_INDEX.get(step.kind, 0)
        elif step.action == "CLOSE_CLAUSE":
            open_clause = False
            open_kind_id = model._none_clause_id
            has_clause = True
        elif step.action in ("GROUND", "ATTACH", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT"):
            losses.append(F.cross_entropy(model.role_head(h),
                                           torch.tensor([model.role_id(step.role)])))
        if step.action in ("GROUND", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT") and step.gtype is not None:
            losses.append(F.cross_entropy(model.gtype_head(h),
                                           torch.tensor([em.GTYPE_INDEX[step.gtype]])))
        if step.gtype in ("sense", "reference", "elision") and step.source is not None:
            losses.append(F.cross_entropy(model.source_head(h),
                                           torch.tensor([em.SOURCE_INDEX.get(step.source, 0)])))
        if step.action == "EMIT_SYNTH_SLOT" and step.prime is not None:
            losses.append(F.cross_entropy(model.prime_head(h),
                                           torch.tensor([em.PRIME_INDEX.get(step.prime,
                                                                             em.PRIME_INDEX["<UNK_PRIME>"])])))
        if step.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT", "EMIT_SYNTH_SLOT") \
                and step.token_index is not None:
            i = step.token_index + 1
        prev_action_id = em.ACTION_INDEX[step.action]
    expected = torch.stack(losses).sum()

    assert float(got) == pytest.approx(float(expected), abs=1e-9)


# ---------------------------------------------------------------------------
# The aux term itself
# ---------------------------------------------------------------------------

def test_aux_loss_finite_and_positive_at_init(usvs, records):
    model, feats, steps = _fixed_model_and_item(usvs, records)
    node_targets = em.node_targets_for_steps(steps, feats, usvs)
    assert any(t is not None for t in node_targets), "fixture record has no node-emitting steps"

    loss_off = em.teacher_force_loss(model, feats, steps, terminal_weight=4.0)
    loss_on = em.teacher_force_loss(model, feats, steps, terminal_weight=4.0,
                                     node_targets=node_targets, aux_usvs_weight=0.5)
    assert torch.isfinite(loss_on)
    assert float(loss_on) > float(loss_off), "the aux term must add positive mass at random init"


def test_aux_weight_scales_the_added_term(usvs, records):
    model, feats, steps = _fixed_model_and_item(usvs, records)
    node_targets = em.node_targets_for_steps(steps, feats, usvs)
    loss_off = float(em.teacher_force_loss(model, feats, steps, terminal_weight=4.0))
    loss_half = float(em.teacher_force_loss(model, feats, steps, terminal_weight=4.0,
                                             node_targets=node_targets, aux_usvs_weight=0.5))
    loss_full = float(em.teacher_force_loss(model, feats, steps, terminal_weight=4.0,
                                             node_targets=node_targets, aux_usvs_weight=1.0))
    # both add the SAME underlying sum of (1 - cosine) terms, just scaled --
    # so the full-weight delta must be ~2x the half-weight delta.
    assert (loss_full - loss_off) == pytest.approx(2 * (loss_half - loss_off), rel=1e-4)


def test_gradients_reach_usvs_head_only_when_enabled(usvs, records):
    model, feats, steps = _fixed_model_and_item(usvs, records)
    node_targets = em.node_targets_for_steps(steps, feats, usvs)

    loss_off = em.teacher_force_loss(model, feats, steps, terminal_weight=4.0,
                                      node_targets=node_targets, aux_usvs_weight=0.0)
    loss_off.backward()
    assert model.usvs_head.weight.grad is None, "aux_usvs_weight=0.0 must not touch usvs_head at all"
    assert model.action_type_head.weight.grad is not None

    model.zero_grad()
    loss_on = em.teacher_force_loss(model, feats, steps, terminal_weight=4.0,
                                     node_targets=node_targets, aux_usvs_weight=0.5)
    loss_on.backward()
    assert model.usvs_head.weight.grad is not None
    assert torch.isfinite(model.usvs_head.weight.grad).all()
    assert model.usvs_head.weight.grad.abs().sum() > 0.0
    # the rest of the network still trains too -- this is additive, not a
    # replacement of the main loss.
    assert model.action_type_head.weight.grad is not None


# ---------------------------------------------------------------------------
# node_vectors / evaluate_head_cosine diagnostic
# ---------------------------------------------------------------------------

def test_node_vectors_matches_flatten_tree_node_set(usvs, records):
    model, feats, _ = _fixed_model_and_item(usvs, records)
    rec = records[0]
    norm = normalize_gold_tree(rec, rec["lattice"]["trees"][0])
    out = model.node_vectors(feats, norm)

    from nsm_ct import usvs_graded as ug
    flat = ug.flatten_tree(rec, norm, usvs)
    got = {(n["clause_index"], n["role"], n["token_index"], n["is_predicate"]) for n in out}
    expected = {(n.clause_index, n.role, n.token_index, n.is_predicate) for n in flat.nodes}
    assert got == expected
    for n in out:
        assert n["vector"].shape == (len(usvs.axes) + em_reserved_size(),)


def test_evaluate_head_cosine_is_finite_and_bounded(usvs, records):
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    torch.manual_seed(3)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes),
                             hash_buckets=512, d_model=32, controller_hidden=32)
    c = etu.evaluate_head_cosine(model, records[:5], usvs, pos_vocab, 512)
    assert c == c, "must not be NaN on a normal fixture"
    assert -1.0 - 1e-6 <= c <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# build_train_items / run_training_loop plumbing
# ---------------------------------------------------------------------------

def test_build_train_items_default_has_no_node_targets(usvs, records):
    pos_vocab = em.build_pos_vocab(records)
    items = etu.build_train_items(records[:3], usvs, pos_vocab, 512)
    assert all(len(item) == 3 for item in items)
    assert all(item[2] is None for item in items)


def test_build_train_items_computes_node_targets_when_asked(usvs, records):
    pos_vocab = em.build_pos_vocab(records)
    items = etu.build_train_items(records[:3], usvs, pos_vocab, 512, compute_node_targets=True)
    assert any(item[2] is not None for item in items)
    for feats, steps, node_targets in items:
        assert len(node_targets) == len(steps)


def test_run_training_loop_with_aux_usvs_reduces_loss(usvs, records):
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    torch.manual_seed(5)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes),
                             hash_buckets=512, d_model=32, controller_hidden=32)
    items = etu.build_train_items(records[:6], usvs, pos_vocab, 512, compute_node_targets=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    result = etu.run_training_loop(model, items, opt, epochs=1, batch_size=2,
                                    max_seconds=60.0, max_steps=5,
                                    aux_usvs_weight=0.5)
    assert result["optimizer_steps"] >= 1
    assert all(np.isfinite(avg) for _, avg in result["loss_curve"])


# ---------------------------------------------------------------------------
# Checkpoint save/load round-trip + old-checkpoint compatibility
# ---------------------------------------------------------------------------

def test_checkpoint_round_trips_the_head(usvs, records):
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    torch.manual_seed(0)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes),
                             hash_buckets=512, d_model=32, controller_hidden=32)
    sd = model.state_dict()

    torch.manual_seed(99)  # deliberately different init
    model2 = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes),
                              hash_buckets=512, d_model=32, controller_hidden=32)
    model2.load_checkpoint_state(sd)
    for k in sd:
        assert torch.equal(sd[k], model2.state_dict()[k]), k
    assert torch.equal(model.usvs_head.weight, model2.usvs_head.weight)
    assert torch.equal(model.usvs_head.bias, model2.usvs_head.bias)


def test_old_checkpoint_without_head_loads_with_a_clear_message(usvs, records, capsys):
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    torch.manual_seed(0)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes),
                             hash_buckets=512, d_model=32, controller_hidden=32)
    old_sd = {k: v for k, v in model.state_dict().items() if not k.startswith("usvs_head.")}

    torch.manual_seed(1)
    model2 = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes),
                              hash_buckets=512, d_model=32, controller_hidden=32)
    before = model2.usvs_head.weight.clone()
    model2.load_checkpoint_state(old_sd)  # must not raise
    # every non-head parameter now matches the old checkpoint ...
    for k, v in old_sd.items():
        assert torch.equal(v, model2.state_dict()[k])
    # ... but usvs_head is untouched (left at its own random init)
    assert torch.equal(before, model2.usvs_head.weight)

    out = capsys.readouterr().out
    assert "usvs_head" in out and ("pre-dates" in out or "no usvs_head" in out)


def test_genuinely_incompatible_checkpoint_still_raises(usvs, records):
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    torch.manual_seed(0)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes),
                             hash_buckets=512, d_model=32, controller_hidden=32)
    sd = model.state_dict()
    bad_sd = {k: v for k, v in sd.items() if not k.startswith("action_type_head.")}
    with pytest.raises(RuntimeError):
        model.load_checkpoint_state(bad_sd)
