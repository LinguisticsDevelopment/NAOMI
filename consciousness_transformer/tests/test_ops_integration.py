"""M60 (op-library integration): RECENCY (a deterministic per-candidate
salience feature for the resolver) and CLEANUP (codebook cleanup on the
answer path with a caution margin) wired into Track A's reactor --
dev/OP_LIBRARY_MAP.md's ``recency``/``cleanup`` rows, closing two gaps
RESEARCH_NOTES "M57 battery #3" recorded: recency-only referent cases
(the ``ambiguous_name`` instance/rich device) sit at CHANCE because
nothing in the register carries discourse-order salience at all (see
tests/test_evidence_interaction.py's ``_train_short`` docstring, which
excludes this exact device from ITS OWN interaction-feature test for this
reason), and ``caution``/margin never gates anything (dev/OP_INVENTORY.md
Sec.5).

Covers:
  1. Byte-identity when ``ClauseBatch.cand_recency`` is absent/``None`` and
     ``ClauseReactor.cleanup`` is off (the default) -- mirrors
     tests/test_evidence_interaction.py's own byte-identity test structure.
  2. ``nsm_ct.ops.recency`` mechanics on a hand-built 3-candidate mention
     log (most-recent flagged, steps_since monotone).
  3. ``build_clause_batch`` actually populates ``cand_recency`` for an
     ``ambiguous_name`` instance episode, with the referent flagged
     ``is_most_recent``.
  4. THE LOAD-BEARING TEST: short training, with vs without the recency
     columns, restricted to the ``ambiguous_name`` device (mirrors
     tests/test_evidence_interaction.py's own load-bearing A/B structure,
     just the opposite device selection).
  5. CLEANUP mechanics: predictions identical with the flag on/off, abstain
     flags set exactly at the ``CAUTION`` margin, on a hand-built
     deterministic response.
  6. Inverse-read routing: ``inverse_direct_logits`` picks the right answer
     on a hand-built batch where the direct entity-axis route is correct.

No parser dependency anywhere in this file -- mirrors
tests/test_instance_curriculum.py's / tests/test_evidence_interaction.py's
own isolation discipline.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from nsm_ct import ops  # noqa: E402
from nsm_ct.clause_reactor import ClauseBatch, ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import generate_instance_episodes  # noqa: E402
from nsm_ct.episode import split_episodes  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.resolver import make_resolver  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

from _train_common import epoch_minibatches  # noqa: E402

DIM = 24


def _meaning():
    return NSMMeaningResolver()


def _codec(dim=DIM):
    return TPRCodec(dim=dim)


def _force_full_write_gate(model: ClauseReactor):
    """Force the write gate to (near) 1 and disable negation -- mirrors
    tests/test_instance_curriculum.py's own helper of the same name, so a
    hand-built write step actually lands with full strength and a
    hand-built read is deterministic."""
    with torch.no_grad():
        model.write_gate.weight.zero_(); model.write_gate.bias.fill_(10.0)
        model.overwrite_gate.weight.zero_(); model.overwrite_gate.bias.fill_(10.0)
        model.decide_truth.weight.zero_(); model.decide_truth.bias.fill_(-10.0)


# ---------------------------------------------------------------------------
# 1. Byte-identity: cand_recency absent/None, cleanup off.
# ---------------------------------------------------------------------------
def test_cand_recency_is_none_for_every_pre_m60_batch():
    """A batch built from plain old-curriculum episodes (no candidate sets
    at all) must leave cand_recency None -- the same "only if present"
    discipline every optional ClauseBatch field establishes."""
    from nsm_ct.episode import CurriculumGenerator

    meaning = _meaning()
    codec = _codec()
    old_eps = CurriculumGenerator(max_level=6, seed=0).generate(10)
    batch_old = build_clause_batch(old_eps, None, meaning, codec)
    assert batch_old.cand_recency is None


def test_forward_output_byte_identical_with_and_without_cand_recency_field_when_none():
    """A hand-built batch with cand_recency explicitly absent (dataclass
    default) vs explicitly None must produce IDENTICAL model output --
    mirrors tests/test_evidence_interaction.py's own
    ``test_forward_output_byte_identical_with_and_without_evidence_target_field_when_none``.
    Also checks ``self.cleanup=False`` (the default) leaves no cleanup_*
    key in the output at all."""
    torch.manual_seed(0)
    b, d, C = 5, 16, 3
    g = torch.Generator().manual_seed(1)
    entity = F.normalize(torch.randn(b, 2, d, generator=g), dim=-1)
    relation = F.normalize(torch.randn(b, 2, d, generator=g), dim=-1)
    value = F.normalize(torch.randn(b, 2, d, generator=g), dim=-1)
    pred = F.normalize(torch.randn(b, 2, d, generator=g), dim=-1)
    is_q = torch.tensor([[0.0, 1.0]] * b)
    mask = torch.ones(b, 2)
    options = F.normalize(torch.randn(b, 4, d, generator=g), dim=-1)
    answer = torch.randint(0, 4, (b,), generator=g)
    cand_entity = F.normalize(torch.randn(b, 2, C, d, generator=g), dim=-1)
    cand_mask = torch.ones(b, 2, C)
    cand_prior = torch.full((b, 2, C), 1.0 / C)
    cand_feature = torch.zeros(b, 2, 6)
    cand_gold = torch.full((b, 2), -1, dtype=torch.long)

    kwargs = dict(entity=entity, relation=relation, value=value, pred=pred, is_q=is_q,
                  mask=mask, options=options, answer=answer, cand_entity=cand_entity,
                  cand_mask=cand_mask, cand_prior=cand_prior, cand_feature=cand_feature,
                  cand_gold=cand_gold)

    for resolver in (make_resolver("A", d, 8),
                     make_resolver("A", d, 8, use_cand_feature=True, cand_feature_extra=3)):
        model = ClauseReactor(dim=d, hidden=8, resolver=resolver, cleanup=False)
        model.eval()

        batch_absent = ClauseBatch(**kwargs)
        assert batch_absent.cand_recency is None
        with torch.no_grad():
            out_absent = model(batch_absent)
        assert "cleanup_index" not in out_absent
        assert "inverse_direct_logits" not in out_absent

        batch_none = ClauseBatch(**kwargs, cand_recency=None)
        with torch.no_grad():
            out_none = model(batch_none)

        for k in out_absent:
            assert torch.equal(out_absent[k], out_none[k]), k


# ---------------------------------------------------------------------------
# 2. ops.recency mechanics on a hand-built 3-candidate mention log.
# ---------------------------------------------------------------------------
def test_recency_mechanics_hand_built_three_candidates():
    """Candidate 0 was last mentioned 5 steps ago, candidate 1 was NEVER
    mentioned, candidate 2 was mentioned most recently (1 step ago) --
    steps_since must be monotone (never < recent), is_most_recent must
    flag exactly candidate 2, and the never-mentioned candidate must get
    the RECENCY_NEVER sentinel, not a negative/garbage value."""
    mention_steps = torch.tensor([[5.0, -1.0, 9.0]])   # [1, 3], current_step=10
    mention_counts = torch.tensor([[2.0, 0.0, 4.0]])
    feats = ops.recency(mention_steps, current_step=10, mention_counts=mention_counts)

    assert feats.steps_since.shape == (1, 3)
    assert feats.steps_since[0, 2].item() < feats.steps_since[0, 0].item()   # more recent -> smaller steps_since
    assert feats.steps_since[0, 1].item() == ops.RECENCY_NEVER               # never mentioned -> sentinel
    assert feats.is_most_recent.tolist() == [[False, False, True]]
    assert feats.log_count[0, 1].item() == 0.0
    assert feats.log_count[0, 2].item() > feats.log_count[0, 0].item()       # monotone in count


# ---------------------------------------------------------------------------
# 3. build_clause_batch actually populates cand_recency for ambiguous_name.
# ---------------------------------------------------------------------------
def test_instance_batch_populates_cand_recency_and_flags_the_referent_most_recent():
    """For the ``ambiguous_name`` device, the referent's OWN baseline trait
    statement is placed LAST (see _instance_steps's own docstring
    paragraph) -- so at the overwrite step, the gold candidate must be the
    one flagged ``is_most_recent``."""
    meaning = _meaning()
    codec = _codec()
    eps = generate_instance_episodes(60, seed=0, inverse_frac=0.0)
    eps = [e for e in eps if e.meta.get("referring_device") == "ambiguous_name"]
    assert len(eps) >= 5, "expected enough ambiguous_name episodes at this seed"
    batch = build_clause_batch(eps, None, meaning, codec)
    assert batch.cand_recency is not None

    # The overwrite step for each row is the earliest gold-bearing step
    # with exactly 2 real candidates (ambiguous_name's cand_roles = the
    # name-matched pair only -- see _instance_steps's docstring).
    has_cand = batch.cand_gold >= 0
    two_cand = batch.cand_mask.sum(-1) == 2
    mask = has_cand & two_cand
    assert bool(mask.any())
    gold = batch.cand_gold[mask]
    is_most_recent = batch.cand_recency[..., 2][mask]        # [-1] would also work; keep explicit
    picked = is_most_recent.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    assert bool((picked > 0).all())


# ---------------------------------------------------------------------------
# 4. THE LOAD-BEARING TEST: short training, with vs without the recency
#    columns, restricted to the ambiguous_name device (dim=24, ~260
#    instance episodes, ~150 steps, batch 32). Mirrors
#    tests/test_evidence_interaction.py's own ``_train_short`` structure.
# ---------------------------------------------------------------------------
def _binding_accuracy_two_candidate(model, batch):
    """Same shape as tests/test_evidence_interaction.py's
    ``_binding_accuracy``, but restricted to gold-bearing steps with
    EXACTLY 2 real candidates -- the ambiguous_name overwrite step's own
    signature (cand_roles = the name-matched pair only), which isolates it
    from the SAME batch's definite_description-style question steps
    (3 candidates, decided by attr:kind, unrelated to recency) without
    having to filter episodes down to a tiny subset."""
    model.eval()
    with torch.no_grad():
        out = model(batch)
    if "resolver_logits" not in out:
        return None
    has_cand = batch.cand_gold >= 0
    two_cand = batch.cand_mask.sum(-1) == 2
    mask = has_cand & two_cand
    pred_idx = out["resolver_logits"].argmax(-1)
    correct = (pred_idx == batch.cand_gold) & mask
    total = int(mask.sum())
    if total == 0:
        return None
    return float(correct.sum()) / total


def _train_short_recency(cand_feature_extra: int, dim: int, seed: int) -> float:
    """``cand_feature_extra=1`` truncates CorefHead's widened register down
    to JUST the evidence*target interaction scalar (uninformative for this
    device by design, see this module's own docstring) -- the "without
    recency" arm. ``cand_feature_extra=4`` keeps the interaction scalar
    PLUS all three recency columns (extra_cols order:
    [evidence_interaction, cand_recency] -- see
    ClauseReactor._collapse's own extra_cols block) -- the "with recency"
    arm. Both otherwise IDENTICAL setup/batch/seed, mirroring
    tests/test_evidence_interaction.py's own ``_train_short``."""
    meaning = _meaning()
    codec = TPRCodec(dim=dim)
    episodes = generate_instance_episodes(260, seed=0, inverse_frac=0.0)
    episodes = [e for e in episodes if e.meta.get("referring_device") == "ambiguous_name"]
    tr_eps, va_eps = split_episodes(episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, None, meaning, codec)
    va = build_clause_batch(va_eps, None, meaning, codec)
    assert tr.cand_recency is not None and va.cand_recency is not None

    torch.manual_seed(seed)
    resolver = make_resolver("A", dim, 64, use_cand_feature=True,
                              cand_feature_extra=cand_feature_extra)
    model = ClauseReactor(dim=dim, hidden=64, resolver=resolver)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    n_tr = len(tr_eps)
    batch_size = 32
    n_steps = 0
    model.train()
    epoch = 0
    while n_steps < 150:
        for mb_idx in epoch_minibatches(n_tr, batch_size, seed, epoch):
            idx_t = torch.from_numpy(mb_idx)
            sub = tr.subset(idx_t)
            out = model(sub)
            cg = sub.cand_gold
            has_cand = cg >= 0
            if not bool(has_cand.any()):
                continue
            loss = F.cross_entropy(out["resolver_logits"][has_cand], cg[has_cand])
            opt.zero_grad(); loss.backward(); opt.step()
            n_steps += 1
            if n_steps >= 150:
                break
        epoch += 1

    return _binding_accuracy_two_candidate(model, va)


def test_short_training_recency_lifts_ambiguous_name_binding_above_chance():
    """THE load-bearing test (director-locked design item 4): with the
    recency columns wired into the register (cand_feature_extra=4),
    ambiguous_name binding accuracy must exceed 0.75 in a short training
    run; with them absent (cand_feature_extra=1, otherwise IDENTICAL
    setup/batch/seed), it must stay at/near chance (<= 0.6) -- this is
    exactly the RESEARCH_NOTES "M57 battery #3" gap (recency-only referent
    cases at chance), closed."""
    # extra_cols order is [evidence_interaction (1), cand_recency (3)] -- see
    # ClauseReactor._collapse's own extra_cols block -- so extra_width=4
    # keeps all four, extra_width=1 truncates down to the interaction
    # scalar alone.
    acc_with = _train_short_recency(cand_feature_extra=4, dim=24, seed=0)
    acc_without = _train_short_recency(cand_feature_extra=1, dim=24, seed=0)
    assert acc_with is not None and acc_without is not None
    assert acc_with > 0.75, f"acc_with={acc_with}"
    assert acc_without <= 0.6, f"acc_without={acc_without}"
    assert acc_with - acc_without > 0.15


# ---------------------------------------------------------------------------
# 5. CLEANUP mechanics: predictions identical on/off, abstain at CAUTION.
# ---------------------------------------------------------------------------
def _cleanup_batch_and_model(d=8):
    """Two rows, one question step each, with the response head PATCHED to
    emit a KNOWN vector (e0) regardless of input -- T=1 makes the respond
    softmax trivially 1.0, so ``r`` (the aggregated response) equals the
    response head's own bias exactly, and options are hand-designed so
    each row's top1-top2 COSINE gap is exactly known: row 0 confident
    (0.5 gap, >= CAUTION), row 1 borderline-abstain (0.05 gap, <
    CAUTION)."""
    torch.manual_seed(0)
    e0 = torch.zeros(d); e0[0] = 1.0
    e1 = torch.zeros(d); e1[1] = 1.0

    def _at_cos(cos_theta: float) -> torch.Tensor:
        s = (1.0 - cos_theta ** 2) ** 0.5
        return cos_theta * e0 + s * e1

    # Row 0: top1=e0 (cos=1.0), top2 at cos=0.5 -> margin=0.5 (confident).
    opts0 = torch.stack([e0, _at_cos(0.5), _at_cos(0.0)])
    # Row 1: top1=e0 (cos=1.0), top2 at cos=0.95 -> margin=0.05 (abstain,
    # CAUTION default 0.1).
    opts1 = torch.stack([e0, _at_cos(0.95), _at_cos(0.0)])
    options = torch.stack([opts0, opts1])                        # [2, 3, d]

    b = 2
    entity = torch.zeros(b, 1, d)
    relation = torch.zeros(b, 1, d)
    value = torch.zeros(b, 1, d)
    pred = torch.zeros(b, 1, d)
    is_q = torch.ones(b, 1)
    mask = torch.ones(b, 1)
    answer = torch.tensor([0, 0])

    model = ClauseReactor(dim=d, hidden=4, resolver=None, cleanup=False)
    with torch.no_grad():
        model.response.weight.zero_()
        model.response.bias.copy_(e0)

    batch = ClauseBatch(entity=entity, relation=relation, value=value, pred=pred,
                         is_q=is_q, mask=mask, options=options, answer=answer)
    return model, batch


def test_cleanup_predictions_identical_and_abstain_at_caution_margin():
    model, batch = _cleanup_batch_and_model()

    model.eval()
    assert model.cleanup is False
    with torch.no_grad():
        out_off = model(batch)
    assert "cleanup_index" not in out_off

    model.cleanup = True
    with torch.no_grad():
        out_on = model(batch)
    pred = out_on["answer_logits"].argmax(-1)
    assert torch.equal(out_on["cleanup_index"], pred), \
        "cleanup must never change the argmax prediction"
    # only the abstain/margin annotation differs -- every other key is
    # byte-identical on vs off.
    for k in out_off:
        assert torch.equal(out_off[k], out_on[k]), k

    assert out_on["cleanup_abstain"].tolist() == [False, True]
    assert float(out_on["cleanup_margin"][0]) > ops.CAUTION
    assert float(out_on["cleanup_margin"][1]) < ops.CAUTION
    assert abs(float(out_on["cleanup_margin"][0]) - 0.5) < 1e-3
    assert abs(float(out_on["cleanup_margin"][1]) - 0.05) < 1e-3


def test_cleanup_is_eval_only_no_op_during_training():
    """``self.cleanup=True`` must be a NO-OP while ``model.training`` is
    True -- "at eval only" read literally (LOCKED DESIGN item 2)."""
    model, batch = _cleanup_batch_and_model()
    model.cleanup = True
    model.train()
    out = model(batch)
    assert "cleanup_index" not in out


# ---------------------------------------------------------------------------
# 6. Inverse-read routing: direct similarity route on a hand-built batch.
# ---------------------------------------------------------------------------
def test_inverse_direct_route_picks_the_correct_option():
    """Step 0 writes (ent0, rel0) <- val0 (write gate forced to 1); step 1
    is an inverse-query step (``inverse_mask=1``) reading THAT SAME
    (rel0, val0) pair on the entity axis -- the direct route
    (similarity(query_entity readout, option atoms)) must recover ent0 as
    the top-scoring option, with NO resolver installed at all (this route
    never touches the resolver/candidate machinery)."""
    torch.manual_seed(0)
    d = 16
    g = torch.Generator().manual_seed(3)
    ent0 = F.normalize(torch.randn(1, d, generator=g), dim=-1)
    other = F.normalize(torch.randn(1, d, generator=g), dim=-1)
    rel0 = F.normalize(torch.randn(1, d, generator=g), dim=-1)
    val0 = F.normalize(torch.randn(1, d, generator=g), dim=-1)
    who_marker = F.normalize(torch.randn(1, d, generator=g), dim=-1)
    z = torch.zeros(1, d)

    entity = torch.stack([ent0, who_marker], dim=1)      # [1, 2, d]
    relation = torch.stack([rel0, rel0], dim=1)
    value = torch.stack([val0, val0], dim=1)
    pred = torch.zeros(1, 2, d)
    is_q = torch.tensor([[0.0, 1.0]])
    mask = torch.ones(1, 2)
    options = torch.stack([ent0, other], dim=1)           # [1, 2, d]
    answer = torch.tensor([0])
    inverse_mask = torch.tensor([[0.0, 1.0]])

    model = ClauseReactor(dim=d, hidden=16, resolver=None)
    _force_full_write_gate(model)
    batch = ClauseBatch(entity=entity, relation=relation, value=value, pred=pred,
                         is_q=is_q, mask=mask, options=options, answer=answer,
                         inverse_mask=inverse_mask)
    with torch.no_grad():
        out = model(batch)
    assert "inverse_direct_logits" in out
    pred_idx = out["inverse_direct_logits"].argmax(-1)
    assert int(pred_idx[0]) == 0
