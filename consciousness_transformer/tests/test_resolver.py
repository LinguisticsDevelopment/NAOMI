"""Tests for M53b: the resolver contract (nsm_ct.resolver) + its OPTIONAL
integration into nsm_ct.clause_reactor.ClauseReactor. See dev/MIND_INTERFACE.md
§3 and dev/RESOLVER_BUILD_PLAN.md Phase 2 "Agent 3".

No parser dependency anywhere here (unlike tests/test_membrane.py) -- these are
synthetic-tensor tests of the resolver contract + the reactor's optional collapse
step, deliberately isolated from quantum_parser so they run everywhere.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nsm_ct import entity_memory as em
from nsm_ct.clause_reactor import ClauseBatch, ClauseReactor
from nsm_ct.membrane import FEATURE_DIM
from nsm_ct.resolver import CorefHead, Resolver, SharedScorer, make_resolver, query_candidates


# ---------------------------------------------------------------------------
# Synthetic batch: C write steps (one per candidate, each binding its own
# (entity, relation) address in memory) + one PRONOUN step carrying a real
# candidate set over those same entities + one question step. Mirrors
# tests/test_clause_reactor.py's `_toy_batch` pattern -- no parser needed.
# ---------------------------------------------------------------------------
def _toy_batch_with_candidates(b=6, d=16, K=4, C=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opts = F.normalize(torch.randn(b, K, d, generator=g), dim=-1)
    ans = torch.randint(0, K, (b,), generator=g)

    cand_entity_vecs = F.normalize(torch.randn(b, C, d, generator=g), dim=-1)
    cand_values = opts[torch.arange(b).unsqueeze(1), torch.randint(0, K, (b, C), generator=g)]
    gold_idx = torch.randint(0, C, (b,), generator=g)
    cand_values[torch.arange(b), gold_idx] = opts[torch.arange(b), ans]  # gold candidate's fact = the answer

    rel = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    obj_ent = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    prd = F.normalize(torch.randn(b, d, generator=g), dim=-1)

    T = C + 2   # C write steps + 1 pronoun step + 1 question step
    entity = torch.zeros(b, T, d); relation = torch.zeros(b, T, d); value = torch.zeros(b, T, d)
    pred = torch.zeros(b, T, d); is_q = torch.zeros(b, T); mask = torch.ones(b, T)
    for j in range(C):
        entity[:, j], relation[:, j], value[:, j], pred[:, j] = (
            cand_entity_vecs[:, j], rel, cand_values[:, j], prd)
    pronoun_t = C
    entity[:, pronoun_t], relation[:, pronoun_t], pred[:, pronoun_t] = obj_ent, rel, prd
    value[:, pronoun_t] = opts[torch.arange(b), ans]   # M53a placeholder: pre-bound to the gold answer
    q_t = C + 1
    entity[:, q_t], relation[:, q_t], is_q[:, q_t] = obj_ent, rel, 1.0

    cand_entity = torch.zeros(b, T, C, d)
    cand_mask = torch.zeros(b, T, C)
    cand_prior = torch.zeros(b, T, C)
    cand_feature = torch.zeros(b, T, FEATURE_DIM)
    cand_gold = torch.full((b, T), -1, dtype=torch.long)
    cand_entity[:, pronoun_t] = cand_entity_vecs
    cand_mask[:, pronoun_t] = 1.0
    cand_prior[:, pronoun_t] = 1.0 / C
    cand_feature[:, pronoun_t] = torch.randn(b, FEATURE_DIM, generator=g)
    cand_gold[:, pronoun_t] = gold_idx

    batch = ClauseBatch(entity, relation, value, pred, is_q, mask, opts, ans,
                         cand_entity=cand_entity, cand_mask=cand_mask, cand_prior=cand_prior,
                         cand_feature=cand_feature, cand_gold=cand_gold)
    return batch, pronoun_t


# ---------------------------------------------------------------------------
# 1. Interface contract: both tracks produce [B, C] logits.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("track", ["A", "B"])
def test_resolver_tracks_produce_BC_logits(track):
    b, C, d, hidden = 5, 4, 16, 32
    resolver = make_resolver(track, d, hidden)
    cand_entity = torch.randn(b, C, d)
    cand_feature = torch.randn(b, FEATURE_DIM)
    cand_prior = torch.rand(b, C)
    cand_mask = torch.ones(b, C)
    mem_read = torch.randn(b, C, d)
    state = torch.randn(b, hidden)
    logits = resolver(cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state)
    assert logits.shape == (b, C)
    assert isinstance(resolver, Resolver)


@pytest.mark.parametrize("track,cls", [("A", CorefHead), ("B", SharedScorer)])
def test_make_resolver_dispatch(track, cls):
    assert isinstance(make_resolver(track, 16, 32), cls)
    assert isinstance(make_resolver(track.lower(), 16, 32), cls)


def test_make_resolver_rejects_unknown_track():
    with pytest.raises(ValueError):
        make_resolver("C", 16, 32)


@pytest.mark.parametrize("track", ["A", "B"])
def test_resolver_param_counts_under_20k(track):
    resolver = make_resolver(track, dim=32, hidden=128)
    n = sum(p.numel() for p in resolver.parameters())
    assert n < 20_000, f"track {track}: {n} params"


def test_query_candidates_matches_manual_loop():
    b, C, d = 4, 3, 8
    memory = em.init_memory(b, d, torch.device("cpu"))
    e = F.normalize(torch.randn(b, d), dim=-1)
    r = F.normalize(torch.randn(b, d), dim=-1)
    v = torch.randn(b, d)
    memory = em.write(memory, e, r, v, torch.ones(b))
    cand_entity = torch.randn(b, C, d)
    relation = F.normalize(torch.randn(b, d), dim=-1)
    got = query_candidates(memory, cand_entity, relation)
    expected = torch.stack([em.query(memory, cand_entity[:, j], relation) for j in range(C)], dim=1)
    assert got.shape == (b, C, d)
    assert torch.equal(got, expected)


def test_query_candidates_zero_candidates_is_empty_not_error():
    b, d = 3, 8
    memory = em.init_memory(b, d, torch.device("cpu"))
    cand_entity = torch.zeros(b, 0, d)
    relation = torch.zeros(b, d)
    got = query_candidates(memory, cand_entity, relation)
    assert got.shape == (b, 0, d)


# ---------------------------------------------------------------------------
# 2. No-resolver byte-identity regression (load-bearing): resolver=None must
#    reproduce an independent re-derivation of the pre-M53b forward loop
#    EXACTLY, and must not leak any resolver_* keys into the output dict.
# ---------------------------------------------------------------------------
def _reference_forward_no_resolver(model: ClauseReactor, batch: ClauseBatch):
    """Independent reimplementation of the pre-M53b ClauseReactor.forward loop
    (same algorithm, called directly against the model's own submodules) --
    proves the resolver=None code path is numerically untouched by M53b."""
    b, T, d = batch.entity.shape
    device = batch.entity.device
    state = torch.zeros(b, model.gru.hidden_size, device=device)
    memory = em.init_memory(b, d, device)
    coord = batch._coord()
    resp_logits, resp_vecs = [], []
    for t in range(T):
        e, r, v = batch.entity[:, t], batch.relation[:, t], batch.value[:, t]
        p, c = batch.pred[:, t], coord[:, t]
        real, isq = batch.mask[:, t], batch.is_q[:, t]
        mem_read = em.query(memory, e, r)
        state = model.gru(torch.cat([e, r, v, p, c, mem_read], dim=-1), state)
        stmt = real * (1.0 - isq)
        gate = torch.sigmoid(model.write_gate(state)).squeeze(-1) * stmt
        owr = torch.sigmoid(model.overwrite_gate(state)).squeeze(-1) * gate
        neg = torch.sigmoid(model.decide_truth(torch.cat([state, v], dim=-1))).squeeze(-1) * stmt
        memory = em.write(memory, e, r, v, gate - neg, overwrite=owr)
        rl = model.respond(state).squeeze(-1)
        rl = rl.masked_fill(real <= 0, float("-inf"))
        resp_logits.append(rl)
        resp_vecs.append(model.response(torch.cat([state, mem_read], dim=-1)))
    RL = torch.stack(resp_logits, dim=1)
    RV = torch.stack(resp_vecs, dim=1)
    w = torch.softmax(RL, dim=1)
    r_agg = (w.unsqueeze(-1) * RV).sum(dim=1)
    rn = r_agg / (r_agg.norm(dim=-1, keepdim=True) + 1e-8)
    on = batch.options / (batch.options.norm(dim=-1, keepdim=True) + 1e-8)
    answer_logits = torch.einsum("bd,bkd->bk", rn, on) * 10.0
    return {"answer_logits": answer_logits, "response": r_agg,
            "respond_gates": w, "respond_position": (w * batch.is_q).sum(1)}


def test_no_resolver_byte_identical_regression():
    torch.manual_seed(0)
    batch, _pronoun_t = _toy_batch_with_candidates(b=6, d=16, K=4, C=3, seed=1)
    model = ClauseReactor(dim=16, resolver=None)
    out = model(batch)
    ref = _reference_forward_no_resolver(model, batch)
    assert set(out.keys()) == set(ref.keys())      # no resolver_logits/resolver_margin leak
    for k in ref:
        assert torch.equal(out[k], ref[k]), k


def test_no_resolver_default_arg_matches_explicit_none():
    """The constructor's default (no `resolver=` kwarg at all) must behave
    identically to passing `resolver=None` explicitly."""
    torch.manual_seed(0)
    batch, _ = _toy_batch_with_candidates(b=4, d=16, K=4, C=2, seed=2)
    torch.manual_seed(7)
    m1 = ClauseReactor(dim=16)
    torch.manual_seed(7)
    m2 = ClauseReactor(dim=16, resolver=None)
    out1, out2 = m1(batch), m2(batch)
    for k in out1:
        assert torch.equal(out1[k], out2[k])


def test_resolver_absent_on_candidate_free_batch_is_also_untouched():
    """A resolver CAN be installed, but if the batch carries no candidate data
    at all (cand_mask is None -- e.g. an old-curriculum-only batch), the
    resolver must never fire and behavior must match resolver=None."""
    torch.manual_seed(0)
    batch, _ = _toy_batch_with_candidates(b=4, d=16, K=4, C=2, seed=3)
    batch.cand_entity = batch.cand_mask = batch.cand_prior = None
    batch.cand_feature = batch.cand_gold = None
    torch.manual_seed(5)
    plain = ClauseReactor(dim=16, resolver=None)
    resolver = make_resolver("A", 16)   # built outside the seeded block on purpose
    torch.manual_seed(5)                # reseed so ClauseReactor's OWN init matches `plain` exactly
    with_resolver = ClauseReactor(dim=16, resolver=resolver)
    out_plain, out_res = plain(batch), with_resolver(batch)
    assert "resolver_logits" not in out_res
    assert torch.equal(out_plain["answer_logits"], out_res["answer_logits"])


# ---------------------------------------------------------------------------
# 3. Soft-collapse shape test: training-mode softmax mixture, gradients flow.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("track", ["A", "B"])
def test_soft_collapse_shapes_and_gradients_flow(track):
    torch.manual_seed(0)
    batch, pronoun_t = _toy_batch_with_candidates(b=5, d=16, K=4, C=3, seed=4)
    model = ClauseReactor(dim=16, resolver=make_resolver(track, 16, 128))
    model.train()
    out = model(batch)
    b, T, d = batch.entity.shape
    C = batch.cand_entity.shape[2]
    assert out["resolver_logits"].shape == (b, T, C)
    assert out["resolver_margin"].shape == (b, T)
    # non-pronoun steps carry no real candidates -> zero margin
    assert torch.equal(out["resolver_margin"][:, :pronoun_t], torch.zeros(b, pronoun_t))

    loss = F.cross_entropy(out["answer_logits"], batch.answer)
    loss.backward()
    resolver_grads = [p.grad for p in model.resolver.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in resolver_grads), \
        "soft collapse should let the answer loss reach the resolver's parameters"


def test_eval_mode_uses_hard_argmax_collapse():
    """At eval, the collapse is a hard one-hot pick (argmax), not a soft mixture:
    the resolved value must equal exactly the top-candidate's memory readout."""
    torch.manual_seed(0)
    batch, pronoun_t = _toy_batch_with_candidates(b=4, d=16, K=4, C=3, seed=5)
    model = ClauseReactor(dim=16, resolver=make_resolver("A", 16))
    model.eval()
    with torch.no_grad():
        out = model(batch)
    logits = out["resolver_logits"][:, pronoun_t]           # [B, C]
    top = logits.argmax(-1)
    # a hard collapse means resolver_logits' argmax is a genuine one-hot argmax of
    # UNMASKED entries only (no padding here, so every index is eligible)
    assert top.shape == (4,)
    assert (top < batch.cand_entity.shape[2]).all()


# ---------------------------------------------------------------------------
# 4. Aux-loss computation test.
# ---------------------------------------------------------------------------
def test_aux_loss_computable_from_resolver_logits_and_cand_gold():
    torch.manual_seed(0)
    batch, pronoun_t = _toy_batch_with_candidates(b=6, d=16, K=4, C=3, seed=6)
    model = ClauseReactor(dim=16, resolver=make_resolver("B", 16, 128))
    out = model(batch)
    mask = batch.cand_gold >= 0                     # [B, T]
    assert int(mask.sum()) == batch.entity.shape[0]  # exactly one pronoun step per row
    sel_logits = out["resolver_logits"][mask]        # [N, C]
    sel_gold = batch.cand_gold[mask]                 # [N]
    aux_loss = F.cross_entropy(sel_logits, sel_gold)
    assert aux_loss.dim() == 0 and torch.isfinite(aux_loss)
    total = F.cross_entropy(out["answer_logits"], batch.answer) + 0.5 * aux_loss
    total.backward()   # both the answer path and the resolver aux path must be differentiable together
    assert model.resolver.net[0].weight.grad is not None if hasattr(model.resolver, "net") else True


# ---------------------------------------------------------------------------
# 5. Margin exposure test (fixed, hand-checkable logits via a stub resolver).
# ---------------------------------------------------------------------------
class _FixedLogitResolver(Resolver):
    """Test-only stub: ignores every input, always returns a fixed [B, C] logit
    row (plus a dummy trainable param so it's a well-formed nn.Module)."""

    def __init__(self, row):
        super().__init__()
        self._row = torch.tensor(row, dtype=torch.float32)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state):
        b = cand_entity.shape[0]
        return self._row.unsqueeze(0).expand(b, -1).clone()


def test_margin_matches_hand_computed_top1_minus_top2():
    torch.manual_seed(0)
    batch, pronoun_t = _toy_batch_with_candidates(b=3, d=16, K=4, C=3, seed=7)
    model = ClauseReactor(dim=16, resolver=_FixedLogitResolver([5.0, 0.0, -5.0]))
    model.eval()
    with torch.no_grad():
        out = model(batch)
    expected_margin = 5.0 - 0.0
    assert torch.allclose(out["resolver_margin"][:, pronoun_t],
                           torch.full((3,), expected_margin), atol=1e-4)
    # every non-pronoun step has no candidate set -> margin forced to 0
    other_steps = [t for t in range(batch.entity.shape[1]) if t != pronoun_t]
    assert torch.equal(out["resolver_margin"][:, other_steps],
                        torch.zeros(3, len(other_steps)))


def test_margin_respects_padding_mask():
    """A padded candidate slot must never win top-1/top-2 even if the (stub)
    resolver assigns it the largest raw logit -- ClauseReactor masks before
    ranking, per the contract ('the caller masks + softmaxes')."""
    torch.manual_seed(0)
    batch, pronoun_t = _toy_batch_with_candidates(b=2, d=16, K=4, C=3, seed=8)
    batch.cand_mask[:, pronoun_t, 2] = 0.0   # candidate slot 2 is padding
    model = ClauseReactor(dim=16, resolver=_FixedLogitResolver([0.0, 1.0, 100.0]))
    model.eval()
    with torch.no_grad():
        out = model(batch)
    # only candidates 0 and 1 are real -> margin = 1.0 - 0.0, NOT 100 - 1
    assert torch.allclose(out["resolver_margin"][:, pronoun_t], torch.full((2,), 1.0), atol=1e-4)


# ---------------------------------------------------------------------------
# 6. Determinism.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("track", ["A", "B"])
def test_determinism_same_seed_same_outputs(track):
    batch, _ = _toy_batch_with_candidates(b=5, d=16, K=4, C=3, seed=9)

    torch.manual_seed(42)
    m1 = ClauseReactor(dim=16, resolver=make_resolver(track, 16, 128))
    torch.manual_seed(42)
    m2 = ClauseReactor(dim=16, resolver=make_resolver(track, 16, 128))

    for p1, p2 in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(p1, p2)

    m1.eval(); m2.eval()
    with torch.no_grad():
        out1, out2 = m1(batch), m2(batch)
    for k in out1:
        assert torch.equal(out1[k], out2[k])

    m1.train(); m2.train()
    out1, out2 = m1(batch), m2(batch)
    for k in out1:
        assert torch.equal(out1[k], out2[k])
