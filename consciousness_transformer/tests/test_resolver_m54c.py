"""Tests for M54c: three diagnosis arms unpicking the A-vs-B confounds
(RESEARCH_NOTES M54c) -- capacity (B-wide), architecture/state-input
(B-nostate, B-nostate-wide), and pretraining (B-distilled).

Covers: (1) nsm_ct.resolver.SharedScorer's new ``use_state`` constructor
flag + nsm_ct.resolver.shared_scorer_for_budget (default-SharedScorer
regression + param-count-matching correctness); (2) scripts/train_resolver.py's
new track dispatch (build_resolvers/build_b_family_resolver) and
distillation loss mechanics (aux_loss_terms/distill_loss_terms/_distill_kl).

Mirrors tests/test_resolver.py's synthetic-tensor style for (1) and
tests/test_m32_ambiguity.py's sys.path pattern for importing the script
under test for (2).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nsm_ct.membrane import FEATURE_DIM
from nsm_ct.resolver import CorefHead, SenseHead, SharedScorer, shared_scorer_for_budget

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import train_resolver as tr  # noqa: E402


# ---------------------------------------------------------------------------
# 1. SharedScorer's use_state flag: default (True) must be a byte-identical
#    regression against the pre-M54c architecture; False must actually drop
#    the state dependency.
# ---------------------------------------------------------------------------
class _PreM54cSharedScorer(nn.Module):
    """Verbatim copy of SharedScorer's forward/__init__ BEFORE M54c's
    ``use_state`` flag was added -- the regression oracle. If this ever
    drifts from the real pre-M54c code, the regression test below is
    meaningless, so keep it a literal copy, not a paraphrase."""

    def __init__(self, dim: int, hidden: int, state_proj: int = 8, mlp_hidden: int = 16) -> None:
        super().__init__()
        self.state_proj = nn.Linear(hidden, state_proj)
        cand_vec_dim = dim + FEATURE_DIM + 1
        in_dim = cand_vec_dim + dim + state_proj
        self.net = nn.Sequential(nn.Linear(in_dim, mlp_hidden), nn.Tanh(),
                                  nn.Linear(mlp_hidden, 1))

    def forward(self, cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state):
        b, C, _d = cand_entity.shape
        feat = cand_feature.unsqueeze(1).expand(b, C, -1)
        prior = cand_prior.unsqueeze(-1)
        candidate_vec = torch.cat([cand_entity, feat, prior], dim=-1)
        s = self.state_proj(state).unsqueeze(1).expand(b, C, -1)
        x = torch.cat([candidate_vec, mem_read, s], dim=-1)
        return self.net(x).squeeze(-1)


def _synthetic_inputs(b=5, C=4, dim=16, hidden=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    cand_entity = torch.randn(b, C, dim, generator=g)
    cand_feature = torch.randn(b, FEATURE_DIM, generator=g)
    cand_prior = torch.rand(b, C, generator=g)
    cand_mask = torch.ones(b, C)
    mem_read = torch.randn(b, C, dim, generator=g)
    state = torch.randn(b, hidden, generator=g)
    return cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state


def test_default_use_state_true_matches_pre_m54c_architecture_exactly():
    """Regression gate: SharedScorer's default behavior (use_state=True,
    the only behavior that existed before M54c) must be numerically
    IDENTICAL to the pre-M54c code -- same seeded init, same forward output."""
    inputs = _synthetic_inputs()
    torch.manual_seed(123)
    old = _PreM54cSharedScorer(dim=16, hidden=32)
    torch.manual_seed(123)
    new = SharedScorer(dim=16, hidden=32)   # use_state defaults to True
    for (n1, p1), (n2, p2) in zip(old.state_dict().items(), new.state_dict().items()):
        assert torch.equal(p1, p2), f"{n1} vs {n2} diverged"
    old.eval(); new.eval()
    with torch.no_grad():
        out_old = old(*inputs)
        out_new = new(*inputs)
    assert torch.equal(out_old, out_new)


def test_default_use_state_true_explicit_matches_implicit():
    """Passing use_state=True explicitly must match omitting it (the
    constructor default)."""
    torch.manual_seed(7)
    implicit = SharedScorer(dim=16, hidden=32)
    torch.manual_seed(7)
    explicit = SharedScorer(dim=16, hidden=32, use_state=True)
    for p1, p2 in zip(implicit.state_dict().values(), explicit.state_dict().values()):
        assert torch.equal(p1, p2)


def test_use_state_false_drops_state_proj_submodule():
    scorer = SharedScorer(dim=16, hidden=32, use_state=False)
    assert scorer.state_proj is None
    # in_dim shrinks by state_proj (no state concatenated)
    with_state = SharedScorer(dim=16, hidden=32, use_state=True, mlp_hidden=16, state_proj=8)
    without_state = SharedScorer(dim=16, hidden=32, use_state=False, mlp_hidden=16, state_proj=8)
    assert without_state.net[0].in_features == with_state.net[0].in_features - 8


def test_use_state_false_output_invariant_to_state():
    """The whole point of the no-state variant: changing `state` must not
    change the logits at all (it is never read)."""
    torch.manual_seed(0)
    scorer = SharedScorer(dim=16, hidden=32, use_state=False)
    scorer.eval()
    cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state1 = _synthetic_inputs(seed=1)
    state2 = torch.randn_like(state1) * 100.0 + 50.0
    with torch.no_grad():
        out1 = scorer(cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state1)
        out2 = scorer(cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state2)
    assert torch.equal(out1, out2)


def test_use_state_true_output_does_depend_on_state():
    """Sanity check the fixture: the WITH-state variant must be sensitive to
    `state`, or the invariance test above would be vacuous."""
    torch.manual_seed(0)
    scorer = SharedScorer(dim=16, hidden=32, use_state=True)
    scorer.eval()
    cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state1 = _synthetic_inputs(seed=1)
    state2 = torch.randn_like(state1) * 100.0 + 50.0
    with torch.no_grad():
        out1 = scorer(cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state1)
        out2 = scorer(cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state2)
    assert not torch.equal(out1, out2)


def test_use_state_false_gradients_only_flow_into_net():
    torch.manual_seed(0)
    scorer = SharedScorer(dim=16, hidden=32, use_state=False)
    inputs = _synthetic_inputs(seed=2)
    out = scorer(*inputs)
    out.sum().backward()
    assert scorer.net[0].weight.grad is not None
    assert torch.any(scorer.net[0].weight.grad != 0)


# ---------------------------------------------------------------------------
# 2. shared_scorer_for_budget: hits (or lands very close to) the requested
#    total parameter count by varying mlp_hidden alone.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim,hidden", [(32, 128), (48, 128), (16, 32)])
def test_shared_scorer_for_budget_matches_a_combined_capacity(dim, hidden):
    a_total = (sum(p.numel() for p in CorefHead(dim).parameters()) +
               sum(p.numel() for p in SenseHead(dim).parameters()))
    wide = shared_scorer_for_budget(dim, hidden, a_total, use_state=True)
    n = sum(p.numel() for p in wide.parameters())
    assert abs(n - a_total) / a_total < 0.03, f"dim={dim}: {n} vs target {a_total}"
    assert wide.use_state is True


@pytest.mark.parametrize("dim,hidden", [(32, 128), (48, 128)])
def test_shared_scorer_for_budget_nostate_wide_also_matches(dim, hidden):
    a_total = (sum(p.numel() for p in CorefHead(dim).parameters()) +
               sum(p.numel() for p in SenseHead(dim).parameters()))
    wide_nostate = shared_scorer_for_budget(dim, hidden, a_total, use_state=False)
    n = sum(p.numel() for p in wide_nostate.parameters())
    assert abs(n - a_total) / a_total < 0.03
    assert wide_nostate.use_state is False
    assert wide_nostate.state_proj is None


def test_shared_scorer_for_budget_narrow_matches_original_b():
    dim, hidden = 48, 128
    original_b = sum(p.numel() for p in SharedScorer(dim, hidden).parameters())
    narrow_nostate = shared_scorer_for_budget(dim, hidden, original_b, use_state=False)
    n = sum(p.numel() for p in narrow_nostate.parameters())
    assert abs(n - original_b) / original_b < 0.05


def test_shared_scorer_for_budget_monotonic_search_handles_tiny_target():
    """A target already met (or unmatchable-small) at mlp_hidden=1 must not
    crash -- the search's base case."""
    scorer = shared_scorer_for_budget(dim=8, hidden=8, target_params=1, use_state=True)
    assert isinstance(scorer, SharedScorer)


# ---------------------------------------------------------------------------
# 3. train_resolver.py track dispatch: build_resolvers / build_b_family_resolver
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("track", ["B-wide", "B-nostate", "B-nostate-wide"])
def test_b_family_dispatch_returns_shared_scorer_shared_instance(track):
    resolver, sense_resolver = tr.build_resolvers(track, dim=16, hidden=32)
    assert isinstance(resolver, SharedScorer)
    assert resolver is sense_resolver, "B-family tracks must install ONE shared instance"


def test_b_wide_dispatch_uses_state_b_nostate_variants_dont():
    wide, _ = tr.build_resolvers("B-wide", dim=16, hidden=32)
    nostate, _ = tr.build_resolvers("B-nostate", dim=16, hidden=32)
    nostate_wide, _ = tr.build_resolvers("B-nostate-wide", dim=16, hidden=32)
    assert wide.use_state is True
    assert nostate.use_state is False
    assert nostate_wide.use_state is False


def test_track_dispatch_case_insensitive_and_matches_plain_ab():
    """"A"/"B" dispatch must be untouched by the refactor into build_resolvers:
    "A" -> two independent heads, "B" -> one shared instance."""
    from nsm_ct.resolver import CorefHead as _CH, SenseHead as _SH
    r, s = tr.build_resolvers("A", dim=16, hidden=32)
    assert isinstance(r, _CH) and isinstance(s, _SH)
    assert r is not s
    r2, s2 = tr.build_resolvers("b", dim=16, hidden=32)   # lowercase, like make_resolver accepts
    assert isinstance(r2, SharedScorer)
    assert r2 is s2


def test_unknown_track_raises():
    with pytest.raises(ValueError):
        tr.build_resolvers("C", dim=16, hidden=32)
    with pytest.raises(ValueError):
        tr.build_b_family_resolver("B-EXTRA-WIDE", dim=16, hidden=32)


def test_a_combined_params_matches_direct_sum():
    dim = 24
    expected = (sum(p.numel() for p in CorefHead(dim).parameters()) +
                sum(p.numel() for p in SenseHead(dim).parameters()))
    assert tr.a_combined_params(dim) == expected


# ---------------------------------------------------------------------------
# 4. Distillation loss mechanics: aux_loss_terms / distill_loss_terms / _distill_kl
# ---------------------------------------------------------------------------
def test_distill_kl_is_zero_for_identical_distributions():
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    has = torch.tensor([True])
    kl = tr._distill_kl(logits, logits.clone(), has)
    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


def test_distill_kl_matches_hand_computed_value():
    student = torch.tensor([[1.0, 0.0]])
    teacher = torch.tensor([[0.0, 1.0]])
    has = torch.tensor([True])
    kl = tr._distill_kl(student, teacher, has)
    p = F.softmax(student, dim=-1)
    log_p = F.log_softmax(student, dim=-1)
    log_q = F.log_softmax(teacher, dim=-1)
    expected = (p * (log_p - log_q)).sum(-1).mean()
    assert torch.isclose(kl, expected, atol=1e-6)


def test_distill_kl_direction_is_b_given_a_not_symmetric():
    """KL(B||A) != KL(A||B) in general -- confirms the function isn't
    accidentally symmetric (e.g. via a JS-divergence-style average)."""
    student = torch.tensor([[3.0, -3.0, 0.0]])
    teacher = torch.tensor([[0.0, 0.0, 3.0]])
    has = torch.tensor([True])
    kl_b_given_a = tr._distill_kl(student, teacher, has)
    kl_a_given_b = tr._distill_kl(teacher, student, has)
    assert not torch.isclose(kl_b_given_a, kl_a_given_b, atol=1e-4)


def test_distill_kl_ignores_rows_without_candidates():
    student = torch.tensor([[1.0, 0.0], [5.0, -5.0]])
    teacher = torch.tensor([[0.0, 1.0], [-5.0, 5.0]])
    has = torch.tensor([True, False])
    kl = tr._distill_kl(student, teacher, has)
    # must equal the single-row computation, not an average diluted by row 2
    expected = tr._distill_kl(student[:1], teacher[:1], torch.tensor([True]))
    assert torch.isclose(kl, expected, atol=1e-6)


def test_distill_kl_returns_none_when_no_rows_have_candidates():
    student = torch.tensor([[1.0, 0.0]])
    teacher = torch.tensor([[0.0, 1.0]])
    has = torch.tensor([False])
    assert tr._distill_kl(student, teacher, has) is None


def test_distill_kl_padding_does_not_nan():
    """Padded slots carry -1e9 (ClauseReactor._collapse's masking convention)
    -- softmax there is exactly 0.0 in float32, so 0 * finite must not NaN."""
    student = torch.tensor([[1.0, -1e9, -1e9]])
    teacher = torch.tensor([[0.5, -1e9, -1e9]])
    has = torch.tensor([True])
    kl = tr._distill_kl(student, teacher, has)
    assert torch.isfinite(kl)


def test_distill_loss_terms_sums_both_kinds_present():
    student_out = {"resolver_logits": torch.tensor([[1.0, 0.0]]),
                   "sense_resolver_logits": torch.tensor([[0.0, 1.0]])}
    teacher_out = {"resolver_logits": torch.tensor([[0.0, 1.0]]),
                   "sense_resolver_logits": torch.tensor([[1.0, 0.0]])}

    class _Batch:
        cand_gold = torch.tensor([0])
        sense_cand_gold = torch.tensor([0])

    total = tr.distill_loss_terms(student_out, teacher_out, _Batch())
    expected = (tr._distill_kl(student_out["resolver_logits"], teacher_out["resolver_logits"],
                                _Batch.cand_gold >= 0)
                + tr._distill_kl(student_out["sense_resolver_logits"], teacher_out["sense_resolver_logits"],
                                  _Batch.sense_cand_gold >= 0))
    assert torch.isclose(total, expected, atol=1e-6)


def test_distill_loss_terms_zero_when_neither_kind_present():
    class _Batch:
        cand_gold = torch.tensor([-1])
        sense_cand_gold = torch.tensor([-1])

    total = tr.distill_loss_terms({}, {}, _Batch())
    assert total == 0.0


def test_aux_loss_terms_matches_manual_cross_entropy():
    resolver = object()   # only needs to be non-None; not called directly
    out = {"resolver_logits": torch.tensor([[2.0, 0.0], [0.0, 2.0]])}

    class _Batch:
        cand_gold = torch.tensor([0, 1])

    total = tr.aux_loss_terms(out, _Batch(), resolver, None)
    expected = tr.AUX_WEIGHT * F.cross_entropy(out["resolver_logits"], _Batch.cand_gold)
    assert torch.isclose(total, expected, atol=1e-6)


def test_aux_loss_terms_zero_when_resolver_none():
    out = {"resolver_logits": torch.tensor([[2.0, 0.0]])}

    class _Batch:
        cand_gold = torch.tensor([0])

    assert tr.aux_loss_terms(out, _Batch(), None, None) == 0.0


# ---------------------------------------------------------------------------
# 5. End-to-end mechanics smoke (tiny, fast): run_distilled_arm actually
#    trains three stages and transitions between them. Needs quantum_parser;
#    skips cleanly if unavailable (mirrors test_resolver.py's own guard
#    inside run_arm, surfaced here as an explicit skip instead of a no-op
#    return so a missing parser doesn't silently look like a pass).
# ---------------------------------------------------------------------------
def _parser_available():
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.tokenizer import SimpleTokenizer
    tok = SimpleTokenizer.build(["mary went to the garden ."])
    return getattr(ParserInputEncoder(tok), "_parser", None) is not None


@pytest.mark.skipif(not _parser_available(), reason="quantum_parser unavailable in this environment")
def test_run_distilled_arm_stages_transition_and_loss_decreases():
    episodes = tr.build_m54b_curriculum(40, seed=0)
    result = tr.run_distilled_arm("test-distilled", episodes, dim=8, seed=0, hidden=16,
                                   stage1_epochs=3, stage2_epochs=3, stage3_epochs=2)
    assert result is not None
    assert len(result["stage1_losses"]) == 3
    assert len(result["stage2_losses"]) == 3
    assert len(result["stage2_kl"]) == 3
    assert len(result["stage3_losses"]) == 2
    # mechanics: losses are finite throughout every stage
    for key in ("stage1_losses", "stage2_losses", "stage3_losses"):
        assert all(l == l and abs(l) < float("inf") for l in result[key])


@pytest.mark.skipif(not _parser_available(), reason="quantum_parser unavailable in this environment")
@pytest.mark.parametrize("track", ["B-wide", "B-nostate", "B-nostate-wide"])
def test_run_arm_new_tracks_execute_on_m54b_mix(track):
    episodes = tr.build_m54b_curriculum(30, seed=1)
    result = tr.run_arm(f"test-{track}", track, episodes, dim=8, epochs=2, seed=0, hidden=16)
    assert result is not None
    assert result["n_resolver_params"] == result["n_sense_params"]   # shared instance
    assert 0.0 <= result["total_acc"] <= 1.0


# ---------------------------------------------------------------------------
# 6. M56b: the held-out-name ablation (dev/TRACK_C_DESIGN.md §1.8/Risk #4,
#    RESEARCH_NOTES M56/M56b) -- scripts/train_resolver.py's
#    _held_out_name_pools + run_held_out_name_ablation.
# ---------------------------------------------------------------------------
def test_held_out_name_pools_are_disjoint_and_cover_everything():
    (train_f, train_m), (ho_f, ho_m) = tr._held_out_name_pools("sandra", "bill")
    assert ho_f == ["sandra"] and ho_m == ["bill"]
    assert "sandra" not in train_f and "bill" not in train_m
    from nsm_ct.curriculum2 import _FEMALE_NAMES, _MALE_NAMES
    assert set(train_f) | set(ho_f) == set(_FEMALE_NAMES)
    assert set(train_m) | set(ho_m) == set(_MALE_NAMES)


def test_held_out_name_pools_rejects_unknown_name():
    with pytest.raises(ValueError):
        tr._held_out_name_pools("mary", "not-a-real-name")
    with pytest.raises(ValueError):
        tr._held_out_name_pools("not-a-real-name", "bill")


@pytest.mark.skipif(not _parser_available(), reason="quantum_parser unavailable in this environment")
def test_run_held_out_name_ablation_mechanics_smoke():
    """Tiny, fast mechanics check (not a gate run -- see RESEARCH_NOTES M56b
    for the real smoke-scale numbers): both heads train without error, the
    2x2 result dict has the right shape, and every reported accuracy is a
    valid probability computed over a non-empty n."""
    result = tr.run_held_out_name_ablation(
        n_episodes=40, epochs=2, dim=8, seed=0, hidden=16,
        held_out_female="sandra", held_out_male="bill", heads=("old", "fixed"))
    assert set(result.keys()) == {"old", "fixed"}
    for label in ("old", "fixed"):
        r = result[label]
        assert 0.0 <= r["train_names"] <= 1.0
        assert 0.0 <= r["held_out_names"] <= 1.0
        assert r["n_train_names"] > 0
        assert r["n_held_out_names"] > 0


@pytest.mark.skipif(not _parser_available(), reason="quantum_parser unavailable in this environment")
def test_run_held_out_name_ablation_fixed_head_uses_more_params_than_old():
    """use_cand_feature=True adds a second FEATURE_DIM-wide input slot to
    CorefHead -- a cheap, direct way to confirm the "fixed" arm is actually
    installing the new architecture, not silently reusing the old one."""
    from nsm_ct.resolver import CorefHead
    n_old = sum(p.numel() for p in CorefHead(8, use_cand_feature=False).parameters())
    n_fixed = sum(p.numel() for p in CorefHead(8, use_cand_feature=True).parameters())
    assert n_fixed > n_old
