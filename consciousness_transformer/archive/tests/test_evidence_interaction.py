"""Tests for M57c.3: the evidence*target INTERACTION feature (RESEARCH_NOTES
"M57c battery #2" -- forced-gold PROVES the read path (0.423->0.856), but the
TRAINED resolver binds instance candidates at CHANCE because it never had
the referring expression's own TARGET vector to compare a candidate's
evidence readout against -- M54c's lesson repeated: a shared/specialist
head needs an explicit INTERACTION PRIOR, not just both vectors present
somewhere in its inputs).

Covers: membrane.EntityCandidateSet.evidence_target, ClauseBatch's new
cand_evidence_target field (byte-identity when absent), the pure
nsm_ct.resolver.evidence_interaction helper, CorefHead's widened
cand_feature_extra register, ClauseReactor's evidence_prior_beta
structural-prior option, and the load-bearing short-training A/B (with vs
without the interaction feature).

No parser dependency anywhere in this file -- mirrors
tests/test_instance_curriculum.py's / tests/test_rich_episodes.py's own
isolation discipline.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from nsm_ct import entity_memory as em  # noqa: E402
from nsm_ct.clause_reactor import ClauseBatch, ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import generate_instance_episodes, generate_rich_episodes  # noqa: E402
from nsm_ct.episode import split_episodes  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.resolver import Resolver, evidence_interaction, make_resolver, query_candidates  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

from _train_common import epoch_minibatches  # noqa: E402

DIM = 24


def _meaning():
    return NSMMeaningResolver()


def _codec(dim=DIM):
    return TPRCodec(dim=dim)


# ---------------------------------------------------------------------------
# 1. Byte-identity: cand_evidence_target absent/None.
# ---------------------------------------------------------------------------
def test_cand_evidence_target_is_none_for_every_pre_m57c3_batch():
    """A batch built from pronoun/writeback episodes -- neither generator's
    candidate sets ever populate EntityCandidateSet.evidence_target -- must
    leave cand_evidence_target None, the same "only if present" discipline
    every optional ClauseBatch field establishes."""
    from nsm_ct.curriculum2 import generate_writeback_episodes
    from nsm_ct.episode import CurriculumGenerator

    meaning = _meaning()
    codec = _codec()
    old_eps = CurriculumGenerator(max_level=6, seed=0).generate(10)
    batch_old = build_clause_batch(old_eps, None, meaning, codec)
    assert batch_old.cand_evidence_target is None

    wb_eps = generate_writeback_episodes(10, seed=0)
    batch_wb = build_clause_batch(wb_eps, None, meaning, codec)
    assert batch_wb.cand_evidence_target is None


def test_instance_batch_populates_cand_evidence_target():
    """An instance-episode batch DOES populate cand_evidence_target at
    every step carrying a candidate set (the overwrite step, and the
    question step when it targets a name-sharing instance)."""
    meaning = _meaning()
    codec = _codec()
    eps = generate_instance_episodes(20, seed=0, inverse_frac=0.0)
    batch = build_clause_batch(eps, None, meaning, codec)
    assert batch.cand_evidence_target is not None
    # every (row, step) with a real candidate set must carry a nonzero target
    has_cand_step = batch.cand_mask.sum(-1) > 0
    target_norm = batch.cand_evidence_target.norm(dim=-1)
    assert bool((target_norm[has_cand_step] > 0).all())


def test_forward_output_byte_identical_with_and_without_evidence_target_field_when_none():
    """A hand-built batch with cand_evidence_target explicitly absent
    (dataclass default) vs explicitly None must produce IDENTICAL model
    output -- the no-op guard in ClauseReactor._collapse's entity branch
    (``s_c = None`` whenever ``batch.cand_evidence_target is None``)."""
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

    # both the plain track A and the widened (cand_feature_extra=1) track A
    # must be byte-identical when cand_evidence_target is absent.
    for resolver in (make_resolver("A", d, 8),
                     make_resolver("A", d, 8, use_cand_feature=True, cand_feature_extra=1)):
        model = ClauseReactor(dim=d, hidden=8, resolver=resolver)
        model.eval()

        batch_absent = ClauseBatch(**kwargs)
        assert batch_absent.cand_evidence_target is None
        with torch.no_grad():
            out_absent = model(batch_absent)

        batch_none = ClauseBatch(**kwargs, cand_evidence_target=None)
        with torch.no_grad():
            out_none = model(batch_none)

        for k in out_absent:
            assert torch.equal(out_absent[k], out_none[k]), k


# ---------------------------------------------------------------------------
# 2. Interaction mechanics: evidence_interaction on a hand-built read.
# ---------------------------------------------------------------------------
def test_evidence_interaction_mechanics_doctor_vs_teacher():
    """Candidate 0's attr:kind slot holds the DOCTOR vector, candidate 1's
    holds the TEACHER vector; the target is the doctor vector -> s_0 must
    be near 1.0 (self-cosine) and s_1 must equal cos(doctor, teacher) --
    exactly the production chain (query_candidates then
    evidence_interaction) ClauseReactor._collapse runs."""
    torch.manual_seed(0)
    d = 16
    cand0 = F.normalize(torch.randn(1, d), dim=-1)
    cand1 = F.normalize(torch.randn(1, d), dim=-1)
    kind_rel = F.normalize(torch.randn(1, d), dim=-1)
    doctor_vec = F.normalize(torch.randn(1, d), dim=-1)
    teacher_vec = F.normalize(torch.randn(1, d), dim=-1)

    memory = em.init_memory(1, d, "cpu")
    gate = torch.ones(1)
    memory = em.write(memory, cand0, kind_rel, doctor_vec, gate)
    memory = em.write(memory, cand1, kind_rel, teacher_vec, gate)

    cand_entity = torch.stack([cand0, cand1], dim=1)                     # [1, 2, d]
    cand_mem_read = query_candidates(memory, cand_entity, kind_rel)      # [1, 2, d]

    s = evidence_interaction(cand_mem_read, doctor_vec)                  # [1, 2]
    expected_cross = F.cosine_similarity(doctor_vec, teacher_vec).item()
    assert s[0, 0].item() > 0.9
    assert abs(s[0, 1].item() - expected_cross) < 1e-3


def test_evidence_interaction_zero_when_target_is_zero_vector():
    """A target of all-zeros (the 'no evidence target at this row' padding
    convention) must yield s_c = 0 for every candidate, with no NaN --
    F.cosine_similarity's own zero-vector floor, not extra masking."""
    torch.manual_seed(0)
    cand_mem_read = F.normalize(torch.randn(2, 3, 8), dim=-1)
    target = torch.zeros(2, 8)
    s = evidence_interaction(cand_mem_read, target)
    assert torch.equal(s, torch.zeros(2, 3))
    assert not torch.isnan(s).any()


# ---------------------------------------------------------------------------
# 3. CorefHead's widened register (cand_feature_extra).
# ---------------------------------------------------------------------------
def test_corefhead_cand_feature_extra_widens_input_and_accepts_the_wider_register():
    from nsm_ct.membrane import FEATURE_DIM
    from nsm_ct.resolver import CorefHead

    plain = CorefHead(dim=16, use_cand_feature=True)
    widened = CorefHead(dim=16, use_cand_feature=True, cand_feature_extra=1)
    assert widened.cand_feature_extra == 1
    assert plain.cand_feature_extra == 0

    b, C, d = 2, 3, 16
    cand_entity = torch.randn(b, C, d)
    cand_feature = torch.randn(b, FEATURE_DIM)
    cand_prior = torch.full((b, C), 1.0 / C)
    cand_mask = torch.ones(b, C)
    mem_read = torch.randn(b, C, d)
    state = torch.randn(b, 8)

    cfpc_wide = torch.randn(b, C, FEATURE_DIM + 1)
    out = widened(cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state,
                  cand_feature_per_candidate=cfpc_wide)
    assert out.shape == (b, C)

    # a widened head with no per-candidate register supplied falls back to
    # its OWN (wider) zero default -- must not shape-mismatch.
    out_default = widened(cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state)
    assert out_default.shape == (b, C)

    # cand_feature_extra=0 (default) is byte-identical to pre-M57c.3 CorefHead.
    torch.manual_seed(0)
    a1 = CorefHead(dim=16, use_cand_feature=True)
    torch.manual_seed(0)
    a2 = CorefHead(dim=16, use_cand_feature=True, cand_feature_extra=0)
    for p1, p2 in zip(a1.parameters(), a2.parameters()):
        assert torch.equal(p1, p2)
    del plain, out


# ---------------------------------------------------------------------------
# 4. --evidence-prior semantics: the prior mass moves toward the matching
#    candidate.
# ---------------------------------------------------------------------------
class _EchoPriorResolver(Resolver):
    """Test-only stub: records the ``cand_prior`` it was actually called
    with (post any structural-prior mixing) and returns arbitrary
    (uniform) logits -- isolates ClauseReactor._collapse's evidence-prior
    arithmetic from anything resolver-learned."""

    def __init__(self):
        super().__init__()
        self.captured_prior = None
        self._dummy = torch.nn.Linear(1, 1)   # so this is a real nn.Module with params

    def forward(self, cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state):
        self.captured_prior = cand_prior.detach().clone()
        b, C, _d = cand_entity.shape
        return torch.zeros(b, C)


def test_evidence_prior_flag_moves_prior_mass_to_matching_candidate():
    torch.manual_seed(0)
    d = 16
    cand0 = F.normalize(torch.randn(1, d), dim=-1)
    cand1 = F.normalize(torch.randn(1, d), dim=-1)
    kind_rel = F.normalize(torch.randn(1, d), dim=-1)
    doctor_vec = F.normalize(torch.randn(1, d), dim=-1)
    teacher_vec = F.normalize(torch.randn(1, d), dim=-1)

    memory = em.init_memory(1, d, "cpu")
    gate = torch.ones(1)
    memory = em.write(memory, cand0, kind_rel, doctor_vec, gate)
    memory = em.write(memory, cand1, kind_rel, teacher_vec, gate)

    resolver = _EchoPriorResolver()
    state = torch.zeros(1, 8)
    e = torch.zeros(1, d)
    r = kind_rel
    v = torch.zeros(1, d)
    mem_read = torch.zeros(1, d)
    cand_entity = torch.stack([cand0, cand1], dim=1)

    common = dict(
        entity=e.unsqueeze(1), relation=r.unsqueeze(1), value=v.unsqueeze(1),
        pred=torch.zeros(1, 1, d), is_q=torch.zeros(1, 1), mask=torch.ones(1, 1),
        options=torch.zeros(1, 1, d), answer=torch.zeros(1, dtype=torch.long),
        cand_entity=cand_entity.unsqueeze(1), cand_mask=torch.ones(1, 1, 2),
        cand_prior=torch.full((1, 1, 2), 0.5), cand_feature=torch.zeros(1, 1, 6),
        cand_gold=torch.full((1, 1), -1, dtype=torch.long),
        cand_evidence_target=doctor_vec.unsqueeze(1),
    )
    batch = ClauseBatch(**common)

    # beta=None (default, inert): captured prior unchanged from the batch's own.
    model_off = ClauseReactor(dim=d, hidden=8, resolver=resolver, evidence_prior_beta=None)
    model_off._collapse(memory, state, mem_read, e, r, v, batch, 0)
    assert torch.allclose(resolver.captured_prior, torch.full((1, 2), 0.5))

    # beta set: prior mass moves toward candidate 0 (the doctor match).
    # NOTE: ``cand_prior * softmax(...)`` is a literal MULTIPLY, not a
    # renormalization -- since cand_prior starts uniform (0.5/0.5, the two
    # factors cancel in the RATIO), the boosted SHARE of prior mass on
    # candidate 0 (not its raw value, which can shrink below the original
    # 0.5 -- softmax mass gets split, never added to) equals softmax(s_c *
    # beta)[0], which must exceed its original uniform share (0.5).
    model_on = ClauseReactor(dim=d, hidden=8, resolver=resolver, evidence_prior_beta=5.0)
    model_on._collapse(memory, state, mem_read, e, r, v, batch, 0)
    boosted = resolver.captured_prior
    assert boosted[0, 0] > boosted[0, 1]
    share0 = boosted[0, 0] / boosted[0].sum()
    assert share0 > 0.5


# ---------------------------------------------------------------------------
# 5. Rich episodes build with the new field.
# ---------------------------------------------------------------------------
def test_rich_episodes_populate_cand_evidence_target():
    meaning = _meaning()
    codec = _codec()
    eps = generate_rich_episodes(20, seed=0, inverse_frac=0.0)
    batch = build_clause_batch(eps, None, meaning, codec)
    assert batch.cand_evidence_target is not None
    has_cand_step = batch.cand_mask.sum(-1) > 0
    target_norm = batch.cand_evidence_target.norm(dim=-1)
    assert bool((target_norm[has_cand_step] > 0).all())


# ---------------------------------------------------------------------------
# 6. THE LOAD-BEARING TEST: short training, with vs without the interaction
#    feature. dim=24, ~200 instance episodes, ~150 steps, batch 32.
# ---------------------------------------------------------------------------
def _binding_accuracy(model, batch, eps):
    model.eval()
    with torch.no_grad():
        out = model(batch)
    if "resolver_logits" not in out:
        return None
    has_cand = batch.cand_gold >= 0
    pred_idx = out["resolver_logits"].argmax(-1)
    correct = (pred_idx == batch.cand_gold) & has_cand
    total = int(has_cand.sum())
    if total == 0:
        return None
    return float(correct.sum()) / total


def _train_short(cand_feature_extra: int, dim: int, seed: int) -> float:
    meaning = _meaning()
    codec = TPRCodec(dim=dim)
    episodes = generate_instance_episodes(260, seed=0, inverse_frac=0.0)
    # The "ambiguous_name" device is, BY DESIGN, not decidable via this
    # interaction feature at all (see _ground_evidence_target's docstring:
    # its evidence_relation reads attr:kind, but the referring expression
    # names no kind -- it disambiguates via discourse RECENCY, a channel
    # CorefHead deliberately never sees). InstanceCurriculumGenerator also
    # bakes in genuine, unresolvable-in-principle GENDER TIES for the
    # "pronoun" device in ~1/3 of episodes (curriculum2's own "a/b share
    # gender in EXACTLY half of episodes ... pronoun's evidence is
    # therefore sometimes tied" note) -- both are curriculum-inherent
    # ceilings on ANY resolver, not an interaction-feature deficiency, so
    # this load-bearing mechanics test isolates the device the design's own
    # worked example uses (definite_description: kind is ALWAYS unique --
    # "definite description's evidence never ties") plus pronoun (still
    # included: mostly solvable, its tied minority is the honest reason the
    # "with" arm's ceiling sits below 1.0, not above the 0.8 bar this test
    # sets).
    episodes = [e for e in episodes if e.meta.get("referring_device") != "ambiguous_name"]
    tr_eps, va_eps = split_episodes(episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, None, meaning, codec)
    va = build_clause_batch(va_eps, None, meaning, codec)

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

    return _binding_accuracy(model, va, va_eps)


def test_short_training_interaction_feature_lifts_instance_binding_above_chance():
    """THE load-bearing test (director-locked design item 5): with the
    evidence*target interaction feature wired into the register
    (cand_feature_extra=1), instance binding accuracy must exceed 0.8 in a
    short training run; with it absent (cand_feature_extra=0, otherwise
    IDENTICAL setup/batch/seed), it must stay near chance. This is the
    smoke-scale mechanics proof -- RESEARCH_NOTES's full-scale battery is
    the real gate, per CLAUDE.md's "smoke-scale never gates curriculum
    validity", but a smoke-scale FAILURE here would mean the mechanism
    itself is broken, not merely under-trained."""
    acc_with = _train_short(cand_feature_extra=1, dim=24, seed=0)
    acc_without = _train_short(cand_feature_extra=0, dim=24, seed=0)
    assert acc_with is not None and acc_without is not None
    assert acc_with > 0.8, f"acc_with={acc_with}"
    assert acc_without < 0.7, f"acc_without={acc_without}"
    assert acc_with - acc_without > 0.2
