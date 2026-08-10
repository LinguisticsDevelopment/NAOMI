"""Tests for M54: sense collapse joins the same membrane.

Covers dev/MIND_INTERFACE.md's v1 "value" IN row and
dev/RESOLVER_BUILD_PLAN.md Phase 3 -- nsm_ct.membrane's SenseCandidateSet,
nsm_ct.resolver's SenseHead (Track A) + SharedScorer reused unchanged
(Track B), and nsm_ct.clause_reactor's batch-build sense packing +
_collapse extension. Mirrors tests/test_membrane.py's and
tests/test_resolver.py's own structure/style (parser-backed tests for
curriculum integration, synthetic-tensor tests for the resolver contract).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nsm_ct import entity_memory as em
from nsm_ct.clause_reactor import ClauseBatch, ClauseReactor, build_clause_batch
from nsm_ct.membrane import FEATURE_DIM, Candidate, SenseCandidateSet, sense_candidate_set
from nsm_ct.resolver import Resolver, SenseHead, SharedScorer, make_resolver, make_sense_resolver
from nsm_ct.tpr import TPRCodec
from nsm_ct.usvs_bridge import usvs_sense_handle


# ---------------------------------------------------------------------------
# 1. SenseCandidateSet construction / determinism (membrane, no parser needed)
# ---------------------------------------------------------------------------
def test_sense_candidate_set_reuses_generic_candidate_shape():
    cs = sense_candidate_set("plant", gold_sense="plant.n.01")
    assert isinstance(cs, SenseCandidateSet)
    assert len(cs) == len(cs.candidates)
    assert all(isinstance(c, Candidate) for c in cs.candidates)


def test_sense_candidate_set_mfs_ordered_and_priors_normalized_and_decreasing():
    cs = sense_candidate_set("bank", gold_sense="depository_financial_institution.n.01")
    assert cs.keys[0] == "bank.n.01"     # MFS is index 0 by construction (episode.py's own convention)
    assert cs.gold_index == cs.keys.index("depository_financial_institution.n.01")
    assert np.isclose(float(cs.priors.sum()), 1.0, atol=1e-5)
    assert all(cs.priors[i] > cs.priors[i + 1] for i in range(len(cs.priors) - 1))


def test_sense_candidate_set_deterministic():
    cs1 = sense_candidate_set("organ", gold_sense="organ.n.01")
    cs2 = sense_candidate_set("organ", gold_sense="organ.n.01")
    assert cs1.keys == cs2.keys
    assert np.allclose(cs1.priors, cs2.priors)
    assert cs1.gold_index == cs2.gold_index


def test_sense_candidate_set_gold_not_among_candidates_is_none():
    cs = sense_candidate_set("bank", gold_sense="not_a_real_synset.n.01")
    assert cs.gold_index is None


def test_sense_candidate_set_context_word_carried_and_defaults_none():
    cs = sense_candidate_set("bank", context_word="river")
    assert cs.context_word == "river"
    assert sense_candidate_set("bank").context_word is None


def test_ungroundable_sense_id_returns_none_not_crash():
    """The defensive precondition build_clause_batch's masking branch relies
    on: usvs_sense_handle really can return None for a well-formed-looking
    but nonexistent synset id (every M32-family real sense id happens to be
    USVS-groundable today, so this path is never hit by the curriculum
    itself -- but the mask-out contract must hold regardless)."""
    assert usvs_sense_handle("not_a_real_synset.n.99", 32) is None


# ---------------------------------------------------------------------------
# 2. Curriculum integration: ambiguity episodes flow through build_clause_batch
# ---------------------------------------------------------------------------
def _amb_parser_env(dim=32, n=40, seed=0):
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer
    from nsm_ct.episode import generate_ambiguity_episodes

    eps = generate_ambiguity_episodes(n, seed=seed)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return parser, NSMMeaningResolver(), TPRCodec(dim=dim), eps


def _mixed_parser_env(dim=32, seed=0):
    """Ambiguity + pronoun episodes sharing one parser vocabulary, for the
    both-kinds-in-one-batch tests."""
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer
    from nsm_ct.episode import generate_ambiguity_episodes
    from nsm_ct.curriculum2 import generate_pronoun_episodes

    amb = generate_ambiguity_episodes(20, seed=seed)
    pron = generate_pronoun_episodes(20, seed=seed)
    eps = amb + pron
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return parser, NSMMeaningResolver(), TPRCodec(dim=dim), amb, pron


def test_ambiguity_episodes_produce_rows_not_silently_dropped():
    """Before M54: the ambiguity question ("what kind of X is it ?") has no
    NAME and no "the", so _question_entity returns None and
    build_clause_batch's old else-branch would `continue` on EVERY episode --
    zero rows. The dedicated homograph branch must actually produce rows."""
    parser, resolver, codec, eps = _amb_parser_env(n=30)
    batch = build_clause_batch(eps, parser, resolver, codec)
    assert batch.entity.shape[0] == len(eps)   # every episode produced a row


def test_batch_carries_sense_candidates_for_homograph_steps():
    parser, resolver, codec, eps = _amb_parser_env(n=40)
    batch = build_clause_batch(eps, parser, resolver, codec)
    assert batch.sense_cand_entity is not None
    b, T, C, d = batch.sense_cand_entity.shape
    assert d == codec.dim
    assert batch.sense_cand_context.shape == (b, T, d)
    assert batch.sense_cand_gold.shape == (b, T)
    has_cand = batch.sense_cand_gold >= 0
    assert bool(has_cand.any())    # at least one episode actually got a sense step


def test_batch_packs_sense_gold_index_and_normalized_priors_correctly():
    from nsm_ct.wordnet import senses as wn_senses

    parser, resolver, codec, eps = _amb_parser_env(n=40)
    batch = build_clause_batch(eps, parser, resolver, codec)
    has_cand = batch.sense_cand_gold >= 0
    checked = 0
    for i, e in enumerate(eps):
        row = has_cand[i]
        if not bool(row.any()):
            continue
        ids = tuple(s["sense_id"] for s in wn_senses(e.meta["homograph"]))
        gold = e.meta["gold_sense"]
        expected_gold_idx = ids.index(gold) if gold in ids else None
        for t in row.nonzero().flatten().tolist():
            assert int(batch.sense_cand_gold[i, t]) == expected_gold_idx
            mask_t = batch.sense_cand_mask[i, t].bool()
            priors_t = batch.sense_cand_prior[i, t][mask_t]
            assert torch.allclose(priors_t.sum(), torch.tensor(1.0), atol=1e-4)
            # MFS-rank ordering preserved in the packed tensor too
            assert bool((priors_t[:-1] >= priors_t[1:]).all())
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# 3. Byte-identity regression: ambiguity-free episodes are UNAFFECTED by M54
#    (the load-bearing test).
# ---------------------------------------------------------------------------
def test_batch_identity_regression_no_ambiguity_episodes():
    from nsm_ct.episode import CurriculumGenerator

    parser, resolver, codec, _amb_eps = _amb_parser_env(n=5)
    eps = CurriculumGenerator(max_level=8, seed=1).generate(30)
    batch = build_clause_batch(eps, parser, resolver, codec)

    # the whole point of the sense_cand_* fields defaulting to None: nothing
    # new is allocated when there is no homograph episode to train on.
    assert batch.sense_cand_entity is None
    assert batch.sense_cand_mask is None
    assert batch.sense_cand_prior is None
    assert batch.sense_cand_context is None
    assert batch.sense_cand_gold is None
    # and M53's own cand_* guarantee (no pronoun episodes here either) holds too
    assert batch.cand_entity is None

    batch2 = build_clause_batch(eps, parser, resolver, codec)
    assert torch.equal(batch.entity, batch2.entity)
    assert torch.equal(batch.relation, batch2.relation)
    assert torch.equal(batch.value, batch2.value)
    assert torch.equal(batch.pred, batch2.pred)
    assert torch.equal(batch.is_q, batch2.is_q)
    assert torch.equal(batch.mask, batch2.mask)
    assert torch.equal(batch.options, batch2.options)
    assert torch.equal(batch.answer, batch2.answer)


def test_to_and_subset_preserve_none_sense_cand_fields():
    from nsm_ct.episode import CurriculumGenerator

    parser, resolver, codec, _amb_eps = _amb_parser_env(n=5)
    eps = CurriculumGenerator(max_level=6, seed=3).generate(10)
    batch = build_clause_batch(eps, parser, resolver, codec)
    assert batch.to("cpu").sense_cand_entity is None
    assert batch.subset(torch.tensor([0, 1])).sense_cand_entity is None


def test_to_and_subset_carry_sense_cand_fields_when_present():
    parser, resolver, codec, eps = _amb_parser_env(n=20)
    batch = build_clause_batch(eps, parser, resolver, codec)
    moved = batch.to("cpu")
    assert moved.sense_cand_entity is not None
    assert torch.equal(moved.sense_cand_gold, batch.sense_cand_gold)
    sub = batch.subset(torch.tensor([0, 1, 2]))
    assert sub.sense_cand_entity is not None
    assert sub.sense_cand_entity.shape[0] == 3
    assert torch.equal(sub.sense_cand_gold, batch.sense_cand_gold[[0, 1, 2]])


# ---------------------------------------------------------------------------
# 4. Both kinds in one batch/episode set: entity (pronoun) + sense (ambiguity).
# ---------------------------------------------------------------------------
def test_batch_and_collapse_handle_both_entity_and_sense_candidates():
    parser, resolver, codec, amb_eps, pron_eps = _mixed_parser_env()
    eps = amb_eps + pron_eps
    batch = build_clause_batch(eps, parser, resolver, codec)
    assert batch.cand_entity is not None
    assert batch.sense_cand_entity is not None

    # a step never carries BOTH kinds of candidate at once in this curriculum
    ent_has = batch.cand_mask.sum(-1) > 0
    sense_has = batch.sense_cand_mask.sum(-1) > 0
    assert not bool((ent_has & sense_has).any())

    model = ClauseReactor(dim=codec.dim, hidden=32,
                           resolver=make_resolver("A", codec.dim, 32),
                           sense_resolver=make_sense_resolver("A", codec.dim, 32))
    model.train()
    out = model(batch)
    assert "resolver_logits" in out and "sense_resolver_logits" in out
    loss = F.cross_entropy(out["answer_logits"], batch.answer)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.resolver.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.sense_resolver.parameters())


def test_track_b_shares_one_instance_across_both_slots():
    """Track B's whole argument: ONE SharedScorer, same weights, used for
    both entity and sense collapse in the SAME forward pass."""
    parser, resolver, codec, amb_eps, pron_eps = _mixed_parser_env()
    eps = amb_eps + pron_eps
    batch = build_clause_batch(eps, parser, resolver, codec)
    shared = make_resolver("B", codec.dim, 32)
    model = ClauseReactor(dim=codec.dim, hidden=32, resolver=shared, sense_resolver=shared)
    assert model.resolver is model.sense_resolver
    n_model = sum(p.numel() for p in model.parameters())
    n_gru_etc = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("resolver", "sense_resolver")))
    n_shared = sum(p.numel() for p in shared.parameters())
    assert n_model == n_gru_etc + n_shared    # shared params counted ONCE, not twice
    model.train()
    out = model(batch)
    F.cross_entropy(out["answer_logits"], batch.answer).backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in shared.parameters())


# ---------------------------------------------------------------------------
# 5. SenseHead + SharedScorer contract tests on sense-shaped tensors
#    (synthetic, no parser -- mirrors tests/test_resolver.py's style).
# ---------------------------------------------------------------------------
def test_sense_head_shape_and_params_under_20k():
    b, C, d, hidden = 6, 5, 32, 32
    head = SenseHead(d, hidden)
    cand = torch.randn(b, C, d)
    feat = torch.zeros(b, FEATURE_DIM)
    prior = torch.rand(b, C)
    mask = torch.ones(b, C)
    mem_read = torch.randn(b, C, d)
    state = torch.randn(b, 64)
    logits = head(cand, feat, prior, mask, mem_read, state)
    assert logits.shape == (b, C)
    n = sum(p.numel() for p in head.parameters())
    assert n < 20_000, f"SenseHead: {n} params"


def test_sense_head_ignores_feature_prior_and_state():
    """Literal M34-chooser parity: only candidate + context (mem_read) drive
    the score -- changing feature/prior/state must not move the logits."""
    torch.manual_seed(0)
    head = SenseHead(16, 16)
    b, C, d = 3, 4, 16
    cand = torch.randn(b, C, d)
    mem_read = torch.randn(b, C, d)
    l1 = head(cand, torch.zeros(b, FEATURE_DIM), torch.rand(b, C), torch.ones(b, C), mem_read, torch.randn(b, 8))
    l2 = head(cand, torch.randn(b, FEATURE_DIM), torch.rand(b, C), torch.ones(b, C), mem_read, torch.randn(b, 8))
    assert torch.allclose(l1, l2)


def test_shared_scorer_accepts_sense_shaped_inputs_unchanged():
    """SharedScorer's own code is untouched; this proves it "just works" when
    fed the sense-adapted tensors clause_reactor._collapse constructs: sense
    vectors in the cand_entity slot, a zero mention-feature placeholder, a
    real MFS-rank prior, and an already-broadcast mem_read+context."""
    b, C, d, hidden = 4, 3, 32, 64
    scorer = SharedScorer(d, hidden)
    cand = torch.randn(b, C, d)                    # sense vectors, not entity atoms
    feat0 = torch.zeros(b, FEATURE_DIM)             # thinnest projection
    prior = torch.rand(b, C)
    mask = torch.ones(b, C)
    ctx = torch.randn(b, 1, d).expand(b, C, d).contiguous()   # broadcast context
    state = torch.randn(b, hidden)
    logits = scorer(cand, feat0, prior, mask, ctx, state)
    assert logits.shape == (b, C)


@pytest.mark.parametrize("track,cls", [("A", SenseHead), ("B", SharedScorer)])
def test_make_sense_resolver_dispatch(track, cls):
    assert isinstance(make_sense_resolver(track, 16, 32), cls)
    assert isinstance(make_sense_resolver(track.lower(), 16, 32), cls)


def test_make_sense_resolver_rejects_unknown_track():
    with pytest.raises(ValueError):
        make_sense_resolver("C", 16, 32)


# ---------------------------------------------------------------------------
# 6. Synthetic sense-collapse toy batch (no parser) + aux-loss / margin tests
#    mirroring tests/test_resolver.py's own synthetic-tensor coverage.
# ---------------------------------------------------------------------------
def _toy_batch_with_sense_candidates(b=6, d=16, K=4, S=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    opts = F.normalize(torch.randn(b, K, d, generator=g), dim=-1)
    ans = torch.randint(0, K, (b,), generator=g)

    sense_vecs = F.normalize(torch.randn(b, S, d, generator=g), dim=-1)
    gold_idx = torch.randint(0, S, (b,), generator=g)
    sense_vecs[torch.arange(b), gold_idx] = opts[torch.arange(b), ans]  # gold candidate IS the answer's meaning

    hom_ent = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    sense_rel = F.normalize(torch.randn(1, d, generator=g), dim=-1).expand(b, d).contiguous()
    prd = F.normalize(torch.randn(b, d, generator=g), dim=-1)

    T = 2   # one sense (write) step + one question step, same address
    entity = torch.zeros(b, T, d); relation = torch.zeros(b, T, d); value = torch.zeros(b, T, d)
    pred = torch.zeros(b, T, d); is_q = torch.zeros(b, T); mask = torch.ones(b, T)
    sense_t = 0
    entity[:, sense_t], relation[:, sense_t], pred[:, sense_t] = hom_ent, sense_rel, prd
    value[:, sense_t] = sense_vecs[torch.arange(b), gold_idx]   # M54 placeholder: gold-bound
    q_t = 1
    entity[:, q_t], relation[:, q_t], is_q[:, q_t] = hom_ent, sense_rel, 1.0

    sense_cand_entity = torch.zeros(b, T, S, d)
    sense_cand_mask = torch.zeros(b, T, S)
    sense_cand_prior = torch.zeros(b, T, S)
    sense_cand_context = torch.zeros(b, T, d)
    sense_cand_gold = torch.full((b, T), -1, dtype=torch.long)
    sense_cand_entity[:, sense_t] = sense_vecs
    sense_cand_mask[:, sense_t] = 1.0
    sense_cand_prior[:, sense_t] = 1.0 / S
    sense_cand_gold[:, sense_t] = gold_idx

    batch = ClauseBatch(entity, relation, value, pred, is_q, mask, opts, ans,
                         sense_cand_entity=sense_cand_entity, sense_cand_mask=sense_cand_mask,
                         sense_cand_prior=sense_cand_prior, sense_cand_context=sense_cand_context,
                         sense_cand_gold=sense_cand_gold)
    return batch, sense_t


@pytest.mark.parametrize("track", ["A", "B"])
def test_sense_soft_collapse_shapes_and_gradients_flow(track):
    torch.manual_seed(0)
    batch, sense_t = _toy_batch_with_sense_candidates(b=5, d=16, K=4, S=3, seed=1)
    model = ClauseReactor(dim=16, sense_resolver=make_sense_resolver(track, 16, 128))
    model.train()
    out = model(batch)
    b, T, d = batch.entity.shape
    S = batch.sense_cand_entity.shape[2]
    assert out["sense_resolver_logits"].shape == (b, T, S)
    assert out["sense_resolver_margin"].shape == (b, T)
    assert "resolver_logits" not in out    # no entity candidates in this toy batch

    loss = F.cross_entropy(out["answer_logits"], batch.answer)
    loss.backward()
    grads = [p.grad for p in model.sense_resolver.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads), \
        "soft collapse should let the answer loss reach the sense_resolver's parameters"


def test_sense_aux_loss_computable_from_sense_resolver_logits_and_cand_gold():
    torch.manual_seed(0)
    batch, sense_t = _toy_batch_with_sense_candidates(b=6, d=16, K=4, S=3, seed=3)
    model = ClauseReactor(dim=16, sense_resolver=make_sense_resolver("B", 16, 128))
    out = model(batch)
    mask = batch.sense_cand_gold >= 0
    assert int(mask.sum()) == batch.entity.shape[0]   # exactly one sense step per row
    sel_logits = out["sense_resolver_logits"][mask]
    sel_gold = batch.sense_cand_gold[mask]
    aux_loss = F.cross_entropy(sel_logits, sel_gold)
    assert aux_loss.dim() == 0 and torch.isfinite(aux_loss)
    total = F.cross_entropy(out["answer_logits"], batch.answer) + 0.5 * aux_loss
    total.backward()   # both the answer path and the sense_resolver aux path differentiable together


class _FixedSenseLogitResolver(Resolver):
    """Test-only stub, mirrors tests/test_resolver.py's _FixedLogitResolver."""

    def __init__(self, row):
        super().__init__()
        self._row = torch.tensor(row, dtype=torch.float32)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state):
        b = cand_entity.shape[0]
        return self._row.unsqueeze(0).expand(b, -1).clone()


def test_sense_margin_matches_hand_computed_top1_minus_top2():
    torch.manual_seed(0)
    batch, sense_t = _toy_batch_with_sense_candidates(b=3, d=16, K=4, S=3, seed=4)
    model = ClauseReactor(dim=16, sense_resolver=_FixedSenseLogitResolver([5.0, 0.0, -5.0]))
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert torch.allclose(out["sense_resolver_margin"][:, sense_t], torch.full((3,), 5.0), atol=1e-4)
    other_steps = [t for t in range(batch.entity.shape[1]) if t != sense_t]
    assert torch.equal(out["sense_resolver_margin"][:, other_steps],
                        torch.zeros(3, len(other_steps)))


def test_sense_margin_respects_padding_mask():
    torch.manual_seed(0)
    batch, sense_t = _toy_batch_with_sense_candidates(b=2, d=16, K=4, S=3, seed=5)
    batch.sense_cand_mask[:, sense_t, 2] = 0.0   # candidate slot 2 is padding
    model = ClauseReactor(dim=16, sense_resolver=_FixedSenseLogitResolver([0.0, 1.0, 100.0]))
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert torch.allclose(out["sense_resolver_margin"][:, sense_t], torch.full((2,), 1.0), atol=1e-4)


def test_sense_collapse_uses_candidate_vector_directly_not_memory_query():
    """The defining semantic difference from entity collapse: a sense
    candidate's resolved value is the CANDIDATE VECTOR ITSELF (weighted by
    the hard/soft collapse), never a memory-query result -- there is no
    entity address a meaning vector can stand in for."""
    torch.manual_seed(0)
    batch, sense_t = _toy_batch_with_sense_candidates(b=4, d=16, K=4, S=3, seed=6)
    model = ClauseReactor(dim=16, sense_resolver=_FixedSenseLogitResolver([10.0, -10.0, -10.0]))
    model.eval()
    with torch.no_grad():
        memory = em.init_memory(4, 16, torch.device("cpu"))
        e0, r0 = batch.entity[:, sense_t], batch.relation[:, sense_t]
        mem_read0 = em.query(memory, e0, r0)
        v_out, *_ = model._collapse(memory, torch.zeros(4, model.gru.hidden_size),
                                     mem_read0, r0, batch.value[:, sense_t], batch, sense_t)
    expected = batch.sense_cand_entity[:, sense_t, 0]     # candidate 0 wins (logit 10.0)
    assert torch.allclose(v_out, expected, atol=1e-5)


@pytest.mark.parametrize("track", ["A", "B"])
def test_sense_determinism_same_seed_same_outputs(track):
    batch, _ = _toy_batch_with_sense_candidates(b=5, d=16, K=4, S=3, seed=9)

    torch.manual_seed(42)
    m1 = ClauseReactor(dim=16, sense_resolver=make_sense_resolver(track, 16, 128))
    torch.manual_seed(42)
    m2 = ClauseReactor(dim=16, sense_resolver=make_sense_resolver(track, 16, 128))
    for p1, p2 in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(p1, p2)

    m1.eval(); m2.eval()
    with torch.no_grad():
        out1, out2 = m1(batch), m2(batch)
    for k in out1:
        assert torch.equal(out1[k], out2[k])


# ---------------------------------------------------------------------------
# 7. No-sense-resolver byte identity (mirrors test_resolver.py's own
#    resolver=None regression, for the sense_resolver=None case).
# ---------------------------------------------------------------------------
def test_no_sense_resolver_leaves_forward_untouched_even_with_sense_data():
    """A sense_resolver CAN be omitted even when the batch DOES carry sense
    candidate data -- the sense-collapse branch must never fire, matching
    resolver=None's existing contract for entity candidates."""
    torch.manual_seed(0)
    batch, sense_t = _toy_batch_with_sense_candidates(b=4, d=16, K=4, S=2, seed=7)
    torch.manual_seed(5)
    plain = ClauseReactor(dim=16)   # resolver=None, sense_resolver=None (both defaults)
    out = plain(batch)
    assert "sense_resolver_logits" not in out
    assert "resolver_logits" not in out
    # the sense step's value must stay exactly the PLACEHOLDER-bound one (never collapsed)
    assert torch.equal(batch.value[:, sense_t], batch.sense_cand_entity[
        torch.arange(4), sense_t, batch.sense_cand_gold[:, sense_t]])
