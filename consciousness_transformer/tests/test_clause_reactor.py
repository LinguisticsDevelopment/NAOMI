"""Tests for the token-free clause reactor + order-3 entity memory."""

import pytest
import torch
import torch.nn.functional as F

from nsm_ct import entity_memory as em
from nsm_ct.clause_reactor import ClauseBatch, ClauseReactor, build_clause_batch


def test_entity_memory_gated_overwrite():
    """gate=1 writes/updates the slot; gate=0 leaves it; a second write overwrites."""
    b, d = 4, 16
    M = em.init_memory(b, d, torch.device("cpu"))
    e = F.normalize(torch.randn(b, d), dim=-1)
    r = F.normalize(torch.randn(b, d), dim=-1)
    v1 = torch.randn(b, d)
    v2 = torch.randn(b, d)
    M = em.write(M, e, r, v1, torch.ones(b))
    q = em.query(M, e, r)
    assert F.cosine_similarity(q, v1).mean() > 0.9            # recovered
    M0 = em.write(M, e, r, v2, torch.zeros(b))               # gate 0 -> unchanged
    assert torch.allclose(em.query(M0, e, r), q, atol=1e-5)
    M2 = em.write(M, e, r, v2, torch.ones(b))                # overwrite
    q2 = em.query(M2, e, r)
    assert (F.cosine_similarity(q2, v2) > F.cosine_similarity(q2, v1)).all()


def test_entity_memory_vote_vs_update():
    """overwrite=0 accumulates (vote: majority value wins); overwrite=gate replaces."""
    b, d = 3, 16
    M = em.init_memory(b, d, torch.device("cpu"))
    e = F.normalize(torch.randn(b, d), dim=-1)
    r = F.normalize(torch.randn(b, d), dim=-1)
    a = F.normalize(torch.randn(b, d), dim=-1)
    z = F.normalize(torch.randn(b, d), dim=-1)
    one, zero = torch.ones(b), torch.zeros(b)
    # vote: write a twice, b once, all additive -> a wins
    Mv = em.write(M, e, r, a, one, overwrite=zero)
    Mv = em.write(Mv, e, r, a, one, overwrite=zero)
    Mv = em.write(Mv, e, r, z, one, overwrite=zero)
    q = em.query(Mv, e, r)
    assert (F.cosine_similarity(q, a) > F.cosine_similarity(q, z)).all()
    # update: write a then b with full overwrite -> b wins (recency)
    Mu = em.write(M, e, r, a, one, overwrite=one)
    Mu = em.write(Mu, e, r, z, one, overwrite=one)
    qu = em.query(Mu, e, r)
    assert (F.cosine_similarity(qu, z) > F.cosine_similarity(qu, a)).all()


def _toy_batch(b=8, d=16, K=4, seed=0):
    """One statement + one question; the statement's value IS the gold option."""
    g = torch.Generator().manual_seed(seed)
    opts = F.normalize(torch.randn(b, K, d, generator=g), dim=-1)
    ans = torch.randint(0, K, (b,), generator=g)
    ent = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    rel = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    val = opts[torch.arange(b), ans]                          # statement value = answer
    prd = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    entity = torch.stack([ent, ent], dim=1)                  # same entity both steps
    relation = torch.stack([rel, rel], dim=1)
    value = torch.stack([val, torch.zeros(b, d)], dim=1)     # question carries no value
    pred = torch.stack([prd, prd], dim=1)
    is_q = torch.tensor([[0.0, 1.0]]).repeat(b, 1)
    mask = torch.ones(b, 2)
    return ClauseBatch(entity, relation, value, pred, is_q, mask, opts, ans)


def test_reactor_forward_shapes():
    batch = _toy_batch()
    out = ClauseReactor(dim=16)(batch)
    assert out["answer_logits"].shape == (8, 4)
    assert out["response"].shape == (8, 16)
    assert out["respond_gates"].shape == (8, 2)


def test_reactor_learns_to_read_memory_and_answer():
    """It learns: write the fact, respond at the question, generate r ≈ the stored
    value → pick the right option. (No tokens; only the reaction policy trains.)"""
    torch.manual_seed(0)
    batch = _toy_batch(b=16, d=24, K=4)
    model = ClauseReactor(dim=24)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    init = float((model(batch)["answer_logits"].argmax(-1) == batch.answer).float().mean())
    for _ in range(300):
        out = model(batch)
        loss = F.cross_entropy(out["answer_logits"], batch.answer)
        opt.zero_grad(); loss.backward(); opt.step()
    final = float((model(batch)["answer_logits"].argmax(-1) == batch.answer).float().mean())
    assert final > max(0.7, init)                            # learns the reaction
    # respond mass concentrates on the question step
    assert float((model(batch)["respond_gates"] * batch.is_q).sum(1).mean()) > 0.5


def test_decide_truth_head_and_coord_channel():
    """The reactor exposes the decide_truth head and accepts the coord channel."""
    model = ClauseReactor(dim=16, hidden=32)
    assert model.decide_truth.in_features == 32 + 16 and model.decide_truth.out_features == 1
    assert model.gru.input_size == 6 * 16            # +coord channel
    batch = _toy_batch(b=4, d=16, K=4)
    b, T, d = batch.entity.shape
    batch.coord = F.normalize(torch.randn(b, T, d), dim=-1)   # a non-trivial coord
    out = model(batch)
    assert out["answer_logits"].shape == (4, 4)      # forward consumes coord cleanly


def _logic_env():
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.episode import CurriculumGenerator
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.tokenizer import SimpleTokenizer
    from nsm_ct.tpr import TPRCodec

    eps = CurriculumGenerator(max_level=8, seed=0).generate(16)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return parser, NSMMeaningResolver(), TPRCodec(dim=48)


def test_reactor_learns_disjunction_resolution_emergently():
    """Answer-only supervision drives decide_truth: on "A or B" + "not A", the
    reactor learns to SUBTRACT the refuted value and answer the survivor — no aux
    truth loss. Overfit a few resolved-OR / negation episodes to acc 1.0."""
    from nsm_ct.episode import CurriculumGenerator

    parser, resolver, codec = _logic_env()
    gen = CurriculumGenerator(max_level=8, seed=7)
    # the logic-necessary episodes: resolved disjunction (L7) and negation-removal (L8)
    pool = gen.generate(300)
    eps = [e for e in pool if (e.level == 7 and e.meta.get("resolved")) or e.level == 8][:12]
    assert eps
    batch = build_clause_batch(eps, parser, resolver, codec)
    torch.manual_seed(0)
    model = ClauseReactor(dim=48)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(400):
        out = model(batch)
        loss = F.cross_entropy(out["answer_logits"], batch.answer)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = float((model(batch)["answer_logits"].argmax(-1) == batch.answer).float().mean())
    assert acc > 0.9          # the answer loss alone teaches the truth policy
