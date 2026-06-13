"""Tests for the token-free clause reactor + order-3 entity memory."""

import torch
import torch.nn.functional as F

from nsm_ct import entity_memory as em
from nsm_ct.clause_reactor import ClauseBatch, ClauseReactor


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


def _toy_batch(b=8, d=16, K=4, seed=0):
    """One statement + one question; the statement's value IS the gold option."""
    g = torch.Generator().manual_seed(seed)
    opts = F.normalize(torch.randn(b, K, d, generator=g), dim=-1)
    ans = torch.randint(0, K, (b,), generator=g)
    ent = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    rel = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    val = opts[torch.arange(b), ans]                          # statement value = answer
    entity = torch.stack([ent, ent], dim=1)                  # same entity both steps
    relation = torch.stack([rel, rel], dim=1)
    value = torch.stack([val, torch.zeros(b, d)], dim=1)     # question carries no value
    is_q = torch.tensor([[0.0, 1.0]]).repeat(b, 1)
    mask = torch.ones(b, 2)
    return ClauseBatch(entity, relation, value, is_q, mask, opts, ans)


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
    for _ in range(150):
        out = model(batch)
        loss = F.cross_entropy(out["answer_logits"], batch.answer)
        opt.zero_grad(); loss.backward(); opt.step()
    final = float((model(batch)["answer_logits"].argmax(-1) == batch.answer).float().mean())
    assert final > max(0.7, init)                            # learns the reaction
    # respond mass concentrates on the question step
    assert float((model(batch)["respond_gates"] * batch.is_q).sum(1).mean()) > 0.5
