"""Stage 5 — ClausePsyche generates a clause meaning-object (Frobenius + decode)."""

import torch
import torch.nn.functional as F

from nsm_ct.clause_psyche import (
    ClausePsyche,
    clause_decode_accuracy,
    compute_clause_psyche_losses,
    gold_matrix,
)
from nsm_ct.clause_reactor import ClauseBatch
from nsm_ct.tpr import TPRCodec


def _toy_batch(b=8, d=16, K=4, seed=0):
    """One statement + one question; the statement's value IS the gold option."""
    g = torch.Generator().manual_seed(seed)
    opts = F.normalize(torch.randn(b, K, d, generator=g), dim=-1)
    ans = torch.randint(0, K, (b,), generator=g)
    ent = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    rel = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    val = opts[torch.arange(b), ans]
    prd = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    entity = torch.stack([ent, ent], dim=1)
    relation = torch.stack([rel, rel], dim=1)
    value = torch.stack([val, torch.zeros(b, d)], dim=1)
    pred = torch.stack([prd, prd], dim=1)
    is_q = torch.tensor([[0.0, 1.0]]).repeat(b, 1)
    mask = torch.ones(b, 2)
    return ClauseBatch(entity, relation, value, pred, is_q, mask, opts, ans)


def test_forward_shapes():
    d = 16
    batch = _toy_batch(b=8, d=d, K=4)
    out = ClausePsyche(TPRCodec(dim=d), hidden=32)(batch)
    assert out["matrix"].shape == (8, d, d)
    assert out["states"].shape == (8, 2, 32)
    assert out["op_logits"].shape == (8, 2, 5)
    assert out["answer_logits"].shape == (8, 4)


def test_consciousness_state_is_carried_and_updated():
    d = 16
    batch = _toy_batch(b=8, d=d)
    out = ClausePsyche(TPRCodec(dim=d), hidden=32)(batch)
    states = out["states"]
    # the spotlight changes from step 0 to step 1 (it is a maintained, evolving state)
    assert (states[:, 1] - states[:, 0]).abs().mean() > 1e-4


def test_overfits_to_generate_the_correct_clause():
    torch.manual_seed(0)
    d = 24
    batch = _toy_batch(b=16, d=d, K=4)
    model = ClausePsyche(TPRCodec(dim=d), hidden=64)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    with torch.no_grad():
        init_acc = clause_decode_accuracy(model(batch), batch)
        init_frob = float(compute_clause_psyche_losses(model(batch), batch, model)["frobenius"])
    for _ in range(300):
        out = model(batch)
        loss = compute_clause_psyche_losses(out, batch, model)["total"]
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        out = model(batch)
        final_acc = clause_decode_accuracy(out, batch)
        final_frob = float(compute_clause_psyche_losses(out, batch, model)["frobenius"])
    assert final_acc > max(0.7, init_acc)          # generates the right place
    assert final_frob < init_frob                  # meaning-object approaches the gold matrix
    # the generated MATRIX itself decodes: unbind the PLACE role -> correct option
    place = torch.einsum("d,bde->be", model.place_role, out["matrix"])
    logits = torch.einsum("bd,bkd->bk", F.normalize(place, dim=-1),
                          F.normalize(batch.options, dim=-1))
    assert float((logits.argmax(-1) == batch.answer).float().mean()) > 0.7


def _answerable_mix_batch(b=16, d=24, K=4, seed=1):
    """A toy where half the episodes are unanswerable (abstain target)."""
    batch = _toy_batch(b=b, d=d, K=K, seed=seed)
    ok = torch.ones(b)
    ok[: b // 2] = 0.0          # first half: should abstain
    batch.answerable = ok
    return batch


def _two_hop_batch(b=64, d=32, K=4, seed=0):
    """A chain task: (A is_a B), (B can C) + a distractor pair; ask (A can ?) -> C.
    The answer C is only reachable by composing A->B->C (retrieval alone fails)."""
    g = torch.Generator().manual_seed(seed)
    n = lambda *s: F.normalize(torch.randn(*s, generator=g), dim=-1)
    R1, R2 = n(d), n(d)                                   # IS_A, CAN (shared relations)
    A, B = n(b, d), n(b, d)                               # subject, intermediate
    A2, B2 = n(b, d), n(b, d)                             # distractor chain
    opts = n(b, K, d)
    ans = torch.randint(0, K, (b,), generator=g)
    C = opts[torch.arange(b), ans]
    C2 = opts[torch.arange(b), (ans + 1) % K]             # distractor ability
    z = torch.zeros(b, d)
    R1b, R2b = R1.expand(b, d), R2.expand(b, d)
    pis, pq = n(d).expand(b, d), n(d).expand(b, d)
    # steps: (A is_a B), (A2 is_a B2), (B can C), (B2 can C2), Q:(A can ?)
    entity = torch.stack([A, A2, B, B2, A], dim=1)
    relation = torch.stack([R1b, R1b, R2b, R2b, R2b], dim=1)
    value = torch.stack([B, B2, C, C2, z], dim=1)
    pred = torch.stack([pis, pis, pis, pis, pq], dim=1)
    is_q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(b, 1)
    mask = torch.ones(b, 5)
    return ClauseBatch(entity, relation, value, pred, is_q, mask, opts, ans,
                       torch.zeros(b, 5, d), torch.ones(b))


def test_hops_extend_states_and_emit_abstain():
    d = 16
    batch = _toy_batch(b=8, d=d)
    out = ClausePsyche(TPRCodec(dim=d), hidden=32, hops=3)(batch)
    assert out["states"].shape == (8, 2 + 3, 32)         # T stream + K hops
    assert out["abstain_prob"].shape == (8,)


def test_abstain_head_learns_the_unanswerable_flag():
    torch.manual_seed(0)
    d = 24
    batch = _answerable_mix_batch(b=16, d=d)
    model = ClausePsyche(TPRCodec(dim=d), hidden=48, hops=2)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    for _ in range(200):
        loss = compute_clause_psyche_losses(model(batch), batch, model)["total"]
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = model(batch)["abstain_prob"]
    should = batch.answerable == 0
    assert float(p[should].mean()) > float(p[~should].mean()) + 0.3   # abstains when it should


def test_hop_loop_learns_two_hop_composition():
    # The looping model learns a 2-hop chain (A is_a B; B can C -> A can C). (A short
    # chain like this can also be carried single-pass by the GRU; the hop loop's real
    # payoff is deeper chains + explicit control — that comparison is measured on the
    # curriculum in train_clause_psyche, not asserted here.)
    torch.manual_seed(0)
    d = 32
    batch = _two_hop_batch(b=64, d=d)
    model = ClausePsyche(TPRCodec(dim=d), hidden=96, hops=3)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
    for _ in range(600):
        loss = compute_clause_psyche_losses(model(batch), batch, model)["total"]
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        assert clause_decode_accuracy(model(batch), batch) > 0.7


def test_gold_matrix_is_recovered_by_its_own_decode():
    # sanity: the gold assembly decodes (unbind place role) back to the answer option
    d = 32
    batch = _toy_batch(b=8, d=d)
    model = ClausePsyche(TPRCodec(dim=d), hidden=16)
    M = gold_matrix(model, batch)
    place = torch.einsum("d,bde->be", model.place_role, M)  # unbind PLACE role
    logits = torch.einsum("bd,bkd->bk", F.normalize(place, dim=-1),
                          F.normalize(batch.options, dim=-1))
    assert float((logits.argmax(-1) == batch.answer).float().mean()) > 0.99
