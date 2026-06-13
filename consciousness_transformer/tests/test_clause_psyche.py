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
