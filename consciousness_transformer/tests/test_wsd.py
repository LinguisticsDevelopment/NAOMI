"""Tests for NSM-grounded, coherence-driven WSD."""

import torch
import torch.nn.functional as F

from nsm_ct.nsm_primes import NUM_PRIMES
from nsm_ct.wsd import (
    IterativeSenseResolver,
    MockSenseInventory,
    SenseResolver,
    WSDModule,
)


# -- inventory ---------------------------------------------------------------
def test_inventory_returns_candidates_and_fallback():
    inv = MockSenseInventory()
    assert len(inv.senses("bank")) == 2
    assert inv.is_ambiguous("bank")
    # unknown word -> a single generic sense (pipeline never empty)
    fallback = inv.senses("zxqv")
    assert len(fallback) == 1 and not inv.is_ambiguous("zxqv")


def test_sense_prime_vector_shape_and_grounding():
    inv = MockSenseInventory()
    river = inv.senses("bank")[0]
    vec = river.prime_vector()
    assert vec.shape == (NUM_PRIMES,)
    assert vec.sum() != 0  # the sense is grounded in some primes


# -- scorer ------------------------------------------------------------------
def test_wsd_module_shapes_and_context_sensitivity():
    torch.manual_seed(0)
    inv = MockSenseInventory()
    resolver = SenseResolver(inv, WSDModule(context_dim=8, hidden=16))
    words = ["bank", "bat", "spring"]
    c0 = torch.randn(3, 8)
    out0 = resolver.resolve(words, c0)
    assert out0["logits"].shape[0] == 3
    assert out0["sense_emb"].shape == (3, 16)
    assert len(out0["chosen"]) == 3
    # a different context generally yields different sense logits
    out1 = resolver.resolve(words, c0 + 5.0)
    assert not torch.allclose(out0["logits"], out1["logits"])


def test_wsd_module_differentiable():
    inv = MockSenseInventory()
    module = WSDModule(context_dim=8, hidden=16)
    sense_lists = [inv.senses("bank")]
    from nsm_ct.wsd import candidates_to_tensor

    vecs, mask = candidates_to_tensor(sense_lists)
    context = torch.randn(1, 8, requires_grad=True)
    logits, _ = module(context, vecs, mask)
    logits.sum().backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()


# -- iterative / coherence-driven re-evaluation ------------------------------
def test_iterative_reevaluates_when_incoherent():
    torch.manual_seed(0)
    # threshold > 1 can never be reached (coherence is a sigmoid in [0,1]),
    # forcing the resolver to use all hops and re-evaluate each pass.
    res = IterativeSenseResolver(MockSenseInventory(), context_dim=8, hidden=16, coherence_threshold=2.0)
    out = res.resolve_iterative(["bank", "bat"], torch.randn(2, 8), max_hops=3)
    assert out["hops"] == 3
    assert out["coherence"].shape == (2,)
    assert len(out["history"]) == 3
    # the state evolves across hops, so the sense scores are re-evaluated
    assert not torch.allclose(out["history"][0]["logits"], out["history"][-1]["logits"])
    # differentiable end-to-end
    out["logits"].sum().backward()


def test_iterative_halts_when_coherent():
    # threshold <= 0 is always satisfied -> halt after the first hop.
    res = IterativeSenseResolver(MockSenseInventory(), context_dim=8, hidden=16, coherence_threshold=-1.0)
    out = res.resolve_iterative(["bank"], torch.randn(1, 8), max_hops=5)
    assert out["hops"] == 1


def test_coherence_signal_is_learnable():
    """The coherence head can learn 'does this interpretation make sense?'.

    No gold sense labels are used; we train only the coherence head on a
    synthetic coherent/incoherent signal, which is what drives re-evaluation.
    """
    torch.manual_seed(0)
    res = IterativeSenseResolver(MockSenseInventory(), context_dim=8, hidden=16)
    n = 256
    state = torch.randn(n, 8)
    sense = torch.randn(n, 16)
    # "coherent" iff state and sense agree in sign on a shared axis.
    label = (state[:, 0] * sense[:, 0] > 0).float()

    opt = torch.optim.AdamW(res.coherence_head.parameters(), lr=0.05)
    for _ in range(250):
        p = res.coherence(state, sense)
        loss = F.binary_cross_entropy(p, label)
        opt.zero_grad()
        loss.backward()
        opt.step()

    acc = ((res.coherence(state, sense) > 0.5).float() == label).float().mean()
    assert acc > 0.9
