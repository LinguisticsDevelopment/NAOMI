"""M26 gate: grounded WordNet sense signatures (the M22->WSD bridge)."""

from __future__ import annotations

import numpy as np
import pytest

import torch

from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.wsd import (
    GroundedWordNetSenseInventory,
    IterativeSenseResolver,
    WSDModule,
    WordNetSenseInventory,
    candidates_to_tensor,
)
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 0 else 0.0


@wn_required
def test_grounded_signatures_are_nonzero_and_valid_primes():
    inv = GroundedWordNetSenseInventory()
    senses = inv.senses("bank")
    assert len(senses) > 1
    prime_set = set(PRIME_NAMES)
    for s in senses:
        # every prime key is a real canonical prime (prime_vector must not KeyError)
        assert set(s.primes).issubset(prime_set)
        v = s.prime_vector()
        assert v.shape == (len(PRIME_NAMES),)
    # at least the frequent senses carry a real (nonzero) grounded signature
    assert any(s.prime_vector().any() for s in senses)


@wn_required
def test_distinct_senses_get_distinct_signatures():
    inv = GroundedWordNetSenseInventory()
    senses = [s for s in inv.senses("bank") if s.prime_vector().any()]
    # a river/geography sense and a finance sense must not be identical vectors
    river = next(s for s in senses if "financial" not in s.gloss and "money" not in s.gloss)
    finance = next(s for s in senses if "financial" in s.gloss or "money" in s.gloss)
    assert not np.allclose(river.prime_vector(), finance.prime_vector())
    assert _cos(river.prime_vector(), finance.prime_vector()) < 0.9  # meaningfully distinct


@wn_required
def test_base_inventory_stays_stubbed_optin_preserved():
    # the grounding is opt-in: the base class still returns empty prime signatures
    assert WordNetSenseInventory().senses("bank")[0].primes == {}


@wn_required
def test_deterministic_and_unknown_degrades():
    inv = GroundedWordNetSenseInventory()
    a = inv.senses("spring")[0].prime_vector()
    b = inv.senses("spring")[0].prime_vector()
    assert np.allclose(a, b)                                   # deterministic
    unknown = inv.senses("zzqxwv")
    assert len(unknown) == 1 and unknown[0].primes == {}       # graceful degrade


# --- M26.1: the grounded signatures make the WSD machinery actually work -------

@wn_required
def test_grounded_signatures_let_the_module_discriminate_senses():
    """With a neutral context, sense logits vary ONLY because the sense signatures
    differ. Grounded signatures are distinct -> real spread; the empty-stub base
    inventory gives identical zero signatures -> no spread (cannot disambiguate)."""
    torch.manual_seed(0)
    mod = WSDModule(context_dim=32, hidden=64)
    ctx = torch.zeros(1, 32)  # neutral: isolates the sense-signature contribution

    def spread(inv):
        vecs, mask = candidates_to_tensor([inv.senses("bank")])
        logits, _ = mod(ctx, vecs, mask)
        real = logits[0][mask[0] > 0]
        return float(real.std())

    grounded_spread = spread(GroundedWordNetSenseInventory())
    base_spread = spread(WordNetSenseInventory())
    assert base_spread < 1e-5          # empty signatures -> senses indistinguishable
    assert grounded_spread > 1e-3      # grounded signatures -> senses discriminated


@wn_required
def test_context_steers_the_chosen_sense():
    """Different contexts pick different senses (context-dependent disambiguation),
    which is only possible because grounded signatures are discriminated."""
    torch.manual_seed(0)
    inv = GroundedWordNetSenseInventory()
    resolver = IterativeSenseResolver(inv, context_dim=32, hidden=64)
    torch.manual_seed(1)
    contexts = torch.randn(16, 32)
    out = resolver.resolve_iterative(["bank"] * 16, contexts, max_hops=3)
    chosen_ids = {s.sense_id for s in out["chosen"]}
    assert len(chosen_ids) > 1                          # context changes the pick


@wn_required
def test_iterative_resolver_runs_and_is_deterministic():
    torch.manual_seed(0)
    inv = GroundedWordNetSenseInventory()
    resolver = IterativeSenseResolver(inv, context_dim=32, hidden=64)
    ctx = torch.zeros(2, 32)
    out = resolver.resolve_iterative(["bank", "spring"], ctx, max_hops=3)
    assert 1 <= out["hops"] <= 3
    assert ((out["coherence"] >= 0) & (out["coherence"] <= 1)).all()
    assert len(out["chosen"]) == 2
    out2 = resolver.resolve_iterative(["bank", "spring"], ctx, max_hops=3)
    assert torch.allclose(out["logits"], out2["logits"])   # deterministic
