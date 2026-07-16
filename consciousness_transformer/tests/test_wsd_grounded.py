"""M26 gate: grounded WordNet sense signatures (the M22->WSD bridge)."""

from __future__ import annotations

import numpy as np
import pytest

from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.wsd import GroundedWordNetSenseInventory, WordNetSenseInventory
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
