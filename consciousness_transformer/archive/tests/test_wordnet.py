"""Tests for nsm_ct.wordnet and WordNetSenseInventory.

Real-WordNet assertions are gated behind ``wordnet_available()`` so that CI
environments without the NLTK corpus still pass.
"""

from __future__ import annotations

import pytest

from nsm_ct import wordnet as wn_mod
from nsm_ct.wordnet import wordnet_available, senses, relations
from nsm_ct.wsd import Sense, WordNetSenseInventory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

wn_available = wordnet_available()


# ---------------------------------------------------------------------------
# wordnet.senses / wordnet.relations
# ---------------------------------------------------------------------------


def test_senses_returns_empty_when_unavailable(monkeypatch):
    """Simulate WordNet being unavailable: senses() returns []."""
    monkeypatch.setattr(wn_mod, "_available", False)
    result = senses("bank")
    assert result == []


def test_relations_returns_none_when_unavailable(monkeypatch):
    """Simulate WordNet being unavailable: relations() returns None."""
    monkeypatch.setattr(wn_mod, "_available", False)
    result = relations("bank.n.01")
    assert result is None


@pytest.mark.skipif(not wn_available, reason="WordNet corpus not installed")
def test_senses_bank_count_and_structure():
    """bank has >= 2 senses and each entry has required keys."""
    result = senses("bank")
    assert len(result) >= 2, f"Expected >= 2 senses for 'bank', got {len(result)}"
    for entry in result:
        assert "sense_id" in entry
        assert "gloss" in entry
        assert "pos" in entry
        assert "lemmas" in entry
        assert "hypernyms" in entry
        assert "hyponyms" in entry
        assert entry["gloss"], "gloss must be non-empty"
        assert entry["sense_id"], "sense_id must be non-empty"


@pytest.mark.skipif(not wn_available, reason="WordNet corpus not installed")
def test_senses_distinct_ids():
    """Sense IDs for 'bank' must all be distinct."""
    result = senses("bank")
    ids = [entry["sense_id"] for entry in result]
    assert len(ids) == len(set(ids)), "Sense IDs should be unique"


@pytest.mark.skipif(not wn_available, reason="WordNet corpus not installed")
def test_relations_known_synset():
    """relations() returns a dict with hypernyms / hyponyms for a known synset."""
    # bank.n.01 = "sloping land (especially the slope beside a body of water)"
    result = relations("bank.n.01")
    assert result is not None
    assert "hypernyms" in result
    assert "hyponyms" in result
    assert isinstance(result["hypernyms"], list)
    assert isinstance(result["hyponyms"], list)
    # bank.n.01 should have at least one hypernym in WordNet
    assert len(result["hypernyms"]) >= 1, (
        f"Expected hypernyms for bank.n.01, got: {result}"
    )


@pytest.mark.skipif(not wn_available, reason="WordNet corpus not installed")
def test_relations_unknown_synset_returns_none():
    """relations() returns None for a synset name that does not exist."""
    result = relations("zzz_nonexistent_word.n.99")
    assert result is None


# ---------------------------------------------------------------------------
# WordNetSenseInventory
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not wn_available, reason="WordNet corpus not installed")
def test_inventory_bank_count_and_glosses():
    """WordNetSenseInventory().senses('bank') should return >= 2 Senses."""
    inv = WordNetSenseInventory()
    result = inv.senses("bank")
    assert len(result) >= 2, f"Expected >= 2 senses, got {len(result)}"
    # All results must be Sense objects with non-empty glosses and distinct IDs
    for s in result:
        assert isinstance(s, Sense)
        assert s.gloss, f"Empty gloss for sense {s.sense_id}"
        assert s.sense_id, "sense_id must be non-empty"
        assert s.word == "bank"
    ids = [s.sense_id for s in result]
    assert len(ids) == len(set(ids)), "Sense IDs should be unique"


@pytest.mark.skipif(not wn_available, reason="WordNet corpus not installed")
def test_inventory_sense_has_empty_primes_and_none_meaning():
    """primes should be {} and meaning should be None (later-stage tasks)."""
    inv = WordNetSenseInventory()
    for s in inv.senses("bank"):
        assert s.primes == {}, "primes must be empty dict in this pass"
        assert s.meaning is None, "meaning must be None in this pass"


@pytest.mark.skipif(not wn_available, reason="WordNet corpus not installed")
def test_inventory_prime_vector_works_with_empty_primes():
    """prime_vector() must still return the correct-shaped array even with empty primes."""
    from nsm_ct.nsm_primes import NUM_PRIMES
    import numpy as np

    inv = WordNetSenseInventory()
    for s in inv.senses("bank"):
        vec = s.prime_vector()
        assert vec.shape == (NUM_PRIMES,)
        assert np.all(vec == 0.0)


def test_inventory_graceful_fallback_when_unavailable(monkeypatch):
    """When WordNet is unavailable, senses() returns a single generic Sense."""
    monkeypatch.setattr(wn_mod, "_available", False)
    inv = WordNetSenseInventory()
    result = inv.senses("bank")
    assert len(result) == 1
    assert isinstance(result[0], Sense)
    assert result[0].word == "bank"
    # Should NOT raise; the generic sense is well-formed
    result[0].prime_vector()


@pytest.mark.skipif(not wn_available, reason="WordNet corpus not installed")
def test_inventory_unknown_word_fallback():
    """A word with no WordNet senses still returns a non-empty list (fallback)."""
    inv = WordNetSenseInventory()
    result = inv.senses("zzz_nonexistent_xkqv")
    assert len(result) >= 1
    assert isinstance(result[0], Sense)
