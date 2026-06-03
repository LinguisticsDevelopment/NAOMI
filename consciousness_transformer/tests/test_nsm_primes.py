"""Tests for the NSM prime inventory."""

from nsm_ct.nsm_primes import (
    NUM_PRIMES,
    PRIME_NAMES,
    PRIMES,
    PRIMES_BY_NAME,
    PrimeCategory,
    prime_index,
    primes_in_category,
)


def test_inventory_size_is_canonical():
    # The canonical NSM inventory is ~65 primes.
    assert 60 <= NUM_PRIMES <= 70
    assert len(PRIMES) == NUM_PRIMES == len(PRIME_NAMES)


def test_names_are_unique():
    assert len(set(PRIME_NAMES)) == len(PRIME_NAMES)


def test_every_category_is_populated():
    for category in PrimeCategory:
        assert primes_in_category(category), f"empty category: {category}"


def test_lookup_and_index_round_trip():
    for i, name in enumerate(PRIME_NAMES):
        assert prime_index(name) == i
        assert PRIMES_BY_NAME[name].name == name


def test_known_primes_present():
    # Spot-check a few primes from different categories.
    for name in ["I", "YOU", "GOOD", "BAD", "THINK", "BECAUSE", "NOT", "WHEN", "WHERE"]:
        assert name in PRIMES_BY_NAME


def test_exponents_nonempty():
    assert all(p.exponent for p in PRIMES)
