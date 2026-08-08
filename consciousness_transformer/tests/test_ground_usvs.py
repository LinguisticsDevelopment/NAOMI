"""Gates for the USVS artifact (M29): determinism, roundtrip, query sanity."""

import numpy as np
import pytest

from nsm_ct.ground.usvs import (
    DEFAULT_ANTONYM_TIERS, antonym_edges_tiered, build_usvs, load_usvs,
    save_usvs, sense_prime_weights,
)
from nsm_ct.ground.sense_graph import gloss_prime_weights
from nsm_ct.wordnet import wordnet_available

pytestmark = pytest.mark.skipif(not wordnet_available(), reason="needs WordNet")


@pytest.fixture(scope="module")
def small_usvs():
    return build_usvs(n_core=400, max_senses=800, depth=2)


def test_sense_prime_weights_matches_m22_grounding():
    """The cached reimplementation must be bit-equal to gloss_prime_weights."""
    for gloss in ("a domesticated carnivorous mammal",
                  "a financial institution that accepts deposits",
                  "move fast by using one's feet"):
        assert sense_prime_weights(gloss) == gloss_prime_weights(gloss)


def test_build_is_deterministic(small_usvs):
    u2 = build_usvs(n_core=400, max_senses=800, depth=2)
    assert u2.fingerprint == small_usvs.fingerprint
    assert u2.sense_ids == small_usvs.sense_ids
    assert np.array_equal(u2.core_coords, small_usvs.core_coords)
    assert np.array_equal(u2.sense_axis_val, small_usvs.sense_axis_val)
    assert u2.antonyms == small_usvs.antonyms


def test_save_load_roundtrip_identity(tmp_path, small_usvs):
    save_usvs(small_usvs, tmp_path)
    u2 = load_usvs(tmp_path)
    assert u2.fingerprint == small_usvs.fingerprint
    assert u2.axes == small_usvs.axes
    assert u2.core_words == small_usvs.core_words
    assert np.array_equal(u2.core_coords, small_usvs.core_coords)
    assert np.array_equal(u2.sense_indptr, small_usvs.sense_indptr)
    assert u2.antonyms == small_usvs.antonyms
    assert u2.genus == small_usvs.genus


def test_sense_signatures_populated(small_usvs):
    n_nonempty = sum(
        1 for sid in small_usvs.sense_ids if small_usvs.sense_signature(sid))
    assert n_nonempty / len(small_usvs.sense_ids) > 0.9
    sig = small_usvs.sense_signature(small_usvs.sense_ids[0])
    for axis in sig:
        assert axis in small_usvs.axes


def test_antonym_tiers_structure():
    edges = antonym_edges_tiered(["hot", "cold", "warm", "good", "bad", "dry", "wet"])
    tiers = {t for _, _, t in edges}
    assert ("cold", "hot", "direct") in edges
    assert tiers <= {"direct", "satellite_head", "satellite_satellite"}
    # strongest tier wins: hot/cold must be direct, not a weaker rediscovery
    assert all(t == "direct" for a, b, t in edges if (a, b) == ("cold", "hot"))


def test_query_api(small_usvs):
    u = small_usvs
    # word coords only for core words
    w = u.core_words[0]
    assert u.word_coord(w) is not None
    assert u.word_coord("zzz-not-a-word") is None
    # similarity is symmetric and self-similarity is maximal-ish
    a, b = u.core_words[0], u.core_words[1]
    assert u.similarity(a, b) == pytest.approx(u.similarity(b, a))
    assert u.similarity(a, a) >= u.similarity(a, b) - 1e-6
    # antonyms_of default excludes the weakest tier
    assert set(DEFAULT_ANTONYM_TIERS) == {"direct", "satellite_head"}
