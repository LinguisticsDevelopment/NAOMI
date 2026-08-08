"""Gates for the M31 USVS->handle bridge (nsm_ct.usvs_bridge)."""

import numpy as np
import pytest

from nsm_ct.usvs_bridge import _DEFAULT_DIR, usvs_handle, usvs_sense_handle
from nsm_ct.wordnet import wordnet_available

pytestmark = pytest.mark.skipif(
    not wordnet_available() or not _DEFAULT_DIR.exists(),
    reason="needs WordNet + a built data/usvs artifact",
)


def test_deterministic():
    a = usvs_handle("dog", 256)
    b = usvs_handle("dog", 256)
    assert a is not None and b is not None
    assert np.array_equal(a, b)


def test_unknown_word_returns_none():
    assert usvs_handle("zzz-not-a-real-word-xyz", 256) is None


def test_distinct_words_distinct_handles():
    dog = usvs_handle("dog", 256)
    cat = usvs_handle("cat", 256)
    assert dog is not None and cat is not None
    assert not np.array_equal(dog, cat)


def test_similarity_structure_survives_projection():
    dog = usvs_handle("dog", 256)
    puppy = usvs_handle("puppy", 256)
    justice = usvs_handle("justice", 256)
    assert dog is not None and puppy is not None and justice is not None
    cos_close = float(np.dot(dog, puppy))
    cos_far = float(np.dot(dog, justice))
    assert cos_close > cos_far


def test_unit_norm():
    h = usvs_handle("dog", 256)
    assert h is not None
    assert np.linalg.norm(h) == pytest.approx(1.0, abs=1e-5)


def test_sense_handle_bank():
    h = usvs_sense_handle("bank.n.01", 256)
    assert h is not None
    assert h.shape == (256,)
    assert np.linalg.norm(h) == pytest.approx(1.0, abs=1e-5)
