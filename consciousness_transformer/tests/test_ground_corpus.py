"""M18.0 gate: gloss-vocabulary corpus + decomposition cache."""

from __future__ import annotations

import pytest

from nsm_ct.ground.cache import DecompCache, apply_extra_axes
from nsm_ct.ground.canonicalization import tree_key
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.definition_graph import naive_decompose
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


@wn_required
def test_gloss_vocabulary_is_ranked_and_deterministic():
    a = gloss_vocabulary(200)
    b = gloss_vocabulary(200)
    assert len(a) == 200
    assert a == b  # deterministic
    # high-frequency defining-vocabulary words show up near the top
    assert any(w in a[:60] for w in ("one", "small", "person", "make", "part"))


@wn_required
def test_cache_reproduces_naive_decompose_with_extra_axes():
    extra = frozenset({"animal", "feeling", "act", "having"})
    cache = DecompCache(depth=3)
    for w in ["dog", "sad", "happy", "justice", "run", "afraid", "give"]:
        via_cache = cache.decompose(w, 3, extra)
        direct = naive_decompose(w, max_depth=3, extra_axes=extra)
        assert tree_key(via_cache) == tree_key(direct)


@wn_required
def test_apply_extra_axes_no_extra_is_identity():
    base = naive_decompose("kitchen", max_depth=3)
    assert tree_key(apply_extra_axes(base, frozenset())) == tree_key(base)


@wn_required
def test_cache_warm_and_len():
    cache = DecompCache(depth=2).warm(["dog", "cat", "good"])
    assert len(cache) == 3
    # decompose returns cached base structure
    assert cache.decompose("dog", 2).num_nodes() >= 1


@wn_required
def test_cache_disk_round_trip(tmp_path):
    p = tmp_path / "dc.json"
    c1 = DecompCache(depth=3, path=str(p)).warm(["dog", "happy", "tree"])
    c1.save()
    c2 = DecompCache(depth=3, path=str(p))  # loads on construct
    assert len(c2) == 3
    for w in ["dog", "happy", "tree"]:
        assert tree_key(c2.base(w)) == tree_key(c1.base(w))
