"""M20.1 gate: ablation/contribution-based minimality on the normalized space."""

from __future__ import annotations

import pytest

from nsm_ct.ground.axes import MeaningAxes
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.honest_minimality import contribution_minimal_axes
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


@pytest.fixture(scope="module")
def setup():
    vocab = gloss_vocabulary(1500)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    return g, cache, ax


@wn_required
def test_normalized_discrimination_beats_raw_space(setup):
    g, cache, ax = setup
    r = contribution_minimal_axes(g.words(), g, ax, cache=cache, depth=3, alpha=0.7)
    # the normalized space discriminates far better than M19.3's raw ~0.69
    assert r["full_discrimination"] > 0.80
    assert r["n_test_ant"] > 0 and r["n_test_syn"] > 0


@wn_required
def test_minimality_compresses_and_is_named(setup):
    g, cache, ax = setup
    r = contribution_minimal_axes(g.words(), g, ax, cache=cache, depth=3, alpha=0.7, keep_frac=0.95)
    assert 1 <= r["minimal_k"] < r["n_axes"]
    valid = set(ax.names)
    assert all(name in valid for name in r["kept_axes"])           # named invariant
    assert sum(r["kept_kinds"].values()) == r["minimal_k"]


@wn_required
def test_curve_rises_then_saturates(setup):
    g, cache, ax = setup
    r = contribution_minimal_axes(g.words(), g, ax, cache=cache, depth=3, alpha=0.7)
    d = dict(r["curve"])
    # adding load-bearing axes early clearly helps
    assert d[20] > d[5]
    # the minimal set is a genuine mix, not one kind monopolizing
    assert sum(1 for v in r["kept_kinds"].values() if v > 0) >= 2
