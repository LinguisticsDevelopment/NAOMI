"""M19.3 gate: minimality — the minimum named axes that reproduce relations."""

from __future__ import annotations

import pytest

from nsm_ct.ground.axes import MeaningAxes
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.minimality import minimal_axes
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
def test_minimality_compresses(setup):
    g, cache, ax = setup
    r = minimal_axes(g.words(), g, ax, cache=cache, depth=3, alpha=0.7, keep_frac=0.95)
    assert r["n_test_ant"] > 0 and r["n_test_syn"] > 0
    # a small named subset reproduces most of the relational fidelity
    assert 1 <= r["minimal_k"] < r["n_axes"]
    assert len(r["kept_axes"]) == r["minimal_k"]


@wn_required
def test_kept_axes_are_all_named(setup):
    g, cache, ax = setup
    r = minimal_axes(g.words(), g, ax, cache=cache, depth=3, alpha=0.7)
    valid = set(ax.names)
    assert all(name in valid for name in r["kept_axes"])  # hard invariant: named only


@wn_required
def test_curve_reaches_full_at_all_axes(setup):
    g, cache, ax = setup
    r = minimal_axes(g.words(), g, ax, cache=cache, depth=3, alpha=0.7)
    last_k, last_d = r["curve"][-1]
    assert last_k == r["n_axes"]
    assert abs(last_d - r["full_discrimination"]) < 1e-6
    # the minimal_k point clears the keep_frac target
    disc_at_min = dict(r["curve"]).get(r["minimal_k"])
    if disc_at_min is not None:
        assert disc_at_min >= 0.95 * r["full_discrimination"] - 1e-6
