"""M19.1 gate: interpretable axis set + dimensionality diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from nsm_ct.ground.axes import MeaningAxes, dimensionality_spectrum, feature_matrix
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.nsm_primes import NUM_PRIMES
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


@pytest.fixture(scope="module")
def setup():
    vocab = gloss_vocabulary(500)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    return vocab, g, cache


@wn_required
def test_axes_are_all_named_and_interpretable(setup):
    _, g, _ = setup
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    # the hard invariant: every axis is named with a known interpretable kind
    for a in ax.axes:
        assert a.name and a.kind in {"prime", "attribute", "lexname"}
    # primes are the seed; attribute + lexname extend it -> more than 65
    assert len(ax.by_kind("prime")) == NUM_PRIMES
    assert ax.dim > NUM_PRIMES
    assert len(ax.by_kind("lexname")) > 0


@wn_required
def test_assemble_is_deterministic(setup):
    _, g, _ = setup
    a = MeaningAxes.assemble(g)
    b = MeaningAxes.assemble(g)
    assert a.names == b.names


@wn_required
def test_feature_matrix_shape_and_binary(setup):
    vocab, g, cache = setup
    ax = MeaningAxes.assemble(g)
    words = g.words()
    M = feature_matrix(words, g, ax, cache=cache, depth=3)
    assert M.shape == (len(words), ax.dim)
    assert set(np.unique(M)).issubset({0.0, 1.0})
    # every word at least loads on its lexname axis -> no all-zero rows
    assert (M.sum(axis=1) > 0).all()


@wn_required
def test_dimensionality_is_a_compression(setup):
    _, g, cache = setup
    ax = MeaningAxes.assemble(g)
    spec = dimensionality_spectrum(g, ax, cache=cache, depth=3)
    # the named axes compress: far fewer effective axes than candidates
    assert 1 <= spec["intrinsic_dim_mass"] <= spec["n_axes"]
    assert 0 < spec["participation_ratio"] <= spec["n_axes"]
    assert spec["intrinsic_dim_mass"] < spec["n_axes"]
    # singular values are sorted descending
    sv = spec["top_singular_values"]
    assert all(a >= b for a, b in zip(sv, sv[1:]))
