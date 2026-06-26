"""M19.2 gate: placement by constraint satisfaction (relations as geometry)."""

from __future__ import annotations

import numpy as np
import pytest

from nsm_ct.ground.axes import MeaningAxes
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.placement import anchored_coordinate, evaluate_placement, place, relax
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


@wn_required
def test_attribute_axis_anchors_antonyms_at_opposite_poles():
    vocab = ["hot", "cold", "warm", "cool", "temperature", "big", "small"]
    g = RelationGraph.build(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    words = g.words()
    M = anchored_coordinate(words, g, ax, depth=3)
    j = ax.index("attr:temperature")
    ihot, icold = words.index("hot"), words.index("cold")
    # hot (gloss "high") and cold (gloss "low") anchor at opposite poles on the axis
    assert M[ihot, j] > 0 and M[icold, j] < 0


@wn_required
def test_relaxation_is_stable():
    anchor = np.random.RandomState(0).randn(50, 8).astype(np.float32)
    A = np.zeros((50, 50), dtype=np.float32)
    A[0, 1] = A[1, 0] = A[2, 3] = A[3, 2] = 1.0  # a couple of edges incl. high-degree-free
    P = relax(anchor, A, iters=30, alpha=0.7)
    assert np.isfinite(P).all()
    assert P.shape == anchor.shape


@wn_required
def test_place_is_deterministic():
    vocab = gloss_vocabulary(400)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    ax = MeaningAxes.assemble(g)
    a = place(g.words(), g, ax, cache=cache, depth=3, alpha=0.5)
    b = place(g.words(), g, ax, cache=cache, depth=3, alpha=0.5)
    w = g.words()[0]
    assert np.allclose(a[w], b[w])


@wn_required
def test_placement_folds_relations_into_position():
    vocab = gloss_vocabulary(1500)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    r = evaluate_placement(g.words(), g, ax, cache=cache, depth=3, train_frac=0.5, iters=20, alpha=0.7)
    assert r["n_test_ant"] > 0 and r["n_test_syn"] > 0
    # relaxation helps, and PLAIN cosine of the placed coordinate clears chance
    # (antonymy folded into the position, no comparison-time penalty).
    assert r["placed"] > r["anchored"]
    assert r["placed"] > 0.5
