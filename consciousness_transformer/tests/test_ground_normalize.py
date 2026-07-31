"""M20.0 gate: normalized coordinates fix overlap + incommensurability."""

from __future__ import annotations

import numpy as np
import pytest

from nsm_ct.ground.axes import MeaningAxes
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.closeness import split_pairs
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.normalize import (
    center,
    minmax_bound,
    normalize_matrix,
    standardize,
    tanh_standardize,
)
from nsm_ct.ground.placement import place
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


def test_standardize_is_zero_mean_unit_var():
    P = np.random.RandomState(0).randn(100, 8) * 5 + 3
    Z = standardize(P)
    assert np.allclose(Z.mean(0), 0, atol=1e-5)
    assert np.allclose(Z.std(0), 1, atol=1e-5)


def test_bounded_transforms_are_in_range():
    P = np.random.RandomState(1).randn(100, 8) * 5
    for fn in (minmax_bound, tanh_standardize):
        out = fn(P)
        assert out.min() >= -1.0 - 1e-6 and out.max() <= 1.0 + 1e-6


def test_center_removes_mean():
    P = np.random.RandomState(2).randn(50, 6) + 10
    assert np.allclose(center(P).mean(0), 0, atol=1e-6)


def test_constant_axis_does_not_blow_up():
    P = np.ones((20, 4))  # zero variance everywhere
    for kind in ("standardize", "minmax", "center", "tanh"):
        out = normalize_matrix(P, kind)
        assert np.isfinite(out).all()


@wn_required
def test_normalization_reduces_overlap_and_holds_discrimination():
    vocab = gloss_vocabulary(1200)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    words = g.words()
    idx = {w: i for i, w in enumerate(words)}
    wset = set(words)

    # HELD-OUT (M24 leakage fix): propagate over TRAIN closeness edges, score on the
    # disjoint test_syn / test_ant (was placing over all synonyms AND scoring on them).
    syn_all = sorted({tuple(sorted((w, s))) for w in words for s in g.synonym.get(w, []) if s in wset and s != w})
    sim_all = sorted({tuple(sorted((w, s))) for w in words for s in g.similar.get(w, []) if s in wset and s != w})
    ant_all = sorted({tuple(sorted(p)) for p in g.typed_pairs("antonym")})
    train_syn, test_syn = split_pairs(syn_all, 0.5)
    train_sim, _ = split_pairs(sim_all, 0.5)
    _, test_ant = split_pairs(ant_all, 0.5)
    P = np.stack([place(words, g, ax, cache=cache, depth=3, alpha=0.7,
                        train_pairs=train_syn + train_sim)[w] for w in words])
    syn = np.array([(idx[a], idx[b]) for a, b in test_syn if a != b])
    ant = np.array([(idx[a], idx[b]) for a, b in test_ant if a != b])
    rng = np.random.RandomState(0)
    rand = rng.randint(0, len(words), (2000, 2))

    def norm_rows(M):
        n = np.linalg.norm(M, axis=1, keepdims=True)
        n[n < 1e-9] = 1.0
        return M / n

    def stats(M):
        N = norm_rows(M)
        rnd = float((N[rand[:, 0]] * N[rand[:, 1]]).sum(1).mean())
        ss = (N[syn[:, 0]] * N[syn[:, 1]]).sum(1)
        aa = (N[ant[:, 0]] * N[ant[:, 1]]).sum(1)
        return rnd, float((ss[:, None] > aa[None, :]).mean())

    raw_rand, raw_disc = stats(P)
    tanh_rand, tanh_disc = stats(tanh_standardize(P))
    # unrelated pairs stop overlapping, and discrimination holds or improves
    assert tanh_rand < raw_rand
    assert tanh_disc >= raw_disc - 0.02
