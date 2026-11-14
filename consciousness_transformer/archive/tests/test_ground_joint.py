"""M25 gate: joint all-signals TRAINED placement — mechanics + the honest negative.

The trainer must stay within the thesis (values on named axes only, Null mask
re-imposed so no content leaks onto inapplicable axes). Performance is an honest
NEGATIVE — free per-word fitting underperforms deterministic propagation — so the
tests assert the mechanics and the held-out discipline, not a win.
"""

from __future__ import annotations

import numpy as np
import pytest

from nsm_ct.ground.axes import MeaningAxes
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.joint_place import joint_place, split_all_relations
from nsm_ct.ground.placement import anchored_coordinate
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


@pytest.fixture(scope="module")
def setup():
    vocab = gloss_vocabulary(800)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    return g, cache, ax


@wn_required
def test_split_all_relations_is_disjoint(setup):
    g, cache, ax = setup
    train, test = split_all_relations(g.words(), g, train_frac=0.5)
    assert set(train) == {"syn", "sim", "ant", "hyp", "rel"}
    for k in train:
        assert not (set(map(tuple, train[k])) & set(map(tuple, test[k])))  # held-out


@wn_required
def test_trained_space_stays_masked_and_named(setup):
    g, cache, ax = setup
    words = g.words()
    train, _ = split_all_relations(words, g, train_frac=0.5)
    coord = joint_place(words, g, ax, cache=cache, depth=3,
                        train_syn=train["syn"], train_sim=train["sim"], train_ant=train["ant"],
                        train_hyp=train["hyp"], train_rel=train["rel"], iters=40)
    A = anchored_coordinate(words, g, ax, cache=cache, depth=3)
    # every output vector is over the NAMED axes, and (thesis) the Null mask holds:
    # no value appears on an axis that did not already apply to the word.
    for i, w in enumerate(words[:200]):
        v = coord[w]
        assert v.shape[0] == len(ax.names)
        off = np.abs(A[i]) <= 1e-9
        assert np.allclose(v[off], 0.0)  # non-overlap preserved


@wn_required
def test_trainer_moves_values_and_stays_finite(setup):
    g, cache, ax = setup
    words = g.words()
    train, _ = split_all_relations(words, g, train_frac=0.5)
    coord = joint_place(words, g, ax, cache=cache, depth=3,
                        train_syn=train["syn"], train_sim=train["sim"], train_ant=train["ant"],
                        train_hyp=train["hyp"], train_rel=train["rel"], iters=60)
    A = anchored_coordinate(words, g, ax, cache=cache, depth=3)
    V = np.stack([coord[w] for w in words])
    assert np.isfinite(V).all()
    # the optimizer actually moved the values off the anchored init (it trained)
    assert not np.allclose(V, A)
