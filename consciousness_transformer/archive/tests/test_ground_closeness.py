"""M18.3 gate: graph-aware closeness beats coordinates on held-out syn/ant."""

from __future__ import annotations

import numpy as np
import pytest

from nsm_ct.ground.basis_search import _value_vec
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.closeness import (
    antonym_linked,
    discrimination,
    evaluate_closeness,
    make_graph_closeness,
    split_pairs,
)
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.definition_graph import DefinitionGraph
from nsm_ct.ground.polarity import polarity_vector
from nsm_ct.ground.semantic_axes import AxisRegistry
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


def test_antonym_linked_propagates_one_synonym_hop():
    train_ant = {"hot": {"cold"}, "cold": {"hot"}}
    syn = {"cold": {"chilly"}, "chilly": {"cold"}}
    # hot's train-antonym 'cold' is a synonym of 'chilly' -> hot/chilly antonym-linked
    assert antonym_linked("hot", "chilly", train_ant, syn) is True
    # no path to an unrelated word
    assert antonym_linked("hot", "warm", train_ant, syn) is False


def test_graph_closeness_penalises_linked_pairs():
    coord = {"hot": np.array([1.0, 0.0]), "chilly": np.array([1.0, 0.0]), "warm": np.array([1.0, 0.0])}
    train_ant = {"hot": {"cold"}}
    syn = {"cold": {"chilly"}}
    close = make_graph_closeness(coord, {k: set(v) for k, v in train_ant.items()}, syn, lam=1.0)
    # identical coordinates -> cos 1.0; the antonym-linked pair is pushed down by lam
    assert close("hot", "warm") == pytest.approx(1.0)
    assert close("hot", "chilly") == pytest.approx(0.0)


def test_discrimination_is_auc_like():
    # synonyms score high, antonyms low -> perfect separation
    coord = {"a": 0, "b": 0, "c": 0, "d": 0}  # presence only; close ignores coord here
    scores = {("a", "b"): 0.9, ("c", "d"): 0.1}
    disc, ns, na = discrimination([("a", "b")], [("c", "d")], lambda x, y: scores[(x, y)], coord)
    assert disc == 1.0 and ns == 1 and na == 1


def test_split_is_disjoint():
    pairs = [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")]
    tr, te = split_pairs(pairs, 0.5)
    assert not (set(tr) & set(te))
    assert sorted(tr + te) == sorted(pairs)


@wn_required
def test_graph_closeness_beats_coordinates_held_out():
    vocab = gloss_vocabulary(1500)
    graph = DefinitionGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    reg = AxisRegistry.seed()
    dec = lambda x: cache.decompose(x, 3)
    coord_u = {w: _value_vec(w, reg, 3, cache) for w in vocab}
    coord_p = {w: polarity_vector(w, axes=reg.axes, depth=3, decompose=dec) for w in vocab}

    r = evaluate_closeness(vocab, graph=graph, coord_unsigned=coord_u, coord_polarity=coord_p,
                           train_frac=0.5, lam=1.0)
    assert r["n_test_ant"] > 0 and r["n_test_syn"] > 0
    # graph-aware closeness clears chance and beats the pure coordinate decisively.
    assert r["graph_closeness"] > 0.5
    assert r["graph_closeness"] > r["pure_coordinate"]
