"""M18.2 gate: multi-signal basis selection (mechanism + honest comparison)."""

from __future__ import annotations

import numpy as np
import pytest

from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.definition_graph import DefinitionGraph
from nsm_ct.ground.multisignal import (
    multisignal_search,
    relation_pairs,
    split_pairs,
    _rel_score,
)
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")

# A synonym-rich vocab so in-vocab relation pairs exist.
_VOCAB = [
    "big", "large", "great", "small", "little", "fast", "quick", "rapid", "slow",
    "happy", "glad", "joyful", "sad", "unhappy", "smart", "clever", "begin",
    "start", "end", "finish", "good", "bad", "rich", "wealthy", "poor",
]


def test_split_pairs_is_deterministic_and_partitions():
    pairs = [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h"), ("i", "j")]
    tr1, te1 = split_pairs(pairs, 0.5)
    tr2, te2 = split_pairs(pairs, 0.5)
    assert tr1 == tr2 and te1 == te2  # deterministic
    assert sorted(tr1 + te1) == sorted(pairs)  # a partition
    assert not (set(tr1) & set(te1))  # disjoint


def test_rel_score_rewards_aligned_coordinates():
    axes = ["X", "Y"]
    coord = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0]), "c": np.array([0.0, 1.0])}
    high = _rel_score(coord, [("a", "b")], [], axes)  # identical -> cos 1
    low = _rel_score(coord, [("a", "c")], [], axes)   # orthogonal -> cos 0
    assert high > low


@wn_required
def test_relation_pairs_finds_in_vocab_pairs():
    g = DefinitionGraph.build(_VOCAB)
    syn, hyp, ant = relation_pairs(_VOCAB, g)
    # big/large/great share synsets -> some synonym pairs exist
    assert len(syn) > 0
    for a, b in syn:
        assert a in _VOCAB and b in _VOCAB


@wn_required
def test_multisignal_runs_and_reports_held_out_comparison():
    g = DefinitionGraph.build(_VOCAB)
    cache = DecompCache(depth=3).warm(_VOCAB)
    res = multisignal_search(_VOCAB, depth=3, max_axes=3, graph=g, cache=cache, train_frac=0.5)
    assert len(res.added) >= 1
    # provenance records the relational delta that drove each promotion
    for axis, _ in res.added:
        assert res.registry.provenance[axis].startswith("multisignal")
    # both bases evaluated on the same held-out split (honest comparison)
    for m in (res.mdl_only_metrics, res.multisignal_metrics):
        assert "synonym_cos" in m and "hypernym_containment" in m


@wn_required
def test_multisignal_is_deterministic():
    g = DefinitionGraph.build(_VOCAB)
    cache = DecompCache(depth=3).warm(_VOCAB)
    a = multisignal_search(_VOCAB, depth=3, max_axes=3, graph=g, cache=cache)
    b = multisignal_search(_VOCAB, depth=3, max_axes=3, graph=g, cache=cache)
    assert [x[0] for x in a.added] == [x[0] for x in b.added]
