"""M19.0 gate: the unified relation store (every source of signal)."""

from __future__ import annotations

import pytest

from nsm_ct.ground.relations import RelationGraph, WORD_RELATIONS
from nsm_ct.wordnet import (
    attributes,
    attributes_of,
    lexname,
    similar_tos,
    wordnet_available,
)

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")

_VOCAB = [
    "hot", "cold", "warm", "big", "small", "large", "heavy", "light",
    "dog", "cat", "run", "walk", "happy", "sad", "good", "bad", "temperature",
]


@wn_required
def test_wordnet_wrappers():
    assert "temperature" in attributes("hot")          # adj -> dimension
    assert "hot" in attributes_of("temperature")        # reverse
    assert lexname("dog") == "noun.animal"
    assert similar_tos("big")                            # adj cluster non-empty


@wn_required
def test_relation_graph_carries_all_sources():
    g = RelationGraph.build(_VOCAB)
    # the feature relations that NAME axes
    assert "temperature" in g.attribute.get("hot", [])
    assert g.lexname.get("dog") == "noun.animal"
    # lexname is universal: every built word has one
    assert len(g.lexname) == len(g.words())
    # word-word relations present
    assert g.synonym and g.antonym and g.is_a


@wn_required
def test_typed_pairs_are_in_vocab():
    g = RelationGraph.build(_VOCAB)
    ant = g.typed_pairs("antonym")
    assert ("cold", "hot") in ant
    for a, b in ant:
        assert a in _VOCAB and b in _VOCAB


@wn_required
def test_candidate_axes_from_relations():
    g = RelationGraph.build(_VOCAB)
    # attribute relation yields named dimensional axes (temperature, size, weight...)
    axes = g.attribute_axes()
    assert "temperature" in axes
    # lexname categories are a bounded named set
    lx = g.lexnames()
    assert len(lx) > 0 and all("." in name for name in lx)


@wn_required
def test_coverage_reports_every_relation():
    g = RelationGraph.build(_VOCAB)
    cov = g.coverage()
    for r in WORD_RELATIONS:
        assert r in cov and 0.0 <= cov[r] <= 1.0
    assert cov["lexname"] == 1.0  # universal
    assert "attribute" in cov
