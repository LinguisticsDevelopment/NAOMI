"""M18.1 gate: polarity-aware coordinates improve syn>ant discrimination."""

from __future__ import annotations

import pytest

from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.definition_graph import DefinitionGraph
from nsm_ct.ground.evaluation import syn_ant_discrimination
from nsm_ct.ground.polarity import (
    gloss_polarity,
    negation_base,
    polarity_vector,
    signed_axes,
)
from nsm_ct.ground.semantic_axes import AxisRegistry
from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")

# A magnitude/negation-rich antonym set where the polarity signal is expected to help.
_VOCAB = [
    "hot", "cold", "warm", "big", "small", "large", "high", "low", "fast", "slow",
    "strong", "weak", "deep", "shallow", "heavy", "light", "rich", "poor", "full",
    "empty", "good", "bad", "happy", "unhappy", "kind", "unkind",
]


@wn_required
def test_negation_base_detects_morphological_negation():
    assert negation_base("unhappy") == "happy"
    assert negation_base("unkind") == "kind"
    assert negation_base("careless") == "care"
    # a non-negation word returns None (no real base)
    assert negation_base("table") is None


@wn_required
def test_gloss_polarity_signs_magnitude_antonyms():
    # hot is defined via "high temperature", cold via "low temperature".
    assert gloss_polarity("hot") > 0
    assert gloss_polarity("cold") < 0
    assert gloss_polarity("hot") > gloss_polarity("cold")


@wn_required
def test_signed_vector_shape_and_pole_flip():
    axes = AxisRegistry.seed().axes
    v = polarity_vector("happy", axes=axes)
    assert v.shape == (len(signed_axes(axes)),)
    # a morphological negation flips the polarity (last) axes vs its base
    import numpy as np
    base = polarity_vector("kind", axes=axes)
    neg = polarity_vector("unkind", axes=axes)
    from nsm_ct.ground.polarity import _N_POLES
    assert np.allclose(neg[-_N_POLES:], -base[-_N_POLES:])


@wn_required
def test_polarity_improves_syn_ant_discrimination():
    reg = AxisRegistry.seed()
    graph = DefinitionGraph.build(_VOCAB)
    cache = DecompCache(depth=3).warm(_VOCAB)

    unsigned = syn_ant_discrimination(_VOCAB, reg, 3, graph)  # default unsigned coord
    signed_coord = {w: polarity_vector(w, axes=reg.axes, depth=3,
                                       decompose=lambda x: cache.decompose(x, 3)) for w in _VOCAB}
    signed = syn_ant_discrimination(_VOCAB, reg, 3, graph, coord=signed_coord)

    assert signed["n"] > 0 and unsigned["n"] > 0
    # On magnitude-rich antonyms the signed coordinate must do strictly better.
    assert signed["accuracy"] > unsigned["accuracy"]
