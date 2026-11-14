"""M17.2 gate: MDL-driven basis discovery + the extensible axis registry."""

from __future__ import annotations

import pytest

from nsm_ct.nsm_primes import NUM_PRIMES, PRIME_NAMES
from nsm_ct.ground.basis_search import mdl, relational_metrics, search
from nsm_ct.ground.clause_self_consistency import SAMPLE_VOCAB
from nsm_ct.ground.definition_graph import DefinitionGraph
from nsm_ct.ground.semantic_axes import AxisRegistry
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")

_PRIME_SET = frozenset(PRIME_NAMES)


def test_axis_registry_seed_is_nsm():
    reg = AxisRegistry.seed()
    assert reg.dim == NUM_PRIMES
    assert reg.beyond_seed() == []
    assert reg.add("animal", "test") is True
    assert "animal" in reg
    assert reg.beyond_seed() == ["animal"]
    assert reg.extra_axes() == frozenset({"animal"})
    # idempotent add
    assert reg.add("animal", "test") is False


@wn_required
def test_search_promotes_interpretable_axes():
    res = search(SAMPLE_VOCAB, depth=3, max_axes=10)
    promoted = res.registry.beyond_seed()
    assert len(promoted) > 0
    # promoted axes are new primitives, never seed primes/molecules
    for a in promoted:
        assert a not in _PRIME_SET
        assert res.registry.provenance[a].startswith("cycle-break")


@wn_required
def test_mdl_curve_is_monotonically_non_increasing():
    res = search(SAMPLE_VOCAB, depth=3, max_axes=10)
    vals = [m for _, m in res.mdl_curve]
    assert all(b <= a for a, b in zip(vals, vals[1:]))  # never increases
    assert vals[-1] < vals[0]  # strictly improves overall


@wn_required
def test_grounding_strictly_improves_over_seed():
    res = search(SAMPLE_VOCAB, depth=3, max_axes=10)
    assert res.final_metrics["grounding_rate"] > res.seed_metrics["grounding_rate"]


@wn_required
def test_search_is_deterministic():
    a = search(SAMPLE_VOCAB, depth=3, max_axes=8)
    b = search(SAMPLE_VOCAB, depth=3, max_axes=8)
    assert [x[0] for x in a.added] == [x[0] for x in b.added]


@wn_required
def test_relational_metrics_keys_present():
    words = ["dog", "cat", "hot", "cold", "happy", "sad"]
    g = DefinitionGraph.build(words)
    m = relational_metrics(words, AxisRegistry.seed(), 3, g)
    for k in ("grounding_rate", "antonym_cos", "synonym_cos", "hypernym_containment"):
        assert k in m


@wn_required
def test_mdl_decreases_when_adding_a_useful_axis():
    words = ["happy", "sad", "afraid"]
    base = mdl(words, frozenset(), 3)
    with_axis = mdl(words, frozenset({"feeling"}), 3)
    # 'feeling' is a frequent gloss word for emotions: grounding it cuts MDL.
    assert with_axis < base
