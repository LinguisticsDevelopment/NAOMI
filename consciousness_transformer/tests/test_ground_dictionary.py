"""M19.4 gate: rebuild the dictionary in grounded space + validate."""

from __future__ import annotations

import pytest

from nsm_ct.ground.axes import MeaningAxes
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.dictionary import evaluate_dictionary, novel_synonyms
from nsm_ct.ground.placement import place
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
def test_grounded_space_reconstructs_relations(setup):
    g, cache, ax = setup
    r = evaluate_dictionary(g.words(), g, ax, cache=cache, depth=3, alpha=0.7, n_neg=1500)
    assert r["n"]["syn"] > 0 and r["n"]["isa"] > 0 and r["n"]["neg"] > 0
    # synonym, similar, hypernym reconstruct from geometry well above chance
    assert r["synonym_auc"] > 0.65
    assert r["hypernym_auc"] > 0.6
    assert r["similar_auc"] is None or r["similar_auc"] > 0.6
    # antonym by distance is near-but-opposite -> a valid score, not asserted > 0.5
    assert 0.0 <= r["antonym_auc"] <= 1.0


@wn_required
def test_novel_pairs_are_ranked(setup):
    g, cache, ax = setup
    placed = place(g.words(), g, ax, cache=cache, depth=3, alpha=0.7)
    nov = novel_synonyms(g.words(), g, placed, top=10)
    assert len(nov) <= 10
    scores = [s for _, _, s in nov]
    assert scores == sorted(scores, reverse=True)  # descending by closeness
    # none of them are already WordNet synonyms
    wset = set(g.words())
    known = {tuple(sorted((w, s))) for w in g.words() for s in g.synonym.get(w, []) if s in wset}
    for a, b, _ in nov:
        assert tuple(sorted((a, b))) not in known
