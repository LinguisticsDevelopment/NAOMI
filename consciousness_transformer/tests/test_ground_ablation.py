"""Gates for the Step B ablation harness (M24 compliance by construction)."""

import pytest

from nsm_ct.ground.ablation import (
    ablate, baseline_table, metric_table, pair_sets, with_antonyms, with_features,
)
from nsm_ct.ground.axes import MeaningAxes
from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.closeness import split_pairs
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.wordnet import wordnet_available

pytestmark = pytest.mark.skipif(not wordnet_available(), reason="needs WordNet")


@pytest.fixture(scope="module")
def small_space():
    vocab = gloss_vocabulary(400)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=2).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    return list(vocab), g, cache, ax


def test_m24_planted_test_pair_is_excluded(small_space):
    """A close_extra pair that IS a held-out test pair must be dropped."""
    words, g, cache, ax = small_space
    ps = pair_sets(words, g)
    _, test_syn = split_pairs(ps["syn"], 0.5)
    assert test_syn, "fixture needs at least one held-out synonym pair"
    planted = test_syn[0]
    t = metric_table(words, g, ax, cache=cache, depth=2, train_frac=0.5,
                     close_extra=[planted], n_neg=200)
    assert t["n"]["extra_close"] == 0
    assert t["n"]["extra_close_dropped_m24"] == 1


def test_harness_is_deterministic(small_space):
    words, g, cache, ax = small_space
    a = metric_table(words, g, ax, cache=cache, depth=2, n_neg=200)
    b = metric_table(words, g, ax, cache=cache, depth=2, n_neg=200)
    for k in ("syn_ant", "synonym_auc", "hypernym_auc"):
        assert a[k] == b[k]


def test_baseline_band_shape(small_space):
    words, g, cache, ax = small_space
    base = baseline_table(words, g, ax, cache=cache, depth=2,
                          jitters=(0.45, 0.55), n_neg=200)
    assert set(base["mean"]) == set(base["band"])
    assert all(v >= 0 for v in base["band"].values())


def test_antonym_extra_reported_separately(small_space):
    """Expanded antonym pairs never contaminate the original held-out score."""
    words, g, cache, ax = small_space
    ps = pair_sets(words, g)
    base = metric_table(words, g, ax, cache=cache, depth=2, n_neg=200)
    fake = [(words[0], words[1])]
    assert tuple(sorted(fake[0])) not in set(ps["ant"])
    t = metric_table(words, g, ax, cache=cache, depth=2, n_neg=200,
                     antonym_extra=fake)
    assert t["syn_ant"] == base["syn_ant"]          # original score untouched
    assert t["ant_expanded"]["n_new_pairs"] == 1     # reported on the side


def test_graph_copies_do_not_mutate(small_space):
    words, g, cache, ax = small_space
    before = {w: list(v) for w, v in g.antonym.items()}
    g2 = with_antonyms(g, [(words[0], words[1])])
    g3 = with_features(g, {words[0]: ["testcat"]})
    assert {w: list(v) for w, v in g.antonym.items()} == before
    assert words[1] in g2.antonym.get(words[0], [])
    assert "testcat" in g3.attribute.get(words[0], [])


def test_ablate_reports_verdict(small_space):
    words, g, cache, ax = small_space
    res = ablate(words, g, ax, cache=cache, name="noop", depth=2,
                 jitters=(0.45, 0.55), n_neg=200)
    assert res["delta"].keys() == res["baseline"]["mean"].keys()
    # a no-op signal must not report movement
    assert res["moved"] == {}
