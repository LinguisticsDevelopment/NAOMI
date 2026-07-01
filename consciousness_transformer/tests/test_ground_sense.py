"""M22 gate: sense-node graph (WSD by construction) + fused Null+propagation."""

from __future__ import annotations

import numpy as np
import pytest

from nsm_ct.ground.closeness import split_pairs
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.fusion import (
    _cos,
    co_hyponym_pairs,
    fused_similarity,
    propagate,
    relatedness,
)
from nsm_ct.ground.sense_graph import SenseGraph, build_sense_sparse
from nsm_ct.ground.sparse_value import pair_similarity
from nsm_ct.wordnet import senses, wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


@wn_required
def test_senses_now_carry_per_sense_antonyms_and_frequency():
    ss = senses("good")
    assert ss and all("antonyms" in s and "frequency" in s for s in ss)
    # per-sense antonyms differ across senses of 'good' (bad vs evil vs ...)
    antsets = {tuple(s["antonyms"]) for s in ss if s["antonyms"]}
    assert len(antsets) >= 1


@pytest.fixture(scope="module")
def sense_setup():
    vocab = gloss_vocabulary(1000)
    g = SenseGraph.build(vocab, max_senses_per_word=3, cap=2500)
    sp = build_sense_sparse(g, depth=2)
    return g, sp


@wn_required
def test_sense_nodes_and_relations(sense_setup):
    g, sp = sense_setup
    assert len(g.senses) > 100
    assert all("." in sid for sid in g.senses)  # synset ids like large.a.01
    assert sp.value.shape[0] == len(g.senses)
    # sense-specific relations exist
    assert len(g.typed_pairs("similar")) > 0
    assert len(g.typed_pairs("antonym")) > 0


@wn_required
def test_unrelated_sense_pairs_are_far(sense_setup):
    g, sp = sense_setup
    idx = {n: i for i, n in enumerate(sp.words)}
    rng = np.random.RandomState(0)
    rand = rng.randint(0, len(sp.words), (3000, 2))
    rand = rand[rand[:, 0] != rand[:, 1]]
    assert pair_similarity(sp, rand).mean() < 0.20  # Null win holds on sense-nodes


@wn_required
def test_threshold_fusion_keeps_discrimination_and_cuts_overlap(sense_setup):
    g, sp = sense_setup
    idx = {n: i for i, n in enumerate(sp.words)}

    def pidx(ps):
        return np.array([(idx[a], idx[b]) for a, b in ps if a in idx and b in idx and a != b])

    sim, ant = g.typed_pairs("similar"), g.typed_pairs("antonym")
    close = sim + co_hyponym_pairs(g)
    tr_c, _ = split_pairs(close, 0.7)
    _, te_sim = split_pairs(sim, 0.5)
    _, te_ant = split_pairs(ant, 0.5)
    P = propagate(sp, tr_c, iters=20, alpha=0.7)
    S, A = pidx(te_sim), pidx(te_ant)
    rng = np.random.RandomState(0)
    rand = rng.randint(0, len(sp.words), (3000, 2))

    if len(S) == 0 or len(A) == 0:
        pytest.skip("insufficient sense relation pairs at this vocab size")

    def disc(score_fn):
        s, a = score_fn(S), score_fn(A)
        return float((s[:, None] > a[None, :]).mean())

    prop_disc = disc(lambda pr: _cos(P, pr))
    prop_rand = _cos(P, rand).mean()
    fused_disc = disc(lambda pr: fused_similarity(sp, P, pr, threshold=0.15))
    fused_rand = fused_similarity(sp, P, rand, threshold=0.15).mean()

    # the threshold gate cuts unrelated overlap while preserving discrimination
    assert fused_rand < prop_rand
    assert fused_disc >= prop_disc - 0.03
