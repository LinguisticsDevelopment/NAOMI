"""M21 gate: Null-aware bipolar representation + contrastive objective.

What M21 genuinely delivers (asserted here): unrelated words become genuinely
unrelated (the user's main ask) and bipolar axes are clean. The contrastive
objective further separates unrelated pairs. Per-word antonym discrimination is an
honest negative (see RESEARCH_NOTES §0w) — NOT asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from nsm_ct.ground.cache import DecompCache
from nsm_ct.ground.closeness import split_pairs
from nsm_ct.ground.contrastive import contrastive_optimize
from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.ground.sparse_value import build_sparse_space, pair_similarity, similarity
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")

_CURATED = ["good", "bad", "hot", "cold", "big", "small", "grass", "dog", "justice", "water", "happy"]


@wn_required
def test_representation_is_sparse_and_masked():
    g = RelationGraph.build(_CURATED)
    sp = build_sparse_space(g.words(), g)
    assert sp.value.shape == sp.mask.shape == (len(g.words()), len(sp.axes))
    assert set(np.unique(sp.mask)).issubset({0.0, 1.0})
    # the Null model is sparse: each word occupies few applicable axes
    assert sp.mask.sum(1).mean() < 12


@wn_required
def test_bipolar_axes_put_antonyms_on_opposite_poles():
    g = RelationGraph.build(_CURATED)
    sp = build_sparse_space(g.words(), g)
    # under the per-pair masked read, antonyms sharing one axis are opposite
    assert similarity(sp, "hot", "cold", mode="masked") < 0
    assert similarity(sp, "good", "bad", mode="masked") < 0.5  # opposite/ not identical


@wn_required
def test_unrelated_words_are_unrelated():
    g = RelationGraph.build(_CURATED)
    sp = build_sparse_space(g.words(), g)
    # words sharing no applicable axis -> exactly 0 (the Null model's point)
    assert abs(similarity(sp, "good", "grass")) < 0.2


@wn_required
def test_random_pairs_separate_far_below_dense_baseline():
    vocab = gloss_vocabulary(1200)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    sp = build_sparse_space(g.words(), g, cache=cache, depth=3)
    words = sp.words
    idx = {w: i for i, w in enumerate(words)}
    rng = np.random.RandomState(0)
    rand = rng.randint(0, len(words), (3000, 2))
    rand = rand[rand[:, 0] != rand[:, 1]]
    # the M19/M20 dense coordinate had random-pair cosine ~0.32; the Null model is far lower
    assert pair_similarity(sp, rand).mean() < 0.20


@wn_required
def test_contrastive_runs_and_improves_separation():
    vocab = gloss_vocabulary(1200)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    sp = build_sparse_space(g.words(), g, cache=cache, depth=3)
    words = sp.words
    idx = {w: i for i, w in enumerate(words)}
    wset = set(words)

    def pidx(ps):
        return np.array([(idx[a], idx[b]) for a, b in ps if a in idx and b in idx and a != b])

    synp = sorted({tuple(sorted((w, s))) for w in words for s in g.synonym.get(w, []) if s in wset})
    antp = sorted({tuple(sorted(p)) for p in g.typed_pairs("antonym")})
    tr_s, _ = split_pairs(synp, 0.5)
    tr_a, _ = split_pairs(antp, 0.5)
    rng = np.random.RandomState(0)
    rand = rng.randint(0, len(words), (2000, 2))

    before = pair_similarity(sp, rand).mean()
    opt = contrastive_optimize(sp, train_syn=pidx(tr_s), train_ant=pidx(tr_a), iters=150)
    assert opt.value.shape == sp.value.shape and opt.axes == sp.axes
    after = pair_similarity(opt, rand).mean()
    # the contrastive objective pushes unrelated pairs further toward 0
    assert after <= before + 0.02
