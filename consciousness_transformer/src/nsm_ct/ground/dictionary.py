"""Rebuild the dictionary in grounded space + validate (M19.4).

The payoff of the unified space: once words are placed, redraw the dictionary's
relationships as geometry and check the reconstruction against held-out WordNet.
Per relation we ask "do true pairs score higher than random pairs?" (an AUC-like
retrieval score, 0.5 = chance):

- synonym / similar -> high cosine of the placed coordinate,
- hypernym (a IS_A b) -> feature containment (a has b's anchored features + more),
- antonym -> far / opposite (low cosine) — the hard one, reported honestly.

Also surfaces NOVEL high-cosine pairs not in WordNet's synonym set — relationships
the grounded space proposes (the generative payoff), for spot-checking.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .axes import MeaningAxes
from .closeness import split_pairs
from .meaning_value import cosine
from .placement import anchored_coordinate, place
from .relations import RelationGraph


def _auc(pos: List[float], neg: List[float]) -> float:
    p, n = np.asarray(pos, dtype=np.float32), np.asarray(neg, dtype=np.float32)
    if len(p) == 0 or len(n) == 0:
        return 0.5
    return float((p[:, None] > n[None, :]).mean())


def _random_pairs(words, n: int, seed: int, exclude: set) -> List[Tuple[str, str]]:
    rng = np.random.RandomState(seed)
    out, W = [], len(words)
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        i, j = int(rng.randint(W)), int(rng.randint(W))
        if i == j:
            continue
        p = tuple(sorted((words[i], words[j])))
        if p not in exclude:
            out.append(p)
    return out


def evaluate_dictionary(
    words,
    graph: RelationGraph,
    axes: MeaningAxes,
    *,
    cache=None,
    depth: int = 3,
    train_frac: float = 0.5,
    alpha: float = 0.7,
    iters: int = 20,
    n_neg: int = 2000,
    seed: int = 0,
) -> Dict:
    """Per-relation reconstruction of the dictionary from grounded positions."""
    words = list(words)
    wset = set(words)
    idx = {w: i for i, w in enumerate(words)}

    syn = sorted({tuple(sorted((w, s))) for w in words for s in graph.synonym.get(w, []) if s in wset and s != w})
    sim = sorted({tuple(sorted((w, s))) for w in words for s in graph.similar.get(w, []) if s in wset and s != w})
    ant = sorted({tuple(sorted(p)) for p in graph.typed_pairs("antonym")})
    isa = sorted(set(graph.typed_pairs("is_a", directed=True)))

    train_syn, test_syn = split_pairs(syn, train_frac)
    train_sim, test_sim = split_pairs(sim, train_frac)
    _, test_ant = split_pairs(ant, train_frac)
    _, test_isa = split_pairs(isa, train_frac)

    placed = place(words, graph, axes, cache=cache, depth=depth,
                   train_pairs=train_syn + train_sim, iters=iters, alpha=alpha)
    anchor = anchored_coordinate(words, graph, axes, cache=cache, depth=depth)

    exclude = set(syn) | set(sim) | {tuple(sorted(p)) for p in ant} | {tuple(sorted(p)) for p in isa}
    neg = _random_pairs(words, n_neg, seed, exclude)

    def cos(a, b):
        return cosine(placed[a], placed[b])

    def contain(a, b):  # a IS_A b: fraction of b's anchored features a also has
        av = anchor[idx[a]] > 0
        bv = anchor[idx[b]] > 0
        nb = int(bv.sum())
        return float((av & bv).sum() / nb) if nb > 0 else 0.0

    neg_cos = [cos(a, b) for a, b in neg]
    result = {
        "synonym_auc": _auc([cos(a, b) for a, b in test_syn], neg_cos),
        "similar_auc": _auc([cos(a, b) for a, b in test_sim], neg_cos) if test_sim else None,
        "antonym_auc": _auc([-cos(a, b) for a, b in test_ant], [-c for c in neg_cos]),
        "hypernym_auc": _auc([contain(a, b) for a, b in test_isa],
                             [contain(a, b) for a, b in neg]),
        "n": {"syn": len(test_syn), "sim": len(test_sim), "ant": len(test_ant),
              "isa": len(test_isa), "neg": len(neg)},
    }
    return result


def novel_synonyms(words, graph: RelationGraph, placed, *, top: int = 15, sample: int = 4000, seed: int = 0) -> List[Tuple[str, str, float]]:
    """High-cosine word pairs NOT in WordNet's synonym set — relationships the
    grounded space proposes (spot-check sample, not exhaustive)."""
    words = list(words)
    wset = set(words)
    known = {tuple(sorted((w, s))) for w in words for s in graph.synonym.get(w, []) if s in wset}
    known |= {tuple(sorted((w, s))) for w in words for s in graph.similar.get(w, []) if s in wset}
    rng = np.random.RandomState(seed)
    seen = set()
    scored = []
    for _ in range(sample):
        i, j = int(rng.randint(len(words))), int(rng.randint(len(words)))
        if i == j:
            continue
        p = tuple(sorted((words[i], words[j])))
        if p in known or p in seen:
            continue
        seen.add(p)
        scored.append((p[0], p[1], cosine(placed[p[0]], placed[p[1]])))
    scored.sort(key=lambda x: -x[2])
    return scored[:top]
