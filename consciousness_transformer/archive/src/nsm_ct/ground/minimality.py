"""Minimality — reverse-engineer the minimum axes (M19.3).

Given words placed over the named axes (M19.2), find the smallest subset of axes
that still reproduces the relations. Axes are ranked by how much they vary across
the placed words (a flat axis carries no discriminative signal), then the held-out
synonym-vs-antonym fidelity is measured as axes are added back top-down — the
fidelity-vs-#axes curve. The minimal set is the fewest axes reaching most of the
full fidelity. Every surviving axis is still NAMED (the hard invariant): pruning
only ever drops named axes, never invents anonymous ones.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .axes import MeaningAxes
from .closeness import discrimination, split_pairs
from .meaning_value import cosine
from .placement import place
from .relations import RelationGraph


def _splits(words, graph: RelationGraph, train_frac: float):
    wset = set(words)
    syn = sorted({tuple(sorted((w, s))) for w in words for s in graph.synonym.get(w, [])
                  if s in wset and s != w})
    sim = sorted({tuple(sorted((w, s))) for w in words for s in graph.similar.get(w, [])
                  if s in wset and s != w})
    ant = sorted({tuple(sorted(p)) for p in graph.typed_pairs("antonym")})
    train_syn, test_syn = split_pairs(syn, train_frac)
    train_sim, _ = split_pairs(sim, train_frac)
    _, test_ant = split_pairs(ant, train_frac)
    return train_syn + train_sim, test_syn, test_ant


def minimal_axes(
    words,
    graph: RelationGraph,
    axes: MeaningAxes,
    *,
    cache=None,
    depth: int = 3,
    train_frac: float = 0.5,
    alpha: float = 0.7,
    iters: int = 20,
    ks: List[int] = None,
    keep_frac: float = 0.95,
) -> Dict:
    """Fidelity-vs-#axes curve + the minimal named axis subset."""
    words = list(words)
    train_close, test_syn, test_ant = _splits(words, graph, train_frac)
    placed = place(words, graph, axes, cache=cache, depth=depth,
                   train_pairs=train_close, iters=iters, alpha=alpha)
    P = np.stack([placed[w] for w in words])
    idx = {w: i for i, w in enumerate(words)}
    names = axes.names

    importance = P.var(axis=0)
    order = list(np.argsort(importance)[::-1])

    def disc_for(keep_cols) -> float:
        mask = np.zeros(axes.dim, dtype=np.float32)
        for j in keep_cols:
            mask[j] = 1.0
        coord = {w: P[idx[w]] * mask for w in words}
        return discrimination(test_syn, test_ant, lambda a, b: cosine(coord[a], coord[b]), coord)[0]

    full = disc_for(order)
    if ks is None:
        ks = sorted({k for k in (5, 10, 15, 20, 30, 40, 60, 80, 120, axes.dim) if k <= axes.dim})
    curve: List[Tuple[int, float]] = [(k, disc_for(order[:k])) for k in ks]

    target = keep_frac * full
    minimal_k = next((k for k, d in curve if d >= target), axes.dim)
    kept = [names[j] for j in order[:minimal_k]]
    return {
        "n_axes": axes.dim,
        "full_discrimination": full,
        "minimal_k": minimal_k,
        "kept_axes": kept,
        "curve": curve,
        "n_test_syn": len(test_syn),
        "n_test_ant": len(test_ant),
    }
