"""Honest, ablation-based minimality (M20.1).

M19.3 ranked axes by *variance* and concluded "30 axes" — but ablation showed the
kept high-variance lexname axes weren't load-bearing while pruned low-variance
attribute/prime axes were. This reworks minimality on the NORMALIZED (tanh) space
and ranks axes by their **measured contribution** to held-out synonym-vs-antonym
fidelity (leave-one-out), not variance. The minimum-axes number it reports is
honest even if larger than 30; every kept axis is still named.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .axes import MeaningAxes
from .closeness import split_pairs
from .normalize import tanh_standardize
from .placement import place
from .relations import RelationGraph


def _norm_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    return M / n


def _disc(P: np.ndarray, syn: np.ndarray, ant: np.ndarray, cols=None) -> float:
    """Vectorized P(synonym pair cosine > antonym pair cosine) on selected columns."""
    M = P if cols is None else P[:, cols]
    if M.shape[1] == 0 or len(syn) == 0 or len(ant) == 0:
        return 0.5
    N = _norm_rows(M)
    ss = (N[syn[:, 0]] * N[syn[:, 1]]).sum(1)
    aa = (N[ant[:, 0]] * N[ant[:, 1]]).sum(1)
    return float((ss[:, None] > aa[None, :]).mean())


def contribution_minimal_axes(
    words,
    graph: RelationGraph,
    axes: MeaningAxes,
    *,
    cache=None,
    depth: int = 3,
    alpha: float = 0.7,
    train_frac: float = 0.5,
    keep_frac: float = 0.95,
    ks: List[int] = None,
) -> Dict:
    """Rank axes by leave-one-out contribution to held-out syn-vs-ant fidelity on
    the normalized space; report the fidelity-vs-#axes curve and the minimal set."""
    words = list(words)
    idx = {w: i for i, w in enumerate(words)}

    wset = set(words)
    syn_pairs = sorted({tuple(sorted((w, s))) for w in words for s in graph.synonym.get(w, []) if s in wset and s != w})
    sim_pairs = sorted({tuple(sorted((w, s))) for w in words for s in graph.similar.get(w, []) if s in wset and s != w})
    ant_pairs = sorted({tuple(sorted(p)) for p in graph.typed_pairs("antonym")})
    train_syn, test_syn = split_pairs(syn_pairs, train_frac)
    train_sim, _ = split_pairs(sim_pairs, train_frac)
    _, test_ant = split_pairs(ant_pairs, train_frac)

    # HELD-OUT (M24 leakage fix): propagate over the TRAIN closeness edges only, so the
    # test_syn pairs scored below were never part of the placement's propagation graph.
    placed = place(words, graph, axes, cache=cache, depth=depth, alpha=alpha,
                   train_pairs=train_syn + train_sim)
    P = tanh_standardize(np.stack([placed[w] for w in words]))
    syn = np.array([(idx[a], idx[b]) for a, b in test_syn])
    ant = np.array([(idx[a], idx[b]) for a, b in test_ant])

    full = _disc(P, syn, ant)
    D = P.shape[1]
    all_cols = np.arange(D)
    # leave-one-out contribution of each axis
    importance = np.array([full - _disc(P, syn, ant, np.delete(all_cols, j)) for j in range(D)])
    order = list(np.argsort(importance)[::-1])
    names = axes.names

    if ks is None:
        ks = sorted({k for k in (5, 10, 15, 20, 30, 40, 60, 80, 120, D) if k <= D})
    curve: List[Tuple[int, float]] = [(k, _disc(P, syn, ant, np.array(order[:k]))) for k in ks]

    target = keep_frac * full
    minimal_k = next((k for k, d in curve if d >= target), D)
    kept = [names[j] for j in order[:minimal_k]]
    # kind breakdown of the kept axes (interpretability of the minimal set)
    kinds = {"prime": 0, "attribute": 0, "lexname": 0}
    for name in kept:
        if name.startswith("attr:"):
            kinds["attribute"] += 1
        elif name.startswith("lex:"):
            kinds["lexname"] += 1
        else:
            kinds["prime"] += 1
    return {
        "n_axes": D,
        "full_discrimination": full,
        "minimal_k": minimal_k,
        "kept_axes": kept,
        "kept_kinds": kinds,
        "curve": curve,
        "n_test_syn": len(test_syn),
        "n_test_ant": len(test_ant),
    }
