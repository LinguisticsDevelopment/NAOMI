"""Placement by constraint satisfaction (M19.2).

Place each word as a position over the NAMED axes so the relations hold as
geometry. Two transparent, deterministic mechanisms (no opaque embedding, no
learned weights):

1. **Anchored coordinate.** A word loads on its prime axes (decomposition), its
   lexname axis, and its `attribute` axes — and the attribute axes are *signed*
   by the gloss-magnitude (hot=+temperature, cold=-temperature). So antonyms that
   share a dimension are anchored at opposite poles *on that axis* — antonymy
   folded INTO the position, non-circular (it never uses the antonym edge).

2. **Relational relaxation.** Minimize an explicit geometric energy —
   Σ_close ‖a-b‖² (pull synonyms/similar together) + μ‖p-anchor‖² (stay near the
   anchored meaning) — by gradient descent over the named axes. Convex and stable;
   the axes never rotate or mix, so a word's coordinate stays readable.

Evaluation is held-out and circularity-free: relaxation uses TRAIN closeness
edges; antonym separation is scored on HELD-OUT pairs whose own edge is never used.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .axes import MeaningAxes, feature_matrix
from .closeness import discrimination, split_pairs
from .meaning_value import cosine
from .polarity import gloss_polarity
from .relations import RelationGraph

_CLOSE_RELATIONS = ("synonym", "similar")


def anchored_coordinate(words, graph: RelationGraph, axes: MeaningAxes, *, cache=None, depth: int = 3) -> np.ndarray:
    """The anchored position matrix [N x D]: features, with attribute axes signed
    by gloss magnitude so antonym pairs sit at opposite poles on a shared axis."""
    M = feature_matrix(words, graph, axes, cache=cache, depth=depth)
    name_idx = {a.name: i for i, a in enumerate(axes.axes)}
    for r, w in enumerate(words):
        sign = gloss_polarity(w)
        if sign != 0.0:
            s = 1.0 if sign > 0 else -1.0
            for d in graph.attribute.get(w, []):
                j = name_idx.get(f"attr:{d}")
                if j is not None:
                    M[r, j] = s
    return M


def _adjacency(words, pairs) -> np.ndarray:
    idx = {w: i for i, w in enumerate(words)}
    n = len(words)
    A = np.zeros((n, n), dtype=np.float32)
    for a, b in pairs:
        i, j = idx.get(a), idx.get(b)
        if i is not None and j is not None:
            A[i, j] = A[j, i] = 1.0
    return A


def relax(anchor: np.ndarray, A_close: np.ndarray, *, iters: int = 20, alpha: float = 0.5) -> np.ndarray:
    """Stable label propagation: each word -> convex blend of its anchored meaning
    and the mean of its closeness neighbors. Pulls synonyms together while staying
    near the anchor; spectral radius <= 1 so it never diverges."""
    deg = A_close.sum(axis=1)
    deg[deg == 0] = 1.0
    Dinv = (1.0 / deg)[:, None]
    P = anchor.copy()
    for _ in range(iters):
        P = (1.0 - alpha) * anchor + alpha * (Dinv * (A_close @ P))
    return P


def place(
    words,
    graph: RelationGraph,
    axes: MeaningAxes,
    *,
    cache=None,
    depth: int = 3,
    train_pairs=None,
    iters: int = 20,
    alpha: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Anchored coordinate refined by relational relaxation -> position per word."""
    words = list(words)
    anchor = anchored_coordinate(words, graph, axes, cache=cache, depth=depth)
    if train_pairs is None:
        train_pairs = []
        for rel in _CLOSE_RELATIONS:
            train_pairs += graph.typed_pairs(rel)
    A_close = _adjacency(words, train_pairs)
    P = relax(anchor, A_close, iters=iters, alpha=alpha)
    return {w: P[i] for i, w in enumerate(words)}


def evaluate_placement(
    words,
    graph: RelationGraph,
    axes: MeaningAxes,
    *,
    cache=None,
    depth: int = 3,
    train_frac: float = 0.5,
    iters: int = 20,
    alpha: float = 0.5,
) -> Dict:
    """Held-out synonym-vs-antonym discrimination with PLAIN cosine: anchored
    coordinate (no relaxation) vs placed (relaxed). Circularity-free: relaxation
    uses only train closeness edges; antonyms are scored on held-out pairs."""
    words = list(words)
    wset = set(words)

    syn = sorted({tuple(sorted((w, s))) for w in words for s in graph.synonym.get(w, [])
                  if s in wset and s != w})
    sim = sorted({tuple(sorted((w, s))) for w in words for s in graph.similar.get(w, [])
                  if s in wset and s != w})
    ant = sorted({tuple(sorted(p)) for p in graph.typed_pairs("antonym")})

    train_syn, test_syn = split_pairs(syn, train_frac)
    train_sim, _ = split_pairs(sim, train_frac)
    _, test_ant = split_pairs(ant, train_frac)

    anchor = anchored_coordinate(words, graph, axes, cache=cache, depth=depth)
    coord_anchor = {w: anchor[i] for i, w in enumerate(words)}
    placed = place(words, graph, axes, cache=cache, depth=depth,
                   train_pairs=train_syn + train_sim, iters=iters, alpha=alpha)

    a_disc = discrimination(test_syn, test_ant, lambda a, b: cosine(coord_anchor[a], coord_anchor[b]), coord_anchor)
    p_disc = discrimination(test_syn, test_ant, lambda a, b: cosine(placed[a], placed[b]), placed)
    return {
        "anchored": a_disc[0],
        "placed": p_disc[0],
        "n_test_syn": a_disc[1],
        "n_test_ant": a_disc[2],
    }
