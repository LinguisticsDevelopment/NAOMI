"""Normalized coordinates + absence-aware similarity (M20.0).

Diagnostics on the M19 space found two coupled problems: unrelated pairs overlap
(random-pair cosine ~0.32, because the coordinate is dense and *absence is encoded
as 0* — no disagreement) and axes are incommensurable (per-axis std varies ~5x, so
cosine is dominated by a few high-variance axes). The user's instinct — lock axes
to a consistent bounded scale — is right; the literal "absence=-1 + cosine" version
backfires (everything agrees on the huge shared -1 background, cosines crush to
~0.91). The correct version normalizes each axis to a commensurable scale AND
centers it, so the shared background falls at the origin and unrelated pairs become
near-orthogonal.

Transforms (each fit on the corpus coordinate, no relation labels — not circular):
- ``standardize``   — per-axis z-score (mean 0, unit variance): commensurable + centered.
- ``minmax_bound``  — per-axis to [-1,1] (the user's bounded interpretable scale).
- ``center``        — subtract the corpus-mean coordinate only (kills shared background).

Axes stay named; normalization only rescales each named axis, never mixes them.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

_EPS = 1e-6


def standardize(P: np.ndarray) -> np.ndarray:
    """Per-axis z-score: (x - mean) / std. Commensurable and mean-centered."""
    mu = P.mean(axis=0)
    sd = P.std(axis=0)
    sd[sd < _EPS] = 1.0
    return (P - mu) / sd


def minmax_bound(P: np.ndarray) -> np.ndarray:
    """Per-axis affine map to [-1, 1] (the user's bounded interpretable scale)."""
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    rng = hi - lo
    rng[rng < _EPS] = 1.0
    return 2.0 * (P - lo) / rng - 1.0


def center(P: np.ndarray) -> np.ndarray:
    """Subtract the corpus-mean coordinate (the shared 'background' direction)."""
    return P - P.mean(axis=0)


def tanh_standardize(P: np.ndarray) -> np.ndarray:
    """z-score then squash to [-1, 1] with tanh: commensurable AND bounded/readable
    (a word's value on each named axis is a scalar in [-1, 1], the user's intent),
    while preserving the clean separation z-scoring buys (tanh is monotonic)."""
    return np.tanh(standardize(P))


# Recommended default: tanh(z-score) — bounded [-1,1] (interpretable, the user's
# intent) AND the best measured synonym-vs-antonym discrimination, while cutting the
# spurious random-pair overlap (raw 0.32 -> 0.15). The literal min-max-to-[-1,1]
# backfires (random ~0.98, shared-background domination), as does raw absence=0.
DEFAULT_NORMALIZATION = "tanh"

_TRANSFORMS = {
    "raw": lambda P: P,
    "standardize": standardize,
    "minmax": minmax_bound,
    "center": center,
    "tanh": tanh_standardize,
}


def normalize_matrix(P: np.ndarray, kind: str = "standardize") -> np.ndarray:
    if kind not in _TRANSFORMS:
        raise ValueError(f"unknown normalization {kind!r}; pick {sorted(_TRANSFORMS)}")
    return _TRANSFORMS[kind](P)


def normalize_coords(coord: Dict[str, np.ndarray], kind: str = "standardize") -> Dict[str, np.ndarray]:
    """Normalize a word->vector mapping, fit on its own corpus."""
    words: List[str] = list(coord)
    P = np.stack([coord[w] for w in words])
    Pn = normalize_matrix(P, kind)
    return {w: Pn[i] for i, w in enumerate(words)}
