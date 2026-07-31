"""Null-aware bipolar representation + shared-axis metric (M21, checkpoint a).

The user's model, made concrete: each word has a value ONLY on the axes that apply
to it (Null elsewhere), every bipolar axis is a single signed scalar, and two words
are compared ONLY on the axes where BOTH are non-Null — so unrelated words (sharing
no applicable axes) are genuinely unrelated, not spuriously 0.3-similar.

- Bipolar axes: antonym prime-pairs collapse to one signed axis — GOOD/BAD→EVAL,
  BIG/SMALL→SIZE, MUCH_MANY/LITTLE_FEW→QTY. good=+1, bad=-1, grass=Null on EVAL.
- Other primes: presence axes (+1 if in the decomposition, Null otherwise).
- attribute axes: signed by gloss magnitude where the word has the dimension, Null otherwise.
- lexname: exactly one axis per word (its category), Null on the rest.

A word is ``(value, mask)``; ``mask`` is the Null indicator (1 = applies). The metric
is a distinctiveness-weighted (IDF) cosine over the *shared* applicable axes, so
generic axes (everyone is a noun) don't manufacture background similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..nsm_primes import PRIME_NAMES
from .canonicalization import canon_label
from .definition_graph import naive_decompose
from .polarity import POLARITY_PAIRS, gloss_polarity

_PRIME_SET = frozenset(PRIME_NAMES)
_PAIRED = {p for _, pos, neg in POLARITY_PAIRS for p in (pos, neg)}
_POS = {name: pos for name, pos, neg in POLARITY_PAIRS}
_NEG = {name: neg for name, pos, neg in POLARITY_PAIRS}


@dataclass
class SparseSpace:
    words: List[str]
    axes: List[str]
    value: np.ndarray   # [N, D] signed values (meaningful where mask=1)
    mask: np.ndarray    # [N, D] 1.0 = axis applies (non-Null), 0.0 = Null
    idf: np.ndarray     # [D] distinctiveness weight per axis

    def index(self, w: str) -> int:
        return self.words.index(w)


def _active_primes(tree) -> set:
    return {canon_label(n.label) for n in tree.iter_preorder() if canon_label(n.label) in _PRIME_SET}


def build_sparse_space(words, graph, *, cache=None, depth: int = 3, min_attribute_freq: int = 2) -> SparseSpace:
    words = list(words)
    N = len(words)

    # --- assemble the named axes ---
    bipolar = [name for name, _, _ in POLARITY_PAIRS]                       # EVAL, SIZE, QTY
    other_primes = [p for p in PRIME_NAMES if p not in _PAIRED]
    from collections import Counter
    attr_freq: Counter = Counter()
    for dims in graph.attribute.values():
        for d in dims:
            attr_freq[d] += 1
    attr_axes = [f"attr:{d}" for d, f in attr_freq.most_common() if f >= min_attribute_freq]
    lex_axes = [f"lex:{lx}" for lx in graph.lexnames()]
    axes = bipolar + other_primes + attr_axes + lex_axes
    col = {a: i for i, a in enumerate(axes)}
    D = len(axes)

    value = np.zeros((N, D), dtype=np.float32)
    mask = np.zeros((N, D), dtype=np.float32)

    for r, w in enumerate(words):
        tree = cache.decompose(w, depth) if cache is not None else naive_decompose(w, max_depth=depth)
        active = _active_primes(tree)
        # bipolar axes
        for name, pos, neg in POLARITY_PAIRS:
            has_pos, has_neg = pos in active, neg in active
            if has_pos or has_neg:
                j = col[name]
                mask[r, j] = 1.0
                value[r, j] = (1.0 if has_pos else 0.0) - (1.0 if has_neg else 0.0)
        # other prime presence axes
        for p in active:
            if p in col and p not in _PAIRED:
                j = col[p]
                mask[r, j] = 1.0
                value[r, j] = 1.0
        # attribute axes (signed by gloss magnitude)
        sign = gloss_polarity(w)
        s = 1.0 if sign > 0 else (-1.0 if sign < 0 else 0.0)
        for d in graph.attribute.get(w, []):
            j = col.get(f"attr:{d}")
            if j is not None:
                mask[r, j] = 1.0
                value[r, j] = s
        # lexname (one axis)
        lx = graph.lexname.get(w)
        if lx is not None:
            j = col.get(f"lex:{lx}")
            if j is not None:
                mask[r, j] = 1.0
                value[r, j] = 1.0

    df = mask.sum(axis=0)
    idf = np.log(N / (df + 1.0)).astype(np.float32)
    return SparseSpace(words=words, axes=axes, value=value, mask=mask, idf=idf)


def pair_similarity(space: SparseSpace, pairs: np.ndarray, *, mode: str = "full") -> np.ndarray:
    """Distinctiveness-weighted (IDF) similarity per pair.

    ``mode="full"`` (default): IDF-cosine over each word's FULL content (Null=0),
    so sharing one coincidental axis of many gives a *small* score — unrelated
    words become genuinely unrelated (random ~0.08). ``mode="masked"``: cosine over
    only the SHARED applicable axes — crisp for single-pair reads (hot/cold = -1.0)
    but ±1 on single-axis coincidences, so not for aggregate scoring."""
    V, M, idf = space.value, space.mask, space.idf
    Ai, Aj = V[pairs[:, 0]], V[pairs[:, 1]]
    if mode == "masked":
        sh = M[pairs[:, 0]] * M[pairs[:, 1]] * idf[None, :]
        dot = (Ai * Aj * sh).sum(1)
        ni = np.sqrt((Ai * Ai * sh).sum(1))
        nj = np.sqrt((Aj * Aj * sh).sum(1))
    else:  # full
        w = idf[None, :]
        dot = (Ai * Aj * w).sum(1)
        ni = np.sqrt((Ai * Ai * w).sum(1))
        nj = np.sqrt((Aj * Aj * w).sum(1))
    den = ni * nj
    ok = den > 1e-9
    return np.where(ok, dot / np.where(ok, den, 1.0), 0.0).astype(np.float32)


def similarity(space: SparseSpace, a: str, b: str, *, mode: str = "full") -> float:
    pr = np.array([[space.index(a), space.index(b)]])
    return float(pair_similarity(space, pr, mode=mode)[0])
