"""Fused Null + propagation scoring (M22.1).

M21 found the Null sparse representation (separation) and relational propagation
(antonyms) are complementary. This fuses them into ONE score:

    fused(a, b) = relatedness(a, b) · propagation_sim(a, b)

- ``relatedness`` — the Null win: IDF-weighted mask-overlap, a "are these even
  comparable?" gate (unrelated senses share no applicable axes → ~0).
- ``propagation_sim`` — the antonym win: cosine of the coordinate after label
  propagation over the CLOSE edges (similar_to + co-hyponyms), which clusters
  near-synonyms and thereby separates antonyms.

So Null decides *if* two senses are related; propagation decides *how* (near vs
opposite). Held-out: relatedness gate uses the fixed sparse mask; propagation uses
train close-edges only.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .placement import relax
from .sparse_value import SparseSpace


def relatedness(space: SparseSpace, pairs: np.ndarray) -> np.ndarray:
    """IDF-weighted overlap of applicable axes in [0, 1] (the comparability gate)."""
    M, idf = space.mask, space.idf
    sh = (M[pairs[:, 0]] * M[pairs[:, 1]] * idf).sum(1)
    ta = (M[pairs[:, 0]] * idf).sum(1)
    tb = (M[pairs[:, 1]] * idf).sum(1)
    denom = np.minimum(ta, tb)
    denom[denom < 1e-9] = 1.0
    return np.clip(sh / denom, 0.0, 1.0)


def co_hyponym_pairs(graph, *, max_per_group: int = 40) -> List[Tuple[str, str]]:
    """Senses sharing a hypernym are co-hyponyms (a dense 'close' signal)."""
    groups = {}
    for sid, hs in graph.hypernym.items():
        for h in hs:
            groups.setdefault(h, []).append(sid)
    pairs = set()
    for members in groups.values():
        members = members[:max_per_group]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add(tuple(sorted((members[i], members[j]))))
    return sorted(pairs)


def propagate(space: SparseSpace, close_pairs, *, iters: int = 20, alpha: float = 0.7) -> np.ndarray:
    """Label-propagate the sparse coordinate over the close edges (antonym lever)."""
    n = space.value.shape[0]
    idx = {w: i for i, w in enumerate(space.words)}
    A = np.zeros((n, n), dtype=np.float32)
    for a, b in close_pairs:
        ia, ib = idx.get(a), idx.get(b)
        if ia is not None and ib is not None:
            A[ia, ib] = A[ib, ia] = 1.0
    return relax(space.value, A, iters=iters, alpha=alpha)


def _cos(P: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    Ai, Aj = P[pairs[:, 0]], P[pairs[:, 1]]
    ni = np.linalg.norm(Ai, axis=1)
    nj = np.linalg.norm(Aj, axis=1)
    den = ni * nj
    ok = den > 1e-9
    return np.where(ok, (Ai * Aj).sum(1) / np.where(ok, den, 1.0), 0.0)


def fused_similarity(space: SparseSpace, P: np.ndarray, pairs: np.ndarray, *, threshold: float = 0.15) -> np.ndarray:
    """The fused score: a THRESHOLD comparability gate on propagated discrimination.

    If two senses share too little (relatedness < threshold) they are unrelated → 0;
    otherwise the propagated cosine supplies the signed near-vs-opposite score. Unlike
    a product gate (which dampens related pairs and *hurt* discrimination), the
    threshold zeroes only the truly-unrelated, so it cuts random overlap while
    preserving synonym-vs-antonym discrimination."""
    gate = relatedness(space, pairs) >= threshold
    return np.where(gate, _cos(P, pairs), 0.0)
