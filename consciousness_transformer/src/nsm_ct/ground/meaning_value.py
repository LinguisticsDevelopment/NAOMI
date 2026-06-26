"""MeaningValue — the coupled (structured object + grounded coordinate) (M17.0).

A *meaning value* is the unit this milestone generates. It deliberately keeps
**both** representations the user insisted on:

- ``tree``     — the arbitrary-shaped :class:`ParseTree` (structure preserved,
  never flattened to a bag — the flattening that made the legacy model ignore
  meaning).
- ``axis_vec`` — a coordinate over an interpretable basis (NSM-65 in M17.0,
  extended in M17.2). Each dimension is a real axis, so closeness has a witness.
- ``handle``   — an optional fast nearest-neighbour address (the TPR encoding).

Closeness is computed over grounded points (``axis_vec``), with the structure
available for the alignment/containment refinement used by the reduction
operator (M17.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Set, Tuple

import numpy as np

from ..data_structures import ParseTree
from ..nsm_primes import PRIME_NAMES
from .canonicalization import canon_label, normalize


def axis_vector(tree: ParseTree, axes: Sequence[str] = PRIME_NAMES) -> np.ndarray:
    """Count occurrences of each basis axis among the tree's (canonical) labels.

    A multiset coordinate: dimension *i* is how many nodes fold to axis ``axes[i]``.
    Non-axis labels (``UNRESOLVED``, molecule names, ordinary words) contribute
    nothing — so an un-grounded decomposition has a *smaller* coordinate, which
    is the honest signal the baseline harness reports.
    """
    index = {a: i for i, a in enumerate(axes)}
    v = np.zeros(len(axes), dtype=np.float32)
    for node in tree.iter_preorder():
        i = index.get(canon_label(node.label))
        if i is not None:
            v[i] += 1.0
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] for non-negative count vectors (0 if either is empty)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard overlap of two label sets (1.0 if both empty)."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


@dataclass
class MeaningValue:
    """A word/sense's generated meaning: structure + grounded coordinate."""

    tree: ParseTree
    axis_vec: np.ndarray
    handle: Optional[np.ndarray] = None
    axes: Tuple[str, ...] = tuple(PRIME_NAMES)

    @classmethod
    def from_tree(
        cls,
        tree: ParseTree,
        *,
        axes: Sequence[str] = PRIME_NAMES,
        codec=None,
    ) -> "MeaningValue":
        """Build a value from a meaning tree (canonicalized first).

        ``codec`` (an optional :class:`~nsm_ct.tpr.TPRCodec`) supplies the fast
        ``handle`` address; omit it for the symbolic-only path.
        """
        norm = normalize(tree)
        vec = axis_vector(norm, axes)
        handle = codec.encode_tree(norm) if codec is not None else None
        return cls(tree=norm, axis_vec=vec, handle=handle, axes=tuple(axes))

    def active_axes(self) -> Set[str]:
        """The set of basis axes present in this value (its interpretable witness)."""
        return {a for a, x in zip(self.axes, self.axis_vec) if x > 0}

    def similarity(self, other: "MeaningValue") -> float:
        """Cosine closeness of the two grounded coordinates."""
        return cosine(self.axis_vec, other.axis_vec)
