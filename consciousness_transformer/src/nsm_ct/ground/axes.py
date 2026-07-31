"""The interpretable axis set + dimensionality diagnostic (M19.1).

Two deliverables:

1. **Assemble the candidate axis set** — every axis NAMED and interpretable (the
   hard invariant): the NSM primes (semantic floor) + the ``attribute`` noun
   dimensions the relations hand us (temperature, size, age, …) + the 44 ``lexname``
   categories. No anonymous word2vec dimensions.

2. **Dimensionality diagnostic** — estimate how many independent axes the relational
   structure actually needs (the empirical "minimum axes of meaning" number) from the
   eigenspectrum of the word-word *closeness* affinity (synonym + similar +
   derivational edges). Reported, not used as an opaque representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..nsm_primes import PRIME_NAMES
from .relations import RelationGraph

# Relations that assert two words are CLOSE in meaning (for the affinity spectrum).
_CLOSE_RELATIONS = ("synonym", "similar", "derivational")


@dataclass(frozen=True)
class Axis:
    name: str
    kind: str          # "prime" | "attribute" | "lexname"
    provenance: str


@dataclass
class MeaningAxes:
    """An ordered, fully-named interpretable axis set."""

    axes: List[Axis] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return [a.name for a in self.axes]

    @property
    def dim(self) -> int:
        return len(self.axes)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def by_kind(self, kind: str) -> List[str]:
        return [a.name for a in self.axes if a.kind == kind]

    @classmethod
    def assemble(cls, graph: RelationGraph, *, min_attribute_freq: int = 2) -> "MeaningAxes":
        """Primes + frequent attribute dimensions + lexname categories — all named."""
        axes: List[Axis] = []
        seen = set()

        def add(name: str, kind: str, prov: str):
            key = (kind, name)
            if key not in seen:
                seen.add(key)
                axes.append(Axis(name=name, kind=kind, provenance=prov))

        for p in PRIME_NAMES:
            add(p, "prime", "nsm-seed")

        # attribute dimensions, frequency-filtered so noise (one-off dims) drops out
        from collections import Counter
        attr_freq: Counter = Counter()
        for dims in graph.attribute.values():
            for d in dims:
                attr_freq[d] += 1
        for d, f in attr_freq.most_common():
            if f >= min_attribute_freq:
                add(f"attr:{d}", "attribute", f"attribute(freq{f})")

        for lx in graph.lexnames():
            add(f"lex:{lx}", "lexname", "lexname")

        return cls(axes=axes)

    def summary(self) -> Dict[str, int]:
        return {
            "total": self.dim,
            "prime": len(self.by_kind("prime")),
            "attribute": len(self.by_kind("attribute")),
            "lexname": len(self.by_kind("lexname")),
        }


def feature_matrix(words, graph: RelationGraph, axes: "MeaningAxes", *, cache=None, depth: int = 3) -> np.ndarray:
    """Binary word x named-axis matrix: a word loads on a prime axis if its
    decomposition contains it, on an attribute axis if it has that dimension, on a
    lexname axis if that is its category. The structure whose rank = #axes used."""
    from .canonicalization import canon_label
    from .definition_graph import naive_decompose

    name_idx = {a.name: i for i, a in enumerate(axes.axes)}
    M = np.zeros((len(words), axes.dim), dtype=np.float32)
    for r, w in enumerate(words):
        tree = cache.decompose(w, depth) if cache is not None else naive_decompose(w, max_depth=depth)
        for node in tree.iter_preorder():
            j = name_idx.get(canon_label(node.label))     # prime axes are bare prime names
            if j is not None:
                M[r, j] = 1.0
        for d in graph.attribute.get(w, []):
            j = name_idx.get(f"attr:{d}")
            if j is not None:
                M[r, j] = 1.0
        lx = graph.lexname.get(w)
        if lx is not None:
            j = name_idx.get(f"lex:{lx}")
            if j is not None:
                M[r, j] = 1.0
    return M


def dimensionality_spectrum(
    graph: RelationGraph,
    axes: "MeaningAxes",
    *,
    cache=None,
    depth: int = 3,
    max_words: int = 3000,
    mass: float = 0.9,
) -> Dict:
    """Empirical "minimum axes" estimate: how many of the named candidate axes carry
    the structure, from the singular-value spectrum of the word x axis matrix.

    Reports the number of axes to explain *mass* of the spectral energy and the
    participation ratio (effective number of axes). Interpretable throughout — the
    axes are named; this just says how many are load-bearing.
    """
    words = graph.words()[:max_words]
    if len(words) < 3 or axes.dim == 0:
        return {"n_words": len(words), "n_axes": axes.dim, "intrinsic_dim_mass": None,
                "participation_ratio": None, "top_singular_values": []}

    M = feature_matrix(words, graph, axes, cache=cache, depth=depth)
    sv = np.linalg.svd(M, compute_uv=False)
    energy = sv ** 2
    total = float(energy.sum()) or 1.0
    cum = np.cumsum(energy) / total
    dim_mass = int(np.searchsorted(cum, mass) + 1)
    participation = float((energy.sum() ** 2) / (np.square(energy).sum() + 1e-12))

    return {
        "n_words": len(words),
        "n_axes": axes.dim,
        "intrinsic_dim_mass": dim_mass,        # #axes for `mass` of the energy
        "participation_ratio": participation,  # effective number of axes
        "top_singular_values": [float(x) for x in sv[:15]],
    }
