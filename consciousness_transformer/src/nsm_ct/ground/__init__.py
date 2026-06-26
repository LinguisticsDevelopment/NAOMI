"""ground/ — the word-meaning-value generator (M17).

A grounding substrate that turns a word/sense into a *meaning value* (a coupled
structured object + grounded coordinate) over an interpretable basis, and
measures understanding by **clause==word self-consistency** rather than by
looking words up in the DeepNSM/gold explication dictionary.

This subpackage *imports* the existing primitives (ParseTree, the NSM primes,
the TPR codec, WordNet) and never modifies them. The DeepNSM explication store
is used only as a held-out external *check*, never as the runtime answer.

Modules (built per milestone):
- ``canonicalization`` (M17.0) — deterministic canonical form for meaning trees.
- ``meaning_value``   (M17.0) — the coupled object + grounded coordinate.
- ``definition_graph``(M17.0) — word -> definition-clause + relational edges.
- ``clause_self_consistency`` (M17.0) — the baseline understanding harness.
- ``reduction``       (M17.1) — the deterministic clause==word reduction operator.
- ``semantic_axes``   (M17.2) — the extensible basis registry.
- ``basis_search``    (M17.2) — MDL-driven basis discovery.
"""

from __future__ import annotations

from .canonicalization import canon_label, normalize, tree_key
from .meaning_value import MeaningValue, axis_vector, cosine, jaccard

__all__ = [
    "canon_label",
    "normalize",
    "tree_key",
    "MeaningValue",
    "axis_vector",
    "cosine",
    "jaccard",
]
