"""The deterministic clause==word reduction operator (M17.1).

This is the hard core. ``reduce`` flattens a definition-clause toward a fixpoint
at the prime floor; ``lexicalize`` collapses a (reduced) clause back to the
single word it defines, realizing *a word == its definition-clause*.

Determinism is structural — there is **no learned component**:
  1. **normalize** the clause to canonical form;
  2. **axis-substitution** — replace every surface WORD leaf with its bounded
     decomposition toward primes (memoized); primes/molecules/UNRESOLVED leaves
     are already at the floor and pass through;
  3. **renormalize** and repeat until the canonical key stops changing (fixpoint).

``lexicalize`` then maps the reduced clause to a word via an inverted
reduced-definition index: **exact normal-form match first**, then a grounded
**coordinate-closeness fallback** (cosine over the prime basis) when no exact
match exists — the alignment/containment step the user endorsed, computed over
grounded points rather than an opaque embedding.

Honest scope: exact round-trip on indexed words validates the full
decompose->reduce->normalize->lookup pipeline; the *generalization* signal is
whether a *perturbed* clause (a paraphrase) still recovers the word via the
closeness fallback. Both are reported, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data_structures import ParseNode, ParseTree
from .canonicalization import canon_label, normalize, tree_key
from .definition_graph import DEFAULT_MAX_DEPTH, definition_clause, naive_decompose
from .meaning_value import MeaningValue, axis_vector, cosine

_MAX_ITERS = 8


@lru_cache(maxsize=4096)
def _decompose_root(word: str, depth: int) -> ParseNode:
    """Memoized bounded decomposition root for a surface word."""
    return naive_decompose(word, max_depth=depth).root


def _substitute(node: ParseNode, depth: int) -> ParseNode:
    """Replace surface WORD leaves with their decomposition; recurse elsewhere."""
    if node.label == "WORD" and node.token:
        return _decompose_root(node.token, depth)
    new = ParseNode(label=node.label, token=node.token, relation=node.relation)
    new.children = [_substitute(c, depth) for c in node.children]
    return new


def reduce(clause: ParseTree, *, depth: int = DEFAULT_MAX_DEPTH, max_iters: int = _MAX_ITERS) -> MeaningValue:
    """Deterministically reduce *clause* to a prime-grounded fixpoint.

    Terminates when the canonical key is unchanged across a pass (or at
    ``max_iters`` as a hard backstop). Idempotent: ``reduce(reduce(c).tree)``
    has the same key as ``reduce(c)``.

    Note: substitution runs *before* the first canonicalization. Canonical form
    ignores surface ``token`` (equality is over meaning), but an un-decomposed
    ``WORD`` leaf's meaning *is* its token — so normalizing first would wrongly
    collapse distinct WORD leaves. Expanding them to primes first avoids that.
    """
    tree = clause
    key: Optional[str] = None
    for _ in range(max_iters):
        nxt = normalize(ParseTree(root=_substitute(tree.root, depth), text=tree.text))
        nkey = tree_key(nxt)
        if nkey == key:
            break
        tree, key = nxt, nkey
    return MeaningValue.from_tree(tree)


@dataclass
class ReducedDefinitionIndex:
    """Inverted index: reduced-definition -> the word it defines.

    ``by_key`` powers the exact lexicalize step; ``coords``/``words`` power the
    coordinate-closeness fallback.
    """

    by_key: Dict[str, str] = field(default_factory=dict)
    words: List[str] = field(default_factory=list)
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    depth: int = DEFAULT_MAX_DEPTH

    @classmethod
    def build(cls, words, *, depth: int = DEFAULT_MAX_DEPTH) -> "ReducedDefinitionIndex":
        by_key: Dict[str, str] = {}
        kept: List[str] = []
        rows: List[np.ndarray] = []
        for raw in words:
            w = raw.lower().strip()
            defn = definition_clause(w)
            if defn is None:
                continue
            mv = reduce(defn, depth=depth)
            k = tree_key(mv.tree)
            # First writer wins so the index is deterministic w.r.t. input order.
            by_key.setdefault(k, w)
            kept.append(w)
            rows.append(mv.axis_vec)
        coords = np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        return cls(by_key=by_key, words=kept, coords=coords, depth=depth)


def lexicalize(
    clause: ParseTree,
    index: ReducedDefinitionIndex,
    *,
    threshold: float = 0.5,
) -> Tuple[Optional[str], str, float]:
    """Collapse *clause* to the single word it defines.

    Returns ``(word_or_None, how, score)`` where ``how`` is ``"exact"`` (key
    match), ``"closest"`` (coordinate fallback above ``threshold``), or
    ``"none"``. ``score`` is 1.0 for exact, the cosine for closest, else 0.0.
    """
    mv = reduce(clause, depth=index.depth)
    key = tree_key(mv.tree)

    exact = index.by_key.get(key)
    if exact is not None:
        return exact, "exact", 1.0

    if index.coords.shape[0] == 0 or float(np.linalg.norm(mv.axis_vec)) == 0.0:
        return None, "none", 0.0

    # Coordinate-closeness fallback over grounded points.
    sims = np.array([cosine(mv.axis_vec, row) for row in index.coords], dtype=np.float32)
    best = int(np.argmax(sims))
    if float(sims[best]) >= threshold:
        return index.words[best], "closest", float(sims[best])
    return None, "none", float(sims[best])


def round_trip(word: str, index: ReducedDefinitionIndex, *, threshold: float = 0.5) -> Tuple[Optional[str], str, float]:
    """Reduce *word*'s own definition clause and try to recover *word*."""
    defn = definition_clause(word)
    if defn is None:
        return None, "none", 0.0
    return lexicalize(defn, index, threshold=threshold)


def perturbed_clause(word: str) -> Optional[ParseTree]:
    """A paraphrase of *word*'s definition: drop its last content word.

    Used to probe generalization — an exact key match will usually miss, so
    recovery must come from the coordinate-closeness fallback.
    """
    defn = definition_clause(word)
    if defn is None:
        return None
    leaves = defn.root.children
    if len(leaves) <= 1:
        return None
    head = ParseNode(label=defn.root.label, token=defn.root.token)
    head.children = [ParseNode(label=c.label, token=c.token) for c in leaves[:-1]]
    return ParseTree(root=head)
