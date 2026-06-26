"""Decomposition cache for scale (M18.0).

The basis search re-decomposes the whole corpus once per candidate axis, which is
hopeless at 10k words if every call hits WordNet. Two observations make it cheap:

1. The **base** decomposition (``extra_axes=∅``) depends only on ``(word, depth)``
   — compute it once, cache it (in memory and optionally on disk via the lossless
   ``serialization``).
2. Promoting a word *w* to an axis only changes nodes whose ``token`` is *w*:
   ``naive_decompose(extra_axes={w, …})`` is reproduced from the base tree by a
   **prune+relabel** of those nodes into atomic axis leaves — pure in-memory tree
   work, no WordNet calls.

So ``DecompCache.decompose(word, depth, extra)`` returns the extra-aware tree at
in-memory speed after a one-time warm of the base trees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Optional

from ..data_structures import ParseNode, ParseTree
from ..serialization import deserialize_thought, serialize_thought
from .definition_graph import DEFAULT_MAX_DEPTH, naive_decompose

# Labels that mark a node as not-yet-grounded (a gloss head or an unresolved leaf)
# — exactly the nodes a promoted axis may collapse. Prime/molecule leaves are
# never relabelled.
_RELABELLABLE = frozenset({"EXPLICATION", "UNRESOLVED"})


def apply_extra_axes(tree: ParseTree, extra: FrozenSet[str]) -> ParseTree:
    """Reproduce ``naive_decompose(..., extra_axes=extra)`` from a *base* tree.

    Any node whose ``token`` is a promoted axis and whose label is a gloss-head /
    unresolved marker is collapsed to an atomic axis leaf (its subtree dropped).
    """
    if not extra:
        return tree

    def rec(node: ParseNode) -> ParseNode:
        if node.token in extra and node.label in _RELABELLABLE:
            return ParseNode(label=node.token, token=node.token)
        new = ParseNode(label=node.label, token=node.token, relation=node.relation)
        new.children = [rec(c) for c in node.children]
        return new

    return ParseTree(root=rec(tree.root), text=tree.text)


class DecompCache:
    """In-memory (optionally disk-persisted) cache of base decompositions."""

    def __init__(self, *, depth: int = DEFAULT_MAX_DEPTH, path: Optional[str] = None) -> None:
        self.depth = depth
        self.path = Path(path) if path else None
        self._base: Dict[str, ParseTree] = {}
        if self.path and self.path.exists():
            self.load()

    def base(self, word: str) -> ParseTree:
        w = word.lower().strip()
        tree = self._base.get(w)
        if tree is None:
            tree = naive_decompose(w, max_depth=self.depth)
            self._base[w] = tree
        return tree

    def warm(self, words: Iterable[str]) -> "DecompCache":
        for w in words:
            self.base(w)
        return self

    def decompose(self, word: str, depth: int, extra: FrozenSet[str] = frozenset()) -> ParseTree:
        """Extra-aware decomposition at in-memory speed (depth must match warm depth)."""
        return apply_extra_axes(self.base(word), extra)

    # -- persistence (lossless via serialize_thought) ----------------------
    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"depth": self.depth,
                "trees": {w: serialize_thought(t) for w, t in self._base.items()}}
        self.path.write_text(json.dumps(blob), encoding="utf-8")

    def load(self) -> None:
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
            if blob.get("depth") == self.depth:
                self._base = {w: deserialize_thought(toks) for w, toks in blob["trees"].items()}
        except Exception:  # pragma: no cover - corrupt cache -> rebuild lazily
            self._base = {}

    def __len__(self) -> int:
        return len(self._base)
