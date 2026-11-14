"""Reverse parser: thought object (tree) -> surface text.

The parser maps text -> tree (lossless, meaning-preserving). This is the other
direction — read a thought object back out as text. It is the **seed** of the real
reverse parser that Stage D will use to read out the thinking model's *output*
trees; the Stage-A version simply emits the word-leaves in source order, which
round-trips the curriculum's simple sentences.

TODO(reverse-parser): a real reverse parser must realize surface form from
structure + meaning (agreement, function words, word order) — the inverse of the
quantum_parser grammar.
"""

from __future__ import annotations

from .data_structures import ParseTree


def thought_to_text(thought) -> str:
    """Render a thought object / parse tree back to surface text.

    Stage A: concatenate the word-leaves in source order (by node ``index`` when
    present, else tree order).
    """
    tree: ParseTree = thought.tree if hasattr(thought, "tree") else thought
    toks = [(n.index if n.index is not None else i, n.token)
            for i, n in enumerate(tree.iter_preorder()) if n.token is not None]
    toks.sort(key=lambda x: x[0])
    return " ".join(t for _, t in toks)
