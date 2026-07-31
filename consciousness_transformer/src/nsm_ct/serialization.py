"""Serialization of parse trees into flat token sequences.

The model consumes parse structure as a flat token stream (concatenated tokens
with special separators), per the project brief. This is the *simple* encoding;
it throws away hierarchy and just emits a pre-order walk with node markers.

TODO(tree-encoding): replace this flat scheme with a hierarchical / tree-aware
encoding (e.g. structural position embeddings, recursive encoders, or
bracketing that the model can actually exploit). The flat form is a placeholder
so the input pathway exists end-to-end.
"""

from __future__ import annotations

from typing import List, Tuple

from .data_structures import ParseNode, ParseTree
from .tokenizer import NODE, TREE, TREE_END

# ---------------------------------------------------------------------------
# Lossless, invertible serialization of thought objects (trees of meaning).
# Unlike serialize_parse_tree (flat, lossy), this round-trips to identity so the
# parser's structure + meaning is preserved without loss, and so a future
# tree-decoder / reverse parser has an exact target. Distinctive marker tokens
# (unlikely to collide with content words) bracket the structure.
# ---------------------------------------------------------------------------
_OPEN, _CLOSE = "[(", ")]"
_LABEL, _REL, _IDX, _TOK, _MEAN = "@L", "@R", "@I", "@T", "@M"
THOUGHT_MARKERS = (_OPEN, _CLOSE, _LABEL, _REL, _IDX, _TOK, _MEAN)


def _serialize_node(node: ParseNode) -> List[str]:
    out: List[str] = [_OPEN, _LABEL, node.label]
    if node.relation is not None:
        out += [_REL, node.relation]
    if node.index is not None:
        out += [_IDX, str(node.index)]
    if node.token is not None:
        out += [_TOK, node.token]
    if node.meaning is not None:
        out += [_MEAN] + _serialize_node(node.meaning.root)
    for child in node.children:
        out += _serialize_node(child)
    out.append(_CLOSE)
    return out


def _deserialize_node(toks: List[str], pos: int) -> Tuple[ParseNode, int]:
    assert toks[pos] == _OPEN, f"expected {_OPEN!r} at {pos}, got {toks[pos]!r}"
    pos += 1
    node = ParseNode(label="")
    while True:
        t = toks[pos]
        if t == _LABEL:
            node.label = toks[pos + 1]; pos += 2
        elif t == _REL:
            node.relation = toks[pos + 1]; pos += 2
        elif t == _IDX:
            node.index = int(toks[pos + 1]); pos += 2
        elif t == _TOK:
            node.token = toks[pos + 1]; pos += 2
        elif t == _MEAN:
            root, pos = _deserialize_node(toks, pos + 1)
            node.meaning = ParseTree(root=root)
        elif t == _OPEN:
            child, pos = _deserialize_node(toks, pos)
            node.children.append(child)
        elif t == _CLOSE:
            pos += 1
            return node, pos
        else:
            raise ValueError(f"unexpected token {t!r} at {pos}")


def serialize_thought(thought) -> List[str]:
    """Serialize a ThoughtObject (or ParseTree) losslessly to string tokens."""
    tree = thought.tree if hasattr(thought, "tree") else thought
    return _serialize_node(tree.root)


def deserialize_thought(toks: List[str]) -> ParseTree:
    """Inverse of :func:`serialize_thought` — reconstruct the parse/meaning tree."""
    root, _ = _deserialize_node(list(toks), 0)
    return ParseTree(root=root)


def serialize_parse_tree(tree: ParseTree) -> List[str]:
    """Serialize a parse tree to a flat list of string tokens.

    Format (pre-order), per node: ``[NODE] <relation?> <label> <token?>`` where
    ``relation`` is the semantic role linking the node to its parent (e.g.
    SUBJECT, DESCRIPTION) when the parser provides one. Including the relation
    lets the model see structure/roles, not just surface words.

    The structure is still flat (hierarchy is not recoverable). TODO(tree-encoding):
    replace with a hierarchical / tree-aware encoding.

    Args:
        tree: The parse tree to serialize.

    Returns:
        A flat list of tokens suitable for :meth:`SimpleTokenizer.encode_tokens`.
    """
    out: List[str] = [TREE]
    for node in tree.iter_preorder():
        out.append(NODE)
        if node.relation is not None:
            out.append(node.relation)
        out.append(node.label)
        if node.token is not None:
            out.append(node.token)
    out.append(TREE_END)
    return out
