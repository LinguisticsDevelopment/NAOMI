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

from typing import List

from .data_structures import ParseTree
from .tokenizer import NODE, TREE, TREE_END


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
