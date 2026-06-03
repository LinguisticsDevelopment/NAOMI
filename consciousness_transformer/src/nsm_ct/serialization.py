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

    Format (pre-order): ``[TREE] [NODE] <label> <token?> [NODE] <label> ... [/TREE]``

    Leaf nodes contribute their label followed by their surface token; internal
    nodes contribute just their label. The structure is intentionally lossy.

    Args:
        tree: The parse tree to serialize.

    Returns:
        A flat list of tokens suitable for :meth:`SimpleTokenizer.encode_tokens`.
    """
    out: List[str] = [TREE]
    for node in tree.iter_preorder():
        out.append(NODE)
        out.append(node.label)
        if node.token is not None:
            out.append(node.token)
    out.append(TREE_END)
    return out
