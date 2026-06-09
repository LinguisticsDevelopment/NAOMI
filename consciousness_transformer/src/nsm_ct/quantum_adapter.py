"""Convert experimental ``quantum_parser`` output into our :class:`ParseTree`.

This is a thin, optional adapter. It is only imported when the
:class:`~nsm_ct.input_encoder.ParserInputEncoder` is used, and it never imports
``quantum_parser`` at module load — the caller passes already-built objects.

The ``quantum_parser`` ``Hypothesis`` is a graph: ``nodes`` plus typed ``edges``
(``edge.type`` is a ``ConnectionType`` like SUBJECT/OBJECT/DESCRIPTION, with
integer ``parent``/``child`` node indices). We turn the unconsumed (root) node
into a tree, labelling each node by its ``NodeType`` and each child by the
``ConnectionType`` that links it to its parent.
"""

from __future__ import annotations

from typing import Any, List

from .data_structures import ParseNode, ParseTree


def _node_label(node: Any) -> str:
    t = getattr(node, "type", None)
    return getattr(t, "name", str(t)) if t is not None else "NODE"


def _node_token(node: Any):
    value = getattr(node, "value", None)
    return getattr(value, "text", None) if value is not None else None


def _node_index(node: Any):
    i = getattr(node, "index", None)
    return int(i) if isinstance(i, int) and i >= 0 else None


def _build(idx: int, nodes: list, edges: list, relation: str | None, seen: set) -> ParseNode:
    node = nodes[idx]
    pnode = ParseNode(
        label=_node_label(node), token=_node_token(node),
        relation=relation, index=_node_index(node),
    )
    if idx in seen:  # guard against cycles in an experimental parser
        return pnode
    seen.add(idx)
    for edge in edges:
        if getattr(edge, "parent", None) == idx:
            child_rel = getattr(getattr(edge, "type", None), "name", "REL")
            pnode.children.append(_build(edge.child, nodes, edges, child_rel, seen))
    return pnode


def hypothesis_to_tree(hyp: Any, text: str = "") -> ParseTree:
    """Convert a ``quantum_parser`` ``Hypothesis`` to a :class:`ParseTree`.

    Args:
        hyp: A ``Hypothesis`` with ``.nodes`` and ``.edges`` (and ``.get_unconsumed()``).
        text: Original sentence, for provenance.
    """
    nodes = list(getattr(hyp, "nodes", []))
    edges = list(getattr(hyp, "edges", []))
    try:
        roots: List[int] = hyp.get_unconsumed()
    except Exception:  # pragma: no cover - defensive
        roots = [0] if nodes else []

    seen: set = set()
    if len(roots) == 1:
        root = _build(roots[0], nodes, edges, None, seen)
    else:
        # Multiple (or zero) roots: wrap them under a synthetic ROOT node.
        root = ParseNode(label="ROOT", token=None)
        for r in roots:
            root.children.append(_build(r, nodes, edges, None, seen))
    return ParseTree(root=root, text=text)
