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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


def _node_flags(node: Any) -> List[str]:
    """Node subtype flag names (e.g. ``["PASSIVE"]``), or ``[]``.

    quantum_parser's ``Node.flags`` is a ``List[SubType]`` (an Enum stamped by
    ``push_subtypes`` -- PASSIVE by ``aux1`` since M38, QUESTION by
    ``question1`` since M49); we only need the names, so this stays decoupled
    from the ``SubType`` enum itself (no quantum_parser import here).
    """
    return [getattr(f, "name", str(f)) for f in (getattr(node, "flags", None) or [])]


def _build(idx: int, nodes: list, edges: list, relation: str | None, seen: set) -> ParseNode:
    node = nodes[idx]
    pnode = ParseNode(
        label=_node_label(node), token=_node_token(node),
        relation=relation, index=_node_index(node),
        flags=_node_flags(node),
    )
    if idx in seen:  # guard against cycles in an experimental parser
        return pnode
    seen.add(idx)
    for edge in edges:
        if getattr(edge, "parent", None) == idx:
            child_rel = getattr(getattr(edge, "type", None), "name", "REL")
            pnode.children.append(_build(edge.child, nodes, edges, child_rel, seen))
    return pnode


@dataclass
class HypGraph:
    """A flat, dependency-free view of a ``quantum_parser`` hypothesis graph.

    Unlike :func:`hypothesis_to_tree` — which walks parent→child only and so
    **drops** coordination/negation structure (coordinated elements point *up* to
    their coordinator, leaving them unreachable from the root) — this keeps every
    node and every typed edge. It is the substrate for
    :func:`nsm_ct.clause.extract_discourse`, which needs the COORDINATION /
    SUBORDINATION / MODIFIER edges the tree view loses.
    """

    nodes: List[Tuple[int, str, Optional[str]]]   # (index, label, token)
    edges: List[Tuple[str, int, int]]             # (type, parent, child)
    roots: List[int] = field(default_factory=list)
    flags: Dict[int, List[str]] = field(default_factory=dict)  # idx -> subtype names (M50)

    def node(self, idx: int) -> Optional[Tuple[int, str, Optional[str]]]:
        return next((n for n in self.nodes if n[0] == idx), None)

    def token(self, idx: int) -> Optional[str]:
        n = self.node(idx)
        return n[2] if n else None

    def label(self, idx: int) -> Optional[str]:
        n = self.node(idx)
        return n[1] if n else None

    def edges_of(self, etype: str) -> List[Tuple[int, int]]:
        return [(p, c) for (t, p, c) in self.edges if t == etype]

    def flags_of(self, idx: int) -> List[str]:
        """Subtype flag names on node ``idx`` (e.g. ``["PASSIVE"]``), or ``[]``."""
        return self.flags.get(idx, [])


def hypothesis_to_graph(hyp: Any) -> HypGraph:
    """Convert a ``quantum_parser`` ``Hypothesis`` to a flat :class:`HypGraph`."""
    nodes: List[Tuple[int, str, Optional[str]]] = []
    flags: Dict[int, List[str]] = {}
    for i, node in enumerate(getattr(hyp, "nodes", [])):
        idx = _node_index(node)
        node_idx = idx if idx is not None else i
        nodes.append((node_idx, _node_label(node), _node_token(node)))
        node_flags = _node_flags(node)
        if node_flags:
            flags[node_idx] = node_flags
    edges: List[Tuple[str, int, int]] = []
    for edge in getattr(hyp, "edges", []):
        etype = getattr(getattr(edge, "type", None), "name", "REL")
        edges.append((etype, getattr(edge, "parent", -1), getattr(edge, "child", -1)))
    try:
        roots = list(hyp.get_unconsumed())
    except Exception:  # pragma: no cover - defensive
        roots = [n[0] for n in nodes[:1]]
    return HypGraph(nodes=nodes, edges=edges, roots=roots, flags=flags)


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
