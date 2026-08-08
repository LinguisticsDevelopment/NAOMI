"""Meaning graph — the ClausePsyche substrate (numpy, no torch).

One graph holds every unit of meaning as a **node**: concept/word nodes,
referent (entity) nodes, clause nodes, and operator nodes. Each node carries a
*lossy* vector **handle** (a fast, content-addressable address) AND a pointer to
its **lossless stored structure** (a :func:`serialize_thought` token list). The
graph is where collapse/expand (definition) and the STM/LTM discourse live; the
vector never has to be invertible because the exact truth is the stored
structure. See the plan / RESEARCH_NOTES.

This module is the deterministic structural layer only — no learned parts. The
collapse/expand operations live in :mod:`nsm_ct.collapse`; operator-nodes in the
``operators`` section here; the STM/LTM discourse engine in
:mod:`nsm_ct.clause_psyche_graph`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .data_structures import ParseTree
from .serialization import serialize_thought
from .tpr import TPRCodec


class NodeKind(Enum):
    """What a graph node represents."""

    CONCEPT = "CONCEPT"      # a word/meaning (collapsed structure → a handle)
    REFERENT = "REFERENT"    # a specific individual (entity variable atom)
    CLAUSE = "CLAUSE"        # a predicate + role-bound argument slots
    OPERATOR = "OPERATOR"    # not / and / or / maybe over clause-argument(s)


# Edge types. SLOT carries a relation label (the role of the bound filler).
SLOT = "SLOT"               # clause -> filler node (rel = SUBJECT / PLACE / ...)
ABOUT = "ABOUT"             # clause -> referent it mentions
COREF = "COREF"             # referent <-> referent (same individual)
OPERATES_ON = "OPERATES_ON" # operator -> clause it scopes
SUPERSEDES = "SUPERSEDES"   # newer clause -> older clause (recency)
DEFINES = "DEFINES"         # concept -> its explication structure (optional)


@dataclass(frozen=True)
class Edge:
    """A typed directed edge ``(etype, src, dst)`` with an optional relation."""

    etype: str
    src: int
    dst: int
    rel: Optional[str] = None


@dataclass
class GraphNode:
    """A node: a lossy vector ``handle`` + a lossless ``structure`` pointer.

    Attributes:
        nid: Stable integer id within its graph.
        kind: One of :class:`NodeKind`.
        handle: The lossy address vector (shape ``(codec.dim,)``).
        structure: Lossless ``serialize_thought`` token list for CONCEPT/CLAUSE
            nodes; ``None`` for atomic REFERENT/OPERATOR nodes.
        label: Concept/operator name or referent name (``None`` for clauses).
        meta: Bookkeeping — recency counter, truth tag, provenance ``text``, ...
    """

    nid: int
    kind: NodeKind
    handle: np.ndarray
    structure: Optional[List[str]] = None
    label: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class ClausePayload:
    """The structured content of a CLAUSE node (kept beside its matrix/handle)."""

    predicate_nid: int
    slots: List[Tuple[str, int]] = field(default_factory=list)  # (relation, filler nid)
    operator_nids: List[int] = field(default_factory=list)


class MeaningGraph:
    """A graph of meaning nodes + typed edges, addressed by handle or by id.

    Co-reference is enforced structurally: there is exactly one REFERENT node per
    name (``referent_index``) and one CONCEPT node per word/label
    (``concept_index``), so the shared ``is`` / ``mary`` node is reused rather
    than duplicated. What distinguishes two clauses is *which* referent nids their
    slots bind to.
    """

    def __init__(self, codec: TPRCodec) -> None:
        self.codec = codec
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: List[Edge] = []
        self.payloads: Dict[int, ClausePayload] = {}      # clause nid -> payload
        self.referent_index: Dict[str, int] = {}          # name -> REFERENT nid
        self.concept_index: Dict[str, int] = {}           # label -> CONCEPT nid
        self._next_nid = 0
        self._out: Dict[int, List[Edge]] = {}
        self._in: Dict[int, List[Edge]] = {}
        self._by_etype: Dict[str, List[Edge]] = {}

    # -- nodes ----------------------------------------------------------------
    def _new_nid(self) -> int:
        nid = self._next_nid
        self._next_nid += 1
        return nid

    def add_node(
        self,
        kind: NodeKind,
        handle: np.ndarray,
        *,
        structure: Optional[List[str]] = None,
        label: Optional[str] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> int:
        """Create a node and return its id."""
        nid = self._new_nid()
        self.nodes[nid] = GraphNode(
            nid=nid, kind=kind, handle=np.asarray(handle, dtype=np.float32),
            structure=structure, label=label, meta=dict(meta or {}),
        )
        return nid

    def add_referent(self, name: str) -> int:
        """One REFERENT node per name (co-reference, not duplication)."""
        key = name.lower()
        if key in self.referent_index:
            return self.referent_index[key]
        nid = self.add_node(
            NodeKind.REFERENT, self.codec.filler_vec("var:" + key), label=key,
        )
        self.referent_index[key] = nid
        return nid

    def add_concept(
        self,
        label: str,
        tree: ParseTree,
        *,
        handle_fn: Optional[Callable[[str], Optional[np.ndarray]]] = None,
    ) -> int:
        """One CONCEPT node per word/label; handle = contract(encode(tree)).

        ``handle_fn`` is an optional M33 hook (e.g. ``nsm_ct.usvs_bridge.
        usvs_handle`` closed over ``d``): if given ``label`` and it returns a
        vector, that (unit-normalized) vector becomes the handle instead of the
        default label/TPR handle. ``None`` (the default) is exactly today's
        behavior — byte-identical, no hook call at all.
        """
        if label in self.concept_index:
            return self.concept_index[label]
        handle = None
        if handle_fn is not None:
            v = handle_fn(label)
            if v is not None:
                v = np.asarray(v, dtype=np.float32)
                if v.shape == (self.codec.dim,):
                    n = float(np.linalg.norm(v))
                    if n > 1e-8:
                        handle = v / n
        if handle is None:
            handle = self.codec.contract(self.codec.encode_matrix(tree.root))
        nid = self.add_node(
            NodeKind.CONCEPT, handle, structure=serialize_thought(tree), label=label,
        )
        self.concept_index[label] = nid
        return nid

    def add_clause(
        self,
        payload: ClausePayload,
        handle: np.ndarray,
        *,
        structure: Optional[List[str]] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> int:
        """Create a CLAUSE node, store its payload, and wire SLOT/ABOUT edges."""
        nid = self.add_node(NodeKind.CLAUSE, handle, structure=structure, meta=meta)
        self.payloads[nid] = payload
        self.add_edge(SLOT, nid, payload.predicate_nid, rel="PREDICATE")
        for rel, filler_nid in payload.slots:
            self.add_edge(SLOT, nid, filler_nid, rel=rel)
            if self.nodes[filler_nid].kind is NodeKind.REFERENT:
                self.add_edge(ABOUT, nid, filler_nid)
        return nid

    def node(self, nid: int) -> GraphNode:
        return self.nodes[nid]

    # -- edges ----------------------------------------------------------------
    def add_edge(self, etype: str, src: int, dst: int, *, rel: Optional[str] = None) -> Edge:
        e = Edge(etype, src, dst, rel)
        self.edges.append(e)
        self._out.setdefault(src, []).append(e)
        self._in.setdefault(dst, []).append(e)
        self._by_etype.setdefault(etype, []).append(e)
        return e

    def edges_of(self, etype: str) -> List[Edge]:
        return list(self._by_etype.get(etype, ()))

    def out(self, nid: int, etype: Optional[str] = None) -> List[Edge]:
        es = self._out.get(nid, ())
        return [e for e in es if etype is None or e.etype == etype]

    def in_(self, nid: int, etype: Optional[str] = None) -> List[Edge]:
        es = self._in.get(nid, ())
        return [e for e in es if etype is None or e.etype == etype]

    def neighbors(self, nid: int) -> List[int]:
        """All nodes adjacent to ``nid`` (either edge direction)."""
        seen: List[int] = []
        for e in self._out.get(nid, ()):
            if e.dst not in seen:
                seen.append(e.dst)
        for e in self._in.get(nid, ()):
            if e.src not in seen:
                seen.append(e.src)
        return seen

    def clauses_about(self, referent_nid: int) -> List[int]:
        """Clause nids that mention ``referent_nid`` (via ABOUT edges)."""
        return [e.src for e in self.in_(referent_nid, ABOUT)]

    def __len__(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Operator-nodes — not / and / or / maybe as NODES that bind their clause
# argument(s) on a reserved orthonormal role. This is deliberately NOT a flag:
# the old quantum_parser flag system (a flat List[SubType] of NEGATIVE/L_NOT on a
# node) is not deconvolvable because a flag carries no binding to which child /
# relation / order set it. Binding the argument on a reserved role makes it
# recoverable — exactly like clause.tag_truth binds TRUTH on a reserved role.
# ---------------------------------------------------------------------------
_OP_ARG = "OP_ARG"  # reserved relation for operator arguments (its own ±1 sign)


def apply_operator(
    graph: MeaningGraph,
    op_name: str,
    clause_nids,
    codec: TPRCodec,
    *,
    set_false: Optional[bool] = None,
) -> int:
    """Create an OPERATOR node scoping one or more clause nodes.

    Builds ``M_op = bind(self_role, filler(op_name)) + Σ_i bind(role_vec(i,
    OP_ARG), clause_i.handle)`` and stores it on the node (``meta["matrix"]``) so
    the argument(s) and the operator label decode back out. ``NOT`` additionally
    flips the target clause's truth tag to FALSE (a lossless tag, for read-time
    filtering — what L8 needs).
    """
    if isinstance(clause_nids, int):
        clause_nids = [clause_nids]
    m_op = codec.bind(codec.self_role, codec.filler_vec(op_name))
    for i, cnid in enumerate(clause_nids):
        # bind a UNIT-normalized argument: contracted clause handles have tiny
        # norm, so an unnormalized arg would be swamped by the operator-label term
        # (recovery is by direction/cosine, so normalizing is loss-free here).
        arg = graph.node(cnid).handle
        n = float(np.linalg.norm(arg))
        arg = arg / n if n > 1e-8 else arg
        m_op = m_op + codec.bind(codec.role_vec(i, _OP_ARG), arg)
    nid = graph.add_node(
        NodeKind.OPERATOR, codec.filler_vec(op_name), label=op_name,
        meta={"matrix": m_op, "arity": len(clause_nids)},
    )
    for cnid in clause_nids:
        graph.add_edge(OPERATES_ON, nid, cnid)
    if set_false is None:
        set_false = op_name == "NOT"
    if set_false:
        for cnid in clause_nids:
            graph.node(cnid).meta["truth"] = "FALSE"
    return nid


def read_operator(graph: MeaningGraph, op_nid: int, codec: TPRCodec):
    """Deconvolve an operator node: ``(label, score, [recovered_arg_vecs])``.

    ``unbind(M_op, self_role)`` → cleanup recovers the operator label; ``unbind(
    M_op, role_vec(i, OP_ARG))`` recovers the i-th clause-argument vector (≈ the
    clause's handle) — the recovery a flat flag could never give.
    """
    node = graph.node(op_nid)
    m_op = node.meta["matrix"]
    label, score = codec.cleanup(codec.unbind(m_op, codec.self_role))
    args = [
        codec.unbind(m_op, codec.role_vec(i, _OP_ARG))
        for i in range(int(node.meta["arity"]))
    ]
    return label, score, args
