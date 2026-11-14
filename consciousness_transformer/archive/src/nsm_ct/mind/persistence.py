"""Disk persistence for the meaning graph (M1).

The symbolic ``MeaningGraph`` is in-memory only; a durable "big big persistent
graph" needs graph-level save/load. This module serializes a graph — nodes
(including their lossy ``handle`` vectors and lossless ``structure`` token lists),
typed edges, clause payloads, and the co-reference indices — to JSON, and
reconstructs it byte-for-byte.

Round-trip is **exact**: ``load(save(g))`` reproduces the same nodes, handles,
edges, payloads, and id allocator, so a knowledge graph survives a restart
identically (the M1 persistence gate). Handles and any ``numpy`` arrays stashed in
node ``meta`` (e.g. an operator node's bound matrix) are encoded losslessly via a
tagged list form.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

from ..meaning_graph import ClausePayload, Edge, GraphNode, MeaningGraph, NodeKind
from ..tpr import TPRCodec

_FORMAT_VERSION = 1


def _enc(obj: Any) -> Any:
    """Encode a value (incl. numpy arrays) into JSON-safe form, losslessly."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.astype(np.float32).tolist()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _enc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_enc(v) for v in obj]
    return obj


def _dec(obj: Any) -> Any:
    """Inverse of :func:`_enc`."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.asarray(obj["__ndarray__"], dtype=np.float32)
        return {k: _dec(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dec(v) for v in obj]
    return obj


def graph_to_dict(graph: MeaningGraph) -> Dict[str, Any]:
    """Serialize ``graph`` to a JSON-safe dict (exact, reversible)."""
    return {
        "format_version": _FORMAT_VERSION,
        "dim": graph.codec.dim,
        "max_pos": graph.codec.max_pos,
        "next_nid": graph._next_nid,
        "nodes": [
            {
                "nid": n.nid,
                "kind": n.kind.value,
                "handle": _enc(n.handle),
                "structure": n.structure,
                "label": n.label,
                "meta": _enc(n.meta),
            }
            for n in graph.nodes.values()
        ],
        "edges": [
            {"etype": e.etype, "src": e.src, "dst": e.dst, "rel": e.rel}
            for e in graph.edges
        ],
        "payloads": [
            {
                "nid": nid,
                "predicate_nid": p.predicate_nid,
                "slots": [[rel, fnid] for rel, fnid in p.slots],
                "operator_nids": list(p.operator_nids),
            }
            for nid, p in graph.payloads.items()
        ],
        "referent_index": dict(graph.referent_index),
        "concept_index": dict(graph.concept_index),
    }


def graph_from_dict(data: Dict[str, Any], codec: TPRCodec | None = None) -> MeaningGraph:
    """Reconstruct a :class:`MeaningGraph` from :func:`graph_to_dict` output."""
    if data.get("format_version") != _FORMAT_VERSION:
        raise ValueError(f"unsupported graph format version: {data.get('format_version')!r}")
    if codec is None:
        codec = TPRCodec(dim=int(data["dim"]), max_pos=int(data.get("max_pos", 64)))
    graph = MeaningGraph(codec)
    # Restore nodes directly (preserve nids; do NOT route through add_* which would
    # re-allocate ids and recompute handles).
    for nd in data["nodes"]:
        graph.nodes[nd["nid"]] = GraphNode(
            nid=nd["nid"],
            kind=NodeKind(nd["kind"]),
            handle=_dec(nd["handle"]),
            structure=nd["structure"],
            label=nd["label"],
            meta=_dec(nd["meta"]),
        )
    for ed in data["edges"]:
        graph.add_edge(ed["etype"], ed["src"], ed["dst"], rel=ed["rel"])
    for pd in data["payloads"]:
        graph.payloads[pd["nid"]] = ClausePayload(
            predicate_nid=pd["predicate_nid"],
            slots=[(rel, fnid) for rel, fnid in pd["slots"]],
            operator_nids=list(pd["operator_nids"]),
        )
    graph.referent_index = dict(data["referent_index"])
    graph.concept_index = dict(data["concept_index"])
    graph._next_nid = int(data["next_nid"])
    return graph


def save_graph(graph: MeaningGraph, path: str) -> None:
    """Write ``graph`` to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(graph_to_dict(graph), fh)


def load_graph(path: str, codec: TPRCodec | None = None) -> MeaningGraph:
    """Read a graph written by :func:`save_graph`."""
    with open(path, "r", encoding="utf-8") as fh:
        return graph_from_dict(json.load(fh), codec=codec)


__all__ = [
    "graph_to_dict", "graph_from_dict", "save_graph", "load_graph",
]
