"""Collapse / expand — definition as a reversible graph operation.

**Collapse** files a meaning structure (a clause or an explication tree) into one
node: it stores the structure losslessly (:func:`serialize_thought`) and returns
a node whose vector ``handle`` is a *lossy* address. **Expand** dereferences a
node back to its exact structure — losslessly, because the structure was stored,
not reconstructed from the vector. The vector is never required to be invertible
(fixed-dim vectors can't be); it only has to *address* reliably enough.

The honest residual problem lives in :func:`dereference_by_vector`: recovering a
node from a (possibly noisy) handle is content-addressable retrieval, and the
``margin`` it returns surfaces how separable the handles are. Correctness always
routes through the exact path (:func:`expand`); the vector is a shortcut.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .data_structures import ParseTree
from .meaning_graph import MeaningGraph, NodeKind
from .serialization import deserialize_thought, serialize_thought
from .tpr import TPRCodec


def collapse(
    graph: MeaningGraph,
    tree: ParseTree,
    codec: TPRCodec,
    *,
    label: Optional[str] = None,
    kind: NodeKind = NodeKind.CONCEPT,
    handle_fn: Optional[Callable[[str], Optional[np.ndarray]]] = None,
) -> int:
    """File ``tree`` as a node and return its id (a.k.a. ``define_concept``).

    Losslessness is guaranteed by the stored ``structure``; the handle is the
    lossy address ``contract(encode_matrix(root))`` unless ``handle_fn`` (the
    M33 opt-in hook, e.g. a USVS handle provider) is given and returns a vector
    for a CONCEPT's ``label`` — then that becomes the handle instead. ``None``
    (the default) reproduces today's behavior exactly; non-CONCEPT nodes and
    CONCEPT nodes with no ``label`` never consult ``handle_fn``.
    """
    if kind is NodeKind.CONCEPT and label is not None:
        # one node per word/label — co-reference of words (the shared "is" node)
        return graph.add_concept(label, tree, handle_fn=handle_fn)
    handle = codec.contract(codec.encode_matrix(tree.root))
    return graph.add_node(
        kind, handle, structure=serialize_thought(tree), label=label,
    )


def expand(graph: MeaningGraph, nid: int) -> ParseTree:
    """Dereference a node to its exact stored structure (lossless, O(1))."""
    node = graph.node(nid)
    if node.structure is None:
        raise ValueError(f"node {nid} ({node.kind.value}) has no stored structure to expand")
    return deserialize_thought(node.structure)


def dereference_by_vector(
    graph: MeaningGraph,
    v: np.ndarray,
    codec: Optional[TPRCodec] = None,
    *,
    kind_filter: Optional[NodeKind] = None,
) -> Tuple[Optional[int], float]:
    """Nearest node to handle ``v`` by cosine; returns ``(nid, margin)``.

    ``margin`` = top-1 minus top-2 cosine — the separability of the match (the
    geometric quantity the dereferencing risk turns on). ``nid`` is ``None`` if
    there are no candidate nodes.
    """
    v = np.asarray(v, dtype=np.float32)
    nv = float(np.linalg.norm(v))
    if nv < 1e-8:
        return None, 0.0
    v = v / nv
    best_nid, best, second = None, -1.0, -1.0
    for nid, node in graph.nodes.items():
        if kind_filter is not None and node.kind is not kind_filter:
            continue
        h = node.handle
        nh = float(np.linalg.norm(h))
        if nh < 1e-8:
            continue
        s = float(v @ (h / nh))
        if s > best:
            best_nid, best, second = nid, s, best
        elif s > second:
            second = s
    if best_nid is None:
        return None, 0.0
    margin = best - (second if second > -1.0 else 0.0)
    return best_nid, margin


def flatten_concept(
    graph: MeaningGraph,
    nid: int,
    *,
    max_depth: int = 8,
    _depth: int = 0,
    _seen: Optional[frozenset] = None,
) -> List[str]:
    """Walk a node's stored structure down to atom labels (definition expansion).

    Mirrors ``flatten_molecule_to_prime_names``: a leaf whose label is itself a
    concept in the graph is expanded (depth- and cycle-guarded); otherwise the
    leaf label is emitted.
    """
    seen = frozenset() if _seen is None else _seen
    if nid in seen or _depth >= max_depth:
        return []
    seen = seen | {nid}
    node = graph.node(nid)
    if node.structure is None:
        return [node.label] if node.label else []
    out: List[str] = []
    for n in deserialize_thought(node.structure).root.iter_preorder():
        if n.children:
            continue
        lbl = n.label
        sub = graph.concept_index.get(lbl)
        if sub is not None and sub != nid and graph.node(sub).structure is not None:
            out.extend(flatten_concept(
                graph, sub, max_depth=max_depth, _depth=_depth + 1, _seen=seen,
            ))
        else:
            out.append(lbl)
    return out
