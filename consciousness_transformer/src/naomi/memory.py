"""NAOMI memory: one MeaningGraph of stored sentence trees, each also a tensor.

Every stored Tree becomes one CLAUSE node (predicate = the tree's root, SLOT edges labelled with
each node's own structural relation to one shared CONCEPT/REFERENT node per distinct lemma), the
lossless Tree is kept beside it, and its *tensor view* is the TPR flattening over (relation-path,
lemma) pairs with nesting: sum over every node of ``w * bind(role_vec(path), filler_vec(lemma))``
where ``path`` is the chain of relation labels from the root ("SUBJECT/DESCRIPTION") and
``w = 1/depth`` (root 1.0). The synthetic ROOT (no token) is skipped. UNATTACHED nodes are bound
like any other (path "UNATTACHED"). Layers are a tag per tree.

Rules (kept deliberately simple):
* Referent vs concept is decided by node ``kind`` only: ``name`` -> REFERENT (a variable with no
  meaning point), ``word`` -> CONCEPT. No word list.
* Sense choice (``sense``) is made only for ``word`` nodes that carry candidate senses.
* Handles: ``Stored.handle`` / ``handle_of`` are the flattened d*d TPR matrix (cosine-comparable).
  The graph node's own handle is the codec's contracted d-vector.
"""

from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.meaning_graph import ClausePayload, MeaningGraph, NodeKind
from nsm_ct.tpr import TPRCodec

from .tree import compact, walk
from .types import KIND_NAME, LAYERS, Node, Tree


def _wn():
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
        return wn
    except Exception:
        return None


@dataclass
class Stored:
    nid: int
    tree: Tree
    layer: str
    provenance: dict
    handle: np.ndarray


@dataclass
class _Rec:
    tree: Tree
    layer: str
    provenance: dict
    handle: np.ndarray


def _nodes(tree: Tree) -> List[Node]:
    return [n for n, _d, _p in walk(tree.root)]


class Memory:
    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim = dim
        self.seed = seed
        self.codec = TPRCodec(dim=dim)
        self.graph = MeaningGraph(self.codec)
        self._recs: Dict[int, _Rec] = {}
        self._lemma_sense: Dict[str, str] = {}
        self._wn = _wn()

    # -- encoding -----------------------------------------------------------
    def _matrix(self, tree: Tree) -> np.ndarray:
        m = np.zeros((self.dim, self.dim), dtype=np.float32)
        stack = [(tree.root, "", 1)]
        while stack:
            n, path, depth = stack.pop()
            here = n.rel if not path else path + "/" + n.rel
            if n.token_index is not None:
                role = self.codec.role_vec(zlib.crc32(here.encode()), here)
                m += (1.0 / depth) * self.codec.bind(role, self.codec.filler_vec("w:" + n.lemma))
            for c in n.children:
                stack.append((c, here, depth + 1))
        return m

    def handle_of(self, tree: Tree) -> np.ndarray:
        return self._matrix(tree).reshape(-1)

    # -- word sense ---------------------------------------------------------
    def _layer_senses(self, layer: str) -> set:
        return {n.sense for r in self._recs.values() if r.layer == layer
                for n in _nodes(r.tree) if n.sense}

    def _choose_sense(self, f: Node, layer: str, own: set) -> Tuple[Optional[str], str]:
        if not f.senses:
            return f.sense, "no_candidates"
        if f.lemma in self._lemma_sense and self._lemma_sense[f.lemma] in f.senses:
            return self._lemma_sense[f.lemma], "reuse_lemma_sense"
        pool = self._layer_senses(layer) | own
        for c in f.senses:
            if c in pool:
                return c, "same_synset_in_layer"
        if self._wn is not None and pool:
            for c in f.senses:
                try:
                    s = self._wn.synset(c)
                    rel = {x.name() for x in s.hypernyms() + s.hyponyms()}
                except Exception:
                    continue
                if rel & pool:
                    return c, "hypernym_hyponym_neighbour"
        return f.senses[0], "first_candidate"

    # -- graph nodes --------------------------------------------------------
    def _node_for(self, f: Node) -> int:
        if f.kind == KIND_NAME:           # kind only: a name is a variable with no meaning point
            return self.graph.add_referent(f.lemma)
        tree = ParseTree(ParseNode(label=f.lemma, token=f.lemma), text=f.lemma)
        return self.graph.add_concept(f.lemma, tree)

    # -- store --------------------------------------------------------------
    def store(self, tree: Tree, layer: str = "current", provenance: Optional[dict] = None) -> int:
        if layer not in LAYERS:
            raise ValueError(f"layer {layer!r} not in {LAYERS}")
        tree = copy.deepcopy(tree)
        rules, own = [], set()
        for f in _nodes(tree):
            if f.kind == KIND_NAME:
                continue
            f.sense, rule = self._choose_sense(f, layer, own)
            if f.sense:
                own.add(f.sense)
                self._lemma_sense.setdefault(f.lemma, f.sense)
                rules.append(f"{f.lemma}:{rule}")
        mat = self._matrix(tree)
        payload = ClausePayload(
            predicate_nid=self._node_for(tree.root),
            slots=[(n.rel, self._node_for(n)) for n in _nodes(tree)[1:] if n.token_index is not None])
        nid = self.graph.add_clause(payload, self.codec.contract(mat),
                                    meta={"layer": layer, "sentence": tree.sentence})
        prov = dict(provenance or {})
        prov["sense_rules"] = rules
        self._recs[nid] = _Rec(tree, layer, prov, mat.reshape(-1))
        return nid

    def store_all(self, trees, layer: str = "current", provenance=None) -> List[int]:
        return [self.store(t, layer, provenance) for t in trees]

    # -- views --------------------------------------------------------------
    def _view(self, nid: int, r: _Rec) -> Stored:
        return Stored(nid, r.tree, r.layer, r.provenance, r.handle)

    def trees(self, layer: Optional[str] = None) -> List[Stored]:
        return [self._view(n, r) for n, r in self._recs.items()
                if layer is None or r.layer == layer]

    def consolidate(self, from_layer: str = "current", to_layer: str = "long_term",
                    where=None) -> int:
        if to_layer not in LAYERS:
            raise ValueError(f"layer {to_layer!r} not in {LAYERS}")
        n = 0
        for nid, r in self._recs.items():
            if r.layer == from_layer and (where is None or where(self._view(nid, r))):
                r.layer = to_layer
                self.graph.node(nid).meta["layer"] = to_layer
                n += 1
        return n

    def clear(self, layer: Optional[str] = None):
        """Forget trees (graph nodes stay as inert orphans; recs are dropped)."""
        for nid in [n for n, r in self._recs.items() if layer is None or r.layer == layer]:
            del self._recs[nid]
        self._lemma_sense = {}
        for r in self._recs.values():
            for f in _nodes(r.tree):
                if f.sense:
                    self._lemma_sense.setdefault(f.lemma, f.sense)

    def similar(self, query_handle: np.ndarray, k: int = 10,
                layers=None) -> List[Tuple[Stored, float]]:
        q = np.asarray(query_handle, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn < 1e-8:
            return []
        out = []
        for nid, r in self._recs.items():
            if layers is not None and r.layer not in layers:
                continue
            hn = float(np.linalg.norm(r.handle))
            if hn < 1e-8:
                continue
            out.append((self._view(nid, r), float(q @ r.handle) / (qn * hn)))
        out.sort(key=lambda t: (-t[1], t[0].nid))
        return out[:k]

    def stats(self) -> dict:
        by = {L: 0 for L in LAYERS}
        for r in self._recs.values():
            by[r.layer] += 1
        kinds = {k.value: sum(1 for n in self.graph.nodes.values() if n.kind is k)
                 for k in NodeKind}
        return {"trees": len(self._recs), "by_layer": by, "nodes": len(self.graph),
                "node_kinds": kinds, "edges": len(self.graph.edges)}

    def dump(self) -> str:
        lines = []
        for L in LAYERS:
            cs = self.trees(L)
            if not cs:
                continue
            lines.append(f"[{L}] ({len(cs)})")
            for c in cs:
                lines.append(f"  #{c.nid} {compact(c.tree)}  prov={c.provenance}")
        return "\n".join(lines)
