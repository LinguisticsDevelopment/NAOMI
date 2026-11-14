"""NAOMI memory: one MeaningGraph of clause trees, each node also a tensor.

Every stored ClauseTree becomes a CLAUSE node (SLOT edges to one shared
CONCEPT/REFERENT node per distinct lemma), the lossless tree is kept beside it,
and its *tensor view* is the TPR flattening of the tree: sum over fillers of
``bind(role_vec, lemma_vec)`` (a d x d matrix, flattened). Negated clauses are
wrapped by a NOT operator node. Layers are a tag per clause.

Rules documented here (kept deliberately simple):
* Referent vs concept: a non-predicate filler is a REFERENT if its lemma is a
  pronoun or WordNet knows no synset for it (names like "zorp"); everything else
  (and every predicate) is a CONCEPT.
* Tensor weights: modifier roles get ``MODIFIER_WEIGHT``; holes are unbound.
* Handles: ``StoredClause.handle`` / ``handle_of`` are the flattened d*d TPR
  matrix (cosine-comparable). The graph node's own handle is the codec's
  contracted d-vector (what ``apply_operator`` needs).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.meaning_graph import (
    ClausePayload, MeaningGraph, NodeKind, apply_operator,
)
from nsm_ct.tpr import TPRCodec

from .types import LAYERS, ClauseTree, Filler

MODIFIER_ROLES = frozenset(
    {"DESCRIPTION", "SPECIFICATION", "COMPLEMENT", "QUANTITY", "ADDITIVE", "FOCUS"})
MODIFIER_WEIGHT = 0.3
PRONOUNS = frozenset(
    "i you he she it we they me him her us them who someone somebody".split())
_ROLE_ORDER = ("PREDICATE", "SUBJECT", "OBJECT", "PLACE", "RECIPIENT", "AGENT",
               "TIME", "INSTRUMENT", "BENEFICIARY", "SOURCE", "GOAL", "MANNER",
               "CAUSE", "POSSESSOR", "ATTRIBUTE")


def _role_index(role: str) -> int:
    if role in _ROLE_ORDER:
        return _ROLE_ORDER.index(role)
    return len(_ROLE_ORDER) + (sum(map(ord, role)) % 5)


def _wn():
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
        return wn
    except Exception:
        return None


@dataclass
class StoredClause:
    nid: int
    tree: ClauseTree
    layer: str
    provenance: dict
    handle: np.ndarray
    negated: bool


@dataclass
class _Rec:
    tree: ClauseTree
    layer: str
    provenance: dict
    handle: np.ndarray
    negated: bool
    op_nid: Optional[int] = None


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
    def _matrix(self, clause: ClauseTree, skip_hole: bool = True) -> np.ndarray:
        m = np.zeros((self.dim, self.dim), dtype=np.float32)
        for f in [clause.predicate] + list(clause.args):
            if f.is_hole and skip_hole:
                continue
            w = MODIFIER_WEIGHT if f.role in MODIFIER_ROLES else 1.0
            role = self.codec.role_vec(_role_index(f.role), f.role)
            m += w * self.codec.bind(role, self.codec.filler_vec("w:" + f.lemma))
        return m

    def handle_of(self, clause: ClauseTree, skip_hole: bool = True) -> np.ndarray:
        return self._matrix(clause, skip_hole).reshape(-1)

    # -- word sense ---------------------------------------------------------
    def _layer_senses(self, layer: str) -> set:
        return {f.sense for r in self._recs.values() if r.layer == layer
                for f in [r.tree.predicate] + r.tree.args if f.sense}

    def _choose_sense(self, f: Filler, layer: str, own: set) -> Tuple[Optional[str], str]:
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
    def _is_referent(self, f: Filler) -> bool:
        if f.role == "PREDICATE":
            return False
        if f.lemma in PRONOUNS:
            return True
        return self._wn is not None and not self._wn.synsets(f.lemma)

    def _node_for(self, f: Filler) -> int:
        if self._is_referent(f):
            return self.graph.add_referent(f.lemma)
        tree = ParseTree(ParseNode(label=f.lemma, token=f.lemma), text=f.lemma)
        return self.graph.add_concept(f.lemma, tree)

    # -- store --------------------------------------------------------------
    def store(self, clause: ClauseTree, layer: str = "current",
              provenance: Optional[dict] = None) -> int:
        if layer not in LAYERS:
            raise ValueError(f"layer {layer!r} not in {LAYERS}")
        tree = copy.deepcopy(clause)
        tree.args = [a for a in tree.args if not a.is_hole]
        rules, own = [], set()
        for f in [tree.predicate] + tree.args:
            f.sense, rule = self._choose_sense(f, layer, own)
            if f.sense:
                own.add(f.sense)
                self._lemma_sense.setdefault(f.lemma, f.sense)
                rules.append(f"{f.lemma}:{rule}")
        mat = self._matrix(tree)
        payload = ClausePayload(
            predicate_nid=self._node_for(tree.predicate),
            slots=[(a.role, self._node_for(a)) for a in tree.args])
        nid = self.graph.add_clause(
            payload, self.codec.contract(mat),
            meta={"layer": layer, "sentence": tree.sentence})
        op = apply_operator(self.graph, "NOT", nid, self.codec) if tree.negated else None
        prov = dict(provenance or {})
        prov["sense_rules"] = rules
        self._recs[nid] = _Rec(tree, layer, prov, mat.reshape(-1), tree.negated, op)
        return nid

    def store_all(self, clauses, layer: str = "current", provenance=None) -> List[int]:
        return [self.store(c, layer, provenance) for c in clauses]

    # -- views --------------------------------------------------------------
    def _view(self, nid: int, r: _Rec) -> StoredClause:
        return StoredClause(nid, r.tree, r.layer, r.provenance, r.handle, r.negated)

    def clauses(self, layer: Optional[str] = None) -> List[StoredClause]:
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
        """Forget clauses (graph nodes stay as inert orphans; recs are dropped)."""
        for nid in [n for n, r in self._recs.items() if layer is None or r.layer == layer]:
            del self._recs[nid]
        self._lemma_sense = {}
        for r in self._recs.values():
            for f in [r.tree.predicate] + r.tree.args:
                if f.sense:
                    self._lemma_sense.setdefault(f.lemma, f.sense)

    def similar(self, query_handle: np.ndarray, k: int = 10,
                layers=None) -> List[Tuple[StoredClause, float]]:
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
        return {"clauses": len(self._recs), "by_layer": by, "nodes": len(self.graph),
                "node_kinds": kinds, "edges": len(self.graph.edges),
                "negated": sum(r.negated for r in self._recs.values())}

    def dump(self) -> str:
        lines = []
        for L in LAYERS:
            cs = self.clauses(L)
            if not cs:
                continue
            lines.append(f"[{L}] ({len(cs)})")
            for c in cs:
                t = c.tree
                args = " ".join(f"{a.role}={a.lemma}" + (f"<{a.sense}>" if a.sense else "")
                                for a in t.args)
                neg = "NOT " if c.negated else ""
                lines.append(f"  #{c.nid} {neg}{t.predicate.lemma}({args})  prov={c.provenance}")
        return "\n".join(lines)
