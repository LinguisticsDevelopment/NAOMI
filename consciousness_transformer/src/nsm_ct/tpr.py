"""Tensor Product Representations (TPR) for meaning trees — prototype.

A NON-FLATTENING vector encoding of meaning trees (primes/molecules + typed
connections): each child is **bound** to a structural **role** (its slot:
position × relation family) via an outer product, and the bindings are summed.
Unbinding inverts it. Smolensky (1990); fixed-order TPR memories trained
end-to-end reached SOTA on bAbI (Schlag & Schmidhuber, NeurIPS 2018) — see
RESEARCH_NOTES.

Design (deterministic, unique, recursive — per the project requirements):

* **Fillers** — every label (prime, molecule, anything) gets a fixed unit vector
  in R^d from a stable hash seed: same label → same vector across runs.
* **Roles** — an orthonormal basis (QR of a seeded matrix), partitioned into
  blocks per role *family* (argument-ish / predicate-ish / other — the
  "noun-axis × verb-axis" factoring): role(pos, rel) = D_rel · Q[:, block+pos],
  where D_rel is a ±1 diagonal per relation label. Roles within one relation are
  exactly orthonormal → **exact** unbinding; across relations quasi-orthogonal.
* **One level** is a d×d matrix: M = role_self ⊗ filler(label) + Σ role_i ⊗ c_i.
* **Recursion** contracts a child matrix back to R^d via a fixed semi-orthogonal
  map C (rows orthonormal, C·Cᵀ = I): c_child = C·vec(M_child). Contraction is
  lossy (d² → d) — decoding below level 1 uses **cleanup** (nearest neighbour
  against the prime∪molecule codebook). This is the HRR-flavoured fixed-dim path.
* An **exact** order-growing encoder is also provided to measure the cost of
  exactness with depth (dimension grows as d^(depth+1) — the reason unbounded
  depth must come from recursion *in time* (the loop), not tensor order).

Prototype only: numpy, no learned parts, not wired into the model. The probe
(scripts/probe_tpr.py) measures round-trip fidelity on REAL explication trees;
integration is gated on those numbers (see RESEARCH_NOTES).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_structures import ParseNode, ParseTree
from .nsm_molecules import MOLECULES
from .nsm_primes import PRIME_NAMES

_GLOBAL_SEED = 7331

# Role families ("axes"): argument-ish vs predicate-ish vs other. Relations not
# listed fall into OTHER (incl. None — typical for explication-tree children).
_ARGUMENT_RELS = {
    "SUBJECT", "OBJECT", "INDIRECT_OBJECT", "SUBJECT_COMPLEMENT", "NOMINAL",
    "SPECIFICATION", "DESCRIPTION", "APPOSITION", "PREPOSITION",
    "PREPOSITION_FROM", "PREPOSITION_TO",
}
_PREDICATE_RELS = {
    "CLAUSE", "PREDICATE", "VERBAL", "MODIFICATION", "COMPLEMENT",
    "COORDINATION", "SUBORDINATION", "SUBORDINATION_FROM", "SUBORDINATION_TO",
}


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


@dataclass
class TPRCodec:
    """Deterministic TPR encoder/decoder over a label codebook.

    Args:
        dim: Vector dimension d (one level lives in a d×d matrix).
        max_pos: Max child positions per role family (capped by block size).
    """

    dim: int = 128
    max_pos: int = 64  # per-family child positions (capped by block size = (d-1)//3)
    _fillers: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _rel_signs: Dict[Optional[str], np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        d = self.dim
        rng = np.random.default_rng(_GLOBAL_SEED)
        # Orthonormal role basis; column 0 reserved for the node's own label.
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        self._Q = q.astype(np.float32)
        # Family blocks over the remaining columns (the "axes" partition).
        usable = d - 1
        b = usable // 3
        self._blocks = {"ARG": 1, "PRED": 1 + b, "OTHER": 1 + 2 * b}
        self._block_size = min(b, self.max_pos)
        # Fixed semi-orthogonal contraction C: R^{d*d} -> R^d (rows orthonormal).
        g = np.random.default_rng(_GLOBAL_SEED + 1).standard_normal((d * d, d))
        qc, _ = np.linalg.qr(g)            # (d*d, d), orthonormal columns
        self._C = qc.T.astype(np.float32)  # (d, d*d): C @ C.T = I
        # Cleanup codebook: primes + molecules (+ common structural labels).
        self.codebook: Dict[str, np.ndarray] = {}
        for name in list(PRIME_NAMES) + [m.name for m in MOLECULES] + ["EXPLICATION"]:
            self.codebook[name] = self.filler_vec(name)

    # -- vectors --------------------------------------------------------------
    def filler_vec(self, label: str) -> np.ndarray:
        """Deterministic unit filler vector for any label."""
        if label not in self._fillers:
            rng = np.random.default_rng(_GLOBAL_SEED ^ _stable_seed(label))
            v = rng.standard_normal(self.dim).astype(np.float32)
            self._fillers[label] = v / np.linalg.norm(v)
        return self._fillers[label]

    def _family(self, rel: Optional[str]) -> str:
        if rel in _ARGUMENT_RELS:
            return "ARG"
        if rel in _PREDICATE_RELS:
            return "PRED"
        return "OTHER"

    def role_vec(self, pos: int, rel: Optional[str] = None) -> np.ndarray:
        """Role for (child position, relation): D_rel · Q[:, block + pos]."""
        col = self._blocks[self._family(rel)] + (pos % self._block_size)
        if rel not in self._rel_signs:
            rng = np.random.default_rng(_GLOBAL_SEED ^ _stable_seed(f"rel:{rel}"))
            self._rel_signs[rel] = np.where(
                rng.standard_normal(self.dim) >= 0, 1.0, -1.0
            ).astype(np.float32)
        return self._rel_signs[rel] * self._Q[:, col]

    @property
    def self_role(self) -> np.ndarray:
        """Reserved role carrying the node's own label."""
        return self._Q[:, 0]

    # -- bind / unbind ---------------------------------------------------------
    @staticmethod
    def bind(role: np.ndarray, filler: np.ndarray) -> np.ndarray:
        """Outer-product binding: one (role, filler) pair as a d×d matrix."""
        return np.outer(role, filler)

    @staticmethod
    def unbind(matrix: np.ndarray, role: np.ndarray) -> np.ndarray:
        """Matched-filter unbinding: exact for orthonormal same-relation roles."""
        return role @ matrix

    def cleanup(self, vec: np.ndarray) -> Tuple[Optional[str], float]:
        """Snap a (noisy) vector to the nearest codebook label by cosine."""
        n = float(np.linalg.norm(vec))
        if n < 1e-8:
            return None, 0.0
        v = vec / n
        best, score = None, -1.0
        for name, c in self.codebook.items():
            s = float(v @ c)
            if s > score:
                best, score = name, s
        return best, score

    # -- recursive fixed-dim encoding -------------------------------------------
    def encode_matrix(self, node: ParseNode) -> np.ndarray:
        """One node as a d×d matrix: self-label + role-bound children."""
        m = self.bind(self.self_role, self.filler_vec(node.label))
        for i, child in enumerate(node.children):
            m = m + self.bind(self.role_vec(i, child.relation), self._child_vec(child))
        return m

    def _child_vec(self, child: ParseNode) -> np.ndarray:
        if child.is_leaf:
            return self.filler_vec(child.label)
        return self.contract(self.encode_matrix(child))

    def contract(self, matrix: np.ndarray) -> np.ndarray:
        """Compress a level matrix to R^d (lossy; semi-orthogonal projection)."""
        return self._C @ matrix.reshape(-1)

    def lift(self, vec: np.ndarray) -> np.ndarray:
        """Approximate inverse of :meth:`contract` (pseudo-inverse = Cᵀ)."""
        return (self._C.T @ vec).reshape(self.dim, self.dim)

    def encode_tree(self, tree: ParseTree) -> np.ndarray:
        """Whole tree as one fixed-dim vector (deterministic, unique)."""
        return self.contract(self.encode_matrix(tree.root))

    # -- guided decode (fidelity measurement) -----------------------------------
    def decode_guided(self, matrix: np.ndarray, template: ParseNode) -> Tuple[int, int]:
        """Recover labels along a known tree shape; returns (correct, total).

        Level 1 unbinding is exact; deeper levels go through contract→lift and
        rely on cleanup — exactly the loss the probe is meant to measure.
        """
        correct = total = 0
        got, _ = self.cleanup(self.unbind(matrix, self.self_role))
        total += 1
        correct += int(got == template.label)
        for i, child in enumerate(template.children):
            u = self.unbind(matrix, self.role_vec(i, child.relation))
            if child.is_leaf:
                got, _ = self.cleanup(u)
                total += 1
                correct += int(got == child.label)
            else:
                c, t = self.decode_guided(self.lift(u), child)
                correct, total = correct + c, total + t
        return correct, total

    # -- exact order-growing encoding (cost demonstration) ----------------------
    def encode_exact(self, node: ParseNode, _depth: int = 0, max_depth: int = 3):
        """Exact TPR: tensor order grows with depth (d^(depth+1) numbers).

        Returns an ndarray of order (tree depth + 1). Guarded by ``max_depth``
        because the size explodes — the empirical case for depth-via-the-loop.
        """
        if _depth >= max_depth:
            raise MemoryError(f"exact TPR beyond depth {max_depth} refused (size d^(depth+1))")
        t = np.tensordot(self.self_role, self.filler_vec(node.label), axes=0)
        order = 2
        for i, child in enumerate(node.children):
            if child.is_leaf:
                sub = self.filler_vec(child.label)
            else:
                sub = self.encode_exact(child, _depth + 1, max_depth)
            bound = np.tensordot(self.role_vec(i, child.relation), sub, axes=0)
            # pad orders so they add (embed lower-order tensors along new axes
            # by binding to a unit "pad" filler)
            while bound.ndim < order:
                bound = np.tensordot(bound, self._pad, axes=0)
            while t.ndim < bound.ndim:
                t = np.tensordot(t, self._pad, axes=0)
            order = max(order, bound.ndim)
            t = t + bound
        return t

    @property
    def _pad(self) -> np.ndarray:
        return self.filler_vec("__PAD__")
