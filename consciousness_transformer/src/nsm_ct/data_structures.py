"""Core data structures for the NSM Consciousness Transformer.

These are deliberately lightweight, dependency-free (numpy only) containers so
they can be unit-tested in isolation and so the boundaries between *mocked* and
*real* components stay obvious. Nothing here imports torch.

The three structures the rest of the system is built around:

* :class:`ParseTree` / :class:`ParseNode` — the output of a syntactic parser.
  In this scaffold these are produced by a mock (see
  :mod:`nsm_ct.parser_interface`); the real NAOMI parser would emit the same
  shape.
* :class:`CausalTable` — a small relational store of cause/effect facts. This
  is a placeholder for the reasoning substrate; the mock semantic mapper
  populates it. The real system would derive causal structure from meaning.
* :class:`ConsciousnessState` — a fixed-width vector representing the model's
  internal "consciousness" at a point in time, plus helpers for comparing
  states (used by the consistency loss).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Parse trees (mock NAOMI parser output)
# ---------------------------------------------------------------------------


@dataclass
class ParseNode:
    """A single node in a parse tree.

    Attributes:
        label: Syntactic/semantic label (e.g. "S", "NP", "PREDICATE").
        token: Surface token for leaf nodes; ``None`` for internal nodes.
        children: Ordered child nodes.
        relation: The semantic role linking this node to its parent (e.g.
            "SUBJECT", "DESCRIPTION"). ``None`` for the root or for parsers that
            do not emit typed relations (the mock).
    """

    label: str
    token: Optional[str] = None
    children: List["ParseNode"] = field(default_factory=list)
    relation: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children."""
        return not self.children

    def iter_preorder(self) -> Iterator["ParseNode"]:
        """Yield this node then its descendants in pre-order."""
        yield self
        for child in self.children:
            yield from child.iter_preorder()


@dataclass
class ParseTree:
    """A parse tree rooted at a single node.

    Attributes:
        root: The root :class:`ParseNode`.
        text: The original surface text that was parsed (for debugging/provenance).
    """

    root: ParseNode
    text: str = ""

    def iter_preorder(self) -> Iterator[ParseNode]:
        """Pre-order traversal over every node in the tree."""
        return self.root.iter_preorder()

    def leaves(self) -> List[ParseNode]:
        """Return the leaf nodes left-to-right (i.e. the surface tokens)."""
        return [n for n in self.iter_preorder() if n.is_leaf]

    def num_nodes(self) -> int:
        """Total node count."""
        return sum(1 for _ in self.iter_preorder())


# ---------------------------------------------------------------------------
# Causal table (placeholder reasoning substrate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalRelation:
    """A single directed causal fact: ``cause`` --[relation]--> ``effect``.

    Attributes:
        cause: Identifier of the cause (typically a token or concept string).
        effect: Identifier of the effect.
        relation: The relation label. Defaults to the NSM prime ``BECAUSE``.
        weight: Confidence/strength in [0, 1].
    """

    cause: str
    effect: str
    relation: str = "BECAUSE"
    weight: float = 1.0


class CausalTable:
    """A minimal store of :class:`CausalRelation` facts.

    This is a stand-in for a richer reasoning structure. It supports adding
    relations and querying forward (effects of a cause) and backward (causes of
    an effect). TODO(reasoning): replace with a real causal graph that supports
    transitive inference and contradiction detection.
    """

    def __init__(self, relations: Optional[List[CausalRelation]] = None) -> None:
        self._relations: List[CausalRelation] = list(relations or [])

    def add(self, cause: str, effect: str, relation: str = "BECAUSE", weight: float = 1.0) -> CausalRelation:
        """Add a causal relation and return it."""
        rel = CausalRelation(cause=cause, effect=effect, relation=relation, weight=weight)
        self._relations.append(rel)
        return rel

    def get_effects(self, cause: str) -> List[CausalRelation]:
        """All relations whose cause matches ``cause``."""
        return [r for r in self._relations if r.cause == cause]

    def get_causes(self, effect: str) -> List[CausalRelation]:
        """All relations whose effect matches ``effect``."""
        return [r for r in self._relations if r.effect == effect]

    @property
    def relations(self) -> List[CausalRelation]:
        """Read-only view of all stored relations."""
        return list(self._relations)

    def __len__(self) -> int:
        return len(self._relations)


# ---------------------------------------------------------------------------
# Consciousness state
# ---------------------------------------------------------------------------


@dataclass
class ConsciousnessState:
    """A fixed-width vector representing the model's internal state.

    This is the quantity the "consciousness state transition" head predicts and
    the consciousness consistency loss constrains. It is intentionally opaque:
    what its dimensions *mean* is an open research question (see RESEARCH_NOTES).

    Attributes:
        vector: A 1-D float vector of length ``dim``.
    """

    vector: np.ndarray

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32).reshape(-1)

    @property
    def dim(self) -> int:
        """Dimensionality of the state vector."""
        return int(self.vector.shape[0])

    @classmethod
    def zeros(cls, dim: int) -> "ConsciousnessState":
        """An all-zero state of the given dimensionality."""
        return cls(np.zeros(dim, dtype=np.float32))

    def distance(self, other: "ConsciousnessState") -> float:
        """Euclidean (L2) distance to another state of the same dimensionality.

        Raises:
            ValueError: if the two states have different dimensionality.
        """
        if self.dim != other.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}")
        return float(np.linalg.norm(self.vector - other.vector))
