"""The extensible semantic-axis registry (M17.2).

NSM-65 is the *seed* basis, not the answer. The basis search (``basis_search``)
may promote additional interpretable primitives — words the definition graph
keeps bottoming out on (definitional cycle-breakers) — into atomic axes. This
registry holds that growing, ordered basis and records *why* each axis exists.

Each axis carries a true (interpretable) meaning: seed axes are NSM primes;
promoted axes are concrete words (e.g. "animal", "person") treated as atomic
because decomposing them further costs more than it saves (MDL).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

from ..nsm_primes import PRIME_NAMES

SEED_PROVENANCE = "nsm-seed"


@dataclass
class AxisRegistry:
    """An ordered, interpretable basis seeded by the NSM primes."""

    axes: List[str] = field(default_factory=lambda: list(PRIME_NAMES))
    provenance: Dict[str, str] = field(
        default_factory=lambda: {a: SEED_PROVENANCE for a in PRIME_NAMES}
    )

    @classmethod
    def seed(cls) -> "AxisRegistry":
        """A fresh registry containing exactly the NSM-65 seed axes."""
        return cls()

    @property
    def dim(self) -> int:
        return len(self.axes)

    def index(self, name: str) -> int:
        return self.axes.index(name)

    def __contains__(self, name: str) -> bool:
        return name in self.provenance

    def add(self, name: str, why: str) -> bool:
        """Promote *name* to an axis with provenance *why*. No-op if present."""
        if name in self.provenance:
            return False
        self.axes.append(name)
        self.provenance[name] = why
        return True

    def beyond_seed(self) -> List[str]:
        """Axes added beyond the NSM-65 seed, in insertion order."""
        return [a for a in self.axes if self.provenance.get(a) != SEED_PROVENANCE]

    def extra_axes(self) -> FrozenSet[str]:
        """The promoted (non-seed) axes as a frozenset — the decomposition floor
        passed to :func:`naive_decompose`."""
        return frozenset(self.beyond_seed())

    def copy(self) -> "AxisRegistry":
        return AxisRegistry(axes=list(self.axes), provenance=dict(self.provenance))

    def summary(self) -> List[Tuple[str, str]]:
        """``(axis, why)`` for every axis beyond the seed."""
        return [(a, self.provenance[a]) for a in self.beyond_seed()]
