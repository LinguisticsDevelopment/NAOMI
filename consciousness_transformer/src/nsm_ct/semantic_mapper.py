"""Abstract semantic mapper plus a mock implementation.

The semantic mapper is the component that turns syntax (a :class:`ParseTree`)
into *meaning* expressed over NSM primes, and seeds the initial consciousness
state. **This is the hard research problem and we explicitly do not solve it**
(see the project brief and RESEARCH_NOTES). It is mocked behind this interface.

PLUG-IN POINT: implement :class:`AbstractSemanticMapper` with the real
NSM-grounded semantic composition logic and pass it where
:class:`MockSemanticMapper` is used.
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .data_structures import CausalTable, ConsciousnessState, ParseTree
from .nsm_primes import NUM_PRIMES, PRIME_NAMES, prime_index


@dataclass
class SemanticRepresentation:
    """The (mock) meaning of a parse, expressed over NSM primes.

    Attributes:
        nsm_activations: Map from NSM prime name -> activation strength.
        causal_table: Extracted causal relations (placeholder).
    """

    nsm_activations: Dict[str, float] = field(default_factory=dict)
    causal_table: CausalTable = field(default_factory=CausalTable)

    def to_prime_vector(self) -> np.ndarray:
        """Dense activation vector over the canonical prime inventory."""
        vec = np.zeros(NUM_PRIMES, dtype=np.float32)
        for name, val in self.nsm_activations.items():
            vec[prime_index(name)] = val
        return vec

    def to_consciousness_state(self, dim: int) -> ConsciousnessState:
        """Project the prime activations into a ``dim``-wide consciousness state.

        Uses a fixed (seeded) random projection so the mapping is deterministic
        and dimension-configurable. This is a *mock* projection — there is no
        claim that these dimensions mean anything.
        """
        prime_vec = self.to_prime_vector()
        rng = np.random.default_rng(1234)  # fixed projection matrix
        proj = rng.standard_normal((NUM_PRIMES, dim)).astype(np.float32)
        state = np.tanh(prime_vec @ proj)
        return ConsciousnessState(state)


class AbstractSemanticMapper(abc.ABC):
    """Interface for mapping syntax to NSM-grounded meaning."""

    @abc.abstractmethod
    def map(self, tree: ParseTree) -> SemanticRepresentation:
        """Map a parse tree to a :class:`SemanticRepresentation`."""
        raise NotImplementedError


def _stable_hash(text: str) -> int:
    """Deterministic non-negative hash (Python's hash() is salted per run)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


class MockSemanticMapper(AbstractSemanticMapper):
    """A deterministic, meaning-free stand-in for the semantic mapper.

    For each content leaf token it lights up a pseudo-randomly chosen NSM prime
    (chosen by hashing the token, so it is stable across runs) and adds a toy
    causal relation between consecutive content tokens. **None of this reflects
    real semantics** — it exists only to give the transformer a populated,
    deterministic NSM/causal signal to consume.

    TODO(semantics): this is the central thing to replace with real work.
    """

    def map(self, tree: ParseTree) -> SemanticRepresentation:
        activations: Dict[str, float] = {}
        causal = CausalTable()
        content_tokens = [n.token for n in tree.leaves() if n.label == "CONTENT" and n.token]
        for tok in content_tokens:
            prime = PRIME_NAMES[_stable_hash(tok) % NUM_PRIMES]
            activations[prime] = activations.get(prime, 0.0) + 1.0
        # Toy "causality": each content token nudges the next one.
        for cause, effect in zip(content_tokens, content_tokens[1:]):
            causal.add(cause=cause, effect=effect, relation="BECAUSE", weight=0.5)
        # Normalize activations to [0, 1]-ish range.
        if activations:
            peak = max(activations.values())
            activations = {k: v / peak for k, v in activations.items()}
        return SemanticRepresentation(nsm_activations=activations, causal_table=causal)
