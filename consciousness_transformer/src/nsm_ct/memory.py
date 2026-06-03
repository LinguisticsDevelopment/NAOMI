"""Abstract retrieved-memory interface plus a mock implementation.

The model's inputs include a "retrieved memory" vector — the output of some
long-term memory subsystem queried with the current context. That subsystem
does not exist yet, so it is mocked behind :class:`AbstractMemory`.

PLUG-IN POINT: implement :class:`AbstractMemory` backed by a real vector store /
episodic memory and pass it where :class:`MockMemoryStore` is used.
"""

from __future__ import annotations

import abc
import hashlib

import numpy as np


class AbstractMemory(abc.ABC):
    """Interface for a context-conditioned memory retriever."""

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimensionality of returned memory vectors."""
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve(self, query: str) -> np.ndarray:
        """Return a single memory vector of length :attr:`dim` for ``query``."""
        raise NotImplementedError


class MockMemoryStore(AbstractMemory):
    """A deterministic stand-in for retrieved memory.

    Returns a fixed pseudo-random vector derived from a stable hash of the query
    string. It carries no real information; it just guarantees the memory input
    pathway is wired up and reproducible.

    TODO(memory): replace with real retrieval (embedding search over episodic
    memory), and see RESEARCH_NOTES on memory pruning.
    """

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def retrieve(self, query: str) -> np.ndarray:
        seed = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self._dim).astype(np.float32)
