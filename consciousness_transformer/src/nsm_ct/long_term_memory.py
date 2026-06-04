"""Long-term memory: a persistent, growing repository of connections.

This is the second tier of memory. The per-episode
:class:`~nsm_ct.memory.WorkingMemory` is the **local context** (it resets every
episode); :class:`LongTermMemory` **persists across episodes** and accumulates a
growing store of consolidated knowledge plus a graph of **connections** between
entries. Over many input→response cycles ("tests") the repo grows — the
system's non-parametric, ever-expanding memory.

Two complementary kinds of learning happen in the lifelong loop
(:mod:`nsm_ct.lifelong`):
* **parametric** — the model weights keep training on each episode, and
* **non-parametric** — this store grows: new facts are consolidated, new
  connections are recorded, and retrieval over the repo conditions future
  reasoning.

Design choices for the scaffold:
* Stored contents are **detached data** (like a vector database / RAG memory),
  not learned parameters — so consolidation never backprops across episodes. The
  read *projections* are learned; the *contents* are constants.
* Connections are a simple growing edge dict (the "repo of connections"); entries
  consolidated together (co-occurring in an episode) get linked.
* :meth:`save` / :meth:`load` persist the repo to disk, so it survives across
  runs/sessions — a genuinely long-lived store.

TODO(pruning): :meth:`_maybe_prune` is a placeholder FIFO cap. A real policy
should weigh connection strength, recency, and redundancy (see RESEARCH_NOTES).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LongTermMemory(nn.Module):
    """Persistent, growing store of consolidated vectors + a connection graph.

    Args:
        mem_dim: Width of stored vectors / reads.
        state_dim: Width of the consciousness state (the read-query source).
        max_size: Cap on stored entries (triggers the pruning placeholder).
        connect_within_episode: If True, entries consolidated together are linked.
    """

    def __init__(
        self,
        mem_dim: int,
        state_dim: int,
        max_size: int = 10000,
        connect_within_episode: bool = True,
    ) -> None:
        super().__init__()
        self.mem_dim = mem_dim
        self.max_size = max_size
        self.connect_within_episode = connect_within_episode
        # Learned read projections (the contents themselves are data).
        self.query_proj = nn.Linear(state_dim, mem_dim)
        self.key_proj = nn.Linear(mem_dim, mem_dim)
        self.value_proj = nn.Linear(mem_dim, mem_dim)
        self.reset()

    # -- repo state ----------------------------------------------------------
    def reset(self) -> None:
        """Empty the repository (contents + connections)."""
        self._slots: torch.Tensor = torch.zeros(0, self.mem_dim)  # CPU home
        self.metas: List[dict] = []
        self.edges: Dict[Tuple[int, int], float] = {}

    def __len__(self) -> int:
        return int(self._slots.shape[0])

    @property
    def num_connections(self) -> int:
        return len(self.edges)

    # -- read ----------------------------------------------------------------
    def read(self, state: torch.Tensor) -> torch.Tensor:
        """Attend over the whole repo with a query from ``state``.

        Args:
            state: ``[B, state_dim]``.

        Returns:
            ``[B, mem_dim]`` (all-zeros while the repo is empty).
        """
        b = state.shape[0]
        if len(self) == 0:
            return torch.zeros(b, self.mem_dim, device=state.device)
        slots = self._slots.to(state.device)
        q = self.query_proj(state)                       # [B, mem]
        k = self.key_proj(slots)                         # [N, mem]
        v = self.value_proj(slots)                       # [N, mem]
        scores = q @ k.t() / math.sqrt(self.mem_dim)     # [B, N]
        attn = torch.softmax(scores, dim=-1)
        return attn @ v                                  # [B, mem]

    # -- write / consolidate -------------------------------------------------
    def consolidate(
        self,
        vectors: torch.Tensor,
        gates: Optional[torch.Tensor] = None,
        metas: Optional[List[dict]] = None,
    ) -> List[int]:
        """Append (detached) vectors to the repo; return their new indices.

        Args:
            vectors: ``[M, mem_dim]`` candidate entries.
            gates: ``[M]`` optional per-entry retention strength (the state's
                consolidation gate). Near-zero entries are dropped.
            metas: Optional per-entry provenance dicts.
        """
        vectors = vectors.detach().cpu()
        if gates is not None:
            vectors = vectors * gates.detach().cpu().unsqueeze(-1)
        keep = vectors.norm(dim=-1) > 1e-6
        if metas is not None:
            metas = [m for m, k in zip(metas, keep.tolist()) if k]
        vectors = vectors[keep]
        if vectors.shape[0] == 0:
            return []
        start = len(self)
        self._slots = torch.cat([self._slots, vectors], dim=0)
        idxs = list(range(start, len(self)))
        self.metas.extend(metas if metas is not None else [{} for _ in idxs])
        if self.connect_within_episode:
            self.connect_group(idxs)
        self._maybe_prune()
        return idxs

    # -- connections (the repo of connections) -------------------------------
    def connect(self, i: int, j: int, weight: float = 1.0) -> None:
        """Add/strengthen an undirected connection between entries ``i`` and ``j``."""
        if i == j:
            return
        key = (min(i, j), max(i, j))
        self.edges[key] = self.edges.get(key, 0.0) + weight

    def connect_group(self, idxs: List[int], weight: float = 1.0) -> None:
        """Pairwise-connect a group of entries (e.g. facts from one episode)."""
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                self.connect(idxs[a], idxs[b], weight)

    def neighbors(self, i: int) -> List[Tuple[int, float]]:
        """Return ``(other_index, weight)`` connections of entry ``i``."""
        out: List[Tuple[int, float]] = []
        for (a, b), w in self.edges.items():
            if a == i:
                out.append((b, w))
            elif b == i:
                out.append((a, w))
        return out

    # -- pruning (placeholder) ----------------------------------------------
    def _maybe_prune(self) -> None:
        if len(self) <= self.max_size:
            return
        drop = len(self) - self.max_size
        self._slots = self._slots[drop:]
        self.metas = self.metas[drop:]
        # Re-index edges; drop any referencing pruned entries.
        self.edges = {
            (a - drop, b - drop): w
            for (a, b), w in self.edges.items()
            if a >= drop and b >= drop
        }

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the repo (contents + provenance + connections) to disk."""
        torch.save({"slots": self._slots, "metas": self.metas, "edges": self.edges}, path)

    def load(self, path: str) -> None:
        """Load a previously saved repo."""
        d = torch.load(path, map_location="cpu", weights_only=False)
        self._slots = d["slots"]
        self.metas = d["metas"]
        self.edges = d["edges"]

    def stats(self) -> Dict[str, int]:
        """Quick size/connection counts for logging."""
        return {"entries": len(self), "connections": self.num_connections}

    def facts(self, limit: Optional[int] = None, kind: Optional[str] = None) -> List[str]:
        """Read the repo back as text — "facts we know about the world".

        Args:
            limit: Optionally cap how many to return.
            kind: Optionally filter by provenance kind ("base" / "learned").
        """
        out = [
            m.get("text", "") for m in self.metas
            if m.get("text") and (kind is None or m.get("kind") == kind)
        ]
        return out[:limit] if limit else out
