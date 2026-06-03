"""Working memory with state-gated writes and state-conditioned reads.

This is the store the model writes facts into "by sentence, as the state machine
tells it to," and reads from when answering. It is a small,
attention-based (MemN2N-flavored) memory:

* **Write** is *gated*: at step ``t`` the content vector is scaled by the
  model's ABSORB gate before being placed in slot ``t``. An ungated (gate≈0)
  write leaves memory essentially untouched, so the state genuinely controls
  what is remembered.
* **Read** is a single attention lookup: the query is a projection of the
  current consciousness state; it attends over the slots written so far.

The slot tensor is *threaded through the unroll* (like the state) rather than
stored on the module, which keeps autograd clean and batching simple. The module
itself only holds the projection layers.

TODO(memory): per-episode working memory only — no long-term/episodic
persistence across episodes yet, and no pruning/forgetting (see RESEARCH_NOTES).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MemoryState:
    """The threaded state of working memory for one batch.

    Attributes:
        slots: ``[B, num_slots, mem_dim]`` slot contents.
        filled: ``[B, num_slots]`` 1.0 where a slot has been written, else 0.0.
    """

    slots: torch.Tensor
    filled: torch.Tensor


class WorkingMemory(nn.Module):
    """Read/write attention memory.

    Args:
        mem_dim: Width of memory slots and the read vector.
        state_dim: Width of the consciousness state (the read query source).
    """

    def __init__(self, mem_dim: int, state_dim: int) -> None:
        super().__init__()
        self.mem_dim = mem_dim
        self.query_proj = nn.Linear(state_dim, mem_dim)
        self.key_proj = nn.Linear(mem_dim, mem_dim)
        self.value_proj = nn.Linear(mem_dim, mem_dim)

    def init_state(self, batch_size: int, num_slots: int, device: torch.device) -> MemoryState:
        """Allocate empty memory for a fresh episode batch."""
        return MemoryState(
            slots=torch.zeros(batch_size, num_slots, self.mem_dim, device=device),
            filled=torch.zeros(batch_size, num_slots, device=device),
        )

    def write(
        self, mem: MemoryState, slot_idx: int, content: torch.Tensor, gate: torch.Tensor
    ) -> MemoryState:
        """Write ``gate * content`` into slot ``slot_idx`` (out-of-place).

        Args:
            mem: Current memory state.
            slot_idx: Which slot to write (typically the context step index).
            content: ``[B, mem_dim]`` content to store.
            gate: ``[B]`` per-example write strength in [0, 1] (the ABSORB gate).
        """
        slots = mem.slots.clone()
        filled = mem.filled.clone()
        slots[:, slot_idx, :] = gate.unsqueeze(-1) * content
        filled[:, slot_idx] = gate
        return MemoryState(slots=slots, filled=filled)

    def read(self, mem: MemoryState, state: torch.Tensor) -> torch.Tensor:
        """Attend over written slots using a query derived from ``state``.

        Returns ``[B, mem_dim]``; all-zeros for examples with no writes yet.
        """
        query = self.query_proj(state)                      # [B, mem_dim]
        keys = self.key_proj(mem.slots)                     # [B, S, mem_dim]
        scores = (keys * query.unsqueeze(1)).sum(-1) / math.sqrt(self.mem_dim)  # [B, S]
        # Mask empty slots.
        mask = mem.filled > 0                               # [B, S]
        scores = scores.masked_fill(~mask, float("-inf"))
        has_mem = mask.any(dim=1, keepdim=True)             # [B, 1]
        # Avoid NaNs when a row has no memory: softmax over all -inf -> set to 0.
        scores = torch.where(has_mem, scores, torch.zeros_like(scores))
        attn = torch.softmax(scores, dim=1)                 # [B, S]
        attn = attn * mask.float()                          # zero out empty slots
        values = self.value_proj(mem.slots)                 # [B, S, mem_dim]
        read = (attn.unsqueeze(-1) * values).sum(dim=1)     # [B, mem_dim]
        return read * has_mem.float()                       # zero read when empty

    def occupancy(self, mem: MemoryState) -> torch.Tensor:
        """Total write strength per example (``[B]``) — useful for tests/metrics."""
        return mem.filled.sum(dim=1)
