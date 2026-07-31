"""The read-encoder — the controller's state-view onto memory (M3, v1).

The controller never holds meaning as a differentiable blob; it *reads* the
order-3 ``entity_memory`` at an attended ``(focus, relation)`` and gets back a
value vector. **v1 is deliberately minimal — it IS the §0n focus-chaining read**
(``entity_memory.query``), the parameter-free TPR unbind that already cracked
held-out multi-hop. A GNN over the symbolic graph's edges + node handles is the
documented upgrade (see ``MIND_ARCHITECTURE.md``), not the v1.

Kept as a tiny module so the "read-encoder" box in the architecture is a real,
swappable seam: replace :meth:`ReadEncoder.read` with a learned encoder later
without touching the controller.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .. import entity_memory as em


class ReadEncoder(nn.Module):
    """Read the value bound at ``(focus, relation)`` in the order-3 memory.

    Args:
        dim: Vector dimension ``d`` (must match the codec / memory).
        project: If True, apply a small learned projection to the read value
            (default False — v1 is the bare parameter-free TPR unbind).
    """

    def __init__(self, dim: int, *, project: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.proj: Optional[nn.Linear] = nn.Linear(dim, dim) if project else None

    def read(self, memory: torch.Tensor, focus: torch.Tensor,
             relation: torch.Tensor) -> torch.Tensor:
        """``[B, d]`` value at ``(focus, relation)`` — the §0n focus-chaining read."""
        v = em.query(memory, focus, relation)
        return self.proj(v) if self.proj is not None else v

    forward = read


__all__ = ["ReadEncoder"]
