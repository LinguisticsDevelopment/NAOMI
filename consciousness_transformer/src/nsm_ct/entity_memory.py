"""Differentiable, parameter-free order-3 entity memory (entity ⊗ relation ⊗ value).

The cross-clause substrate (TPR-RNN shape): a batched ``[B, d, d, d]`` tensor that
binds an entity-variable to a relation to a value. It has **no learned parameters**
— the controller (``clause_reactor``) supplies the write *gate*; the binding /
unbinding are fixed TPR ops. Writes are **gated overwrites** (move the (entity,
relation) slot toward the new value by ``gate``), so gate≈1 updates a fact
(recency) and gate≈0 ignores it — the learned reaction.
"""

from __future__ import annotations

import torch


def init_memory(batch: int, dim: int, device: torch.device) -> torch.Tensor:
    """An empty ``[B, d, d, d]`` entity⊗relation⊗value memory."""
    return torch.zeros(batch, dim, dim, dim, device=device)


def query(memory: torch.Tensor, entity: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
    """Recover the value bound to (entity, relation): ``[B, d]``.

    Exact when the stored (entity, relation) keys are orthonormal; otherwise noisy
    (interference from other bindings), to be cleaned up against the value codebook.
    """
    return torch.einsum("bijk,bi,bj->bk", memory, entity, relation)


def write(
    memory: torch.Tensor,
    entity: torch.Tensor,
    relation: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Gated overwrite of the (entity, relation) slot toward ``value`` (out-of-place).

    ``gate=1`` → the slot becomes ``value`` (old removed, new written: an update);
    ``gate=0`` → unchanged. Differentiable in gate, value, and the keys.

    Args:
        memory: ``[B, d, d, d]``.
        entity, relation, value: ``[B, d]``.
        gate: ``[B]`` write strength in [0, 1] (from the controller).
    """
    old = query(memory, entity, relation)                 # [B, d]
    delta = gate.unsqueeze(-1) * (value - old)            # move slot toward value
    return memory + torch.einsum("bi,bj,bk->bijk", entity, relation, delta)
