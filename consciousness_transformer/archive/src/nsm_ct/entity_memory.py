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


def query_entity(memory: torch.Tensor, relation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Recover the ENTITY bound to (relation, value): ``[B, d]`` -- the entity-axis
    twin of :func:`query`, unbinding along the FIRST (entity) axis instead of the
    THIRD (value) axis. ``query`` answers "what does entity/relation hold?" (a
    known address, an unknown value); this answers "who holds relation/value?" (a
    known relation+value, an unknown address) -- the M57c.2 inverse-query read
    ("who is tall?" needs the entity dimension unbound, not a value read off an
    address that doesn't exist yet).

    Exact when the stored (relation, value) keys are orthonormal; otherwise noisy
    (interference from other bindings), same caveat as :func:`query`.
    """
    return torch.einsum("bijk,bj,bk->bi", memory, relation, value)


def write(
    memory: torch.Tensor,
    entity: torch.Tensor,
    relation: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    overwrite: torch.Tensor = None,
) -> torch.Tensor:
    """Write ``value`` into the (entity, relation) slot (out-of-place).

    Two decoupled gates let the controller choose its reaction:
    ``delta = gate·value − overwrite·old``. ``overwrite≈gate≈1`` → the slot becomes
    ``value`` (an UPDATE / recency); ``overwrite≈0`` → ``value`` is *added* (a vote —
    repeated assertions accumulate so the majority wins). Differentiable throughout.
    Defaults to the gated-overwrite (``overwrite = gate``).

    Args:
        memory: ``[B, d, d, d]``.
        entity, relation, value: ``[B, d]``.
        gate: ``[B]`` write strength in [0, 1].
        overwrite: ``[B]`` how much of the old value to clear (default = ``gate``).
    """
    old = query(memory, entity, relation)                 # [B, d]
    o = gate if overwrite is None else overwrite
    delta = gate.unsqueeze(-1) * value - o.unsqueeze(-1) * old
    return memory + torch.einsum("bi,bj,bk->bijk", entity, relation, delta)
