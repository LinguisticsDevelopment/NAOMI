"""Evaluation metrics for the emergent reasoning loop."""

from __future__ import annotations

import torch


@torch.no_grad()
def answer_accuracy(out: dict, batch) -> float:
    """Fraction of episodes whose RESPOND-weighted answer matches the target."""
    preds = out["answer_logits"].argmax(dim=-1)
    return float((preds == batch.answer_target).float().mean())


@torch.no_grad()
def mean_respond_position(out, batch) -> float:
    """Where in the stream the model concentrates its response mass (0..1).

    A neutral diagnostic for emergent timing: 0 = responds at the first item,
    1 = responds at the last. The model is never told when to respond; this just
    shows where it chose to. (It often responds as soon as it has the answer,
    which for easy episodes is before the question — a valid emergent strategy.)
    """
    gates = out["respond_gates"] * batch.step_mask              # [B, T]
    t = gates.shape[1]
    if t <= 1:
        return 0.0
    w = gates / gates.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, T]
    positions = torch.arange(t, device=gates.device).float()
    pos = (w * positions).sum(dim=1) / (t - 1)                  # [B]
    return float(pos.mean())
