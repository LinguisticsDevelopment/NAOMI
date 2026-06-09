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


@torch.no_grad()
def trust_gap(out, batch) -> float:
    """Does the model trust corroborated info more than contradicted info?

    Diagnostic only (the trust labels are never used in training). Returns
    mean(trust on trustworthy items) - mean(trust on contradicted items) over
    real items where both groups are present; 0.0 if no contradictions in batch.
    """
    if not hasattr(batch, "trust_label") or batch.trust_label is None:
        return 0.0
    trust = out["trust"]                         # [B, T]
    lab = batch.trust_label                      # [B, T] (1 trustworthy, 0 contradicted)
    real = batch.step_mask > 0
    trustworthy = (lab > 0.5) & real
    contradicted = (lab < 0.5) & real
    if contradicted.sum() == 0 or trustworthy.sum() == 0:
        return 0.0
    return float(trust[trustworthy].mean() - trust[contradicted].mean())
