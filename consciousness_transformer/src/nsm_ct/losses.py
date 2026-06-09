"""Losses for the emergent reasoning loop.

Nothing about the action choice **or** trust is supervised:

    total = w_answer * answer + w_consistency * consistency

* ``answer`` — cross-entropy of the RESPOND-weighted answer against the gold
  answer. This is the sole task signal. It teaches *what* to answer, *when* (via
  the RESPOND gate), and — because trust scales how strongly each item is written
  into memory — *whom to trust* (the model must discount contradicted info to
  answer correctly). All emergent, no labels.
* ``consistency`` — placeholder consciousness consistency term (L2 between
  consecutive states). TODO(consciousness-loss).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    """All loss components, for logging."""

    total: torch.Tensor
    answer: torch.Tensor
    consistency: torch.Tensor


def consciousness_consistency_loss(states: torch.Tensor) -> torch.Tensor:
    """Mean L2 between consecutive consciousness states (placeholder)."""
    if states.shape[1] < 2:
        return states.new_zeros(())
    diffs = states[:, 1:, :] - states[:, :-1, :]
    return diffs.pow(2).mean()


def compute_losses(
    out: dict,
    batch,
    weight_answer: float,
    weight_consistency: float,
) -> LossBreakdown:
    """Combine the answer objective with the placeholder consistency term."""
    answer_loss = F.cross_entropy(out["answer_logits"], batch.answer_target)
    consistency_loss = consciousness_consistency_loss(out["states"])
    total = weight_answer * answer_loss + weight_consistency * consistency_loss
    return LossBreakdown(total=total, answer=answer_loss, consistency=consistency_loss)
