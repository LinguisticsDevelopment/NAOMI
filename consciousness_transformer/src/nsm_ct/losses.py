"""Losses for the emergent reasoning loop.

No positional action labels: the action choice (absorb / append / respond / skip)
is **not** supervised. It is shaped only by:

    total = w_answer * answer + w_novelty * novelty + w_consistency * consistency

* ``answer`` — cross-entropy of the RESPOND-weighted answer against the gold
  answer. This is the sole task signal; it teaches both *what* to answer and,
  via the RESPOND gate, *when* (the model learns to put response mass on the
  step after the question, with nothing telling it which item that is).
* ``novelty`` — a small **label-free** auxiliary for the APPEND action: commit to
  long-term memory in proportion to how novel an item is vs. what the repo
  already knows. APPEND only pays off in *future* episodes, so it gets no
  within-episode answer gradient; this stands in until cross-episode credit
  assignment (RL) is built. See RESEARCH_NOTES.
* ``consistency`` — the placeholder consciousness consistency term (L2 between
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
    novelty: torch.Tensor
    consistency: torch.Tensor


def consciousness_consistency_loss(states: torch.Tensor) -> torch.Tensor:
    """Mean L2 between consecutive consciousness states (placeholder)."""
    if states.shape[1] < 2:
        return states.new_zeros(())
    diffs = states[:, 1:, :] - states[:, :-1, :]
    return diffs.pow(2).mean()


def novelty_append_loss(append_gates: torch.Tensor, novelty_target: torch.Tensor,
                        step_mask: torch.Tensor) -> torch.Tensor:
    """Push the APPEND gate toward item novelty (masked MSE over real items)."""
    denom = step_mask.sum().clamp(min=1.0)
    return ((append_gates - novelty_target) ** 2 * step_mask).sum() / denom


def compute_losses(
    out: dict,
    batch,
    weight_answer: float,
    weight_novelty: float,
    weight_consistency: float,
) -> LossBreakdown:
    """Combine the answer, novelty, and consistency objectives."""
    answer_loss = F.cross_entropy(out["answer_logits"], batch.answer_target)
    novelty_loss = novelty_append_loss(
        out["append_gates"], out["novelty_target"], batch.step_mask
    )
    consistency_loss = consciousness_consistency_loss(out["states"])

    total = (
        weight_answer * answer_loss
        + weight_novelty * novelty_loss
        + weight_consistency * consistency_loss
    )
    return LossBreakdown(
        total=total, answer=answer_loss, novelty=novelty_loss, consistency=consistency_loss
    )
