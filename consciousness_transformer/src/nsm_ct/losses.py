"""Loss functions for the Consciousness Transformer.

The training objective is a weighted sum of three terms (per the brief):

    total = w_lm * lm_loss + w_answer * answer_loss + w_consistency * consistency_loss

* ``lm_loss`` — language modeling over the correct response (the gold option's
  answer region).
* ``answer_loss`` — cross-entropy over the 4 options, using the LM-derived
  option log-likelihoods as logits.
* ``consistency_loss`` — **placeholder** auxiliary consciousness consistency
  term. Currently the L2 distance between the predicted next state and the
  current state (i.e. a "don't drift" prior). TODO(consciousness-loss): replace
  with the real formulation; an inertial L2 prior is almost certainly not it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .features import Batch
from .model import ModelOutput


def consciousness_consistency_loss(
    current_state: torch.Tensor, next_state: torch.Tensor
) -> torch.Tensor:
    """L2 distance between consecutive consciousness states (placeholder).

    Args:
        current_state: ``[*, dim]`` the state fed in.
        next_state: ``[*, dim]`` the state the transition head predicts.

    Returns:
        Scalar mean squared distance.

    TODO(consciousness-loss): the real auxiliary objective should encode
    *coherent* state evolution (consistency with evidence, memory, and the
    answer), not mere temporal stability.
    """
    return F.mse_loss(next_state, current_state)


@dataclass
class LossBreakdown:
    """All loss components for logging."""

    total: torch.Tensor
    lm: torch.Tensor
    answer: torch.Tensor
    consistency: torch.Tensor


def compute_losses(
    output: ModelOutput,
    batch: Batch,
    weight_lm: float,
    weight_answer: float,
    weight_consistency: float,
) -> LossBreakdown:
    """Combine the three objectives into a weighted total.

    Args:
        output: Forward-pass output.
        batch: The batch that produced ``output`` (for the gold answer indices
            and the current consciousness state).
        weight_lm / weight_answer / weight_consistency: Term weights.
    """
    device = output.option_logits.device
    num_options = batch.num_options
    gold = batch.answer_idx  # [B]

    # Indices of the gold rows within the flattened [N] dimension.
    b = gold.shape[0]
    base = torch.arange(b, device=device) * num_options
    gold_rows = base + gold  # [B]

    # 1) LM loss: only over the correct response.
    lm_loss = output.lm_loss_per_row[gold_rows].mean()

    # 2) Answer loss: cross-entropy over options.
    answer_loss = F.cross_entropy(output.option_logits, gold)

    # 3) Consistency loss: predicted-next vs current state on the gold rows.
    current_state = batch.consciousness[gold_rows]
    next_state = output.next_consciousness[gold_rows]
    consistency_loss = consciousness_consistency_loss(current_state, next_state)

    total = (
        weight_lm * lm_loss
        + weight_answer * answer_loss
        + weight_consistency * consistency_loss
    )
    return LossBreakdown(
        total=total, lm=lm_loss, answer=answer_loss, consistency=consistency_loss
    )
