"""Losses for the emergent reasoning loop.

Nothing about the action choice **or** trust is supervised:

    total = w_answer·answer + w_mem·mem_answer + w_multi·multi_answer
            + w_consistency·consistency

* ``answer`` — cross-entropy of the RESPOND-weighted answer against the gold
  answer. The core task signal: it teaches *what* to answer, *when* (via the
  RESPOND gate), and — because trust scales how strongly each item is written
  into memory — *whom to trust*. All emergent, no labels.
* ``mem_answer`` — the same answer read out through the **memory bottleneck**
  (response head with the state zeroed). The answer must be recoverable from
  memory alone, which makes the trust-gated writes load-bearing: the state can't
  smuggle the answer past memory.
* ``multi_answer`` — per-question cross-entropy for multi-question streams
  (several questions answered in one unreset run); trains chained-question
  consistency as a capability. Zero when the batch has a single question or the
  weight is 0.
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
    mem_answer: torch.Tensor
    multi_answer: torch.Tensor
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
    weight_mem_answer: float = 0.0,
    weight_multi: float = 0.0,
) -> LossBreakdown:
    """Combine the answer objectives with the placeholder consistency term."""
    answer_loss = F.cross_entropy(out["answer_logits"], batch.answer_target)
    consistency_loss = consciousness_consistency_loss(out["states"])

    mem_loss = answer_loss.new_zeros(())
    if weight_mem_answer > 0 and out.get("answer_logits_mem") is not None:
        mem_loss = F.cross_entropy(out["answer_logits_mem"], batch.answer_target)

    multi_loss = answer_loss.new_zeros(())
    if weight_multi > 0 and out.get("question_logits") is not None:
        valid = batch.q_targets >= 0
        if bool(valid.any()):
            multi_loss = F.cross_entropy(
                out["question_logits"][valid], batch.q_targets[valid]
            )

    total = (
        weight_answer * answer_loss
        + weight_mem_answer * mem_loss
        + weight_multi * multi_loss
        + weight_consistency * consistency_loss
    )
    return LossBreakdown(
        total=total, answer=answer_loss, mem_answer=mem_loss,
        multi_answer=multi_loss, consistency=consistency_loss,
    )
