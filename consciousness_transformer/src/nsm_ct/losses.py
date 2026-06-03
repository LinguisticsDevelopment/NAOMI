"""Losses for the stateful reasoning loop.

The objective is a weighted sum of three terms:

    total = w_answer * answer + w_action * action + w_consistency * consistency

* ``answer`` — did it answer correctly? Cross-entropy over multiple-choice
  options, or over the open-ended answer vocabulary. This is the main signal.
* ``action`` — weak supervision that teaches the *procedure*: each context
  statement should be ABSORBed, and the question should trigger RESPOND. This is
  what makes the model "know it needs to respond."
* ``consistency`` — the **placeholder** auxiliary consciousness term (L2 between
  consecutive states; a "don't thrash" prior). TODO(consciousness-loss): replace
  with the real formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import ACTION_ABSORB, ACTION_RESPOND


@dataclass
class LossBreakdown:
    """All loss components, for logging."""

    total: torch.Tensor
    answer: torch.Tensor
    action: torch.Tensor
    consistency: torch.Tensor


def consciousness_consistency_loss(states: torch.Tensor) -> torch.Tensor:
    """Mean L2 between consecutive consciousness states (placeholder).

    Args:
        states: ``[B, S, dim]`` the state trajectory across an episode.
    """
    if states.shape[1] < 2:
        return states.new_zeros(())
    diffs = states[:, 1:, :] - states[:, :-1, :]
    return diffs.pow(2).mean()


def compute_losses(
    out: dict,
    batch,
    weight_answer: float,
    weight_action: float,
    weight_consistency: float,
) -> LossBreakdown:
    """Combine answer, action, and consistency objectives.

    Args:
        out: Output dict from :meth:`nsm_ct.agent.Mind.forward`.
        batch: The :class:`~nsm_ct.dataset.EpisodeBatch` that produced ``out``.
        weight_*: Term weights.
    """
    answer_logits = out["answer_logits"]
    answer_loss = F.cross_entropy(answer_logits, batch.answer_target)

    # Action supervision: context steps -> ABSORB (masked by real steps),
    # question step -> RESPOND.
    ctx_logits = out["ctx_action_logits"]            # [B, T, A]
    b, t, a = ctx_logits.shape
    if t > 0:
        targets = torch.full((b, t), ACTION_ABSORB, dtype=torch.long, device=ctx_logits.device)
        ce = F.cross_entropy(
            ctx_logits.reshape(b * t, a), targets.reshape(b * t), reduction="none"
        ).reshape(b, t)
        step_mask = batch.step_mask
        ctx_action_loss = (ce * step_mask).sum() / step_mask.sum().clamp(min=1.0)
    else:
        ctx_action_loss = ctx_logits.new_zeros(())

    q_targets = torch.full(
        (b,), ACTION_RESPOND, dtype=torch.long, device=ctx_logits.device
    )
    q_action_loss = F.cross_entropy(out["q_action_logits"], q_targets)
    action_loss = ctx_action_loss + q_action_loss

    consistency_loss = consciousness_consistency_loss(out["states"])

    total = (
        weight_answer * answer_loss
        + weight_action * action_loss
        + weight_consistency * consistency_loss
    )
    return LossBreakdown(
        total=total, answer=answer_loss, action=action_loss, consistency=consistency_loss
    )
