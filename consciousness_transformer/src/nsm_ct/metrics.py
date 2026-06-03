"""Evaluation metrics for the reasoning loop."""

from __future__ import annotations

import torch

from .model import ACTION_ABSORB, ACTION_RESPOND


@torch.no_grad()
def answer_accuracy(out: dict, batch) -> float:
    """Fraction of episodes whose predicted answer matches the target."""
    preds = out["answer_logits"].argmax(dim=-1)
    return float((preds == batch.answer_target).float().mean())


@torch.no_grad()
def action_accuracy(out: dict, batch) -> float:
    """Fraction of step decisions that match the weak-supervision targets.

    Context statements should be ABSORBed; the question should trigger RESPOND.
    Padded context steps are excluded.
    """
    ctx_logits = out["ctx_action_logits"]            # [B, T, A]
    q_logits = out["q_action_logits"]                # [B, A]
    step_mask = batch.step_mask                       # [B, T]

    if ctx_logits.shape[1] > 0:
        ctx_correct = (ctx_logits.argmax(-1) == ACTION_ABSORB).float() * step_mask
        ctx_hits = ctx_correct.sum()
    else:
        ctx_hits = q_logits.new_zeros(())
    q_hits = (q_logits.argmax(-1) == ACTION_RESPOND).float().sum()

    denom = step_mask.sum() + q_logits.shape[0]
    return float((ctx_hits + q_hits) / denom.clamp(min=1.0))
