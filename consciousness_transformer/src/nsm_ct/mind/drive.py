"""The learned drive (M15, L6) — calibrated initiative as a learned policy.

M14 built the conversational substrate (L2 ask, L4 continuity) and instrumented every
turn with a ``TurnOutcome``; M15-L3 added the *volunteer* action. The drive is the
**policy that decides when to use those actions** — answer-terse vs answer+volunteer vs
ask vs stay-quiet — the line between a yappy chatbot and old Siri.

In the grain of this codebase (100% supervised-imitation-of-a-symbolic-oracle, zero RL),
the drive is a small discrete-choice head trained with **cross-entropy against a grounded
gold action**, exactly like M12's learned act-routing (:mod:`nsm_ct.mind.routing`). What
makes it *genuinely learned* rather than a hand-coded rule: the gold "ideal action" is set
by a synthetic-user environment (:mod:`nsm_ct.mind.drive_env`) that knows a **latent
usefulness** the policy never sees — the policy must learn to predict it from grounded,
decision-time features (relevance/novelty, backlog, focus). Sequential RL over the
``TurnOutcome`` reward is the documented next step, not this milestone.

Boundary (the "no information in weights" invariant): the drive learns *when to act*; it
can only ever gate **grounded, feasible** actions (you cannot volunteer a fact that was not
derived, nor ask for a premise that no rule needs). It never invents content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# The action vocabulary. In an *answered* context the live choice is {ANSWER, VOLUNTEER};
# in a *blocked* context it is {ASK, QUIET}. Feasibility masking (below) restricts the
# 4-way head to the live pair, so one policy learns both binary initiative decisions.
ANSWER, VOLUNTEER, ASK, QUIET = 0, 1, 2, 3
DRIVE_ACTS = ("ANSWER", "VOLUNTEER", "ASK", "QUIET")
N_ACTS = 4

# Normalisation caps for the (otherwise unbounded) count features — keeps inputs ~[0,1]
# and shared identically by the environment and the live conversation.
_CAP = {"cand": 5.0, "depth": 6.0, "pending": 6.0, "gained": 5.0, "focus": 5.0}
N_FEATURES = 8


@dataclass
class DriveContext:
    """The grounded, decision-time signals the drive conditions on — computed the same
    way by the synthetic environment and the live :class:`~nsm_ct.mind.conversation.Conversation`.
    None of these directly encodes the latent usefulness; the policy must learn the proxy."""

    derivable: bool        # the query could be answered (answered context)
    has_premise: bool      # blocked, but a missing premise exists (ask is feasible)
    n_candidates: int      # size of the volunteer pool (on-topic, derived, unsaid)
    top_depth: int         # derivation depth of the most-relevant candidate/premise (novelty)
    answer_steps: int      # derivation depth of the answer (effort)
    pending: int           # backlog of unresolved questions (overload signal)
    knowledge_gained: int  # newly-derivable facts from the last statement
    focus: int             # focus persistence on the current subject (continuity)


def features_vec(ctx: DriveContext) -> torch.Tensor:
    """``DriveContext`` → a fixed-length float feature vector (the policy input)."""
    return torch.tensor([
        1.0 if ctx.derivable else 0.0,
        1.0 if ctx.has_premise else 0.0,
        min(ctx.n_candidates, _CAP["cand"]) / _CAP["cand"],
        min(ctx.top_depth, _CAP["depth"]) / _CAP["depth"],
        min(ctx.answer_steps, _CAP["depth"]) / _CAP["depth"],
        min(ctx.pending, _CAP["pending"]) / _CAP["pending"],
        min(ctx.knowledge_gained, _CAP["gained"]) / _CAP["gained"],
        min(ctx.focus, _CAP["focus"]) / _CAP["focus"],
    ], dtype=torch.float32)


def feasible_mask(ctx: DriveContext) -> torch.Tensor:
    """The grounded feasibility of each action in this context (the invariant that keeps
    the drive honest): in an answered context only ANSWER/VOLUNTEER are possible (and
    VOLUNTEER only if there is something true to add); in a blocked context only ASK/QUIET."""
    m = torch.zeros(N_ACTS)
    if ctx.derivable:
        m[ANSWER] = 1.0
        if ctx.n_candidates > 0:
            m[VOLUNTEER] = 1.0
    else:                                                # blocked-with-premise decision
        m[ASK] = 1.0
        m[QUIET] = 1.0
    return m


class DrivePolicy(nn.Module):
    """A small MLP over the grounded feature vector → a 4-way action distribution. A new
    member of the controller family: it governs *initiative*, just as the reasoning
    controller governs *derivation* — and it holds no facts, only the when-to-act policy.

    M16 splits the trunk from the action head and adds an optional **value head** (a
    state-value baseline for the sequential-RL advantage). The split is transparent to M15:
    ``forward`` still returns the action logits, and M15 checkpoints warm-start via
    :func:`load_m15_into` (legacy ``net.*`` keys remap onto ``trunk``/``act_head``)."""

    def __init__(self, hidden: int = 16, *, value_head: bool = False) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(N_FEATURES, hidden), nn.ReLU())
        self.act_head = nn.Linear(hidden, N_ACTS)
        self.val_head = nn.Linear(hidden, 1) if value_head else None

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.act_head(self.trunk(feats))          # [..., 4] logits (M15-compatible)

    def forward_with_value(self, feats: torch.Tensor):
        """``feats`` → ``(action logits [...,4], state value [...])`` — the RL forward."""
        assert self.val_head is not None, "DrivePolicy was built without a value head"
        h = self.trunk(feats)
        return self.act_head(h), self.val_head(h).squeeze(-1)


def _remap_legacy(state: dict) -> dict:
    """Remap an M15 ``DrivePolicy`` state-dict (``net.0.*`` / ``net.2.*``) onto the M16
    ``trunk``/``act_head`` layout, so a supervised checkpoint warm-starts the RL policy."""
    out = {}
    for k, v in state.items():
        if k.startswith("net.0."):
            out["trunk.0." + k[len("net.0."):]] = v
        elif k.startswith("net.2."):
            out["act_head." + k[len("net.2."):]] = v
        else:
            out[k] = v
    return out


def load_m15_into(policy: "DrivePolicy", state: dict) -> "DrivePolicy":
    """Warm-start ``policy`` from an M15 (or M16) state-dict. The trunk+act_head load
    exactly (so ``forward`` is bit-identical to M15); a fresh value head is left intact."""
    policy.load_state_dict(_remap_legacy(state), strict=False)
    return policy


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Push infeasible actions to -inf so they are never chosen / never carry loss mass."""
    return logits.masked_fill(mask < 0.5, float("-inf"))


def _safe_entropy(masked_logits: torch.Tensor) -> torch.Tensor:
    """Entropy of a masked categorical, NaN-safe in *both* forward and backward: masked
    actions have ``-inf`` log-probs (prob 0). We replace those ``-inf`` with a finite 0
    *before* the multiply, so the product is never ``0·-inf`` (which yields NaN gradients
    even when the forward value is masked away)."""
    lp = F.log_softmax(masked_logits, dim=-1)
    lp_safe = lp.masked_fill(torch.isinf(lp), 0.0)       # prob is ~0 there; value is irrelevant
    p = lp.exp()                                         # exactly 0 at masked actions
    return -(p * lp_safe).sum(-1)


def drive_loss(logits: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked cross-entropy of the policy's action logits vs the environment's gold action
    (the M12 routing pattern, restricted to feasible actions). ``logits``/``mask`` are
    ``[B, 4]``, ``gold`` is ``[B]``."""
    return F.cross_entropy(_masked_logits(logits, mask), gold)


def predict_action(policy: DrivePolicy, feats: torch.Tensor, mask: torch.Tensor,
                   *, temperature: float = 1.0) -> int:
    """The drive's chosen action — masked argmax over feasible actions (temperature is the
    soft→discrete knob; at runtime it is sharp)."""
    policy.eval()
    with torch.no_grad():
        logits = _masked_logits(policy(feats) / max(temperature, 1e-6), mask)
    return int(logits.argmax(-1))


# -- sequential RL (M16): sampling + REINFORCE-with-baseline ----------------------
def sample_action(policy: DrivePolicy, feats: torch.Tensor, mask: torch.Tensor,
                  *, temperature: float = 1.0):
    """Sample an action from the masked policy (the soft→discrete temperature is the
    exploration knob). Infeasible actions get ``-inf`` logits ⇒ exactly zero probability,
    so sampling can never pick an ungrounded action. Returns ``(action, log_prob, entropy)``."""
    with torch.no_grad():
        masked = _masked_logits(policy(feats) / max(temperature, 1e-6), mask)
        dist = Categorical(logits=masked)
        a = dist.sample()
    return int(a), dist.log_prob(a), _safe_entropy(masked)


def kl_to_reference(logits: torch.Tensor, ref_logits: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """KL(π ‖ π_ref) over feasible actions — the trust-region anchor to the warm-start
    (M16's analog of clause_psyche's geometric exploration prior, anchored to the teacher)."""
    lp = F.log_softmax(_masked_logits(logits, mask), dim=-1)
    lq = F.log_softmax(_masked_logits(ref_logits, mask), dim=-1)
    infeasible = torch.isinf(lp)
    lp_safe = lp.masked_fill(infeasible, 0.0)
    lq_safe = lq.masked_fill(infeasible, 0.0)            # same masked positions
    p = lp.exp()                                         # 0 at masked actions
    return (p * (lp_safe - lq_safe)).sum(-1).mean()


def drive_rl_loss(policy: DrivePolicy, feats: torch.Tensor, mask: torch.Tensor,
                  actions: torch.Tensor, returns: torch.Tensor, *, temperature: float = 1.0,
                  w_value: float = 0.5, w_entropy: float = 0.01,
                  ref_policy: "DrivePolicy" = None, w_anchor: float = 0.0):
    """REINFORCE-with-baseline over a flattened batch of rollout transitions (mirrors the
    clause_psyche ``-reward`` objective: minimize ``-E[return]`` via the score function,
    plus a value baseline, an entropy exploration term, and an optional KL anchor to the
    M15 policy). Recomputes log-probs/values/entropy with grad from the stored
    ``(feats, mask, actions)`` — so no autograd graph is held across the rollout."""
    logits, values = policy.forward_with_value(feats)
    masked = _masked_logits(logits / max(temperature, 1e-6), mask)
    dist = Categorical(logits=masked)
    log_probs = dist.log_prob(actions)
    advantage = (returns - values).detach()              # baseline-centered score weight
    policy_loss = -(log_probs * advantage).mean()
    value_loss = F.smooth_l1_loss(values, returns)
    entropy = _safe_entropy(masked).mean()
    total = policy_loss + w_value * value_loss - w_entropy * entropy
    anchor = total.new_zeros(())
    if ref_policy is not None and w_anchor > 0.0:
        with torch.no_grad():
            ref_logits = ref_policy(feats)
        anchor = kl_to_reference(logits, ref_logits, mask)
        total = total + w_anchor * anchor
    return {"total": total, "policy": policy_loss.detach(), "value": value_loss.detach(),
            "entropy": entropy.detach(), "anchor": anchor.detach(),
            "mean_return": returns.mean().detach()}


# -- live-conversation hooks (used by Conversation when a drive is attached) -----
def drive_features(conv, query, answer, candidates, *, derivable=True,
                   has_premise=False, premise=None) -> "DriveContext":
    """Build the grounded :class:`DriveContext` from the *live* conversation state — the
    same signals the environment trains on. Reuses the conversation's own closure helpers."""
    from . import verbalize
    from ..reasoning_oracle import forward_chain
    facts = conv._facts()
    _known, chain = forward_chain(list(facts), conv._rules())
    top = candidates[0] if candidates else premise
    top_depth = len(verbalize._relevant_steps(chain, top)) if top is not None else 0
    answered_lit = (query[0], query[1], query[2], "+") if len(query) >= 4 \
        else (query[0], query[1], answer, "+")
    answer_steps = len(verbalize._relevant_steps(chain, answered_lit)) if derivable else 0
    gained = next((o.knowledge_gained for o in reversed(conv.log)
                   if o.kind == "learned"), 0)
    focus = sum(1 for st in conv.statements if len(st) > 1 and st[1] == query[0])
    return DriveContext(
        derivable=derivable, has_premise=has_premise, n_candidates=len(candidates),
        top_depth=top_depth, answer_steps=answer_steps, pending=len(conv.pending),
        knowledge_gained=gained, focus=focus)


def wants_volunteer(policy: DrivePolicy, ctx: DriveContext) -> bool:
    """Answered-context gate: does the drive choose to surface the extra fact?"""
    return predict_action(policy, features_vec(ctx), feasible_mask(ctx)) == VOLUNTEER


def wants_ask(policy: DrivePolicy, ctx: DriveContext) -> bool:
    """Blocked-context gate: does the drive choose to ask the premise vs stay quiet?"""
    return predict_action(policy, features_vec(ctx), feasible_mask(ctx)) == ASK


__all__ = [
    "ANSWER", "VOLUNTEER", "ASK", "QUIET", "DRIVE_ACTS", "N_ACTS", "N_FEATURES",
    "DriveContext", "features_vec", "feasible_mask", "DrivePolicy", "drive_loss",
    "predict_action", "drive_features", "wants_volunteer", "wants_ask",
    "load_m15_into", "sample_action", "drive_rl_loss", "kl_to_reference",
]
