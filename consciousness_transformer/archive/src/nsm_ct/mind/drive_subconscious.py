"""Subconscious-style self-training for the sequential-RL drive (M16).

Parallel to :class:`~nsm_ct.mind.subconscious_loop.SubconsciousLoop` (separate model, batch,
and loss — so the reasoning controller's path is untouched), this rides the same round/replay/
anneal/checkpoint shape: each round samples on-policy dialogue rollouts against the grounded
user simulator (:mod:`nsm_ct.mind.drive_rollout`), then takes REINFORCE-with-baseline steps
(:func:`nsm_ct.mind.drive.drive_rl_loss`). Temperature and the entropy bonus anneal
high→low across rounds (the project's soft→discrete regime); the reward is the symbolic floor
(goal-progress), so there is no learned reward model to drift.

Empirical note (M16): warm-starting the policy from the M15 supervised weights helps as an
*initialization* (it starts competent on goal turns), but a KL **anchor** back to that teacher
*hurts* — the supervised policy is "always-act", which is wrong on distractor turns, so anchoring
fights the very behaviour RL must learn. Hence ``w_anchor`` defaults to 0.0; it is kept as a knob.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch

from . import drive
from .drive_rollout import (RewardCfg, build_rl_batch, greedy_act_fn, rl_act_fn,
                            rollout, rollout_metrics, sample_goal_world)


class DriveSelfTrain:
    """RL self-training of a value-headed :class:`~nsm_ct.mind.drive.DrivePolicy`."""

    def __init__(self, policy: "drive.DrivePolicy", *, lr: float = 2e-3, total_rounds: int = 30,
                 gamma: float = 0.9, w_value: float = 0.3, t0: float = 1.0, t1: float = 0.3,
                 ent0: float = 0.01, ent1: float = 0.003, w_anchor0: float = 0.0,
                 ref_policy: Optional["drive.DrivePolicy"] = None,
                 reward_cfg: Optional[RewardCfg] = None, episodes: int = 300, steps: int = 20,
                 seed: int = 0) -> None:
        # Defaults are the M16-validated robust recipe (goal-rate 0.975 across seeds 0/1/2).
        # Gentle lr + moderate initial temperature + a larger batch avoid the early
        # "push VOLUNTEER down everywhere before `focus` is learned" collapse.
        assert policy.val_head is not None, "RL needs a value-headed DrivePolicy"
        self.policy = policy
        self.opt = torch.optim.AdamW(policy.parameters(), lr=lr)
        self.total_rounds = max(total_rounds, 1)
        self.gamma = gamma
        self.w_value = w_value
        self.t0, self.t1, self.ent0, self.ent1, self.w_anchor0 = t0, t1, ent0, ent1, w_anchor0
        self.ref = ref_policy
        self.cfg = reward_cfg or RewardCfg(gamma=gamma)
        self.episodes, self.steps = episodes, steps
        self.rng = random.Random(seed)
        self._round = 0

    def _anneal(self):
        frac = self._round / max(self.total_rounds - 1, 1)
        return (self.t0 * (1 - frac) + self.t1 * frac,                 # temperature
                self.ent0 * (1 - frac) + self.ent1 * frac,             # entropy weight
                self.w_anchor0 * (1 - frac))                           # anchor weight (→0)

    def self_train(self) -> Dict[str, float]:
        """One round: on-policy rollouts → REINFORCE-with-baseline steps under the anneal."""
        temp, w_ent, w_anc = self._anneal()
        eps = [rollout(rl_act_fn(self.policy, temperature=temp),
                       sample_goal_world(self.rng), self.rng, self.cfg)
               for _ in range(self.episodes)]
        batch = build_rl_batch(eps)
        self.policy.train()
        last = {}
        for _ in range(self.steps):
            self.opt.zero_grad()
            loss = drive.drive_rl_loss(self.policy, batch["feats"], batch["mask"],
                                       batch["actions"], batch["returns"], temperature=temp,
                                       w_value=self.w_value, w_entropy=w_ent,
                                       ref_policy=self.ref, w_anchor=w_anc)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            last = {k: float(v.detach()) for k, v in loss.items()}
        reached = sum(e.reached for e in eps) / max(len(eps), 1)
        self._round += 1
        return {"round": self._round, "temp": temp, "train_goal_rate": reached, **last}

    def run(self, rounds: int, *, eval_every: int = 0, eval_n: int = 2000,
            verbose: bool = True) -> List[Dict[str, float]]:
        history = []
        for _ in range(rounds):
            m = self.self_train()
            if eval_every and (self._round % eval_every == 0):
                ev = rollout_metrics(greedy_act_fn(self.policy), n=eval_n,
                                     seed=10_000 + self._round, cfg=self.cfg)
                m.update({f"val_{k}": v for k, v in ev.items()})
            history.append(m)
            if verbose:
                extra = (f" val_goal={m.get('val_goal_rate', float('nan')):.3f}"
                         if "val_goal_rate" in m else "")
                print(f"round {m['round']:>3} | return={m.get('mean_return', float('nan')):.2f}"
                      f" train_goal={m['train_goal_rate']:.3f} temp={m['temp']:.2f}{extra}",
                      flush=True)
        return history


__all__ = ["DriveSelfTrain"]
