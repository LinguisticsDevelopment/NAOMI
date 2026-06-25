"""Multi-turn rollout environment for the sequential-RL drive (M16).

M15's drive learns from *independent* decision points; this module gives it **dialogues**,
so it can learn from consequences. A simulated user has a hidden goal — the deepest
derivable fact of a real `sample_world` chain — reachable only by establishing its support
literals *in order*. The user reacts to the drive's action each turn; the episode return is
computed from **symbolic goal progress** (the floor), no model/judge.

What makes the gain genuinely *sequential* (un-learnable by the single-turn teacher) — three
grounded couplings:

* **Distractor stream.** Half the turns are off-goal. The only reliable signal that separates
  a goal turn from a distractor is ``focus`` (high on-goal, low off-goal) — a feature M15 was
  trained to *ignore* (its gold never depended on focus). The right policy learns to spend
  initiative only on focused turns and hold through distractors.
* **Patience budget.** Each off-path volunteer/ask (a yap/nag) costs user patience; when it
  runs out the user disengages and the goal is *missed*. So wasting initiative on distractors
  directly lowers goal-completion — a whole-dialogue cost.
* **Backlog gating.** Asking creates pending obligations; while overloaded, an on-path ask
  *fails to make progress*. Over-asking now blocks the real goal later.

With the couplings OFF (``distractor_rate=0``) the sequential optimum collapses to the
single-turn optimum and RL ties the supervised teacher — the ablation that proves the gain
is sequential, not a re-tuned proxy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..reasoning_oracle import forward_chain
from .drive import (ANSWER, ASK, QUIET, VOLUNTEER, DriveContext, feasible_mask,
                    features_vec, predict_action, sample_action)
from .drive_env import FOCUS_CAPACITY, sample_world


@dataclass
class RewardCfg:
    r_progress: float = 1.0      # on-path action advances the goal
    r_yap: float = 0.5           # off-path volunteer (wasted initiative)
    r_nag: float = 0.5           # off-path ask (wasted initiative)
    r_missed: float = 0.3        # held back when an on-path action was available
    r_quiet_ok: float = 0.2      # correct restraint on a distractor
    r_goal: float = 3.0          # terminal: the goal became derivable
    patience: int = 3            # off-path actions the user tolerates before disengaging
    max_turns: int = 12
    distractor_rate: float = 0.5
    gamma: float = 0.9


@dataclass
class RolloutWorld:
    subj: str
    chain: List[Tuple]           # the ordered on-path support literals L1..LD (depths 1..D)
    branches: List[Tuple]        # off-path distractor facts
    goal_fact: Tuple


@dataclass
class Transition:
    ctx: DriveContext
    action: int
    reward: float


@dataclass
class Episode:
    transitions: List[Transition]
    reached: bool
    turns: int
    returns: List[float] = field(default_factory=list)   # discounted return-to-go per turn


def sample_goal_world(rng: random.Random, *, max_depth: int = 4) -> RolloutWorld:
    """A real world whose deepest chain literal is the hidden goal; the chain literals are
    the ordered on-path supports, the depth-1 branches are off-path distractors."""
    subj, facts, rules = sample_world(rng, max_depth=max_depth)
    known, chain = forward_chain(list(facts), rules)
    derived = [f for f in known if f not in set(facts) and f[0] == subj and f[3] == "+"]
    chain_lits = sorted((f for f in derived if f[1].startswith("r")),
                        key=lambda f: int(f[1][1:]))     # r1,r2,… in order
    branches = [f for f in derived if f[1].startswith("b")]
    if not chain_lits:                                   # degenerate world → retry
        return sample_goal_world(rng, max_depth=max_depth)
    return RolloutWorld(subj, chain_lits, branches, chain_lits[-1])


class UserSimulator:
    """A grounded user with a hidden goal-path, reacting to the drive's actions (the three
    couplings live here). ``next_context`` poses a turn; ``step`` rewards an action and
    advances state."""

    def __init__(self, world: RolloutWorld, rng: random.Random, cfg: RewardCfg) -> None:
        self.w = world
        self.rng = rng
        self.cfg = cfg
        self.delivered = 0                               # on-path support steps established (in order)
        self.pending = 0
        self.patience = cfg.patience
        self.last_gain = 0
        self.turn = 0

    @property
    def depth(self) -> int:
        return len(self.w.chain)

    def next_context(self, rng: random.Random):
        """Pose the next turn → ``(DriveContext, meta)``. ``meta.on_path`` is the single
        feasible action that advances the goal this turn (``None`` on a distractor)."""
        overload = self.pending > FOCUS_CAPACITY
        goal_turn = (self.delivered < self.depth) and (rng.random() >= self.cfg.distractor_rate)
        blocked = rng.random() < 0.5
        if goal_turn:
            focus = rng.randint(3, 4)                    # HIGH focus ⇒ on-goal (the discriminator)
            if blocked:
                ctx = DriveContext(False, True, 0, 1, 0, self.pending, self.last_gain, focus)
                return ctx, {"on_path": ASK, "overload": overload}
            n_branch = len(self.w.branches)
            ctx = DriveContext(True, False, 1 + n_branch, 1, 1, self.pending, self.last_gain, focus)
            return ctx, {"on_path": VOLUNTEER, "overload": overload}
        # distractor turn (or goal already complete): only off-path initiative is possible
        focus = rng.randint(0, 1)                        # LOW focus ⇒ distractor
        if blocked:
            ctx = DriveContext(False, True, 0, 1, 0, self.pending, self.last_gain, focus)
            return ctx, {"on_path": None, "overload": overload}
        ctx = DriveContext(True, False, max(len(self.w.branches), 1), 1, 1,
                           self.pending, self.last_gain, focus)
        return ctx, {"on_path": None, "overload": overload}

    def step(self, action: int, meta: dict) -> float:
        """Reward ``action`` and advance state (the floor: progress only for a real on-path
        move; patience/backlog couplings make wasted initiative cost the goal)."""
        cfg, overload, on_path = self.cfg, meta["overload"], meta["on_path"]
        r, progress = 0.0, False
        if on_path is not None and action == on_path:
            if action == ASK:
                self.pending += 1
                if not overload:                         # backlog gating: overloaded ask stalls
                    self.delivered += 1; progress = True; r += cfg.r_progress
            else:                                        # on-path VOLUNTEER delivers immediately
                self.delivered += 1; progress = True; r += cfg.r_progress
        elif action in (VOLUNTEER, ASK):                 # initiative spent off-path (distractor)
            r -= (cfg.r_yap if action == VOLUNTEER else cfg.r_nag) * (1 + int(overload))
            self.patience -= (1 + int(overload))
            if action == ASK:
                self.pending += 1
        else:                                            # terse / quiet
            r += (-cfg.r_missed) if on_path is not None else cfg.r_quiet_ok
        self.pending = max(0, self.pending - 1)          # the user clears one obligation per turn
        self.last_gain = 1 if progress else 0
        self.turn += 1
        return r

    def reached(self) -> bool:
        return self.delivered >= self.depth

    def disengaged(self) -> bool:
        return self.patience <= 0


ActFn = Callable[[DriveContext], int]


def rollout(act_fn: ActFn, world: RolloutWorld, rng: random.Random, cfg: RewardCfg) -> Episode:
    """Play one dialogue: each turn the simulator poses a context, ``act_fn`` chooses an
    action, the simulator rewards it. Ends on goal, disengagement, or the turn budget.
    Returns the episode with discounted returns-to-go (incl. the terminal goal bonus)."""
    sim = UserSimulator(world, rng, cfg)
    trans: List[Transition] = []
    reached = False
    for _ in range(cfg.max_turns):
        ctx, meta = sim.next_context(rng)
        action = act_fn(ctx)
        r = sim.step(action, meta)
        if sim.reached():
            r += cfg.r_goal; reached = True
        trans.append(Transition(ctx, action, r))
        if reached or sim.disengaged():
            break
    ep = Episode(trans, reached, len(trans))
    g = 0.0                                              # discounted return-to-go, backward
    returns: List[float] = []
    for t in reversed(trans):
        g = t.reward + cfg.gamma * g
        returns.append(g)
    ep.returns = list(reversed(returns))
    return ep


# -- policy adapters (one interface for RL / supervised / baselines) -------------
def rl_act_fn(policy, *, temperature: float = 1.0) -> ActFn:
    return lambda ctx: sample_action(policy, features_vec(ctx), feasible_mask(ctx),
                                     temperature=temperature)[0]


def greedy_act_fn(policy) -> ActFn:
    return lambda ctx: predict_action(policy, features_vec(ctx), feasible_mask(ctx))


def baseline_act_fn(kind: str) -> ActFn:
    """``always`` = maximal initiative (yappy), ``never`` = none (Siri)."""
    def act(ctx: DriveContext) -> int:
        if ctx.derivable:
            return VOLUNTEER if (kind == "always" and ctx.n_candidates > 0) else ANSWER
        return ASK if kind == "always" else QUIET
    return act


# -- training batch + evaluation -------------------------------------------------
def build_rl_batch(episodes: List[Episode]) -> Dict[str, torch.Tensor]:
    """Flatten transitions across episodes → ``feats/mask/actions/returns`` for one loss call."""
    feats, masks, actions, returns = [], [], [], []
    for ep in episodes:
        for tr, g in zip(ep.transitions, ep.returns):
            feats.append(features_vec(tr.ctx))
            masks.append(feasible_mask(tr.ctx))
            actions.append(tr.action)
            returns.append(g)
    return {"feats": torch.stack(feats), "mask": torch.stack(masks),
            "actions": torch.tensor(actions, dtype=torch.long),
            "returns": torch.tensor(returns, dtype=torch.float32)}


def rollout_metrics(act_fn: ActFn, *, n: int, seed: int, cfg: RewardCfg,
                    max_depth: int = 4) -> Dict[str, float]:
    """Whole-dialogue outcomes over ``n`` held-out worlds — the sequential gate metric:
    goal-completion rate, mean turns-to-goal (over reached episodes), and mean return."""
    rng = random.Random(seed)
    reached = 0
    turns_to_goal: List[int] = []
    rets: List[float] = []
    for _ in range(n):
        world = sample_goal_world(rng, max_depth=max_depth)
        ep = rollout(act_fn, world, rng, cfg)
        rets.append(ep.returns[0] if ep.returns else 0.0)
        if ep.reached:
            reached += 1; turns_to_goal.append(ep.turns)
    return {"goal_rate": reached / max(n, 1),
            "mean_turns_to_goal": (sum(turns_to_goal) / len(turns_to_goal)) if turns_to_goal else float("nan"),
            "mean_return": sum(rets) / max(len(rets), 1)}


__all__ = ["RewardCfg", "RolloutWorld", "Transition", "Episode", "sample_goal_world",
           "UserSimulator", "rollout", "rl_act_fn", "greedy_act_fn", "baseline_act_fn",
           "build_rl_batch", "rollout_metrics"]
