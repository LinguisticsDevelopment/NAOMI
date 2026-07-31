"""M16 gates: sequential RL for the learned drive (L6).

The drive RL-finetunes the M15 supervised policy on multi-turn dialogue consequences against
a grounded hidden-goal user simulator. These gates cover the new RL machinery (warm-start
fidelity, mask-safe sampling/loss, grounded reward signs + no-hack, discounted returns) and
the research headline: the RL drive beats the supervised drive on a **whole-dialogue** outcome
(goal-completion) that the single-turn teacher cannot capture — and the coupling-OFF ablation
ties, proving the gain is sequential.
"""

from __future__ import annotations

import random

import torch

from nsm_ct.mind import drive, drive_env
from nsm_ct.mind import drive_rollout as R
from nsm_ct.mind.drive import ANSWER, ASK, QUIET, VOLUNTEER
from nsm_ct.mind.drive_subconscious import DriveSelfTrain


def _m15(seed: int, n: int = 2000, iters: int = 200) -> drive.DrivePolicy:
    torch.manual_seed(seed)
    b = drive_env.build_batch(drive_env.generate(n, seed=seed))
    p = drive.DrivePolicy()
    opt = torch.optim.AdamW(p.parameters(), lr=5e-3)
    for _ in range(iters):
        opt.zero_grad(); drive.drive_loss(p(b["feats"]), b["gold"], b["mask"]).backward(); opt.step()
    return p


def _focus_oracle(ctx):
    act = VOLUNTEER if ctx.derivable else ASK
    hold = ANSWER if ctx.derivable else QUIET
    return act if ctx.focus >= 2 else hold


# -- warm-start + mask-safe RL primitives ----------------------------------------
def test_value_head_warm_start_is_bit_identical():
    """An M15 checkpoint warm-starts a value-headed policy; ``forward`` (the action logits)
    is bit-for-bit the M15 policy — the trunk+act_head transferred, value head fresh."""
    m15 = drive.DrivePolicy()
    m16 = drive.load_m15_into(drive.DrivePolicy(value_head=True), m15.state_dict())
    x = torch.rand(16, drive.N_FEATURES)
    assert torch.equal(m15(x), m16(x))
    assert m16.val_head is not None


def test_sample_action_feasible_and_finite():
    """Sampling never picks a masked action (zero probability), and log-prob/entropy are
    finite even with ``-inf`` masked logits (the NaN-safety the loss depends on)."""
    pol = drive.DrivePolicy(value_head=True)
    ctx = drive.DriveContext(False, True, 0, 2, 0, 0, 0, 0)        # blocked ⇒ only ASK/QUIET
    f, m = drive.features_vec(ctx), drive.feasible_mask(ctx)
    seen = set()
    for _ in range(200):
        a, lp, ent = drive.sample_action(pol, f, m, temperature=1.5)
        seen.add(a); assert torch.isfinite(lp) and torch.isfinite(ent)
    assert seen <= {ASK, QUIET}


def test_rl_loss_has_finite_gradients():
    """REINFORCE loss + backward produce finite gradients across mixed feasibility — the
    regression guard for the masked-entropy/KL ``0·-inf`` NaN."""
    pol = drive.DrivePolicy(value_head=True)
    ref = drive.DrivePolicy()
    ctxs = [drive.DriveContext(True, False, 2, 1, 1, 0, 0, 4),
            drive.DriveContext(False, True, 0, 1, 0, 3, 0, 0)]
    feats = torch.stack([drive.features_vec(c) for c in ctxs])
    mask = torch.stack([drive.feasible_mask(c) for c in ctxs])
    actions = torch.tensor([VOLUNTEER, QUIET])
    returns = torch.tensor([2.0, -0.5])
    out = drive.drive_rl_loss(pol, feats, mask, actions, returns, w_entropy=0.05,
                              ref_policy=ref, w_anchor=0.2)
    out["total"].backward()
    assert torch.isfinite(out["total"])
    assert all(torch.isfinite(p.grad).all() for p in pol.parameters() if p.grad is not None)


# -- grounded world + reward signs (the symbolic floor) --------------------------
def test_goal_world_is_grounded():
    rng = random.Random(0)
    for _ in range(50):
        w = R.sample_goal_world(rng)
        assert w.goal_fact == w.chain[-1] and len(w.chain) >= 1
        for i, lit in enumerate(w.chain):                          # ordered r1,r2,… depths 1..D
            assert lit[1] == f"r{i+1}" and lit[3] == "+"


def _sim():
    return R.UserSimulator(R.sample_goal_world(random.Random(1)), random.Random(1), R.RewardCfg())


def test_reward_on_path_advances_off_path_penalised():
    """On a goal turn the on-path action advances the goal (+); holding back is a miss (−).
    On a distractor turn initiative is wasted (− and costs patience); restraint is rewarded."""
    cfg = R.RewardCfg()
    s = _sim(); r = s.step(VOLUNTEER, {"on_path": VOLUNTEER, "overload": False})
    assert r == cfg.r_progress and s.delivered == 1
    s = _sim(); assert s.step(ANSWER, {"on_path": VOLUNTEER, "overload": False}) == -cfg.r_missed

    s = _sim(); p0 = s.patience
    r = s.step(VOLUNTEER, {"on_path": None, "overload": False})    # off-path volunteer (distractor)
    assert r < 0 and s.patience < p0
    s = _sim(); assert s.step(QUIET, {"on_path": None, "overload": False}) == cfg.r_quiet_ok


def test_overloaded_ask_makes_no_progress():
    """Backlog gating: an on-path ask while overloaded stalls (no delivery) — over-asking now
    blocks the real goal later."""
    s = _sim(); s.pending = R.FOCUS_CAPACITY + 2
    d0 = s.delivered
    s.step(ASK, {"on_path": ASK, "overload": True})
    assert s.delivered == d0


def test_no_hack_indiscriminate_initiative_is_net_negative():
    """A policy that always takes initiative — including through distractors — burns patience
    and earns net-negative return on distractor-heavy worlds (the yap/nag penalty defeats
    farming initiative; ``TurnOutcome`` kinds are telemetry, not reward)."""
    cfg = R.RewardCfg(distractor_rate=1.0)                          # all distractors: no real goal turns
    m = R.rollout_metrics(R.baseline_act_fn("always"), n=1000, seed=5, cfg=cfg)
    assert m["mean_return"] < 0.0 and m["goal_rate"] == 0.0


# -- discounted returns + terminal goal bonus ------------------------------------
def test_returns_discounting_and_terminal_bonus():
    """Returns-to-go satisfy the discount recurrence, and reaching the goal (terminal bonus)
    yields a strictly higher return than a policy that never advances on the same worlds."""
    cfg = R.RewardCfg()
    rng = random.Random(2)
    world = R.sample_goal_world(rng)
    ep = R.rollout(_focus_oracle, world, random.Random(3), cfg)
    for t in range(len(ep.returns) - 1):                           # G_t = r_t + γ G_{t+1}
        assert abs(ep.returns[t] - (ep.transitions[t].reward + cfg.gamma * ep.returns[t + 1])) < 1e-5
    reached = R.rollout_metrics(_focus_oracle, n=800, seed=8, cfg=cfg)
    idle = R.rollout_metrics(R.baseline_act_fn("never"), n=800, seed=8, cfg=cfg)
    assert reached["goal_rate"] > 0.8 and reached["mean_return"] > idle["mean_return"]


# -- the research headline: RL beats supervised on a SEQUENTIAL outcome -----------
def test_rl_beats_supervised_sequential_and_ablation_ties():
    """RL-finetuning the supervised drive on dialogue consequences strictly raises the
    whole-dialogue goal-completion rate (coupling ON), while the coupling-OFF ablation ties —
    the proof the gain is sequential, not a re-tuned single-turn proxy."""
    m15 = _m15(0)
    rl = drive.load_m15_into(drive.DrivePolicy(value_head=True), m15.state_dict())
    DriveSelfTrain(rl, seed=0, total_rounds=18, episodes=150).run(18, verbose=False)

    on = R.RewardCfg(); off = R.RewardCfg(distractor_rate=0.0)
    rl_on = R.rollout_metrics(R.greedy_act_fn(rl), n=1500, seed=999, cfg=on)
    m15_on = R.rollout_metrics(R.greedy_act_fn(m15), n=1500, seed=999, cfg=on)
    rl_off = R.rollout_metrics(R.greedy_act_fn(rl), n=1500, seed=999, cfg=off)
    m15_off = R.rollout_metrics(R.greedy_act_fn(m15), n=1500, seed=999, cfg=off)

    assert rl_on["goal_rate"] > m15_on["goal_rate"] + 0.2          # decisive sequential win
    assert rl_on["mean_return"] > m15_on["mean_return"]
    assert abs(rl_off["goal_rate"] - m15_off["goal_rate"]) < 0.05  # ablation ties ⇒ gain is sequential
