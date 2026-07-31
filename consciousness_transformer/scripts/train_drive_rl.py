"""Train the sequential-RL drive (M16) and report the calibrated-initiative-over-dialogues gate.

The M15 drive learns from independent decision points; M16 RL-finetunes it on **multi-turn
consequences** against a grounded hidden-goal user simulator (`drive_rollout`), warm-started
from the M15 weights, reward = symbolic goal-progress (no judge), soft→discrete annealed.

Headline gate (a WHOLE-DIALOGUE outcome the single-turn teacher cannot capture): the RL drive
beats the M15 supervised drive (and always/never) on **goal-completion rate** and **mean
return** — AND the coupling-OFF ablation (no distractors) **ties**, proving the gain is
sequential, not a re-tuned proxy.

Run:
    python scripts/train_drive_rl.py --rounds 30 --episodes 200
    python scripts/train_drive_rl.py --warm runs/drive_m15.pt --rounds 30
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.mind import drive, drive_env  # noqa: E402
from nsm_ct.mind import drive_rollout as R  # noqa: E402
from nsm_ct.mind.drive_subconscious import DriveSelfTrain  # noqa: E402


def _train_m15(seed: int) -> drive.DrivePolicy:
    """The single-turn supervised baseline (the policy RL warm-starts from and must beat)."""
    torch.manual_seed(seed)
    batch = drive_env.build_batch(drive_env.generate(4000, seed=seed))
    pol = drive.DrivePolicy()
    opt = torch.optim.AdamW(pol.parameters(), lr=5e-3)
    for _ in range(300):
        opt.zero_grad()
        drive.drive_loss(pol(batch["feats"]), batch["gold"], batch["mask"]).backward()
        opt.step()
    return pol


def _row(name, m):
    return (f"{name:<10} goal_rate={m['goal_rate']:.3f}  mean_return={m['mean_return']:.2f}"
            f"  (turns_to_goal={m['mean_turns_to_goal']:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=str, default="", help="M15 checkpoint; trained inline if absent")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--w-value", type=float, default=0.3)
    ap.add_argument("--w-anchor", type=float, default=0.0)
    ap.add_argument("--eval-n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()

    m15 = drive.DrivePolicy()
    if args.warm:
        drive.load_m15_into(m15, torch.load(args.warm))
        print(f"loaded M15 warm-start <- {args.warm}")
    else:
        print("training the M15 supervised baseline inline …")
        m15 = _train_m15(args.seed)

    rl = drive.load_m15_into(drive.DrivePolicy(value_head=True), m15.state_dict())
    trainer = DriveSelfTrain(rl, lr=args.lr, total_rounds=args.rounds, gamma=args.gamma,
                             w_value=args.w_value, w_anchor0=args.w_anchor,
                             episodes=args.episodes, steps=args.steps, seed=args.seed)
    print("RL self-training (sequential rollouts) …")
    trainer.run(args.rounds, eval_every=max(args.rounds // 5, 1))

    cfg, cfg0 = R.RewardCfg(gamma=args.gamma), R.RewardCfg(gamma=args.gamma, distractor_rate=0.0)
    ev = lambda fn, c: R.rollout_metrics(fn, n=args.eval_n, seed=999, cfg=c)
    rl_on, sup_on = ev(R.greedy_act_fn(rl), cfg), ev(R.greedy_act_fn(m15), cfg)

    print("\n=== HEADLINE: calibrated initiative over whole dialogues (coupling ON) ===")
    print(_row("RL (M16)", rl_on))
    print(_row("M15 sup", sup_on))
    print(_row("always", ev(R.baseline_act_fn("always"), cfg)))
    print(_row("never", ev(R.baseline_act_fn("never"), cfg)))

    rl_off, sup_off = ev(R.greedy_act_fn(rl), cfg0), ev(R.greedy_act_fn(m15), cfg0)
    print("\n=== ABLATION: coupling OFF (no distractors) — should TIE ===")
    print(_row("RL (M16)", rl_off))
    print(_row("M15 sup", sup_off))

    headline = rl_on["goal_rate"] > sup_on["goal_rate"] and rl_on["mean_return"] > sup_on["mean_return"]
    tie = abs(rl_off["goal_rate"] - sup_off["goal_rate"]) < 0.05
    print(f"\nGATE — RL beats supervised on goal-rate & return (coupling ON): "
          f"{'PASS' if headline else 'FAIL'}")
    print(f"ABLATION — RL ≈ supervised with coupling OFF (gain is sequential): "
          f"{'PASS' if tie else 'FAIL'}")

    # Guardrail: the RL drive stays grounded/non-degenerate. Its single-turn calibration
    # *shifts* (it learned to use `focus`, which the single-turn teacher ignores) — that is
    # the expected price of the sequential gain, reported, not gated.
    val = drive_env.generate(2000, seed=7)
    for nm, pol in [("RL (M16)", rl), ("M15 sup", m15)]:
        preds = [drive.predict_action(pol, drive.features_vec(c), drive.feasible_mask(c))
                 for c, _ in val]
        im = drive_env.initiative_metrics(val, preds)
        print(f"  single-turn {nm:9} acc={im['acc']:.3f} vol_f1={im['volunteer_f1']:.2f} "
              f"ask_f1={im['ask_f1']:.2f}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(rl.state_dict(), args.save)
        print(f"saved RL drive -> {args.save}")


if __name__ == "__main__":
    main()
