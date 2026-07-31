"""Train the learned drive (M15, L6) — calibrated initiative.

The drive is a small policy that decides, per turn, whether to answer-terse / volunteer /
ask / stay-quiet. It is trained by **supervised imitation** of the synthetic-user
environment (:mod:`nsm_ct.mind.drive_env`), whose gold action is set by a *latent
usefulness the policy never sees* — so this is genuine learned calibration, in the grain of
the codebase (CE over a discrete head, like M12's routing; no RL).

The headline gate is **calibrated initiative**: on held-out decision points the learned
drive's VOLUNTEER and ASK precision/recall vs ground-truth usefulness must beat BOTH the
always-on (yappy) and never (Siri) baselines — the quantified "line between the two".

Run:
    python scripts/train_drive.py --train 4000 --val 2000 --epochs 300
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.mind import drive, drive_env  # noqa: E402


def _eval(policy, examples):
    """Held-out predictions (masked argmax) → calibration metrics."""
    preds = [drive.predict_action(policy, drive.features_vec(c), drive.feasible_mask(c))
             for c, _ in examples]
    return drive_env.initiative_metrics(examples, preds), preds


def _baseline(examples, kind):
    preds = [drive_env.baseline_action(c, kind) for c, _ in examples]
    return drive_env.initiative_metrics(examples, preds)


def _row(name, m):
    return (f"{name:<22} acc={m['acc']:.3f} | volunteer P/R/F1="
            f"{m['volunteer_p']:.2f}/{m['volunteer_r']:.2f}/{m['volunteer_f1']:.2f}"
            f" | ask P/R/F1={m['ask_p']:.2f}/{m['ask_r']:.2f}/{m['ask_f1']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    train = drive_env.generate(args.train, seed=args.seed)
    val = drive_env.generate(args.val, seed=args.seed + 1)
    tb = drive_env.build_batch(train)

    policy = drive.DrivePolicy(hidden=args.hidden)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        policy.train()
        logits = policy(tb["feats"])
        loss = drive.drive_loss(logits, tb["gold"], tb["mask"])
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % max(args.epochs // 5, 1) == 0:
            m, _ = _eval(policy, val)
            print(f"epoch {epoch+1:>4} | loss={float(loss.detach()):.3f} | val acc={m['acc']:.3f}"
                  f" vol_f1={m['volunteer_f1']:.2f} ask_f1={m['ask_f1']:.2f}", flush=True)

    learned, _ = _eval(policy, val)
    print("\n=== held-out calibrated initiative (the gate) ===")
    print(_row("learned drive (L6)", learned))
    print(_row("always-volunteer (yappy)", _baseline(val, "always")))
    print(_row("never (old Siri)", _baseline(val, "never")))
    gate = (learned["volunteer_f1"] > _baseline(val, "always")["volunteer_f1"]
            and learned["volunteer_f1"] > _baseline(val, "never")["volunteer_f1"]
            and learned["ask_f1"] > _baseline(val, "always")["ask_f1"]
            and learned["ask_f1"] > _baseline(val, "never")["ask_f1"])
    print(f"\nGATE (learned beats both yappy and Siri on volunteer & ask F1): "
          f"{'PASS' if gate else 'FAIL'}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save(policy.state_dict(), args.save)
        print(f"saved drive policy -> {args.save}")


if __name__ == "__main__":
    main()
