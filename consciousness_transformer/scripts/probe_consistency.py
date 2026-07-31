"""Chained-question consistency: train it as a capability, then measure it.

Several questions are asked in ONE unreset run — ``Q1`` (about A), ``Q2``
(about B), ``Q1'`` (A again). Psyche's controlled loop reads out a per-question
answer through a soft pointer window (``question_logits``), so the per-question
loss (``weight_multi``) trains the model to answer each question as it arrives
and to keep its story straight.

This script trains twice on the same data — once WITHOUT the per-question loss
(consistency is incidental) and once WITH it — and reports held-out consistency
(Q1 vs the repeated Q1') and per-question accuracy for both.

Run:
    python scripts/probe_consistency.py [--config configs/default.yaml]
                                        [--steps 800] [--eval-episodes 64]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct import build_default_stack, load_config  # noqa: E402
from nsm_ct.dataset import EpisodeDataset, make_dataloader  # noqa: E402
from nsm_ct.episode import (  # noqa: E402
    CurriculumGenerator,
    chained_question_episode,
    generate_chained_episodes,
)
from nsm_ct.losses import compute_losses  # noqa: E402
from nsm_ct.metrics import answer_consistency  # noqa: E402


def _train(cfg, episodes, steps: int, weight_multi: float, device):
    torch.manual_seed(cfg.train.seed)
    stack = build_default_stack(cfg, episodes)
    psyche = stack.psyche.to(device)
    ds = EpisodeDataset(episodes, stack.encoder, stack.tokenizer, stack.answer_vocab, cfg)
    loader = make_dataloader(ds, stack.tokenizer.pad_id, cfg.train.batch_size, shuffle=True)
    opt = torch.optim.AdamW(psyche.parameters(), lr=cfg.train.learning_rate)
    psyche.train()
    step = 0
    while step < steps:
        for batch in loader:
            batch = batch.to(device)
            loss = compute_losses(
                psyche(batch), batch, cfg.train.weight_answer, cfg.train.weight_consistency,
                weight_mem_answer=cfg.train.weight_mem_answer, weight_multi=weight_multi,
            ).total
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(psyche.parameters(), cfg.train.grad_clip)
            opt.step()
            step += 1
            if step >= steps:
                break
    return stack


@torch.no_grad()
def _eval_chained(stack, cfg, n: int, device, seed0: int = 777_000):
    eps, pairs = [], None
    for i in range(n):
        ep, _pos, _gold, pairs = chained_question_episode(seed=seed0 + i)
        eps.append(ep)
    ds = EpisodeDataset(eps, stack.encoder, stack.tokenizer, stack.answer_vocab, cfg)
    batch = next(iter(make_dataloader(ds, stack.tokenizer.pad_id, n, shuffle=False))).to(device)
    stack.psyche.eval()
    preds = stack.psyche(batch)["question_logits"].argmax(-1).cpu()  # [B, Q]
    gold = batch.q_targets.cpu()
    acc = (preds == gold).float().mean(dim=0).tolist()
    return answer_consistency(preds, pairs), acc


def main() -> None:
    ap = argparse.ArgumentParser(description="Chained-question consistency (train + probe)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--curriculum-episodes", type=int, default=192)
    ap.add_argument("--chained-episodes", type=int, default=96)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.model.loop_mode = "controlled"  # the probe is about the self-controlled loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same data both times: single-question curriculum + chained multi-question streams.
    episodes = (
        CurriculumGenerator(max_level=3, seed=cfg.data.seed).generate(args.curriculum_episodes)
        + generate_chained_episodes(args.chained_episodes, seed=cfg.data.seed + 1)
    )

    print(f"training WITHOUT the per-question loss ({args.steps} steps) ...")
    stack0 = _train(cfg, episodes, args.steps, weight_multi=0.0, device=device)
    c0, a0 = _eval_chained(stack0, cfg, args.eval_episodes, device)

    print(f"training WITH the per-question loss ({args.steps} steps) ...")
    stack1 = _train(cfg, episodes, args.steps, weight_multi=1.0, device=device)
    c1, a1 = _eval_chained(stack1, cfg, args.eval_episodes, device)

    print(
        f"\nheld-out chained questions (n={args.eval_episodes}, one unreset run each):\n"
        f"  without multi-question loss: consistency={c0:.3f} | "
        f"acc Q1={a0[0]:.3f} Q2={a0[1]:.3f} Q1'={a0[2]:.3f}\n"
        f"  WITH multi-question loss:    consistency={c1:.3f} | "
        f"acc Q1={a1[0]:.3f} Q2={a1[1]:.3f} Q1'={a1[2]:.3f}"
    )


if __name__ == "__main__":
    main()
