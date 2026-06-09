"""Consistency probe: chain several questions in ONE unreset run.

Trains Psyche (the default self-controlled loop) on the reasoning curriculum,
then asks each episode three questions in a single run *without resetting* state
or memory — ``Q1`` (about A), ``Q2`` (about B), ``Q1'`` (A again). If Psyche
reasons stably, the repeat ``Q1'`` should match ``Q1`` despite the intervening
``Q2``. Reports answer consistency (Q1 vs Q1') and per-question accuracy.

Run:
    python scripts/probe_consistency.py [--config configs/default.yaml]
                                        [--episodes 64] [--steps 600]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct import build_default_stack, load_config  # noqa: E402
from nsm_ct.dataset import EpisodeDataset, make_dataloader  # noqa: E402
from nsm_ct.episode import CurriculumGenerator, chained_question_episode  # noqa: E402
from nsm_ct.losses import compute_losses  # noqa: E402
from nsm_ct.metrics import answer_accuracy, answer_consistency  # noqa: E402


def _chained_batch(stack, cfg, n: int, seed0: int = 10_000):
    eps, positions, gold, pairs = [], None, None, None
    for i in range(n):
        ep, positions, gold, pairs = chained_question_episode(seed=seed0 + i)
        eps.append(ep)
    ds = EpisodeDataset(eps, stack.encoder, stack.tokenizer, stack.answer_vocab, cfg)
    batch = next(iter(make_dataloader(ds, stack.tokenizer.pad_id, n, shuffle=False)))
    pos = torch.tensor([positions] * n, dtype=torch.long)
    gold = torch.tensor([gold] * n, dtype=torch.long)
    return batch, pos, gold, pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Chained-question consistency probe")
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.model.loop_mode = "controlled"  # the probe is about the self-controlled loop
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train on the reasoning curriculum so the model can answer "where is X ?".
    train_eps = CurriculumGenerator(max_level=3, seed=cfg.data.seed).generate(256)
    stack = build_default_stack(cfg, train_eps)
    psyche = stack.psyche.to(device)
    ds = EpisodeDataset(train_eps, stack.encoder, stack.tokenizer, stack.answer_vocab, cfg)
    loader = make_dataloader(ds, stack.tokenizer.pad_id, cfg.train.batch_size, shuffle=True)
    opt = torch.optim.AdamW(psyche.parameters(), lr=cfg.train.learning_rate)

    psyche.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            batch = batch.to(device)
            loss = compute_losses(psyche(batch), batch, cfg.train.weight_answer,
                                  cfg.train.weight_consistency).total
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(psyche.parameters(), cfg.train.grad_clip)
            opt.step()
            step += 1
            if step >= args.steps:
                break

    # Probe: three questions, one unreset run.
    psyche.eval()
    batch, positions, gold, pairs = _chained_batch(stack, cfg, args.episodes)
    batch = batch.to(device)
    answers = psyche.answer_at_positions(batch, positions.to(device)).cpu()

    consistency = answer_consistency(answers, pairs)
    per_q_acc = (answers == gold).float().mean(dim=0).tolist()
    print(
        f"chained-question probe | episodes={args.episodes} steps={args.steps}\n"
        f"  single-question train acc (level 1-3): "
        f"{answer_accuracy(psyche(batch), batch):.3f}\n"
        f"  per-question acc  Q1={per_q_acc[0]:.3f}  Q2(other)={per_q_acc[1]:.3f}  "
        f"Q1'(repeat)={per_q_acc[2]:.3f}\n"
        f"  consistency (Q1 == Q1', after the intervening Q2): {consistency:.3f}"
    )


if __name__ == "__main__":
    main()
