"""Phase 1 training: teach the reasoning loop on episodes (kindergartener regime).

Run:
    python scripts/train_phase1.py [--config configs/default.yaml]
                                   [--source curriculum|babi] [--out runs/phase1.pt]

Each episode is a context stream + a question. The model unrolls the stream
(absorbing facts into memory under its own action gating) and answers at the
question. We report answer accuracy AND action-gating accuracy (does it ABSORB
statements and RESPOND to questions?).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct import build_default_stack, load_config  # noqa: E402
from nsm_ct.dataset import EpisodeDataset, make_dataloader, split_episodes  # noqa: E402
from nsm_ct.episode import make_source  # noqa: E402
from nsm_ct.losses import compute_losses  # noqa: E402
from nsm_ct.metrics import answer_accuracy, mean_respond_position  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 training (reasoning loop)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="Override data.source (curriculum|babi)")
    ap.add_argument("--out", default="runs/phase1.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.data.source = args.source
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- episodes + stack --------------------------------------------------
    source = make_source(
        cfg.data.source, seed=cfg.data.seed, max_level=cfg.data.max_level,
        babi_task=cfg.data.babi_task, babi_path=cfg.data.babi_path,
    )
    episodes = source.generate(cfg.data.num_episodes)
    train_eps, val_eps = split_episodes(episodes, cfg.data.val_fraction, seed=cfg.data.seed)
    stack = build_default_stack(cfg, episodes)

    train_ds = EpisodeDataset(train_eps, stack.encoder, stack.tokenizer, stack.answer_vocab, cfg)
    loader = make_dataloader(train_ds, stack.tokenizer.pad_id, cfg.train.batch_size, shuffle=True)

    mind = stack.mind.to(device)
    optimizer = torch.optim.AdamW(mind.parameters(), lr=cfg.train.learning_rate)

    print(
        f"source={cfg.data.source} mode={cfg.data.answer_mode} encoder={cfg.input_encoder} "
        f"| vocab={stack.tokenizer.vocab_size} answers={len(stack.answer_vocab)} "
        f"| train={len(train_eps)} val={len(val_eps)} "
        f"| params={sum(p.numel() for p in mind.parameters()):,}"
    )

    # --- training loop -----------------------------------------------------
    mind.train()
    for epoch in range(cfg.train.epochs):
        agg = {"total": 0.0, "answer": 0.0, "novelty": 0.0, "consistency": 0.0, "ans_acc": 0.0, "resp_pos": 0.0}
        n = 0
        for batch in loader:
            batch = batch.to(device)
            out = mind(batch)
            losses = compute_losses(
                out, batch,
                weight_answer=cfg.train.weight_answer,
                weight_novelty=cfg.train.weight_novelty,
                weight_consistency=cfg.train.weight_consistency,
            )
            optimizer.zero_grad()
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(mind.parameters(), cfg.train.grad_clip)
            optimizer.step()

            agg["total"] += float(losses.total.detach())
            agg["answer"] += float(losses.answer.detach())
            agg["novelty"] += float(losses.novelty.detach())
            agg["consistency"] += float(losses.consistency.detach())
            agg["ans_acc"] += answer_accuracy(out, batch)
            agg["resp_pos"] += mean_respond_position(out, batch)
            n += 1

        avg = {k: v / max(n, 1) for k, v in agg.items()}
        print(
            f"epoch {epoch + 1}/{cfg.train.epochs} | total={avg['total']:.4f} "
            f"answer={avg['answer']:.4f} novelty={avg['novelty']:.4f} "
            f"consistency={avg['consistency']:.4f} | ans_acc={avg['ans_acc']:.3f} "
            f"resp_pos={avg['resp_pos']:.2f}"
        )

    # --- save --------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(
        {
            "model_state": mind.state_dict(),
            "token_to_id": stack.tokenizer.token_to_id,
            "answer_vocab": stack.answer_vocab,
            "config_path": args.config,
            "source": cfg.data.source,
        },
        args.out,
    )
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
