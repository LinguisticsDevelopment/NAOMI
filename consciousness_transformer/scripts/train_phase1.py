"""Phase 1 training: fit the Consciousness Transformer on toy comprehension data.

Run:
    python scripts/train_phase1.py [--config configs/default.yaml] [--out runs/phase1.pt]

This trains the tiny model on the synthetic toy dataset and saves a checkpoint
(plus the tokenizer vocabulary) so ``eval.py`` can load it. Everything upstream
of the transformer (parser, semantic mapper, memory) is mocked — see the README.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Allow running as a plain script without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct import build_default_stack, load_config  # noqa: E402
from nsm_ct.dataset import (  # noqa: E402
    ComprehensionDataset,
    generate_toy_dataset,
    make_dataloader,
    split_examples,
)
from nsm_ct.losses import compute_losses  # noqa: E402
from nsm_ct.model import ConsciousnessTransformer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 training")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--out", default="runs/phase1.pt", help="Checkpoint output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data + mock NLP stack ---------------------------------------------
    examples = generate_toy_dataset(cfg.data.num_examples, seed=cfg.data.seed)
    train_examples, val_examples = split_examples(examples, cfg.data.val_fraction, seed=cfg.data.seed)
    tokenizer, feature_builder = build_default_stack(cfg, examples)

    train_ds = ComprehensionDataset(train_examples, feature_builder)
    train_loader = make_dataloader(
        train_ds, pad_id=tokenizer.pad_id, batch_size=cfg.train.batch_size, shuffle=True
    )

    # --- model -------------------------------------------------------------
    model = ConsciousnessTransformer(tokenizer.vocab_size, cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    print(
        f"Phase {cfg.curriculum_phase} | vocab={tokenizer.vocab_size} "
        f"| train={len(train_examples)} val={len(val_examples)} "
        f"| params={sum(p.numel() for p in model.parameters()):,}"
    )

    # --- training loop -----------------------------------------------------
    model.train()
    for epoch in range(cfg.train.epochs):
        running = {"total": 0.0, "lm": 0.0, "answer": 0.0, "consistency": 0.0}
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            output = model(batch)
            losses = compute_losses(
                output,
                batch,
                weight_lm=cfg.train.weight_lm,
                weight_answer=cfg.train.weight_answer,
                weight_consistency=cfg.train.weight_consistency,
            )
            optimizer.zero_grad()
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            running["total"] += float(losses.total.detach())
            running["lm"] += float(losses.lm.detach())
            running["answer"] += float(losses.answer.detach())
            running["consistency"] += float(losses.consistency.detach())
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        print(
            f"epoch {epoch + 1}/{cfg.train.epochs} "
            f"| total={avg['total']:.4f} lm={avg['lm']:.4f} "
            f"answer={avg['answer']:.4f} consistency={avg['consistency']:.4f}"
        )

    # --- save --------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "token_to_id": tokenizer.token_to_id,
            "config_path": args.config,
        },
        args.out,
    )
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
