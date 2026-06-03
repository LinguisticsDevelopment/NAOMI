"""Evaluate a trained Consciousness Transformer on a held-out toy set.

Run:
    python scripts/eval.py [--config configs/default.yaml] [--ckpt runs/phase1.pt]

Reports multiple-choice accuracy. With the mock semantic stack and a tiny model
on toy data, expect a (probably terrible) number near chance — that is fine; the
point is a working eval pathway, not a good score.

If no checkpoint is given (or it is missing), evaluates a randomly-initialized
model so the script still runs end-to-end.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct import build_default_stack, load_config  # noqa: E402
from nsm_ct.dataset import (  # noqa: E402
    ComprehensionDataset,
    generate_toy_dataset,
    make_dataloader,
    split_examples,
)
from nsm_ct.model import ConsciousnessTransformer  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402


def evaluate(model: ConsciousnessTransformer, loader, device: torch.device) -> float:
    """Return multiple-choice accuracy over a dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model.predict(batch)
            correct += int((pred == batch.answer_idx).sum())
            total += int(batch.answer_idx.shape[0])
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the toy held-out set")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--ckpt", default="runs/phase1.pt", help="Checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples = generate_toy_dataset(cfg.data.num_examples, seed=cfg.data.seed)
    _, val_examples = split_examples(examples, cfg.data.val_fraction, seed=cfg.data.seed)
    tokenizer, feature_builder = build_default_stack(cfg, examples)

    # Prefer the checkpoint's tokenizer/weights if available.
    model = ConsciousnessTransformer(tokenizer.vocab_size, cfg.model)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        tokenizer = SimpleTokenizer(ckpt["token_to_id"])
        feature_builder.tok = tokenizer
        model = ConsciousnessTransformer(tokenizer.vocab_size, cfg.model)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint {args.ckpt}")
    else:
        print(f"No checkpoint at {args.ckpt}; evaluating an untrained model.")
    model.to(device)

    val_ds = ComprehensionDataset(val_examples, feature_builder)
    val_loader = make_dataloader(
        val_ds, pad_id=tokenizer.pad_id, batch_size=cfg.train.batch_size, shuffle=False
    )

    acc = evaluate(model, val_loader, device)
    print(f"Held-out toy accuracy: {acc:.3f}  (random baseline = 0.250, n={len(val_examples)})")


if __name__ == "__main__":
    main()
