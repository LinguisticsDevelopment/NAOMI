"""Evaluate the reasoning loop on a held-out episode set.

Run:
    python scripts/eval.py [--config configs/default.yaml]
                           [--source curriculum|babi] [--ckpt runs/phase1.pt]

Reports answer accuracy and action-gating accuracy. Episodes are regenerated
deterministically (same seed/source as training) so the held-out split matches.
If the checkpoint is missing, evaluates an untrained model so the script still
runs end-to-end.
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
from nsm_ct.metrics import action_accuracy, answer_accuracy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the reasoning loop")
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--ckpt", default="runs/phase1.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.data.source = args.source
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source = make_source(
        cfg.data.source, seed=cfg.data.seed, max_level=cfg.data.max_level,
        babi_task=cfg.data.babi_task, babi_path=cfg.data.babi_path,
    )
    episodes = source.generate(cfg.data.num_episodes)
    _, val_eps = split_episodes(episodes, cfg.data.val_fraction, seed=cfg.data.seed)
    stack = build_default_stack(cfg, episodes)

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        try:
            stack.mind.load_state_dict(ckpt["model_state"])
            print(f"Loaded checkpoint {args.ckpt}")
        except Exception as exc:
            print(f"Checkpoint incompatible ({exc}); evaluating an untrained model.")
    else:
        print(f"No checkpoint at {args.ckpt}; evaluating an untrained model.")

    mind = stack.mind.to(device)
    mind.eval()

    val_ds = EpisodeDataset(val_eps, stack.encoder, stack.tokenizer, stack.answer_vocab, cfg)
    loader = make_dataloader(val_ds, stack.tokenizer.pad_id, cfg.train.batch_size, shuffle=False)

    ans, act, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = mind(batch)
            ans += answer_accuracy(out, batch)
            act += action_accuracy(out, batch)
            n += 1

    chance = 1.0 / (val_eps[0].options and len(val_eps[0].options) or len(stack.answer_vocab)) if val_eps else 0.0
    print(
        f"Held-out: answer_acc={ans / max(n,1):.3f} (chance≈{chance:.3f}) "
        f"action_acc={act / max(n,1):.3f}  (n={len(val_eps)} episodes)"
    )


if __name__ == "__main__":
    main()
