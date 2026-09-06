"""Standalone teacher-forced next-action-accuracy eval for a trained encoder
checkpoint (dev/ENCODER_MODEL_SPEC.md S3.1/S3.2).

This is the exposure-bias diagnostic's OTHER half: `eval_encoder.py` reports
free-running beam-decode edge-F1 (candidate-set recall); this script reports
how often the policy picks the gold action when conditioned on the gold
PREFIX at every step of every gold derivation (never its own prior output --
see `nsm_ct.encoder_model.teacher_forced_action_accuracy`). A large gap
between the two (teacher-forced accuracy high, decode edge-F1 low) is
exactly the exposure-bias signature scheduled sampling (`--sched-samp-max`
in scripts/train_encoder.py / scripts/scaling_curve.py) is meant to close.

Recomputes the SAME held-out split the checkpoint's training run used (same
seed + split sizes), exactly like eval_encoder.py.

Usage:
    python scripts/eval_teacher_forced.py --checkpoint runs/encoder_smoke.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from train_encoder import load_gold, stratified_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "usvs"))
    ap.add_argument("--split", choices=("train", "dev", "test", "all"), default="test")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    cfg = ckpt["config"]
    pos_vocab = ckpt["pos_vocab"]
    role_vocab = ckpt["role_vocab"]

    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=ckpt["d_axes"],
                             hash_buckets=ckpt["hash_buckets"], d_model=ckpt["d_model"],
                             controller_hidden=ckpt["d_model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"policy params: {model.num_policy_params():,}")

    records = load_gold(args.gold)
    train_recs, dev_recs, test_recs = stratified_split(records, cfg["seed"], cfg["n_train"],
                                                         cfg["n_dev"], cfg["n_test"])
    splits = {"train": train_recs, "dev": dev_recs, "test": test_recs}

    usvs = load_usvs(args.usvs_dir)

    targets = splits if args.split == "all" else {args.split: splits[args.split]}
    for name, recs in targets.items():
        acc = em.teacher_forced_accuracy_corpus(model, recs, usvs, pos_vocab, ckpt["hash_buckets"])
        print(f"=== {name} (n={len(recs)}) === teacher_forced_action_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
