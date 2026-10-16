"""Standalone candidate-set-recall eval for a trained encoder checkpoint
(dev/ENCODER_MODEL_SPEC.md S6 / dev/ENCODER_IO_CONTRACT_V2.md S7).

Recomputes the SAME held-out split the checkpoint's training run used (same
seed + split sizes, so dev/test are exactly the records the model never
trained on), decodes each record's candidate forest by beam search, and
reports sense / structure / slot recall plus the all-gold-recalled record
rate -- against both the trained policy and a random-legal-action baseline.

Usage:
    python scripts/eval_encoder.py --checkpoint runs/encoder_smoke.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from nsm_ct import encoder_train_util as etu
from train_encoder import load_gold, stratified_split  # noqa: E402
from cheat_encoder_baseline import evaluate_heuristic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "usvs"))
    ap.add_argument("--beam-width", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--split", choices=("train", "dev", "test", "all"), default="test")
    ap.add_argument("--commit-margin", type=float, default=0.0,
                     help="confidence-gated decode (policy=model only); 0.0 = off (original "
                          "always-branch-top-3 decode)")
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
    rng = random.Random(cfg["seed"])

    targets = splits if args.split == "all" else {args.split: splits[args.split]}
    for name, recs in targets.items():
        model_metrics = em.evaluate(model, recs, usvs, pos_vocab, ckpt["hash_buckets"],
                                     beam_width=args.beam_width, k=args.k, policy="model",
                                     commit_margin=args.commit_margin)
        random_metrics = em.evaluate(model, recs, usvs, pos_vocab, ckpt["hash_buckets"],
                                      beam_width=args.beam_width, k=args.k, policy="random", rng=rng)
        # dev/AUDIT_2026-10-15.md / dev/K12_BENCHMARK_FIX.md: the POS-heuristic
        # cheat baseline, by default, next to every encoder number (CLAUDE.md's
        # "every capability curriculum ships with its cheat-baseline at floor").
        heuristic_metrics = evaluate_heuristic(recs)
        heuristic_metrics["edge_f1"] = etu._f1(heuristic_metrics["edge_precision"],
                                                heuristic_metrics["edge_recall"])
        print(f"=== {name} (n={len(recs)}) ===")
        print("  model    :", json.dumps(model_metrics, indent=2))
        print("  random   :", json.dumps(random_metrics, indent=2))
        print("  heuristic:", json.dumps(heuristic_metrics, indent=2))


if __name__ == "__main__":
    main()
