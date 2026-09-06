"""Re-score an existing (already-trained) encoder checkpoint with the NEW
edge-precision / over-generation metrics (dev/RESEARCH_NOTES.md DIAGNOSIS
2026-09-06 entry: "encoder structure-0.000 is REAL; 93/96 sense/slot was a
MIRAGE"). NO retraining here -- this only re-decodes the checkpoint's own
reproduced held-out test split (seed + split sizes are stored in the
checkpoint's own `config`) and re-scores it with
`nsm_ct.encoder_model.evaluate`'s new fields, for policy=model, random, and
dump side by side.

This is the headline diagnostic: does the trained model beat the "dump
everything" cheat baseline on PRECISION (not just recall, which dump wins
trivially by construction)?

Usage:
    python scripts/rescore_encoder.py --checkpoint runs/encoder_colab.pt \
        | tee runs/rescore_output.txt
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
from train_encoder import load_gold, stratified_split  # noqa: E402


def fmt(m: dict) -> str:
    return (f"sense_recall={m['sense_recall']:.3f}  slot_recall={m['slot_recall']:.3f}  "
            f"structure_recall={m['structure_recall']:.3f}\n"
            f"      edge_precision={m['edge_precision']:.3f}  edge_recall={m['edge_recall']:.3f}  "
            f"overgen_ratio={m['overgen_ratio']:.3f}  (n={m['n_records']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_colab.pt"))
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "usvs"))
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--split", choices=("train", "dev", "test"), default="test")
    args = ap.parse_args()

    print(f"loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    pos_vocab = ckpt["pos_vocab"]
    role_vocab = ckpt["role_vocab"]

    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=ckpt["d_axes"],
                             hash_buckets=ckpt["hash_buckets"], d_model=ckpt["d_model"],
                             controller_hidden=ckpt["d_model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"policy params: {model.num_policy_params():,}")
    print(f"checkpoint config: {cfg}")
    old_metrics = ckpt.get("metrics", {})
    if old_metrics:
        print(f"checkpoint's OLD reported metrics (recall-only, no precision gate): "
              f"{json.dumps(old_metrics, default=str)}")

    records = load_gold(args.gold)
    train_recs, dev_recs, test_recs = stratified_split(records, cfg["seed"], cfg["n_train"],
                                                         cfg["n_dev"], cfg["n_test"])
    splits = {"train": train_recs, "dev": dev_recs, "test": test_recs}
    recs = splits[args.split]
    print(f"reproduced split: train={len(train_recs)} dev={len(dev_recs)} test={len(test_recs)} (seed={cfg['seed']})")
    print(f"scoring split={args.split} (n={len(recs)})")

    usvs = load_usvs(args.usvs_dir)
    rng = random.Random(cfg["seed"])

    results = {}
    for policy in ("model", "random", "dump"):
        results[policy] = em.evaluate(model, recs, usvs, pos_vocab, ckpt["hash_buckets"],
                                       beam_width=args.beam_width, k=args.k, policy=policy,
                                       rng=rng if policy == "random" else None)

    print("\n" + "=" * 78)
    print("RE-SCORE: model vs random vs dump (NEW precision + overgen metrics)")
    print("=" * 78)
    for policy in ("model", "random", "dump"):
        print(f"\n[{policy.upper()}]")
        print("    " + fmt(results[policy]))

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    mp, mo = results["model"]["edge_precision"], results["model"]["overgen_ratio"]
    dp, do = results["dump"]["edge_precision"], results["dump"]["overgen_ratio"]
    dr = results["dump"]["sense_recall"]
    mr = results["model"]["sense_recall"]
    print(f"  MODEL edge_precision = {mp:.3f}    MODEL overgen_ratio = {mo:.3f}")
    print(f"  DUMP  edge_precision = {dp:.3f}    DUMP  overgen_ratio = {do:.3f}")
    print(f"  MODEL sense_recall   = {mr:.3f}    DUMP  sense_recall  = {dr:.3f}")
    print("  (the old headline claim was model sense/slot recall alone; if MODEL's")
    print("   overgen_ratio is close to DUMP's and MODEL's edge_precision is low,")
    print("   that recall was inflated by over-generation, not real structure.)")


if __name__ == "__main__":
    main()
