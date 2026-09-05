"""Standalone candidate-set-recall eval for a trained encoder checkpoint against
an arbitrary gold file (no split recomputation).

Unlike scripts/eval_encoder.py, this does NOT recompute the English
train/dev/test split -- it evaluates every record in --gold as-is. Used for
the cross-lingual grammar-swap acceptance test: an English-trained checkpoint
evaluated on the never-seen Spanish gold set. Reuses
nsm_ct.encoder_model.evaluate / score_record verbatim -- no metric
reimplementation.

Usage:
    python scripts/eval_encoder_on.py --checkpoint runs/encoder_full.pt \
        --gold runs/spanish_gold_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em


def load_gold(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate_with_totals(model, records, usvs, pos_vocab, hash_buckets,
                          beam_width, k, policy, rng=None):
    """Same walk as encoder_model.evaluate(), but also keeps the raw
    RecordRecall objects so we can report site counts (n) alongside the
    aggregate recall. Calls score_record/aggregate_recall verbatim."""
    scores = []
    for record in records:
        feats = em.build_features(record, usvs, pos_vocab, hash_buckets)
        forest = em.beam_decode(model, feats, beam_width=beam_width, k=k, policy=policy, rng=rng)
        scores.append(em.score_record(record, forest))
    agg = em.aggregate_recall(scores)
    agg["sense_total"] = sum(s.sense_total for s in scores)
    agg["slot_total"] = sum(s.slot_total for s in scores)
    agg["tree_total"] = sum(s.tree_total for s in scores)
    agg["sense_hits"] = sum(s.sense_hits for s in scores)
    agg["slot_hits"] = sum(s.slot_hits for s in scores)
    agg["tree_hits"] = sum(s.tree_hits for s in scores)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--usvs-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "usvs"))
    ap.add_argument("--beam-width", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
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
    usvs = load_usvs(args.usvs_dir)
    rng = random.Random(cfg["seed"])

    model_metrics = evaluate_with_totals(model, records, usvs, pos_vocab, ckpt["hash_buckets"],
                                          args.beam_width, args.k, "model")
    random_metrics = evaluate_with_totals(model, records, usvs, pos_vocab, ckpt["hash_buckets"],
                                           args.beam_width, args.k, "random", rng=rng)
    print(f"=== {Path(args.gold).name} (n={len(records)}) ===")
    print("  model :", json.dumps(model_metrics, indent=2))
    print("  random:", json.dumps(random_metrics, indent=2))


if __name__ == "__main__":
    main()
