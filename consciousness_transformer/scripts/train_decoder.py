"""Teacher-forced training of the learned reconstruction decoder
(RESEARCH_NOTES "DECODER PLAN UPDATE", 2026-09-05).

Self-supervised reconstruction: for every `encoder_gold_v2.jsonl` record, the
committed structure is its TOP tree (`lattice.trees[0]`) and the label is the
record's own `text`/`tokens` -- no separate decoder gold. Trains
`nsm_ct.decoder_trained.DecoderTrainedModel` with `reconstruction_loss`
(teacher-forced next-token CE over the extended copy+closed-function-vocab
label space) and reports the ROUND-TRIP eval (exact_match + token_f1) on a
held-out split.

Usage:
    python scripts/train_decoder.py --records runs/encoder_gold_v2.jsonl \\
        --epochs 1 --device cpu --out runs/decoder_trained_tiny.pt

This script does not itself do a real training run -- that happens on Colab
GPU (RESEARCH_NOTES). The one thing a real run needs to set: --records
(the full corpus) and --epochs (a real budget; CPU here only smoke-tests
that the loop runs).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from nsm_ct import decoder_trained as dt
from nsm_ct import encoder_model as em


def load_records(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def split_records(records: list, seed: int, n_train: int, n_dev: int) -> tuple:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    train = shuffled[:n_train]
    dev = shuffled[n_train:n_train + n_dev]
    return train, dev


def evaluate_round_trip(model: dt.DecoderTrainedModel, records: list) -> dict:
    exact_matches = []
    token_f1s = []
    for record in records:
        tree = record["lattice"]["trees"][0]
        pred = dt.round_trip(record, tree, model)
        m = dt.reconstruction_accuracy(pred, record["text"])
        exact_matches.append(m["exact_match"])
        token_f1s.append(m["token_f1"])
    n = max(len(records), 1)
    return {
        "n": len(records),
        "exact_match": sum(exact_matches) / n,
        "token_f1": sum(token_f1s) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "runs" / "decoder_trained.pt"))
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-dev", type=int, default=None)
    ap.add_argument("--hash-buckets", type=int, default=2048)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=650.0, help="hard training-time cutoff")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] loading records from {args.records}")
    records = load_records(args.records)
    print(f"[{time.time()-t0:6.1f}s] {len(records)} records")

    n_train = args.n_train or max(1, int(0.8 * len(records)))
    n_dev = args.n_dev or max(1, len(records) - n_train)
    train_recs, dev_recs = split_records(records, args.seed, n_train, n_dev)
    print(f"[{time.time()-t0:6.1f}s] split: train={len(train_recs)} dev={len(dev_recs)}")

    # Vocabularies over the FULL corpus (closed label spaces, same discipline
    # as train_encoder.py's pos/role vocabs -- never a per-split target).
    relation_vocab = em.build_role_vocab(records)
    function_vocab = dt.build_function_vocab(records)
    print(f"[{time.time()-t0:6.1f}s] relation_vocab={len(relation_vocab)} function_vocab={len(function_vocab)}")

    model = dt.DecoderTrainedModel(relation_vocab, function_vocab,
                                    hash_buckets=args.hash_buckets, d_model=args.d_model).to(device)
    n_params = model.num_params()
    print(f"[{time.time()-t0:6.1f}s] decoder params: {n_params:,} (~{n_params*4/1e6:.3f} MB fp32)")

    print(f"[{time.time()-t0:6.1f}s] building features for {len(train_recs)} train records")
    train_feats = [
        dt.build_decoder_features(r, r["lattice"]["trees"][0], function_vocab, relation_vocab, args.hash_buckets)
        for r in train_recs
    ]
    print(f"[{time.time()-t0:6.1f}s] {len(train_feats)} reconstruction items")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_curve = []
    step_count = 0
    train_start = time.time()
    stopped_early = False
    for epoch in range(args.epochs):
        if time.time() - train_start > args.max_seconds:
            stopped_early = True
            print(f"[{time.time()-t0:6.1f}s] max-seconds budget hit before epoch {epoch}; stopping")
            break
        random.shuffle(train_feats)
        epoch_loss = 0.0
        epoch_n = 0
        opt.zero_grad()
        for idx, feats in enumerate(train_feats):
            if time.time() - train_start > args.max_seconds:
                stopped_early = True
                break
            loss = dt.reconstruction_loss(model, feats) / args.batch_size
            loss.backward()
            epoch_loss += float(loss.item()) * args.batch_size
            epoch_n += 1
            step_count += 1
            if (idx + 1) % args.batch_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
            if step_count % 50 == 0:
                avg = epoch_loss / max(epoch_n, 1)
                loss_curve.append((step_count, avg))
                print(f"[{time.time()-t0:6.1f}s] epoch {epoch} step {step_count} avg_loss={avg:.3f}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad()
        avg = epoch_loss / max(epoch_n, 1)
        loss_curve.append((step_count, avg))
        print(f"[{time.time()-t0:6.1f}s] === epoch {epoch} done: avg_loss={avg:.3f} (n={epoch_n}) ===")
        if stopped_early:
            break

    train_wall = time.time() - train_start
    print(f"[{time.time()-t0:6.1f}s] training wall-clock: {train_wall:.1f}s (stopped_early={stopped_early})")

    print(f"[{time.time()-t0:6.1f}s] round-trip eval (train sample) ...")
    train_metrics = evaluate_round_trip(model, train_recs[:min(len(train_recs), 200)])
    print(f"[{time.time()-t0:6.1f}s] train round-trip: {train_metrics}")

    print(f"[{time.time()-t0:6.1f}s] round-trip eval (held-out dev) ...")
    dev_metrics = evaluate_round_trip(model, dev_recs)
    print(f"[{time.time()-t0:6.1f}s] dev round-trip: {dev_metrics}")

    ckpt = {
        "model_state": model.state_dict(),
        "relation_vocab": relation_vocab,
        "function_vocab": function_vocab,
        "hash_buckets": args.hash_buckets,
        "d_model": args.d_model,
        "config": {"n_train": len(train_recs), "n_dev": len(dev_recs), "epochs": args.epochs,
                   "batch_size": args.batch_size, "seed": args.seed},
        "loss_curve": loss_curve,
        "train_round_trip": train_metrics,
        "dev_round_trip": dev_metrics,
        "train_wallclock_s": train_wall,
        "n_params": n_params,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[{time.time()-t0:6.1f}s] saved checkpoint -> {args.out}")
    print(f"[{time.time()-t0:6.1f}s] TOTAL wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
