"""Teacher-forced training of the candidate-lattice encoder (dev/ENCODER_MODEL_SPEC.md).

Trains `nsm_ct.encoder_model.EncoderModel` on `runs/encoder_gold_v2.jsonl`
with the spec S3.2 loss (action-type CE + typed-arg CE + grounding-type CE +
source CE -- there is no sense-selection term anywhere) under the S3.3
grammar-constrained action mask. Reports the S2.3 smoke wall-clock and the
S6 candidate-set recall on a held-out, forest-width-stratified split.

Usage:
    python scripts/train_encoder.py --smoke --out runs/encoder_smoke.pt
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

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em


def load_gold(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def forest_width_bucket(record: dict) -> int:
    n = len(record["lattice"]["trees"])
    if n <= 1:
        return 0
    if n <= 3:
        return 1
    return 2


def stratified_split(records: list, seed: int, n_train: int, n_dev: int, n_test: int) -> tuple:
    """Seeded, forest-width-stratified split (spec S6): buckets = {1-tree,
    2-3 trees, 4+ trees}, shuffled within bucket, then interleaved so
    train/dev/test each see all three widths, disjoint records throughout."""
    rng = random.Random(seed)
    buckets = {0: [], 1: [], 2: []}
    for r in records:
        buckets[forest_width_bucket(r)].append(r)
    for b in buckets.values():
        rng.shuffle(b)

    # round-robin merge across buckets (proportional representation)
    order = []
    idxs = {0: 0, 1: 0, 2: 0}
    total = sum(len(b) for b in buckets.values())
    while len(order) < total:
        for k in (0, 1, 2):
            if idxs[k] < len(buckets[k]):
                order.append(buckets[k][idxs[k]])
                idxs[k] += 1

    train = order[:n_train]
    dev = order[n_train:n_train + n_dev]
    test = order[n_train + n_dev:n_train + n_dev + n_test]
    return train, dev, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "usvs"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_smoke.pt"))
    ap.add_argument("--smoke", action="store_true", help="S2.3 smoke config: 150 train, d_model=64, 2^12 buckets, batch 16, 2 epochs")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-dev", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--hash-buckets", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=650.0, help="hard training-time cutoff")
    ap.add_argument("--beam-width", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    if args.smoke:
        n_train = args.n_train or 150
        n_dev = args.n_dev or 40
        n_test = args.n_test or 40
        d_model = args.d_model or 64
        hash_buckets = args.hash_buckets or 4096
        epochs = args.epochs or 2
        batch_size = args.batch_size or 16
    else:
        n_train = args.n_train or 788
        n_dev = args.n_dev or 98
        n_test = args.n_test or 98
        d_model = args.d_model or 128
        hash_buckets = args.hash_buckets or 32768
        epochs = args.epochs or 15
        batch_size = args.batch_size or 32

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] loading gold from {args.gold}")
    records = load_gold(args.gold)
    print(f"[{time.time()-t0:6.1f}s] {len(records)} gold records")

    train_recs, dev_recs, test_recs = stratified_split(records, args.seed, n_train, n_dev, n_test)
    print(f"[{time.time()-t0:6.1f}s] split: train={len(train_recs)} dev={len(dev_recs)} test={len(test_recs)}")

    print(f"[{time.time()-t0:6.1f}s] loading USVS from {args.usvs_dir}")
    usvs = load_usvs(args.usvs_dir)
    d_axes = len(usvs.axes)
    print(f"[{time.time()-t0:6.1f}s] USVS loaded: {len(usvs.sense_ids)} senses, d_axes={d_axes}")

    # Vocabs are built over the FULL gold file: POS tags and role labels are
    # closed/open LINGUISTIC label spaces (the frozen role vocabulary,
    # contract S3), not evaluation targets -- unlike every recalled
    # candidate (senses, slots, trees), which is always copied from
    # per-record retrieval and never drawn from a learned vocabulary.
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    print(f"[{time.time()-t0:6.1f}s] pos_vocab={len(pos_vocab)} role_vocab={len(role_vocab)}")

    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=d_axes, hash_buckets=hash_buckets,
                             d_model=d_model, controller_hidden=d_model)
    n_params = model.num_policy_params()
    n_bytes = n_params * 4
    print(f"[{time.time()-t0:6.1f}s] policy params: {n_params:,} (~{n_bytes/1e6:.3f} MB fp32)")

    print(f"[{time.time()-t0:6.1f}s] building features + derivations for {len(train_recs)} train records")
    train_items = []
    for r in train_recs:
        feats = em.build_features(r, usvs, pos_vocab, hash_buckets)
        for tree in r["lattice"]["trees"]:
            steps = em.linearize_tree(r, tree)
            train_items.append((feats, steps))
    print(f"[{time.time()-t0:6.1f}s] {len(train_items)} teacher-forced derivations")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_curve = []
    step_count = 0
    train_start = time.time()
    stopped_early = False
    for epoch in range(epochs):
        if time.time() - train_start > args.max_seconds:
            stopped_early = True
            print(f"[{time.time()-t0:6.1f}s] max-seconds budget hit before epoch {epoch}; stopping")
            break
        random.shuffle(train_items)
        epoch_loss = 0.0
        epoch_n = 0
        opt.zero_grad()
        for idx, (feats, steps) in enumerate(train_items):
            if time.time() - train_start > args.max_seconds:
                stopped_early = True
                break
            loss = em.teacher_force_loss(model, feats, steps) / batch_size
            loss.backward()
            epoch_loss += float(loss.item()) * batch_size
            epoch_n += 1
            step_count += 1
            if (idx + 1) % batch_size == 0:
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
        print(f"[{time.time()-t0:6.1f}s] === epoch {epoch} done: avg_loss={avg:.3f} (n={epoch_n} derivations) ===")
        if stopped_early:
            break

    train_wall = time.time() - train_start
    print(f"[{time.time()-t0:6.1f}s] training wall-clock: {train_wall:.1f}s (stopped_early={stopped_early})")

    print(f"[{time.time()-t0:6.1f}s] evaluating (model policy) train/dev/test ...")
    rng = random.Random(args.seed)
    metrics = {}
    for split_name, split_recs in (("train", train_recs), ("dev", dev_recs), ("test", test_recs)):
        m = em.evaluate(model, split_recs, usvs, pos_vocab, hash_buckets,
                         beam_width=args.beam_width, k=args.k, policy="model")
        metrics[split_name] = m
        print(f"[{time.time()-t0:6.1f}s] {split_name}: {m}")

    print(f"[{time.time()-t0:6.1f}s] evaluating RANDOM baseline on test ...")
    random_metrics = em.evaluate(model, test_recs, usvs, pos_vocab, hash_buckets,
                                  beam_width=args.beam_width, k=args.k, policy="random", rng=rng)
    print(f"[{time.time()-t0:6.1f}s] test (random baseline): {random_metrics}")

    ckpt = {
        "model_state": model.state_dict(),
        "pos_vocab": pos_vocab,
        "role_vocab": role_vocab,
        "d_axes": d_axes,
        "hash_buckets": hash_buckets,
        "d_model": d_model,
        "config": {"n_train": len(train_recs), "n_dev": len(dev_recs), "n_test": len(test_recs),
                   "epochs": epochs, "batch_size": batch_size, "seed": args.seed},
        "loss_curve": loss_curve,
        "metrics": metrics,
        "random_baseline_test": random_metrics,
        "train_wallclock_s": train_wall,
        "n_policy_params": n_params,
        "split_record_texts": {"train": [r["text"] for r in train_recs],
                                "dev": [r["text"] for r in dev_recs],
                                "test": [r["text"] for r in test_recs]},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[{time.time()-t0:6.1f}s] saved checkpoint -> {args.out}")
    print(f"[{time.time()-t0:6.1f}s] TOTAL wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
