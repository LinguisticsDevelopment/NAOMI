"""Scaling-curve driver for the candidate-lattice encoder
(dev/ENCODER_MODEL_SPEC.md): is encoder quality DATA-limited or
CAPACITY/exposure-bias-limited?

Trains scripts/train_encoder.py's exact model/loss (`nsm_ct.encoder_model`,
called verbatim -- no reimplemented loss or metric math) at several
`--n-train` sizes, WITH scheduled sampling on by default (see
`nsm_ct.encoder_model.teacher_force_loss`'s `p_ss` / DIAGNOSIS 2026-09-06:
teacher-forced next-action accuracy was measured at 0.888 while free-running
beam-decode edge-F1 was only ~0.70 -- a ~0.19 exposure-bias gap from
training only ever on gold prefixes). For each size, reports:

  - free-running beam-decode edge_precision / edge_recall / overgen_ratio
    (nsm_ct.encoder_model.evaluate, policy="model") on the held-out test split
  - teacher-forced next-action accuracy (nsm_ct.encoder_model
    .teacher_forced_accuracy_corpus, i.e. scripts/eval_teacher_forced.py's
    metric) on the SAME test split

printed as one edge-F1 & TF-acc vs n table. A rising edge-F1 with n says the
ceiling is data-limited; a big jump in edge-F1 from the scheduled-sampling
run vs. the pure-teacher-forcing 0.70 baseline says the encoder had learned
the right thing and was only exposure-bias-limited at decode time.

Uses the SAME seeded, forest-width-stratified split machinery as
scripts/train_encoder.py / scripts/rescore_encoder.py (`stratified_split`),
so the n_train records at each grid point are the same prefix of records
regardless of `--n-list` ordering (larger n_train supersets a smaller one's
train split at the same seed -- see `stratified_split`'s round-robin merge).

CPU-only (see `resolve_device`, imported verbatim from
scripts/colab_train_encoder.py: nsm_ct.encoder_model's controller builds
plain CPU index tensors at every step, so CUDA is never actually exploited
by this model regardless of what `--device` requests).

Usage:
    python scripts/scaling_curve.py
    # tiny smoke (must exit 0, table need not be meaningful):
    python scripts/scaling_curve.py --n-list 40 80 --n-dev 8 --n-test 8 \
        --epochs 2 --sched-samp-max 0.25 --device cpu --out /tmp/scaling_smoke.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from train_encoder import load_gold, stratified_split  # noqa: E402
from colab_train_encoder import resolve_device  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def train_one(records, usvs, pos_vocab, role_vocab, d_axes, *, n_train, n_dev, n_test,
              epochs, sched_samp_max, sched_samp_warmup_epochs, terminal_weight,
              d_model, hash_buckets, batch_size, lr, max_seconds, beam_width, k,
              seed, device, log) -> dict:
    """Train + evaluate one (n_train,) scaling-grid cell. Calls
    `em.teacher_force_loss` / `em.evaluate` / `em.teacher_forced_accuracy_corpus`
    verbatim -- this function only owns the loop/split bookkeeping (same
    shape as scripts/train_encoder.py's main() and scripts/colab_train_encoder
    .py's `train()`)."""
    torch.manual_seed(seed)
    random.seed(seed)

    train_recs, dev_recs, test_recs = stratified_split(records, seed, n_train, n_dev, n_test)
    log(f"n_train={n_train}: split train={len(train_recs)} dev={len(dev_recs)} test={len(test_recs)}")

    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=d_axes, hash_buckets=hash_buckets,
                             d_model=d_model, controller_hidden=d_model)
    model.to(device)

    train_items = []
    for r in train_recs:
        feats = em.build_features(r, usvs, pos_vocab, hash_buckets)
        for tree in r["lattice"]["trees"]:
            steps = em.linearize_tree(r, tree)
            train_items.append((feats, steps))
    log(f"n_train={n_train}: {len(train_items)} teacher-forced derivations")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_start = time.time()
    stopped_early = False
    for epoch in range(epochs):
        if time.time() - train_start > max_seconds:
            stopped_early = True
            log(f"n_train={n_train}: max-seconds budget hit before epoch {epoch}; stopping")
            break
        p_ss = (sched_samp_max * min(1.0, epoch / max(1, sched_samp_warmup_epochs))
                if sched_samp_max > 0.0 else 0.0)
        random.shuffle(train_items)
        opt.zero_grad()
        epoch_loss, epoch_n = 0.0, 0
        for idx, (feats, steps) in enumerate(train_items):
            if time.time() - train_start > max_seconds:
                stopped_early = True
                break
            loss = em.teacher_force_loss(model, feats, steps, terminal_weight=terminal_weight,
                                          p_ss=p_ss) / batch_size
            loss.backward()
            epoch_loss += float(loss.item()) * batch_size
            epoch_n += 1
            if (idx + 1) % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad()
        log(f"n_train={n_train}: epoch {epoch+1}/{epochs} done avg_loss={epoch_loss/max(epoch_n,1):.3f} "
            f"p_ss={p_ss:.3f}")
        if stopped_early:
            break
    train_wall = time.time() - train_start

    model.eval()
    test_edge = em.evaluate(model, test_recs, usvs, pos_vocab, hash_buckets,
                             beam_width=beam_width, k=k, policy="model")
    tf_acc = em.teacher_forced_accuracy_corpus(model, test_recs, usvs, pos_vocab, hash_buckets)
    log(f"n_train={n_train}: test edge_precision={test_edge['edge_precision']:.3f} "
        f"edge_recall={test_edge['edge_recall']:.3f} overgen={test_edge['overgen_ratio']:.3f} "
        f"tf_acc={tf_acc:.3f} (train_wall={train_wall:.1f}s, stopped_early={stopped_early})")

    return {"n_train": len(train_recs), "n_dev": len(dev_recs), "n_test": len(test_recs),
            "epochs": epoch + 1, "stopped_early": stopped_early, "train_wallclock_s": train_wall,
            "test_edge_precision": test_edge["edge_precision"], "test_edge_recall": test_edge["edge_recall"],
            "test_overgen_ratio": test_edge["overgen_ratio"], "test_edge_f1": _f1(test_edge),
            "teacher_forced_accuracy": tf_acc}


def _f1(m: dict) -> float:
    p, r = m["edge_precision"], m["edge_recall"]
    if not (p == p and r == r) or (p + r) == 0.0:  # NaN-safe (NaN != NaN)
        return float("nan")
    return 2 * p * r / (p + r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-list", type=int, nargs="+", default=[200, 400, 788],
                     help="n_train values to sweep (default: the scaling-curve grid)")
    ap.add_argument("--n-dev", type=int, default=98)
    ap.add_argument("--n-test", type=int, default=98)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--sched-samp-max", type=float, default=0.25,
                     help="scheduled sampling on by default for this driver (0.0 to disable, "
                          "reproducing pure teacher forcing); see em.teacher_force_loss")
    ap.add_argument("--sched-samp-warmup-epochs", type=int, default=10)
    ap.add_argument("--terminal-weight", type=float, default=4.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--hash-buckets", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-seconds", type=float, default=16000.0,
                     help="hard per-size training-time cutoff. Default 16000s (~4.4h) "
                          "so every size in --n-list reaches --epochs before the cutoff "
                          "(n=788 @ 100 epochs ~= 3.7h at ~130s/epoch); a smaller cap "
                          "confounds the scaling curve by giving larger n fewer epochs.")
    ap.add_argument("--beam-width", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--gold", default=str(ROOT / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(ROOT / "data" / "usvs"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "scaling_curve_report.json"))
    args = ap.parse_args()

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

    device = resolve_device(args.device, log)

    records = load_gold(args.gold)
    log(f"{len(records)} gold records available")
    max_needed = max(args.n_list) + args.n_dev + args.n_test
    if max_needed > len(records):
        sys.exit(f"ERROR: largest cell needs {max_needed} records "
                  f"(n_train={max(args.n_list)}+n_dev={args.n_dev}+n_test={args.n_test}), "
                  f"only {len(records)} available")

    usvs = load_usvs(args.usvs_dir)
    d_axes = len(usvs.axes)
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    log(f"USVS loaded (d_axes={d_axes}); pos_vocab={len(pos_vocab)} role_vocab={len(role_vocab)}")

    results = []
    for n_train in args.n_list:
        r = train_one(records, usvs, pos_vocab, role_vocab, d_axes, n_train=n_train,
                       n_dev=args.n_dev, n_test=args.n_test, epochs=args.epochs,
                       sched_samp_max=args.sched_samp_max,
                       sched_samp_warmup_epochs=args.sched_samp_warmup_epochs,
                       terminal_weight=args.terminal_weight, d_model=args.d_model,
                       hash_buckets=args.hash_buckets, batch_size=args.batch_size, lr=args.lr,
                       max_seconds=args.max_seconds, beam_width=args.beam_width, k=args.k,
                       seed=args.seed, device=device, log=log)
        results.append(r)

    print()
    print("=" * 78)
    print(f"SCALING TABLE (sched_samp_max={args.sched_samp_max}, epochs={args.epochs})")
    print("=" * 78)
    print(f"{'n_train':>8s} {'edge_P':>8s} {'edge_R':>8s} {'edge_F1':>8s} {'overgen':>8s} "
          f"{'tf_acc':>8s} {'minutes':>8s}")
    for r in results:
        print(f"{r['n_train']:8d} {r['test_edge_precision']:8.3f} {r['test_edge_recall']:8.3f} "
              f"{r['test_edge_f1']:8.3f} {r['test_overgen_ratio']:8.3f} "
              f"{r['teacher_forced_accuracy']:8.3f} {r['train_wallclock_s']/60:8.1f}")
    print()
    print("read: edge-F1 rising with n_train => data-limited, more gold helps.")
    print("      edge-F1 here vs. the pure-teacher-forcing ~0.70 baseline at n=788 => "
          "how much of the exposure-bias gap scheduled sampling recovered.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    log(f"saved report -> {args.out}")
    log(f"TOTAL wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
