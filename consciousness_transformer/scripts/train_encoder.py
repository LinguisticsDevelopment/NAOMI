"""Teacher-forced training of the candidate-lattice encoder (dev/ENCODER_MODEL_SPEC.md).

Trains `nsm_ct.encoder_model.EncoderModel` on `runs/encoder_gold_v2.jsonl`
with the spec S3.2 loss (action-type CE + typed-arg CE + grounding-type CE +
source CE -- there is no sense-selection term anywhere) under the S3.3
grammar-constrained action mask. Reports the S2.3 smoke wall-clock and the
S6 candidate-set recall on a held-out, forest-width-stratified split.

dev/AUDIT_2026-09-08.md finding 7 + recommendation (c): also supports an
exact optimizer-step budget (--max-steps) and a shared, gold-file-independent
held-out sentence set (--holdout-file) so scripts/run_encoder_arms.sh can run
arms that separate MORE data from CLEANER data at equal optimization budget,
scored on the SAME held-out sentences. All of this is additive: without
--max-steps/--holdout-file, behavior is unchanged from before.

Usage:
    python scripts/train_encoder.py --smoke --out runs/encoder_smoke.pt
    python scripts/train_encoder.py --smoke --max-steps 60 \
        --holdout-file runs/holdout_sentences.txt --gold runs/encoder_gold_v2.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from nsm_ct import encoder_train_util as etu

# Re-exported for scripts that do `from train_encoder import load_gold,
# stratified_split` (scripts/rescore_encoder.py, colab_train_encoder.py,
# colab_train_all.py, eval_encoder.py, diagnose_colab_ckpts.py,
# _roundtrip_aliasfix.py) -- the implementations now live in
# nsm_ct.encoder_train_util so scripts/colab_train_encoder.py doesn't
# duplicate them.
load_gold = etu.load_gold
forest_width_bucket = etu.forest_width_bucket
stratified_split = etu.stratified_split


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
    ap.add_argument("--terminal-weight", type=float, default=4.0,
                     help="up-weight the STOP/CLOSE_CLAUSE action-type CE loss by this factor "
                          "(spec fix, DIAGNOSIS 2026-09-06: rare terminal actions in the oracle "
                          "are under-trained by plain CE, so the policy never learns to stop and "
                          "over-attaches instead); 1.0 = unweighted (original loss)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=650.0, help="hard training-time cutoff")
    ap.add_argument("--beam-width", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=None,
                     help="(item A) train for exactly N optimizer steps (gradient updates); "
                          "--epochs becomes a ceiling only. dev/AUDIT_2026-09-08.md finding 7: "
                          "fixed-EPOCH runs give bigger n_train arms more gradient updates for "
                          "free, confounding 'more data' with 'more optimization'.")
    ap.add_argument("--subset-seed", type=int, default=None,
                     help="(item B) seed for a stratified subset of size --n-train drawn from the "
                          "training pool (or, with --holdout-file, from the pool minus the held-out "
                          "sentences); defaults to --seed. Independent of the split seed, so "
                          "v3@788 and v3@3000 are separate, reproducible draws.")
    ap.add_argument("--holdout-file", default=None,
                     help="(item C) text file of sentences (one per line) EXCLUDED from training "
                          "and used as the eval set instead of the gold-derived dev/test split. "
                          "See scripts/make_holdout.py.")
    ap.add_argument("--eval-gold", default=None,
                     help="(item C) gold file whose records for the holdout sentences are the eval "
                          "targets; only used with --holdout-file. Default: --gold.")
    ap.add_argument("--eval-gold-alt", default=None,
                     help="(item C) second gold file to ALSO score the holdout sentences against "
                          "(e.g. score a v3-trained arm against v2 targets for continuity AND v3 "
                          "targets); only used with --holdout-file.")
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
        hash_buckets = args.hash_buckets or 4096
        epochs = args.epochs or 15
        batch_size = args.batch_size or 32

    if args.max_steps is not None and args.epochs is None:
        # item A: "--max-steps N ... regardless of epochs; epochs becomes a
        # ceiling only". The user didn't pin --epochs, so raise the ceiling
        # far above what --max-steps could plausibly need -- --max-steps (or
        # --max-seconds) is the real budget now, not the smoke/full default.
        epochs = max(epochs, 100_000)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] loading gold from {args.gold}")
    records = load_gold(args.gold)
    print(f"[{time.time()-t0:6.1f}s] {len(records)} gold records")

    holdout_sentences = None
    dev_recs, test_recs = [], []
    if args.holdout_file:
        holdout_sentences = etu.load_holdout_sentences(args.holdout_file)
        pool = etu.exclude_by_text(records, holdout_sentences)
        subset_seed = args.subset_seed if args.subset_seed is not None else args.seed
        train_recs = etu.seeded_subset(pool, n_train, subset_seed)
        print(f"[{time.time()-t0:6.1f}s] holdout mode: {len(holdout_sentences)} held-out sentences, "
              f"pool={len(pool)}, train subset={len(train_recs)} (subset_seed={subset_seed})")
    else:
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
    train_items = etu.build_train_items(train_recs, usvs, pos_vocab, hash_buckets)
    print(f"[{time.time()-t0:6.1f}s] {len(train_items)} teacher-forced derivations")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def on_step_50(epoch, step_count, avg):
        print(f"[{time.time()-t0:6.1f}s] epoch {epoch} step {step_count} avg_loss={avg:.3f}")

    def on_epoch_done(epoch, avg, epoch_n):
        print(f"[{time.time()-t0:6.1f}s] === epoch {epoch} done: avg_loss={avg:.3f} (n={epoch_n} derivations) ===")

    def on_max_seconds(epoch):
        print(f"[{time.time()-t0:6.1f}s] max-seconds budget hit before epoch {epoch}; stopping")

    result = etu.run_training_loop(
        model, train_items, opt, epochs=epochs, batch_size=batch_size,
        max_seconds=args.max_seconds, max_steps=args.max_steps,
        terminal_weight=args.terminal_weight,
        on_step_50=on_step_50, on_epoch_done=on_epoch_done, on_max_seconds=on_max_seconds)

    loss_curve = result["loss_curve"]
    train_wall = result["train_wall"]
    stopped_early = result["stopped_early"]
    print(f"[{time.time()-t0:6.1f}s] training wall-clock: {train_wall:.1f}s (stopped_early={stopped_early})")
    if args.max_steps is not None:
        opt_steps = result["optimizer_steps"]
        throughput = opt_steps / train_wall if train_wall > 0 else float("nan")
        s_per_step = train_wall / opt_steps if opt_steps else float("nan")
        print(f"[{time.time()-t0:6.1f}s] max-steps budget: optimizer_steps={opt_steps} "
              f"(cap={args.max_steps}, stop_reason={result['stop_reason']}) "
              f"epoch_fraction={result['epoch_fraction']:.3f} "
              f"throughput={throughput:.4f} steps/s ({s_per_step:.4f} s/step)")

    print(f"[{time.time()-t0:6.1f}s] evaluating (model policy) ...")
    metrics = {}
    if holdout_sentences is not None:
        eval_gold_path = args.eval_gold or args.gold
        eval_records = records if eval_gold_path == args.gold else load_gold(eval_gold_path)
        eval_targets, n_missing = etu.records_for_sentences(eval_records, holdout_sentences)
        print(f"[{time.time()-t0:6.1f}s] holdout eval targets from {eval_gold_path}: "
              f"matched={len(eval_targets)} missing={n_missing}")
        m = etu.evaluate_full(model, eval_targets, usvs, pos_vocab, hash_buckets,
                               beam_width=args.beam_width, k=args.k, policy="model")
        metrics["holdout"] = m
        print(f"[{time.time()-t0:6.1f}s] holdout (best-of-{args.k} + rank1 + forest width): {m}")

        if args.eval_gold_alt:
            alt_records = load_gold(args.eval_gold_alt)
            alt_targets, alt_missing = etu.records_for_sentences(alt_records, holdout_sentences)
            print(f"[{time.time()-t0:6.1f}s] holdout eval targets (alt) from {args.eval_gold_alt}: "
                  f"matched={len(alt_targets)} missing={alt_missing}")
            m_alt = etu.evaluate_full(model, alt_targets, usvs, pos_vocab, hash_buckets,
                                       beam_width=args.beam_width, k=args.k, policy="model")
            metrics["holdout_alt"] = m_alt
            print(f"[{time.time()-t0:6.1f}s] holdout_alt (best-of-{args.k} + rank1 + forest width): {m_alt}")
    else:
        for split_name, split_recs in (("train", train_recs), ("dev", dev_recs), ("test", test_recs)):
            m = etu.evaluate_full(model, split_recs, usvs, pos_vocab, hash_buckets,
                                   beam_width=args.beam_width, k=args.k, policy="model")
            metrics[split_name] = m
            print(f"[{time.time()-t0:6.1f}s] {split_name}: {m}")

    random_target = eval_targets if holdout_sentences is not None else test_recs
    random_target_name = "holdout" if holdout_sentences is not None else "test"
    print(f"[{time.time()-t0:6.1f}s] evaluating RANDOM baseline on {random_target_name} ...")
    rng = random.Random(args.seed)
    random_metrics = em.evaluate(model, random_target, usvs, pos_vocab, hash_buckets,
                                  beam_width=args.beam_width, k=args.k, policy="random", rng=rng)
    print(f"[{time.time()-t0:6.1f}s] {random_target_name} (random baseline): {random_metrics}")

    ckpt = {
        "model_state": model.state_dict(),
        "pos_vocab": pos_vocab,
        "role_vocab": role_vocab,
        "d_axes": d_axes,
        "hash_buckets": hash_buckets,
        "d_model": d_model,
        "config": {"n_train": len(train_recs), "n_dev": len(dev_recs), "n_test": len(test_recs),
                   "epochs": epochs, "batch_size": batch_size, "seed": args.seed,
                   "terminal_weight": args.terminal_weight, "max_steps": args.max_steps,
                   "subset_seed": args.subset_seed, "holdout_file": args.holdout_file,
                   "eval_gold": args.eval_gold, "eval_gold_alt": args.eval_gold_alt,
                   "optimizer_steps": result["optimizer_steps"], "stop_reason": result["stop_reason"],
                   "epoch_fraction": result["epoch_fraction"]},
        "loss_curve": loss_curve,
        "metrics": metrics,
        "random_baseline_test": random_metrics,
        "train_wallclock_s": train_wall,
        "n_policy_params": n_params,
        "split_record_texts": {"train": [r["text"] for r in train_recs],
                                "dev": [r["text"] for r in dev_recs],
                                "test": [r["text"] for r in test_recs],
                                "holdout": list(holdout_sentences) if holdout_sentences is not None else []},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[{time.time()-t0:6.1f}s] saved checkpoint -> {args.out}")
    print(f"[{time.time()-t0:6.1f}s] TOTAL wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
