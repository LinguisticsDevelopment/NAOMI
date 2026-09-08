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

encoder-train-arms-v2 (equal-STEPS is unfair to single-tree gold: v3 gold
has ~1 derivation/record vs ~3.3 for v2 forest gold, so the SAME
--max-steps is a wildly different number of epochs over the corpus):
--eval-every runs a cheap periodic rank-1 dev eval and logs a learning
curve (<out>.curve.tsv); --keep-best selects the checkpoint at the best
dev score instead of always taking the final one; --patience stops early
on a dev plateau; --extra-eval scores the saved checkpoint on additional
gold files in full, reported per meta.family. All additive: without
--eval-every, none of this changes anything.

Usage:
    python scripts/train_encoder.py --smoke --out runs/encoder_smoke.pt
    python scripts/train_encoder.py --smoke --max-steps 60 \
        --holdout-file runs/holdout_sentences.txt --gold runs/encoder_gold_v2.jsonl
    python scripts/train_encoder.py --smoke --max-steps 60 --eval-every 20 --keep-best \
        --holdout-file runs/holdout_sentences.txt --gold runs/encoder_gold_v2.jsonl
"""

from __future__ import annotations

import argparse
import copy
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

ROOT = Path(__file__).resolve().parent.parent


def _rank1_f1(m: dict) -> float:
    return m.get("rank1_edge_f1", float("nan"))


def evaluate_extra_by_family(model, gold_path: str, usvs, pos_vocab, hash_buckets: int,
                              beam_width: int, k: int) -> dict:
    """(item D) Score `model` on EVERY record of `gold_path` (not matched
    against a holdout sentence list -- the whole file is the eval set),
    grouped by `record["meta"]["family"]` (runs/hard_gold_test_filler.jsonl
    / runs/hard_gold_test_template.jsonl's own test splits). Returns
    {family: {n, rank1_edge_f1, rank1_edge_precision, rank1_edge_recall}}."""
    records = load_gold(gold_path)
    by_family: dict = {}
    for r in records:
        fam = (r.get("meta") or {}).get("family", "unknown")
        by_family.setdefault(fam, []).append(r)
    out = {}
    for fam, fam_recs in sorted(by_family.items()):
        m = etu.evaluate_full(model, fam_recs, usvs, pos_vocab, hash_buckets,
                               beam_width=beam_width, k=k, policy="model")
        out[fam] = {"n": len(fam_recs), "rank1_edge_f1": m["rank1_edge_f1"],
                    "rank1_edge_precision": m["rank1_edge_precision"],
                    "rank1_edge_recall": m["rank1_edge_recall"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(ROOT / "data" / "usvs"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "encoder_smoke.pt"))
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
    ap.add_argument("--n-train-per-file", default=None,
                     help="(arms-3 minimal support for fixed source ratios) comma list of "
                          "per-file train-subset sizes, one count per file in --gold's comma "
                          "list, same order (e.g. --gold a.jsonl,b.jsonl --n-train-per-file "
                          "788,200). Each file's own records (after --holdout-file / "
                          "--dev-holdout-file exclusion, and after dropping any text already "
                          "taken from an earlier file in the list) get their own seeded_subset "
                          "of the requested size, then all subsets are concatenated -- unlike "
                          "plain --n-train, which draws ONE stratified subset from the files' "
                          "UNION and can under-represent a smaller source file. Only valid with "
                          "--holdout-file; ignored (with --n-train used instead) otherwise.")
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
    ap.add_argument("--eval-every", type=int, default=0,
                     help="(encoder-train-arms-v2 item A) every N optimizer steps, run a cheap "
                          "single-hypothesis (beam_width=1, k=1) rank-1 dev eval on "
                          "--dev-holdout-file and append a row to <out>.curve.tsv (step, "
                          "train_avg_loss, dev_rank1_f1, dev_edge_p, dev_edge_r, wall_s). Training "
                          "keeps going regardless (no early stop unless --patience is also given). "
                          "0 (default) disables periodic eval entirely -- unchanged behavior.")
    ap.add_argument("--dev-holdout-file", default=str(ROOT / "runs" / "holdout_dev_sentences.txt"),
                     help="(item A) text file of dev sentences (one per line) used for the "
                          "periodic --eval-every eval; in --holdout-file mode these are ALSO "
                          "excluded from the training pool (same as --holdout-file's own "
                          "sentences). Only consulted when --eval-every > 0. See "
                          "scripts/make_holdout.py.")
    ap.add_argument("--keep-best", action="store_true",
                     help="(item B) track the best --eval-every dev rank-1 F1 seen during "
                          "training; save that checkpoint as --out and the checkpoint from the "
                          "final step as <out>.last.pt instead. No effect without --eval-every "
                          "(or if no eval ever ran, e.g. training stopped before step 1).")
    ap.add_argument("--patience", type=int, default=None,
                     help="(item B) stop training if --eval-every dev rank-1 F1 hasn't improved "
                          "for this many evals in a row. No effect without --eval-every.")
    ap.add_argument("--extra-eval", default=None,
                     help="(item D) comma list of additional gold files to evaluate the saved "
                          "checkpoint on IN FULL (every record in the file is an eval target, not "
                          "matched against a holdout sentence list), reported per "
                          "record['meta']['family'] with rank-1 edge F1 -- e.g. "
                          "runs/hard_gold_test_filler.jsonl,runs/hard_gold_test_template.jsonl.")
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

    do_dev_eval = args.eval_every > 0

    holdout_sentences = None
    dev_recs, test_recs = [], []
    dev_eval_sentences: list = []
    if args.holdout_file:
        holdout_sentences = etu.load_holdout_sentences(args.holdout_file)
        subset_seed = args.subset_seed if args.subset_seed is not None else args.seed
        if do_dev_eval:
            dev_eval_sentences = etu.load_holdout_sentences(args.dev_holdout_file)
        if args.n_train_per_file:
            per_file_n = [int(x) for x in args.n_train_per_file.split(",")]
            gold_paths = [p.strip() for p in args.gold.split(",") if p.strip()]
            if len(per_file_n) != len(gold_paths):
                raise SystemExit(f"--n-train-per-file has {len(per_file_n)} counts but --gold "
                                  f"lists {len(gold_paths)} files")
            train_recs = []
            seen_text = set()
            per_file_report = []
            for gp, n_i in zip(gold_paths, per_file_n):
                file_pool = etu.exclude_by_text(etu.load_gold(gp), holdout_sentences)
                if do_dev_eval:
                    file_pool = etu.exclude_by_text(file_pool, dev_eval_sentences)
                file_pool = [r for r in file_pool if r["text"] not in seen_text]
                file_subset = etu.seeded_subset(file_pool, n_i, subset_seed)
                seen_text.update(r["text"] for r in file_subset)
                train_recs.extend(file_subset)
                per_file_report.append(f"{gp}: pool={len(file_pool)} subset={len(file_subset)}")
            print(f"[{time.time()-t0:6.1f}s] holdout mode (per-file subsets): "
                  f"{len(holdout_sentences)} held-out sentences, "
                  f"train subset={len(train_recs)} (subset_seed={subset_seed}) -- "
                  + "; ".join(per_file_report))
        else:
            pool = etu.exclude_by_text(records, holdout_sentences)
            if do_dev_eval:
                pool = etu.exclude_by_text(pool, dev_eval_sentences)
            train_recs = etu.seeded_subset(pool, n_train, subset_seed)
            print(f"[{time.time()-t0:6.1f}s] holdout mode: {len(holdout_sentences)} held-out sentences, "
                  f"pool={len(pool)}, train subset={len(train_recs)} (subset_seed={subset_seed})")
    else:
        train_recs, dev_recs, test_recs = stratified_split(records, args.seed, n_train, n_dev, n_test)
        print(f"[{time.time()-t0:6.1f}s] split: train={len(train_recs)} dev={len(dev_recs)} test={len(test_recs)}")
        if do_dev_eval:
            dev_eval_sentences = [r["text"] for r in dev_recs]

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
    total_items = len(train_items)

    dev_eval_targets: list = []
    if do_dev_eval:
        if holdout_sentences is not None:
            dev_eval_gold_path = args.eval_gold or args.gold
            dev_eval_pool = records if dev_eval_gold_path == args.gold else load_gold(dev_eval_gold_path)
            dev_eval_targets, dev_missing = etu.records_for_sentences(dev_eval_pool, dev_eval_sentences)
            print(f"[{time.time()-t0:6.1f}s] dev-holdout eval targets from {dev_eval_gold_path}: "
                  f"matched={len(dev_eval_targets)} missing={dev_missing}")
        else:
            dev_eval_targets = dev_recs

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def on_step_50(epoch, step_count, avg):
        print(f"[{time.time()-t0:6.1f}s] epoch {epoch} step {step_count} avg_loss={avg:.3f}")

    def on_epoch_done(epoch, avg, epoch_n):
        print(f"[{time.time()-t0:6.1f}s] === epoch {epoch} done: avg_loss={avg:.3f} (n={epoch_n} derivations) ===")

    def on_max_seconds(epoch):
        print(f"[{time.time()-t0:6.1f}s] max-seconds budget hit before epoch {epoch}; stopping")

    curve_rows: list = []
    best_state = None
    best_step = None
    best_dev_rank1_f1 = None
    patience_counter = 0

    def on_optimizer_step(step, avg_loss, wall_s):
        nonlocal best_state, best_step, best_dev_rank1_f1, patience_counter
        if step % args.eval_every != 0:
            return False
        dev_m = etu.evaluate_dev_fast(model, dev_eval_targets, usvs, pos_vocab, hash_buckets)
        f1 = dev_m["rank1_edge_f1"]
        curve_rows.append((step, avg_loss, f1, dev_m["rank1_edge_precision"], dev_m["rank1_edge_recall"], wall_s))
        print(f"[{time.time()-t0:6.1f}s] eval step={step} train_avg_loss={avg_loss:.3f} "
              f"dev_rank1_f1={f1:.4f} dev_edge_p={dev_m['rank1_edge_precision']:.4f} "
              f"dev_edge_r={dev_m['rank1_edge_recall']:.4f}")
        is_better = (f1 == f1) and (best_dev_rank1_f1 is None or f1 > best_dev_rank1_f1)
        if is_better:
            best_dev_rank1_f1 = f1
            best_step = step
            if args.keep_best:
                best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if args.patience is not None and patience_counter >= args.patience:
            print(f"[{time.time()-t0:6.1f}s] --patience {args.patience} exhausted "
                  f"({patience_counter} evals without dev improvement); stopping")
            return True
        return False

    result = etu.run_training_loop(
        model, train_items, opt, epochs=epochs, batch_size=batch_size,
        max_seconds=args.max_seconds, max_steps=args.max_steps,
        terminal_weight=args.terminal_weight,
        on_step_50=on_step_50, on_epoch_done=on_epoch_done, on_max_seconds=on_max_seconds,
        on_optimizer_step=on_optimizer_step if do_dev_eval else None)

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

    curve_path = None
    if do_dev_eval:
        curve_path = args.out + ".curve.tsv"
        Path(curve_path).parent.mkdir(parents=True, exist_ok=True)
        with open(curve_path, "w") as f:
            f.write("step\ttrain_avg_loss\tdev_rank1_f1\tdev_edge_p\tdev_edge_r\twall_s\n")
            for step, avg_loss, f1, p, r, wall_s in curve_rows:
                f.write(f"{step}\t{avg_loss:.6f}\t{f1:.6f}\t{p:.6f}\t{r:.6f}\t{wall_s:.2f}\n")
        print(f"[{time.time()-t0:6.1f}s] wrote {len(curve_rows)} curve rows -> {curve_path}")

    eval_targets = None
    alt_targets = None
    if holdout_sentences is not None:
        eval_gold_path = args.eval_gold or args.gold
        eval_records = records if eval_gold_path == args.gold else load_gold(eval_gold_path)
        eval_targets, n_missing = etu.records_for_sentences(eval_records, holdout_sentences)
        print(f"[{time.time()-t0:6.1f}s] holdout eval targets from {eval_gold_path}: "
              f"matched={len(eval_targets)} missing={n_missing}")

        if args.eval_gold_alt:
            alt_records = load_gold(args.eval_gold_alt)
            alt_targets, alt_missing = etu.records_for_sentences(alt_records, holdout_sentences)
            print(f"[{time.time()-t0:6.1f}s] holdout eval targets (alt) from {args.eval_gold_alt}: "
                  f"matched={len(alt_targets)} missing={alt_missing}")

    def run_final_eval(tag: str = None) -> dict:
        prefix = f"[{tag}] " if tag else ""
        metrics_ = {}
        print(f"[{time.time()-t0:6.1f}s] evaluating {prefix}(model policy) ...")
        if holdout_sentences is not None:
            m = etu.evaluate_full(model, eval_targets, usvs, pos_vocab, hash_buckets,
                                   beam_width=args.beam_width, k=args.k, policy="model")
            metrics_["holdout"] = m
            print(f"[{time.time()-t0:6.1f}s] {prefix}holdout (best-of-{args.k} + rank1 + forest width): {m}")

            if args.eval_gold_alt:
                m_alt = etu.evaluate_full(model, alt_targets, usvs, pos_vocab, hash_buckets,
                                           beam_width=args.beam_width, k=args.k, policy="model")
                metrics_["holdout_alt"] = m_alt
                print(f"[{time.time()-t0:6.1f}s] {prefix}holdout_alt (best-of-{args.k} + rank1 + forest width): {m_alt}")
        else:
            for split_name, split_recs in (("train", train_recs), ("dev", dev_recs), ("test", test_recs)):
                m = etu.evaluate_full(model, split_recs, usvs, pos_vocab, hash_buckets,
                                       beam_width=args.beam_width, k=args.k, policy="model")
                metrics_[split_name] = m
                print(f"[{time.time()-t0:6.1f}s] {prefix}{split_name}: {m}")
        return metrics_

    # item C: without --keep-best (or if no eval ever improved on the
    # initial state), there is only ONE checkpoint -- run_final_eval(None)
    # reproduces the original untagged prints exactly, byte-for-byte.
    show_both = args.keep_best and best_state is not None
    metrics = run_final_eval("last" if show_both else None)

    last_state = None
    metrics_best = None
    if show_both:
        last_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        metrics_best = run_final_eval("best")
    # From here on `model` holds whatever will be saved as --out: the BEST
    # checkpoint if show_both, otherwise unchanged (the final/only state).

    random_target = eval_targets if holdout_sentences is not None else test_recs
    random_target_name = "holdout" if holdout_sentences is not None else "test"
    print(f"[{time.time()-t0:6.1f}s] evaluating RANDOM baseline on {random_target_name} ...")
    rng = random.Random(args.seed)
    random_metrics = em.evaluate(model, random_target, usvs, pos_vocab, hash_buckets,
                                  beam_width=args.beam_width, k=args.k, policy="random", rng=rng)
    print(f"[{time.time()-t0:6.1f}s] {random_target_name} (random baseline): {random_metrics}")

    extra_eval_results = {}
    if args.extra_eval:
        for gp in args.extra_eval.split(","):
            gp = gp.strip()
            if not gp:
                continue
            print(f"[{time.time()-t0:6.1f}s] extra-eval: {gp}")
            fam_metrics = evaluate_extra_by_family(model, gp, usvs, pos_vocab, hash_buckets,
                                                    args.beam_width, args.k)
            extra_eval_results[gp] = fam_metrics
            for fam, m in fam_metrics.items():
                print(f"[{time.time()-t0:6.1f}s]   family={fam} n={m['n']} "
                      f"rank1_edge_f1={m['rank1_edge_f1']:.4f} p={m['rank1_edge_precision']:.4f} "
                      f"r={m['rank1_edge_recall']:.4f}")

    last_dev_rank1_f1 = curve_rows[-1][2] if curve_rows else None
    last_epochs_equivalent = etu.epochs_equivalent(result["optimizer_steps"], batch_size, total_items)

    def build_config(step_for_epochs_equiv: int) -> dict:
        return {"n_train": len(train_recs), "n_dev": len(dev_recs), "n_test": len(test_recs),
                "epochs": epochs, "batch_size": batch_size, "seed": args.seed,
                "terminal_weight": args.terminal_weight, "max_steps": args.max_steps,
                "subset_seed": args.subset_seed, "holdout_file": args.holdout_file,
                "eval_gold": args.eval_gold, "eval_gold_alt": args.eval_gold_alt,
                "optimizer_steps": result["optimizer_steps"], "stop_reason": result["stop_reason"],
                "epoch_fraction": result["epoch_fraction"],
                "eval_every": args.eval_every,
                "dev_holdout_file": args.dev_holdout_file if do_dev_eval else None,
                "keep_best": args.keep_best, "patience": args.patience,
                "curve_path": curve_path,
                "best_step": best_step, "best_dev_rank1_f1": best_dev_rank1_f1,
                "last_dev_rank1_f1": last_dev_rank1_f1,
                "epochs_equivalent": etu.epochs_equivalent(step_for_epochs_equiv, batch_size, total_items)}

    def build_ckpt(model_state, metrics_, config) -> dict:
        return {
            "model_state": model_state,
            "pos_vocab": pos_vocab,
            "role_vocab": role_vocab,
            "d_axes": d_axes,
            "hash_buckets": hash_buckets,
            "d_model": d_model,
            "config": config,
            "loss_curve": loss_curve,
            "metrics": metrics_,
            "random_baseline_test": random_metrics,
            "extra_eval": extra_eval_results,
            "train_wallclock_s": train_wall,
            "n_policy_params": n_params,
            "split_record_texts": {"train": [r["text"] for r in train_recs],
                                    "dev": [r["text"] for r in dev_recs],
                                    "test": [r["text"] for r in test_recs],
                                    "holdout": list(holdout_sentences) if holdout_sentences is not None else []},
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if show_both:
        ckpt_best = build_ckpt(best_state, metrics_best, build_config(best_step))
        ckpt_last = build_ckpt(last_state, metrics, build_config(result["optimizer_steps"]))
        torch.save(ckpt_best, args.out)
        torch.save(ckpt_last, args.out + ".last.pt")
        print(f"[{time.time()-t0:6.1f}s] saved BEST checkpoint (step={best_step}, "
              f"dev_rank1_f1={best_dev_rank1_f1:.4f}) -> {args.out}")
        print(f"[{time.time()-t0:6.1f}s] saved LAST checkpoint -> {args.out}.last.pt")
    else:
        ckpt = build_ckpt(model.state_dict(), metrics, build_config(result["optimizer_steps"]))
        torch.save(ckpt, args.out)
        print(f"[{time.time()-t0:6.1f}s] saved checkpoint -> {args.out}")
    print(f"[{time.time()-t0:6.1f}s] TOTAL wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
