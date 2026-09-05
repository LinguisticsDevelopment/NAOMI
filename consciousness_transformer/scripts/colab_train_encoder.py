"""Self-contained Colab driver: train the candidate-lattice encoder
(dev/ENCODER_MODEL_SPEC.md) to convergence from a single command, then
report EN candidate-set recall and the Spanish grammar-swap eval.

This is glue, not a second implementation: the actual training step
(teacher_force_loss) and every metric (evaluate / beam_decode /
score_record / aggregate_recall) are imported and called VERBATIM from
scripts/train_encoder.py and scripts/eval_encoder_on.py -- this file only
adds argument parsing, USVS/device bootstrap, and the printed RESULTS
block. Do not duplicate loss or metric math here; if something about the
training loop needs to change, change it in train_encoder.py and this
driver will pick it up.

Intended flow, run from a fresh clone of this repo with deps installed
(see colab/Encoder_Train.ipynb for the exact three Colab cells):

    pip install -e .
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
    python scripts/build_usvs.py                      # skipped here if already built
    git show origin/encoder-gold-v2:consciousness_transformer/runs/encoder_gold_v2.jsonl \
        > runs/encoder_gold_v2.jsonl
    git show origin/spanish-gold-v2:consciousness_transformer/runs/spanish_gold_v2.jsonl \
        > runs/spanish_gold_v2.jsonl
    python scripts/colab_train_encoder.py --records 900 --epochs 15 --out runs/encoder_colab.pt

Smoke (tiny, proves the driver runs end to end -- numbers are meaningless):

    python scripts/colab_train_encoder.py --records 40 --epochs 1 --device cpu \
        --out /tmp/enc_smoke.pt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from nsm_ct.ground.usvs import build_usvs, load_usvs, save_usvs
from nsm_ct import encoder_model as em
from train_encoder import load_gold, stratified_split  # noqa: E402
from eval_encoder_on import evaluate_with_totals  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def ensure_usvs(usvs_dir: Path, log) -> None:
    """Build the USVS artifact if it isn't already on disk (fresh-clone case).
    Reuses nsm_ct.ground.usvs.build_usvs/save_usvs verbatim -- same call
    scripts/build_usvs.py makes."""
    if (usvs_dir / "usvs.npz").exists():
        log(f"USVS already present at {usvs_dir}")
        return
    log(f"USVS not found at {usvs_dir}; building (~1-2 min)...")
    u = build_usvs(log=lambda m: log(f"  [usvs] {m}"))
    save_usvs(u, str(usvs_dir))
    log(f"USVS built and saved -> {usvs_dir}")


def resolve_device(requested: str, log) -> str:
    """nsm_ct.encoder_model never calls .to(device) or passes device= to any
    tensor constructor -- EncoderModel.controller_step and _apply_action
    build their per-step index tensors with plain torch.tensor([...]), which
    always land on CPU regardless of where the model's own parameters live.
    Moving the model to CUDA would break the very first embedding lookup
    against those CPU index tensors (device-mismatch RuntimeError). Until
    that's fixed upstream, CUDA genuinely isn't exploited by this
    implementation -- report that honestly instead of erroring or silently
    claiming GPU use."""
    want_cuda = requested == "cuda" or (requested == "auto" and torch.cuda.is_available())
    if not want_cuda:
        if requested == "cuda":
            log("NOTE: --device cuda requested but CUDA is not available on this machine; using CPU.")
        return "cpu"
    log("NOTE: CUDA is available, but nsm_ct.encoder_model has no device-placement support "
        "(its controller builds plain CPU index tensors for every embedding lookup each step) -- "
        "GPU is NOT exploited by the current model/training code. Training on CPU. "
        "This is an accurate limitation of the current implementation, not a driver bug; "
        "it will need EncoderModel/controller_step to grow real device support to change.")
    return "cpu"


def split_sizes(n_records: int, n_available: int, log) -> tuple:
    if n_records > n_available:
        log(f"NOTE: --records {n_records} exceeds the {n_available} gold records available; "
            f"clamping to {n_available}.")
        n_records = n_available
    n_dev = max(1, round(n_records * 0.1))
    n_test = max(1, round(n_records * 0.1))
    n_train = max(1, n_records - n_dev - n_test)
    return n_train, n_dev, n_test


def train(model, train_items, epochs, batch_size, lr, max_seconds, t0, log):
    """Same teacher-forced training loop as scripts/train_encoder.py's
    main(), calling em.teacher_force_loss verbatim -- no reimplemented loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_curve = []
    step_count = 0
    train_start = time.time()
    stopped_early = False
    for epoch in range(epochs):
        if time.time() - train_start > max_seconds:
            stopped_early = True
            log(f"max-seconds budget ({max_seconds}s) hit before epoch {epoch}; stopping")
            break
        random.shuffle(train_items)
        epoch_loss = 0.0
        epoch_n = 0
        opt.zero_grad()
        for idx, (feats, steps) in enumerate(train_items):
            if time.time() - train_start > max_seconds:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad()
        avg = epoch_loss / max(epoch_n, 1)
        loss_curve.append((step_count, avg))
        log(f"epoch {epoch + 1}/{epochs} done: avg_loss={avg:.4f} (n={epoch_n} derivations)")
        if stopped_early:
            break
    train_wall = time.time() - train_start
    return loss_curve, train_wall, stopped_early


def fmt_recall(m: dict) -> str:
    return (f"sense={m.get('sense_recall', float('nan')):.3f} "
            f"slot={m.get('slot_recall', float('nan')):.3f} "
            f"structure={m.get('structure_recall', float('nan')):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=int, default=984,
                     help="total gold records to split into train/dev/test (~80/10/10); "
                          "984 reproduces the dev/ENCODER_MODEL_SPEC.md S2.3 full-Stage-i split "
                          "(788/98/98) out of the 985 available")
    ap.add_argument("--epochs", type=int, default=50,
                     help="~50-60 min end-to-end on this project's CPU dev box at the default "
                          "--records (see scripts/colab_train_encoder.py smoke-timing notes); "
                          "Colab CPU speed may differ")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--out", default=str(ROOT / "runs" / "encoder_colab.pt"))
    ap.add_argument("--gold", default=str(ROOT / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--spanish-gold", default=str(ROOT / "runs" / "spanish_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(ROOT / "data" / "usvs"))
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--hash-buckets", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=5400.0, help="hard training-time cutoff")
    ap.add_argument("--beam-width", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    device = resolve_device(args.device, log)

    gold_path = Path(args.gold)
    if not gold_path.exists():
        sys.exit(f"ERROR: English gold file not found at {gold_path}. Fetch it first, e.g.:\n"
                  f"  git show origin/encoder-gold-v2:consciousness_transformer/runs/encoder_gold_v2.jsonl "
                  f"> {gold_path}")
    spanish_path = Path(args.spanish_gold)
    if not spanish_path.exists():
        sys.exit(f"ERROR: Spanish gold file not found at {spanish_path}. Fetch it first, e.g.:\n"
                  f"  git show origin/spanish-gold-v2:consciousness_transformer/runs/spanish_gold_v2.jsonl "
                  f"> {spanish_path}")

    ensure_usvs(Path(args.usvs_dir), log)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    log(f"loading English gold from {gold_path}")
    records = load_gold(str(gold_path))
    log(f"{len(records)} English gold records available")

    n_train, n_dev, n_test = split_sizes(args.records, len(records), log)
    train_recs, dev_recs, test_recs = stratified_split(records, args.seed, n_train, n_dev, n_test)
    log(f"split: train={len(train_recs)} dev={len(dev_recs)} test={len(test_recs)} "
        f"(held out from training, used for candidate-set-recall eval)")

    log(f"loading USVS from {args.usvs_dir}")
    usvs = load_usvs(args.usvs_dir)
    d_axes = len(usvs.axes)
    log(f"USVS loaded: {len(usvs.sense_ids)} senses, d_axes={d_axes}")

    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    log(f"pos_vocab={len(pos_vocab)} role_vocab={len(role_vocab)}")

    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=d_axes, hash_buckets=args.hash_buckets,
                             d_model=args.d_model, controller_hidden=args.d_model)
    model.to(device)
    n_params = model.num_policy_params()
    n_bytes = n_params * 4
    log(f"policy params: {n_params:,} (~{n_bytes / 1e6:.3f} MB fp32)")

    log(f"building features + teacher-forced derivations for {len(train_recs)} train records")
    train_items = []
    for r in train_recs:
        feats = em.build_features(r, usvs, pos_vocab, args.hash_buckets)
        for tree in r["lattice"]["trees"]:
            steps = em.linearize_tree(r, tree)
            train_items.append((feats, steps))
    log(f"{len(train_items)} teacher-forced derivations, batch_size={args.batch_size}, "
        f"epochs={args.epochs}")

    loss_curve, train_wall, stopped_early = train(
        model, train_items, args.epochs, args.batch_size, args.lr, args.max_seconds, t0, log)
    log(f"training wall-clock: {train_wall:.1f}s (stopped_early={stopped_early})")

    model.eval()

    log("evaluating English candidate-set recall (model policy) on held-out test split ...")
    en_model_metrics = em.evaluate(model, test_recs, usvs, pos_vocab, args.hash_buckets,
                                    beam_width=args.beam_width, k=args.k, policy="model")
    log(f"English test (model) : {fmt_recall(en_model_metrics)}")

    rng = random.Random(args.seed)
    log("evaluating English candidate-set recall (random-legal baseline) on held-out test split ...")
    en_random_metrics = em.evaluate(model, test_recs, usvs, pos_vocab, args.hash_buckets,
                                     beam_width=args.beam_width, k=args.k, policy="random", rng=rng)
    log(f"English test (random): {fmt_recall(en_random_metrics)}")

    log(f"loading Spanish gold from {spanish_path} for the grammar-swap eval "
        f"(EN-trained weights -> Spanish, zero Spanish training)")
    spanish_records = load_gold(str(spanish_path))
    log(f"{len(spanish_records)} Spanish gold records")

    log("evaluating Spanish candidate-set recall (model policy, EN-trained weights) ...")
    es_model_metrics = evaluate_with_totals(model, spanish_records, usvs, pos_vocab, args.hash_buckets,
                                             args.beam_width, args.k, "model")
    log(f"Spanish (model) : {fmt_recall(es_model_metrics)}")

    es_rng = random.Random(args.seed)
    log("evaluating Spanish candidate-set recall (random-legal baseline) ...")
    es_random_metrics = evaluate_with_totals(model, spanish_records, usvs, pos_vocab, args.hash_buckets,
                                              args.beam_width, args.k, "random", rng=es_rng)
    log(f"Spanish (random): {fmt_recall(es_random_metrics)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.to("cpu").state_dict(),
        "pos_vocab": pos_vocab,
        "role_vocab": role_vocab,
        "d_axes": d_axes,
        "hash_buckets": args.hash_buckets,
        "d_model": args.d_model,
        "config": {"n_train": len(train_recs), "n_dev": len(dev_recs), "n_test": len(test_recs),
                   "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
                   "device": device},
        "loss_curve": loss_curve,
        "metrics": {"english_test": en_model_metrics, "english_test_random": en_random_metrics,
                    "spanish": es_model_metrics, "spanish_random": es_random_metrics},
        "train_wallclock_s": train_wall,
        "n_policy_params": n_params,
        "split_record_texts": {"train": [r["text"] for r in train_recs],
                                "dev": [r["text"] for r in dev_recs],
                                "test": [r["text"] for r in test_recs]},
    }
    torch.save(ckpt, out_path)
    log(f"saved checkpoint -> {out_path}")

    total_wall = time.time() - t0
    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"records: {len(train_recs)} train / {len(dev_recs)} dev / {len(test_recs)} test "
          f"(of {len(records)} EN gold available)  |  epochs: {args.epochs}  |  device: {device}")
    print(f"policy params: {n_params:,} (~{n_bytes / 1e6:.3f} MB fp32)")
    print()
    print("English (held-out test split):")
    print(f"  model : {fmt_recall(en_model_metrics)}")
    print(f"  random: {fmt_recall(en_random_metrics)}")
    print()
    print(f"Spanish grammar-swap ({len(spanish_records)} records, EN-trained weights, zero ES training):")
    print(f"  model : {fmt_recall(es_model_metrics)}")
    print(f"  random: {fmt_recall(es_random_metrics)}")
    print()
    print(f"training wall-clock: {train_wall:.1f}s  |  total wall-clock: {total_wall:.1f}s")
    print(f"checkpoint: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
