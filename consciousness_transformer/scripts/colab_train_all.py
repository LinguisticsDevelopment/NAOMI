"""Self-contained Colab driver: train BOTH the candidate-lattice ENCODER
(dev/ENCODER_MODEL_SPEC.md) and the learned reconstruction DECODER
(RESEARCH_NOTES.md "DECODER PLAN UPDATE") in a single run, then chain them
into the autoencoder ROUND-TRIP test the lead asked for:

    text -> ENCODER.beam_decode -> top committed tree -> DECODER.realize -> text

This is glue, not a third implementation. Every piece of real math is
imported and called verbatim from the existing single-purpose scripts/
modules -- this file only adds: argument parsing, phase sequencing, the
encoder-forest -> decoder-structure bridge (the two tree shapes differ; see
`predicted_tree_to_structure` below), and one combined printed RESULTS
block.

  - Encoder training loop, EN candidate-set recall, Spanish grammar-swap
    eval: reused verbatim from scripts/colab_train_encoder.py
    (`train`, `resolve_device`, `ensure_usvs`, `split_sizes`, `fmt_recall`)
    and scripts/eval_encoder_on.py (`evaluate_with_totals`), which in turn
    call `nsm_ct.encoder_model.teacher_force_loss/evaluate/beam_decode`.
  - Decoder training loss, round-trip-on-gold-tree eval, reconstruction
    metric: reused verbatim from `nsm_ct.decoder_trained`
    (`reconstruction_loss`, `round_trip`, `reconstruction_accuracy`,
    `realize`, `sever_structure_content`, `build_function_vocab`,
    `build_decoder_features`) and scripts/train_decoder.py
    (`load_records`, `split_records`).
  - The one genuinely new piece: `predicted_tree_to_structure`. The
    encoder's `beam_decode` forest and a gold `lattice.trees[*]` entry use
    different clause shapes (predicted clauses nest a `{token_index,
    grounding}` node under `clause["predicate"]`; gold clauses store the
    predicate's surface word directly under `clause["predicate"]` plus a
    sibling `predicate_grounding`) -- `decoder_trained.extract_nodes` reads
    the gold shape only (it calls `encoder_model.clause_node_order`, whose
    `_predicate_token_index` expects a word string). `predicted_tree_to_
    structure` is the same left-to-right, token-index-ordered node walk
    applied to the *predicted* shape, so the trained decoder -- which never
    reads sense candidates, only `(relation, grounding.type, token_index)`
    -- can consume either a gold or a predicted tree identically.

Intended flow (see colab/Train_Encoder_And_Decoder.ipynb for the exact
Colab cells):

    pip install -e .
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
    python scripts/build_usvs.py
    git show origin/encoder-gold-v2:consciousness_transformer/runs/encoder_gold_v2.jsonl \\
        > runs/encoder_gold_v2.jsonl
    git show origin/spanish-gold-v2:consciousness_transformer/runs/spanish_gold_v2.jsonl \\
        > runs/spanish_gold_v2.jsonl
    python scripts/colab_train_all.py --outdir runs/colab_all

Smoke (tiny, proves the driver runs end to end -- numbers are meaningless):

    python scripts/colab_train_all.py --enc-records 40 --enc-epochs 1 \\
        --dec-records 40 --dec-epochs 1 --device cpu --outdir /tmp/all_smoke
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

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from nsm_ct import decoder_trained as dt
from train_encoder import load_gold, stratified_split  # noqa: E402
from eval_encoder_on import evaluate_with_totals  # noqa: E402
from colab_train_encoder import (  # noqa: E402
    ensure_usvs, resolve_device, split_sizes, train as train_encoder_epochs, fmt_recall,
)
from train_decoder import load_records as load_decoder_records, split_records as split_decoder_records  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The encoder-forest -> decoder-structure bridge (see module docstring)
# ---------------------------------------------------------------------------

def predicted_tree_to_structure(record: dict, tree: dict) -> dt.CommittedStructure:
    """A `beam_decode` forest entry (predicted clause shape) -> the same
    `CommittedStructure` `decoder_trained.build_structure` would build from
    a gold tree: one `Node` per (predicate + role) with a real surface
    `token_index`, in left-to-right token order within each clause (nulls
    last) -- the same ordering discipline as `encoder_model.
    clause_node_order`, just applied to the predicted node shape directly
    instead of via that gold-only helper."""
    nodes = []
    for clause in tree.get("clauses", []):
        clause_nodes = []
        pred = clause.get("predicate")
        if pred is not None:
            clause_nodes.append(("PREDICATE", pred.get("grounding") or {}, pred.get("token_index")))
        for role in clause.get("roles", []):
            clause_nodes.append((role.get("relation", "PREDICATE"), role.get("grounding") or {},
                                  role.get("token_index")))
        clause_nodes.sort(key=lambda n: (n[2] is None, n[2] if n[2] is not None else 0))
        for relation, grounding, tidx in clause_nodes:
            if tidx is None:
                continue
            nodes.append(dt.Node(token_index=tidx, word=record["tokens"][tidx],
                                  relation=relation, gtype=grounding.get("type")))
    return dt.CommittedStructure(nodes=nodes, tokens=list(record["tokens"]))


# ---------------------------------------------------------------------------
# Decoder training loop -- same shape as scripts/train_decoder.py's main(),
# calling dt.reconstruction_loss verbatim; train_decoder.py doesn't factor
# its loop into an importable function, so this is the one loop re-stated
# here (no loss/metric math is reimplemented, only the optimizer/batch
# bookkeeping, mirroring how colab_train_encoder.py's own `train()` already
# re-states the encoder's loop shape rather than importing train_encoder.py's
# inline main()).
# ---------------------------------------------------------------------------

def train_decoder_epochs(model, train_feats, epochs, batch_size, lr, max_seconds, t0, log):
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
        random.shuffle(train_feats)
        epoch_loss = 0.0
        epoch_n = 0
        opt.zero_grad()
        for idx, feats in enumerate(train_feats):
            if time.time() - train_start > max_seconds:
                stopped_early = True
                break
            loss = dt.reconstruction_loss(model, feats) / batch_size
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
        log(f"decoder epoch {epoch + 1}/{epochs} done: avg_loss={avg:.4f} (n={epoch_n} records)")
        if stopped_early:
            break
    train_wall = time.time() - train_start
    return loss_curve, train_wall, stopped_early


def evaluate_round_trip_on_gold_tree(model: dt.DecoderTrainedModel, records: list) -> dict:
    """Reconstruction sanity check: decoder alone, fed the GOLD committed
    tree (no encoder in the loop) -- reuses dt.round_trip/reconstruction_
    accuracy verbatim, same metric as scripts/train_decoder.py's own
    evaluate_round_trip."""
    exact_matches, token_f1s = [], []
    for record in records:
        tree = record["lattice"]["trees"][0]
        pred = dt.round_trip(record, tree, model)
        m = dt.reconstruction_accuracy(pred, record["text"])
        exact_matches.append(m["exact_match"])
        token_f1s.append(m["token_f1"])
    n = max(len(records), 1)
    return {"n": len(records), "exact_match": sum(exact_matches) / n, "token_f1": sum(token_f1s) / n}


def evaluate_autoencoder_round_trip(encoder: em.EncoderModel, decoder: dt.DecoderTrainedModel,
                                     records: list, usvs, pos_vocab, hash_buckets: int,
                                     beam_width: int, k: int, log) -> dict:
    """THE metric the lead asked for: text -> encoder.beam_decode -> top
    committed tree -> decoder.realize -> text, end to end, no gold tree
    anywhere in this path."""
    exact_matches, token_f1s = [], []
    n_empty_forest = 0
    examples = []
    for record in records:
        feats = em.build_features(record, usvs, pos_vocab, hash_buckets)
        forest = em.beam_decode(encoder, feats, beam_width=beam_width, k=k, policy="model")
        if not forest:
            n_empty_forest += 1
            pred_text = ""
        else:
            structure = predicted_tree_to_structure(record, forest[0])
            pred_text = " ".join(dt.realize(decoder, structure))
        m = dt.reconstruction_accuracy(pred_text, record["text"])
        exact_matches.append(m["exact_match"])
        token_f1s.append(m["token_f1"])
        if len(examples) < 5:
            examples.append({"gold": record["text"], "pred": pred_text})
    n = max(len(records), 1)
    metrics = {"n": len(records), "exact_match": sum(exact_matches) / n,
               "token_f1": sum(token_f1s) / n, "empty_forest": n_empty_forest, "examples": examples}
    for ex in examples:
        log(f"  round-trip example: gold={ex['gold']!r} pred={ex['pred']!r}")
    return metrics


def no_confab_spot_check(decoder: dt.DecoderTrainedModel, records: list, encoder: em.EncoderModel,
                          usvs, pos_vocab, hash_buckets: int, beam_width: int, k: int,
                          n: int, log) -> dict:
    """Ablation from the decoder_trained module docstring: sever a
    committed structure's content (null every node's surface word, keep
    shape) and confirm realize() abstains (returns []) BEFORE the decoder
    network ever runs -- structural no-confab, not a trained behaviour."""
    checked = 0
    abstained = 0
    for record in records[:n]:
        feats = em.build_features(record, usvs, pos_vocab, hash_buckets)
        forest = em.beam_decode(encoder, feats, beam_width=beam_width, k=k, policy="model")
        structure = (predicted_tree_to_structure(record, forest[0]) if forest
                     else dt.build_structure(record, record["lattice"]["trees"][0]))
        severed = dt.sever_structure_content(structure)
        out = dt.realize(decoder, severed)
        checked += 1
        if out == []:
            abstained += 1
        else:
            log(f"  NO-CONFAB VIOLATION on {record['text']!r}: severed structure produced {out!r}")
    return {"checked": checked, "abstained": abstained, "all_abstained": checked > 0 and checked == abstained}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enc-records", type=int, default=984,
                     help="EN gold records for the encoder split (~80/10/10 train/dev/test); "
                          "984 reproduces the dev/ENCODER_MODEL_SPEC.md S2.3 full-Stage-i split")
    ap.add_argument("--enc-epochs", type=int, default=50,
                     help="~50-60 min end-to-end on this project's CPU dev box at --enc-records 984 "
                          "(see scripts/colab_train_encoder.py)")
    ap.add_argument("--dec-records", type=int, default=984,
                     help="EN gold records for the decoder split (80/20 train/dev, scripts/train_decoder.py)")
    ap.add_argument("--dec-epochs", type=int, default=80,
                     help="decoder is much smaller (48-d GRU) and its per-record features are cheap to "
                          "rebuild (~15-17s/epoch at --dec-records 984 on this project's CPU dev box), "
                          "so 80 epochs (~20-22 min) converges far faster per-epoch than the encoder; "
                          "see the printed decoder training wall-clock to recalibrate")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--outdir", default=str(ROOT / "runs" / "colab_all"))
    ap.add_argument("--gold", default=str(ROOT / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--spanish-gold", default=str(ROOT / "runs" / "spanish_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(ROOT / "data" / "usvs"))
    # Encoder hyperparameters (nsm_ct.encoder_model.EncoderModel / colab_train_encoder.py defaults)
    ap.add_argument("--enc-d-model", type=int, default=128)
    ap.add_argument("--enc-hash-buckets", type=int, default=4096)
    ap.add_argument("--enc-batch-size", type=int, default=32)
    ap.add_argument("--enc-lr", type=float, default=1e-3)
    ap.add_argument("--enc-max-seconds", type=float, default=5400.0)
    # Decoder hyperparameters (nsm_ct.decoder_trained.DecoderTrainedModel / train_decoder.py defaults)
    ap.add_argument("--dec-d-model", type=int, default=48)
    ap.add_argument("--dec-hash-buckets", type=int, default=2048)
    ap.add_argument("--dec-batch-size", type=int, default=16)
    ap.add_argument("--dec-lr", type=float, default=1e-3)
    ap.add_argument("--dec-max-seconds", type=float, default=2400.0)
    # Shared eval knobs
    ap.add_argument("--beam-width", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--round-trip-n", type=int, default=20,
                     help="held-out English sentences for the encoder+decoder autoencoder round-trip test")
    ap.add_argument("--no-confab-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    usvs = load_usvs(args.usvs_dir)
    d_axes = len(usvs.axes)
    log(f"USVS loaded: {len(usvs.sense_ids)} senses, d_axes={d_axes}")

    # =========================================================================
    # PHASE 1: ENCODER (reuses colab_train_encoder.py machinery verbatim)
    # =========================================================================
    log("=" * 72)
    log("PHASE 1: encoder training")
    log("=" * 72)

    log(f"loading English gold from {gold_path}")
    en_records = load_gold(str(gold_path))
    log(f"{len(en_records)} English gold records available")

    n_train, n_dev, n_test = split_sizes(args.enc_records, len(en_records), log)
    enc_train, enc_dev, enc_test = stratified_split(en_records, args.seed, n_train, n_dev, n_test)
    log(f"encoder split: train={len(enc_train)} dev={len(enc_dev)} test={len(enc_test)}")

    pos_vocab = em.build_pos_vocab(en_records)
    role_vocab = em.build_role_vocab(en_records)
    log(f"pos_vocab={len(pos_vocab)} role_vocab={len(role_vocab)}")

    encoder = em.EncoderModel(pos_vocab, role_vocab, d_axes=d_axes, hash_buckets=args.enc_hash_buckets,
                               d_model=args.enc_d_model, controller_hidden=args.enc_d_model)
    encoder.to(device)
    n_enc_params = encoder.num_policy_params()
    log(f"encoder policy params: {n_enc_params:,} (~{n_enc_params * 4 / 1e6:.3f} MB fp32)")

    log(f"building features + teacher-forced derivations for {len(enc_train)} encoder train records")
    enc_train_items = []
    for r in enc_train:
        feats = em.build_features(r, usvs, pos_vocab, args.enc_hash_buckets)
        for tree in r["lattice"]["trees"]:
            steps = em.linearize_tree(r, tree)
            enc_train_items.append((feats, steps))
    log(f"{len(enc_train_items)} teacher-forced derivations, batch_size={args.enc_batch_size}, "
        f"epochs={args.enc_epochs}")

    enc_loss_curve, enc_train_wall, enc_stopped_early = train_encoder_epochs(
        encoder, enc_train_items, args.enc_epochs, args.enc_batch_size, args.enc_lr,
        args.enc_max_seconds, t0, log)
    log(f"encoder training wall-clock: {enc_train_wall:.1f}s (stopped_early={enc_stopped_early})")

    encoder.eval()

    log("evaluating English candidate-set recall (model policy) on held-out test split ...")
    en_model_metrics = em.evaluate(encoder, enc_test, usvs, pos_vocab, args.enc_hash_buckets,
                                    beam_width=args.beam_width, k=args.k, policy="model")
    log(f"English test (model) : {fmt_recall(en_model_metrics)}")

    rng = random.Random(args.seed)
    en_random_metrics = em.evaluate(encoder, enc_test, usvs, pos_vocab, args.enc_hash_buckets,
                                     beam_width=args.beam_width, k=args.k, policy="random", rng=rng)
    log(f"English test (random): {fmt_recall(en_random_metrics)}")

    log(f"loading Spanish gold from {spanish_path} for the grammar-swap eval")
    spanish_records = load_gold(str(spanish_path))
    log(f"{len(spanish_records)} Spanish gold records")

    es_model_metrics = evaluate_with_totals(encoder, spanish_records, usvs, pos_vocab, args.enc_hash_buckets,
                                             args.beam_width, args.k, "model")
    log(f"Spanish (model) : {fmt_recall(es_model_metrics)}")

    es_rng = random.Random(args.seed)
    es_random_metrics = evaluate_with_totals(encoder, spanish_records, usvs, pos_vocab, args.enc_hash_buckets,
                                              args.beam_width, args.k, "random", rng=es_rng)
    log(f"Spanish (random): {fmt_recall(es_random_metrics)}")

    enc_out_path = outdir / "encoder_colab.pt"
    enc_ckpt = {
        "model_state": encoder.to("cpu").state_dict(),
        "pos_vocab": pos_vocab,
        "role_vocab": role_vocab,
        "d_axes": d_axes,
        "hash_buckets": args.enc_hash_buckets,
        "d_model": args.enc_d_model,
        "config": {"n_train": len(enc_train), "n_dev": len(enc_dev), "n_test": len(enc_test),
                   "epochs": args.enc_epochs, "batch_size": args.enc_batch_size, "seed": args.seed,
                   "device": device},
        "loss_curve": enc_loss_curve,
        "metrics": {"english_test": en_model_metrics, "english_test_random": en_random_metrics,
                    "spanish": es_model_metrics, "spanish_random": es_random_metrics},
        "train_wallclock_s": enc_train_wall,
        "n_policy_params": n_enc_params,
    }
    torch.save(enc_ckpt, enc_out_path)
    log(f"saved encoder checkpoint -> {enc_out_path}")

    # =========================================================================
    # PHASE 2: DECODER (reuses nsm_ct.decoder_trained + train_decoder.py machinery)
    # =========================================================================
    log("=" * 72)
    log("PHASE 2: decoder training")
    log("=" * 72)

    dec_records_all = load_decoder_records(str(gold_path))
    n_dec = min(args.dec_records, len(dec_records_all))
    if args.dec_records > len(dec_records_all):
        log(f"NOTE: --dec-records {args.dec_records} exceeds the {len(dec_records_all)} gold records "
            f"available; clamping to {n_dec}.")
    dec_pool = dec_records_all[:]
    random.Random(args.seed).shuffle(dec_pool)
    dec_pool = dec_pool[:n_dec]
    n_dec_train = max(1, int(0.8 * n_dec))
    n_dec_dev = max(1, n_dec - n_dec_train)
    dec_train, dec_dev = split_decoder_records(dec_pool, args.seed, n_dec_train, n_dec_dev)
    log(f"decoder split: train={len(dec_train)} dev={len(dec_dev)} (of {n_dec} records drawn from "
        f"{len(dec_records_all)} available)")

    relation_vocab = em.build_role_vocab(dec_pool)
    function_vocab = dt.build_function_vocab(dec_pool)
    log(f"relation_vocab={len(relation_vocab)} function_vocab={len(function_vocab)}")

    decoder = dt.DecoderTrainedModel(relation_vocab, function_vocab,
                                      hash_buckets=args.dec_hash_buckets, d_model=args.dec_d_model)
    n_dec_params = decoder.num_params()
    log(f"decoder params: {n_dec_params:,} (~{n_dec_params * 4 / 1e6:.3f} MB fp32)")

    log(f"building reconstruction features for {len(dec_train)} decoder train records")
    dec_train_feats = [
        dt.build_decoder_features(r, r["lattice"]["trees"][0], function_vocab, relation_vocab,
                                   args.dec_hash_buckets)
        for r in dec_train
    ]
    log(f"{len(dec_train_feats)} reconstruction items, batch_size={args.dec_batch_size}, "
        f"epochs={args.dec_epochs}")

    dec_loss_curve, dec_train_wall, dec_stopped_early = train_decoder_epochs(
        decoder, dec_train_feats, args.dec_epochs, args.dec_batch_size, args.dec_lr,
        args.dec_max_seconds, t0, log)
    log(f"decoder training wall-clock: {dec_train_wall:.1f}s (stopped_early={dec_stopped_early})")

    decoder.eval()

    log("evaluating reconstruction (decoder alone, gold committed tree) on held-out dev split ...")
    dec_dev_metrics = evaluate_round_trip_on_gold_tree(decoder, dec_dev)
    log(f"decoder dev reconstruction: exact_match={dec_dev_metrics['exact_match']:.3f} "
        f"token_f1={dec_dev_metrics['token_f1']:.3f} (n={dec_dev_metrics['n']})")

    dec_out_path = outdir / "decoder_colab.pt"
    dec_ckpt = {
        "model_state": decoder.state_dict(),
        "relation_vocab": relation_vocab,
        "function_vocab": function_vocab,
        "hash_buckets": args.dec_hash_buckets,
        "d_model": args.dec_d_model,
        "config": {"n_train": len(dec_train), "n_dev": len(dec_dev), "epochs": args.dec_epochs,
                   "batch_size": args.dec_batch_size, "seed": args.seed},
        "loss_curve": dec_loss_curve,
        "dev_reconstruction": dec_dev_metrics,
        "train_wallclock_s": dec_train_wall,
        "n_params": n_dec_params,
    }
    torch.save(dec_ckpt, dec_out_path)
    log(f"saved decoder checkpoint -> {dec_out_path}")

    # =========================================================================
    # PHASE 3: THE AUTOENCODER ROUND-TRIP TEST (encoder + decoder together)
    # =========================================================================
    log("=" * 72)
    log("PHASE 3: round-trip test (encoder.beam_decode -> decoder.realize)")
    log("=" * 72)

    dec_train_texts = {r["text"] for r in dec_train}
    round_trip_pool = [r for r in enc_test if r["text"] not in dec_train_texts] or enc_test
    round_trip_records = round_trip_pool[:args.round_trip_n]
    log(f"round-trip test on {len(round_trip_records)} held-out English sentences "
        f"(encoder test split, not used for decoder training)")
    round_trip_metrics = evaluate_autoencoder_round_trip(
        encoder, decoder, round_trip_records, usvs, pos_vocab, args.enc_hash_buckets,
        args.beam_width, args.k, log)
    log(f"round-trip: exact_match={round_trip_metrics['exact_match']:.3f} "
        f"token_f1={round_trip_metrics['token_f1']:.3f} "
        f"(n={round_trip_metrics['n']}, empty_forest={round_trip_metrics['empty_forest']})")

    # =========================================================================
    # PHASE 4: NO-CONFAB SPOT CHECK
    # =========================================================================
    log("=" * 72)
    log("PHASE 4: no-confab spot check (severed structure -> decoder must abstain)")
    log("=" * 72)
    no_confab = no_confab_spot_check(decoder, round_trip_records, encoder, usvs, pos_vocab,
                                      args.enc_hash_buckets, args.beam_width, args.k,
                                      args.no_confab_n, log)
    log(f"no-confab: {no_confab['abstained']}/{no_confab['checked']} severed structures abstained "
        f"(all_abstained={no_confab['all_abstained']})")

    total_wall = time.time() - t0

    print()
    print("=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)
    print(f"device: {device} (encoder is CPU-only -- see resolve_device()'s note; decoder trained on "
          f"the same device)")
    print()
    print("--- ENCODER ---")
    print(f"records: {len(enc_train)} train / {len(enc_dev)} dev / {len(enc_test)} test "
          f"(of {len(en_records)} EN gold available)  |  epochs: {args.enc_epochs}")
    print(f"policy params: {n_enc_params:,} (~{n_enc_params * 4 / 1e6:.3f} MB fp32)  |  "
          f"training wall-clock: {enc_train_wall:.1f}s")
    print("English (held-out test split):")
    print(f"  model : {fmt_recall(en_model_metrics)}")
    print(f"  random: {fmt_recall(en_random_metrics)}")
    print(f"Spanish grammar-swap ({len(spanish_records)} records, EN-trained weights, zero ES training):")
    print(f"  model : {fmt_recall(es_model_metrics)}")
    print(f"  random: {fmt_recall(es_random_metrics)}")
    print()
    print("--- DECODER ---")
    print(f"records: {len(dec_train)} train / {len(dec_dev)} dev (of {n_dec} drawn, "
          f"{len(dec_records_all)} EN gold available)  |  epochs: {args.dec_epochs}")
    print(f"params: {n_dec_params:,} (~{n_dec_params * 4 / 1e6:.3f} MB fp32)  |  "
          f"training wall-clock: {dec_train_wall:.1f}s")
    print(f"reconstruction (gold committed tree, held-out dev): "
          f"exact_match={dec_dev_metrics['exact_match']:.3f} token_f1={dec_dev_metrics['token_f1']:.3f} "
          f"(n={dec_dev_metrics['n']})")
    print()
    print("--- ROUND-TRIP (encoder+decoder autoencoder, no gold tree in the loop) ---")
    print(f"n={round_trip_metrics['n']} held-out English sentences  |  "
          f"empty_forest={round_trip_metrics['empty_forest']}")
    print(f"  exact_match={round_trip_metrics['exact_match']:.3f}  token_f1={round_trip_metrics['token_f1']:.3f}")
    print()
    print("--- NO-CONFAB SPOT CHECK ---")
    print(f"  {no_confab['abstained']}/{no_confab['checked']} severed structures abstained "
          f"(all_abstained={no_confab['all_abstained']})")
    print()
    print(f"encoder checkpoint: {enc_out_path}")
    print(f"decoder checkpoint: {dec_out_path}")
    print(f"encoder wall-clock: {enc_train_wall:.1f}s  |  decoder wall-clock: {dec_train_wall:.1f}s  |  "
          f"TOTAL wall-clock: {total_wall:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
