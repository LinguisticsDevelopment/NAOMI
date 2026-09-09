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
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from nsm_ct import usvs_graded as ug
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
    ap.add_argument("--metric", choices=("edge", "graded", "both"), default="edge",
                     help="(lead directive 2026-09-08; dev/USVS_GRADED_SCORING.md) 'edge' = "
                          "the original binary edge-F1 report only. 'graded'/'both' ALSO print "
                          "the USVS-graded P/R/F table, computed on the SAME decoded forests "
                          "(nothing is re-decoded, no existing number changes).")
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

    part_a_report(model, recs, usvs, pos_vocab, ckpt, results["model"], args)


# ---------------------------------------------------------------------------
# PART A (audit 2026-09-08, action 2) -- does the best-of-k oracle number
# survive if the model has to COMMIT to the single tree it would actually
# emit, instead of being scored by whichever of up to k candidate trees
# happens to overlap gold best? `evaluate(..., policy="model")` above is
# already the best-of-k oracle view (`_best_tree_overlap` picks the
# max-overlap tree out of the forest); this section re-scores the SAME
# decoded records against (1) only the forest's rank-1 (highest-logprob)
# tree, and (2) a forest independently decoded with k=1, plus forest-width
# and forest-level (pooled-across-the-whole-forest) over-generation, which
# `overgen_ratio` above does NOT capture since it only measures the size of
# whichever single tree `_best_tree_overlap` selected.
# ---------------------------------------------------------------------------

# The ledger's (dev/CURRENT_STATE.md, dev/AUDIT_2026-09-08.md) previously
# reported best-of-8 numbers for THIS checkpoint (run-2, post aliasing-fix).
# NOTE: the `rescore_prev.txt`/`rescore_output.txt` file fetched by the
# setup step is a STALE re-score of the run-1 checkpoint (commit 921f8af,
# before the beam_decode aliasing fix) -- it is not comparable to run-2 and
# is not used for the sanity check below.
PREV_BEST8_EDGE_PRECISION = 0.70
PREV_BEST8_OVERGEN_RATIO = 1.08
PREV_BEST8_STRUCTURE_RECALL = 0.212
_REPRO_TOL = 0.02


def _f1(p: float, r: float) -> float:
    if p != p or r != r or (p + r) == 0:
        return float("nan")
    return 2 * p * r / (p + r)


def _nanmean(vals):
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _forest_level_overgen(record: dict, forest: list) -> tuple:
    """(pooled_edge_count, gold_edge_count, forest_overgen_ratio) -- the
    TOTAL edges emitted across every tree in the forest (deduped, union'd),
    vs gold, as opposed to `score_record`'s best-SINGLE-tree overgen."""
    _sense, _slots, gold_trees = em._gold_sites(record)
    edge_sets = [em._tree_edge_set(em._tree_skeleton(t)) for t in forest]
    pooled = frozenset().union(*edge_sets) if edge_sets else frozenset()
    overgens = []
    gold_edge_total = 0
    for gold_sk in gold_trees:
        ge = em._tree_edge_set(gold_sk)
        gold_edge_total += len(ge)
        if ge:
            overgens.append(len(pooled) / len(ge))
    return len(pooled), gold_edge_total, (sum(overgens) / len(overgens) if overgens else float("nan"))


def part_a_report(model, recs, usvs, pos_vocab, ckpt, best8_agg, args) -> None:
    lines = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("\n" + "=" * 78)
    emit(f"PART A: RANK-1 COMMITTED-TREE RE-SCORE (best-of-{args.k} oracle vs commitment)")
    emit("=" * 78)

    want_graded = args.metric in ("graded", "both")
    rank1_scores, k1_scores = [], []
    graded_best, graded_rank1, graded_k1 = [], [], []
    widths, pooled_edges_list, gold_edges_list, forest_overgens = [], [], [], []
    rank1_vs_k1_mismatches = 0
    for record in recs:
        feats = em.build_features(record, usvs, pos_vocab, ckpt["hash_buckets"])
        forest8 = em.beam_decode(model, feats, beam_width=args.beam_width, k=args.k, policy="model")
        forest1 = em.beam_decode(model, feats, beam_width=args.beam_width, k=1, policy="model")
        top1_of_8 = [forest8[0]] if forest8 else []

        sk_top1 = em._tree_skeleton(top1_of_8[0]) if top1_of_8 else None
        sk_k1 = em._tree_skeleton(forest1[0]) if forest1 else None
        if sk_top1 != sk_k1:
            rank1_vs_k1_mismatches += 1

        rank1_scores.append(em.score_record(record, top1_of_8))
        k1_scores.append(em.score_record(record, forest1))
        if want_graded:
            graded_best.append(ug.score_record_graded(record, forest8, usvs))
            graded_rank1.append(ug.score_record_graded(record, top1_of_8, usvs))
            graded_k1.append(ug.score_record_graded(record, forest1, usvs))
        widths.append(len(forest8))
        pooled_n, gold_n, forest_over = _forest_level_overgen(record, forest8)
        pooled_edges_list.append(pooled_n)
        gold_edges_list.append(gold_n)
        forest_overgens.append(forest_over)

    rank1_agg = em.aggregate_recall(rank1_scores)
    k1_agg = em.aggregate_recall(k1_scores)

    emit(f"\nsanity check: rank-1-vs-k=1 tree mismatches = {rank1_vs_k1_mismatches}/{len(recs)} "
         "(should be 0 -- k only truncates beam_decode's already-sorted output)")

    emit("\nreproduction check vs ledger's previously reported best-of-{} numbers "
         "(dev/CURRENT_STATE.md, dev/AUDIT_2026-09-08.md; NOT the stale "
         "runs/rescore_prev.txt, which re-scored the pre-aliasing-fix run-1 "
         "checkpoint):".format(args.k))
    for name, prev, now in (
        ("edge_precision", PREV_BEST8_EDGE_PRECISION, best8_agg["edge_precision"]),
        ("overgen_ratio", PREV_BEST8_OVERGEN_RATIO, best8_agg["overgen_ratio"]),
        ("structure_recall", PREV_BEST8_STRUCTURE_RECALL, best8_agg["structure_recall"]),
    ):
        ok = abs(prev - now) <= _REPRO_TOL
        emit(f"  {name}: prev={prev:.3f}  now={now:.3f}  "
             f"{'MATCH' if ok else 'MISMATCH'} (tol={_REPRO_TOL})")

    rows = [
        (f"best-of-{args.k} (oracle)", best8_agg["edge_precision"], best8_agg["edge_recall"],
         _f1(best8_agg["edge_precision"], best8_agg["edge_recall"]), best8_agg["structure_recall"]),
        (f"rank-1 (top of {args.k}-forest)", rank1_agg["edge_precision"], rank1_agg["edge_recall"],
         _f1(rank1_agg["edge_precision"], rank1_agg["edge_recall"]), rank1_agg["structure_recall"]),
        ("k=1 (beam_decode k=1)", k1_agg["edge_precision"], k1_agg["edge_recall"],
         _f1(k1_agg["edge_precision"], k1_agg["edge_recall"]), k1_agg["structure_recall"]),
    ]

    emit("\n| decode view                  | edge_P | edge_R | edge_F1 | structure_exact |")
    emit("|-------------------------------|--------|--------|---------|-----------------|")
    for name, p, r, f1, sx in rows:
        emit(f"| {name:<29} | {p:.3f}  | {r:.3f}  | {f1:.3f}   | {sx:.3f}           |")

    if want_graded:
        emit("\n" + "-" * 78)
        emit("USVS-GRADED re-score of the SAME decoded forests "
             "(dev/USVS_GRADED_SCORING.md; graded_F is NOT comparable in LEVEL to edge_F1 "
             "-- read it against the random-tree floor, see the design doc S2.1)")
        emit("-" * 78)
        emit("| decode view                  | graded_P | graded_R | graded_F | clause_struct | overall |")
        emit("|-------------------------------|----------|----------|----------|---------------|---------|")
        for name, sc in ((f"best-of-{args.k} (oracle)", graded_best),
                          (f"rank-1 (top of {args.k}-forest)", graded_rank1),
                          ("k=1 (beam_decode k=1)", graded_k1)):
            g = ug.aggregate_graded(sc)
            emit(f"| {name:<29} | {g['graded_p']:.3f}    | {g['graded_r']:.3f}    | "
                 f"{g['graded_f']:.3f}    | {g['clause_struct']:.3f}         | "
                 f"{g['graded_overall']:.3f}   |")

    mean_width = statistics.mean(widths) if widths else float("nan")
    median_width = statistics.median(widths) if widths else float("nan")
    total_pooled = sum(pooled_edges_list)
    total_gold = sum(gold_edges_list)
    mean_forest_overgen = _nanmean(forest_overgens)

    emit(f"\nforest width (trees emitted per sentence, n={len(recs)}): "
         f"mean={mean_width:.2f}  median={median_width:.1f}  "
         f"min={min(widths)}  max={max(widths)}")
    emit(f"forest-level over-generation (ALL forest trees pooled, deduped edges, "
         f"vs gold -- distinct from the best-SINGLE-tree overgen_ratio above): "
         f"total emitted edges={total_pooled}  total gold edges={total_gold}  "
         f"ratio={total_pooled / total_gold if total_gold else float('nan'):.3f}  "
         f"(mean per-record ratio={mean_forest_overgen:.3f})")

    rank1_f1 = _f1(rank1_agg["edge_precision"], rank1_agg["edge_recall"])
    emit(f"\nGATE (audit): rank-1 edge-F1 >= 0.55 -> 'encoder works' verdict survives commitment.")
    if rank1_f1 != rank1_f1:
        emit("  rank-1 edge-F1 is NaN (no gold edges in split?) -- gate INCONCLUSIVE.")
    elif rank1_f1 >= 0.55:
        emit(f"  rank-1 edge-F1 = {rank1_f1:.3f} >= 0.55 -> PASSES. Commitment survives.")
    else:
        emit(f"  rank-1 edge-F1 = {rank1_f1:.3f} < 0.55 -> FAILS. The best-of-{args.k} "
             "headline does NOT survive commitment to the model's own top hypothesis.")

    out_path = Path(__file__).resolve().parent.parent / "runs" / "rescore_rank1.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote -> {out_path}")


if __name__ == "__main__":
    main()
