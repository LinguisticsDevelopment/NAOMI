"""Re-score EVERY arms-1/arms-2 checkpoint with the USVS-graded metric next
to the old binary edge-F1 -- no retraining (lead directive 2026-09-08, part
C; dev/USVS_GRADED_SCORING.md).

The question the lead asked: does the ARM RANKING change when a near-miss
stops scoring like a phantom, and how far do the old "wrong" edges actually
sit in USVS space?

Every checkpoint is decoded ONCE per target set on the shared 98-sentence
test holdout (`runs/holdout_sentences.txt`, the same sentences every arm was
scored on in arms-1/2), and both metrics are computed on the SAME forests --
so the edge column here is directly comparable to `runs/arms/summary.tsv`
and the graded column cannot differ because of a different decode.

Also runs the three sanity controls the directive names, on the gold itself:
identity (must be 1.0), a corrupted copy (SUBJECT/OBJECT swap, predicate to a
non-verb, one phantom node), and a different sentence's tree (the ambient
USVS floor -- the number the arm column must be read against, NOT 0).

Usage:
    python scripts/rescore_graded_arms.py | tee runs/rescore_graded.txt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from nsm_ct import encoder_model as em
from nsm_ct import encoder_train_util as etu
from nsm_ct import usvs_graded as ug
from nsm_ct.ground.usvs import load_usvs
from nsm_ct.tree_render import normalize_gold_tree

ROOT = Path(__file__).resolve().parent.parent


def _fmt(v: float) -> str:
    return "  n/a " if v != v else f"{v:.3f}"


def sanity_controls(records, usvs, n: int = 40) -> list:
    """The three controls, averaged over the first `n` holdout records."""
    ident, corrupt, random_other = [], [], []
    usable = [r for r in records if r["lattice"]["trees"]][:n]
    for i, rec in enumerate(usable):
        gold = rec["lattice"]["trees"][0]
        ident.append(ug.usvs_tree_similarity(
            normalize_gold_tree(rec, gold), gold, rec, usvs, gold_is_lattice=True)["graded_f"])
        corrupt.append(ug.usvs_tree_similarity(
            ug.corrupt_tree(rec, gold), gold, rec, usvs, gold_is_lattice=True)["graded_f"])
        other = usable[(i + 7) % len(usable)]
        if other is rec:
            continue
        random_other.append(ug.usvs_tree_similarity(
            normalize_gold_tree(other, other["lattice"]["trees"][0]), gold, rec, usvs,
            gold_is_lattice=True, pred_record=other)["graded_f"])

    def mean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    return [("gold vs itself (must be 1.000)", mean(ident), len(ident)),
            ("gold vs corrupted copy", mean(corrupt), len(corrupt)),
            ("gold vs a random other sentence's tree", mean(random_other), len(random_other))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-dir", default=str(ROOT / "runs" / "arms"))
    ap.add_argument("--holdout-file", default=str(ROOT / "runs" / "holdout_sentences.txt"))
    ap.add_argument("--gold-v2", default=str(ROOT / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--gold-v4b", default=str(ROOT / "runs" / "encoder_gold_v4b.jsonl"))
    ap.add_argument("--usvs-dir", default=str(ROOT / "data" / "usvs"))
    ap.add_argument("--beam-width", type=int, default=6)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--only-best", action="store_true", default=True,
                     help="score only the kept-best checkpoint (<arm>.pt), skipping the "
                          "<arm>.pt.last.pt siblings. Default on: 'kept-best where present'.")
    ap.add_argument("--include-last", dest="only_best", action="store_false")
    args = ap.parse_args()

    t0 = time.time()
    usvs = load_usvs(args.usvs_dir)
    holdout = etu.load_holdout_sentences(args.holdout_file)
    gold_v2 = etu.load_gold(args.gold_v2)
    gold_v4b = etu.load_gold(args.gold_v4b)
    targets = {}
    for name, pool in (("v2", gold_v2), ("v4b", gold_v4b)):
        recs, missing = etu.records_for_sentences(pool, holdout)
        targets[name] = recs
        print(f"# targets[{name}]: matched={len(recs)} missing={missing} "
              f"of {len(holdout)} holdout sentences")

    print("\n" + "=" * 100)
    print("SANITY CONTROLS (on v4b gold targets; graded_F)")
    print("=" * 100)
    for label, val, n in sanity_controls(targets["v4b"], usvs):
        print(f"  {label:<42} {_fmt(val)}   (n={n})")
    print("  NOTE: the random-tree number is the ambient USVS floor -- USVS sense signatures")
    print("        are non-negative, so unrelated senses have a positive cosine. Read every")
    print("        graded_F below AGAINST that floor, never against 0.")

    ckpts = sorted(glob.glob(os.path.join(args.arms_dir, "*.pt")))
    if args.only_best:
        ckpts = [c for c in ckpts if not c.endswith(".last.pt")]

    rows = []
    for path in ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = em.EncoderModel(ckpt["pos_vocab"], ckpt["role_vocab"], d_axes=ckpt["d_axes"],
                                 hash_buckets=ckpt["hash_buckets"], d_model=ckpt["d_model"],
                                 controller_hidden=ckpt["d_model"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        arm = Path(path).name[: -len(".pt")]
        kind = "kept-best" if cfg.get("keep_best") else "final"
        for tname in ("v2", "v4b"):
            m = etu.evaluate_full(model, targets[tname], usvs, ckpt["pos_vocab"],
                                   ckpt["hash_buckets"], beam_width=args.beam_width,
                                   k=args.k, policy="model", metric="graded")
            rows.append({
                "arm": arm, "kind": kind, "targets": tname,
                "best_step": cfg.get("best_step"),
                "edge_f1": m["rank1_edge_f1"],
                "edge_p": m["rank1_edge_precision"], "edge_r": m["rank1_edge_recall"],
                "graded_f": m["rank1_graded_f"], "graded_p": m["rank1_graded_p"],
                "graded_r": m["rank1_graded_r"],
                "graded_clause": m["rank1_clause_struct"],
                "best_of_k_edge_f1": etu._f1(m["edge_precision"], m["edge_recall"]),
                "best_of_k_graded_f": m["graded_f"],
                "forest_width": m["mean_forest_width"],
            })
            print(f"[{time.time()-t0:7.1f}s] {arm:<28} {kind:<9} {tname:<4} "
                  f"edge_F1={_fmt(m['rank1_edge_f1'])} graded_F={_fmt(m['rank1_graded_f'])}")

    for tname in ("v2", "v4b"):
        sub = [r for r in rows if r["targets"] == tname]
        print("\n" + "=" * 100)
        print(f"RANK-1 (COMMITTED TREE) RE-SCORE on {tname} targets, n={len(targets[tname])} "
              f"holdout sentences")
        print("=" * 100)
        print("| arm                          | ckpt      | edge-F1 | graded-F | graded-P | "
              "graded-R | clause | width |")
        print("|------------------------------|-----------|---------|----------|----------|"
              "----------|--------|-------|")
        for r in sorted(sub, key=lambda r: -(r["graded_f"] if r["graded_f"] == r["graded_f"] else -1)):
            print(f"| {r['arm']:<28} | {r['kind']:<9} | {_fmt(r['edge_f1'])}   | "
                  f"{_fmt(r['graded_f'])}    | {_fmt(r['graded_p'])}    | {_fmt(r['graded_r'])}    | "
                  f"{_fmt(r['graded_clause'])}  | {r['forest_width']:.2f}  |")

        edge_rank = [r["arm"] for r in sorted(sub, key=lambda r: -r["edge_f1"])]
        graded_rank = [r["arm"] for r in sorted(sub, key=lambda r: -r["graded_f"])]
        print(f"\n  ranking by edge-F1  : {' > '.join(edge_rank)}")
        print(f"  ranking by graded-F : {' > '.join(graded_rank)}")
        print(f"  RANKING CHANGED: {'YES' if edge_rank != graded_rank else 'NO'}"
              f"  (top arm: edge={edge_rank[0]}  graded={graded_rank[0]})")

    print("\n" + "=" * 100)
    print("BEST-OF-K (ORACLE) VIEW, for continuity with runs/arms/summary.tsv")
    print("=" * 100)
    print("| arm                          | targets | best-of-k edge-F1 | best-of-k graded-F |")
    print("|------------------------------|---------|-------------------|--------------------|")
    for r in rows:
        print(f"| {r['arm']:<28} | {r['targets']:<7} | {_fmt(r['best_of_k_edge_f1'])}             | "
              f"{_fmt(r['best_of_k_graded_f'])}              |")

    print(f"\n[{time.time()-t0:.1f}s] done ({len(rows)} arm x target-set re-scores)")


if __name__ == "__main__":
    main()
