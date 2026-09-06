"""Confirmation-run diagnostic (mask-fix, Step 4): decompose over-generation.

Under the LOOSE mask (strict_ground=False, i.e. current default decode
behavior), decode the reproduced test split with policy=model and, over
each record's emitted "best tree" (the eval-view forest tree matched to
each gold tree, same selection `score_record`/`_best_tree_overlap` use),
count nodes with `token_index=None` broken down by grounding type (sense /
entity / prime / reference / elision), as a fraction of all emitted nodes
in those best trees.

This tells us how much of the ~8x over-generation is GROUND (sense/entity
nulls -- exactly what `strict_ground` targets) vs EMIT_SYNTH_SLOT (prime
nulls, always token_index=None by construction) vs EMIT_UNRESOLVED_SLOT
(reference/elision nulls -- NOT touched by `strict_ground`).

Usage:
    python scripts/spam_composition.py --checkpoint runs/encoder_colab.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from train_encoder import load_gold, stratified_split  # noqa: E402


def _best_matching_tree(gold_sk, forest, forest_edge_sets):
    """Index of the forest tree with max edge-overlap to `gold_sk` (same
    selection as `em._best_tree_overlap`, but returning the tree itself)."""
    gold_edges = em._tree_edge_set(gold_sk)
    best_idx, best_overlap, best_size = None, -1, None
    for idx, edges in enumerate(forest_edge_sets):
        overlap = len(gold_edges & edges)
        size = len(edges)
        if best_idx is None or overlap > best_overlap or (overlap == best_overlap and size < best_size):
            best_idx, best_overlap, best_size = idx, overlap, size
    return best_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_colab.pt"))
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "usvs"))
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--split", choices=("train", "dev", "test"), default="test")
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

    records = load_gold(args.gold)
    train_recs, dev_recs, test_recs = stratified_split(records, cfg["seed"], cfg["n_train"],
                                                         cfg["n_dev"], cfg["n_test"])
    recs = {"train": train_recs, "dev": dev_recs, "test": test_recs}[args.split]
    print(f"reproduced split: train={len(train_recs)} dev={len(dev_recs)} test={len(test_recs)} (seed={cfg['seed']})")
    print(f"decoding split={args.split} (n={len(recs)}) under LOOSE mask (strict_ground=False), policy=model")

    usvs = load_usvs(args.usvs_dir)

    null_by_gtype = Counter()
    total_nodes = 0
    total_null = 0
    n_best_trees = 0

    for record in recs:
        feats = em.build_features(record, usvs, pos_vocab, ckpt["hash_buckets"])
        forest = em.beam_decode(model, feats, beam_width=args.beam_width, k=args.k,
                                 policy="model", strict_ground=False)
        if not forest:
            continue
        _, _, gold_trees = em._gold_sites(record)
        forest_sk = [em._tree_skeleton(t) for t in forest]
        forest_edge_sets = [em._tree_edge_set(sk) for sk in forest_sk]

        matched_idxs = set()
        for gold_sk in gold_trees:
            idx = _best_matching_tree(gold_sk, forest, forest_edge_sets)
            if idx is not None:
                matched_idxs.add(idx)

        for idx in matched_idxs:
            tree = forest[idx]
            n_best_trees += 1
            for clause in tree["clauses"]:
                nodes = [clause["predicate"]] + clause["roles"]
                for node in nodes:
                    total_nodes += 1
                    if node["token_index"] is None:
                        total_null += 1
                        gtype = node["grounding"]["type"] or "entity"
                        null_by_gtype[gtype] += 1

    print(f"\nbest trees examined: {n_best_trees}")
    print(f"total emitted nodes (in best trees): {total_nodes}")
    print(f"total null (token_index=None) nodes: {total_null}  "
          f"({total_null / total_nodes:.3f} of all emitted nodes)" if total_nodes else "")

    print("\nnull-node composition by grounding type (fraction of ALL emitted nodes):")
    for gtype in ("sense", "entity", "prime", "reference", "elision"):
        n = null_by_gtype.get(gtype, 0)
        frac = n / total_nodes if total_nodes else 0.0
        print(f"  {gtype:10s}  count={n:5d}  frac_of_all_nodes={frac:.4f}")
    other = set(null_by_gtype) - {"sense", "entity", "prime", "reference", "elision"}
    for gtype in sorted(other):
        n = null_by_gtype[gtype]
        frac = n / total_nodes if total_nodes else 0.0
        print(f"  {gtype:10s}  count={n:5d}  frac_of_all_nodes={frac:.4f}  (unexpected gtype)")

    ground_null = null_by_gtype.get("sense", 0) + null_by_gtype.get("entity", 0)
    synth_null = null_by_gtype.get("prime", 0)
    unresolved_null = null_by_gtype.get("reference", 0) + null_by_gtype.get("elision", 0)
    print("\nby producing action (fraction of ALL emitted nodes):")
    print(f"  GROUND nulls (sense+entity, killed by strict_ground)          : "
          f"count={ground_null:5d}  frac={ground_null / total_nodes:.4f}" if total_nodes else "")
    print(f"  EMIT_SYNTH_SLOT nulls (prime, NOT touched by strict_ground)   : "
          f"count={synth_null:5d}  frac={synth_null / total_nodes:.4f}" if total_nodes else "")
    print(f"  EMIT_UNRESOLVED_SLOT nulls (reference+elision, NOT touched)  : "
          f"count={unresolved_null:5d}  frac={unresolved_null / total_nodes:.4f}" if total_nodes else "")


if __name__ == "__main__":
    main()
