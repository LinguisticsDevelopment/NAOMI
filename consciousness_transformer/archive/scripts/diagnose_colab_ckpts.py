#!/usr/bin/env python3
"""Diagnose Colab run-1 checkpoints: WHY is encoder structure-recall 0.000 while
sense/slot are 0.93/0.96, and WHAT does the trained decoder drop?

CPU-only eval, NO training. Loads encoder_colab.pt + decoder_colab.pt, reproduces
the exact Colab held-out test split (seed + sizes stored in the ckpt), and
decomposes the all-or-nothing whole-tree metric into edge-level agreement.

Structure-recall (current metric) = whole gold-tree SKELETON present exactly in
the emitted forest. This tool asks the finer questions:
  A. node recall  -- of gold (label, token_index, type) nodes, how many appear
     anywhere in the emitted forest? (grouping-agnostic; sanity vs 0.93 sense)
  B. clause-exact -- of gold CLAUSES (frozenset), how many appear exactly in the
     forest's clause pool? (per-clause version of structure recall)
  C. edges-off histogram -- per gold clause, min symmetric-difference to any
     emitted clause. 0 = exact; 1 = one node off; ... Distinguishes
     "metric-too-strict (mostly 0-1 off)" from "genuinely mis-structured".
  D. predicate token_index agreement, and clause-count (emitted vs gold).
  E. ~15 rendered gold-vs-best-emitted tree examples.
Decoder:
  F. reconstruction-from-gold + round-trip, token-F1 split into CONTENT vs
     FUNCTION words (what is dropped?), + ~15 side-by-side examples.

Usage (cloud routine, CPU):
  git show origin/encoder-gold-v2:consciousness_transformer/runs/encoder_gold_v2.jsonl > runs/encoder_gold_v2.jsonl
  git show origin/colab-trained-ckpts:consciousness_transformer/runs/colab_all/encoder_colab.pt > runs/encoder_colab.pt
  git show origin/colab-trained-ckpts:consciousness_transformer/runs/colab_all/decoder_colab.pt > runs/decoder_colab.pt
  python scripts/diagnose_colab_ckpts.py \
      --gold runs/encoder_gold_v2.jsonl \
      --encoder runs/encoder_colab.pt --decoder runs/decoder_colab.pt \
      --usvs-dir <usvs_dir> --max-records 120 --beam 8 --k 8
"""
import argparse, json, random, sys, os
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import nsm_ct.encoder_model as em
import nsm_ct.decoder_trained as dt
from nsm_ct.ground.usvs import load_usvs
from train_encoder import load_gold, stratified_split

# Reuse the exact internal skeleton helpers the metric uses so our decomposition
# is definitionally consistent with structure_recall.
from nsm_ct.encoder_model import (
    _gold_sites, _clause_skeleton, _tree_skeleton, beam_decode, build_features,
    score_record, aggregate_recall,
)

FUNCTION_WORDS = set("""
a an the this that these those my your his her its our their some any no
i you he she it we they me him us them
is am are was were be been being do does did have has had will would shall should
can could may might must of to in on at by for with from into onto about as
and or but nor so yet if then than because while when where which who whom whose
not n't very too also just only even more most such
""".split())


def render_node(record, node):
    """node = (label, token_index, gtype) -> 'LABEL:word#idx:gtype'."""
    label, tidx, gtype = node
    tok = record["tokens"][tidx] if (tidx is not None and 0 <= tidx < len(record["tokens"])) else "<null>"
    return f"{label}:{tok}#{tidx}:{gtype}"


def render_clause(record, clause_fs):
    return "{" + ", ".join(sorted(render_node(record, n) for n in clause_fs)) + "}"


def min_symdiff(gold_clause, emitted_pool):
    """Smallest symmetric-difference size between a gold clause and any emitted
    clause in the pool (len 0 = exact match)."""
    if not emitted_pool:
        return len(gold_clause)
    return min(len(gold_clause ^ ec) for ec in emitted_pool)


def diagnose_encoder(model, usvs, pos_vocab, hash_buckets, test, beam, k, n_examples):
    print("\n" + "=" * 72)
    print("ENCODER STRUCTURE DIAGNOSIS")
    print("=" * 72)

    node_hits = node_tot = 0
    clause_exact_hits = clause_tot = 0
    tree_exact_hits = tree_tot = 0
    pred_idx_hits = pred_tot = 0
    edges_off_hist = Counter()          # per gold clause: min symdiff bucket
    clause_count_delta = Counter()      # (emitted_best - gold) clause count
    empty_forest = 0
    examples = []

    for rec in test:
        feats = build_features(rec, usvs, pos_vocab, hash_buckets)
        forest = beam_decode(model, feats, beam_width=beam, k=k)
        if not forest:
            empty_forest += 1

        # gold clauses (frozensets) via the metric's own path
        _, _, gold_trees = _gold_sites(rec)
        gold_clauses = [c for t in gold_trees for c in t]

        # emitted clause pool (union over all forest trees)
        emit_pool = []
        for t in forest:
            emit_pool.extend(_tree_skeleton(t))
        emit_pool_set = set(emit_pool)
        emit_nodes = set().union(*emit_pool) if emit_pool else set()

        # A. node recall
        gold_nodes = set().union(*gold_clauses) if gold_clauses else set()
        node_hits += len(gold_nodes & emit_nodes)
        node_tot += len(gold_nodes)

        # B. clause-exact recall
        for gc in gold_clauses:
            clause_tot += 1
            if gc in emit_pool_set:
                clause_exact_hits += 1
            # C. edges-off
            edges_off_hist[min(min_symdiff(gc, emit_pool), 4)] += 1  # cap bucket at 4+
            # D. predicate token_index agreement
            gpred = [n for n in gc if n[0] == "PREDICATE"]
            if gpred:
                pred_tot += 1
                gp = gpred[0]
                if any(n[0] == "PREDICATE" and n[1] == gp[1] for n in emit_nodes):
                    pred_idx_hits += 1

        # whole-tree exact (should reproduce reported structure_recall)
        sc = score_record(rec, forest)
        tree_exact_hits += sc.tree_hits
        tree_tot += sc.tree_total

        # clause-count delta (best emitted tree vs gold tree, first gold tree)
        gold_n = len(gold_trees[0]) if gold_trees else 0
        if forest:
            best = max(forest, key=lambda t: len(set(_tree_skeleton(t)) & set(gold_clauses)))
            clause_count_delta[len(best["clauses"]) - gold_n] += 1

        if len(examples) < n_examples:
            best_tree = None
            if forest:
                best_tree = max(forest, key=lambda t: len(set(_tree_skeleton(t)) & set(gold_clauses)))
            examples.append((rec, gold_clauses, best_tree))

    n = len(test)
    print(f"n_test={n}  empty_forest={empty_forest}  beam={beam} k={k}")
    print(f"[A] node recall (grouping-agnostic) : {node_hits}/{node_tot} = {node_hits/max(node_tot,1):.3f}")
    print(f"[B] clause-exact recall             : {clause_exact_hits}/{clause_tot} = {clause_exact_hits/max(clause_tot,1):.3f}")
    print(f"    whole-tree exact (metric repro) : {tree_exact_hits}/{tree_tot} = {tree_exact_hits/max(tree_tot,1):.3f}")
    print(f"[D] predicate token_index agreement : {pred_idx_hits}/{pred_tot} = {pred_idx_hits/max(pred_tot,1):.3f}")
    print(f"[C] edges-off per gold clause (0=exact match to an emitted clause):")
    tot_c = sum(edges_off_hist.values())
    for b in range(5):
        lbl = f"{b}+" if b == 4 else str(b)
        c = edges_off_hist.get(b, 0)
        print(f"      {lbl} off : {c:4d}  ({100*c/max(tot_c,1):5.1f}%)")
    print(f"    clause-count delta (emitted_best - gold):")
    for d in sorted(clause_count_delta):
        print(f"      {d:+d} : {clause_count_delta[d]}")

    print("\n--- EXAMPLES (gold clauses  vs  best-emitted tree) ---")
    for rec, gc, best in examples:
        print(f"\nTEXT: {rec['text'][:120]}")
        print(f"  GOLD  ({len(gc)} clause):")
        for c in gc:
            print(f"    {render_clause(rec, c)}")
        if best is None:
            print("  EMIT : <empty forest>")
        else:
            print(f"  EMIT  ({len(best['clauses'])} clause, best of forest):")
            for c in _tree_skeleton(best):
                print(f"    {render_clause(rec, c)}")


def token_split_f1(gold_text, pred_tokens):
    gold = gold_text.split()
    pred = list(pred_tokens)
    def f1(gset_toks, pset_toks):
        gc, pc = Counter(gset_toks), Counter(pset_toks)
        overlap = sum((gc & pc).values())
        p = overlap / max(sum(pc.values()), 1)
        r = overlap / max(sum(gc.values()), 1)
        return (2*p*r/(p+r)) if (p+r) else 0.0
    g_content = [t for t in gold if t.lower() not in FUNCTION_WORDS]
    p_content = [t for t in pred if t.lower() not in FUNCTION_WORDS]
    g_func = [t for t in gold if t.lower() in FUNCTION_WORDS]
    p_func = [t for t in pred if t.lower() in FUNCTION_WORDS]
    return f1(gold, pred), f1(g_content, p_content), f1(g_func, p_func)


def diagnose_decoder(dec, usvs, pos_vocab, enc_model, enc_hash, function_vocab,
                     relation_vocab, dec_hash, records, n_examples):
    print("\n" + "=" * 72)
    print("DECODER RECONSTRUCTION DIAGNOSIS")
    print("=" * 72)
    all_f1 = all_c = all_fn = 0.0
    m = 0
    examples = []
    for rec in records:
        tree = rec["lattice"]["trees"][0]
        # reconstruction from GOLD tree (isolates the decoder from encoder error;
        # the encode->decode true round-trip number is already in the ckpt = 0.323)
        rt_text = dt.round_trip(rec, tree, dec)
        f, c, fn = token_split_f1(rec["text"], rt_text.split())
        all_f1 += f; all_c += c; all_fn += fn
        m += 1
        if len(examples) < n_examples:
            examples.append((rec["text"], rt_text))
    m = max(m, 1)
    print(f"n={m}")
    print(f"reconstruction-from-gold token-F1 : all={all_f1/m:.3f}  content={all_c/m:.3f}  function={all_fn/m:.3f}")
    print("  (content>>function => decoder drops function words / morphology / order)")
    print("\n--- EXAMPLES (gold text  vs  realized) ---")
    for g, p in examples:
        print(f"  GOLD: {g[:110]}")
        print(f"  PRED: {p[:110]}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--usvs-dir", required=True)
    ap.add_argument("--max-records", type=int, default=120)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()

    print("loading USVS ...")
    usvs = load_usvs(args.usvs_dir)
    d_axes = usvs.matrix.shape[1] if hasattr(usvs, "matrix") else None

    print("loading encoder ckpt ...")
    enc_ck = torch.load(args.encoder, map_location="cpu", weights_only=False)
    pos_vocab = enc_ck["pos_vocab"]; role_vocab = enc_ck["role_vocab"]
    enc_hash = enc_ck["hash_buckets"]; enc_dmodel = enc_ck["d_model"]
    d_axes = enc_ck.get("d_axes", d_axes)
    enc = em.EncoderModel(pos_vocab, role_vocab, d_axes=d_axes, hash_buckets=enc_hash,
                          d_model=enc_dmodel, controller_hidden=enc_dmodel)
    enc.load_state_dict(enc_ck["model_state"]); enc.eval()
    cfg = enc_ck["config"]
    print(f"  encoder cfg: {cfg}")
    print(f"  reported metrics: {json.dumps(enc_ck.get('metrics', {}).get('english_test', {}))}")

    print("loading decoder ckpt ...")
    dec_ck = torch.load(args.decoder, map_location="cpu", weights_only=False)
    dec = dt.DecoderTrainedModel(dec_ck["relation_vocab"], dec_ck["function_vocab"],
                                 hash_buckets=dec_ck["hash_buckets"], d_model=dec_ck["d_model"])
    dec.load_state_dict(dec_ck["model_state"]); dec.eval()

    # Reproduce the EXACT Colab test split from stored seed + sizes.
    records = load_gold(args.gold)
    seed = cfg["seed"]; n_tr = cfg["n_train"]; n_dv = cfg["n_dev"]; n_te = cfg["n_test"]
    _, _, test = stratified_split(records, seed, n_tr, n_dv, n_te)
    print(f"reproduced split: train={n_tr} dev={n_dv} test={len(test)} (seed={seed})")
    if args.max_records and len(test) > args.max_records:
        test = test[:args.max_records]
        print(f"  capped test to {len(test)} for CPU eval")

    diagnose_encoder(enc, usvs, pos_vocab, enc_hash, test, args.beam, args.k, args.examples)
    diagnose_decoder(dec, usvs, pos_vocab, enc, enc_hash, dec_ck["function_vocab"],
                     dec_ck["relation_vocab"], dec_ck["hash_buckets"], test, args.examples)
    print("\nDONE.")


if __name__ == "__main__":
    main()
