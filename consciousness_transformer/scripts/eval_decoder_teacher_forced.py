"""Decoder training-path audit: teacher-forced token accuracy + unreachable-target
rate + free-running exact-match-vs-token-F1 gap.

READ-ONLY analysis script (no training, no edits to decoder_trained.py /
train_decoder.py / colab_train_all.py logic). Reproduces the EXACT train/dev
split that scripts/colab_train_all.py used to produce runs/decoder_colab.pt
(seed=0, dec_records=984, 0.8/0.8 split via train_decoder.split_records),
so "held-out dev" here is the same 197 records the checkpoint's own
dev_reconstruction metric was computed on.

Usage:
    python scripts/eval_decoder_teacher_forced.py --records runs/encoder_gold_v2.jsonl \\
        --ckpt runs/decoder_colab.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from nsm_ct import decoder_trained as dt
from train_decoder import load_records, split_records  # noqa: E402


def reproduce_colab_split(records: list, seed: int = 0, dec_records: int = 984):
    """Mirrors scripts/colab_train_all.py's decoder split exactly."""
    dec_pool = records[:]
    random.Random(seed).shuffle(dec_pool)
    n_dec = min(dec_records, len(dec_pool))
    dec_pool = dec_pool[:n_dec]
    n_dec_train = max(1, int(0.8 * n_dec))
    n_dec_dev = max(1, n_dec - n_dec_train)
    return split_records(dec_pool, seed, n_dec_train, n_dec_dev)


# ---------------------------------------------------------------------------
# PART A: unreachable-target rate
# ---------------------------------------------------------------------------

def unreachable_stats(records: list, function_vocab_set: set) -> dict:
    total = 0
    unreachable = 0
    copy_n = 0
    function_n = 0
    unreachable_examples = Counter()
    for r in records:
        trees = r.get("lattice", {}).get("trees") or []
        if not trees:
            continue
        nodes = dt.extract_nodes(r, trees[0])
        covered = {n.token_index for n in nodes}
        for i, tok in enumerate(r["tokens"]):
            total += 1
            if i in covered:
                copy_n += 1
                continue
            function_n += 1
            if tok.lower() not in function_vocab_set:
                unreachable += 1
                unreachable_examples[tok.lower()] += 1
    return {
        "total_tokens": total,
        "copy_tokens": copy_n,
        "function_tokens": function_n,
        "unreachable_tokens": unreachable,
        "unreachable_rate_overall": unreachable / total if total else 0.0,
        "unreachable_rate_of_function_slots": unreachable / function_n if function_n else 0.0,
        "top_unreachable": unreachable_examples.most_common(15),
    }


# ---------------------------------------------------------------------------
# PART B: teacher-forced next-token accuracy, broken down by target class
# ---------------------------------------------------------------------------

def teacher_forced_eval(model: dt.DecoderTrainedModel, records: list, relation_vocab, function_vocab,
                         hash_buckets: int) -> dict:
    model.eval()
    correct = Counter()
    support = Counter()
    with torch.no_grad():
        for r in records:
            trees = r.get("lattice", {}).get("trees") or []
            if not trees:
                continue
            feats = dt.build_decoder_features(r, trees[0], function_vocab, relation_vocab, hash_buckets)
            node_enc = model.encode_structure(feats.node_word_hash, feats.node_relation_id, feats.node_gtype_id)
            M = node_enc.shape[0]
            eos_label = M + len(function_vocab)
            h = model.init_hidden()
            for t in range(feats.target_labels.shape[0]):
                combined, h = model.decode_step(feats.prev_token_hash[t], h, node_enc)
                pred = int(torch.argmax(combined).item())
                gold = int(feats.target_labels[t].item())
                if gold < M:
                    cls = "COPY"
                elif gold == eos_label:
                    cls = "EOS"
                else:
                    cls = "FUNCTION"
                support[cls] += 1
                support["ALL"] += 1
                if pred == gold:
                    correct[cls] += 1
                    correct["ALL"] += 1
    return {
        "accuracy_overall": correct["ALL"] / support["ALL"] if support["ALL"] else 0.0,
        "support_overall": support["ALL"],
        "by_class": {
            cls: {
                "accuracy": correct[cls] / support[cls] if support[cls] else 0.0,
                "support": support[cls],
            }
            for cls in ("COPY", "FUNCTION", "EOS")
        },
    }


# ---------------------------------------------------------------------------
# PART C: free-running realize() vs gold -- exact-match harshness quantification
# ---------------------------------------------------------------------------

def try_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")  # touch wordnet to fail fast if data missing
        return lem
    except Exception as e:  # pragma: no cover
        print(f"[warn] lemmatizer unavailable ({e}); falling back to lowercase-only", file=sys.stderr)
        return None


def normalize_tokens(toks, lemmatizer):
    if lemmatizer is None:
        return [t.lower() for t in toks]
    return [lemmatizer.lemmatize(t.lower()) for t in toks]


def token_f1(pred_toks, gold_toks) -> float:
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0 or not pred_toks or not gold_toks:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def free_running_eval(model: dt.DecoderTrainedModel, records: list, lemmatizer) -> dict:
    exact = []
    f1_raw = []
    f1_lemma = []
    examples = []
    for r in records:
        trees = r.get("lattice", {}).get("trees") or []
        if not trees:
            continue
        structure = dt.build_structure(r, trees[0])
        pred_toks = dt.realize(model, structure)
        gold_toks = r["tokens"]
        pred_l = [t.lower() for t in pred_toks]
        gold_l = [t.lower() for t in gold_toks]
        exact.append(1.0 if pred_l == gold_l else 0.0)
        f1_raw.append(token_f1(pred_l, gold_l))
        f1_lemma.append(token_f1(normalize_tokens(pred_toks, lemmatizer), normalize_tokens(gold_toks, lemmatizer)))
        examples.append({"gold": " ".join(gold_toks), "pred": " ".join(pred_toks),
                          "exact": exact[-1], "f1_raw": f1_raw[-1], "f1_lemma": f1_lemma[-1]})
    n = max(len(exact), 1)
    return {
        "n": len(exact),
        "exact_match": sum(exact) / n,
        "token_f1_raw": sum(f1_raw) / n,
        "token_f1_lemma": sum(f1_lemma) / n,
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parent.parent / "runs" / "decoder_colab.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dec-records", type=int, default=984)
    ap.add_argument("--tf-n", type=int, default=150, help="dev records for teacher-forced accuracy (Part B)")
    ap.add_argument("--fr-n", type=int, default=30, help="dev records for free-running eval (Part C)")
    ap.add_argument("--dump-n", type=int, default=8, help="gold-vs-realized pairs to print (Part C)")
    args = ap.parse_args()

    records = load_records(args.records)
    print(f"loaded {len(records)} records from {args.records}")

    train_recs, dev_recs = reproduce_colab_split(records, seed=args.seed, dec_records=args.dec_records)
    print(f"reproduced colab_train_all.py decoder split: train={len(train_recs)} dev={len(dev_recs)}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    relation_vocab = ckpt["relation_vocab"]
    function_vocab = ckpt["function_vocab"]
    hash_buckets = ckpt["hash_buckets"]
    d_model = ckpt["d_model"]
    print(f"checkpoint: d_model={d_model} hash_buckets={hash_buckets} "
          f"relation_vocab={len(relation_vocab)} function_vocab={len(function_vocab)} "
          f"config={ckpt.get('config')} n_params={ckpt.get('n_params')}")
    print(f"checkpoint's own dev_reconstruction: {ckpt.get('dev_reconstruction')}")

    model = dt.DecoderTrainedModel(relation_vocab, function_vocab, hash_buckets=hash_buckets, d_model=d_model)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    func_set = set(function_vocab)

    print("\n" + "=" * 78)
    print("PART A -- UNREACHABLE-TARGET RATE")
    print("=" * 78)
    train_unreach = unreachable_stats(train_recs, func_set)
    dev_unreach = unreachable_stats(dev_recs, func_set)
    all_unreach = unreachable_stats(records, func_set)
    for name, stats in (("TRAIN", train_unreach), ("DEV", dev_unreach), ("ALL", all_unreach)):
        print(f"[{name}] total_tokens={stats['total_tokens']} copy={stats['copy_tokens']} "
              f"function_slots={stats['function_tokens']} unreachable={stats['unreachable_tokens']} "
              f"unreachable_rate_overall={stats['unreachable_rate_overall']:.4f} "
              f"unreachable_rate_of_function_slots={stats['unreachable_rate_of_function_slots']:.4f}")
        if stats["top_unreachable"]:
            print(f"    top unreachable tokens: {stats['top_unreachable']}")
    print(f"\nfunction_vocab size = {len(function_vocab)} (built via dt.build_function_vocab over the "
          f"colab_train_all.py `dec_pool`, i.e. train+dev COMBINED before the split -- see note below)")
    print(f"function_vocab sample (30 random, excluding <unk>): "
          f"{random.Random(1).sample([w for w in function_vocab if w != dt.UNK_FUNC], 30)}")

    print("\n" + "=" * 78)
    print("PART B -- TEACHER-FORCED NEXT-TOKEN ACCURACY (held-out dev)")
    print("=" * 78)
    tf_dev = dev_recs[: args.tf_n]
    tf_result = teacher_forced_eval(model, tf_dev, relation_vocab, function_vocab, hash_buckets)
    print(f"records used: {len(tf_dev)}")
    print(f"DEV OVERALL accuracy: {tf_result['accuracy_overall']:.4f} (support={tf_result['support_overall']})")
    for cls, d in tf_result["by_class"].items():
        print(f"  DEV {cls:8s}: accuracy={d['accuracy']:.4f}  support={d['support']}")

    tf_train = train_recs[: args.tf_n]
    tf_train_result = teacher_forced_eval(model, tf_train, relation_vocab, function_vocab, hash_buckets)
    print(f"\n[generalization-gap check] same metric on {len(tf_train)} TRAIN records "
          f"(the model's own training distribution):")
    print(f"TRAIN OVERALL accuracy: {tf_train_result['accuracy_overall']:.4f} "
          f"(support={tf_train_result['support_overall']})")
    for cls, d in tf_train_result["by_class"].items():
        print(f"  TRAIN {cls:8s}: accuracy={d['accuracy']:.4f}  support={d['support']}")
    print(f"\n=> TRAIN-DEV accuracy gap (overall): "
          f"{tf_train_result['accuracy_overall'] - tf_result['accuracy_overall']:.4f}")

    print("\n" + "=" * 78)
    print("PART C -- FREE-RUNNING realize() vs GOLD: exact-match vs token-F1 gap")
    print("=" * 78)
    lemmatizer = try_lemmatizer()
    fr_dev = dev_recs[: args.fr_n]
    fr_result = free_running_eval(model, fr_dev, lemmatizer)
    print(f"records used: {fr_result['n']}")
    print(f"exact_match       = {fr_result['exact_match']:.4f}")
    print(f"token_f1 (raw)    = {fr_result['token_f1_raw']:.4f}")
    print(f"token_f1 (lemma)  = {fr_result['token_f1_lemma']:.4f}")
    print(f"\n{args.dump_n} gold-vs-realized pairs:")
    for i, ex in enumerate(fr_result["examples"][: args.dump_n]):
        print(f"\n  [{i}] exact={ex['exact']:.0f} f1_raw={ex['f1_raw']:.3f} f1_lemma={ex['f1_lemma']:.3f}")
        print(f"      GOLD: {ex['gold']}")
        print(f"      PRED: {ex['pred']}")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
