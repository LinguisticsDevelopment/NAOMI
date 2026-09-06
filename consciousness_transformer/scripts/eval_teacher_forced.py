"""Teacher-forced next-action accuracy for the candidate-lattice encoder --
the training-path's OWN report card, and an UPPER BOUND on decode quality.

This is a deliberately independent check from `scripts/rescore_encoder.py`:
it NEVER calls `em.beam_decode` (the function the 2026-09-06 aliasing bug
lived in). Instead it walks the SAME gold oracle action sequence
(`em.linearize_tree`) `em.teacher_force_loss` trains on, feeding the model
the gold prefix at every step (never its own prediction), and records
whether `argmax(masked action-type logits) == gold action`. If teacher-forced
accuracy is high, the training/loss path learned the right thing regardless
of whatever the (now-fixed) decode-side bug did to eval numbers; if it's low,
the bug is in loss/training, not decode.

Also reports, at the steps where the head applies, the same treatment for
the typed sub-decisions (role @ GROUND/EMIT_*, grounding-type, source) --
mirroring exactly which sub-losses `em.teacher_force_loss` computes at each
step (see that function's step loop), so "does this head fire here" is never
guessed at separately from the training code it audits.

Usage:
    python scripts/eval_teacher_forced.py --checkpoint runs/encoder_colab.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from nsm_ct.ground.usvs import load_usvs
from nsm_ct import encoder_model as em
from train_encoder import load_gold, stratified_split  # noqa: E402


def teacher_forced_predict(model: em.EncoderModel, feats: em.SentenceFeatures,
                            steps, counters: dict) -> None:
    """One derivation's teacher-forced pass -- control flow copied verbatim
    from `em.teacher_force_loss` (same state transitions, same masking, same
    per-step sub-head gating), with CE replaced by argmax-vs-gold bookkeeping
    into `counters`. Never touches `em.beam_decode`."""
    enc = model.encode(feats)
    T = enc.shape[0]
    h = model.init_controller_state()
    open_clause = False
    open_kind_id = model._none_clause_id
    prev_action_id = model._start_action_id
    i = 0
    has_clause = False

    for step in steps:
        i_clamped = min(i, T - 1) if T > 0 else 0
        enc_i = enc[i_clamped] if T > 0 else torch.zeros(model.d_model)
        h = model.controller_step(enc_i, open_kind_id, prev_action_id, h)

        legal = em.legal_action_types(open_clause, i, T, has_clause)
        mask = em._mask_vector(legal)
        type_logits = model.action_type_head(h).squeeze(0) + mask
        pred_type = int(torch.argmax(type_logits))
        gold_type = em.ACTION_INDEX[step.action]

        counters["overall_total"] += 1
        counters["overall_hits"] += int(pred_type == gold_type)
        counters[f"type_total::{step.action}"] += 1
        counters[f"type_hits::{step.action}"] += int(pred_type == gold_type)

        if step.action == "OPEN_CLAUSE":
            kind_logits = model.kind_head(h).squeeze(0)
            pred_kind = int(torch.argmax(kind_logits))
            gold_kind = em.KIND_INDEX.get(step.kind, 0)
            counters["kind_total"] += 1
            counters["kind_hits"] += int(pred_kind == gold_kind)
            open_clause = True
            open_kind_id = gold_kind
        elif step.action == "CLOSE_CLAUSE":
            open_clause = False
            open_kind_id = model._none_clause_id
            has_clause = True
        elif step.action in ("GROUND", "ATTACH", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT"):
            role_logits = model.role_head(h).squeeze(0)
            pred_role = int(torch.argmax(role_logits))
            gold_role = model.role_id(step.role)
            counters["role_total"] += 1
            counters["role_hits"] += int(pred_role == gold_role)

        if step.action in ("GROUND", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT") and step.gtype is not None:
            gtype_logits = model.gtype_head(h).squeeze(0)
            pred_gtype = int(torch.argmax(gtype_logits))
            gold_gtype = em.GTYPE_INDEX[step.gtype]
            counters["gtype_total"] += 1
            counters["gtype_hits"] += int(pred_gtype == gold_gtype)

        if step.gtype in ("sense", "reference", "elision") and step.source is not None:
            source_logits = model.source_head(h).squeeze(0)
            pred_source = int(torch.argmax(source_logits))
            gold_source = em.SOURCE_INDEX.get(step.source, 0)
            counters["source_total"] += 1
            counters["source_hits"] += int(pred_source == gold_source)

        if step.action == "EMIT_SYNTH_SLOT" and step.prime is not None:
            prime_logits = model.prime_head(h).squeeze(0)
            pred_prime = int(torch.argmax(prime_logits))
            gold_prime = em.PRIME_INDEX.get(step.prime, em.PRIME_INDEX["<UNK_PRIME>"])
            counters["prime_total"] += 1
            counters["prime_hits"] += int(pred_prime == gold_prime)

        if step.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT") and step.token_index is not None:
            i = step.token_index + 1
        prev_action_id = em.ACTION_INDEX[step.action]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_colab.pt"))
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--usvs-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "usvs"))
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
    print(f"policy params: {model.num_policy_params():,}")
    print(f"checkpoint config: {cfg}")

    records = load_gold(args.gold)
    train_recs, dev_recs, test_recs = stratified_split(records, cfg["seed"], cfg["n_train"],
                                                         cfg["n_dev"], cfg["n_test"])
    splits = {"train": train_recs, "dev": dev_recs, "test": test_recs}
    recs = splits[args.split]
    print(f"reproduced split: train={len(train_recs)} dev={len(dev_recs)} test={len(test_recs)} (seed={cfg['seed']})")
    print(f"teacher-forcing split={args.split} (n={len(recs)} records)")

    usvs = load_usvs(args.usvs_dir)

    counters = defaultdict(int)
    n_derivations = 0
    with torch.no_grad():
        for r in recs:
            feats = em.build_features(r, usvs, pos_vocab, ckpt["hash_buckets"])
            for tree in r["lattice"]["trees"]:
                steps = em.linearize_tree(r, tree)
                teacher_forced_predict(model, feats, steps, counters)
                n_derivations += 1

    print(f"\nteacher-forced derivations scored: {n_derivations} (from {len(recs)} records, "
          f"forest-expanded -- one derivation per gold tree)")

    overall_acc = counters["overall_hits"] / counters["overall_total"] if counters["overall_total"] else float("nan")
    print("\n" + "=" * 78)
    print("TEACHER-FORCED NEXT-ACTION ACCURACY (never calls beam_decode)")
    print("=" * 78)
    print(f"OVERALL micro accuracy: {overall_acc:.4f}  (n={counters['overall_total']} steps)")

    print(f"\n{'action type':<22}{'accuracy':>10}{'support':>10}")
    print("-" * 42)
    per_action = {}
    for a in em.ACTION_TYPES:
        tot = counters[f"type_total::{a}"]
        hits = counters[f"type_hits::{a}"]
        acc = hits / tot if tot else float("nan")
        per_action[a] = {"accuracy": acc, "support": tot}
        print(f"{a:<22}{acc:>10.4f}{tot:>10d}")

    print("\n" + "-" * 42)
    print("TERMINAL ACTIONS (the ones over-attachment/aliasing would hide):")
    for a in ("CLOSE_CLAUSE", "STOP"):
        print(f"  {a:<20} accuracy={per_action[a]['accuracy']:.4f}  support={per_action[a]['support']}")

    print("\n" + "=" * 78)
    print("TYPED SUB-DECISION teacher-forced accuracy (where each head fires)")
    print("=" * 78)
    sub_decisions = {}
    for name in ("kind", "role", "gtype", "source", "prime"):
        tot = counters[f"{name}_total"]
        hits = counters[f"{name}_hits"]
        acc = hits / tot if tot else float("nan")
        sub_decisions[name] = {"accuracy": acc, "support": tot}
        label = {"kind": "clause-kind @ OPEN_CLAUSE", "role": "role label @ GROUND/EMIT_*",
                 "gtype": "grounding-type @ GROUND/EMIT_*", "source": "source @ sense/reference/elision",
                 "prime": "prime @ EMIT_SYNTH_SLOT"}[name]
        print(f"  {label:<38} accuracy={acc:.4f}  support={tot}")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print("Teacher-forced accuracy is the training/loss path's own report card and")
    print("an UPPER BOUND on decode quality (decode compounds errors across steps;")
    print("this does not). If OVERALL is high (~0.85+) and roughly consistent with")
    print("the fixed-decode edge-F1 (~0.70), the gap is normal exposure bias and the")
    print("loss/training path is sound. If OVERALL (or specifically CLOSE_CLAUSE/")
    print("STOP) is LOW, that points to a genuine training-path bug, not decode.")

    out = {
        "split": args.split,
        "n_records": len(recs),
        "n_derivations": n_derivations,
        "overall_accuracy": overall_acc,
        "overall_support": counters["overall_total"],
        "per_action_type": per_action,
        "sub_decisions": sub_decisions,
    }
    print("\nJSON summary:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
