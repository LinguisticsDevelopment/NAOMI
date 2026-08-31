"""M58e: THE FIRST PROSE-TRAINING SCRIPT (dev/AURORA_SPRINT.md, RESEARCH_NOTES
M58a-d). M58b/d measured the FROZEN M60 checkpoint zero-shot on converted
prose episodes (scripts/eval_prose.py): 0.583 -> 0.558 after M58d's PLACE
grounding fix, with the honest diagnosis that the fix is now structurally
right but OUT-OF-DISTRIBUTION for a checkpoint that only ever saw ``var:``
atoms in PLACE training -- "the fix pays at the NEXT training run, not on a
frozen model". M58d's own "Next" line: corpus scale-up -> retrain on
corrected grounding -> re-measure zero-shot on HELD-OUT DOCUMENTS. This
script is that retraining step.

Two ideas, glued together:

1. **Document-held-out zero-shot** (:func:`split_by_document`): prose
   episodes are grouped by ``meta["source_doc"]`` (the SAME key
   scripts/eval_prose.py's own ``_group_of`` reads, set by
   ``nsm_ct.corpus.make_episodes``) and a seeded subset of whole DOCUMENTS
   is held out entirely -- every episode from a held-out document goes to
   eval, none of its sentences are ever seen in training. This is the
   "zero-shot-on-unseen-documents" measurement M58b/d could only take
   against a checkpoint that saw prose NEVER; this script measures it
   against a checkpoint that saw SOME prose, held to the same standard
   (no sentence-level leakage -- a document's later sentence never
   trains on the same document's earlier sentence being in eval, and
   vice versa).

2. **Curriculum-mixed training** (:func:`build_mixed_training_set`):
   training on prose ALONE risks catastrophically forgetting the proven
   synthetic-curriculum capabilities (instance binding, write-back,
   inverse query, rich multi-entity discourse -- the M57 memory schema
   RESEARCH_NOTES spent a full milestone proving). Every training epoch
   therefore mixes in ``scripts.train_instances.build_instance_curriculum``
   episodes at a target ``--curriculum-frac`` of the combined training set
   (default 0.5 -- half prose, half synthetic), and a SEPARATE,
   never-trained-on synthetic curriculum split (``--curriculum-val-episodes``,
   independently seeded, not carved out of the training mix -- there is no
   fixed curriculum "dataset" to leak from, since ``build_instance_curriculum``
   generates fresh episodes on demand) is scored every ``--log-interval``
   epochs and in the final report, alongside the prose numbers -- the
   retention check.

Reuses, not duplicates: :func:`scripts.eval_prose.load_episodes` (the JSONL
loader) and its aggregation helpers (``per_group_accuracy``,
``random_guess_floor``, ``majority_baseline``, ``_group_of``) for the
held-out-document report -- SAME keys as ``scripts/eval_prose.py``'s own
"THE ZERO-SHOT PROSE NUMBER" table, so a before/after run pair is directly
comparable line for line. :func:`scripts.train_instances.build_instance_curriculum`
for the synthetic side (unchanged, not re-implemented). ``_train_common``'s
``build_model``/``epoch_minibatches``/``eval_minibatched``/``peak_rss_mb``
(the SAME minibatching/footprint machinery every other training script uses).
``nsm_ct.checkpoint``'s ``save_checkpoint``/``load_checkpoint`` (the SAME
config-driven reconstruction path -- a ``--load``ed M60 checkpoint and this
script's own fresh-built model come from the identical code path).

Usage:
    # normal arm: warm-start from M60, train on prose + curriculum mix
    python scripts/train_prose.py --episodes runs/prose_episodes_large.jsonl \\
        --load runs/m60_checkpoint.pt --save runs/prose_checkpoint.pt \\
        --batch-size 64 --threads 2

    # the before/after comparator: the loaded checkpoint AS-IS, no training
    python scripts/train_prose.py --episodes runs/prose_episodes_large.jsonl \\
        --load runs/m60_checkpoint.pt --frozen-eval --batch-size 64 --threads 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
# scripts/ is already on sys.path for anything invoked as `python scripts/train_prose.py`
# (argv[0]'s directory); tests add it explicitly, same contract eval_prose.py's own
# module docstring documents ("scripts/ already on sys.path -- this is a sibling module").

import eval_prose  # noqa: E402
from _train_common import (  # noqa: E402
    add_footprint_args, apply_threads, build_model, epoch_minibatches, eval_minibatched, peak_rss_mb,
)
from nsm_ct.checkpoint import git_commit, load_checkpoint, save_checkpoint  # noqa: E402
from nsm_ct.clause_reactor import build_clause_batch  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402
from train_instances import RECENCY_EXTRA, build_instance_curriculum  # noqa: E402

AUX_WEIGHT = 0.5   # resolver cross-entropy weight -- same constant every M57+ training script uses
_CURRICULUM_VAL_SEED_OFFSET = 900   # kept apart from build_mixed_training_set's own train-side offset (500)
_CURRICULUM_TRAIN_SEED_OFFSET = 500
_MIX_SHUFFLE_SEED_OFFSET = 1
_DEFAULT_HOLDOUT_FRAC = 0.2

# JSONL loader: REUSED verbatim, not duplicated -- see module docstring.
load_episodes = eval_prose.load_episodes


# ---------------------------------------------------------------------------
# 1. document-held-out split
# ---------------------------------------------------------------------------
def split_by_document(episodes, holdout_docs=None, holdout_frac=None, seed: int = 0):
    """Groups ``episodes`` by ``meta["source_doc"]`` and holds out a seeded
    subset of whole DOCUMENTS -- every episode belonging to a held-out
    document goes to the eval half, none of its sentences are ever seen by
    the other half. ``holdout_docs`` (an exact document count) takes
    priority over ``holdout_frac`` (a fraction of the DISTINCT document
    count, rounded); if neither is given, defaults to
    :data:`_DEFAULT_HOLDOUT_FRAC`. Document order is the first-seen order
    in ``episodes`` (deterministic given the input list); which documents
    get held out is a seeded permutation of that order, so
    ``(episodes, holdout_docs/frac, seed)`` alone determines the split --
    same call, same split, every time.

    Returns ``(train_eps, eval_eps)``, together covering every input
    episode exactly once (a document with NO episodes can't occur -- the
    grouping is built FROM the episode list).
    """
    docs: dict = {}
    order: list = []
    for e in episodes:
        d = e.meta.get("source_doc", "")
        if d not in docs:
            docs[d] = []
            order.append(d)
        docs[d].append(e)

    n_docs = len(order)
    if holdout_docs is not None:
        n_hold = int(holdout_docs)
    else:
        frac = _DEFAULT_HOLDOUT_FRAC if holdout_frac is None else holdout_frac
        n_hold = int(round(n_docs * frac))
    n_hold = max(0, min(n_hold, n_docs))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_docs) if n_docs else np.array([], dtype=int)
    hold_idx = set(perm[:n_hold].tolist())

    train_eps: list = []
    eval_eps: list = []
    for i, d in enumerate(order):
        (eval_eps if i in hold_idx else train_eps).extend(docs[d])
    return train_eps, eval_eps


# ---------------------------------------------------------------------------
# 2. curriculum-mixed training set
# ---------------------------------------------------------------------------
def _build_curriculum_val(n: int, seed: int):
    return build_instance_curriculum(n, seed + _CURRICULUM_VAL_SEED_OFFSET) if n > 0 else []


def build_mixed_training_set(prose_train_eps, curriculum_frac: float, seed: int,
                              curriculum_val_episodes: int = 200):
    """Builds the epoch training set: ``prose_train_eps`` (every prose
    episode NOT held out by :func:`split_by_document`) plus freshly
    generated :func:`~scripts.train_instances.build_instance_curriculum`
    episodes, sized so curriculum episodes make up ~``curriculum_frac`` of
    the COMBINED training set (``curriculum_frac=0`` -> prose-only
    training; the ratio is anchored to ``len(prose_train_eps)``, so it
    stays correct regardless of corpus size). The combined list is then
    seed-shuffled once so a training epoch's minibatches interleave both
    kinds rather than seeing one kind, then the other.

    ``curriculum_val_episodes`` is a SEPARATE, independently-seeded
    synthetic split (offset :data:`_CURRICULUM_VAL_SEED_OFFSET` from the
    training-side curriculum's own offset) -- never mixed into the
    returned training set, used only for the retention report every
    ``--log-interval`` epochs. There is no fixed curriculum "dataset" to
    carve a val split OUT of (the generator makes fresh episodes on
    demand), so this is "generate a different, disjoint-by-seed batch",
    not "hold out a slice of the training batch" -- functionally
    equivalent (never trained on, scored throughout), simpler to reason
    about (the training-mix fraction below is exact, not diluted by a
    val carve-out).

    Returns ``(mixed_train_eps, curriculum_train_eps, curriculum_val_eps)``.
    """
    n_prose = len(prose_train_eps)
    if curriculum_frac <= 0.0:
        curriculum_train_eps: list = []
    else:
        cf = min(curriculum_frac, 0.99)
        if n_prose == 0:
            # No prose to anchor the ratio to (e.g. every document was held
            # out): fall back to the val-set size as "some curriculum",
            # rather than producing zero training data outright.
            n_curriculum = max(1, curriculum_val_episodes)
        else:
            n_curriculum = max(1, int(round(n_prose * cf / (1.0 - cf))))
        curriculum_train_eps = build_instance_curriculum(n_curriculum, seed + _CURRICULUM_TRAIN_SEED_OFFSET)

    curriculum_val_eps = _build_curriculum_val(curriculum_val_episodes, seed)

    combined = list(prose_train_eps) + curriculum_train_eps
    order = (np.random.RandomState(seed + _MIX_SHUFFLE_SEED_OFFSET).permutation(len(combined))
             if combined else np.array([], dtype=int))
    mixed = [combined[i] for i in order]
    return mixed, curriculum_train_eps, curriculum_val_eps


# ---------------------------------------------------------------------------
# 3. eval helpers
# ---------------------------------------------------------------------------
def _quick_acc(model, batch, gold, batch_size) -> float:
    """One minibatched forward pass -> accuracy, or NaN when there's
    nothing to score (an empty split -- e.g. ``--holdout-docs`` larger
    than the corpus, or ``--curriculum-val-episodes 0``)."""
    if batch is None or gold is None or len(gold) == 0:
        return float("nan")
    pred = eval_minibatched(model, batch, batch_size)["answer_logits"].argmax(-1)
    return float((pred == gold).float().mean())


def prose_eval_report(model, eps, batch, gold, batch_size: int) -> dict:
    """The held-out-document zero-shot table -- SAME keys/shape as
    ``scripts/eval_prose.py``'s own "THE ZERO-SHOT PROSE NUMBER" report
    (overall/per-relation/per-source + cleanup abstain), reusing that
    module's aggregation helpers directly rather than a second hand-rolled
    implementation. Assumes ``model.cleanup`` is already set and the model
    is already in eval mode (the caller sets this once for the whole final
    report, not per-helper)."""
    print("\n=== HELD-OUT-DOCUMENT ZERO-SHOT (prose) ===")
    if not eps:
        print("no held-out prose episodes -- nothing to score.")
        return {}
    out = eval_minibatched(model, batch, batch_size)
    pred = out["answer_logits"].argmax(-1)
    correct = (pred == gold)
    n = len(eps)
    overall_acc = float(correct.float().mean())
    floor = eval_prose.random_guess_floor(eps)
    maj_acc, maj_text = eval_prose.majority_baseline(eps)

    print(f"n episodes evaluated: {n}")
    print(f"overall accuracy: {overall_acc:.3f}")
    print(f"random-guess floor (mean 1/n_options): {floor:.3f}")
    print(f"majority baseline (always {maj_text!r}): {maj_acc:.3f}")

    print("per-relation:")
    rel_acc = eval_prose.per_group_accuracy(eps, correct, lambda e: e.meta.get("relation", "?"))
    for rel in sorted(rel_acc):
        hits, cnt = rel_acc[rel]
        print(f"  {rel:<10} {hits}/{cnt} = {hits / cnt:.3f}")

    print("per-source:")
    src_acc = eval_prose.per_group_accuracy(eps, correct, eval_prose._group_of)
    for grp in ("synthetic", "real"):
        if grp not in src_acc:
            continue
        hits, cnt = src_acc[grp]
        print(f"  {grp:<10} {hits}/{cnt} = {hits / cnt:.3f}")

    report = {"n": n, "overall_acc": overall_acc, "floor": floor, "majority_acc": maj_acc,
              "per_relation": rel_acc, "per_source": src_acc}

    if "cleanup_index" in out:
        cleanup_abstain = out["cleanup_abstain"].bool()
        abstain_rate = float(cleanup_abstain.float().mean())
        confident = ~cleanup_abstain
        acc_confident = (float((pred[confident] == gold[confident]).float().mean())
                          if bool(confident.any()) else float("nan"))
        margins = out["cleanup_margin"]
        print(f"CLEANUP: abstain_rate={abstain_rate:.3f} acc_when_confident={acc_confident:.3f} "
              f"(n={int(confident.sum())}) vs overall={overall_acc:.3f} (n={n})")
        print(f"  margin: min={float(margins.min()):.3f} mean={float(margins.mean()):.3f} "
              f"max={float(margins.max()):.3f}")
        report.update({"abstain_rate": abstain_rate, "acc_when_confident": acc_confident})
    return report


def curriculum_retention_report(model, eps, batch, gold, batch_size: int) -> dict:
    """The synthetic-curriculum retention table: overall accuracy on a
    never-trained-on curriculum split, plus a per-kind (old/writeback/
    instance/rich) breakdown -- "did prose training catastrophically
    forget the proven capabilities". Assumes ``model`` is already in eval
    mode (see :func:`prose_eval_report`)."""
    print("\n=== CURRICULUM RETENTION (synthetic, never trained on) ===")
    if not eps:
        print("no curriculum retention episodes -- nothing to score.")
        return {}
    out = eval_minibatched(model, batch, batch_size)
    pred = out["answer_logits"].argmax(-1)
    correct = (pred == gold)
    n = len(eps)
    overall_acc = float(correct.float().mean())
    print(f"n episodes evaluated: {n}")
    print(f"overall accuracy: {overall_acc:.3f}")

    per_kind: dict = {}
    for i, e in enumerate(eps):
        per_kind.setdefault(str(e.meta.get("kind", "old")), []).append(bool(correct[i]))
    for k in sorted(per_kind):
        w = per_kind[k]
        print(f"  kind {k}: {sum(w)}/{len(w)} = {sum(w) / len(w):.3f}")

    return {"n": n, "overall_acc": overall_acc,
            "per_kind": {k: (sum(w), len(w)) for k, w in per_kind.items()}}


# ---------------------------------------------------------------------------
# 4. main run
# ---------------------------------------------------------------------------
def _check_args(args) -> None:
    """Raises :class:`ValueError` on an invalid combination -- factored
    out of :func:`main` so tests can exercise the validation without
    going through argparse/SystemExit."""
    if args.frozen_eval and not args.load:
        raise ValueError("--frozen-eval requires --load (nothing to evaluate as-is otherwise)")
    if args.save and args.frozen_eval:
        raise ValueError("--save with --frozen-eval would just re-save the untouched --load checkpoint; omit one")


def run(args) -> dict:
    episodes = load_episodes(args.episodes)
    print(f"=== train_prose: loaded {len(episodes)} prose episodes from {args.episodes} ===", flush=True)

    prose_train_eps, prose_eval_eps = split_by_document(
        episodes, holdout_docs=args.holdout_docs, holdout_frac=args.holdout_frac, seed=args.seed)
    n_train_docs = len({e.meta.get("source_doc", "") for e in prose_train_eps})
    n_eval_docs = len({e.meta.get("source_doc", "") for e in prose_eval_eps})
    print(f"=== document split: {len(prose_train_eps)} prose train episodes ({n_train_docs} docs), "
          f"{len(prose_eval_eps)} prose eval episodes ({n_eval_docs} docs, FULLY held out) ===", flush=True)

    if args.frozen_eval:
        mixed_train_eps, curriculum_train_eps = [], []
        curriculum_val_eps = _build_curriculum_val(args.curriculum_val_episodes, args.seed)
    else:
        mixed_train_eps, curriculum_train_eps, curriculum_val_eps = build_mixed_training_set(
            prose_train_eps, args.curriculum_frac, args.seed,
            curriculum_val_episodes=args.curriculum_val_episodes)
    print(f"=== training mix: {len(mixed_train_eps)} total ({len(prose_train_eps)} prose + "
          f"{len(curriculum_train_eps)} curriculum, target curriculum_frac={args.curriculum_frac}); "
          f"{len(curriculum_val_eps)} curriculum retention-val episodes (never trained on) ===", flush=True)

    all_build_eps = mixed_train_eps + prose_eval_eps + curriculum_val_eps
    texts = [t for e in all_build_eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="en")
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping.")
        return {}
    meaning_resolver = NSMMeaningResolver()

    model_config = {
        "dim": args.dim, "hidden": args.hidden, "track": args.track,
        "use_cand_feature": True, "cand_feature_extra": 1 + RECENCY_EXTRA,
        "evidence_prior_beta": None,
    }
    dim, hidden = args.dim, args.hidden

    if args.load:
        model, ckpt_config, ckpt_extra = load_checkpoint(args.load)
        print(f"=== loaded checkpoint {args.load}: dim={ckpt_config.get('dim')} "
              f"hidden={ckpt_config.get('hidden')} track={ckpt_config.get('track')} "
              f"use_cand_feature={ckpt_config.get('use_cand_feature')} "
              f"cand_feature_extra={ckpt_config.get('cand_feature_extra')} "
              f"trained_total_acc={ckpt_extra.get('total_acc')} "
              f"git_commit={ckpt_config.get('git_commit')} ===", flush=True)
        mismatches = [f"{k}: requested={model_config.get(k)!r} checkpoint={ckpt_config.get(k)!r}"
                      for k in ("dim", "hidden", "track", "use_cand_feature", "cand_feature_extra")
                      if ckpt_config.get(k) != model_config.get(k)]
        if mismatches:
            print("  NOTE: overriding requested config with the checkpoint's own (compatibility): "
                  + "; ".join(mismatches), flush=True)
        dim = ckpt_config.get("dim", dim)
        hidden = ckpt_config.get("hidden", hidden)
        model_config = ckpt_config
    else:
        torch.manual_seed(args.seed)
        model = build_model(model_config)

    resolver = model.resolver
    n_resolver_params = sum(p.numel() for p in resolver.parameters()) if resolver is not None else 0
    codec = TPRCodec(dim=dim, max_pos=model_config.get("codec_max_pos", 64))

    prose_eval_batch = (build_clause_batch(prose_eval_eps, parser, meaning_resolver, codec,
                                            writeback_cheat=args.cheat, writeback_no_gold=args.no_gold_eval)
                         if prose_eval_eps else None)
    curriculum_val_batch = (build_clause_batch(curriculum_val_eps, parser, meaning_resolver, codec,
                                                writeback_cheat=args.cheat, writeback_no_gold=args.no_gold_eval)
                             if curriculum_val_eps else None)
    gold_prose_eval = torch.tensor([e.answer_idx for e in prose_eval_eps]) if prose_eval_eps else None
    gold_curriculum_val = torch.tensor([e.answer_idx for e in curriculum_val_eps]) if curriculum_val_eps else None

    losses: list = []
    elapsed_min = 0.0
    if args.frozen_eval:
        print("=== --frozen-eval: skipping training, evaluating the loaded checkpoint AS-IS "
              "(the before/after comparator) ===", flush=True)
    elif not mixed_train_eps:
        print("=== no training episodes (0 prose-train + 0 curriculum) -- skipping the training loop ===",
              flush=True)
    else:
        tr = build_clause_batch(mixed_train_eps, parser, meaning_resolver, codec, writeback_cheat=args.cheat)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        gold_tr = torch.tensor([e.answer_idx for e in mixed_train_eps])
        n_tr = len(mixed_train_eps)
        t0 = time.time()
        model.train()
        for i in range(args.epochs):
            epoch_losses = []
            for mb_idx in epoch_minibatches(n_tr, args.batch_size, args.seed, i):
                idx_t = torch.from_numpy(mb_idx)
                sub = tr.subset(idx_t)
                sub_gold = gold_tr[idx_t]
                out = model(sub)
                loss = F.cross_entropy(out["answer_logits"], sub_gold)
                if resolver is not None and "resolver_logits" in out:
                    cg = sub.cand_gold
                    has_cand = cg >= 0
                    if bool(has_cand.any()):
                        aux = F.cross_entropy(out["resolver_logits"][has_cand], cg[has_cand])
                        loss = loss + AUX_WEIGHT * aux
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_losses.append(float(loss.item()))
            last_loss = epoch_losses[-1]
            losses.append(last_loss)
            if (i + 1) % args.log_interval == 0 or i == 0 or i == args.epochs - 1:
                model.eval()
                prose_acc = _quick_acc(model, prose_eval_batch, gold_prose_eval, args.batch_size)
                curr_acc = _quick_acc(model, curriculum_val_batch, gold_curriculum_val, args.batch_size)
                model.train()
                print(f"  epoch {i + 1:3d} loss={last_loss:.3f} prose_val={prose_acc:.3f} "
                      f"curriculum_val={curr_acc:.3f}", flush=True)
        elapsed_min = (time.time() - t0) / 60

    # M60 CLEANUP wiring (reused verbatim from scripts/train_instances.py's run_arm /
    # scripts/eval_prose.py's run: a loaded checkpoint's model.cleanup defaults to
    # False, and this training loop never touches it mid-training -- set it only NOW,
    # for the final report).
    model.cleanup = True
    model.eval()
    prose_report = prose_eval_report(model, prose_eval_eps, prose_eval_batch, gold_prose_eval, args.batch_size)
    curriculum_report = curriculum_retention_report(
        model, curriculum_val_eps, curriculum_val_batch, gold_curriculum_val, args.batch_size)

    rss = peak_rss_mb()
    print(f"\npeak_rss_mb: {rss:.1f}  elapsed_min: {elapsed_min:.2f}", flush=True)

    return {
        "prose_eval": prose_report, "curriculum_retention": curriculum_report,
        "peak_rss_mb": rss, "elapsed_min": elapsed_min, "losses": losses,
        "n_resolver_params": n_resolver_params,
        "model": model, "model_config": model_config,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=str, required=True,
                     help="Prose episodes JSONL (scripts/convert_corpus.py --out).")
    ap.add_argument("--holdout-docs", type=int, default=None,
                     help="Exact number of DOCUMENTS to fully hold out for the zero-shot eval "
                          "(takes priority over --holdout-frac).")
    ap.add_argument("--holdout-frac", type=float, default=None,
                     help=f"Fraction of documents to fully hold out. Default {_DEFAULT_HOLDOUT_FRAC} "
                          "when neither this nor --holdout-docs is given.")
    ap.add_argument("--curriculum-frac", type=float, default=0.5,
                     help="Target fraction of the TRAINING mix that is synthetic curriculum "
                          "(train_instances.build_instance_curriculum) rather than prose -- guards "
                          "against catastrophic forgetting of the proven M57 capabilities. 0 = "
                          "prose-only training.")
    ap.add_argument("--curriculum-val-episodes", type=int, default=200,
                     help="Size of the separately, independently-seeded synthetic curriculum "
                          "retention-eval split (never trained on), scored every --log-interval "
                          "epochs and in the final report alongside the prose numbers.")
    ap.add_argument("--track", choices=["A", "B"], default="A",
                     help="Resolver track for a FRESH (non-warm-started) model; ignored (overridden) "
                          "when --load is given.")
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--log-interval", type=int, default=20,
                     help="Print prose_val/curriculum_val every N epochs (plus epoch 1 and the last).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cheat", action="store_true",
                     help="Cheat-baseline arm: candidate sets stripped everywhere -- no resolver "
                          "signal reaches any candidate-bearing (curriculum writeback/instance) "
                          "episode. Inert for prose episodes, which never carry candidate sets.")
    ap.add_argument("--no-gold-eval", action="store_true",
                     help="Held-out EVAL batches only (prose zero-shot + curriculum retention) built "
                          "with no gold grounding anywhere. Training is unaffected.")
    ap.add_argument("--frozen-eval", action="store_true",
                     help="Skip training entirely -- evaluate the --load checkpoint AS-IS on the "
                          "held-out prose documents + curriculum retention set (the before/after "
                          "comparator this script's own module docstring describes). Requires --load.")
    ap.add_argument("--load", type=str, default=None,
                     help="Warm-start from a checkpoint (e.g. runs/m60_checkpoint.pt) instead of a "
                          "fresh model.")
    ap.add_argument("--save", type=str, default=None,
                     help="Save the trained model to this path at the end. Ignored (rejected) with "
                          "--frozen-eval, since nothing new was trained.")
    add_footprint_args(ap)
    args = ap.parse_args()
    apply_threads(args)
    try:
        _check_args(args)
    except ValueError as exc:
        ap.error(str(exc))

    result = run(args)
    if args.save and result.get("model") is not None:
        config = dict(result["model_config"])
        config.update({
            "codec_dim": args.dim, "codec_max_pos": 64, "meaning_source": "usvs",
            "episodes": args.episodes, "holdout_docs": args.holdout_docs, "holdout_frac": args.holdout_frac,
            "curriculum_frac": args.curriculum_frac, "curriculum_val_episodes": args.curriculum_val_episodes,
            "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
            "cheat": args.cheat, "no_gold_eval": args.no_gold_eval,
            "git_commit": git_commit(), "argv": sys.argv[:],
        })
        extra = {"prose_eval": result["prose_eval"], "curriculum_retention": result["curriculum_retention"],
                 "n_resolver_params": result["n_resolver_params"], "peak_rss_mb": result["peak_rss_mb"],
                 "elapsed_min": result["elapsed_min"]}
        save_checkpoint(args.save, result["model"], config=config, extra=extra)
        size = os.path.getsize(args.save)
        print(f"  saved checkpoint to {args.save} ({size} bytes)", flush=True)


if __name__ == "__main__":
    main()
