"""M59b training script: the CROSS-PASSAGE curriculum for episodic LTM
(M59a). Sibling to scripts/train_instances.py -- SAME arm/report pattern
(track dispatch, aux loss, per-kind accuracy, checkpoint save/load) --
extended for MULTI-PASSAGE DOCUMENTS instead of single-passage episodes:
episodes are grouped by ``meta["doc_id"]`` (nsm_ct.curriculum2.
DocumentGenerator) and each document is driven through
``scripts._train_common.DocumentRunner.run_document``, one document at a
time (DocumentRunner's own contract: one InstanceRegistry per call -- see
its docstring), summing/averaging losses across a minibatch of documents.

Every document's own (registry, per-passage ClauseBatch list) is BUILT ONCE
up front (mirrors every other training script's "build batches once, only
the model's forward/backward re-runs per epoch" convention) -- a fresh
``nsm_ct.instances.ProvenanceLog`` is still created per document PER RUN
(train step or eval pass), since the log is a pure per-pass audit trail with
no need to persist across epochs.

Arms:
    python scripts/train_ltm.py --track A                         # normal arm
    python scripts/train_ltm.py --track A --force-binding gold    # forced-gold EVAL (ceiling)
    python scripts/train_ltm.py --track A --force-binding wrong   # forced-wrong EVAL (floor) --
                                                                      THE full-scale validity check
                                                                      (CLAUDE.md: smoke-scale never
                                                                      gates curriculum validity).
    python scripts/train_ltm.py --track A --cheat                 # cheat-baseline floor (no candidates
                                                                      anywhere, train AND eval)
    python scripts/train_ltm.py --track A --no-gold-eval          # EVAL batches only, no gold anywhere
                                                                      (candidates + the TRAINED resolver only)
    python scripts/train_ltm.py --track A --no-ltm                # "consolidate nothing" floor: every
                                                                      passage reads ltm=None regardless of
                                                                      what was consolidated
    python scripts/train_ltm.py --track A --trust-ltm 0.8         # a stricter/looser STM->LTM promotion gate
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

from _train_common import (  # noqa: E402
    DocumentRunner, add_footprint_args, apply_threads, build_model, epoch_minibatches, peak_rss_mb,
)
from nsm_ct.checkpoint import git_commit, load_checkpoint, save_checkpoint  # noqa: E402
from nsm_ct.clause_reactor import build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import generate_document_episodes  # noqa: E402
from nsm_ct.instances import InstanceRegistry, ProvenanceLog  # noqa: E402
from nsm_ct.ltm import NEW, link_decision  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

AUX_WEIGHT = 0.5   # resolver cross-entropy weight, same constant train_instances.py/train_writeback.py use


def _group_by_document(episodes):
    """Group document-kind episodes by ``doc_id``, sorted by
    ``passage_index`` -- the caller-side grouping ``nsm_ct.ltm``'s module
    docstring requires of every ``DocumentRunner.run_document`` caller.
    Returns a list of ``(doc_id, [Episode, ...])`` pairs, in first-seen
    ``doc_id`` order (deterministic given the input episode list's own
    order, which ``generate_document_episodes`` produces deterministically)."""
    docs: dict = {}
    for ep in episodes:
        docs.setdefault(ep.meta["doc_id"], []).append(ep)
    for passages in docs.values():
        passages.sort(key=lambda e: e.meta["passage_index"])
    return list(docs.items())


def build_documents(doc_items, dim: int, meaning_resolver, codec: TPRCodec, *,
                     cheat: bool = False, no_gold: bool = False, force: str = None):
    """Build every document's own (registry, per-passage ClauseBatch list)
    ONCE -- ``document_registry`` is threaded through every passage's
    ``build_clause_batch`` call for that document, IN PASSAGE ORDER (the
    ltm.py contract: one registry per document, constructed once, threaded
    through every passage's batch-build call)."""
    built = []
    for doc_id, passages in doc_items:
        registry = InstanceRegistry(dim=dim, seed=passages[0].meta["instance_seed"])
        batches = [
            build_clause_batch([ep], None, meaning_resolver, codec,
                                writeback_cheat=cheat, writeback_no_gold=no_gold,
                                writeback_force=force, document_registry=registry)
            for ep in passages
        ]
        built.append({"doc_id": doc_id, "registry": registry, "batches": batches, "episodes": passages})
    return built


def loss_fn(out, batch):
    """Answer cross-entropy (K=1 registration/filler passages contribute
    IDENTICALLY ZERO -- softmax over a single class is always probability 1
    at that class, so ``F.cross_entropy`` is 0 regardless of the model's
    logit, contributing no gradient; only the final, real-question passage's
    4-option answer_logits actually drives this term) plus the resolver's
    own cand_gold-supervised aux loss at the mention step (mirrors
    scripts/train_instances.py's own loss_fn exactly)."""
    loss = F.cross_entropy(out["answer_logits"], batch.answer)
    if "resolver_logits" in out and batch.cand_gold is not None:
        cg = batch.cand_gold
        has_cand = cg >= 0
        if bool(has_cand.any()):
            aux = F.cross_entropy(out["resolver_logits"][has_cand], cg[has_cand])
            loss = loss + AUX_WEIGHT * aux
    return loss


def run_documents_train(runner: DocumentRunner, docs, codec: TPRCodec, opt) -> float:
    """One minibatch SGD step over ``docs`` (a list of prebuilt document
    dicts): ``opt.zero_grad()`` once, then EACH document's own
    ``run_document(train=True, loss_fn=loss_fn)`` call backward()s
    IMMEDIATELY (gradient accumulation via repeated ``.backward()`` calls,
    not a single combined-graph backward -- frees each document's own
    forward graph as soon as its gradient is accumulated, instead of
    holding every document's graph alive simultaneously until one shared
    backward call), then ``opt.step()`` once. Returns the minibatch's mean
    per-document loss (float, for logging)."""
    opt.zero_grad()
    losses = []
    for doc in docs:
        log = ProvenanceLog()
        _, doc_loss = runner.run_document(doc["batches"], doc["registry"], log, codec,
                                           train=True, loss_fn=loss_fn)
        doc_loss.backward()
        losses.append(float(doc_loss.detach()))
    opt.step()
    return float(np.mean(losses)) if losses else 0.0


def evaluate(runner: DocumentRunner, docs, codec: TPRCodec) -> dict:
    """No-grad pass over EVERY document in ``docs``, one ``run_document``
    call per document (eval mode). Returns a metrics dict: overall
    final-question accuracy, per-question_type / per-condition accuracy,
    link accuracy + NEW-rate (resolver argmax vs gold at the mention step),
    mean n_promoted, and per-passage-index accuracy."""
    n_correct = 0
    n_total = 0
    link_correct = 0
    link_total = 0
    new_num = 0
    new_den = 0
    n_promoted_all = []
    by_qtype: dict = {}
    by_condition: dict = {}
    per_passage: dict = {}
    for doc in docs:
        log = ProvenanceLog()
        with torch.no_grad():
            reports, _ = runner.run_document(doc["batches"], doc["registry"], log, codec, train=False)
        eps = doc["episodes"]
        final_idx = len(eps) - 1
        for p_idx, (report, ep, batch) in enumerate(zip(reports, eps, doc["batches"])):
            n_promoted_all.append(report["n_promoted"])
            out = report["out"]
            pred = int(out["answer_logits"][0].argmax())
            gold = int(batch.answer[0])
            hit = pred == gold
            per_passage.setdefault(p_idx, []).append(hit)
            if p_idx == final_idx:
                n_total += 1
                n_correct += int(hit)
                by_qtype.setdefault(ep.meta["question_type"], []).append(hit)
                by_condition.setdefault(ep.meta["condition"], []).append(hit)
            if "resolver_logits" in out and batch.cand_gold is not None:
                cg = batch.cand_gold[0]
                has_cand = cg >= 0
                if bool(has_cand.any()):
                    probs = torch.softmax(out["resolver_logits"][0][has_cand], dim=-1)
                    pred_idx = int(probs.argmax(-1)[0])
                    gold_idx = int(cg[has_cand][0])
                    link_correct += int(pred_idx == gold_idx)
                    link_total += 1
                    decided = int(link_decision(probs[0]))
                    new_den += 1
                    new_num += int(decided == NEW or decided == 1)   # index 1 = the NEW candidate, by construction

    return {
        "total_acc": n_correct / n_total if n_total else 0.0,
        "n_total": n_total,
        "by_qtype": {k: (sum(v) / len(v), len(v)) for k, v in by_qtype.items()},
        "by_condition": {k: (sum(v) / len(v), len(v)) for k, v in by_condition.items()},
        "link_acc": link_correct / link_total if link_total else None,
        "link_total": link_total,
        "new_rate": new_num / new_den if new_den else None,
        "n_promoted_mean": float(np.mean(n_promoted_all)) if n_promoted_all else 0.0,
        "per_passage_acc": {k: sum(v) / len(v) for k, v in per_passage.items()},
    }


def print_report(name: str, metrics: dict) -> None:
    print(f"  [{name}] kind document: total_acc={metrics['total_acc']:.3f} (n={metrics['n_total']})", flush=True)
    for k, (acc, n) in sorted(metrics["by_qtype"].items()):
        print(f"    document/{k}: {acc:.3f} (n={n})")
    for k, (acc, n) in sorted(metrics["by_condition"].items()):
        print(f"    document/{k}: {acc:.3f} (n={n})")
    if metrics["link_acc"] is not None:
        print(f"  [{name}] LINK ACCURACY: {metrics['link_acc']:.3f} (n={metrics['link_total']}) "
              f"NEW-rate={metrics['new_rate']:.3f}", flush=True)
    print(f"  [{name}] n_promoted mean={metrics['n_promoted_mean']:.2f}", flush=True)
    for p_idx, acc in sorted(metrics["per_passage_acc"].items()):
        print(f"    per_passage/{p_idx}: {acc:.3f}")


def run_arm(name: str, track, n_documents: int, dim: int, epochs: int, seed: int, hidden: int = 128,
            n_passages: int = 2, cheat: bool = False, no_gold_eval: bool = False,
            force_binding: str = None, no_ltm: bool = False, trust_ltm: float = None,
            batch_size: int = 16, load: str = None) -> dict:
    """``track``: "A" | "B" | None (no resolver installed -- linking never
    fires at all, a genuine floor arm: every mention step's candidate set
    still exists, but nothing ever scores it).

    Training is ALWAYS built with ``cheat`` only (never force/no_gold --
    mirrors scripts/train_instances.py's own contract: forced-binding/
    no-gold-eval are EVAL-ONLY arms). ``load`` skips training entirely and
    reconstructs the model via ``nsm_ct.checkpoint.load_checkpoint``, same
    "frozen-model path" contract as scripts/train_instances.py's own
    ``run_arm``.
    """
    meaning_resolver = NSMMeaningResolver()
    episodes = generate_document_episodes(n_documents, seed=seed, n_passages=n_passages)
    doc_items = _group_by_document(episodes)
    n_val = max(1, int(round(len(doc_items) * 0.2)))
    order = np.random.RandomState(seed).permutation(len(doc_items))
    val_idx = set(order[:n_val].tolist())
    tr_items = [d for i, d in enumerate(doc_items) if i not in val_idx]
    va_items = [d for i, d in enumerate(doc_items) if i in val_idx]

    model_config = {
        "dim": dim, "hidden": hidden, "track": track,
        "use_cand_feature": True, "cand_feature_extra": 2,   # slot 0: evidence-interaction; slot 1: cand_from_ltm
    }

    if load:
        model, ckpt_config, _ckpt_extra = load_checkpoint(load)
        if ckpt_config.get("dim") != dim or ckpt_config.get("hidden", 128) != hidden:
            print(f"  [{name}] NOTE: overriding --dim/--hidden ({dim}/{hidden}) with the "
                  f"checkpoint's own dim={ckpt_config.get('dim')} hidden={ckpt_config.get('hidden', 128)}",
                  flush=True)
        dim = ckpt_config.get("dim", dim)
        hidden = ckpt_config.get("hidden", hidden)
        model_config = ckpt_config
        elapsed_min = 0.0
        losses: list = []
    else:
        torch.manual_seed(seed)
        model = build_model(model_config)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    codec = TPRCodec(dim=dim)
    va_docs = build_documents(va_items, dim, meaning_resolver, codec,
                               cheat=cheat, no_gold=no_gold_eval, force=force_binding)
    runner = DocumentRunner(model, trust_ltm=trust_ltm, zero_ltm=no_ltm)

    if not load:
        tr_docs = build_documents(tr_items, dim, meaning_resolver, codec, cheat=cheat)
        n_tr = len(tr_docs)
        t0 = time.time()
        losses = []
        for i in range(epochs):
            epoch_losses = []
            for mb_idx in epoch_minibatches(n_tr, batch_size, seed, i):
                mb_docs = [tr_docs[j] for j in mb_idx]
                epoch_losses.append(run_documents_train(runner, mb_docs, codec, opt))
            last_loss = epoch_losses[-1] if epoch_losses else 0.0
            losses.append(last_loss)
            if (i + 1) % 5 == 0 or i == 0:
                metrics = evaluate(runner, va_docs, codec)
                print(f"  [{name}] epoch {i+1:3d} loss={last_loss:.3f} val_acc={metrics['total_acc']:.3f} "
                      f"link_acc={metrics['link_acc'] if metrics['link_acc'] is not None else float('nan'):.3f}",
                      flush=True)
        elapsed_min = (time.time() - t0) / 60

    n_resolver_params = sum(p.numel() for p in model.resolver.parameters()) if model.resolver is not None else 0
    metrics = evaluate(runner, va_docs, codec)
    print(f"  [{name}] FINAL n_documents={len(doc_items)} (train={len(tr_items)}, val={len(va_items)}) "
          f"resolver_params={n_resolver_params} time={elapsed_min:.2f} min peak_rss_mb={peak_rss_mb():.1f}",
          flush=True)
    print_report(name, metrics)

    return {"losses": losses, "metrics": metrics, "n_resolver_params": n_resolver_params,
            "peak_rss_mb": peak_rss_mb(), "elapsed_min": elapsed_min,
            "model": model, "model_config": model_config}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B"], default=None,
                     help="A = CorefHead, B = SharedScorer. Omit (None) for a no-resolver floor arm.")
    ap.add_argument("--force-binding", choices=["gold", "wrong"], default=None,
                     help="EVAL-ONLY: the held-out eval batches' mention-step collapse is teacher-forced "
                          "to the gold or the wrong link candidate. FULL-SCALE ONLY (CLAUDE.md: smoke-scale "
                          "results never gate curriculum validity): forced-gold must sit near ceiling and "
                          "forced-wrong near floor -- that GAP is the proof the answer flows through the link.")
    ap.add_argument("--cheat", action="store_true",
                     help="Cheat-baseline arm: the mention step's candidate set is stripped entirely (train "
                          "AND eval) -- no resolver data at all, even an installed resolver never fires. Must "
                          "sit at floor for type-(ii)/(iii) questions if linking is genuinely load-bearing.")
    ap.add_argument("--no-gold-eval", action="store_true",
                     help="EVAL-ONLY: no gold grounding anywhere in the eval batches (candidates + the "
                          "TRAINED resolver only). Training is unaffected.")
    ap.add_argument("--no-ltm", action="store_true",
                     help="The 'consolidate nothing' floor: every passage's forward pass reads ltm=None "
                          "regardless of what was consolidated (DocumentRunner's zero_ltm) -- the model has "
                          "STM only, every passage. Question type (i) (pure LTM recall) must crater to chance.")
    ap.add_argument("--trust-ltm", type=float, default=None,
                     help="STM->LTM promotion gate dial (nsm_ct.ltm.TRUST_LTM default 0.5). A record's write "
                          "gate must clear this to be consolidated into LTM.")
    ap.add_argument("--documents", type=int, default=800)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-passages", type=int, choices=[2, 3], default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=None,
                     help="After training + final eval, save the reactor+resolver to this path via "
                          "nsm_ct.checkpoint.save_checkpoint. Ignored when --load is given.")
    ap.add_argument("--load", type=str, default=None,
                     help="Load a checkpoint from this path instead of training, then run this script's own "
                          "eval/report on it (the FROZEN-model path). Mutually exclusive with --save.")
    add_footprint_args(ap)
    args = ap.parse_args()
    apply_threads(args)
    if args.save and args.load:
        ap.error("--save and --load are mutually exclusive (there is nothing new to save when loading)")

    print(f"=== document (LTM) "
          f"{f'(force-binding={args.force_binding}) ' if args.force_binding else ''}"
          f"{'(cheat) ' if args.cheat else ''}{'(no-gold-eval) ' if args.no_gold_eval else ''}"
          f"{'(no-ltm) ' if args.no_ltm else ''}"
          f"{'(load) ' if args.load else ''}"
          f"track={args.track}: {args.documents} documents (n_passages={args.n_passages}), "
          f"dim={args.dim}, epochs={args.epochs}, batch_size={args.batch_size}, "
          f"trust_ltm={args.trust_ltm} ===", flush=True)
    result = run_arm(f"track-{args.track}", args.track, args.documents, args.dim, args.epochs, args.seed,
                      args.hidden, n_passages=args.n_passages, cheat=args.cheat, no_gold_eval=args.no_gold_eval,
                      force_binding=args.force_binding, no_ltm=args.no_ltm, trust_ltm=args.trust_ltm,
                      batch_size=args.batch_size, load=args.load)

    if args.save and result:
        config = dict(result["model_config"])
        config.update({
            "codec_dim": args.dim, "codec_max_pos": 64, "meaning_source": "usvs",
            "documents": args.documents, "n_passages": args.n_passages,
            "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
            "cheat": args.cheat, "no_gold_eval": args.no_gold_eval, "force_binding": args.force_binding,
            "no_ltm": args.no_ltm, "trust_ltm": args.trust_ltm,
            "git_commit": git_commit(), "argv": sys.argv[:],
        })
        extra = {"metrics": result["metrics"], "n_resolver_params": result["n_resolver_params"],
                 "peak_rss_mb": result["peak_rss_mb"], "elapsed_min": result["elapsed_min"]}
        save_checkpoint(args.save, result["model"], config=config, extra=extra)
        size = os.path.getsize(args.save)
        print(f"  saved checkpoint to {args.save} ({size} bytes)", flush=True)


if __name__ == "__main__":
    main()
