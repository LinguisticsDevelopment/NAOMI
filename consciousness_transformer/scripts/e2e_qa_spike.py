"""M62: THE END-TO-END QA SPIKE (dev/E2E_QA_SPIKE.md, dev/AUDIT_2026-10-15.md
section F item 2 -- "the thesis test"). Teacher-built episodes (the
DETERMINISTIC parser path, ``nsm_ct.clause.extract_discourse`` -- NOT the
learned encoder) over real prose MC benchmarks (FairytaleQA entity items,
MCTest mc160), respecting each dataset's OWN story split, through the
existing resolver/QA head (``scripts/train_prose.py``'s model/training
loop). Reports, per dataset, on the SAME test items: random, the
``scripts/cheat_baselines_k12.py`` cheats, the trained model, and a
MEMORY-SEVERED control (the trained model with every passage WRITE step
masked out before the question step -- see :func:`sever_memory`) -- the
gate dev/AUDIT_2026-10-15.md names: test accuracy >= best cheat + 10pts AND
memory-severed accuracy within 3pts of random.

Differs from ``train_prose.py`` in exactly one structural way:
:func:`split_by_native_field` replaces that script's own
``split_by_document`` (a random holdout of documents) with the datasets'
OWN pre-assigned ``meta["split"]`` (FairytaleQA train/val/test,
MCTest train/dev) -- everything downstream (model config, training loop,
keep-best, eval_minibatched) is reused, not re-implemented.

Usage:
    python scripts/e2e_qa_spike.py --dataset ftqa \\
        --episodes runs/fairytaleqa_episodes_samepassage.jsonl \\
        --answer-type entity --save runs/e2e_qa_spike_ftqa.pt \\
        --report-json runs/e2e_qa_spike_ftqa.json

    python scripts/e2e_qa_spike.py --dataset mctest \\
        --episodes runs/mctest_mc160_episodes.jsonl \\
        --save runs/e2e_qa_spike_mctest.pt \\
        --report-json runs/e2e_qa_spike_mctest.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random as pyrandom
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
import torch.nn.functional as F

import cheat_baselines_k12 as cheats  # noqa: E402  (scripts/ sibling module)
import eval_prose  # noqa: E402
from train_prose import _safe_build_clause_batch  # noqa: E402
from _train_common import apply_threads, build_model, epoch_minibatches, eval_minibatched, peak_rss_mb  # noqa: E402
import nsm_ct.clause_reactor as creac  # noqa: E402
from nsm_ct.clause import extract_discourse  # noqa: E402
from nsm_ct.clause_reactor import build_clause_batch  # noqa: E402
from nsm_ct.episode import Episode  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402
from train_instances import RECENCY_EXTRA  # noqa: E402

AUX_WEIGHT = 0.5
load_episodes = eval_prose.load_episodes


# ---------------------------------------------------------------------------
# 0. parse-failure negative cache (perf only -- see module docstring on
#    _prose_parse_graph: the per-parser SUCCESS cache it already has does
#    nothing for a ParseResourceExceeded sentence, which is retried at full
#    cost every time it recurs across the many episodes that share a
#    passage). This wraps the SAME function with a (sentence -> exception)
#    cache so a capped sentence pays its ~10s wall-clock cap once, not once
#    per episode that shares its passage. No change to what gets built --
#    only to how many times a failing parse is attempted.
# ---------------------------------------------------------------------------
_orig_prose_parse_graph = creac._prose_parse_graph
_fail_cache: Dict[str, Exception] = {}


def _cached_prose_parse_graph(parser, sent):
    if sent in _fail_cache:
        raise _fail_cache[sent]
    try:
        return _orig_prose_parse_graph(parser, sent)
    except Exception as exc:  # noqa: BLE001 -- re-raised immediately, just memoized
        _fail_cache[sent] = exc
        raise


creac._prose_parse_graph = _cached_prose_parse_graph


# ---------------------------------------------------------------------------
# 1. load / filter / native-split / passage-count cap
# ---------------------------------------------------------------------------
def load_and_filter(path: str, answer_type: str) -> List[Episode]:
    eps = load_episodes(path)
    if answer_type:
        eps = [e for e in eps if e.meta.get("answer_type") == answer_type]
    eps = [e for e in eps if e.options is not None and e.answer_idx is not None]
    return eps


def split_by_native_field(episodes: List[Episode], field: str = "split") -> Dict[str, List[Episode]]:
    """Groups by ``meta[field]`` -- the dataset's OWN pre-assigned split
    (FairytaleQA train/val/test; MCTest train/dev), unlike
    ``train_prose.split_by_document``'s random per-run document holdout."""
    out: Dict[str, List[Episode]] = defaultdict(list)
    for e in episodes:
        out[str(e.meta.get(field, "?"))].append(e)
    return dict(out)


def _passage_key(ep: Episode) -> str:
    return ep.meta.get("story_id") or ep.meta.get("source_doc") or "|".join(ep.context)


def carve_val_from_train(train_eps: List[Episode], n_passages: int, seed: int) -> Tuple[List[Episode], List[Episode]]:
    """MCTest has no native val split (only train/dev -- dev is held for
    the FINAL test number, never touched for model selection). Carves a
    seeded subset of TRAIN's own passages out for keep-best instead of
    reusing dev as both val and test (which would leak the checkpoint
    SELECTION, not just the final number, into the reported test set).
    No-op (returns ``(train_eps, [])``) when ``n_passages <= 0``."""
    if n_passages <= 0:
        return train_eps, []
    passages = sorted({_passage_key(e) for e in train_eps})
    if len(passages) <= n_passages:
        return [], train_eps  # degenerate: everything becomes val (shouldn't happen in practice)
    rng = pyrandom.Random(seed + 777)
    val_passages = set(rng.sample(passages, n_passages))
    inner_train = [e for e in train_eps if _passage_key(e) not in val_passages]
    val_eps = [e for e in train_eps if _passage_key(e) in val_passages]
    return inner_train, val_eps


def cap_by_passage(episodes: List[Episode], max_passages: Optional[int], seed: int) -> List[Episode]:
    """Caps the number of DISTINCT passages (stories) kept -- not items --
    since parse cost is governed by unique sentences, which are shared by
    every item drawn from the same passage. ``None``/``0`` = no cap (keep
    every episode). Deterministic given ``(episodes, max_passages, seed)``."""
    if not max_passages:
        return episodes
    passages = sorted({_passage_key(e) for e in episodes})
    if len(passages) <= max_passages:
        return episodes
    rng = pyrandom.Random(seed)
    keep = set(rng.sample(passages, max_passages))
    return [e for e in episodes if _passage_key(e) in keep]


# ---------------------------------------------------------------------------
# 2. coverage + answerable-from-memory ceiling (per split, passage-deduped)
# ---------------------------------------------------------------------------
def _ctx_key(ep: Episode) -> tuple:
    """Keys the parse/token cache by the EXACT context sentence tuple, not
    ``_passage_key``'s story id -- two items on the same story can carry
    DIFFERENT context slices (FairytaleQA's per-section context), so
    deduping by story alone would silently skip-cache a sibling item's
    extra sentences (measured: this was the actual cause of a fallback
    build re-hitting ``ParseResourceExceeded`` on a sentence the coverage
    pass never actually saw -- see dev/E2E_QA_SPIKE.md's diagnosis).
    ``_passage_key`` (story-id-based) stays the right granularity for
    :func:`cap_by_passage`'s subsampling, where this finer key would just
    make every item its own singleton "passage"."""
    return tuple(ep.context)


def coverage_and_answerable(episodes: List[Episode], parser) -> dict:
    """Per split: items, distinct passages, % of passage SENTENCES that
    produced >=1 clause (the same :func:`_prose_parse_graph` +
    :func:`extract_discourse` call ``_prose_steps`` makes when actually
    writing memory -- not a separate/different parse path), and the
    answerable-from-memory CEILING: the % of items whose gold answer has at
    least one (lowercased) token among the tokens ``_prose_steps`` would
    actually write to memory for that passage (every clause's predicate +
    argument tokens, pooled across the whole passage). Passage-deduped (one
    parse per distinct passage, reused across every item on it) via the
    module's own failure-negative-cache + the parser's built-in success
    cache -- this is also what pre-warms both caches before batch-building
    below, so that step pays almost nothing extra."""
    passages: Dict[tuple, List[str]] = {}
    for e in episodes:
        k = _ctx_key(e)
        if k not in passages:
            passages[k] = list(e.context)
    n_distinct_passages = len({_passage_key(e) for e in episodes})

    n_sent = 0
    n_ok = 0
    passage_tokens: Dict[tuple, set] = {}
    for key, ctx in passages.items():
        toks: set = set()
        for sent in ctx:
            n_sent += 1
            try:
                graph = creac._prose_parse_graph(parser, sent)
                clauses, _links = extract_discourse(graph)
            except Exception:  # noqa: BLE001 -- a capped/failed sentence: no clauses, no tokens
                clauses = []
            if clauses:
                n_ok += 1
            for cl in clauses:
                if cl.predicate:
                    toks.add(cl.predicate.lower())
                for _rel, arg in cl.args:
                    if arg.token:
                        toks.add(arg.token.lower())
        passage_tokens[key] = toks

    n_answerable = 0
    for e in episodes:
        gold_toks = set(e.answer_text.lower().split())
        if gold_toks & passage_tokens[_ctx_key(e)]:
            n_answerable += 1

    return {
        "items": len(episodes),
        "passages": n_distinct_passages,
        "distinct_context_variants": len(passages),
        "sentence_clause_coverage": (n_ok / n_sent) if n_sent else float("nan"),
        "n_sentences": n_sent,
        "answerable_from_memory_ceiling": (n_answerable / len(episodes)) if episodes else float("nan"),
    }


# ---------------------------------------------------------------------------
# 3. build model batches (parser cache is already warm from step 2 above)
# ---------------------------------------------------------------------------
def _has_known_bad_sentence(ep: Episode) -> bool:
    """True iff the coverage pass (:func:`coverage_and_answerable`, which
    touches every sentence of every episode before this is ever called)
    already recorded a parse failure for one of ``ep.context``'s sentences
    in the module's negative cache."""
    return any(s in _fail_cache for s in ep.context)


def build_split_batch(episodes: List[Episode], parser, resolver, codec):
    """``train_prose._safe_build_clause_batch`` (reused, not duplicated) --
    but pre-filtered: an episode already known (from the coverage pass) to
    contain a capped/failing sentence is dropped BEFORE the optimistic
    whole-split batch call, instead of being allowed to blow that call and
    force the slow fallback. The fallback
    (``eval_prose.partition_buildable``'s per-episode probe) builds ONE
    FRESH content-vector/USVS-grounding cache per episode (``build_clause_
    batch`` only shares its cache dict across episodes PASSED IN THE SAME
    CALL) -- measured at ~3s/episode even with every sentence parse-cache
    warm (dev/E2E_QA_SPIKE.md's diagnosis), against a shared-cache whole-
    batch call that builds the same split in a few seconds total. Avoiding
    the fallback for every PREDICTABLE failure (anything the coverage pass
    already saw fail) keeps the optimistic, fast, shared-cache path as the
    common case; only a build failure that ISN'T explained by a known-bad
    sentence (should not occur, see docstring above) still pays the slow
    fallback, on whatever residual it's left to probe."""
    if not episodes:
        return None, [], []
    pre_failed = [e for e in episodes if _has_known_bad_sentence(e)]
    survivors = [e for e in episodes if not _has_known_bad_sentence(e)]
    batch, buildable = (_safe_build_clause_batch(survivors, parser, resolver, codec)
                         if survivors else (None, []))
    kept_ids = {id(e) for e in buildable}
    residual_failed = [e for e in survivors if id(e) not in kept_ids] if batch is not None else survivors
    failures = ([(e, "known-bad sentence (coverage pass)") for e in pre_failed]
                + [(e, "dropped by _safe_build_clause_batch") for e in residual_failed])
    return batch, buildable, failures


# ---------------------------------------------------------------------------
# 4. memory-severed control
# ---------------------------------------------------------------------------
def sever_memory(batch):
    """A clone of ``batch`` with every passage WRITE step's mask zeroed --
    ``forward()``'s own write gate is ``stmt = real * (1 - is_q)``
    (clause_reactor.py), so ``mask := mask * is_q`` makes every write step
    contribute nothing: ``memory`` stays at its ``em.init_memory`` zero
    state all the way to the question step. The question step itself is
    untouched (still masked real, still is_q=1), so the model still runs
    end to end and still scores ``answer_logits`` against the SAME
    options -- it just never saw the passage. This is "no change to the
    model, sever the input" rather than a second code path."""
    severed_mask = batch.mask * batch.is_q
    return dataclasses.replace(batch, mask=severed_mask)


# ---------------------------------------------------------------------------
# 5. training loop (mirrors train_prose.py's run(), native split instead of
#    split_by_document; keep-best on val/dev instead of a single final eval)
# ---------------------------------------------------------------------------
def run(args) -> dict:
    t_start = time.time()
    episodes = load_and_filter(args.episodes, args.answer_type)
    print(f"=== e2e_qa_spike[{args.dataset}]: {len(episodes)} scorable episodes "
          f"loaded from {args.episodes} ===", flush=True)

    by_split = split_by_native_field(episodes, "split")
    print(f"=== native splits: {[(k, len(v)) for k, v in sorted(by_split.items())]} ===", flush=True)

    train_key, val_key, test_key = args.train_split, args.val_split, args.test_split
    raw_train_eps = by_split.get(train_key, [])
    if args.val_from_train_passages > 0:
        # No native val split (MCTest: train/dev only) -- carve val out of
        # TRAIN itself so keep-best is never selected against the same
        # episodes used for the final test number (see carve_val_from_train).
        raw_train_eps, raw_val_eps = carve_val_from_train(raw_train_eps, args.val_from_train_passages, args.seed)
    else:
        raw_val_eps = by_split.get(val_key, [])
    train_eps = cap_by_passage(raw_train_eps, args.train_passage_cap, args.seed)
    val_eps = cap_by_passage(raw_val_eps, args.val_passage_cap, args.seed)
    test_eps = cap_by_passage(by_split.get(test_key, []), args.test_passage_cap, args.seed)
    print(f"=== after passage cap: train={len(train_eps)} ({len({_passage_key(e) for e in train_eps})} passages) "
          f"val={len(val_eps)} ({len({_passage_key(e) for e in val_eps})} passages) "
          f"test={len(test_eps)} ({len({_passage_key(e) for e in test_eps})} passages) ===", flush=True)

    all_eps = train_eps + val_eps + test_eps
    texts = [t for e in all_eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok, lang="en")
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping.")
        return {}
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=args.dim, max_pos=64)

    print("=== coverage + answerable-from-memory (passage-deduped parse; also warms the "
          "parse cache for batch-building below) ===", flush=True)
    coverage = {}
    for name, eps in (("train", train_eps), ("val", val_eps), ("test", test_eps)):
        t0 = time.time()
        coverage[name] = coverage_and_answerable(eps, parser)
        coverage[name]["elapsed_sec"] = round(time.time() - t0, 1)
        print(f"  {name:5s}: {coverage[name]}", flush=True)

    print("=== building model batches (parse cache warm from coverage pass) ===", flush=True)
    t0 = time.time()
    train_batch, train_eps_b, train_fail = build_split_batch(train_eps, parser, resolver, codec)
    val_batch, val_eps_b, val_fail = build_split_batch(val_eps, parser, resolver, codec)
    test_batch, test_eps_b, test_fail = build_split_batch(test_eps, parser, resolver, codec)
    build_elapsed = time.time() - t0
    print(f"  build-failures: train={len(train_fail)}/{len(train_eps)} val={len(val_fail)}/{len(val_eps)} "
          f"test={len(test_fail)}/{len(test_eps)}  (build elapsed {build_elapsed:.1f}s)", flush=True)

    gold_train = torch.tensor([e.answer_idx for e in train_eps_b]) if train_eps_b else None
    gold_val = torch.tensor([e.answer_idx for e in val_eps_b]) if val_eps_b else None
    gold_test = torch.tensor([e.answer_idx for e in test_eps_b]) if test_eps_b else None

    model_config = {
        "dim": args.dim, "hidden": args.hidden, "track": "A",
        "use_cand_feature": True, "cand_feature_extra": 1 + RECENCY_EXTRA,
        "evidence_prior_beta": None,
    }
    torch.manual_seed(args.seed)
    model = build_model(model_config)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    best_val_acc = -1.0
    best_state = None
    losses = []
    t0 = time.time()
    if train_batch is not None:
        n_tr = len(train_eps_b)
        model.train()
        for i in range(args.epochs):
            epoch_losses = []
            for mb_idx in epoch_minibatches(n_tr, args.batch_size, args.seed, i):
                idx_t = torch.from_numpy(mb_idx)
                sub = train_batch.subset(idx_t)
                sub_gold = gold_train[idx_t]
                out = model(sub)
                loss = F.cross_entropy(out["answer_logits"], sub_gold)
                if model.resolver is not None and "resolver_logits" in out:
                    cg = sub.cand_gold
                    has_cand = cg >= 0
                    if bool(has_cand.any()):
                        aux = F.cross_entropy(out["resolver_logits"][has_cand], cg[has_cand])
                        loss = loss + AUX_WEIGHT * aux
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_losses.append(float(loss.item()))
            losses.append(epoch_losses[-1])
            if val_batch is not None:
                model.eval()
                val_pred = eval_minibatched(model, val_batch, args.batch_size)["answer_logits"].argmax(-1)
                val_acc = float((val_pred == gold_val).float().mean())
                model.train()
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (i + 1) % args.log_interval == 0 or i == 0 or i == args.epochs - 1:
                print(f"  epoch {i + 1:3d} loss={losses[-1]:.3f} val_acc={best_val_acc:.3f}", flush=True)
    train_elapsed_min = (time.time() - t0) / 60.0

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cleanup = True
    model.eval()

    report: dict = {"dataset": args.dataset, "coverage": coverage,
                     "n_train": len(train_eps_b) if train_eps_b else 0,
                     "n_val": len(val_eps_b) if val_eps_b else 0,
                     "n_test": len(test_eps_b) if test_eps_b else 0,
                     "best_val_acc": best_val_acc, "train_elapsed_min": round(train_elapsed_min, 2),
                     "build_elapsed_sec": round(build_elapsed, 1),
                     "total_elapsed_min": round((time.time() - t_start) / 60.0, 2)}

    if test_batch is not None:
        out = eval_minibatched(model, test_batch, args.batch_size)
        pred = out["answer_logits"].argmax(-1)
        correct = (pred == gold_test)
        model_acc = float(correct.float().mean())

        severed_batch = sever_memory(test_batch)
        out_sev = eval_minibatched(model, severed_batch, args.batch_size)
        pred_sev = out_sev["answer_logits"].argmax(-1)
        severed_acc = float((pred_sev == gold_test).float().mean())

        floor = eval_prose.random_guess_floor(test_eps_b)
        answerable_mask = [bool(set(e.answer_text.lower().split())
                                 & _passage_answer_tokens(e, parser)) for e in test_eps_b]
        n_answerable = sum(answerable_mask)
        if n_answerable:
            acc_answerable = float(correct[torch.tensor(answerable_mask)].float().mean())
        else:
            acc_answerable = float("nan")

        cheat_report = cheats.report(f"{args.dataset} cheat baselines (test/dev)", test_eps_b, seed=args.seed)
        best_cheat = max(cheat_report.get("passage_substring", 0.0), cheat_report.get("word_overlap", 0.0))

        gate_acc = model_acc >= best_cheat + 0.10
        gate_severed = abs(severed_acc - floor) <= 0.03
        gate_pass = gate_acc and gate_severed

        report.update({
            "random_floor": floor,
            "cheat_baselines": cheat_report,
            "best_cheat": best_cheat,
            "model_acc": model_acc,
            "memory_severed_acc": severed_acc,
            "n_answerable_subset": n_answerable,
            "acc_on_answerable_subset": acc_answerable,
            "gate_acc_vs_cheat": gate_acc,
            "gate_severed_vs_random": gate_severed,
            "GATE_PASS": gate_pass,
        })

        print(f"\n=== {args.dataset} RESULTS (test/dev, n={len(test_eps_b)}) ===")
        print(f"  random floor          : {floor:.3f}")
        print(f"  best cheat             : {best_cheat:.3f}")
        print(f"  model accuracy         : {model_acc:.3f}")
        print(f"  memory-severed accuracy: {severed_acc:.3f}")
        print(f"  answerable subset      : {n_answerable}/{len(test_eps_b)}  acc={acc_answerable:.3f}")
        print(f"  GATE (model>=cheat+0.10 AND severed within 0.03 of random): "
              f"{'PASS' if gate_pass else 'FAIL'}")
    else:
        print("no test/dev batch built -- nothing to score.")

    rss = peak_rss_mb()
    report["peak_rss_mb"] = rss
    print(f"\npeak_rss_mb: {rss:.1f}  train_elapsed_min: {train_elapsed_min:.2f}  "
          f"total_elapsed_min: {report['total_elapsed_min']:.2f}", flush=True)

    if args.save and best_state is not None:
        from nsm_ct.checkpoint import save_checkpoint, git_commit
        cfg = dict(model_config)
        cfg.update({"codec_dim": args.dim, "codec_max_pos": 64, "meaning_source": "usvs",
                    "git_commit": git_commit(), "argv": sys.argv[:]})
        save_checkpoint(args.save, model, config=cfg, extra={"report": report})
        print(f"  saved checkpoint -> {args.save} ({os.path.getsize(args.save)} bytes)", flush=True)

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"  wrote report -> {args.report_json}", flush=True)

    return report


def _passage_answer_tokens(ep: Episode, parser) -> set:
    """Re-derives the one passage's written-memory token set for a single
    episode (used only for the answerable-subset accuracy slice above --
    the bulk per-split computation is :func:`coverage_and_answerable`).
    Cheap: the parser's own success/failure caches make every sentence here
    an O(1) cache hit by this point in the run."""
    toks: set = set()
    for sent in ep.context:
        try:
            graph = creac._prose_parse_graph(parser, sent)
            clauses, _links = extract_discourse(graph)
        except Exception:  # noqa: BLE001
            clauses = []
        for cl in clauses:
            if cl.predicate:
                toks.add(cl.predicate.lower())
            for _rel, arg in cl.args:
                if arg.token:
                    toks.add(arg.token.lower())
    return toks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ftqa", "mctest"], required=True)
    ap.add_argument("--episodes", type=str, required=True)
    ap.add_argument("--answer-type", type=str, default="", help="'entity' for FairytaleQA; '' (no filter) for MCTest.")
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--val-split", type=str, default="val")
    ap.add_argument("--test-split", type=str, default="test")
    ap.add_argument("--val-from-train-passages", type=int, default=0,
                     help="MCTest hook: carve this many passages OUT of --train-split itself for "
                          "keep-best (see carve_val_from_train), instead of reading --val-split. "
                          "Use when the dataset has no native val split (MCTest: train/dev only) -- "
                          "keeps model SELECTION off the final test/dev set. 0 = use --val-split as-is.")
    ap.add_argument("--train-passage-cap", type=int, default=0, help="0 = no cap (every passage in the split).")
    ap.add_argument("--val-passage-cap", type=int, default=0)
    ap.add_argument("--test-passage-cap", type=int, default=0)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--report-json", type=str, default=None)
    args = ap.parse_args()
    apply_threads(args)
    run(args)


if __name__ == "__main__":
    main()
