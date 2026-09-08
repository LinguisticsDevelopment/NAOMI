"""Shared code between scripts/train_encoder.py and
scripts/colab_train_encoder.py: gold loading, the forest-width-stratified
split (spec S6), seeded pool subsetting, the shared held-out-sentence eval
path, and the teacher-forced training loop with an optional exact
optimizer-step budget.

dev/AUDIT_2026-09-08.md finding 7 + recommendation (c): the encoder's
data-limited verdict rested on FIXED-EPOCH runs (n=788 got ~4x the gradient
steps of n=200), and the v3-gold retrain bundled top-1 pruning + richer
extraction + an 11x corpus into one uncontrolled run. This module gives
scripts/run_encoder_arms.sh arms that hold the optimization budget and the
eval sentences fixed while varying only the gold file / n_train, so "more
data" and "cleaner data" are measured separately, on the same held-out
sentences.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from nsm_ct import encoder_model as em


def load_gold(path: str) -> list:
    """`path` is one file, or a comma-separated list (e.g. a teacher gold file
    plus `runs/hard_gold_train.jsonl`): files are concatenated in order and
    deduped by surface `text`, keeping the first occurrence -- so a hard-gold
    file placed after teacher gold never overrides an existing record, and a
    file listed twice is a no-op rather than a duplicate."""
    records = []
    seen_text = set()
    for p in path.split(","):
        p = p.strip()
        if not p:
            continue
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                text = rec.get("text")
                if text in seen_text:
                    continue
                seen_text.add(text)
                records.append(rec)
    return records


def forest_width_bucket(record: dict) -> int:
    n = len(record["lattice"]["trees"])
    if n <= 1:
        return 0
    if n <= 3:
        return 1
    return 2


def _stratified_order(records: list, seed: int) -> list:
    """Seeded forest-width-stratified interleave (spec S6): buckets =
    {1-tree, 2-3 trees, 4+ trees}, shuffled within bucket, then round-robin
    merged (proportional representation) so any prefix of the returned
    order sees all three widths."""
    rng = random.Random(seed)
    buckets: Dict[int, list] = {0: [], 1: [], 2: []}
    for r in records:
        buckets[forest_width_bucket(r)].append(r)
    for b in buckets.values():
        rng.shuffle(b)

    order = []
    idxs = {0: 0, 1: 0, 2: 0}
    total = sum(len(b) for b in buckets.values())
    while len(order) < total:
        for k in (0, 1, 2):
            if idxs[k] < len(buckets[k]):
                order.append(buckets[k][idxs[k]])
                idxs[k] += 1
    return order


def stratified_split(records: list, seed: int, n_train: int, n_dev: int, n_test: int) -> tuple:
    """Seeded, forest-width-stratified split (spec S6): buckets = {1-tree,
    2-3 trees, 4+ trees}, shuffled within bucket, then interleaved so
    train/dev/test each see all three widths, disjoint records throughout."""
    order = _stratified_order(records, seed)
    train = order[:n_train]
    dev = order[n_train:n_train + n_dev]
    test = order[n_train + n_dev:n_train + n_dev + n_test]
    return train, dev, test


def seeded_subset(records: list, n: int, seed: int) -> list:
    """Seeded, forest-width-stratified subset of `records`, size
    min(n, len(records)) (--n-train/--subset-seed, item B): the same
    stratification `stratified_split` uses, keyed by its OWN seed so
    v3@788 and v3@3000 are independent, reproducible draws from the same
    pool -- not a prefix relationship with a `stratified_split` call keyed
    by the training seed proper."""
    order = _stratified_order(records, seed)
    return order[:n]


def load_holdout_sentences(path: str) -> List[str]:
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def exclude_by_text(records: Sequence[dict], sentences: Sequence[str]) -> List[dict]:
    """Training pool for --holdout-file (item C): every record whose `text`
    is NOT one of the held-out sentences."""
    excluded = set(sentences)
    return [r for r in records if r["text"] not in excluded]


def records_for_sentences(gold_records: Sequence[dict], sentences: Sequence[str]) -> Tuple[List[dict], int]:
    """Look up each held-out sentence's gold record(s) by exact `text` match
    in `gold_records` (item C). A sentence missing from this particular gold
    file is counted in the returned n_missing, never a crash -- v3-trained
    arms are expected to be scored against v2 targets too, and not every
    v2 test sentence need appear verbatim in a v3 gold file."""
    by_text: Dict[str, List[dict]] = {}
    for r in gold_records:
        by_text.setdefault(r["text"], []).append(r)
    matched: List[dict] = []
    n_missing = 0
    for s in sentences:
        hits = by_text.get(s)
        if hits:
            matched.extend(hits)
        else:
            n_missing += 1
    return matched, n_missing


def build_train_items(records: Sequence[dict], usvs, pos_vocab, hash_buckets: int) -> list:
    items = []
    for r in records:
        feats = em.build_features(r, usvs, pos_vocab, hash_buckets)
        for tree in r["lattice"]["trees"]:
            steps = em.linearize_tree(r, tree)
            items.append((feats, steps))
    return items


def _f1(p: float, r: float) -> float:
    if p != p or r != r or (p + r) == 0:
        return float("nan")
    return 2 * p * r / (p + r)


#: The graded-metric field names `evaluate_full(metric=...)` adds, in both
#: the best-of-k and the rank-1 (`rank1_`-prefixed) views. See
#: dev/USVS_GRADED_SCORING.md S3.
GRADED_FIELDS = ("graded_p", "graded_r", "graded_f", "graded_overall",
                  "clause_count", "clause_kind", "clause_struct")


def _empty_graded(prefix: str = "") -> Dict[str, float]:
    return {prefix + k: float("nan") for k in GRADED_FIELDS}


def evaluate_full(model, records: Sequence[dict], usvs, pos_vocab, hash_buckets: int,
                   beam_width: int = 6, k: int = 6, policy: str = "model",
                   rng: Optional[random.Random] = None,
                   metric: str = "edge") -> Dict[str, float]:
    """`em.evaluate`'s best-of-k oracle metrics PLUS the audit's rank-1
    committed-tree edge P/R/F1 and mean forest width (dev/AUDIT_2026-09-08.md
    finding 5 + recommendation (b); ported from the Part A section added to
    scripts/rescore_encoder.py on branch `encoder-complement-probe`). A
    forest of up to k trees is decoded ONCE per record; `em.score_record` is
    applied both to the whole forest (best-of-k, matching `em.evaluate`
    exactly) and to just its rank-1 (highest-logprob) tree, so every arm
    reports the "encoder works" number alongside the number that survives
    commitment to a single hypothesis.

    `metric` (lead directive 2026-09-08; dev/USVS_GRADED_SCORING.md):
    `"edge"` (default) is the original binary edge-F1 only and is unchanged;
    `"graded"` ADDS the USVS-graded P/R/F fields (same two views, `graded_*`
    and `rank1_graded_*`) computed on the SAME decoded forests -- nothing is
    re-decoded and no existing field changes value; `"both"` is a synonym,
    kept so a caller can be explicit that it wants the old numbers too."""
    want_graded = metric in ("graded", "both")
    if not records:
        empty = em.aggregate_recall([])
        empty.update({
            "rank1_edge_precision": float("nan"),
            "rank1_edge_recall": float("nan"),
            "rank1_edge_f1": float("nan"),
            "rank1_structure_recall": float("nan"),
            "mean_forest_width": float("nan"),
        })
        if want_graded:
            empty.update(_empty_graded())
            empty.update(_empty_graded("rank1_"))
        return empty

    if want_graded:
        from nsm_ct import usvs_graded as ug

    scores = []
    rank1_scores = []
    widths = []
    graded_scores = []
    rank1_graded_scores = []
    for record in records:
        feats = em.build_features(record, usvs, pos_vocab, hash_buckets)
        forest = em.beam_decode(model, feats, beam_width=beam_width, k=k, policy=policy, rng=rng)
        rank1 = [forest[0]] if forest else []
        scores.append(em.score_record(record, forest))
        rank1_scores.append(em.score_record(record, rank1))
        widths.append(len(forest))
        if want_graded:
            graded_scores.append(ug.score_record_graded(record, forest, usvs))
            rank1_graded_scores.append(ug.score_record_graded(record, rank1, usvs))

    agg = em.aggregate_recall(scores)
    rank1_agg = em.aggregate_recall(rank1_scores)
    agg["rank1_edge_precision"] = rank1_agg["edge_precision"]
    agg["rank1_edge_recall"] = rank1_agg["edge_recall"]
    agg["rank1_edge_f1"] = _f1(rank1_agg["edge_precision"], rank1_agg["edge_recall"])
    agg["rank1_structure_recall"] = rank1_agg["structure_recall"]
    agg["mean_forest_width"] = statistics.mean(widths) if widths else float("nan")
    if want_graded:
        g = ug.aggregate_graded(graded_scores)
        g1 = ug.aggregate_graded(rank1_graded_scores)
        for key in GRADED_FIELDS:
            agg[key] = g[key]
            agg["rank1_" + key] = g1[key]
    return agg


def epochs_equivalent(step: int, batch_size: int, total_items: int) -> float:
    """How many passes over `total_items` `step` optimizer steps of
    `batch_size` amount to. v3 gold's ~1 derivation/record vs v2 forest
    gold's ~3.3 means the SAME --max-steps budget on the SAME n_train is a
    wildly different number of epochs over the corpus (WHY:
    encoder-train-arms-v2) -- this is the number that makes that visible,
    for whatever step a checkpoint (best or last) was actually taken at."""
    return (step * batch_size / total_items) if total_items else float("nan")


def evaluate_dev_fast(model, records: Sequence[dict], usvs, pos_vocab, hash_buckets: int) -> Dict[str, float]:
    """Cheap periodic dev-holdout eval for --eval-every: rank-1 edge-F1 with
    a single-hypothesis `beam_decode` (beam_width=1, k=1) -- no beam search
    branching, no best-of-k forest, just the argmax decode -- so it's cheap
    enough to run every N optimizer steps without materially slowing down
    training. With k=1 the "best-of-k" and "rank-1" numbers from
    `evaluate_full` coincide; only the rank-1 fields are meaningful here."""
    return evaluate_full(model, records, usvs, pos_vocab, hash_buckets, beam_width=1, k=1, policy="model")


def run_training_loop(model, train_items: list, opt, *, epochs: int, batch_size: int,
                       max_seconds: float, max_steps: Optional[int] = None,
                       terminal_weight: float = 4.0,
                       soft_targets=None,
                       on_step_50: Optional[Callable] = None,
                       on_epoch_done: Optional[Callable] = None,
                       on_max_seconds: Optional[Callable] = None,
                       on_optimizer_step: Optional[Callable[[int, float, float], Optional[bool]]] = None) -> dict:
    """Teacher-forced training loop shared by scripts/train_encoder.py and
    scripts/colab_train_encoder.py. Reproduces both scripts' original
    inline loops EXACTLY (same control flow, same clip/step/zero_grad
    placement) when `max_steps` is None -- item A requires existing
    behavior without --max-steps to be byte-identical, so all logging stays
    with the caller via the `on_*` hooks (each script keeps its own
    timestamp/format) and this function only touches loop control and step
    accounting.

    When `max_steps` is given, it caps the number of completed *optimizer
    steps* (gradient updates -- i.e. `opt.step()` calls, including the
    unconditional per-epoch flush of a partial trailing batch) rather than
    epochs; `epochs` becomes a ceiling only, so v3@788 and v3@3000 can be
    trained to the SAME optimization budget (dev/AUDIT_2026-09-08.md
    finding 7).

    `soft_targets` (default `None`) is passed straight through to
    `em.teacher_force_loss` -- an `em.SoftTargetConfig` enables the opt-in
    USVS/semantics-graded soft CE targets (`--loss usvs-soft`;
    dev/USVS_GRADED_SCORING.md S5). `None` keeps the original one-hot loss,
    numerically identical.

    `on_optimizer_step`, if given, is called after EVERY completed
    optimizer step (both mid-epoch batches and the per-epoch trailing
    partial-batch flush) with `(optimizer_steps, avg_loss_so_far,
    train_wall_s)` -- the periodic-dev-eval / --keep-best / --patience hook
    (train-arms v2 item A/B). Returning a truthy value requests an early
    stop, handled exactly like hitting --max-steps (stop_reason=
    "patience"). Defaulting to None keeps every existing caller (and the
    no-flags path) byte-identical: the hook is simply never invoked."""
    loss_curve = []
    step_count = 0
    optimizer_steps = 0
    train_start = time.time()
    stopped_early = False
    stop_reason = None
    total_items = len(train_items)

    def _fire_on_optimizer_step() -> bool:
        """Calls `on_optimizer_step` (if given) with the current
        optimizer-step count / running avg loss / wall clock; returns
        whether it requested an early stop."""
        if on_optimizer_step is None:
            return False
        avg_so_far = epoch_loss / max(epoch_n, 1)
        return bool(on_optimizer_step(optimizer_steps, avg_so_far, time.time() - train_start))

    for epoch in range(epochs):
        if time.time() - train_start > max_seconds:
            stopped_early = True
            stop_reason = "max_seconds"
            if on_max_seconds:
                on_max_seconds(epoch)
            break
        random.shuffle(train_items)
        epoch_loss = 0.0
        epoch_n = 0
        opt.zero_grad()
        step_budget_hit = False
        budget_hit_reason = "max_steps"
        for idx, (feats, steps) in enumerate(train_items):
            if time.time() - train_start > max_seconds:
                stopped_early = True
                stop_reason = "max_seconds"
                break
            loss = em.teacher_force_loss(model, feats, steps, terminal_weight=terminal_weight,
                                          soft_targets=soft_targets) / batch_size
            loss.backward()
            epoch_loss += float(loss.item()) * batch_size
            epoch_n += 1
            step_count += 1
            if (idx + 1) % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
                optimizer_steps += 1
                if max_steps is not None and optimizer_steps >= max_steps:
                    step_budget_hit = True
                patience_stop = _fire_on_optimizer_step()
                if patience_stop and not step_budget_hit:
                    step_budget_hit = True
                    budget_hit_reason = "patience"
            if step_count % 50 == 0:
                avg = epoch_loss / max(epoch_n, 1)
                loss_curve.append((step_count, avg))
                if on_step_50:
                    on_step_50(epoch, step_count, avg)
            if step_budget_hit:
                stopped_early = True
                stop_reason = budget_hit_reason
                break
        # Unconditional per-epoch flush of a partial trailing batch -- SKIPPED
        # only when the inner loop already stopped exactly on the
        # --max-steps cap (or an --patience stop, same shape): that break
        # happens right after a real opt.step() + opt.zero_grad() (no
        # leftover accumulated gradient), so an extra flush here would
        # silently perform one more optimizer step on stale/zero gradients,
        # violating "train for exactly N optimizer steps." A max-seconds
        # break can land mid-batch with real accumulated gradient, so it
        # still gets the flush.
        do_trailing_step = stop_reason not in ("max_steps", "patience")
        trailing_patience_stop = False
        if do_trailing_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            opt.zero_grad()
            optimizer_steps += 1
            trailing_patience_stop = _fire_on_optimizer_step()
        avg = epoch_loss / max(epoch_n, 1)
        loss_curve.append((step_count, avg))
        if on_epoch_done:
            on_epoch_done(epoch, avg, epoch_n)
        if max_steps is not None and optimizer_steps >= max_steps:
            stopped_early = True
            stop_reason = stop_reason or "max_steps"
        if trailing_patience_stop and not stopped_early:
            stopped_early = True
            stop_reason = "patience"
        if stopped_early:
            break

    train_wall = time.time() - train_start
    epoch_fraction = epochs_equivalent(optimizer_steps, batch_size, total_items)
    return {
        "loss_curve": loss_curve,
        "step_count": step_count,
        "optimizer_steps": optimizer_steps,
        "epoch_fraction": epoch_fraction,
        "train_wall": train_wall,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }
