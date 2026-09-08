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


def evaluate_full(model, records: Sequence[dict], usvs, pos_vocab, hash_buckets: int,
                   beam_width: int = 6, k: int = 6, policy: str = "model",
                   rng: Optional[random.Random] = None) -> Dict[str, float]:
    """`em.evaluate`'s best-of-k oracle metrics PLUS the audit's rank-1
    committed-tree edge P/R/F1 and mean forest width (dev/AUDIT_2026-09-08.md
    finding 5 + recommendation (b); ported from the Part A section added to
    scripts/rescore_encoder.py on branch `encoder-complement-probe`). A
    forest of up to k trees is decoded ONCE per record; `em.score_record` is
    applied both to the whole forest (best-of-k, matching `em.evaluate`
    exactly) and to just its rank-1 (highest-logprob) tree, so every arm
    reports the "encoder works" number alongside the number that survives
    commitment to a single hypothesis."""
    if not records:
        empty = em.aggregate_recall([])
        empty.update({
            "rank1_edge_precision": float("nan"),
            "rank1_edge_recall": float("nan"),
            "rank1_edge_f1": float("nan"),
            "rank1_structure_recall": float("nan"),
            "mean_forest_width": float("nan"),
        })
        return empty

    scores = []
    rank1_scores = []
    widths = []
    for record in records:
        feats = em.build_features(record, usvs, pos_vocab, hash_buckets)
        forest = em.beam_decode(model, feats, beam_width=beam_width, k=k, policy=policy, rng=rng)
        scores.append(em.score_record(record, forest))
        rank1_scores.append(em.score_record(record, [forest[0]] if forest else []))
        widths.append(len(forest))

    agg = em.aggregate_recall(scores)
    rank1_agg = em.aggregate_recall(rank1_scores)
    agg["rank1_edge_precision"] = rank1_agg["edge_precision"]
    agg["rank1_edge_recall"] = rank1_agg["edge_recall"]
    agg["rank1_edge_f1"] = _f1(rank1_agg["edge_precision"], rank1_agg["edge_recall"])
    agg["rank1_structure_recall"] = rank1_agg["structure_recall"]
    agg["mean_forest_width"] = statistics.mean(widths) if widths else float("nan")
    return agg


def run_training_loop(model, train_items: list, opt, *, epochs: int, batch_size: int,
                       max_seconds: float, max_steps: Optional[int] = None,
                       terminal_weight: float = 4.0,
                       on_step_50: Optional[Callable] = None,
                       on_epoch_done: Optional[Callable] = None,
                       on_max_seconds: Optional[Callable] = None) -> dict:
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
    finding 7)."""
    loss_curve = []
    step_count = 0
    optimizer_steps = 0
    train_start = time.time()
    stopped_early = False
    stop_reason = None
    total_items = len(train_items)

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
        for idx, (feats, steps) in enumerate(train_items):
            if time.time() - train_start > max_seconds:
                stopped_early = True
                stop_reason = "max_seconds"
                break
            loss = em.teacher_force_loss(model, feats, steps, terminal_weight=terminal_weight) / batch_size
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
            if step_count % 50 == 0:
                avg = epoch_loss / max(epoch_n, 1)
                loss_curve.append((step_count, avg))
                if on_step_50:
                    on_step_50(epoch, step_count, avg)
            if step_budget_hit:
                stopped_early = True
                stop_reason = "max_steps"
                break
        # Unconditional per-epoch flush of a partial trailing batch -- SKIPPED
        # only when the inner loop already stopped exactly on the
        # --max-steps cap: that break happens right after a real opt.step()
        # + opt.zero_grad() (no leftover accumulated gradient), so an extra
        # flush here would silently perform one more optimizer step on
        # stale/zero gradients, violating "train for exactly N optimizer
        # steps." A max-seconds break can land mid-batch with real
        # accumulated gradient, so it still gets the flush.
        do_trailing_step = not (max_steps is not None and stop_reason == "max_steps")
        if do_trailing_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            opt.zero_grad()
            optimizer_steps += 1
        avg = epoch_loss / max(epoch_n, 1)
        loss_curve.append((step_count, avg))
        if on_epoch_done:
            on_epoch_done(epoch, avg, epoch_n)
        if max_steps is not None and optimizer_steps >= max_steps:
            stopped_early = True
            stop_reason = stop_reason or "max_steps"
        if stopped_early:
            break

    train_wall = time.time() - train_start
    epoch_fraction = (optimizer_steps * batch_size / total_items) if total_items else float("nan")
    return {
        "loss_curve": loss_curve,
        "step_count": step_count,
        "optimizer_steps": optimizer_steps,
        "epoch_fraction": epoch_fraction,
        "train_wall": train_wall,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }
