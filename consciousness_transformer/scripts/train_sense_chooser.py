"""M32.1 — train the sense chooser, close the M32 gap.

``probe_m32_ambiguity.py`` measured the gap with no learning at all: MFS
grounding scores 0.247 on the sense-flipped half of
``episode.generate_ambiguity_episodes``; the gold-sense oracle scores 1.000.
This script trains ``nsm_ct.sense_chooser.SenseChooser`` — a <50k-param policy
over USVS-space sense vectors — supervised on the gold sense ids, and reports
its score on that SAME benchmark (reusing ``probe_m32_ambiguity.score_episode``
so the numbers are directly comparable to the floor/ceiling).

Two evaluations:

1. **In-distribution val** (Part 2): train/val split by episode seed
   (scaled with the family count so per-family density stays roughly
   constant), report accuracy overall and on the flipped half vs the
   0.247 floor / 1.000 ceiling.

2. **Leave-one-family-out** (Part 3, the scientifically decisive one): train
   on N-1 of the N homograph families in ``episode._AMBIGUITY_FAMILIES``,
   evaluate on the held-out one, for EVERY rotation (N of them — originally
   4, now however many families the M32.2 batch grew the curriculum to). A
   held-out family the chooser still resolves means it learned *how* to
   match a context vector to a sense vector in USVS geometry (generalizes);
   flopping to floor on the held-out family would mean it only learned
   *which* sense each specific word means (memorization — does not
   generalize). This is also the direct test of the "too few shots" fix:
   the original 4-family run left "bank" stuck at floor in its one rotation;
   with N-1 training families instead of 3, does bank's (and everyone
   else's) rotation clear the floor?

Run:
    python scripts/train_sense_chooser.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nsm_ct.episode import _AMBIGUITY_FAMILIES, generate_ambiguity_episodes  # noqa: E402
from nsm_ct.sense_chooser import (  # noqa: E402
    SenseChooser,
    build_example,
    collate,
    predicted_sense_ids,
)
from nsm_ct.usvs_bridge import usvs_sense_handle  # noqa: E402

from probe_m32_ambiguity import score_episode  # noqa: E402

FAMILIES = tuple(sorted(_AMBIGUITY_FAMILIES))  # N families (was a hardcoded 4)
N_FAMILIES = len(FAMILIES)
# Keep roughly the same per-family sample density as the original 4-family
# run (~200 train / ~500 eval per family) instead of holding total counts
# fixed — with N_FAMILIES now ~8x the original 4, a fixed total would dilute
# each family's training signal 8x for no reason.
TRAIN_PER_FAMILY = 200
EVAL_PER_FAMILY = 500
D = 128  # NOTE: the pre-registered 0.247/1.000 floor/ceiling were measured at
         # d=256 (probe_m32_ambiguity.py's default). usvs_sense_handle's fixed
         # random projection is itself lossier at lower d, so the *same*
         # MFS/GOLD baseline shifts with d (verified empirically: at d=64 even
         # GOLD only reaches ~0.76 on the flipped half). d=128 keeps GOLD's
         # ceiling at exactly 1.000 (matches canonical) while staying tiny; we
         # additionally recompute MFS/GOLD at this exact d on this exact eval
         # set below (`baseline_rates`) so the comparison is apples-to-apples,
         # and report the canonical d=256 numbers alongside for context.
SEED = 0
EPOCHS = 300
LR = 5e-3
BATCH_SIZE = 16384  # >= any split size used below -> effectively full-batch;
                    # the model is tiny and the dataset is tiny, so full-batch
                    # gradient steps avoid thread-pool overhead from many
                    # small-matmul minibatches dominating wall time on multi-core boxes.
FLOOR_CANONICAL = 0.247   # MFS-grounding win rate on the flipped half @ d=256 (M32 probe)
CEILING_CANONICAL = 1.000  # gold-sense (oracle) win rate on the flipped half @ d=256


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def train(model: SenseChooser, examples: list, *, epochs: int = EPOCHS, lr: float = LR,
          batch_size: int = BATCH_SIZE, seed: int = SEED) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(examples)
    g = torch.Generator().manual_seed(seed)
    for _epoch in range(epochs):
        perm = torch.randperm(n, generator=g).tolist()
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = collate([examples[i] for i in idx])
            logits = model(batch["candidates"], batch["mask"], batch["context"])
            loss = F.cross_entropy(logits, batch["gold_idx"], ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()


def _rate(bools: list) -> float:
    return sum(bools) / len(bools) if bools else float("nan")


def baseline_rates(episodes: list, d: int) -> dict:
    """Recompute the MFS floor / GOLD ceiling at ``d`` on THIS exact episode
    set, using the probe's own ``score_episode`` — so the chooser's
    ``benchmark_acc`` is compared against a baseline measured under the same
    projection dimensionality, not just the canonical d=256 numbers."""
    buckets = {"MFS": {"all": [], "flipped": [], "unflipped": []},
               "GOLD": {"all": [], "flipped": [], "unflipped": []}}
    for ep in episodes:
        bucket = "flipped" if ep.meta["mfs_sense"] != ep.meta["gold_sense"] else "unflipped"
        for label, sense_key in (("MFS", "mfs_sense"), ("GOLD", "gold_sense")):
            v = usvs_sense_handle(ep.meta[sense_key], d)
            win = score_episode(ep, v, d) if v is not None else None
            if win is not None:
                buckets[label]["all"].append(win)
                buckets[label][bucket].append(win)
    return {label: {k: _rate(v) for k, v in sub.items()} for label, sub in buckets.items()}


def evaluate(model: SenseChooser, examples: list, d: int = D) -> dict:
    """Score the chooser's predictions two ways, bucketed by all/flipped/unflipped:

    - ``sense_acc``: predicted sense id == gold sense id (direct classification).
    - ``benchmark_acc``: the M32 probe's OWN metric — ground the predicted sense
      via ``usvs_sense_handle`` and rank the episode's two MC options by cosine
      (``probe_m32_ambiguity.score_episode``). This is the number directly
      comparable to the 0.247 floor / 1.000 ceiling.
    """
    model.eval()
    if not examples:
        empty = {"all": float("nan"), "flipped": float("nan"), "unflipped": float("nan")}
        return {"n": {"all": 0, "flipped": 0, "unflipped": 0}, "sense_acc": dict(empty),
                "benchmark_acc": dict(empty)}

    with torch.no_grad():
        batch = collate(examples)
        logits = model(batch["candidates"], batch["mask"], batch["context"])
    pred_ids = predicted_sense_ids(examples, logits)

    sense_buckets = {"all": [], "flipped": [], "unflipped": []}
    bench_buckets = {"all": [], "flipped": [], "unflipped": []}
    for ex, pid in zip(examples, pred_ids):
        ep = ex.episode
        bucket = "flipped" if ep.meta["mfs_sense"] != ep.meta["gold_sense"] else "unflipped"
        correct = pid == ep.meta["gold_sense"]
        sense_buckets["all"].append(correct)
        sense_buckets[bucket].append(correct)

        v = usvs_sense_handle(pid, d)
        win = score_episode(ep, v, d) if v is not None else None
        if win is not None:
            bench_buckets["all"].append(win)
            bench_buckets[bucket].append(win)

    return {
        "n": {k: len(v) for k, v in sense_buckets.items()},
        "sense_acc": {k: _rate(v) for k, v in sense_buckets.items()},
        "benchmark_acc": {k: _rate(v) for k, v in bench_buckets.items()},
    }


def _print_eval_table(title: str, result: dict) -> None:
    print(f"\n{title}")
    header = f"{'subset':<12}{'n':>6}{'sense_acc':>12}{'benchmark_acc':>16}"
    print(header)
    print("-" * len(header))
    for subset in ("all", "flipped", "unflipped"):
        n = result["n"][subset]
        sa = result["sense_acc"][subset]
        ba = result["benchmark_acc"][subset]
        print(f"{subset:<12}{n:>6}{sa:>12.3f}{ba:>16.3f}")


# ---------------------------------------------------------------------------
# Part 2 — supervised train/val
# ---------------------------------------------------------------------------


def run_supervised(d: int = D, seed: int = SEED) -> dict:
    train_eps = generate_ambiguity_episodes(TRAIN_PER_FAMILY * N_FAMILIES, seed=seed + 100)
    val_eps = generate_ambiguity_episodes(TRAIN_PER_FAMILY * N_FAMILIES // 4, seed=seed + 101)

    train_examples = [build_example(e, d) for e in train_eps]
    val_examples = [build_example(e, d) for e in val_eps]

    model = SenseChooser(d=d)
    train(model, train_examples, seed=seed)
    result = evaluate(model, val_examples, d)
    return {"model": model, "result": result,
            "n_train": len(train_examples), "n_val": len(val_examples)}


# ---------------------------------------------------------------------------
# Part 3 — leave-one-family-out generalization test
# ---------------------------------------------------------------------------


def run_leave_one_family_out(d: int = D, seed: int = SEED) -> dict:
    # Held-in training set target: same ~200/family density as the original
    # 4-family run, scaled to however many families remain after holding one
    # out (N_FAMILIES - 1) instead of the old fixed 800 (which would have
    # diluted to ~27/family with N_FAMILIES ~= 31).
    train_target = TRAIN_PER_FAMILY * (N_FAMILIES - 1)
    # Generate generous pools (with margin) so filtering by family still
    # leaves enough to hit train_target / a full EVAL_PER_FAMILY-sized eval set.
    pool_train = generate_ambiguity_episodes(int(train_target * N_FAMILIES / (N_FAMILIES - 1) * 1.5), seed=seed + 200)
    pool_eval = generate_ambiguity_episodes(EVAL_PER_FAMILY * N_FAMILIES * 2, seed=seed + 300)

    rotations = {}
    for held_out in FAMILIES:
        train_eps = [e for e in pool_train if e.meta["family"] != held_out][:train_target]
        eval_eps = [e for e in pool_eval if e.meta["family"] == held_out]

        train_examples = [build_example(e, d) for e in train_eps]
        eval_examples = [build_example(e, d) for e in eval_eps]

        model = SenseChooser(d=d)
        train(model, train_examples, seed=seed)
        result = evaluate(model, eval_examples, d)
        rotations[held_out] = {
            "result": result, "n_train": len(train_examples), "n_eval": len(eval_examples),
            "episodes": eval_eps,
        }
    return rotations


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    # This model/dataset is tiny; a 12-core box spawning full BLAS thread-pools
    # for thousands of small matmuls is *slower* than single-threaded here.
    torch.set_num_threads(1)
    t0 = time.time()
    model_probe = SenseChooser(d=D)
    n_params = sum(p.numel() for p in model_probe.parameters())
    print(f"SenseChooser: d={D}, params={n_params} (<50k target: {'OK' if n_params < 50_000 else 'FAIL'})")

    print(f"families: {N_FAMILIES} -> {FAMILIES}")

    print("\n=== Part 2: supervised train/val ===")
    sup = run_supervised()
    val_n = TRAIN_PER_FAMILY * N_FAMILIES // 4
    val_eps = generate_ambiguity_episodes(val_n, seed=SEED + 101)  # same call as run_supervised's val split
    same_d = baseline_rates(val_eps, D)
    print(f"train episodes: {sup['n_train']}, val episodes: {sup['n_val']}")
    _print_eval_table(f"val (benchmark_acc = MC-option win rate via chosen sense, d={D})",
                       sup["result"])
    flipped_bench = sup["result"]["benchmark_acc"]["flipped"]
    same_d_floor = same_d["MFS"]["flipped"]
    same_d_ceiling = same_d["GOLD"]["flipped"]
    print(f"\nsame-d (d={D}) recomputed baseline on THIS val set, flipped half:"
          f"  MFS floor={same_d_floor:.3f}  GOLD ceiling={same_d_ceiling:.3f}")
    print(f"canonical (d=256, probe_m32_ambiguity.py) reference:"
          f"          MFS floor={FLOOR_CANONICAL:.3f}  GOLD ceiling={CEILING_CANONICAL:.3f}")
    span = same_d_ceiling - same_d_floor
    closed = (flipped_bench - same_d_floor) / span * 100 if span > 1e-9 else float("nan")
    print(f"\nchooser flipped-half benchmark_acc = {flipped_bench:.3f}  "
          f"(same-d floor {same_d_floor:.3f}, same-d ceiling {same_d_ceiling:.3f}, "
          f"gap closed: {closed:.1f}%)")

    print(f"\n=== Part 3: leave-one-family-out generalization (all {N_FAMILIES} rotations) ===")
    rotations = run_leave_one_family_out()
    header = (f"{'held_out':<10}{'n_train':>9}{'n_eval':>8}{'flipped_n':>11}"
              f"{'flipped_bench_acc':>19}{'same_d_floor':>14}{'same_d_ceiling':>16}")
    print(header)
    print("-" * len(header))
    flipped_accs = []
    for family, r in rotations.items():
        res = r["result"]
        eval_eps = r["episodes"]
        base = baseline_rates(eval_eps, D)
        acc = res["benchmark_acc"]["flipped"]
        if acc == acc:  # not NaN
            flipped_accs.append(acc)
        print(f"{family:<10}{r['n_train']:>9}{r['n_eval']:>8}"
              f"{res['n']['flipped']:>11}{acc:>19.3f}"
              f"{base['MFS']['flipped']:>14.3f}{base['GOLD']['flipped']:>16.3f}")
    print("-" * len(header))
    mean_acc = sum(flipped_accs) / len(flipped_accs) if flipped_accs else float("nan")
    print(f"mean flipped_bench_acc across {len(flipped_accs)}/{N_FAMILIES} rotations: {mean_acc:.3f}")
    print("(original M32/M34 result, 4 families / 3-of-4 rotations: 3/4 rotations transferred "
          "perfectly, bank-held-out stuck at floor — this table is the direct re-test of that "
          "with N-1-of-N training families instead of 3-of-4)")

    print(f"\ntotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
