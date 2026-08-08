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
   (~800/200), report accuracy overall and on the flipped half vs the
   0.247 floor / 1.000 ceiling.

2. **Leave-one-family-out** (Part 3, the scientifically decisive one): train
   on 3 of the 4 homograph families (bank/bat/plant/organ), evaluate on the
   held-out 4th, for all 4 rotations. A held-out family the chooser still
   resolves means it learned *how* to match a context vector to a sense
   vector in USVS geometry (generalizes); flopping to floor on the held-out
   family would mean it only learned *which* sense each specific word means
   (memorization — does not generalize).

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

from nsm_ct.episode import generate_ambiguity_episodes  # noqa: E402
from nsm_ct.sense_chooser import (  # noqa: E402
    SenseChooser,
    build_example,
    collate,
    predicted_sense_ids,
)
from nsm_ct.usvs_bridge import usvs_sense_handle  # noqa: E402

from probe_m32_ambiguity import score_episode  # noqa: E402

FAMILIES = ("bank", "bat", "plant", "organ")
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
BATCH_SIZE = 1024  # >= any split size here (800 train) -> effectively full-batch;
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
    train_eps = generate_ambiguity_episodes(800, seed=seed + 100)
    val_eps = generate_ambiguity_episodes(200, seed=seed + 101)

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
    pool_train = generate_ambiguity_episodes(4000, seed=seed + 200)
    pool_eval = generate_ambiguity_episodes(2000, seed=seed + 300)

    rotations = {}
    for held_out in FAMILIES:
        train_eps = [e for e in pool_train if e.meta["family"] != held_out][:800]
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

    print("\n=== Part 2: supervised train/val ===")
    sup = run_supervised()
    val_eps = generate_ambiguity_episodes(200, seed=SEED + 101)  # same seed as run_supervised's val split
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

    print("\n=== Part 3: leave-one-family-out generalization ===")
    rotations = run_leave_one_family_out()
    header = (f"{'held_out':<10}{'n_train':>9}{'n_eval':>8}{'flipped_n':>11}"
              f"{'flipped_bench_acc':>19}{'same_d_floor':>14}{'same_d_ceiling':>16}")
    print(header)
    print("-" * len(header))
    for family, r in rotations.items():
        res = r["result"]
        eval_eps = r["episodes"]
        base = baseline_rates(eval_eps, D)
        print(f"{family:<10}{r['n_train']:>9}{r['n_eval']:>8}"
              f"{res['n']['flipped']:>11}{res['benchmark_acc']['flipped']:>19.3f}"
              f"{base['MFS']['flipped']:>14.3f}{base['GOLD']['flipped']:>16.3f}")

    print(f"\ntotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
