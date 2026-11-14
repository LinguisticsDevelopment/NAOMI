"""Probe: the M-scaling experiment -- is the clause reactor's curriculum
accuracy DATA-limited or CAPACITY-limited?

M35 verified 12 surface templates parse cleanly (100% success,
``curriculum2.generate_varied_episodes``, template sets A/B) plus a 61-noun
place pool (``vocab_scale``), and showed zero template overfitting. This
script takes the natural next step: hold everything else fixed (mixed A+B
templates, vocab_scale on, denser "scaled"-mode episodes, USVS fillers, seed
fixed) and vary ONE axis at a time against scripts/train_clause.py's
full-budget reference point (val 0.885 at 480 episodes / 80 epochs / dim 48):

  data axis:     480 / 1000 / 2000 curriculum2 "scaled" episodes, dim 48
  capacity axis: dim 48 / 64 / 96, at 2000 episodes

Perception is fixed/grounded (usvs meaning source); only the ClauseReactor's
GRU reaction policy trains -- this script reuses that machinery
(ClauseReactor, build_clause_batch) exactly as scripts/train_clause.py and
scripts/probe_template_overfit.py do, swapping the episode source for
:func:`nsm_ct.curriculum2.generate_scaled_episodes` (mixed A+B templates, the
61-noun vocab pool, 4-8 facts / 2-4 entities per episode -- see
curriculum2.py's "scaled" mode).

Each cell runs under a wall-clock budget (``--time-budget-min``, default 40);
if a cell exceeds it, its epoch count is halved once and retried (noted in
the report) rather than left to run indefinitely.

Run:
    python scripts/probe_scaled_training.py [--epochs 80] [--time-budget-min 40]
"""

from __future__ import annotations

import argparse
import collections
import os
import signal
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import generate_scaled_episodes  # noqa: E402
from nsm_ct.episode import split_episodes  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


class _TimeLimitExceeded(Exception):
    pass


def _alarm(signum, frame):
    raise _TimeLimitExceeded()


def _acc(out, batch) -> float:
    return float((out["answer_logits"].argmax(-1) == batch.answer).float().mean())


def _per_level(model, batch, episodes):
    """Val accuracy broken down by curriculum level (matches train_clause.py)."""
    with torch.no_grad():
        correct = (model(batch)["answer_logits"].argmax(-1) == batch.answer).tolist()
    agg = collections.defaultdict(lambda: [0, 0])
    for ep, ok in zip(episodes, correct):
        agg[f"L{ep.level}"][0] += int(ok)
        agg[f"L{ep.level}"][1] += 1
    return {k: (c, n) for k, (c, n) in sorted(agg.items())}


def _build_parser(episodes) -> ParserInputEncoder:
    texts = [t for e in episodes
             for t in e.context + [e.question] + (e.options or []) + (e.post_context or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    return ParserInputEncoder(tok)


def train_cell(*, episodes: int, dim: int, epochs: int, seed: int, val_fraction: float = 0.2):
    """Train one (episodes, dim) cell of the scaling grid; mirrors
    scripts/train_clause.py's loop exactly (full-batch AdamW lr=3e-3,
    meaning_source='usvs') so results are comparable to its reported
    reference point. Returns ``(final_val_acc, per_level_dict, n_params)``.
    """
    torch.manual_seed(seed)
    eps = generate_scaled_episodes(episodes, seed=seed)
    tr, va = split_episodes(eps, val_fraction, seed=seed)
    parser = _build_parser(eps)
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    train = build_clause_batch(tr, parser, resolver, codec, "usvs")
    val = build_clause_batch(va, parser, resolver, codec, "usvs")
    print(f"  encoded: train={len(tr)} val={len(va)} T_train={train.entity.shape[1]}")

    model = ClauseReactor(dim=dim)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for epoch in range(epochs):
        model.train()
        out = model(train)
        loss = F.cross_entropy(out["answer_logits"], train.answer)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 20 == 19 or epoch == 0:
            model.eval()
            with torch.no_grad():
                to_, vo = model(train), model(val)
            print(f"    epoch {epoch + 1:3d} | loss={loss.item():.3f} "
                  f"train_acc={_acc(to_, train):.3f} val_acc={_acc(vo, val):.3f}")

    model.eval()
    with torch.no_grad():
        vo = model(val)
    final_acc = _acc(vo, val)
    levels = _per_level(model, val, va)
    return final_acc, levels, n_params


def run_cell_capped(label: str, *, episodes: int, dim: int, epochs: int, seed: int,
                     budget_min: float) -> dict:
    """Run :func:`train_cell` under a wall-clock cap; on timeout, halve
    epochs once and retry (noted in the returned record), per the probe's
    stop-loss policy -- never left to run indefinitely."""
    attempt_epochs = epochs
    for _attempt in range(2):
        print(f"=== {label}: episodes={episodes} dim={dim} epochs={attempt_epochs} "
              f"(budget={budget_min:.0f} min) ===")
        old_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(budget_min * 60))
        t0 = time.time()
        try:
            acc, levels, n_params = train_cell(episodes=episodes, dim=dim,
                                                epochs=attempt_epochs, seed=seed)
            elapsed_min = (time.time() - t0) / 60
            note = "" if attempt_epochs == epochs else f"epochs halved from {epochs} after a timeout"
            per_level_str = "  ".join(f"{k}={c}/{n}={c / n:.2f}" for k, (c, n) in levels.items())
            print(f"  -> val_acc={acc:.3f}  time={elapsed_min:.1f} min  params={n_params:,}"
                  + (f"  [{note}]" if note else ""))
            print(f"  per-level: {per_level_str}")
            return {"label": label, "episodes": episodes, "dim": dim, "epochs": attempt_epochs,
                    "val_acc": acc, "minutes": elapsed_min, "levels": levels, "note": note}
        except _TimeLimitExceeded:
            elapsed_min = (time.time() - t0) / 60
            print(f"  !! exceeded {budget_min:.0f} min budget ({elapsed_min:.1f} min elapsed); "
                  f"halving epochs {attempt_epochs} -> {max(1, attempt_epochs // 2)} and retrying")
            attempt_epochs = max(1, attempt_epochs // 2)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    raise RuntimeError(f"{label} exceeded the time budget even after halving epochs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget-min", type=float, default=40.0)
    ap.add_argument("--axis", choices=("both", "data", "capacity"), default="both",
                    help="run only one axis's cells (resume aid: a killed run's "
                         "completed axis doesn't need re-running — cells are "
                         "deterministic given the same seed/args)")
    args = ap.parse_args()

    # Overlap by design: (2000, dim=48) appears on both axes -- run it twice
    # (once per axis) so each axis's own table is self-contained and each
    # cell gets its own independent wall-time measurement.
    data_points = [(480, 48), (1000, 48), (2000, 48)] if args.axis != "capacity" else []
    capacity_points = [(2000, 48), (2000, 64), (2000, 96)] if args.axis != "data" else []

    results = []
    print("#" * 78)
    print("DATA AXIS -- episodes 480 / 1000 / 2000 at dim 48, "
          f"{args.epochs} epochs (curriculum2 scaled: mixed A+B templates, vocab_scale on)")
    print("#" * 78)
    for episodes, dim in data_points:
        r = run_cell_capped(f"data episodes={episodes}", episodes=episodes, dim=dim,
                             epochs=args.epochs, seed=args.seed, budget_min=args.time_budget_min)
        r["axis"] = "data"
        results.append(r)

    print()
    print("#" * 78)
    print(f"CAPACITY AXIS -- dim 48 / 64 / 96 at 2000 episodes, {args.epochs} epochs")
    print("#" * 78)
    for episodes, dim in capacity_points:
        r = run_cell_capped(f"capacity dim={dim}", episodes=episodes, dim=dim,
                             epochs=args.epochs, seed=args.seed, budget_min=args.time_budget_min)
        r["axis"] = "capacity"
        results.append(r)

    print()
    print("=" * 78)
    print("SCALING TABLE")
    print("=" * 78)
    print(f"{'axis':10s} {'episodes':>9s} {'dim':>5s} {'epochs':>7s} {'val_acc':>8s} "
          f"{'minutes':>8s}  per-level")
    for r in results:
        per_level_str = "  ".join(f"{k}={c}/{n}={c / n:.2f}" for k, (c, n) in r["levels"].items())
        note = f"  [{r['note']}]" if r["note"] else ""
        print(f"{r['axis']:10s} {r['episodes']:9d} {r['dim']:5d} {r['epochs']:7d} "
              f"{r['val_acc']:8.3f} {r['minutes']:8.1f}  {per_level_str}{note}")

    best = max(results, key=lambda r: r["val_acc"])
    print()
    print(f"best cell: {best['axis']} episodes={best['episodes']} dim={best['dim']} "
          f"epochs={best['epochs']} val_acc={best['val_acc']:.3f}")
    print("  per-level: " + "  ".join(f"{k}={c}/{n}={c / n:.2f}" for k, (c, n) in best["levels"].items()))
    print()
    print("reference (train_clause.py, plain curriculum): val 0.885 @ 480 episodes / "
          "80 epochs / dim 48 (usvs fillers)")


if __name__ == "__main__":
    main()
