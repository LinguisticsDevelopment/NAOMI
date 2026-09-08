#!/usr/bin/env python3
"""Stratified small-mix sample of `runs/hard_gold_train.jsonl` (hard-gold-gen-v3):
a seeded 200-record subset, evenly split across the 8 families, for the 1:4
hard-gold-to-teacher-gold mixing arm (a full ~1000-record hard_gold_train.jsonl
would swamp a small mixing ratio; this gives a fixed-size, reproducible slice
instead of re-sampling ad hoc per run).

Run:  python scripts/make_hard_gold_small.py [--n 200] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RUNS = _HERE.parent / "runs"


def make_small(records: list, n_target: int, seed: int) -> list:
    by_family = defaultdict(list)
    for r in records:
        by_family[r["meta"]["family"]].append(r)
    families = sorted(by_family)
    rng = random.Random(seed)

    base = n_target // len(families)
    rem = n_target - base * len(families)
    counts = {f: base for f in families}
    # give the remainder to the largest families first, deterministically
    for f in sorted(families, key=lambda f: (-len(by_family[f]), f))[:rem]:
        counts[f] += 1

    out = []
    for f in families:
        pool = list(by_family[f])
        rng.shuffle(pool)
        out.extend(pool[:min(counts[f], len(pool))])
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=str(RUNS / "hard_gold_train.jsonl"))
    ap.add_argument("--out", default=str(RUNS / "hard_gold_train_small.jsonl"))
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.src).open()]
    small = make_small(records, args.n, args.seed)

    with Path(args.out).open("w") as fh:
        for r in small:
            fh.write(json.dumps(r) + "\n")

    by_family: dict = defaultdict(int)
    for r in small:
        by_family[r["meta"]["family"]] += 1
    print(f"wrote {args.out} ({len(small)} records)")
    for f in sorted(by_family):
        print(f"  {f:20s} {by_family[f]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
