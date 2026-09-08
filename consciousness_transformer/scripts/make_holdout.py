"""Write the shared held-out sentence files for the encoder training arms
(dev/AUDIT_2026-09-08.md finding 7 + recommendation (c)).

`runs/holdout_sentences.txt` = the sentence texts of the CURRENT seeded v2
test split (n~98, same seed/split sizes as run-2 -- default seed=0,
n_train=788, n_dev=98, n_test=98 against runs/encoder_gold_v2.jsonl -- see
scripts/train_encoder.py's non-smoke defaults), so it is the EXACT set the
existing 0.70/0.68 numbers were measured on. `runs/holdout_dev_sentences.txt`
= that same split's dev sentences, as a second held-out file.

Every arm (v2_788 / v3_788 / v3_3000) trains on a pool that EXCLUDES these
sentences (--holdout-file) and is scored on them regardless of which gold
file trains it (--eval-gold / --eval-gold-alt), so "more data" and "cleaner
data" are compared on identical held-out sentences.

Usage:
    python scripts/make_holdout.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nsm_ct import encoder_train_util as etu

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default=str(ROOT / "runs" / "encoder_gold_v2.jsonl"))
    ap.add_argument("--seed", type=int, default=0, help="must match the split this holdout is meant to reproduce (run-2: 0)")
    ap.add_argument("--n-train", type=int, default=788)
    ap.add_argument("--n-dev", type=int, default=98)
    ap.add_argument("--n-test", type=int, default=98)
    ap.add_argument("--out-test", default=str(ROOT / "runs" / "holdout_sentences.txt"))
    ap.add_argument("--out-dev", default=str(ROOT / "runs" / "holdout_dev_sentences.txt"))
    args = ap.parse_args()

    records = etu.load_gold(args.gold)
    print(f"{len(records)} gold records loaded from {args.gold}")

    _train_recs, dev_recs, test_recs = etu.stratified_split(
        records, args.seed, args.n_train, args.n_dev, args.n_test)
    print(f"seeded split (seed={args.seed}, n_train={args.n_train}, n_dev={args.n_dev}, "
          f"n_test={args.n_test}): train={args.n_train} dev={len(dev_recs)} test={len(test_recs)}")

    for path, recs, name in ((args.out_test, test_recs, "test"), (args.out_dev, dev_recs, "dev")):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(r["text"] for r in recs) + "\n", encoding="utf-8")
        print(f"wrote {len(recs)} {name}-split sentences -> {out_path}")


if __name__ == "__main__":
    main()
