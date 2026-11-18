"""Evaluate orderer checkpoints on gappy fragments cut from the TEST-split sentences (never trained on) and on full text.
  python scripts/eval_orderer_fragments.py --ckpts old=runs/naomi_orderer/large_best.pt new=runs/naomi_orderer_frag/frag_best.pt --out runs/naomi_orderer_frag/eval_fragments.json
Fragments: sizes drawn per fragment from the training distribution, 4 per piece, fixed seed.  Metrics per size: exact order
and all-pairs accuracy (1 - Kendall distance).  Also the full-sentence sets of eval_naomi_orderer (connected / pieces)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402

from naomi import orderer as O  # noqa: E402
from train_naomi_orderer import fragments_from_items, load_items, tins_from_items  # noqa: E402

BANDS = [("2-4", 2, 4), ("5-8", 5, 8), ("9-12", 9, 12), ("13-15", 13, 15)]


def frag_metrics(tins, orders):
    rows = {}
    for t, o in zip(tins, orders):
        n = t.n_tokens
        ex = int(O.emit(t, o) == O.true_order(t))
        kt = O.kendall_distance([int(t.true_pos[i]) for i in O.emit(t, o)])
        rows.setdefault(n, []).append((ex, 1 - kt))
    def agg(sel):
        v = [x for n in sel for x in rows.get(n, [])]
        return {"n": len(v), "exact": round(float(np.mean([a for a, _ in v])), 4) if v else None,
                "pairwise": round(float(np.mean([b for _, b in v])), 4) if v else None}
    out = {"by_size": {str(n): agg([n]) for n in sorted(rows)}, "all": agg(list(rows))}
    out["by_band"] = {f"{n}": agg(range(lo, hi + 1)) for n, lo, hi in BANDS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    by = load_items()
    res = {}
    for spec in a.ckpts:
        name, path = spec.split("=")
        orr = O.Orderer.load(str(ROOT / path))
        test_frag = fragments_from_items(by["test"], orr.builder, orr.rel_ids, 4, seed=777)
        full = tins_from_items(by["test"], orr.builder, orr.rel_ids, connected_only=True)
        pieces = tins_from_items(by["test"], orr.builder, orr.rel_ids)
        r = {"fragments": frag_metrics(test_frag, orr.group_orders(test_frag)), "n_fragments": len(test_frag),
             "test_split_connected": O.evaluate(full, orr.group_orders)["all"],
             "test_split_pieces": O.evaluate(pieces, orr.group_orders)["all"]}
        res[name] = r
        print(name, json.dumps({"frag_all": r["fragments"]["all"], "bands": r["fragments"]["by_band"],
                                "connected": r["test_split_connected"], "pieces": r["test_split_pieces"]}), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
