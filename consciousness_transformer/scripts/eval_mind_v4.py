"""Evaluate Mind v4 (and the v3 / v2b checkpoints on the same items) on one split: the v2 evaluation code with the thinker dispatch,
plus the walk diagnostics (steps, final anchor depth, budget exhaustion) saved to runs/mind_v4/steps_<split><tag>.json.
    python scripts/eval_mind_v4.py --dir mind_v4 --split val --models v4_all=runs/mind_v4/thinker_v4_all_best.pt v3_s2=runs/mind_v3/thinker_v3_s2_best.pt --main v4_all
Run on the test split ONCE, after every choice (checkpoint, --nstart) is made on val."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_v2 as m2  # noqa: E402
from naomi import mind_v4 as m4  # noqa: E402
import eval_mind_v2 as E  # noqa: E402
import train_mind_v4 as T4  # noqa: E402

EXTRA = {}          # model name -> {item: walk diagnostics}
NAMES = {}          # id(model) -> name


def take(flag, default, cast):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = cast(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        return v
    return default


class Shim:
    """Loader that remembers which model object carries which name (names come from --models name=path, in order)."""

    def __init__(self, names):
        self.names, self.n = names, 0

    def load_model(self, path):
        out = T4.load_model(path)
        NAMES[id(out[0])] = self.names[self.n]
        self.n += 1
        return out


@torch.no_grad()
def run_model(model, stories, pairs, tab, bs=48):
    out = {}
    nm = NAMES.get(id(model))
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        b = m2.collate(ch, stories, tab, with_targets=False)
        e, anch, res = E.m2.decide(model, b)
        kept, _ = m2.keep_sets(model, e, b, stories, ch, anch)
        for p, a, k, r in zip(ch, anch, kept, res):
            out[p["id"]] = (a, k, r["p_abstain"])
            if isinstance(model, m4.Thinker4) and p["kind"] == "normal":
                arr = stories[p["skey"]]
                EXTRA.setdefault(nm, {})[p["item"]] = {"steps": r["steps"], "exhausted": r["exhausted"], "path": r["path"], "answered": a >= 0,
                                                       "depth": int(arr["depth"][a]) if a >= 0 else None, "anchor": a,
                                                       "sent": int(arr["sent"][a]) if a >= 0 else None}
    return out


def main():
    nstart = take("--nstart", 1, int)
    start = take("--start", "value", str)
    budget = take("--budget", m4.BUDGET, int)
    here_bias = take("--here_bias", 0.0, float)
    E.m2 = m4.Mind(1, nstart, start, budget, here_bias)
    E.run_model = run_model
    names = [s.split("=")[0] for s in sys.argv[sys.argv.index("--models") + 1:] if "=" in s and not s.startswith("--")]
    E.T = Shim(names)
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "val"
    tag = sys.argv[sys.argv.index("--out_tag") + 1] if "--out_tag" in sys.argv else ""
    E.main()
    json.dump(EXTRA, open(CT / "runs" / "mind_v4" / f"steps_{split}{tag}.json", "w"))


if __name__ == "__main__":
    main()
