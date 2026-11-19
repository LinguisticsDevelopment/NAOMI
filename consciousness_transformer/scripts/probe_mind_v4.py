"""Descent probe: the answer SENTENCE is given (the walk starts at its root and may not move between sentences), so only the
within-sentence choice is measured. v4 walks down; v3 takes the best node of that sentence (its within-sentence softmax).
Isolates the recursion's transfer from short to long sentences from the sentence-selection problem.
    python scripts/probe_mind_v4.py val v4_all=runs/mind_v4/thinker_v4_all_best.pt v4_short=... v3_s1=... v3_short=...
Writes runs/mind_v4/probe_<split>.json and prints a markdown table (by band of the answer sentence)."""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_v2 as m2  # noqa: E402
from naomi import mind_v3 as m3  # noqa: E402
from naomi import mind_v4 as m4  # noqa: E402
import train_mind_v4 as T4  # noqa: E402

BANDS = ["<=10", "11-20", "21+"]
D = CT / "runs" / "mind_v4"


@torch.no_grad()
def anchors_given_sentence(model, b, chunk, stories):
    """Anchor (local gid) per item inside the oracle sentence (the sentence of the first accepted anchor), no sentence moves."""
    model.eval()
    out = []
    if isinstance(model, m4.Thinker4):
        e = model.encode(b)
        e["move"] = torch.cat([e["move"][:, :1], torch.full_like(e["move"][:, 1:], m4.NEG)], 1)
        kids = m4._kids_of(b["s_parent"])
        for i, p in enumerate(chunk):
            o0 = int(b["s_off"][i])
            a0 = p["tgt"]["anchors"][0][0]
            arr = stories[p["skey"]]
            root = next(g for g in range(a0, -1, -1) if arr["parent"][g] < 0)
            fin, steps, path, _lp, _pab = m4._walk(e, kids, o0 + root, m4.BUDGET)
            out.append((fin - o0 if fin >= 0 else -1, steps))
        return e, out
    e = model.encode(b)
    for i, p in enumerate(chunk):
        o0, o1, n, gl = m3._item_view(e, b, i)
        a0 = p["tgt"]["anchors"][0][0]
        s = int(gl[a0])
        idx = torch.where(gl == s)[0]
        out.append((int(idx[int(e["within"][o0:o1][idx].argmax())]), 0))
    return e, out


def main():
    split = sys.argv[1]
    specs = [a.split("=") for a in sys.argv[2:]]
    T4.T3.DATA = CT / "runs" / "mind_v2b"
    stories = pickle.load(open(D / "stories.pkl", "rb"))
    meta = json.load(open(D / "items_meta.json"))
    pairs = [p for p in pickle.load(open(D / f"pairs_{split}.pkl", "rb")) if p["kind"] == "normal" and p["tgt"]["anchors"] and p["tgt"]["best_f1"] >= m2.ACCEPT_F1]
    tab = m2.SenseTable()
    res = {}
    print(f"## Descent probe ({split}): answer sentence given, items with a reachable answer\n")
    print("| model | band | n | anchor acceptable | chosen depth | oracle depth | ROUGE-L (true order, v4 or v3 keep head) |\n|---|---|---|---|---|---|---|")
    for name, path in specs:
        model, _ = T4.load_model(path)
        rows = defaultdict(list)
        for i in range(0, len(pairs), 48):
            ch = pairs[i:i + 48]
            b = m2.collate(ch, stories, tab, with_targets=False)
            e, ans = anchors_given_sentence(model, b, ch, stories)
            anch = [a for a, _ in ans]
            kept, _ = m2.keep_sets(model, e, b, stories, ch, anch)
            for p, (a, st), k in zip(ch, ans, kept):
                arr = stories[p["skey"]]
                acc = {x for x, _k, _f in p["tgt"]["anchors"]}
                band = m2.band_of(p["tgt"]["ans_len"])
                if band == "none":
                    continue
                rows[band].append({"ok": a in acc, "depth": int(arr["depth"][a]) if a >= 0 else None, "od": int(arr["depth"][p["tgt"]["anchors"][0][0]]),
                                   "rouge": m2.rouge_multi(m2.true_order_text(arr, k), meta[p["item"]]["golds"]) if a >= 0 else 0.0})
        res[name] = {}
        for b_ in BANDS:
            r = rows[b_]
            d = [x["depth"] for x in r if x["depth"] is not None]
            res[name][b_] = {"n": len(r), "acceptable": float(np.mean([x["ok"] for x in r])), "depth": float(np.mean(d)), "oracle_depth": float(np.mean([x["od"] for x in r])),
                             "rouge": float(np.mean([x["rouge"] for x in r]))}
            x = res[name][b_]
            print(f"| {name} | {b_} | {x['n']} | {x['acceptable']:.3f} | {x['depth']:.2f} | {x['oracle_depth']:.2f} | {x['rouge']:.3f} |")
    json.dump(res, open(D / f"probe_{split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
