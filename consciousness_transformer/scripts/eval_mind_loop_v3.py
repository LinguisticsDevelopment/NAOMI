"""Evaluate looped thinker v3 on one split (val first: picks thr/tau; test ONCE). Reuses eval_mind_loop.main (tables, cheat, controls, tree score) with outputs
redirected to runs/mind_loop_v3/, then eval_mind_loop_v2.extras (counterfactual accuracy, per-step loss, halting, story adds), then the v3 additions:
coverage, ROUGE-L / tree score on answered items only, and a risk-coverage curve over the existence threshold.
    python scripts/eval_mind_loop_v3.py --split val --ckpt runs/mind_loop_v3/v3_s0_best.pt --tag _s0
    python scripts/eval_mind_loop_v3.py --split test --ckpt ... --tag _s0        (reads the val choice of the same tag)"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_v2 as m2  # noqa: E402
import eval_mind_loop as EL  # noqa: E402
import eval_mind_loop_v2 as E2X  # noqa: E402
import train_mind_loop as TL  # noqa: E402
import train_mind_loop_v3 as T3  # noqa: E402

OUT = T3.OUT
FINE = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
EL.THR_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85)


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


@torch.no_grad()
def v3_extras(split, ckpt, tag):
    D = TL.Data((split,))
    items = D.items[split]
    stories = pickle.load(open(TL.V2B / "stories.pkl", "rb"))
    dc = TL.Decode(D, stories)
    model, _ = TL.load_model(ckpt)
    ch = json.load(open(OUT / f"val_choice{tag}.json"))["chosen"]
    thr, tau = ch["thr"], ch["tau"]
    # risk-coverage over the existence threshold (halting rule fixed at the chosen tau)
    recs = EL.forward_mode(model, D, items, "normal")
    curve = []
    for t in FINE:
        rows = EL.decode_recs(dc, recs, t, tau)
        sc = np.array([m2.rouge_multi(txt, x["golds"]) if txt else 0.0 for x, _q, txt, _s, _n in rows])
        ans = np.array([bool(txt) for _x, _q, txt, _s, _n in rows])
        curve.append({"thr": t, "coverage": float(ans.mean()), "rouge_all": float(sc.mean()), "rouge_answered": float(sc[ans].mean()) if ans.any() else None,
                      "mean_nodes": float(np.mean([n for *_a, n in rows]))})
        print("risk-coverage", curve[-1], flush=True)
    # answered-only metrics at the chosen setting, from the rows file written by eval_mind_loop.main
    rows = [json.loads(l) for l in open(OUT / f"eval_rows_{split}{tag}.jsonl")]
    ans = [r for r in rows if r["text"]]
    cheat_ans = [r["cheat"] for r in ans]
    tr = [r for r in rows if "tree_loop" in r]
    tra = [r for r in tr if r["text"]]
    res = {"thr": thr, "tau": tau, "n": len(rows), "coverage": len(ans) / len(rows), "rouge_all": float(np.mean([r["rouge"] for r in rows])),
           "rouge_answered": float(np.mean([r["rouge"] for r in ans])) if ans else None, "cheat_on_answered": float(np.mean(cheat_ans)) if ans else None,
           "tree_all_300": float(np.mean([r["tree_loop"] for r in tr])) if tr else None, "tree_answered_only": float(np.mean([r["tree_loop"] for r in tra])) if tra else None,
           "n_tree": len(tr), "n_tree_answered": len(tra), "risk_coverage": curve}
    bands = {}
    for b in EL.BANDS:
        rb = [r for r in rows if r["band"] == b]
        ab = [r for r in rb if r["text"]]
        bands[b] = {"n": len(rb), "coverage": len(ab) / max(len(rb), 1), "rouge_all": float(np.mean([r["rouge"] for r in rb])) if rb else None,
                    "rouge_answered": float(np.mean([r["rouge"] for r in ab])) if ab else None}
    res["bands"] = bands
    json.dump(res, open(OUT / f"v3extras_{split}{tag}.json", "w"), indent=1)
    L = [f"### v3 additions ({split}, {tag or 'run'}): thr={thr} tau={tau}", "", "```", json.dumps({k: v for k, v in res.items() if k != "risk_coverage"}, indent=1), "```", "",
         "| thr | coverage | ROUGE-L all | ROUGE-L answered | mean nodes |", "|---|---|---|---|---|"]
    for c in curve:
        L.append(f"| {c['thr']} | {c['coverage']:.3f} | {c['rouge_all']:.3f} | {'' if c['rouge_answered'] is None else format(c['rouge_answered'], '.3f')} | {c['mean_nodes']:.2f} |")
    open(OUT / f"tables_{split}{tag}.md", "a").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    tag = arg("--tag", "")
    EL.OUT = OUT
    E2X.OUT = OUT
    rb = OUT / "read_cache" / "readback.jsonl"
    if not rb.exists():
        rb.parent.mkdir(exist_ok=True)
        shutil.copy(TL.OUT / "read_cache" / "readback.jsonl", rb)         # copy of the v1 read-back cache; v1 file untouched
    cp = OUT / f"val_choice{tag}.json"
    plain = OUT / "val_choice.json"
    if arg("--split") == "test" and cp.exists():
        shutil.copy(cp, plain)
    EL.main()
    if arg("--split") == "val":
        shutil.copy(plain, cp)
    E2X.extras(arg("--split"), arg("--ckpt"), tag)
    v3_extras(arg("--split"), arg("--ckpt"), tag)
