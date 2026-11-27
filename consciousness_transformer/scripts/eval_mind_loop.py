"""Evaluate the looped thinker on one split (test ONCE, after every setting is fixed on val).
    python scripts/eval_mind_loop.py --split val --ckpt runs/mind_loop/loop_all_best.pt [--short runs/mind_loop/loop_short_best.pt]   # picks thr/tau on val
    python scripts/eval_mind_loop.py --split test --ckpt ... [--short ...]                                                            # reads val_choice.json
Writes runs/mind_loop/eval_<split>.json, tables_<split>.md, handread_<split>.json (rows jsonl is large: not for the commit)."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_loop as ML  # noqa: E402
from naomi import mind_v1 as m1  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402
from naomi.tree import tree_from_dict  # noqa: E402
import eval_mind_v2 as E2  # noqa: E402
import eval_mind_v5 as E5  # noqa: E402
import train_mind_loop as TL  # noqa: E402

OUT = TL.OUT
BANDS = ["<=10", "11-20", "21+", "none"]
STATS = ("vec", "has", "idc", "rel", "pos", "par", "ex")
V5_TEST = {"cheat": 0.200, "v3_s2": 0.346, "v5_all": 0.298}          # MIND_V5.md test table, full-sentence ROUGE-L (a), same 1007 items
V5_TREE = {"cheat": 11.2, "v3_s2": 38.3, "v5_all": 33.6}


@torch.no_grad()
def forward_mode(model, D, items, mode, bs=24, seed=0):
    """Batches of (items, memory keys, stacked outputs [N, B, ...], halting dist, salts, question item per row)."""
    model.eval()
    rng = random.Random(seed)
    recs = []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        reqs, qitems, mems = [], [], []
        for x in ch:
            q, mem = x, x["skey"]
            if mode == "empty":
                mem = None
            elif mode == "wrong":
                mem = x["wrong_skey"]
            elif mode == "qonly":
                mem = ("q", x["item"])
            elif mode == "shuffled":
                q = rng.choice([y for y in items if y["skey"] != x["skey"]])
            if mode == "wrong" and mem is None:
                continue
            reqs.append({"mem": mem, "q": q["q"], "t": None, "qid": x["item"]})
            qitems.append(q)
            mems.append(mem)
        if not reqs:
            continue
        outs, hl, _ = TL.forward_batch(model, D, reqs, block_read=(mode == "block"))
        recs.append({"items": [x for x in ch if not (mode == "wrong" and x["wrong_skey"] is None)], "qitems": qitems, "mems": mems,
                     "st": {k: torch.stack([o[k] for o in outs]) for k in STATS}, "p": ML.halting(hl), "salts": [TL.zlib_salt(r["mem"]) for r in reqs]})
    return recs


def decode_recs(dc, recs, thr, tau):
    """-> list of (item, qitem, text, steps taken, n output nodes) for every row."""
    rows = []
    for r in recs:
        trees, hs = [], ML.halt_step(r["p"], tau)
        for b, (x, qx, mk) in enumerate(zip(r["items"], r["qitems"], r["mems"])):
            nodes = list(dc.stories[mk]["nodes"]) if mk is not None and not isinstance(mk, tuple) else []
            pool = m2.Pool(nodes, qx["qnodes"])
            tout = {k: v[int(hs[b]), b].unsqueeze(0) for k, v in r["st"].items()}
            trees.append(ML.decode_tree(tout, 0, pool, ML.CodeBook(r["salts"][b]), dc.D.dims["rels"], dc.D.dims["poss"], thr))
        ok = [i for i, t in enumerate(trees) if t is not None]
        txt = [""] * len(trees)
        for i, s in zip(ok, dc.dec.texts([trees[i] for i in ok])):
            txt[i] = s
        for b, x in enumerate(r["items"]):
            n = int((r["st"]["ex"][int(hs[b]), b] > thr).sum())
            rows.append((x, r["qitems"][b], txt[b], int(hs[b]) + 1, n))
    return rows


def rouge_of(rows):
    return [m2.rouge_multi(t, x["golds"]) if t else 0.0 for x, _q, t, _s, _n in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--short", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rt_n", type=int, default=300)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(4)
    D = TL.Data((args.split,))
    items = D.items[args.split]
    if args.limit:
        items = random.Random(0).sample(items, args.limit)
    stories = pickle.load(open(TL.V2B / "stories.pkl", "rb"))
    dc = TL.Decode(D, stories)
    model, _ = TL.load_model(args.ckpt)
    short = TL.load_model(args.short)[0] if args.short else None
    ctx = m1.Ctx.get()
    E2.OUT = OUT
    E2.RB = OUT / "read_cache" / "readback.jsonl"
    suffix = args.tag
    choice_path = OUT / "val_choice.json"

    # ------------------------------------------------------------ settings chosen on val: existence threshold and halting rule
    recs = forward_mode(model, D, items, "normal")
    if args.split == "val":
        grid = []
        for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            for tau in (0.3, 0.5, 0.7):
                r = float(np.mean(rouge_of(decode_recs(dc, recs, thr, tau))))
                grid.append({"thr": thr, "tau": tau, "rouge": r})
                print("grid", thr, tau, round(r, 4), flush=True)
        best = max(grid, key=lambda g: g["rouge"])
        json.dump({"chosen": best, "grid": grid}, open(choice_path, "w"), indent=1)
    choice = json.load(open(choice_path))["chosen"]
    thr, tau = choice["thr"], choice["tau"]
    print("settings", choice, flush=True)

    rows = decode_recs(dc, recs, thr, tau)
    R = {x["item"]: {"id": x["item"], "story": x["story"], "band": x["band"], "question": x["question"], "reference": x["golds"][0], "golds": x["golds"],
                     "text": t, "rouge": m2.rouge_multi(t, x["golds"]) if t else 0.0, "steps": s, "n_nodes": n} for x, _q, t, s, n in rows}

    # ------------------------------------------------------------ cheat (best-overlap sentence, same item set)
    by_id = {x["item"]: x for x in items}
    sents = {}
    ctrees = {}
    for x in items:
        arr = stories[x["skey"]]
        k = E2.idf_pick(x["question"], arr["sent_text"])
        R[x["item"]]["cheat_raw"] = m2.rouge_multi(arr["sent_text"][k], x["golds"]) if k >= 0 else 0.0
        if k >= 0:
            ctrees[x["item"]] = E5.sent_tree(arr, k)
    ids = list(ctrees)
    for i, t in zip(ids, dc.dec.texts([ctrees[i] for i in ids])):
        R[i]["cheat_text"] = t
        R[i]["cheat"] = m2.rouge_multi(t, by_id[i]["golds"])
    for r in R.values():
        r.setdefault("cheat", 0.0)

    # ------------------------------------------------------------ controls
    ctl = {}
    for mode in ("empty", "wrong", "qonly", "shuffled", "block"):
        rr = decode_recs(dc, forward_mode(model, D, items, mode, seed=1), thr, tau)
        sc = []
        if mode == "shuffled":
            sc = [m2.rouge_multi(t, x["golds"]) if t else 0.0 for x, _q, t, _s, _n in rr]
        ctl[mode] = {"n": len(rr), "abstain_rate": float(np.mean([not t for _x, _q, t, _s, _n in rr])), "mean_nodes": float(np.mean([n for *_a, n in rr])),
                     "rouge_vs_own_reference": float(np.mean([m2.rouge_multi(t, x["golds"]) if t else 0.0 for x, _q, t, _s, _n in rr])),
                     "steps_mean": float(np.mean([s for *_a, s, _n in rr]))}
        if mode == "block":
            ctl[mode]["rouge_delta_vs_normal"] = ctl[mode]["rouge_vs_own_reference"] - float(np.mean([r["rouge"] for r in R.values()]))
        print("control", mode, ctl[mode], flush=True)

    # ------------------------------------------------------------ tree score of the decoded text read back (seeded subsample)
    rng = random.Random(0)
    reads = TL.T5.reads_all() if hasattr(TL, "T5") else None
    if reads is None:
        import train_mind_v5 as T5
        reads = T5.reads_all()
    meta = json.load(open(TL.V2B / "items_meta.json"))
    ref_trees = {i: [t for t in (m1.tree_of(reads, g) for g in R[i]["golds"]) if t is not None] for i in R}
    sub = [i for i in R if ref_trees[i]]
    if args.rt_n and args.rt_n < len(sub):
        sub = rng.sample(sub, args.rt_n)
    systems = {"loop": "text", "cheat": "cheat_text"}
    rb = E2._read_texts([R[i][f] for f in systems.values() for i in sub if R[i].get(f)])
    reasons = {}
    for s, f in systems.items():
        rc = Counter()
        for i in sub:
            x = rb.get(R[i].get(f)) if R[i].get(f) else None
            R[i][f"tree_{s}"] = 0.0
            if x is None or x["tree"] is None:
                continue
            tt = tree_from_dict(x["tree"])
            ps = max((ctx.score.score_pair(g, tt, ctx.usvs) for g in ref_trees[i]), key=lambda z: z.overall)
            R[i][f"tree_{s}"] = ps.overall
            rc.update(z["kind"] for z in ps.reasons)
        reasons[s] = dict(rc.most_common())

    # ------------------------------------------------------------ short-trained model
    if short is not None:
        rs = decode_recs(dc, forward_mode(short, D, items, "normal"), thr, tau)
        for x, _q, t, s, n in rs:
            R[x["item"]]["short_rouge"] = m2.rouge_multi(t, x["golds"]) if t else 0.0
            R[x["item"]]["short_steps"] = s

    # ------------------------------------------------------------ tables
    allr = list(R.values())
    mean = E2.mean
    summ = {"n": len(allr), "thr": thr, "tau": tau, "rouge_loop": mean([r["rouge"] for r in allr]), "rouge_cheat_decoded": mean([r["cheat"] for r in allr]),
            "rouge_cheat_raw": mean([r["cheat_raw"] for r in allr]), "answer_rate": mean([bool(r["text"]) for r in allr]),
            "tree_loop": mean([r["tree_loop"] for r in allr if "tree_loop" in r]), "tree_cheat": mean([r["tree_cheat"] for r in allr if "tree_cheat" in r]),
            "tree_n": sum(1 for r in allr if "tree_loop" in r), "tree_reasons": reasons, "controls": ctl,
            "mean_nodes": mean([r["n_nodes"] for r in allr]), "steps_dist": {str(k): v for k, v in sorted(Counter(r["steps"] for r in allr).items())},
            "steps_mean": mean([r["steps"] for r in allr])}
    cis = {}
    for nm, g in (("cheat", lambda r: r["cheat"]),):
        d, bi, bs = E5.boot(allr, lambda r: r["rouge"], g)
        cis[f"loop_minus_{nm}"] = {"mean": d, "items": [float(bi[0]), float(bi[1])], "stories": [float(bs[0]), float(bs[1])]}
    if short is not None:
        d, bi, bs = E5.boot(allr, lambda r: r["rouge"], lambda r: r["short_rouge"])
        cis["loop_minus_short"] = {"mean": d, "items": [float(bi[0]), float(bi[1])], "stories": [float(bs[0]), float(bs[1])]}
    summ["cis"] = cis
    bands = {}
    for b in BANDS:
        rb_ = [r for r in allr if r["band"] == b]
        bands[b] = {"n": len(rb_), "loop": mean([r["rouge"] for r in rb_]), "cheat": mean([r["cheat"] for r in rb_]), "steps": mean([r["steps"] for r in rb_]),
                    "nodes": mean([r["n_nodes"] for r in rb_]), "short": mean([r["short_rouge"] for r in rb_]) if short is not None else None}
    summ["bands"] = bands
    json.dump(summ, open(OUT / f"eval_{args.split}{suffix}.json", "w"), indent=1)
    with open(OUT / f"eval_rows_{args.split}{suffix}.jsonl", "w") as f:
        for r in allr:
            f.write(json.dumps(r) + "\n")
    L = [f"## {args.split}: n={len(allr)} items, thr={thr}, tau={tau}", "", "| system | full-sentence ROUGE-L | tree score % |", "|---|---|---|",
         f"| loop (decoded output tree, setting (b)) | {summ['rouge_loop']:.3f} | {summ['tree_loop']:.1f} (n={summ['tree_n']}) |",
         f"| cheat: best-overlap sentence (decoded) | {summ['rouge_cheat_decoded']:.3f} (raw text {summ['rouge_cheat_raw']:.3f}) | {summ['tree_cheat']:.1f} |"]
    if args.split == "test":
        L += [f"| v3_s2 (MIND_V5.md, same items) | {V5_TEST['v3_s2']:.3f} | {V5_TREE['v3_s2']} |", f"| v5_all (MIND_V5.md) | {V5_TEST['v5_all']:.3f} | {V5_TREE['v5_all']} |"]
    L += ["", f"answer rate {summ['answer_rate']:.3f}, mean output nodes {summ['mean_nodes']:.2f}, mean steps {summ['steps_mean']:.2f}, steps distribution {summ['steps_dist']}", "",
          f"loop minus cheat: {json.dumps(cis['loop_minus_cheat'])}", "", "| band | n | loop | cheat | short-only | mean steps | mean nodes |", "|---|---|---|---|---|---|---|"]
    for b in BANDS:
        x = bands[b]
        L.append(f"| {b} | {x['n']} | {x['loop']:.3f} | {x['cheat']:.3f} | {'' if x['short'] is None else format(x['short'], '.3f')} | {x['steps']:.2f} | {x['nodes']:.2f} |")
    L += ["", "### Controls", "", "| control | n | abstain (empty tree) rate | mean nodes | ROUGE vs own reference | mean steps |", "|---|---|---|---|---|---|"]
    for k, v in ctl.items():
        L.append(f"| {k} | {v['n']} | {v['abstain_rate']:.3f} | {v['mean_nodes']:.2f} | {v['rouge_vs_own_reference']:.3f} | {v['steps_mean']:.2f} |")
    L += ["", f"block-reads ROUGE delta vs normal: {ctl['block'].get('rouge_delta_vs_normal', 0):+.3f}", "", f"tree-score reasons (loop): {reasons['loop']}"]
    open(OUT / f"tables_{args.split}{suffix}.md", "w").write("\n".join(L) + "\n")
    hr = random.Random(7).sample([r for r in allr if r["text"]], min(20, sum(1 for r in allr if r["text"])))
    json.dump([{k: r[k] for k in ("id", "question", "reference", "text", "rouge", "steps", "n_nodes", "band")} for r in hr], open(OUT / f"handread_{args.split}{suffix}.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
