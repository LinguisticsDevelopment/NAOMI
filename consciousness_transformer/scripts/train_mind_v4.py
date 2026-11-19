"""Train the Mind v4 thinker (steps down the tree; see dev/reports/MIND_V4.md). Stage 1 only: imitate the path from the start
sentence's root to the oracle anchor (answer-derived targets), plus the keep head.
    nohup nice -n 5 python scripts/train_mind_v4.py --variant all --epochs 24 --tag v4_all > runs/mind_v4/v4_all.log 2>&1 &
Data (stories / pairs / vocab) is read from runs/mind_v2b; outputs go to runs/mind_v4."""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_v1 as m1  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402
from naomi import mind_v4 as m4  # noqa: E402
import train_mind_v3 as T3  # noqa: E402

OUT = CT / "runs" / "mind_v4"


def make_model(vocab, seed=0, **kw):
    torch.manual_seed(seed)
    return m4.Thinker4(m1.Ctx.get().D, len(vocab["rels"]), len(vocab["poss"]), **kw)


def save(model, vocab, path, extra=None):
    torch.save({"kind": "v4", "state": model.state_dict(), "vocab": vocab, "cfg": {"H": model.H}, "extra": extra or {}}, path)


def load_model(path, vocab=None):
    """v4 checkpoints -> Thinker4; anything else (v3, v2 / v2b) -> the older loaders."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != "v4":
        return T3.load_model(path)
    model = make_model(ck["vocab"], **ck["cfg"])
    model.load_state_dict(ck["state"])
    return model.eval(), ck["vocab"]


@torch.no_grad()
def val_rouge(model, stories, pairs, meta, tab, decoder=None, bs=64, nstart=1, **kw):
    model.eval()
    sc, ans, wa, wn, steps, exh = [], 0, 0, 0, [], 0
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        b = m2.collate(ch, stories, tab, with_targets=False)
        e, anch, res = m4.decide(model, b, nstart=nstart, **kw)
        kept, _ = m2.keep_sets(model, e, b, stories, ch, anch)
        trees, idx = [], []
        for p, a, k, r in zip(ch, anch, kept, res):
            if p["kind"] == "wrong":
                wn += 1
                wa += a < 0
                continue
            if p["kind"] != "normal":
                continue
            steps.append(r["steps"])
            exh += r["exhausted"]
            if a < 0:
                sc.append(0.0)
                continue
            ans += 1
            arr = stories[p["skey"]]
            if decoder is None:
                sc.append(m2.rouge_multi(m2.true_order_text(arr, k), meta[p["item"]]["golds"]))
            else:
                trees.append(m2.answer_tree(arr, a, k))
                idx.append((len(sc), p))
                sc.append(0.0)
        if trees:
            for (pos, p), t in zip(idx, decoder.texts(trees)):
                sc[pos] = m2.rouge_multi(t, meta[p["item"]]["golds"])
    n = len(sc)
    return {"rouge": float(np.mean(sc)) if n else 0.0, "answer_rate": ans / max(n, 1), "wrong_abstain": wa / max(wn, 1), "n": n,
            "steps": float(np.mean(steps)) if steps else 0.0, "exhausted": exh / max(n, 1)}


def stage1(args, stories, pairs, meta, vocab, tab):
    rng = random.Random(args.seed)
    tr = pairs["train"]
    normal, wrong, skipped = [], [], 0
    short_items = {p["item"] for p in tr if p["kind"] == "normal" and p.get("tgt") and p["tgt"]["ans_len"] is not None and p["tgt"]["ans_len"] <= 10}
    for p in tr:
        if p["kind"] == "wrong":
            if args.variant == "all" or p["item"] in short_items:
                wrong.append(p)
        elif p["kind"] == "normal":
            t = p.get("tgt")
            if not t or t["best_f1"] < m2.ACCEPT_F1:
                skipped += 1
                continue
            if args.variant == "short" and not (t["ans_len"] is not None and t["ans_len"] <= 10):
                continue
            normal.append(p)
    print(f"stage1 v4 variant={args.variant}: normal {len(normal)} (skipped unreachable {skipped}), wrong-passage {len(wrong)}", flush=True)
    val = [p for p in pairs["val"] if p["kind"] in ("normal", "wrong")]
    model = make_model(vocab, args.seed, H=args.H, drop=args.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, curve = -1.0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        data = normal + wrong
        rng.shuffle(data)
        tl = 0.0
        for i in range(0, len(data), args.bs):
            ch = data[i:i + args.bs]
            b = m2.collate(ch, stories, tab)
            lp, ls, lk = m4.stage1_loss(model, b, ch, stories, rng, rand_start=bool(args.rand_start))
            loss = lp + args.sent_w * ls + args.keep_w * lk
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss.detach()) * len(ch)
        v = val_rouge(model, stories, val, meta, tab)
        curve.append((ep, tl / len(data), v["rouge"], v["answer_rate"], v["wrong_abstain"], v["steps"], v["exhausted"]))
        print(f"ep {ep} loss {tl / len(data):.4f} val_rouge {v['rouge']:.4f} answer_rate {v['answer_rate']:.3f} wrong_abstain {v['wrong_abstain']:.3f} steps {v['steps']:.2f} exhausted {v['exhausted']:.3f} {time.time() - t0:.0f}s", flush=True)
        if v["rouge"] > best:
            best = v["rouge"]
            save(model, vocab, OUT / f"thinker_{args.tag}_best.pt", {"epoch": ep, "val": v, "args": vars(args)})
    T3.write_curve(OUT / f"thinker_{args.tag}.curve.tsv", "epoch\tloss\tval_rouge\tanswer_rate\twrong_abstain\tsteps\texhausted", curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["all", "short"], default="all")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--keep_w", type=float, default=1.0)
    ap.add_argument("--sent_w", type=float, default=1.0, help="weight of the sentence-value softmax loss (the value the moves climb and the start uses)")
    ap.add_argument("--rand_start", type=int, default=1, help="1: also train from one random start sentence per item (teaches NEXT/PREV from anywhere)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v4_all")
    ap.add_argument("--dir", default="mind_v2b")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_items", type=int, default=0, help="smoke runs: cap the number of training items")
    args = ap.parse_args()
    T3.DATA = CT / "runs" / args.dir
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    stories, pairs, meta, vocab = T3.load_all()
    if args.max_items:
        pairs["train"] = pairs["train"][:args.max_items]
        pairs["val"] = pairs["val"][:args.max_items]
    tab = m2.SenseTable()
    stage1(args, stories, pairs, meta, vocab, tab)


if __name__ == "__main__":
    main()
