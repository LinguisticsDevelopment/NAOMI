"""Train the Mind v3 thinker (sentence step + sentence context + length-invariant shape; see dev/reports/MIND_V3.md).
Stage 1: toward the reference answer tree (sentence softmax + node-in-sentence softmax + keep), ALL lengths uniformly (no curriculum).
Stage 2: REINFORCE on the ROUGE-L of the decoded answer (abstain reward as v2).
    nohup nice -n 5 python scripts/train_mind_v3.py --dir mind_v2b --stage 1 --epochs 24 --tag v3_s1 > runs/mind_v3/v3_s1.log 2>&1 &
Data (stories / pairs / vocab) is read from runs/<--dir> (the v2b build); outputs go to runs/mind_v3."""
from __future__ import annotations

import argparse
import json
import pickle
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
from naomi import mind_v3 as m3  # noqa: E402
import train_mind_v2 as T2  # noqa: E402

DATA = CT / "runs" / "mind_v2b"
OUT = CT / "runs" / "mind_v3"
ABSTAIN_REWARD = T2.ABSTAIN_REWARD            # as v2 (declared before any test number)
WRONG_ABSTAIN_REWARD = T2.WRONG_ABSTAIN_REWARD
K_SAMPLES = T2.K_SAMPLES


def make_model(vocab, seed=0, inv=True, ctx=True, **kw):
    torch.manual_seed(seed)
    return m3.Thinker3(m1.Ctx.get().D, len(vocab["rels"]), len(vocab["poss"]), inv=inv, ctx=ctx, **kw)


def save(model, vocab, path, extra=None):
    cfg = {"inv": model.inv, "ctx": model.use_ctx, "H": model.H}
    torch.save({"kind": "v3", "state": model.state_dict(), "vocab": vocab, "cfg": cfg, "extra": extra or {}}, path)


def load_model(path, vocab=None):
    """v3 checkpoints -> Thinker3; anything else (v2 / v2b checkpoints) -> the v2 thinker."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") != "v3":
        return T2.load_model(path)
    model = make_model(ck["vocab"], **ck["cfg"])
    model.load_state_dict(ck["state"])
    return model.eval(), ck["vocab"]


@torch.no_grad()
def val_rouge(model, stories, pairs, meta, tab, decoder=None, bs=64, topk=1):
    model.eval()
    sc, ans, wa, wn = [], 0, 0, 0
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        b = m2.collate(ch, stories, tab, with_targets=False)
        e, anch, _ = m3.decide(model, b, topk=topk)
        kept, _ = m2.keep_sets(model, e, b, stories, ch, anch)
        trees, idx = [], []
        for p, a, k in zip(ch, anch, kept):
            if p["kind"] == "wrong":
                wn += 1
                wa += a < 0
                continue
            if p["kind"] != "normal":
                continue
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
    return {"rouge": float(np.mean(sc)) if n else 0.0, "answer_rate": ans / max(n, 1), "wrong_abstain": wa / max(wn, 1), "n": n}


def load_all():
    stories = pickle.load(open(DATA / "stories.pkl", "rb"))
    pairs = {sp: pickle.load(open(DATA / f"pairs_{sp}.pkl", "rb")) for sp in ("train", "val")}
    meta = json.load(open(DATA / "items_meta.json"))
    vocab = json.load(open(DATA / "vocab.json"))
    return stories, pairs, meta, vocab


def write_curve(path, head, curve):
    with open(path, "w") as f:
        f.write(head + "\n")
        for c in curve:
            f.write("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in c) + "\n")


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
    print(f"stage1 v3 variant={args.variant} inv={args.inv} ctx={args.ctx}: normal {len(normal)} (skipped unreachable {skipped}), wrong-passage {len(wrong)}", flush=True)
    val = [p for p in pairs["val"] if p["kind"] in ("normal", "wrong")]
    model = make_model(vocab, args.seed, inv=args.inv, ctx=bool(args.ctx), H=args.H, drop=args.drop)
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
            ls, la, lk = m3.stage1_loss(model, b, ch, stories, [1.0] * len(ch))          # uniform: no length curriculum
            loss = ls + args.anchor_w * la + args.keep_w * lk
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss) * len(ch)
        v = val_rouge(model, stories, val, meta, tab)
        curve.append((ep, tl / len(data), v["rouge"], v["answer_rate"], v["wrong_abstain"]))
        print(f"ep {ep} loss {tl / len(data):.4f} val_rouge {v['rouge']:.4f} answer_rate {v['answer_rate']:.3f} wrong_abstain {v['wrong_abstain']:.3f} {time.time() - t0:.0f}s", flush=True)
        if v["rouge"] > best:
            best = v["rouge"]
            save(model, vocab, OUT / f"thinker_{args.tag}_best.pt", {"epoch": ep, "val": v, "args": vars(args)})
    write_curve(OUT / f"thinker_{args.tag}.curve.tsv", "epoch\tloss\tval_rouge\tanswer_rate\twrong_abstain", curve)


def stage2(args, stories, pairs, meta, vocab, tab):
    rng = random.Random(args.seed)
    model, _ = load_model(args.init)
    dec = m2.Decoder(str(CT / "runs/naomi_orderer/large_best.pt"))
    data = [p for p in pairs["train"] if p["kind"] in ("normal", "wrong")]
    val = [p for p in pairs["val"] if p["kind"] in ("normal", "wrong")]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    v0 = val_rouge(model, stories, val, meta, tab, dec)
    print(f"stage2 start: val_rouge(decoded) {v0['rouge']:.4f} answer_rate {v0['answer_rate']:.3f} wrong_abstain {v0['wrong_abstain']:.3f}", flush=True)
    best = v0["rouge"]
    save(model, vocab, OUT / f"thinker_{args.tag}_best.pt", {"epoch": -1, "val": v0, "args": vars(args)})
    curve = [(-1, 0.0, v0["rouge"], v0["answer_rate"], v0["wrong_abstain"])]
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(data)
        rs = []
        for i in range(0, len(data), args.bs):
            ch = data[i:i + args.bs]
            b = m2.collate(ch, stories, tab, with_targets=False)
            e = model.encode(b)
            B = len(ch)
            rew = np.zeros((K_SAMPLES, B), dtype=np.float32)
            logp, kept_k, anch_k = [], [], []
            for k in range(K_SAMPLES):
                anch, lps = m3.sample(e, b)
                kept, klp = m2.keep_sets(model, e, b, stories, ch, anch, sample=True)
                logp.append(torch.stack([lps[j] + klp[j] for j in range(B)]))
                anch_k.append(anch)
                kept_k.append(kept)
            trees, where = [], []
            for k in range(K_SAMPLES):
                for j, p in enumerate(ch):
                    a = anch_k[k][j]
                    wrong = p["kind"] == "wrong"
                    if a < 0:
                        rew[k, j] = WRONG_ABSTAIN_REWARD if wrong else ABSTAIN_REWARD
                    elif wrong:
                        rew[k, j] = 0.0
                    else:
                        trees.append(m2.answer_tree(stories[p["skey"]], a, kept_k[k][j]))
                        where.append((k, j))
            for (k, j), t in zip(where, dec.texts(trees)):
                rew[k, j] = m2.rouge_multi(t, meta[ch[j]["item"]]["golds"])
            base = (rew.sum(0, keepdims=True) - rew) / (K_SAMPLES - 1)
            adv = torch.from_numpy(rew - base)
            loss = -(adv * torch.stack(logp)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rs.append(float(rew.mean()))
        v = val_rouge(model, stories, val, meta, tab, dec)
        curve.append((ep, float(np.mean(rs)), v["rouge"], v["answer_rate"], v["wrong_abstain"]))
        print(f"ep {ep} train_reward {np.mean(rs):.4f} val_rouge {v['rouge']:.4f} answer_rate {v['answer_rate']:.3f} wrong_abstain {v['wrong_abstain']:.3f} {time.time() - t0:.0f}s", flush=True)
        if v["rouge"] > best:
            best = v["rouge"]
            save(model, vocab, OUT / f"thinker_{args.tag}_best.pt", {"epoch": ep, "val": v, "args": vars(args)})
    write_curve(OUT / f"thinker_{args.tag}.curve.tsv", "epoch\ttrain_reward_or_loss\tval_rouge\tanswer_rate\twrong_abstain", curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--variant", choices=["all", "short"], default="all")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--keep_w", type=float, default=1.0)
    ap.add_argument("--anchor_w", type=float, default=1.0)
    ap.add_argument("--inv", type=int, default=1, help="1: length-invariant shape features / pooling (size = share of the sentence); 2: same with size = share of the parent's subtree; 0: v2's raw depth, size, max + count pooling (ablation)")
    ap.add_argument("--ctx", type=int, default=1, help="1: cross-sentence context; 0: sentences are scored alone (ablation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--tag", default="v3_s1")
    ap.add_argument("--dir", default="mind_v2b", help="data directory under runs/ (the v2b build)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    global DATA
    DATA = CT / "runs" / args.dir
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    stories, pairs, meta, vocab = load_all()
    tab = m2.SenseTable()
    (stage1 if args.stage == 1 else stage2)(args, stories, pairs, meta, vocab, tab)


if __name__ == "__main__":
    main()
