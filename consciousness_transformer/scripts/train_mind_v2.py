"""Train the Mind v2 thinker.  Stage 1: toward the reference answer tree (anchor + keep labels by lemma alignment, 20-25%
wrong-passage items -> ABSTAIN, length curriculum).  Stage 2: REINFORCE on the ROUGE-L of the decoded answer text.
    nohup nice -n 5 python scripts/train_mind_v2.py --stage 1 --variant all --epochs 24 --tag s1_all > runs/mind_v2/s1_all.log 2>&1 &
    nohup nice -n 5 python scripts/train_mind_v2.py --stage 2 --init runs/mind_v2/thinker_s1_all_best.pt --epochs 6 --tag s2_all > runs/mind_v2/s2_all.log 2>&1 &
"""
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
import torch.nn.functional as F

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
from naomi import mind_v1 as m1  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402

OUT = CT / "runs" / "mind_v2"
ABSTAIN_REWARD = 0.10          # declared before any test number: abstaining on a normal item pays 0.10 (a wrong answer ~0, a good one >> 0.10)
WRONG_ABSTAIN_REWARD = 1.0     # wrong-passage item: abstaining is right
K_SAMPLES = 4


def curriculum_weight(band: str, ep: int, epochs: int) -> float:
    """Stage-1 item weight: short answer sentences always 1; 11-20 words ramps 0.3 -> 1 by 40% of training; 21+ ramps 0.1 -> 1 by 70%."""
    f = ep / max(epochs - 1, 1)
    if band == "<=10":
        return 1.0
    if band == "11-20":
        return min(1.0, 0.3 + 0.7 * f / 0.4)
    return min(1.0, 0.1 + 0.9 * f / 0.7)


def load_all():
    stories = pickle.load(open(OUT / "stories.pkl", "rb"))
    pairs = {sp: pickle.load(open(OUT / f"pairs_{sp}.pkl", "rb")) for sp in ("train", "val")}
    meta = json.load(open(OUT / "items_meta.json"))
    vocab = json.load(open(OUT / "vocab.json"))
    return stories, pairs, meta, vocab


def make_model(vocab, seed=0, **kw):
    torch.manual_seed(seed)
    return m2.Thinker(m1.Ctx.get().D, len(vocab["rels"]), len(vocab["poss"]), **kw)


def save(model, vocab, path, extra=None):
    torch.save({"state": model.state_dict(), "vocab": vocab, "extra": extra or {}}, path)


def load_model(path, vocab=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    v = ck["vocab"]
    model = make_model(v)
    model.load_state_dict(ck["state"])
    return model.eval(), v


@torch.no_grad()
def val_rouge(model, stories, pairs, meta, tab, decoder=None, bs=64):
    """Mean ROUGE-L (abstain = 0) on normal pairs, answered share, wrong-passage abstain rate. True word order unless a decoder is given."""
    model.eval()
    sc, ans, wa, wn = [], 0, 0, 0
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        b = m2.collate(ch, stories, tab, with_targets=False)
        e, anch, _ = m2.decide(model, b)
        kept, _ = m2.keep_sets(model, e, b, stories, ch, anch)
        trees, idx = [], []
        for j, (p, a, k) in enumerate(zip(ch, anch, kept)):
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


def stage1_loss(model, b, pairs, stories, weights):
    e = model.encode(b)
    B = b["B"]
    A, Nn, It, Y = [], [], [], []
    anc_loss = []
    for i, p in enumerate(pairs):
        o0, o1 = int(b["s_off"][i]), int(b["s_off"][i + 1])
        lg = torch.cat([e["alog"][o0:o1], e["ablog"][i:i + 1]])
        lse = torch.logsumexp(lg, 0)
        if p["kind"] == "wrong":
            anc_loss.append(lse - e["ablog"][i])
        else:
            acc = [a for a, _k, _f in p["tgt"]["anchors"]]
            anc_loss.append(lse - torch.logsumexp(e["alog"][o0:o1][torch.tensor(acc)], 0))
            arr = stories[p["skey"]]
            for a, kept, _f in p["tgt"]["anchors"]:
                d = arr["desc"][a]
                if len(d) == 0:
                    continue
                ks = set(kept)
                A += [o0 + a] * len(d)
                Nn += (o0 + d).tolist()
                It += [i] * len(d)
                Y += [1.0 if g in ks else 0.0 for g in d.tolist()]
    w = torch.tensor(weights, dtype=torch.float32)
    loss_a = (torch.stack(anc_loss) * w).sum() / w.sum().clamp(min=1e-6)
    loss_k = torch.zeros(())
    if A:
        lg = model.keep_logits(e, b, torch.tensor(A), torch.tensor(Nn), torch.tensor(It))
        bce = F.binary_cross_entropy_with_logits(lg, torch.tensor(Y), reduction="none")
        wi = w[torch.tensor(It)]
        loss_k = (bce * wi).sum() / wi.sum().clamp(min=1e-6)
    return loss_a, loss_k


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
    print(f"stage1 variant={args.variant}: normal {len(normal)} (skipped unreachable {skipped}), wrong-passage {len(wrong)}", flush=True)
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
            ws = [1.0 if p["kind"] == "wrong" else (curriculum_weight(m2.band_of(p["tgt"]["ans_len"]), ep, args.epochs) if args.variant == "all" else 1.0) for p in ch]
            b = m2.collate(ch, stories, tab)
            la, lk = stage1_loss(model, b, ch, stories, ws)
            loss = la + args.keep_w * lk
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
    with open(OUT / f"thinker_{args.tag}.curve.tsv", "w") as f:
        f.write("epoch\tloss\tval_rouge\tanswer_rate\twrong_abstain\n")
        for c in curve:
            f.write("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in c) + "\n")


def stage2(args, stories, pairs, meta, vocab, tab):
    rng = random.Random(args.seed)
    model, _ = load_model(args.init)
    dec = m2.Decoder(str(CT / "runs/naomi_orderer/large_best.pt"))
    data = [p for p in pairs["train"] if p["kind"] in ("normal", "wrong")]
    if args.variant == "short":
        short_items = {p["item"] for p in data if p["kind"] == "normal" and p.get("tgt") and p["tgt"]["ans_len"] is not None and p["tgt"]["ans_len"] <= 10}
        data = [p for p in data if p["item"] in short_items]
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
            samples = []          # per k: anchors, kept, logp
            lps = [[] for _ in range(B)]
            rew = np.zeros((K_SAMPLES, B), dtype=np.float32)
            logp = [[None] * B for _ in range(K_SAMPLES)]
            kept_k, anch_k = [], []
            for k in range(K_SAMPLES):
                anch = []
                for j in range(B):
                    o0, o1 = int(b["s_off"][j]), int(b["s_off"][j + 1])
                    lg = torch.cat([e["alog"][o0:o1], e["ablog"][j:j + 1]])
                    lp = torch.log_softmax(lg, 0)
                    c = int(torch.multinomial(lp.exp().detach(), 1))
                    logp[k][j] = lp[c]
                    anch.append(-1 if c == o1 - o0 else c)
                kept, klp = m2.keep_sets(model, e, b, stories, ch, anch, sample=True)
                for j in range(B):
                    logp[k][j] = logp[k][j] + klp[j]
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
            base = (rew.sum(0, keepdims=True) - rew) / (K_SAMPLES - 1)      # leave-one-out baseline
            adv = torch.from_numpy(rew - base)
            lp_all = torch.stack([torch.stack(logp[k]) for k in range(K_SAMPLES)])
            loss = -(adv * lp_all).mean()
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
    with open(OUT / f"thinker_{args.tag}.curve.tsv", "w") as f:
        f.write("epoch\ttrain_reward_or_loss\tval_rouge\tanswer_rate\twrong_abstain\n")
        for c in curve:
            f.write("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in c) + "\n")


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--tag", default="s1_all")
    ap.add_argument("--dir", default="mind_v2", help="run directory under runs/ (mind_v2b for the full FairytaleQA)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    global OUT
    OUT = CT / "runs" / args.dir
    torch.set_num_threads(args.threads)
    stories, pairs, meta, vocab = load_all()
    tab = m2.SenseTable()
    (stage1 if args.stage == 1 else stage2)(args, stories, pairs, meta, vocab, tab)


if __name__ == "__main__":
    main()
