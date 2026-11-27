"""Train looped thinker v2 (dev/reports/MIND_LOOP_V2.md). Reuses runs/mind_loop prep files (read only); writes runs/mind_loop_v2/.
    nice -n 5 python scripts/train_mind_loop_v2.py train --tag v2_s0 --seed 0 --epochs 12 > runs/mind_loop_v2/v2_s0.log
Schedule: epochs [0, --warm) hold the halting distribution at the geometric prior (lambda 1/3); then PonderNet learned halt with KL weight --beta.
Counterfactual items: a fraction --p_cf of training items get an extra item with swapped memory slots + swapped reference. Wrong/empty items as v1.
Learning rate: linear warm-up 100 steps, cosine decay to 10% over --epochs. Checkpoint = best val score AFTER release."""
from __future__ import annotations

import argparse
import json
import math
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
from naomi import mind_loop as ML  # noqa: E402
from naomi import mind_loop_v2 as V2  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402
import train_mind_loop as TL  # noqa: E402

OUT = CT / "runs" / "mind_loop_v2"


def cf_items(cf: V2.CF, items, stories, D, seed=0, limit=0):
    """Deterministic counterfactual val/test items: dicts like items plus cf key, reference form, swapped gold strings. Registers memory in D.sf / stories."""
    out = []
    for x in items:
        rng = random.Random(TL.zlib_salt(x["item"]) + seed)
        r = cf.build(x["skey"], x["q"], x["refs"][0], rng)
        if r is None:
            continue
        golds = [V2.swap_text(g, r["swaps"]) for g in x["golds"]]
        if golds == x["golds"]:
            continue                                  # the swapped token never shows in the answer text: not scoreable on text
        k = "cf|" + x["item"]
        D.sf[k] = {"mem": r["mem"], "keys": r["keys"]}
        stories[k] = {"nodes": r["nodes"]}
        out.append({**x, "cf_key": k, "cf_ref": r["ref"], "cf_golds": golds, "swaps": r["swaps"]})
    return out[:limit] if limit else out


def train_reqs_v2(items, D, rng, cf, p_wrong=0.12, p_empty=0.06, p_cf=0.4):
    reqs = TL.train_reqs(items, D, rng, p_wrong, p_empty)
    keys = []
    for x in items:
        if rng.random() < p_cf:
            r = cf.build(x["skey"], x["q"], rng.choice(x["refs"]), rng)
            if r is not None:
                k = f"cf|{x['item']}|{rng.randrange(1 << 30)}"
                D.sf[k] = {"mem": r["mem"], "keys": r["keys"]}
                keys.append(k)
                reqs.append({"mem": k, "q": x["q"], "t": r["ref"], "qid": x["item"]})
    return reqs, keys


@torch.no_grad()
def cf_val(model, D, dc, items, thr=0.5, tau=0.5, bs=24):
    """ROUGE-L of decoded output on counterfactual items vs the SWAPPED golds; and rate of the new token in the output text."""
    model.eval()
    sc, new, old = [], [], []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        reqs = [{"mem": x["cf_key"], "q": x["q"], "t": None, "qid": x["item"]} for x in ch]
        outs, hl, _ = TL.forward_batch(model, D, reqs)
        txt, _ = dc.texts(model, outs, ML.halting(hl), ch, [x["cf_key"] for x in ch], thr, tau, [TL.zlib_salt(x["cf_key"]) for x in ch])
        for t, x in zip(txt, ch):
            sc.append(m2.rouge_multi(t, x["cf_golds"]) if t else 0.0)
            w = {c.strip(".,;:!?\"'()").lower() for c in t.split()}
            new.append(float(any(d.lower() in w for _o, d in x["swaps"])))
            old.append(float(any(o.lower() in w for o, _d in x["swaps"])))
    return float(np.mean(sc)), float(np.mean(new)), float(np.mean(old))


@torch.no_grad()
def val_halt(model, D, items, warm_lam=V2.LAM_PRIOR, bs=24):
    """Per-step tree loss (mean over items), mean halting probability per step, mean halting step (tau 0.5); learned halt."""
    model.eval()
    ls, ps, n = 0, 0, 0
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        reqs = [{"mem": x["skey"], "q": x["q"], "t": x["refs"][0], "qid": x["item"]} for x in ch]
        outs, hl, tg = TL.forward_batch(model, D, reqs)
        Ls = torch.stack([ML.tree_loss(o, tg) for o in outs], 1)
        ls = ls + Ls.sum(0)
        ps = ps + ML.halting(hl).sum(0)
        n += len(ch)
    ls, ps = ls / n, ps / n
    return ls.tolist(), ps.tolist(), float(sum((k + 1) * p for k, p in enumerate(ps.tolist())))


def train(a):
    rng = random.Random(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    D = TL.Data()
    stories = pickle.load(open(TL.V2B / "stories.pkl", "rb"))
    dc = TL.Decode(D, stories)
    tr, va = D.items["train"], D.items["val"]
    cf_tr = V2.CF(D.sf, stories, {x["skey"] for x in tr})
    cf_va = V2.CF(D.sf, stories, {x["skey"] for x in va})
    vsub = random.Random(0).sample(va, min(a.val_n, len(va)))
    vcf = cf_items(cf_va, va, stories, D, limit=0)
    vcf = random.Random(0).sample(vcf, min(a.val_n, len(vcf)))
    by_story = {}
    if a.limit_stories:
        keep = set(sorted({x['skey'] for x in tr})[:a.limit_stories])
        tr = [x for x in tr if x['skey'] in keep]
    for x in tr:
        by_story.setdefault(x["skey"], []).append(x)
    model = TL.make_model(D.dims, a.seed, H=a.H)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    nb_ep = math.ceil(len(by_story) / a.S)
    total = a.epochs * nb_ep
    print(f"items {len(tr)} stories {len(by_story)} val_cf {len(vcf)} params {sum(p.numel() for p in model.parameters())} warm {a.warm} lam {a.lam:.4f} beta {a.beta} p_cf {a.p_cf}", flush=True)
    t0, step, best = time.time(), 0, -9.0
    curve = open(OUT / f"{a.tag}.curve.tsv", "w")
    curve.write("epoch\tminutes\twarm\ttrain_loss\tval_loss\tval_steps\tval_rouge\tcf_rouge\tcf_new\tcf_old\tper_step_loss\thalt_p\n")
    for ep in range(a.epochs):
        model.train()
        warm = V2.is_warm(ep, a.warm)
        sk = sorted(by_story)
        rng.shuffle(sk)
        tl, nb = 0.0, 0
        for i in range(0, len(sk), a.S):
            items = [x for s in sk[i:i + a.S] for x in by_story[s]]
            reqs, ck = train_reqs_v2(items, D, rng, cf_tr, 0.12, 0.06, a.p_cf)
            outs, hl, tg = TL.forward_batch(model, D, reqs, train=True, rng=rng)
            loss, _Ls, _p = V2.ponder_loss_v2(outs, hl, tg, a.lam, a.beta, warm)
            cos = 0.5 * (1 + math.cos(math.pi * step / total))
            lr = a.lr * (0.1 + 0.9 * cos) * min(1.0, (step + 1) / 100)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in ck:
                del D.sf[k]
            step += 1
            tl += float(loss)
            nb += 1
        vl, vls, vst = TL.val_loss(model, D, vsub)
        pl, pp, ms = val_halt(model, D, vsub)
        vr = TL.val_rouge(model, D, dc, vsub)
        cr, cn, co = cf_val(model, D, dc, vcf)
        mins = (time.time() - t0) / 60
        print(f"ep {ep} {mins:.1f}min warm={int(warm)} train {tl / nb:.4f} val_loss {vl:.4f} steps {ms:.2f} val_rouge {vr:.4f} cf_rouge {cr:.4f} cf_new {cn:.3f} cf_old {co:.3f} per-step {[round(x, 3) for x in pl]} halt_p {[round(x, 3) for x in pp]}", flush=True)
        curve.write(f"{ep}\t{mins:.2f}\t{int(warm)}\t{tl / nb:.4f}\t{vl:.4f}\t{ms:.3f}\t{vr:.4f}\t{cr:.4f}\t{cn:.3f}\t{co:.3f}\t{json.dumps([round(x, 4) for x in pl])}\t{json.dumps([round(x, 4) for x in pp])}\n")
        curve.flush()
        TL.save(model, D.dims, OUT / f"{a.tag}_last.pt")
        score = 0.5 * (vr + cr) - 1e-3 * vl
        if not warm and score > best:
            best = score
            TL.save(model, D.dims, OUT / f"{a.tag}_best.pt", {"epoch": ep, "val_rouge": vr, "cf_rouge": cr, "val_loss": vl})
        if a.max_minutes and mins > a.max_minutes:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--tag", default="v2_s0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--S", type=int, default=12)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=V2.LAM_PRIOR)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--p_cf", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_stories", type=int, default=0)
    ap.add_argument("--val_n", type=int, default=150)
    a = ap.parse_args()
    train(a)
