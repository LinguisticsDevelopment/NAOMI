"""Train looped thinker v3 (dev/reports/MIND_LOOP_V3.md). Reuses runs/mind_loop prep files (read only); writes runs/mind_loop_v3/.
    nohup nice -n 5 python scripts/train_mind_loop_v3.py train --tag v3_s0 --seed 0 --epochs 16 > runs/mind_loop_v3/v3_s0.log 2>&1 &
Only the training grade differs from v2 (naomi/mind_loop_v3.py): expected asymmetric reward over the candidate pool. Same halt schedule (warm-up at the
prior for --warm epochs, then PonderNet + KL), same counterfactual items, same lr schedule. --commit epochs (default = --warm) bypass the existence head.
Checkpoint = best (lowest) val loss AFTER release (the loss is the negative expected reward + structure)."""
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
from naomi import mind_loop_v3 as V3  # noqa: E402
import train_mind_loop as TL  # noqa: E402
import train_mind_loop_v2 as T2  # noqa: E402

OUT = CT / "runs" / "mind_loop_v3"


def forward_batch(model, D, reqs, nstep=ML.NSTEP, block_read=False, train=False, rng=None):
    """As ``train_mind_loop.forward_batch`` (same books, memory, targets) and also returns the candidate pool. -> outs, halting logits, targets, pool."""
    keys = []
    for r in reqs:
        if r["mem"] is not None and r["mem"] not in keys:
            keys.append(r["mem"])
    salts = {k: (rng.randrange(1 << 30) if train else TL.zlib_salt(k)) for k in keys}
    books = {k: ML.CodeBook(salts[k]) for k in keys}
    mems = []
    for k in keys:
        if isinstance(k, tuple):
            q = [r["q"] for r in reqs if r["mem"] == k][0]
            mems.append({**q, "sent": np.zeros(q["n"], dtype=np.int64), "tidx": np.arange(q["n"], dtype=np.int64)})
        else:
            mems.append(D.sf[k]["mem"])
    state = model.init_state(len(keys) + 1)
    mt = ML.memory_tensors(mems + [None], [books[k] for k in keys] + [ML.CodeBook(0)], D.SM, D.R, D.P)
    mem = model.encode_memory(mt)
    ix = torch.tensor([keys.index(r["mem"]) if r["mem"] is not None else len(keys) for r in reqs])
    bk = [books[r["mem"]] if r["mem"] is not None else ML.CodeBook(0) for r in reqs]
    known = [(D.sf[r["mem"]]["keys"] if (r["mem"] is not None and not isinstance(r["mem"], tuple)) else set()) | {k for k in r["q"]["key"] if k} for r in reqs]
    qin, _ = ML.tree_tensors([r["q"] for r in reqs], bk, D.SM, D.R, D.P)
    _, tg = ML.tree_tensors([r["t"] for r in reqs], bk, D.SM, D.R, D.P, known)
    pool = V3.build_pool([mems[keys.index(r["mem"])] if r["mem"] is not None else None for r in reqs], [r["q"] for r in reqs], bk, D.SM, [r["t"] for r in reqs])
    outs, hl = model.ask(qin, state[ix], (mem[0][ix], mem[1][ix]), nstep, block_read)
    return outs, hl, tg, pool


@torch.no_grad()
def val_halt(model, D, items, bs=24):
    """Per-step v3 loss (mean over items), mean halting probability per step, mean halting step, and the last-step grade stats."""
    model.eval()
    ls, ps, n = 0, 0, 0
    st = {}
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        reqs = [{"mem": x["skey"], "q": x["q"], "t": x["refs"][0], "qid": x["item"]} for x in ch]
        outs, hl, tg, pool = forward_batch(model, D, reqs)
        res = [V3.tree_loss_v3(o, tg, pool) for o in outs]
        ls = ls + torch.stack([r[0] for r in res], 1).sum(0)
        ps = ps + ML.halting(hl).sum(0)
        n += len(ch)
        for k, v in res[-1][1].items():
            st[k] = st.get(k, 0.0) + float(v) * len(ch)
    ls, ps = ls / n, ps / n
    p = ps.tolist()
    # val loss = expected loss under the mean halting distribution is not the per-item value; recompute per item below is costly: use the mean-p approximation
    return ls.tolist(), p, float(sum((k + 1) * q for k, q in enumerate(p))), {k: v / n for k, v in st.items()}


@torch.no_grad()
def val_loss(model, D, items, lam, bs=24):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        reqs = [{"mem": x["skey"], "q": x["q"], "t": x["refs"][0], "qid": x["item"]} for x in ch]
        outs, hl, tg, pool = forward_batch(model, D, reqs)
        loss, _Ls, _p, _st = V3.ponder_loss_v3(outs, hl, tg, pool, lam, 0.0)
        tot += float(loss) * len(ch)
        n += len(ch)
    return tot / max(n, 1)


@torch.no_grad()
def val_coverage(model, D, dc, items, thr, tau=0.5, bs=24):
    """Decoded val: ROUGE-L overall, coverage (non-empty text), ROUGE on answered items."""
    from naomi import mind_v2 as m2
    model.eval()
    sc, ans = [], []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        reqs = [{"mem": x["skey"], "q": x["q"], "t": None, "qid": x["item"]} for x in ch]
        outs, hl, _tg, _pool = forward_batch(model, D, reqs)
        txt, _ = dc.texts(model, outs, ML.halting(hl), ch, [x["skey"] for x in ch], thr, tau, [TL.zlib_salt(r["mem"]) for r in reqs])
        for t, x in zip(txt, ch):
            sc.append(m2.rouge_multi(t, x["golds"]) if t else 0.0)
            ans.append(bool(t))
    sc, ans = np.array(sc), np.array(ans)
    return float(sc.mean()), float(ans.mean()), float(sc[ans].mean()) if ans.any() else 0.0


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
    vcf = T2.cf_items(cf_va, va, stories, D, limit=0)
    vcf = random.Random(0).sample(vcf, min(a.val_n, len(vcf)))
    if a.limit_stories:
        keep = set(sorted({x['skey'] for x in tr})[:a.limit_stories])
        tr = [x for x in tr if x['skey'] in keep]
    by_story = {}
    for x in tr:
        by_story.setdefault(x["skey"], []).append(x)
    model = TL.make_model(D.dims, a.seed, H=a.H)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    nb_ep = math.ceil(len(by_story) / a.S)
    total = a.epochs * nb_ep
    commit_ep = a.warm if a.commit is None else a.commit
    print(f"items {len(tr)} stories {len(by_story)} val_cf {len(vcf)} params {sum(p.numel() for p in model.parameters())} warm {a.warm} commit {commit_ep} lam {a.lam:.4f} beta {a.beta} p_cf {a.p_cf} tau {V3.TAU} rewards {V3.R_CORRECT}/{V3.R_NONE}/{V3.R_WRONG}", flush=True)
    t0, step, best = time.time(), 0, 1e9
    curve = open(OUT / f"{a.tag}.curve.tsv", "w")
    curve.write("epoch\tminutes\twarm\tcommit\ttrain_loss\tval_loss\tval_steps\tval_rouge\tval_cov\tval_rouge_ans\tcf_rouge\tcf_new\tcf_old\tunreach_frac\tval_s_corr\tex_matched\tex_unmatched\tper_step_loss\thalt_p\n")
    for ep in range(a.epochs):
        model.train()
        warm = V2.is_warm(ep, a.warm)
        commit = ep < commit_ep
        sk = sorted(by_story)
        rng.shuffle(sk)
        tl, nb = 0.0, 0
        for i in range(0, len(sk), a.S):
            items = [x for s in sk[i:i + a.S] for x in by_story[s]]
            reqs, ck = T2.train_reqs_v2(items, D, rng, cf_tr, 0.12, 0.06, a.p_cf)
            outs, hl, tg, pool = forward_batch(model, D, reqs, train=True, rng=rng)
            loss, _Ls, _p, _st = V3.ponder_loss_v3(outs, hl, tg, pool, a.lam, a.beta, warm, commit, a.decouple)
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
            tl += float(loss.detach())
            nb += 1
            if a.max_batches and nb >= a.max_batches:
                break
        vl = val_loss(model, D, vsub, a.lam)
        pl, pp, ms, st = val_halt(model, D, vsub)
        vr, vc, vra = val_coverage(model, D, dc, vsub, a.thr)
        cr, cn, co = T2.cf_val(model, D, dc, vcf, a.thr)
        mins = (time.time() - t0) / 60
        unreach = st["unreach"] / max(st["ref"], 1e-9)
        print(f"ep {ep} {mins:.1f}min warm={int(warm)} commit={int(commit)} train {tl / nb:.4f} val_loss {vl:.4f} steps {ms:.2f} val_rouge {vr:.4f} cov {vc:.3f} rouge_ans {vra:.4f} cf_rouge {cr:.4f} cf_new {cn:.3f} "
              f"unreach {unreach:.3f} s_corr {st['s_corr']:.3f} ex_m {st['ex_matched']:.3f} ex_u {st['ex_unmatched']:.3f} per-step {[round(x, 3) for x in pl]} halt_p {[round(x, 3) for x in pp]}", flush=True)
        curve.write(f"{ep}\t{mins:.2f}\t{int(warm)}\t{int(commit)}\t{tl / nb:.4f}\t{vl:.4f}\t{ms:.3f}\t{vr:.4f}\t{vc:.3f}\t{vra:.4f}\t{cr:.4f}\t{cn:.3f}\t{co:.3f}\t{unreach:.3f}\t{st['s_corr']:.3f}\t{st['ex_matched']:.3f}\t{st['ex_unmatched']:.3f}\t"
                    f"{json.dumps([round(x, 4) for x in pl])}\t{json.dumps([round(x, 4) for x in pp])}\n")
        curve.flush()
        TL.save(model, D.dims, OUT / f"{a.tag}_last.pt")
        if not warm and not commit and vl < best:
            best = vl
            TL.save(model, D.dims, OUT / f"{a.tag}_best.pt", {"epoch": ep, "val_rouge": vr, "val_cov": vc, "cf_rouge": cr, "val_loss": vl})
        if a.max_minutes and mins > a.max_minutes:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--tag", default="v3_s0")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--commit", type=int, default=None, help="epochs with the existence head bypassed (default = --warm)")
    ap.add_argument("--decouple", action="store_true", help="v3b: choice gradient without the ex factor, ex trained on the true expected reward")
    ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--S", type=int, default=12)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=V2.LAM_PRIOR)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--p_cf", type=float, default=0.4)
    ap.add_argument("--thr", type=float, default=0.5, help="existence threshold for the decoded val monitor only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_stories", type=int, default=0)
    ap.add_argument("--val_n", type=int, default=150)
    a = ap.parse_args()
    train(a)
