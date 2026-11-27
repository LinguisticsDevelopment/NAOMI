"""Train the Mind v5 thinker (question completion; dev/reports/MIND_V5.md).
    python scripts/train_mind_v5.py prep                      # recompute slot-free features + imitation targets (runs/mind_v5/data_*.pkl)
    nohup nice -n 5 python scripts/train_mind_v5.py train --variant all --tag v5_all --init runs/mind_v3/thinker_v3_s1_best.pt > runs/mind_v5/v5_all.log 2>&1 &
Stage 1 imitates the filler (reference nodes aligned to memory by lemma). Stage 2 (``stage2``) trains the replace-choice ONLY by
REINFORCE on the ROUGE-L of the decoded full output sentence vs the references. Nothing here reads a slot label or the lexicon's unknown axis."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_v1 as m1  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402
from naomi import mind_v5 as m5  # noqa: E402
import build_mind_data as B  # noqa: E402

DATA = CT / "runs" / "mind_v2b"
OUT = CT / "runs" / "mind_v5"


def reads_all():
    return m2.load_reads([B.CACHE, CT / "runs/mind_v2/read_cache/reads_v2.jsonl", DATA / "read_cache" / "reads_v2.jsonl", DATA / "read_cache" / "reads_v2b.jsonl"])


def prep():
    OUT.mkdir(parents=True, exist_ok=True)
    reads = reads_all()
    ctx = m1.Ctx.get()
    stories = pickle.load(open(DATA / "stories.pkl", "rb"))
    meta = json.load(open(DATA / "items_meta.json"))
    stats = {}
    for sp in ("train", "val", "test"):
        pairs = pickle.load(open(DATA / f"pairs_{sp}.pkl", "rb"))
        out, t0, unread, unreach = [], time.time(), 0, 0
        for n, p in enumerate(pairs):
            if p["kind"] not in ("normal", "wrong"):
                continue                                            # shuffled / severed / empty controls are rebuilt in eval
            m = meta[p["item"]]
            qt = m1.tree_of(reads, m["question"])
            if qt is None:
                unread += 1
                continue
            qn = m2.flat_tree(qt)[0]
            arr = stories[p["skey"]]
            if len(qn) != len(p["q"]["parent"]):
                unread += 1
                continue
            np_ = m5.refeature(p, arr, qn, ctx)
            np_["tgt0"] = p.get("tgt")                               # the v2 target (anchors, ans_len): bands and the v2-v4 oracle only
            if p["kind"] == "normal":
                tg = m5.build_targets5(arr, p.get("tgt"))
                np_["tg"] = tg
                if tg is None:
                    unreach += 1
                at = m1.tree_of(reads, m["answer"])
                np_["ref_cnt"] = dict(m2.ref_lemmas(at)) if at is not None else {}
            out.append(np_)
            if n % 2000 == 0:
                print(sp, n, len(pairs), f"{time.time() - t0:.0f}s", flush=True)
        pickle.dump(out, open(OUT / f"data_{sp}.pkl", "wb"))
        stats[sp] = {"pairs": len(out), "unread": unread, "normal_unreachable": unreach, "kinds": dict(Counter(p["kind"] for p in out))}
        print(sp, stats[sp], flush=True)
    json.dump(stats, open(OUT / "prep_stats.json", "w"), indent=1)


def make_model(vocab, seed=0, **kw):
    torch.manual_seed(seed)
    return m5.Thinker5(m1.Ctx.get().D, len(vocab["rels"]), len(vocab["poss"]), **kw)


def save(model, vocab, path, extra=None):
    torch.save({"kind": "v5", "state": model.state_dict(), "vocab": vocab, "cfg": {"H": model.H}, "extra": extra or {}}, path)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(ck["vocab"], **ck["cfg"])
    model.load_state_dict(ck["state"], strict=False)          # stage-1 checkpoints predate the keep head (it then starts at its init)
    return model.eval(), ck["vocab"]


def load_data(splits=("train", "val")):
    stories = pickle.load(open(DATA / "stories.pkl", "rb"))
    vocab = json.load(open(DATA / "vocab.json"))
    return stories, {sp: pickle.load(open(OUT / f"data_{sp}.pkl", "rb")) for sp in splits}, vocab


@torch.no_grad()
def val_metrics(model, stories, pairs, tab, bs=48):
    """Stage-1 selection proxy (declared before any test number): mean lemma-F1 of the filler vs the reference (abstain = 0) over
    normal items; sel = 0.9 * F1 + 0.1 * wrong-story abstain rate."""
    model.eval()
    f1s, wa, wn, ans = [], 0, 0, 0
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        b = m2.collate(ch, stories, tab, with_targets=False)
        for p, r in zip(ch, model.decode(b)):
            if p["kind"] == "wrong":
                wn += 1
                wa += r["abstain"]
                continue
            if r["abstain"] or not r["pos"]:
                f1s.append(0.0)
                continue
            ans += 1
            arr = stories[p["skey"]]
            keys = []
            for pos in r["pos"]:
                k, ix = m5.source_of(pos, r["Nmax"])
                if k == "m":
                    keys.append((arr["nodes"][ix].kind, arr["nodes"][ix].lemma))
            f1s.append(m5.lemma_f1(keys, Counter(p["ref_cnt"])) if p.get("ref_cnt") else 0.0)
    n = max(len(f1s), 1)
    wab = wa / max(wn, 1)
    return {"sel": 0.9 * sum(f1s) / n + 0.1 * wab, "f1": sum(f1s) / n, "answer_rate": ans / n, "wrong_abstain": wab}


def train(args):
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    stories, data, vocab = load_data()
    tab = m2.SenseTable()
    empty = m2.Story([], {}, m2.Vocab(vocab["rels"], vocab["poss"])).arrays()
    stories["empty"] = empty
    tr = data["train"]
    short_items = {p["item"] for p in tr if p["kind"] == "normal" and p.get("tgt0") and p["tgt0"]["ans_len"] is not None and p["tgt0"]["ans_len"] <= 10}
    normal, wrong, skipped = [], [], 0
    for p in tr:
        if p["kind"] == "wrong":
            if args.variant == "all" or p["item"] in short_items:
                wrong.append(p)
        else:
            if p.get("tg") is None:
                skipped += 1
                continue
            t0_ = p["tgt0"]
            if args.variant == "short" and not (t0_["ans_len"] is not None and t0_["ans_len"] <= 10):
                continue
            normal.append(p)
    nvars = [dict(p, kind="empty", skey="empty", tg=None, sb=np.zeros((0, m2.NB), np.float32), itf=np.zeros(m2.N_ITEM, np.float32)) for p in rng.sample(normal, max(1, int(args.empty_frac * len(normal))))]
    print(f"v5 variant={args.variant}: normal {len(normal)} (unreachable skipped {skipped}), wrong-story {len(wrong)}, empty-memory {len(nvars)}", flush=True)
    for p in wrong + nvars:
        p["tg"] = None
    val = [p for p in data["val"] if p["kind"] in ("normal", "wrong")]
    if args.variant == "short":
        pass                                                         # selection is on the full val set for both variants (declared)
    model = make_model(vocab, args.seed, H=args.H, drop=args.drop)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        miss = model.load_state_dict(ck["state"], strict=False)
        print(f"warm start from {args.init}: {len(miss.missing_keys)} new tensors, {len(miss.unexpected_keys)} unexpected", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, curve, t0, ep0 = -1.0, [], time.time(), 0
    last = OUT / f"thinker_{args.tag}_last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        best, curve, ep0 = ck["best"], ck["curve"], ck["epoch"] + 1
        rng.setstate(ck["rng"])
        torch.set_rng_state(ck["trng"])
        print(f"resumed after epoch {ck['epoch']} (best sel {best:.4f})", flush=True)
    for ep in range(ep0, args.epochs):
        model.train()
        items = normal + wrong + nvars
        rng.shuffle(items)
        tl = [0.0, 0.0, 0.0, 0.0]
        nb = 0
        for i in range(0, len(items), args.bs):
            ch = items[i:i + args.bs]
            b = m2.collate(ch, stories, tab, with_targets=False)
            lp, lpar = model.tf_loss(b, ch)
            loss = lp + args.par_w * lpar
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in enumerate((float(lp), float(lpar), 0.0, 0.0)):
                tl[k] += v
            nb += 1
        v = val_metrics(model, stories, val, tab)
        curve.append((ep, tl[0] / nb, tl[1] / nb, v["sel"], v["f1"], v["answer_rate"], v["wrong_abstain"]))
        print(f"ep {ep} ptr {tl[0] / nb:.3f} par {tl[1] / nb:.3f} | val sel {v['sel']:.4f} f1 {v['f1']:.3f} "
              f"answer_rate {v['answer_rate']:.3f} wrong_abstain {v['wrong_abstain']:.3f} {time.time() - t0:.0f}s", flush=True)
        if v["sel"] > best:
            best = v["sel"]
            save(model, vocab, OUT / f"thinker_{args.tag}_best.pt", {"epoch": ep, "val": v, "args": vars(args)})
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "best": best, "curve": curve, "epoch": ep, "rng": rng.getstate(), "trng": torch.get_rng_state()}, last)
        with open(OUT / f"thinker_{args.tag}.curve.tsv", "w") as f:
            f.write("epoch\tptr_loss\tpar_loss\tval_sel\tval_f1\tanswer_rate\twrong_abstain\n")
            for c in curve:
                f.write("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in c) + "\n")
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print("time budget for this call reached; rerun with --resume", flush=True)
            break


def stage2(args):
    """REINFORCE on the replace-choice only. Reward = ROUGE-L (max over references) of the decoded (order-only decoder) FULL output
    sentence = the question tree with the sampled node replaced by the (greedy, fixed) filler. Abstained items get no gradient."""
    from naomi.tree import tree_from_dict  # noqa: F401
    rng = random.Random(args.seed)
    stories, data, vocab = load_data()
    tab = m2.SenseTable()
    meta = json.load(open(DATA / "items_meta.json"))
    reads = reads_all()
    dec = m2.Decoder(str(CT / "runs/naomi_orderer/large_best.pt"))
    model, _ = load_model(args.init)
    for n_, p_ in model.named_parameters():
        p_.requires_grad = n_.startswith(("rep.", "keep_q."))
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr, weight_decay=args.wd)
    qcache = {}

    def qinfo(i):
        if i not in qcache:
            qt = m1.tree_of(reads, meta[i]["question"])
            qcache[i] = (qt, m2.flat_tree(qt)[0])
        return qcache[i]

    def outputs(ch, res, js, keeps):
        trees, where = [], []
        for n, (p, r) in enumerate(zip(ch, res)):
            if r["abstain"] or not r["pos"]:
                continue
            qt, qn = qinfo(p["item"])
            arr = stories[p["skey"]]
            tr, _ = m5.filler_tree(arr, qn, [m5.source_of(x, r["Nmax"]) for x in r["pos"]], r["par"])
            for k, j in enumerate(js[n]):
                trees.append(m5.assemble(qt, qn, j, tr, keeps[n][k]))
                where.append((n, k))
        return trees, where

    def val_reward(pairs):
        tot = []
        for i in range(0, len(pairs), 48):
            ch = pairs[i:i + 48]
            b = m2.collate(ch, stories, tab, with_targets=False)
            res = model.decode(b)
            trees, where = outputs(ch, res, [[r["j"]] for r in res], [[r["keep"]] for r in res])
            sc = {w: 0.0 for w in range(len(ch))}
            for (n, _k), t in zip(where, dec.texts(trees)):
                sc[n] = m2.rouge_multi(t, meta[ch[n]["item"]]["golds"])
            tot += list(sc.values())
        return float(np.mean(tot))

    val = [p for p in data["val"] if p["kind"] == "normal"]
    val = rng.sample(val, min(args.n_val2, len(val)))
    train_items = [p for p in data["train"] if p["kind"] == "normal"]
    if args.variant == "short":
        train_items = [p for p in train_items if p["tgt0"] and p["tgt0"]["ans_len"] is not None and p["tgt0"]["ans_len"] <= 10]
    best = val_reward(val)
    print(f"stage2 start: val full-sentence ROUGE-L {best:.4f}", flush=True)
    save(model, vocab, OUT / f"thinker_{args.tag}_best.pt", {"epoch": -1, "val": best})
    curve, t0 = [(-1, 0.0, best)], time.time()
    for ep in range(args.epochs):
        items = rng.sample(train_items, min(args.items_per_epoch, len(train_items)))
        rs = []
        for i in range(0, len(items), args.bs):
            ch = items[i:i + args.bs]
            b = m2.collate(ch, stories, tab, with_targets=False)
            res = model.decode(b)
            with torch.enable_grad():
                rl, kl = model.replace_logits_for(b, res)
            lp = torch.log_softmax(rl, 1)
            js = [torch.multinomial(lp[n].exp().detach(), args.K, replacement=True).tolist() for n in range(len(ch))]
            keeps, lkeep = [], torch.zeros(len(ch), args.K)
            for n, p in enumerate(ch):
                qn_ = len(p["q"]["parent"])
                pk = torch.sigmoid(kl[n, :qn_]).detach()
                row = []
                for k in range(args.K):
                    kk = torch.bernoulli(pk).bool()
                    row.append(kk.tolist())
                    rem = m5.removed_by(p["q"]["parent"], js[n][k])
                    live = torch.tensor([i not in rem for i in range(qn_)])
                    ll = torch.where(kk, F.logsigmoid(kl[n, :qn_]), F.logsigmoid(-kl[n, :qn_]))
                    lkeep[n, k] = (ll * live.float()).sum()
                keeps.append(row)
            trees, where = outputs(ch, res, js, keeps)
            rew = np.zeros((len(ch), args.K), dtype=np.float32)
            for (n, k), t in zip(where, dec.texts(trees)):
                rew[n, k] = m2.rouge_multi(t, meta[ch[n]["item"]]["golds"])
            act = np.array([not (r["abstain"] or not r["pos"]) for r in res], dtype=np.float32)
            base = (rew.sum(1, keepdims=True) - rew) / (args.K - 1)
            adv = torch.from_numpy(rew - base) * torch.from_numpy(act)[:, None]
            logp = torch.stack([lp[n, js[n]] for n in range(len(ch))]) + lkeep
            loss = -(adv * logp).sum() / max(float(act.sum()), 1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rs.append(float(rew[act > 0].mean()) if act.sum() else 0.0)
        v = val_reward(val)
        curve.append((ep, float(np.mean(rs)), v))
        print(f"stage2 ep {ep} train_reward {np.mean(rs):.4f} val_full_rouge {v:.4f} {time.time() - t0:.0f}s", flush=True)
        if v > best:
            best = v
            save(model, vocab, OUT / f"thinker_{args.tag}_best.pt", {"epoch": ep, "val": v})
        with open(OUT / f"thinker_{args.tag}.curve.tsv", "w") as f:
            f.write("epoch\ttrain_reward\tval_full_rouge\n" + "".join(f"{c[0]}\t{c[1]:.4f}\t{c[2]:.4f}\n" for c in curve))
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "stage2"])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n_val2", type=int, default=300)
    ap.add_argument("--items_per_epoch", type=int, default=2000)
    ap.add_argument("--variant", choices=["all", "short"], default="all")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--par_w", type=float, default=0.5)
    ap.add_argument("--rep_w", type=float, default=1.0)
    ap.add_argument("--empty_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--tag", default="v5_all")
    ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.cmd == "prep":
        prep()
    elif args.cmd == "train":
        train(args)
    else:
        stage2(args)


if __name__ == "__main__":
    main()
