"""Train the Mind v5 thinker (question completion; dev/reports/MIND_V5.md).
    python scripts/train_mind_v5.py prep                      # recompute slot-free features + imitation targets (runs/mind_v5/data_*.pkl)
    nohup nice -n 5 python scripts/train_mind_v5.py train --variant all --tag v5_all --init runs/mind_v3/thinker_v3_s1_best.pt > runs/mind_v5/v5_all.log 2>&1 &
The filler is imitated (reference nodes aligned to memory by lemma); the replace-choice is trained ONLY by the expected frame
reward over placements (see naomi.mind_v5.frame_rewards). Nothing here reads a slot label or the lexicon's unknown axis."""
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
                tg = m5.build_targets5(arr, p.get("tgt"), qn, p["q"]["parent"])
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
    model.load_state_dict(ck["state"])
    return model.eval(), ck["vocab"]


def load_data(splits=("train", "val")):
    stories = pickle.load(open(DATA / "stories.pkl", "rb"))
    vocab = json.load(open(DATA / "vocab.json"))
    return stories, {sp: pickle.load(open(OUT / f"data_{sp}.pkl", "rb")) for sp in splits}, vocab


@torch.no_grad()
def val_metrics(model, stories, pairs, tab, bs=48):
    """Cheap proxy used for checkpoint selection (declared before any test number): per normal item
    0.5 * lemma-F1(filler vs reference, abstain = 0) + 0.5 * frame share of the SUPPORTED question nodes kept at the chosen node
    (the frame reward the replace-choice is trained on); plus the wrong-story abstain rate. sel = 0.9 * normal + 0.1 * wrong_abstain."""
    model.eval()
    f1s, frs, wa, wn, ans = [], [], 0, 0, 0
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        b = m2.collate(ch, stories, tab, with_targets=False)
        res = model.decode(b)
        for p, r in zip(ch, res):
            if p["kind"] == "wrong":
                wn += 1
                wa += r["abstain"]
                continue
            if r["abstain"] or not r["pos"]:
                f1s.append(0.0)
                frs.append(0.0)
                continue
            ans += 1
            arr = stories[p["skey"]]
            keys = []
            for pos in r["pos"]:
                k, ix = m5.source_of(pos, r["Nmax"])
                keys.append((arr["nodes"][ix].kind, arr["nodes"][ix].lemma) if k == "m" else None)
            keys = [k for k in keys if k is not None]
            f1 = m5.lemma_f1(keys, Counter(p["ref_cnt"])) if p.get("ref_cnt") else 0.0
            tg = p.get("tg")
            fr = float(tg["rew"][r["j"]]) if (tg and tg["rew"] is not None) else 1.0
            f1s.append(f1)
            frs.append(fr)
    n = max(len(f1s), 1)
    normal = 0.5 * sum(f1s) / n + 0.5 * sum(frs) / n
    wab = wa / max(wn, 1)
    return {"sel": 0.9 * normal + 0.1 * wab, "f1": sum(f1s) / n, "frame": sum(frs) / n, "answer_rate": ans / n, "wrong_abstain": wab}


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
            lp, lpar, lrep, st = model.tf_loss(b, ch)
            loss = lp + args.par_w * lpar + args.rep_w * lrep
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in enumerate((float(lp), float(lpar), float(lrep), st["exp_frame"])):
                tl[k] += v
            nb += 1
        v = val_metrics(model, stories, val, tab)
        curve.append((ep, tl[0] / nb, tl[1] / nb, tl[2] / nb, tl[3] / nb, v["sel"], v["f1"], v["frame"], v["answer_rate"], v["wrong_abstain"]))
        print(f"ep {ep} ptr {tl[0] / nb:.3f} par {tl[1] / nb:.3f} rep {tl[2] / nb:.3f} train_exp_frame {tl[3] / nb:.3f} | val sel {v['sel']:.4f} f1 {v['f1']:.3f} frame {v['frame']:.3f} "
              f"answer_rate {v['answer_rate']:.3f} wrong_abstain {v['wrong_abstain']:.3f} {time.time() - t0:.0f}s", flush=True)
        if v["sel"] > best:
            best = v["sel"]
            save(model, vocab, OUT / f"thinker_{args.tag}_best.pt", {"epoch": ep, "val": v, "args": vars(args)})
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "best": best, "curve": curve, "epoch": ep, "rng": rng.getstate(), "trng": torch.get_rng_state()}, last)
        with open(OUT / f"thinker_{args.tag}.curve.tsv", "w") as f:
            f.write("epoch\tptr_loss\tpar_loss\trep_loss\ttrain_exp_frame\tval_sel\tval_f1\tval_frame\tanswer_rate\twrong_abstain\n")
            for c in curve:
                f.write("\t".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in c) + "\n")
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print("time budget for this call reached; rerun with --resume", flush=True)
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train"])
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
    else:
        train(args)


if __name__ == "__main__":
    main()
