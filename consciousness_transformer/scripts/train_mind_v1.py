"""Train the Mind v1 policy from the answer alone (cross-entropy over candidates + ABSTAIN).

Target: the set of correct candidates (marginal likelihood) when the answer is among the candidates, otherwise ABSTAIN;
plus wrong-passage items (question against another story), target ABSTAIN, at a fixed share.
    python scripts/train_mind_v1.py --variant all|short --epochs 60 [--resume]
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

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
from naomi import mind_v1 as m1  # noqa: E402

OUT = CT / "runs" / "mind_v1"
SHORT_MAX = 10


def load(split):
    return pickle.load(open(OUT / f"items_{split}.pkl", "rb"))


def keep(items, variant):
    if variant == "short":
        return [r for r in items if r.get("answer_sentence_words", 10 ** 9) <= SHORT_MAX]
    return items


def examples(items, wp_share, rng):
    """(extracted, positives or None -> ABSTAIN). Wrong-passage items are added so they make wp_share of the total."""
    ex = [(r["ex"], np.flatnonzero(r["correct"])) for r in items]
    n_wp = int(round(len(ex) * wp_share / (1 - wp_share)))
    wps = [(r["ex_wp"], np.zeros(0, dtype=int)) for r in items if "ex_wp" in r]
    rng.shuffle(wps)
    return ex + wps[:n_wp]


def loss_fn(logits, ex_batch):
    ls = []
    for i, (e, pos) in enumerate(ex_batch):
        c = len(e.cand_gid)
        lg = torch.cat([logits[i, :c], logits[i, -1:]])
        lse = torch.logsumexp(lg, 0)
        tgt = lg[pos] if len(pos) else lg[-1:]
        ls.append(lse - torch.logsumexp(tgt, 0))
    return torch.stack(ls).mean()


@torch.no_grad()
def evaluate(policy, val):
    policy.eval()
    hit = ans = wp_abs = n = nw = acc = 0
    for r in val:
        k, _ = m1.decide(policy, r["ex"])
        n += 1
        ans += k >= 0
        hit += k >= 0 and bool(r["correct"][k])
        if k >= 0 and r["meaning"][k].max() > 0:       # mapped to the closest option, as the evaluation does
            acc += int(np.argmax(r["meaning"][k] * 1000.0 + r["overall"][k])) == r["answer_idx"]
        if "ex_wp" in r:
            nw += 1
            wp_abs += m1.decide(policy, r["ex_wp"])[0] < 0
    return {"acc": acc / max(n, 1), "hit": hit / max(n, 1), "answer_rate": ans / max(n, 1), "wp_abstain": wp_abs / max(nw, 1), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=["all", "short"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--wp-share", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--P", type=int, default=32)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--novec", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    name = f"policy_{args.variant}{args.tag}"
    rel_vocab = json.load(open(OUT / "rel_vocab.json"))
    train, val = keep(load("train"), args.variant), keep(load("val"), args.variant)   # short model: selection on short items only
    if args.limit:
        train, val = train[:args.limit], val[:args.limit]
    R = len(rel_vocab)
    D = train[0]["ex"].wv.shape[0]
    policy = m1.Policy(D, m1.n_cand_features(R), m1.n_q_features(R), P=args.P, H=args.H, drop=args.drop, use_vec=not args.novec)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.wd)
    start, best = 0, -1.0
    if args.resume and (OUT / f"{name}_last.pt").exists():
        ck = torch.load(OUT / f"{name}_last.pt", weights_only=False)
        policy.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"] + 1, ck["best"]
        print("resumed at epoch", start, flush=True)
    curve = OUT / f"{name}.curve.tsv"
    if start == 0:
        curve.write_text("epoch\ttrain_loss\tval_acc\tval_hit\tval_answer_rate\tval_wp_abstain\tscore\tseconds\n")
    print(f"{name}: train items {len(train)} (of which with a correct candidate "
          f"{sum(bool(r['correct'].any()) for r in train)}), val {len(val)}", flush=True)
    t0 = time.time()
    for ep in range(start, args.epochs):
        policy.train()
        exs = examples(train, args.wp_share, rng)
        rng.shuffle(exs)
        tot, nb = 0.0, 0
        for i in range(0, len(exs), args.bs):
            chunk = exs[i:i + args.bs]
            logits = policy(m1.collate([e for e, _ in chunk]))
            loss = loss_fn(logits, chunk)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        v = evaluate(policy, val)
        score = v["acc"] + 0.5 * v["wp_abstain"]
        with open(curve, "a") as f:
            f.write(f"{ep}\t{tot / nb:.4f}\t{v['acc']:.4f}\t{v['hit']:.4f}\t{v['answer_rate']:.4f}\t{v['wp_abstain']:.4f}\t{score:.4f}\t{time.time() - t0:.0f}\n")
        print(f"ep {ep} loss {tot / nb:.4f} val acc {v['acc']:.3f} hit {v['hit']:.3f} ans {v['answer_rate']:.3f} wp_abs {v['wp_abstain']:.3f} {time.time() - t0:.0f}s", flush=True)
        if score > best:
            best = score
            m1.save_policy(policy, rel_vocab, OUT / f"{name}_best.pt", {"epoch": ep, "val": v, "args": vars(args)})
        torch.save({"state": policy.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best}, OUT / f"{name}_last.pt")
    print("done", name, "best val score", round(best, 4), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
