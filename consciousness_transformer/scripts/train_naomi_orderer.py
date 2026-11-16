"""Train / evaluate the NAOMI decoder v1 (word order from a tree) on gold v7 TRAIN sources (same split as the encoder).

  nohup nice -n 5 python scripts/train_naomi_orderer.py --tag main --max-minutes 60 > runs/naomi_orderer_main.log 2>&1 &
  python scripts/train_naomi_orderer.py --eval runs/naomi_orderer/main_best.pt --out runs/naomi_orderer/eval_orderer.json

Training data: every fully connected tree plus each internally connected piece (>= 2 tokens) of the fragmented
sentences, treated as its own small tree; tokens keep their original relative order as the target. Punctuation is
not in the tree. Position is never an input (see naomi.orderer).
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from naomi import orderer as O  # noqa: E402
from naomi.encoder import FeatureBuilder  # noqa: E402
from naomi.reader import _sense_space  # noqa: E402
from naomi.types import REL_ROOT  # noqa: E402

ENC_DIR = ROOT / "runs" / "naomi_encoder"
OUT = ROOT / "runs" / "naomi_orderer"


# ----------------------------------------------------------------------------- pieces
def pieces_of(it, min_size=2):
    """Internally connected pieces of one prepared item: [(token positions in original order, parent (local), rels)]."""
    sv, infos = it["sv"], it["infos"]
    n = len(infos)
    par = [-1] * n
    for i in range(n):
        if infos[i].is_node and sv.sup[i] and sv.head[i] > 0 and infos[sv.head[i] - 1].is_node:
            par[i] = sv.head[i] - 1

    def top(i):
        while par[i] >= 0:
            i = par[i]
        return i

    comps = {}
    for i in range(n):
        if infos[i].is_node:
            comps.setdefault(top(i), []).append(i)
    out = []
    for t, members in sorted(comps.items()):
        if len(members) < min_size:
            continue
        loc = {g: k for k, g in enumerate(members)}
        out.append((members, [(-1 if par[g] < 0 else loc[par[g]]) for g in members],
                    [(REL_ROOT if par[g] < 0 else sv.rel[g]) for g in members]))
    return out


def tins_from_items(items, builder, rel_ids, connected_only=False, seed=0):
    res = []
    for k, it in enumerate(items):
        if connected_only and not it["sv"].connected:
            continue
        for members, par, rels in pieces_of(it):
            res.append(O.make_tree_in([it["infos"][g] for g in members], rels, par, builder, rel_ids,
                                      true_pos=members, seed=seed + k))
    return res


def load_items():
    cache = pickle.load(open(ENC_DIR / "prep_cache.pkl", "rb"))
    splits = json.load(open(ENC_DIR / "splits.json"))["split_of_source"]
    items, keys = cache["items"], cache["keys"]
    by = {s: [it for it, k in zip(items, keys) if splits[k] == s] for s in ("train", "dev", "test")}
    return by


def make_builder():
    ck = torch.load(ENC_DIR / "main_best.pt", map_location="cpu", weights_only=False)
    usvs = _sense_space()
    rel_labels = list(ck["rel_labels"]) + [O.UNK_REL]
    return FeatureBuilder(usvs, ck["pos_vocab"], ck["sub_vocab"], 0), rel_labels, usvs


# ----------------------------------------------------------------------------- batches / loss
def batches(tins, max_nodes, rng):
    idx = list(range(len(tins)))
    rng.shuffle(idx)
    out, cur, n = [], [], 0
    for i in idx:
        if cur and n + tins[i].n > max_nodes:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += tins[i].n
    if cur:
        out.append(cur)
    return out


def labels(sel):
    ys = []
    for t in sel:
        ys.append((t.true_pos[t.pa] < t.true_pos[t.pb]).astype(np.float32))
    return torch.from_numpy(np.concatenate(ys))


@torch.no_grad()
def dev_eval(net, tins, builder):
    net.eval()
    o = O.Orderer(net, builder, [], {})
    o.rel_ids = {}
    m = O.Metrics()
    for t, s in zip(tins, o.scores(tins)):
        m.add(t, O.sort_groups_by_scores(t, s))
    net.train()
    return m.out()


def train(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    by = load_items()
    if a.limit:
        by = {k: v[:a.limit if k == "train" else max(a.limit // 5, 50)] for k, v in by.items()}
    builder, rel_labels, usvs = make_builder()
    rel_ids = {r: i for i, r in enumerate(rel_labels)}
    tr = tins_from_items(by["train"], builder, rel_ids)
    dv = tins_from_items(by["dev"], builder, rel_ids)
    print(f"pieces train {len(tr)} ({sum(t.n for t in tr)} nodes, {sum(len(t.pa) for t in tr)} pairs), dev {len(dv)} "
          f"in {time.time() - t0:.0f}s", flush=True)
    cfg = dict(d=a.d, rounds=a.rounds, d_pair=192, dropout=0.2)
    net = O.OrdererNet(len(builder.pos_vocab), len(builder.sub_vocab), len(rel_labels), len(usvs.axes), **cfg)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params {n_params}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    rng = random.Random(a.seed)
    table, n_sub = builder.table, len(builder.sub_vocab)
    curve = open(OUT / f"{a.tag}_curve.tsv", "w")
    curve.write("epoch\tminutes\ttrain_loss\tdev_pair_acc\tdev_heads_right\tdev_exact\tlr\n")
    best, bad, start = -1.0, 0, time.time()

    def save(path, ep, dm):
        torch.save({"config": cfg, "state": net.state_dict(), "pos_vocab": builder.pos_vocab,
                    "sub_vocab": builder.sub_vocab, "rel_labels": rel_labels,
                    "meta": {"epoch": ep, "dev": dm, "n_params": n_params, "args": vars(a)}}, path)

    for ep in range(1, a.epochs + 1):
        net.train()
        te, tl, tn = time.time(), 0.0, 0
        bl = batches(tr, a.batch_nodes, rng)
        for k, ids in enumerate(bl):
            sel = [tr[i] for i in ids]
            if not sum(len(t.pa) for t in sel):
                continue
            b = O.collate_trees(sel, table, n_sub)
            loss = F.binary_cross_entropy_with_logits(net(b), labels(sel))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tl, tn = tl + float(loss), tn + 1
            if k % 100 == 0:
                print(f"  ep {ep} batch {k}/{len(bl)} loss {float(loss):.3f} {time.time() - te:.0f}s", flush=True)
            if (time.time() - start) / 60 > a.max_minutes:
                break
        dm = dev_eval(net, dv, builder)
        sched.step(-dm["heads_fully_right"])
        mins = (time.time() - start) / 60
        curve.write(f"{ep}\t{mins:.1f}\t{tl / max(tn, 1):.4f}\t{dm['pairwise_acc']}\t{dm['heads_fully_right']}\t"
                    f"{dm['exact_order']}\t{opt.param_groups[0]['lr']:.5f}\n")
        curve.flush()
        print(f"epoch {ep} {time.time() - te:.0f}s (total {mins:.1f} min) loss {tl / max(tn, 1):.4f} | dev pair "
              f"{dm['pairwise_acc']} heads-right {dm['heads_fully_right']} exact {dm['exact_order']}", flush=True)
        save(OUT / f"{a.tag}_last.pt", ep, dm)
        if dm["heads_fully_right"] > best:
            best, bad = dm["heads_fully_right"], 0
            save(OUT / f"{a.tag}_best.pt", ep, dm)
            print(f"  new best {best} -> {a.tag}_best.pt", flush=True)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop", flush=True)
                break
        if mins > a.max_minutes:
            print("time budget reached", flush=True)
            break
    print(f"done in {(time.time() - t0) / 60:.1f} min; best dev heads-right {best}", flush=True)


# ----------------------------------------------------------------------------- evaluation
def ceiling(tins):
    res = {}
    for name, grp in (("all", tins),) + tuple((b, [t for t in tins if O.band_of(t.n_tokens) == b])
                                             for b in ("<=10", "11-20", "21-35", "36+")):
        if grp:
            res[name] = {"trees": len(grp), "contiguous_share": round(sum(O.contiguous(t) for t in grp) / len(grp), 4)}
    return res


def evaluate_all(a):
    from eval_naomi_encoder import items_from_records  # noqa: E402
    from naomi.reader import Reader  # noqa: E402
    from train_naomi_encoder import load_jsonl  # noqa: E402
    t0 = time.time()
    orr = O.Orderer.load(a.eval)
    builder, rel_ids = orr.builder, orr.rel_ids
    by = load_items()
    tr = tins_from_items(by["train"], builder, rel_ids)
    base = O.LookupBaseline([], [])
    base.fit(tr)
    sets = {"test_split_connected": tins_from_items(by["test"], builder, rel_ids, connected_only=True),
            "test_split_pieces": tins_from_items(by["test"], builder, rel_ids),
            "train_pieces_sample": random.Random(0).sample(tr, 2000)}
    rd = Reader(use_encoder=False, cache_path=None)
    for name, p in (("frozen_holdout_connected", "holdout_gold_v7.jsonl"), ("frozen_holdout_dev_connected", "holdout_dev_gold_v7.jsonl")):
        sets[name] = tins_from_items(items_from_records(load_jsonl(ROOT / "runs" / p), rd), builder, rel_ids, connected_only=True)
    res = {"ckpt": a.eval, "meta": torch.load(a.eval, map_location="cpu", weights_only=False)["meta"]}
    for name, tins in sets.items():
        res[name] = {"n_trees": len(tins), "ceiling_contiguous": ceiling(tins),
                     "decoder": O.evaluate(tins, orr.group_orders),
                     "cheat_lookup_baseline": O.evaluate(tins, lambda ts: [base.group_order(t) for t in ts]),
                     "truth_blocks_ceiling_check": O.evaluate(tins, lambda ts: [O.true_group_order(t) for t in ts])}
        d, c, b = res[name]["decoder"]["all"], res[name]["ceiling_contiguous"]["all"], res[name]["cheat_lookup_baseline"]["all"]
        print(f"{name}: n={len(tins)} ceiling {c['contiguous_share']} | decoder exact {d['exact_order']} heads {d['heads_fully_right']} "
              f"pair {d['pairwise_acc']} kt {d['kendall_distance']} | baseline exact {b['exact_order']} heads {b['heads_fully_right']} "
              f"pair {b['pairwise_acc']} kt {b['kendall_distance']}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--eval", default=None)
    ap.add_argument("--out", default=str(OUT / "eval_orderer.json"))
    ap.add_argument("--max-minutes", type=float, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-nodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.eval:
        torch.set_num_threads(a.threads)
        evaluate_all(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
