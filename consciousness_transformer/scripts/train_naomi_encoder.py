"""Train the stage-A NAOMI encoder (text -> head + relation per token) on gold v7, "unknown, not wrong".

  nohup nice -n 5 python scripts/train_naomi_encoder.py --tag main --max-minutes 150 > runs/naomi_encoder_main.log 2>&1 &

Splits are by SOURCE (a whole FairytaleQA story, or a contiguous chunk of a Gutenberg book), fixed seed,
~90/5/5 train/dev/test by sentence count; written to runs/naomi_encoder/splits.json. The frozen holdout
files are never trained on. The best checkpoint on dev (supervised head+relation accuracy) is kept.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naomi.encoder import (EncoderNet, FeatureBuilder, collate, head_mask, infos_from_record,  # noqa: E402
                           supervision_from_tree)
from naomi.reader import Reader, _sense_space  # noqa: E402
from naomi.tree import tree_from_dict  # noqa: E402
from naomi.types import REL_ROOT  # noqa: E402

OUT = ROOT / "runs" / "naomi_encoder"
GOLD = ROOT / "runs" / "gold_v7.jsonl.gz"
CHUNK = 400          # sentences per contiguous chunk of a Gutenberg book (a "source" for splitting)
SUB_MIN = 20         # a structural subtype enters the vocabulary with at least this many training tokens


def load_jsonl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def source_keys(records):
    """Splitting unit per record: whole story for K-12, contiguous chunks for books."""
    seen, keys = Counter(), []
    for r in records:
        s = r["source"]
        if s.startswith("real_gutenberg"):
            keys.append(f"{s}#{seen[s] // CHUNK}")
            seen[s] += 1
        else:
            keys.append(s)
    return keys


def make_splits(keys, seed=0, dev_frac=0.05, test_frac=0.05):
    cnt = Counter(keys)
    total = sum(cnt.values())
    order = sorted(cnt)
    random.Random(seed).shuffle(order)
    out, acc = {}, {"dev": 0, "test": 0}
    want = {"dev": dev_frac * total, "test": test_frac * total}
    for k in order:
        for name in ("dev", "test"):
            if acc[name] < want[name] and cnt[k] <= 0.02 * total:
                out[k], acc[name] = name, acc[name] + cnt[k]
                break
        else:
            out[k] = "train"
    return out


def prep(records, builder, reader, vocab_from=None):
    """Records -> list of dicts with Feats + targets (heads/rels filled in later once rel ids are known)."""
    items, skipped = [], 0
    for r in records:
        infos = infos_from_record(reader, r)
        if infos is None:
            skipped += 1
            continue
        sv = supervision_from_tree(tree_from_dict(r["tree"]), len(infos))
        items.append({"id": r["id"], "source": r["source"], "infos": infos, "sv": sv})
    return items, skipped


def attach_feats(items, builder):
    for it in items:
        it["feats"] = builder.build(it["infos"])


def targets(items, rel_ids):
    for it in items:
        sv = it["sv"]
        it["head"] = np.array(sv.head, dtype=np.int64)
        it["relid"] = np.array([rel_ids.get(x, 0) if x else 0 for x in sv.rel], dtype=np.int64)
        it["sup"] = np.array(sv.sup, dtype=bool)


def batches(items, max_tokens, max_sents, rng=None):
    idx = sorted(range(len(items)), key=lambda i: len(items[i]["feats"].pos))
    out, cur, mx = [], [], 0
    for i in idx:
        n = len(items[i]["feats"].pos)
        if cur and (max(mx, n) * (len(cur) + 1) > max_tokens or len(cur) >= max_sents):
            out.append(cur)
            cur, mx = [], 0
        cur.append(i)
        mx = max(mx, n)
    if cur:
        out.append(cur)
    if rng:
        rng.shuffle(out)
    return out


def make_batch(items, ids, table, n_sub):
    sel = [items[i] for i in ids]
    b = collate([s["feats"] for s in sel], table, n_sub)
    B, T = b["pos"].shape
    head = torch.full((B, T), -1, dtype=torch.long)
    rel = torch.zeros((B, T), dtype=torch.long)
    sup = torch.zeros((B, T), dtype=torch.bool)
    for k, s in enumerate(sel):
        n = len(s["head"])
        head[k, :n], rel[k, :n], sup[k, :n] = (torch.from_numpy(s["head"]), torch.from_numpy(s["relid"]),
                                               torch.from_numpy(s["sup"]))
    b.update(head=head, rel=rel, sup=sup)
    return b


def forward_loss(net, b):
    H = net.encode(b["pos"], b["subs"], b["meaning"], b["name"], b["hashed"], b["lengths"])
    s = net.arc_scores(H).masked_fill(~head_mask(b["node"]), -1e9)
    sup = b["sup"]
    if not sup.any():
        return H.sum() * 0, 0, 0, 0
    bi, di = sup.nonzero(as_tuple=True)
    gh = b["head"][bi, di]
    la = F.cross_entropy(s[bi, di], gh)
    lg = net.rel_logits(H, bi, di, gh)
    lr = F.cross_entropy(lg, b["rel"][bi, di])
    ok_h = (s[bi, di].argmax(-1) == gh)
    ok_l = ok_h & (lg.argmax(-1) == b["rel"][bi, di])
    return la + lr, int(ok_h.sum()), int(ok_l.sum()), len(bi)


@torch.no_grad()
def dev_eval(net, items, table, n_sub):
    net.eval()
    tot = h = l = 0
    loss = 0.0
    for ids in batches(items, 4000, 64):
        b = make_batch(items, ids, table, n_sub)
        ls, ch, cl, n = forward_loss(net, b)
        loss += float(ls) * n
        h, l, tot = h + ch, l + cl, tot + n
    return {"loss": loss / max(tot, 1), "uas": h / max(tot, 1), "las": l / max(tot, 1), "n": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--max-minutes", type=float, default=150)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hash-buckets", type=int, default=0, help="optional word-identity ablation (0 = off)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-tokens", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="pilot: use only the first N training sentences")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cache = OUT / "prep_cache.pkl"
    if cache.exists() and not a.limit:
        d = pickle.load(open(cache, "rb"))
        items, keys, skipped = d["items"], d["keys"], d["skipped"]
        print(f"loaded prep cache {cache.name}: {len(items)} items", flush=True)
    else:
        recs = load_jsonl(a.gold)
        keys_all = source_keys(recs)
        if a.limit:
            recs, keys_all = recs[::max(1, len(recs) // a.limit)][:a.limit], None
        print(f"{len(recs)} records loaded in {time.time() - t0:.0f}s", flush=True)
        rd = Reader(use_encoder=False, cache_path=None)
        items, skipped = prep(recs, None, rd)
        ids = {r["id"]: k for r, k in zip(recs, keys_all or source_keys(recs))}
        keys = [ids[it["id"]] for it in items]
        if not a.limit:
            pickle.dump({"items": items, "keys": keys, "skipped": skipped}, open(cache, "wb"))
        print(f"prepped {len(items)} items (skipped {skipped} token mismatches) in {time.time() - t0:.0f}s", flush=True)

    splits = make_splits(keys, seed=a.seed) if not a.limit else make_splits(keys, seed=a.seed, dev_frac=0.1, test_frac=0.1)
    for it, k in zip(items, keys):
        it["split"] = splits[k]
    if not a.limit:
        json.dump({"seed": a.seed, "chunk": CHUNK, "split_of_source": splits,
                   "counts": {s: sum(1 for it in items if it["split"] == s) for s in ("train", "dev", "test")}},
                  open(OUT / "splits.json", "w"), indent=1, sort_keys=True)
    tr = [it for it in items if it["split"] == "train"]
    dv = [it for it in items if it["split"] == "dev"]
    te = [it for it in items if it["split"] == "test"]
    print(f"sentences train/dev/test: {len(tr)}/{len(dv)}/{len(te)}", flush=True)
    for name, grp in (("train", tr), ("dev", dv), ("test", te)):
        nt = sum(int(sum(t.is_node for t in it["infos"])) for it in grp)
        ns = sum(sum(it["sv"].sup) for it in grp)
        nu = sum(sum(it["sv"].unknown) for it in grp)
        print(f"  {name}: node tokens {nt}, supervised {ns} ({100 * ns / max(nt, 1):.1f}%), unknown {nu}", flush=True)

    # vocabularies from TRAIN only
    pos_cnt = Counter(t.pos for it in tr for t in it["infos"])
    pos_vocab = {None: 0, **{p: i + 1 for i, p in enumerate(sorted(pos_cnt))}}
    sub_cnt = Counter(s for it in tr for t in it["infos"] for s in t.subtypes)
    sub_vocab = {s: i for i, s in enumerate(sorted(s for s, c in sub_cnt.items() if c >= SUB_MIN))}
    rel_cnt = Counter(x for it in tr for x, s in zip(it["sv"].rel, it["sv"].sup) if s)
    rel_labels = sorted(rel_cnt)
    assert REL_ROOT in rel_labels
    rel_ids = {r: i for i, r in enumerate(rel_labels)}
    print(f"pos {len(pos_vocab)}, subtypes {len(sub_vocab)}, relations {rel_labels}", flush=True)

    usvs = _sense_space()
    builder = FeatureBuilder(usvs, pos_vocab, sub_vocab, a.hash_buckets)
    attach_feats(items, builder)
    targets(items, rel_ids)
    table = builder.table
    print(f"meaning table {table.shape}", flush=True)
    cfg = dict(d_in=192, hidden=a.hidden, layers=a.layers, hash_buckets=a.hash_buckets, d_arc=128, d_rel=64,
               dropout=0.3)
    net = EncoderNet(len(pos_vocab), len(sub_vocab), len(rel_labels), len(usvs.axes), **cfg)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"model params {n_params}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    n_sub = len(sub_vocab)
    rng = random.Random(a.seed)
    best, bad = -1.0, 0
    ckpt_best = OUT / f"{a.tag}_best.pt"
    curve = open(OUT / f"{a.tag}_curve.tsv", "w")
    curve.write("epoch\tminutes\ttrain_loss\ttrain_uas\ttrain_las\tdev_loss\tdev_uas\tdev_las\tlr\n")
    train_start = time.time()

    def save(path, epoch, dm):
        torch.save({"config": cfg, "state": net.state_dict(), "pos_vocab": pos_vocab, "sub_vocab": sub_vocab,
                    "rel_labels": rel_labels,
                    "meta": {"epoch": epoch, "dev": dm, "args": vars(a), "n_params": n_params,
                             "train_sentences": len(tr)}}, path)

    for ep in range(1, a.epochs + 1):
        net.train()
        te0 = time.time()
        tl = th = tlc = tn = 0
        bl = batches(tr, a.batch_tokens, 64, rng)
        for k, ids in enumerate(bl):
            b = make_batch(tr, ids, table, n_sub)
            loss, ch, cl, n = forward_loss(net, b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tl, th, tlc, tn = tl + float(loss) * n, th + ch, tlc + cl, tn + n
            if k % 100 == 0:
                print(f"  ep {ep} batch {k}/{len(bl)} loss {float(loss):.3f} {time.time() - te0:.0f}s", flush=True)
            if (time.time() - train_start) / 60 > a.max_minutes:
                break
        dm = dev_eval(net, dv, table, n_sub)
        sched.step(-dm["las"])
        mins = (time.time() - train_start) / 60
        curve.write(f"{ep}\t{mins:.1f}\t{tl / max(tn, 1):.4f}\t{th / max(tn, 1):.4f}\t{tlc / max(tn, 1):.4f}\t"
                    f"{dm['loss']:.4f}\t{dm['uas']:.4f}\t{dm['las']:.4f}\t{opt.param_groups[0]['lr']:.5f}\n")
        curve.flush()
        print(f"epoch {ep} {time.time() - te0:.0f}s (total {mins:.1f} min) train loss {tl / max(tn, 1):.3f} "
              f"uas {th / max(tn, 1):.3f} | dev uas {dm['uas']:.4f} las {dm['las']:.4f} loss {dm['loss']:.3f}", flush=True)
        save(OUT / f"{a.tag}_last.pt", ep, dm)
        if dm["las"] > best:
            best, bad = dm["las"], 0
            save(ckpt_best, ep, dm)
            print(f"  new best dev LAS {best:.4f} -> {ckpt_best.name}", flush=True)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop", flush=True)
                break
        if mins > a.max_minutes:
            print("time budget reached", flush=True)
            break
    print(f"done in {(time.time() - t0) / 60:.1f} min; best dev LAS {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
