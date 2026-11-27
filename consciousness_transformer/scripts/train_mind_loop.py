"""Train the looped thinker (dev/reports/MIND_LOOP.md).
    python scripts/train_mind_loop.py prep
    nice -n 5 python scripts/train_mind_loop.py train --tag loop_all --max_minutes 40 > runs/mind_loop/loop_all.log
Only the output tree at the halting step is trained (PonderNet expectation over halting steps vs the reference answer read into a tree).
Nothing supervises memory writes, intermediate steps or halting time. Wrong-story / empty-memory items train the EMPTY tree as the reference
(still the output tree at the halt vs a reference)."""
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
from naomi import mind_loop as ML  # noqa: E402
from naomi import mind_v1 as m1  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402

V5 = CT / "runs" / "mind_v5"
V2B = CT / "runs" / "mind_v2b"
OUT = CT / "runs" / "mind_loop"


# ----------------------------------------------------------------------------- data
def prep():
    import build_mind_data as B
    import train_mind_v5 as T5
    OUT.mkdir(parents=True, exist_ok=True)
    reads = T5.reads_all()
    stories = pickle.load(open(V2B / "stories.pkl", "rb"))
    meta = json.load(open(V2B / "items_meta.json"))
    vj = json.load(open(V2B / "vocab.json"))
    vocab = m2.Vocab(vj["rels"], vj["poss"])
    table = m2.SenseTable()
    sf = {}
    needed = set()
    data = {sp: pickle.load(open(V5 / f"data_{sp}.pkl", "rb")) for sp in ("train", "val", "test")}
    for sp, d in data.items():
        needed |= {p["skey"] for p in d}
    for sk in sorted(needed):
        mm = ML.story_memory(stories[sk], table)
        sf[sk] = {"mem": mm, "keys": {k for k in mm["key"] if k}}
    for sp, d in data.items():
        items, wrong = {}, {}
        for p in d:
            if p["kind"] == "wrong":
                wrong[p["item"]] = p["skey"]
        for p in d:
            if p["kind"] != "normal":
                continue
            m = meta[p["item"]]
            qt = m1.tree_of(reads, m["question"])
            qn, qpar, _ = m2.flat_tree(qt)
            qf = ML.form_of_nodes(qn, qpar, [vocab.rel_id.get(n.rel, 0) for n in qn], [vocab.pos_id.get(n.pos, 0) for n in qn], table)
            refs = []
            for g in [m["answer"]] + [x for x in m["golds"] if x != m["answer"]]:
                t = m1.tree_of(reads, g)
                if t is None:
                    continue
                rn, rpar, _ = m2.flat_tree(t)
                if rn:
                    refs.append(ML.form_of_nodes(rn, rpar, [vocab.rel_id.get(n.rel, 0) for n in rn], [vocab.pos_id.get(n.pos, 0) for n in rn], table))
            t0 = p["tgt0"]
            items[p["item"]] = {"item": p["item"], "skey": p["skey"], "story": m["story"], "q": qf, "qnodes": list(qn), "refs": refs,
                                "golds": m["golds"], "question": m["question"], "band": m2.band_of(t0["ans_len"]) if t0["ans_len"] is not None else "none",
                                "ans_len": t0["ans_len"], "wrong_skey": wrong.get(p["item"])}
        pickle.dump(list(items.values()), open(OUT / f"items_{sp}.pkl", "wb"))
        print(sp, len(items), "items", sum(1 for x in items.values() if not x["refs"]), "without a reference tree", flush=True)
    pickle.dump(sf, open(OUT / "stories_f.pkl", "wb"))
    np.save(OUT / "sense_matrix.npy", table.matrix().numpy())
    json.dump({"R": len(vj["rels"]) + 1, "P": len(vj["poss"]) + 1, "rels": vj["rels"], "poss": vj["poss"]}, open(OUT / "dims.json", "w"))
    print("rows", len(table.vecs), flush=True)


class Data:
    def __init__(self, splits=("train", "val")):
        self.sf = pickle.load(open(OUT / "stories_f.pkl", "rb"))
        self.SM = torch.from_numpy(np.load(OUT / "sense_matrix.npy"))
        self.dims = json.load(open(OUT / "dims.json"))
        self.dims["D"] = int(self.SM.size(1))
        self.R, self.P = self.dims["R"], self.dims["P"]
        self.items = {sp: [x for x in pickle.load(open(OUT / f"items_{sp}.pkl", "rb")) if x["refs"]] for sp in splits}


def make_model(dims, seed=0, **kw):
    torch.manual_seed(seed)
    return ML.Loop(int(dims["D"]), dims["R"], dims["P"], **kw)


def save(model, dims, path, extra=None):
    torch.save({"state": model.state_dict(), "dims": dims, "cfg": {"H": model.H}, "extra": extra or {}}, path)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(ck["dims"], **ck["cfg"])
    model.load_state_dict(ck["state"])
    return model.eval(), ck["dims"]


# ----------------------------------------------------------------------------- batches
def forward_batch(model, D: Data, reqs, nstep=ML.NSTEP, block_read=False, train=False, rng=None):
    """reqs: list of dicts {mem: key | None, q: question form, t: reference form | None, qid: item id}. ``mem`` = a story key (read first), None (empty memory:
    nothing is read), or ("q", qid) (the memory is only the question, read as one chunk). Returns (outs per step, halting logits, targets)."""
    keys = []
    for r in reqs:
        if r["mem"] is not None and r["mem"] not in keys:
            keys.append(r["mem"])
    salts = {k: (rng.randrange(1 << 30) if train else zlib_salt(k)) for k in keys}
    books = {k: ML.CodeBook(salts[k]) for k in keys}
    mems = []
    for k in keys:
        if isinstance(k, tuple):                      # memory = only the question, stored as a tree (a stored "sentence")
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
    outs, hl = model.ask(qin, state[ix], (mem[0][ix], mem[1][ix]), nstep, block_read)
    return outs, hl, tg


def zlib_salt(k):
    import zlib
    return zlib.crc32(str(k).encode()) & 0x3FFFFFFF


def train_reqs(items, D: Data, rng, p_wrong=0.12, p_empty=0.06):
    reqs = []
    skeys = sorted({x["skey"] for x in items})
    for x in items:
        reqs.append({"mem": x["skey"], "q": x["q"], "t": rng.choice(x["refs"]), "qid": x["item"]})
        u = rng.random()
        if u < p_wrong and len(skeys) > 1:
            o = rng.choice([s for s in skeys if s != x["skey"]])
            reqs.append({"mem": o, "q": x["q"], "t": None, "qid": x["item"]})
        elif u < p_wrong + p_empty:
            reqs.append({"mem": None, "q": x["q"], "t": None, "qid": x["item"]})
    return reqs


@torch.no_grad()
def val_loss(model, D: Data, items, bs=24, nstep=ML.NSTEP):
    model.eval()
    tot, n, ls, steps = 0.0, 0, [], []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        reqs = [{"mem": x["skey"], "q": x["q"], "t": x["refs"][0], "qid": x["item"]} for x in ch]
        outs, hl, tg = forward_batch(model, D, reqs, nstep)
        loss, Ls, p = ML.ponder_loss(outs, hl, tg)
        tot += float(loss) * len(ch)
        n += len(ch)
        ls.append(Ls * len(ch))
        steps.append(ML.halt_step(p).float().sum())
    return tot / max(n, 1), (sum(ls) / max(n, 1)).tolist(), float(sum(steps)) / max(n, 1) + 1


# ----------------------------------------------------------------------------- decode + rouge (used for checkpoint choice on val and by eval)
class Decode:
    def __init__(self, D: Data, stories, orderer="runs/naomi_orderer/large_best.pt"):
        self.D, self.stories = D, stories
        self.dec = m2.Decoder(str(CT / orderer))

    def texts(self, model, outs, p, items, mems, thr=0.5, tau=0.5, bk=None):
        """Decode the output tree at each item's halting step. items = item dicts (question nodes), mems = story keys used for the pool."""
        hs = ML.halt_step(p, tau)
        trees = []
        st = {k: torch.stack([o[k] for o in outs]) for k in ("vec", "has", "idc", "rel", "pos", "par", "ex")}
        for b, (x, mk) in enumerate(zip(items, mems)):
            nodes = list(self.stories[mk]["nodes"]) if mk is not None and not isinstance(mk, tuple) else []
            pool = m2.Pool(nodes, x["qnodes"])
            tout = {k: v[int(hs[b]), b] for k, v in st.items()}
            salt = bk[b] if bk else 0
            tout = {k: v.unsqueeze(0) for k, v in tout.items()}
            t = ML.decode_tree(tout, 0, pool, ML.CodeBook(salt), self.D.dims["rels"], self.D.dims["poss"], thr)
            trees.append(t)
        ok = [i for i, t in enumerate(trees) if t is not None]
        txt = [""] * len(trees)
        for i, s in zip(ok, self.dec.texts([trees[i] for i in ok])):
            txt[i] = s
        return txt, hs


@torch.no_grad()
def val_rouge(model, D: Data, dc: Decode, items, bs=24, thr=0.5, tau=0.5):
    model.eval()
    sc = []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        reqs = [{"mem": x["skey"], "q": x["q"], "t": None, "qid": x["item"]} for x in ch]
        outs, hl, _tg = forward_batch(model, D, reqs)
        salts = [zlib_salt(r["mem"]) for r in reqs]
        txt, _ = dc.texts(model, outs, ML.halting(hl), ch, [x["skey"] for x in ch], thr, tau, salts)
        sc += [m2.rouge_multi(t, x["golds"]) if t else 0.0 for t, x in zip(txt, ch)]
    return float(np.mean(sc))


def train(args):
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    D = Data()
    D.dims["D"] = int(D.SM.size(1))
    stories = pickle.load(open(V2B / "stories.pkl", "rb"))
    dc = Decode(D, stories)
    tr = D.items["train"]
    if args.short:
        tr = [x for x in tr if x["ans_len"] is not None and x["ans_len"] <= 10]
    if args.overfit:
        sk = sorted({x["skey"] for x in tr})
        random.Random(1).shuffle(sk)
        keep = set(sk[:args.overfit])
        tr = [x for x in tr if x["skey"] in keep]
    va = D.items["val"]
    vsub = random.Random(0).sample(va, min(args.val_n, len(va)))
    if args.overfit:
        vsub = tr[:args.val_n]
    by_story = {}
    for x in tr:
        by_story.setdefault(x["skey"], []).append(x)
    model = make_model(D.dims, args.seed, H=args.H)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["state"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"items {len(tr)} stories {len(by_story)} params {sum(p.numel() for p in model.parameters())}", flush=True)
    t0, step, best = time.time(), 0, -1.0
    curve = open(OUT / f"{args.tag}.curve.tsv", "w")
    curve.write("epoch\tminutes\ttrain_loss\tval_loss\tval_steps\tval_rouge\tper_step_loss\n")
    for ep in range(args.epochs):
        model.train()
        sk = sorted(by_story)
        rng.shuffle(sk)
        tl, nb = 0.0, 0
        for i in range(0, len(sk), args.S):
            items = [x for s in sk[i:i + args.S] for x in by_story[s]]
            reqs = train_reqs(items, D, rng, 0.0 if args.overfit else 0.12, 0.0 if args.overfit else 0.06)
            outs, hl, tg = forward_batch(model, D, reqs, train=True, rng=rng)
            loss, Ls, p = ML.ponder_loss(outs, hl, tg, args.lam, args.beta)
            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (step + 1) / 100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tl += float(loss)
            nb += 1
            if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
                break
        vl, vls, vst = val_loss(model, D, vsub)
        vr = val_rouge(model, D, dc, vsub)
        mins = (time.time() - t0) / 60
        print(f"ep {ep} {mins:.1f}min train {tl / max(nb, 1):.4f} val_loss {vl:.4f} steps {vst:.2f} val_rouge {vr:.4f} per-step {[round(x, 3) for x in vls]}", flush=True)
        curve.write(f"{ep}\t{mins:.2f}\t{tl / max(nb, 1):.4f}\t{vl:.4f}\t{vst:.3f}\t{vr:.4f}\t{json.dumps([round(x, 4) for x in vls])}\n")
        curve.flush()
        save(model, D.dims, OUT / f"{args.tag}_last.pt")
        score = vr - 1e-3 * vl
        if score > best:
            best = score
            save(model, D.dims, OUT / f"{args.tag}_best.pt", {"epoch": ep, "val_rouge": vr, "val_loss": vl})
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train"])
    ap.add_argument("--tag", default="loop_all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--S", type=int, default=12)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--val_n", type=int, default=150)
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    prep() if a.cmd == "prep" else train(a)
