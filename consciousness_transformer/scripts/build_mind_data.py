"""Mind v1 data build.

Stage ``read``: read every distinct sentence / question / option string of the two datasets into a tree
(parser; the A2 encoder fills the joining links, parser links kept) and mark it trusted or not (three
models agree with head probability >= 0.8 on every added link; reporting only). Cached to
runs/mind_v1/read_cache/reads.jsonl (resumable, NOT committed). Stage ``items``: per-item candidates,
features and labels (see naomi.mind_v1). Usage:
    python scripts/build_mind_data.py read [--workers 4] [--limit N]
    python scripts/build_mind_data.py items [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
RUNS = CT / "runs"
OUT = RUNS / "mind_v1"
CACHE = OUT / "read_cache" / "reads.jsonl"
MC = RUNS / "mctest_mc160_episodes.jsonl"
FT = RUNS / "fairytaleqa_episodes_samepassage.jsonl"


def load_items():
    from naomi import mind_v1 as m1
    return m1.load_items(MC, FT)


def all_strings(items):
    seen, out = set(), []
    for it in items:
        for s in list(it["context"]) + [it["question"]] + list(it["options"]):
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


# ---------------------------------------------------------------- reading
_W = {}


def _init_reader():
    import torch
    torch.set_num_threads(1)
    from naomi import encoder as E
    from naomi.reader import Reader, _sense_space
    usvs = _sense_space()
    _W["reader"] = Reader(use_encoder=False, cache_path=None)
    _W["encs"] = {n: E.NaomiEncoder.load(str(RUNS / p), usvs) for n, p in
                  (("A", "naomi_encoder/main_best.pt"), ("A2", "naomi_encoder_a2/a2_best.pt"),
                   ("C", "naomi_encoder_a2/a2ctrl_best.pt"))}


def read_one(s: str) -> dict:
    import numpy as np
    from naomi import encoder as E
    from naomi.tree import tree_to_dict
    from naomi.types import REL_ROOT
    reader, encs = _W["reader"], _W["encs"]
    t0 = time.time()
    infos = E.infos_from_sentence(reader, s)
    nodes = [i for i, t in enumerate(infos) if t.is_node]
    rec = {"k": s, "n_words": len(nodes), "tree": None, "trusted": False, "parser": "ok", "n_added": 0}
    if not nodes:
        return rec
    if len(nodes) == 1:
        i = nodes[0]
        rec["tree"] = tree_to_dict(E.build_tree(s, infos, {i: 0}, {i: REL_ROOT}, "parser"))
        rec.update(trusted=True, parser="trivial")
        return rec
    rr = None
    try:
        rr = reader.read_parser(s)
    except Exception:
        rr = None
    forced, sv = {}, None
    if rr is None or rr.tree is None:
        rec["parser"] = "failed"
    else:
        sv = E.supervision_from_tree(rr.tree, len(infos))
        if max([n.token_index for n in _walk(rr.tree) if n.token_index is not None] + [0]) >= len(infos):
            rec["parser"] = "failed"
            sv = None
        else:
            forced = {i: sv.head[i] for i in nodes if sv.sup[i] and (sv.head[i] == 0 or infos[sv.head[i] - 1].is_node)}
    unknown = [i for i in nodes if i not in forced]
    outs = {}
    for n, enc in encs.items():
        H, logp, _ = enc.arc_logprobs(infos)
        _n, W, h, heads = enc._decode(infos, H, logp, forced)
        ids, _ = enc._rels(H, [(i, heads[i]) for i in unknown])
        outs[n] = (np.exp(logp), heads, {i: enc.rel_labels[k] for i, k in zip(unknown, ids)})
    heads = outs["A2"][1]
    rels = {}
    ok = rec["parser"] == "ok"
    for i in nodes:
        if i in forced:
            rels[i] = REL_ROOT if heads[i] == 0 else sv.rel[i]
        else:
            rels[i] = outs["A2"][2][i]
            h2 = heads[i]
            pa, pc, p2 = outs["A"][0][i, h2], outs["C"][0][i, h2], outs["A2"][0][i, h2]
            chosen = outs["A"][1][i] == h2 and outs["C"][1][i] == h2
            ok = ok and chosen and min(pa, pc, p2) >= 0.8
    rec["n_added"] = len(unknown)
    rec["trusted"] = bool(ok)
    rec["tree"] = tree_to_dict(E.build_tree(s, infos, heads, rels, "parser" if not unknown else "encoder"))
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def _walk(tree):
    from naomi.tree import walk
    return [n for n, _d, _p in walk(tree.root)]


def stage_read(args):
    items = load_items()
    strs = all_strings(items)
    if args.limit:
        strs = strs[:args.limit]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if CACHE.exists():
        for line in open(CACHE, encoding="utf-8"):
            try:
                done.add(json.loads(line)["k"])
            except Exception:
                pass
    todo = [s for s in strs if s not in done]
    print(f"strings {len(strs)} done {len(done)} todo {len(todo)}", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init_reader) as pool, open(CACHE, "a", encoding="utf-8") as f:
        for n, rec in enumerate(pool.imap_unordered(read_one, todo, chunksize=4), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                f.flush()
                print(f"read {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("read done", time.time() - t0, flush=True)


def load_reads():
    out = {}
    for line in open(CACHE, encoding="utf-8"):
        r = json.loads(line)
        out[r["k"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["read", "items"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "read":
        stage_read(args)
    else:
        from naomi import mind_v1 as m1
        m1.build_items_stage(args, load_items, load_reads, OUT)


if __name__ == "__main__":
    main()
