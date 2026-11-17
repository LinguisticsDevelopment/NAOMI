"""Build gold records for UD English sentences: the project's parser's pieces + converted treebank links for the
tops of unconnected pieces (parser wins everywhere it committed).

  python scripts/build_treebank_gold.py --split dev  --workers 4
  python scripts/build_treebank_gold.py --split test --workers 4
  python scripts/build_treebank_gold.py --split train --limit 6000 --workers 4     # seeded subset of the train file

Per split: parse (resumable shards in runs/treebank/work_<split>/), then compose -> runs/treebank/ewt_<split>.jsonl.gz
(gold v7 schema + per-token sup_head / sup_rel / sup_source / sup_flags) and runs/treebank/stats_<split>.json.
Licensed data: nothing under data/ud/ or runs/treebank/ is committed; this script + data/treebank/conversion_table.json
reproduce it (``naomi.treebank.fetch`` downloads the files).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naomi import treebank as TB  # noqa: E402

UD_DIR = ROOT / "data" / "ud" / "EWT"
OUT = ROOT / "runs" / "treebank"
REUSE = str(ROOT / "dev" / "reports" / "treebank_feasibility_scripts" / "out" / "parser_ewt.*.jsonl")


def parse_shard(items, path, shard, nsh):
    from naomi.reader import Reader, normalize
    from naomi.tree import tree_to_dict
    done = set()
    if Path(path).exists():
        done = {json.loads(ln)["id"] for ln in open(path) if ln.strip()}
    rd = Reader(use_encoder=False, cache_path=None, parser_seconds=10)
    with open(path, "a") as out:
        for k, (sid, text) in enumerate(items):
            if k % nsh != shard or sid in done:
                continue
            t0 = time.time()
            try:
                toks, pos = rd.tag(normalize(text))
                rr = rd.read_parser(text)
                tree = tree_to_dict(rr.tree) if rr and rr.tree else None
            except Exception:  # noqa: BLE001
                toks, pos, tree = [], [], None
            rd._cache.clear()
            out.write(json.dumps({"id": sid, "text": text, "tokens": toks, "pos": pos, "secs": round(time.time() - t0, 2),
                                  "tree": tree}) + "\n")
            out.flush()


def collect(work: Path):
    res = {}
    for f in sorted(glob.glob(str(work / "shard.*.jsonl"))) + sorted(glob.glob(REUSE)):
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln)
                res.setdefault(r["id"], r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0, help="seeded random subset of this many sentences")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compose-only", action="store_true")
    a = ap.parse_args()
    table = TB.load_table()
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / f"work_{a.split}"
    work.mkdir(exist_ok=True)
    sents = [s for s in TB.read_conllu(UD_DIR / f"en_ewt-ud-{a.split}.conllu") if sum(t.upos != "PUNCT" for t in s.toks)]
    if a.limit and a.limit < len(sents):
        keep = set(random.Random(a.seed).sample(range(len(sents)), a.limit))
        sents = [s for i, s in enumerate(sents) if i in keep]
    items = [(s.sent_id, s.text) for s in sents]
    print(f"{a.split}: {len(items)} sentences", flush=True)
    if not a.compose_only:
        have = collect(work)
        todo = [it for it in items if it[0] not in have]
        print(f"already parsed {len(items) - len(todo)}, to parse {len(todo)}", flush=True)
        if todo:
            t0 = time.time()
            ps = [mp.Process(target=parse_shard, args=(todo, str(work / f"shard.{k}.jsonl"), k, a.workers))
                  for k in range(a.workers)]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
            print(f"parsed in {(time.time() - t0) / 60:.1f} min", flush=True)
    compose_split(a.split, sents, collect(work), table)


def compose_split(split, sents, parsed, table):
    from naomi.encoder import supervision_from_tree
    from naomi.tree import tree_from_dict
    st = Counter()
    by_rule, by_orig, by_ops = Counter(), Counter(), Counter()
    out = gzip.open(OUT / f"ewt_{split}.jsonl.gz", "wt", encoding="utf-8")
    for s in sents:
        st["sentences"] += 1
        r = parsed.get(s.sent_id)
        if r is None or r["tree"] is None or not r["tokens"]:
            st["parser_failed"] += 1
            continue
        tree = tree_from_dict(r["tree"])
        n = len(r["tokens"])
        sv = supervision_from_tree(tree, n)
        is_node = [p != "PUNCT" for p in r["pos"]]
        groups, status = TB.align(s, r["tokens"])
        st["align_" + status] += 1
        links = {}
        if groups is not None:
            conv = TB.convert(s, table)
            if conv.well_formed:
                links = TB.project_links(conv, groups, is_node)
            else:
                st["conversion_not_well_formed"] += 1
        c = TB.compose(sv.head, sv.rel, sv.sup, sv.unknown, links, is_node)
        st["parser_connected" if sv.connected else "parser_pieces"] += 1
        for k, v in c.counts.items():
            st["links_" + k] += v
        st["unknown_tops_before"] += sum(sv.unknown)
        for i in range(n):
            if is_node[i]:
                st["node_tokens"] += 1
                st["tok_" + c.source[i]] += 1
            if c.source[i] == TB.SRC_TREEBANK:
                by_rule[c.flags[i].split('+')[-1]] += 1
                by_ops[c.flags[i]] += 1
                by_orig[links[i].orig] += 1
        out.write(json.dumps({"id": "ewt-" + s.sent_id, "text": r["text"], "source": "ud_ewt_" + split,
                              "tokens": [[t, p] for t, p in zip(r["tokens"], r["pos"])], "tree": r["tree"],
                              "align": status, "sup_head": c.head, "sup_rel": c.rel, "sup_source": c.source,
                              "sup_flags": c.flags}) + "\n")
    out.close()
    stats = {"split": split, "counts": dict(st), "treebank_links_by_rule": dict(by_rule.most_common()),
             "treebank_links_by_operations_and_rule": dict(by_ops.most_common()),
             "treebank_links_by_treebank_relation": dict(by_orig.most_common())}
    json.dump(stats, open(OUT / f"stats_{split}.json", "w"), indent=1)
    print(json.dumps(stats["counts"], indent=1), flush=True)


if __name__ == "__main__":
    main()
