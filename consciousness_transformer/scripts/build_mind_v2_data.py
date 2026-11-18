"""Mind v2 data build.  Stage ``read``: read every string of the FairytaleQA free-form items (story sentences, question,
first reference answer) not already in the v1 read cache, with the same reader as v1 (parser + A2 encoder links).
Stage ``items`` (added below): per (story, question) pair numeric features + training targets.
    python scripts/build_mind_v2_data.py read [--workers 4]
    python scripts/build_mind_v2_data.py items [--workers 4]
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
sys.path.insert(0, str(CT / "scripts"))
RUNS = CT / "runs"
OUT = RUNS / "mind_v2"
CACHE2 = OUT / "read_cache" / "reads_v2.jsonl"
V2B = RUNS / "mind_v2b"                                  # full FairytaleQA (scripts/build_fairytaleqa_full_items.py)
CACHE_B = V2B / "read_cache" / "reads_v2b.jsonl"
ITEMS_B = V2B / "items_full.jsonl"


def stage_read(args):
    import build_mind_data as B
    from naomi import mind_v2 as m2
    global CACHE2
    if args.full:
        items = m2.load_items(ITEMS_B)
        have = set(m2.load_reads([B.CACHE, CACHE2]))
        CACHE2 = CACHE_B
    else:
        items = m2.load_items(B.FT)
        have = set(B.load_reads())
    seen, todo = set(), []
    for it in items:
        # train items: first answer only (the training target); val/test: every gold answer (tree score takes the max)
        golds = [it["answer"]] if (it["split"] == "train" or not args.full) else it["golds"]
        for s in list(it["context"]) + [it["question"]] + golds:
            if s and s not in have and s not in seen:
                seen.add(s)
                todo.append(s)
    CACHE2.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if CACHE2.exists():
        for line in open(CACHE2, encoding="utf-8"):
            try:
                done.add(json.loads(line)["k"])
            except Exception:
                pass
    todo = [s for s in todo if s not in done]
    print(f"to read {len(todo)}", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers, initializer=B._init_reader) as pool, open(CACHE2, "a", encoding="utf-8") as f:
        for n, rec in enumerate(pool.imap_unordered(B.read_one, todo, chunksize=4), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                f.flush()
                print(f"read {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("read done", time.time() - t0, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["read", "items"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--full", action="store_true", help="Mind v2b: the full FairytaleQA items, cache and outputs in runs/mind_v2b")
    args = ap.parse_args()
    if args.stage == "read":
        stage_read(args)
    else:
        import build_mind_data as B
        from naomi import mind_v2 as m2
        if args.full:
            m2.build_items_stage(args, B, V2B, items_path=ITEMS_B, base_caches=[B.CACHE, CACHE2])
        else:
            m2.build_items_stage(args, B, OUT)


if __name__ == "__main__":
    main()
