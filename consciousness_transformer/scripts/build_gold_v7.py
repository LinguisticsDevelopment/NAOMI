"""Gold v7: the grammar parser's trees in the NEW tree format (naomi.types.Tree) over the gold-v6 population.

Population = every data/corpus/real_gutenberg_*.txt sentence (sorted files) + the seeded (seed 0, budget 14000)
whole-story K-12 FairytaleQA train subset, first-occurrence dedup, holdout sentences excluded.
Parser only (no encoder fallback), 10 s cap, one Reader per worker, no cache file. Sharded (i::n over the
ordered list), resumable per record, merged back in global order.

  python scripts/build_gold_v7.py --workers 4 --out-dir runs/gold_v7_work          # build shards
  python scripts/build_gold_v7.py --merge --out-dir runs/gold_v7_work              # -> runs/gold_v7*.jsonl.gz
  python scripts/build_gold_v7.py --sentences-file runs/holdout_sentences.txt --out runs/holdout_gold_v7.jsonl
  python scripts/build_gold_v7.py --pilot 200 --workers 4 --out-dir runs/gold_v7_pilot
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from naomi.reader import _PUNCT_POS, Reader, normalize  # noqa: E402
from naomi.score import score_coverage  # noqa: E402
from naomi.tree import tree_to_dict  # noqa: E402
from nsm_ct.corpus import CORPUS_MAX_PARSE_SECONDS, iter_sentences  # noqa: E402

CORPUS_GLOB = str(ROOT / "data" / "corpus" / "real_gutenberg_*.txt")
MANIFEST = ROOT / "runs" / "k12_fairytaleqa_manifest.jsonl"
HOLDOUT_FILES = [ROOT / "runs" / "holdout_sentences.txt", ROOT / "runs" / "holdout_dev_sentences.txt"]
K12_BUDGET, K12_SEED = 14000, 0


def _read_set(paths) -> set:
    out = set()
    for p in paths:
        out |= {ln.strip() for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()}
    return out


def select_k12(exclude: set, budget: int = K12_BUDGET, seed: int = K12_SEED) -> List[Tuple[str, str]]:
    """Seeded whole-story subset (same algorithm as the v6 select_k12_whole_stories.py): [(sentence, story_id)]."""
    by_story: Dict[str, List[str]] = defaultdict(list)
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s = r["sentence"].strip()
            if r["split"] == "train" and s and s not in exclude:
                by_story[r["story_id"]].append(s)
    ids = sorted(by_story)
    random.Random(seed).shuffle(ids)
    chosen, total = [], 0
    for sid in ids:
        if chosen and total + len(by_story[sid]) > budget:
            continue
        chosen.append(sid)
        total += len(by_story[sid])
        if total >= budget:
            break
    return [(s, sid) for sid in chosen for s in by_story[sid]]


def load_population(exclude: Optional[set] = None) -> List[Tuple[str, str]]:
    """[(sentence, source)] in deterministic order: Gutenberg files (sorted), then the K-12 subset."""
    exclude = exclude if exclude is not None else _read_set(HOLDOUT_FILES)
    seen, out = set(), []
    for f in sorted(glob.glob(CORPUS_GLOB)):
        for s in iter_sentences(Path(f).read_text(encoding="utf-8")):
            s = s.strip()
            if s and s not in seen and s not in exclude:
                seen.add(s)
                out.append((s, Path(f).name))
    for s, sid in select_k12(exclude):
        if s not in seen and s not in exclude:
            seen.add(s)
            out.append((s, "k12_fairytaleqa_train:" + sid))
    return out


def make_reader() -> Reader:
    return Reader(parser_seconds=CORPUS_MAX_PARSE_SECONDS, cache_path=None, use_encoder=False)


def build_record(reader: Reader, rid: str, sentence: str, source: str, exclude: Optional[set] = None):
    """-> ("ok", record) | ("fail", failure record) | ("skipped", None)."""
    sentence = sentence.strip()
    if exclude and sentence in exclude:
        return "skipped", None
    t0 = time.perf_counter()
    try:
        rr = reader.read_parser(sentence)
    except Exception as exc:  # noqa: BLE001
        rr, err = None, type(exc).__name__
    else:
        err = None
    secs = time.perf_counter() - t0
    reader._cache.clear()
    norm = normalize(sentence)
    if rr is None or rr.tree is None:
        kind = err or ("cap_hit" if secs >= CORPUS_MAX_PARSE_SECONDS else "no_hypothesis")
        return "fail", {"id": rid, "text": sentence, "source": source, "kind": kind,
                        "n_words": len(norm.split()), "parse_seconds": round(secs, 3)}
    toks, pos = reader.tag(norm)
    cov = score_coverage(rr.tree)
    cd = asdict(cov)
    n_tok = sum(v[1] for v in cov.attached_by_pos.values())
    n_att = sum(v[0] for v in cov.attached_by_pos.values())
    counts: Dict[str, int] = defaultdict(int)
    for d in rr.tree.diagnostics:
        counts[d.get("kind", "?")] += 1
    return "ok", {"id": rid, "text": sentence, "source": source, "tokens": [list(p) for p in zip(toks, pos)],
                  "tree": tree_to_dict(rr.tree), "coverage": cd, "diagnostics": dict(counts),
                  "n_tokens": n_tok, "n_attached": n_att, "parse_seconds": round(secs, 3)}


def run_shard(items: List[Tuple[str, str, str]], ok_path: Path, fail_path: Path, tag: str) -> None:
    """items = [(id, sentence, source)]; appends per record, skips ids already in either file."""
    done = set()
    for p in (ok_path, fail_path):
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                try:
                    done.add(json.loads(ln)["id"])
                except Exception:
                    pass  # torn last line from an interruption: redo it
    reader = make_reader()
    t0, n = time.time(), 0
    with open(ok_path, "a", encoding="utf-8") as fo, open(fail_path, "a", encoding="utf-8") as ff:
        for rid, s, src in items:
            if rid in done:
                continue
            st, rec = build_record(reader, rid, s, src)
            (fo if st == "ok" else ff).write(json.dumps(rec, ensure_ascii=False) + "\n")
            (fo if st == "ok" else ff).flush()
            n += 1
            if n % 200 == 0:
                print(f"[{tag}] {n} new ({len(done) + n}/{len(items)}) {time.time() - t0:.0f}s", flush=True)
    print(f"[{tag}] done: {n} new records in {time.time() - t0:.0f}s", flush=True)


def shard_paths(wd: Path, i: int, n: int) -> Tuple[Path, Path]:
    return wd / f"ok.{i}of{n}.jsonl", wd / f"fail.{i}of{n}.jsonl"


def merge(wd: Path, n: int, out_ok: Path, out_fail: Path, plain: bool = False) -> Tuple[int, int]:
    def collect(idx):
        rows = []
        for i in range(n):
            p = shard_paths(wd, i, n)[idx]
            if p.exists():
                rows += [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return sorted(rows, key=lambda ln: json.loads(ln)["id"])
    res = []
    for idx, out in ((0, out_ok), (1, out_fail)):
        rows = collect(idx)
        op = (lambda p: open(p, "wt", encoding="utf-8")) if plain else (lambda p: gzip.open(p, "wt", encoding="utf-8", compresslevel=9))
        with op(out) as f:
            f.writelines(r + "\n" for r in rows)
        res.append(len(rows))
    return tuple(res)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "gold_v7_work"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", default=None, help="i/n: run only this shard in-process")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--pilot", type=int, default=0, help="first N sentences only (every ~k-th, spread over the corpus)")
    ap.add_argument("--sentences-file", default=None, help="build a gold from a frozen sentence list")
    ap.add_argument("--out", default=None, help="with --sentences-file: output jsonl (failures go to <out>.failures)")
    a = ap.parse_args()
    wd = Path(a.out_dir)
    wd.mkdir(parents=True, exist_ok=True)

    if a.sentences_file:
        sents = [ln.strip() for ln in Path(a.sentences_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
        reader, ok, fail = make_reader(), [], []
        for i, s in enumerate(sents):
            st, rec = build_record(reader, f"h7-{i:05d}", s, Path(a.sentences_file).name)
            (ok if st == "ok" else fail).append(rec)
        Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ok), encoding="utf-8")
        Path(str(a.out) + ".failures").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fail), encoding="utf-8")
        print(f"{a.sentences_file}: {len(ok)} ok, {len(fail)} failed -> {a.out}")
        return 0

    pop = load_population()
    if a.pilot:
        step = max(1, len(pop) // a.pilot)
        pop = pop[::step][:a.pilot]
    items = [(f"v7-{i:06d}", s, src) for i, (s, src) in enumerate(pop)]
    print(f"population: {len(items)} sentences", flush=True)
    n = a.workers
    if a.merge:
        o, f = merge(wd, n, ROOT / "runs" / "gold_v7.jsonl.gz", ROOT / "runs" / "gold_v7_failures.jsonl.gz")
        print(f"merged: {o} ok, {f} failed")
        return 0
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        run_shard(items[i::n], *shard_paths(wd, i, n), tag=f"shard {i}/{n}")
        return 0
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = []
    for i in range(n):
        p = ctx.Process(target=run_shard, args=(items[i::n], *shard_paths(wd, i, n), f"shard {i}/{n}"))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print("exit codes:", [p.exitcode for p in procs], flush=True)
    return 0 if all(p.exitcode == 0 for p in procs) else 1


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
