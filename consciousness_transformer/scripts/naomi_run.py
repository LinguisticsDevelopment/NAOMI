#!/usr/bin/env python
"""NAOMI runner: `read` prints the trees of a text; `score` grades parser vs encoder trees."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))

from naomi.mind import render_clause, split_sentences  # noqa: E402
from naomi.reader import Reader, normalize  # noqa: E402
from naomi.score import score_against, score_tree  # noqa: E402


def _p(path):
    p = Path(path)
    return p if p.is_absolute() else CT / p


def cmd_read(a):
    r = Reader(cache_path=a.cache)
    text = _p(a.text).read_text() if _p(a.text).exists() else a.text
    for i, s in enumerate(split_sentences(text)):
        rr = r.read(s, i)
        print(f"[{rr.source}] {s}")
        for c in rr.clauses:
            print("   ", render_clause(c), " | ", " ".join(f"{x.role}@{x.token_index}={x.token}"
                                                          for x in [c.predicate] + c.args))
    r.save_cache()


def load_sentences(path, limit):
    """One sentence per line, or episodes JSONL (key `context` = list of sentences)."""
    out, seen = [], set()
    for line in open(_p(path), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        sents = [line]
        if line.startswith("{"):
            sents = json.loads(line).get("context", [])
        for s in sents:
            if s not in seen:
                seen.add(s)
                out.append(s)
        if len(out) >= limit:
            break
    return out[:limit]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _hist(diags):
    return Counter(d["kind"] + (":" + d["pos"] if d["kind"] == "token_dropped" else "") for d in diags)


def cmd_score(a):
    r = Reader(cache_path=a.cache)
    sents = load_sentences(a.sentences, a.limit)
    t0, done = time.time(), 0
    agg = {"parser": [], "encoder": []}
    comps = {"parser": {}, "encoder": {}}
    hist = {"parser": Counter(), "encoder": Counter()}
    ref, n_fail = [], 0
    for s in sents:
        if time.time() - t0 > a.time_limit:
            print(f"time limit hit after {done} sentences (rerun to continue; results are cached)")
            break
        pr, er = r.read_both(s)
        toks, pos = r.tag(normalize(s))
        n_fail += pr is None
        row = {}
        for name, rr in (("parser", pr), ("encoder", er)):
            if rr is None:
                continue
            sc = score_tree(toks, pos, rr.clauses)
            row[name] = sc
            agg[name].append(sc.percent)
            for k, v in sc.components.items():
                comps[name].setdefault(k, []).append(v)
            hist[name].update(_hist(sc.diagnostics))
        if pr is not None:
            ref.append(score_against(pr.clauses, er.clauses, toks))
        done += 1
        if not a.quiet:
            print(f"- {s}")
            for name, sc in row.items():
                print(f"    {name:8s} {sc.percent:5.1f}%  " + json.dumps(sc.diagnostics)[:300])
    print(f"\n== {done} sentences; parser failed on {n_fail} ({100 * n_fail / max(done, 1):.1f}%)")
    for name in ("parser", "encoder"):
        cm = {k: round(_mean(v), 3) for k, v in comps[name].items()}
        print(f"{name}: mean coverage {_mean(agg[name]):.1f}% over {len(agg[name])}  components {cm}")
        print(f"   diagnostics: {dict(hist[name].most_common())}")
    if ref:
        print(f"encoder vs parser (parser succeeded, n={len(ref)}): mean {_mean([x.percent for x in ref]):.1f}%  "
              f"edge P/R/F1 {_mean([x.edge_precision for x in ref]):.3f}/{_mean([x.edge_recall for x in ref]):.3f}/"
              f"{_mean([x.edge_f1 for x in ref]):.3f}  pred recall {_mean([x.predicate_recall for x in ref]):.3f}")
        print(f"   diagnostics: {dict(Counter(d['kind'] for x in ref for d in x.diagnostics).most_common())}")
    r.save_cache()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("read", help="print the trees of a text (string or file, one sentence per line)")
    t.add_argument("--text", required=True)
    t.add_argument("--cache", default=None)
    s = sp.add_parser("score", help="coverage score of parser vs encoder trees")
    s.add_argument("--sentences", required=True, help="text file (one per line) or episodes .jsonl")
    s.add_argument("--limit", type=int, default=100)
    s.add_argument("--cache", default="runs/naomi_reader_cache.jsonl")
    s.add_argument("--time-limit", type=float, default=110.0)
    s.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    {"read": cmd_read, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
