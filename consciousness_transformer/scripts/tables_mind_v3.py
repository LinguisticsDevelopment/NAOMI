"""Markdown tables for dev/reports/MIND_V3.md from runs/mind_v3/eval_<split>.json and eval_rows_<split>.jsonl (no model is run).
    python scripts/tables_mind_v3.py val v3_s2 v2b v3_s1 --short v3_short s1_short --alllen v3_s1 s1_all"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
from naomi import mind_v2 as m2  # noqa: E402

BANDS = ["<=10", "11-20", "21+", "none"]


def score(r, name, kind="a"):
    if not r.get(f"{name}_answered"):
        return 0.0
    key = {"a": "text_a", "b": "text_b", "t": "true_order_text"}[kind]
    return m2.rouge_multi(r.get(f"{name}_{key}") or "", r["golds"])


def boot(rows, f, g, n=2000, seed=0):
    """Paired bootstrap of mean(f - g) over items and over stories."""
    rng = np.random.default_rng(seed)
    d = np.array([f(r) - g(r) for r in rows])
    st = defaultdict(list)
    for r, x in zip(rows, d):
        st[r["story"]].append(x)
    sv = [np.array(v) for v in st.values()]
    bi = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(sv), len(sv))
        arr = np.concatenate([sv[i] for i in pick])
        bs.append(arr.mean())
    return float(d.mean()), np.percentile(bi, [2.5, 97.5]), np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("models", nargs="+", help="models for the headline / band tables; the first is the main model, the second the v2b comparison")
    ap.add_argument("--short", nargs=2, action="append", default=[], metavar=("SHORT", "ALLLEN"), help="short-only / all-lengths pair (repeatable)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dir", default="mind_v3", help="run directory holding eval_<split>.json (mind_v4 for the v4 tables)")
    args = ap.parse_args()
    D = CT / "runs" / args.dir
    rows = [json.loads(l) for l in open(D / f"eval_rows_{args.split}{args.tag}.jsonl")]
    S = json.load(open(D / f"eval_{args.split}{args.tag}.json"))
    main_m, base_m = args.models[0], args.models[1]
    print(f"## {args.split}: n={len(rows)} items, {len({r['story'] for r in rows})} stories; bands {S['band_counts']}\n")
    print("### Headline ROUGE-L (max over golds, abstain = 0)\n\n| system | ROUGE-L | answered-only | answer rate |\n|---|---|---|---|")
    for k, v in S["overall"].items():
        if k.split(":")[0] in args.models or not any(k.startswith(x + ":") for x in S["controls"]):
            print(f"| {k} | {v['rouge_abstain0']:.3f} | {v.get('rouge_answered_only', float('nan')):.3f} | {v.get('answer_rate', float('nan')):.3f} |")
    print("\n### By answer-sentence length band (decoder (a))\n")
    cols = [m for m in args.models] 
    print("| band | n | cheat | " + " | ".join(cols) + " | oracle |\n|---|---|" + "---|" * (len(cols) + 2))
    for b in BANDS:
        sel = [r for r in rows if r["band"] == b]
        if not sel:
            continue
        cheat = np.mean([m2.rouge_multi(r["cheat_text"], r["golds"]) for r in sel])
        orc = np.mean([m2.rouge_multi(r["oracle_subtree_text"], r["golds"]) for r in sel])
        print(f"| {b} | {len(sel)} | {cheat:.3f} | " + " | ".join(f"{np.mean([score(r, m) for r in sel]):.3f}" for m in cols) + f" | {orc:.3f} |")
    print("\n### Short-to-long (decoder (a))\n\n| pair | band | n | all lengths | <=10 only |\n|---|---|---|---|---|")
    for sh, al in args.short:
        for b in BANDS:
            sel = [r for r in rows if r["band"] == b]
            print(f"| {sh} vs {al} | {b} | {len(sel)} | {np.mean([score(r, al) for r in sel]):.3f} | {np.mean([score(r, sh) for r in sel]):.3f} |")
    print("\n### Paired differences (decoder (a); mean, 95% interval over items, over stories)\n")
    cheat = lambda r: m2.rouge_multi(r["cheat_text"], r["golds"])
    for nm, f, g in ((f"{main_m} - {base_m}", lambda r: score(r, main_m), lambda r: score(r, base_m)), (f"{main_m} - cheat", lambda r: score(r, main_m), cheat)) + tuple(
            (f"{main_m} - {m}", (lambda r: score(r, main_m)), (lambda r, m=m: score(r, m))) for m in args.models[2:]):
        d, ci, cs = boot(rows, f, g)
        print(f"- {nm}: {d:+.3f} (items {ci[0]:+.3f} to {ci[1]:+.3f}; stories {cs[0]:+.3f} to {cs[1]:+.3f})")
    # attribute
    ids = {}
    n = 0
    for line in open(CT / "runs" / "mind_v2b" / "items_full.jsonl", encoding="utf-8"):
        r = json.loads(line)
        m = r["meta"]
        golds = [g for g in (m.get("all_gold_answers") or [r.get("answer_text")]) if g]
        if not r.get("answer_text") or not golds:
            continue
        ids[f"ft{n}"] = m.get("attribute")
        n += 1
    print("\n### By FairytaleQA attribute (decoder (a))\n\n| attribute | n | cheat | " + " | ".join(cols) + " | oracle | answer rate (main) |\n|---|---|---|" + "---|" * (len(cols) + 2))
    by = defaultdict(list)
    for r in rows:
        by[ids.get(r["id"]) or "?"].append(r)
    for a, sel in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"| {a} | {len(sel)} | {np.mean([m2.rouge_multi(r['cheat_text'], r['golds']) for r in sel]):.3f} | " + " | ".join(f"{np.mean([score(r, m) for r in sel]):.3f}" for m in cols)
              + f" | {np.mean([m2.rouge_multi(r['oracle_subtree_text'], r['golds']) for r in sel]):.3f} | {np.mean([bool(r.get(main_m + '_answered')) for r in sel]):.3f} |")
    print("\n### Controls\n\n| model | empty-memory abstain | wrong story abstain | shuffled questions: answer rate / ROUGE | memory = question only: abstain |\n|---|---|---|---|---|")
    for k, v in S["controls"].items():
        print(f"| {k} | {v['empty_abstain']:.3f} | {v['wrong_abstain']:.3f} | {v['shuf_answer_rate']:.3f} / {v['shuf_rouge']:.3f} | {v['severed_abstain']:.3f} |")
    t = S["tree_and_roundtrip_main"]
    print(f"\n### Tree score and round trip (main = {S['main']}; threshold {t['a']['rt2_threshold']})\n")
    for tag in ("a", "b"):
        x = t[tag]
        print(f"- ({tag}) tree score answered {x['tree_score_answered_mean']:.1f}, abstain=0 {x['tree_score_abstain0_mean']:.1f}; new round-trip pass {x['rt2_pass_rate']:.3f}; by length {json.dumps(x['rt2_by_len'])}; reasons {json.dumps(x['reason_counts'])}")
    print(f"- cheat tree score {t['cheat_tree_score_mean']:.1f}")
    print("\n### Needs-new-words items\n")
    for k, v in S["needs_new_words"].items():
        if k.split(":")[0] in args.models:
            print(f"- {k}: {v['rouge_abstain0']:.3f} (n={v['n']})")


if __name__ == "__main__":
    main()
