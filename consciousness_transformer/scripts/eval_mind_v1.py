"""Evaluate Mind v1 on held-out stories (test split), by dataset, with controls, length / trust breakdowns,
the short-to-long comparison and 30 traces for hand-reading.
    python scripts/eval_mind_v1.py [--all runs/mind_v1/policy_all_best.pt] [--short runs/mind_v1/policy_short_best.pt]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from naomi import mind_v1 as m1  # noqa: E402
from naomi.memory import Memory  # noqa: E402

OUT = CT / "runs" / "mind_v1"
RUNS = CT / "runs"


# ----------------------------------------------------------------------------- baselines (surface tokens; eval only)
def toks(s: str):
    return re.findall(r"[a-z0-9']+", s.lower())


def idf_table(items):
    df, n = Counter(), 0
    seen = set()
    for it in items:
        for s in it["context"]:
            if s in seen:
                continue
            seen.add(s)
            n += 1
            df.update(set(toks(s)))
    return lambda w: math.log((n + 1.0) / (df.get(w, 0) + 1.0))


def cheat_pick(it, idf):
    """Word overlap: the story sentence best matching the question, then the option with most (IDF-weighted)
    overlap with it. Returns the expected credit (ties split)."""
    q = set(toks(it["question"]))
    best, bs = None, -1.0
    for s in it["context"]:
        sc = sum(idf(w) for w in set(toks(s)) & q)
        if sc > bs:
            best, bs = s, sc
    st = set(toks(best or ""))
    scores = [sum(idf(w) for w in set(toks(o)) & st) for o in it["options"]]
    top = max(scores)
    winners = [i for i, v in enumerate(scores) if abs(v - top) < 1e-9]
    return (1.0 / len(winners)) if it["answer_idx"] in winners else 0.0


# ----------------------------------------------------------------------------- helpers
def band(n, edges=(10, 20)):
    if n is None:
        return "unknown"
    return f"<={edges[0]}" if n <= edges[0] else (f"{edges[0] + 1}-{edges[1]}" if n <= edges[1] else f"{edges[1] + 1}+")


def qband(n):
    return "<=6" if n <= 6 else ("7-9" if n <= 9 else "10+")


def rates(rows):
    """rows: dicts with 'ok' (1/0/None for abstain) -> accuracy figures."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    ans = [r for r in rows if r["stated"]]
    right = sum(1 for r in ans if r["ok"])
    return {"n": n, "acc": round(right / n, 4), "answer_rate": round(len(ans) / n, 4),
            "acc_when_answering": round(right / len(ans), 4) if ans else None,
            "acc_random_on_abstain": round((right + 0.25 * (n - len(ans))) / n, 4)}


def pred_option(refs, gid, opts):
    idx, m = m1.to_option(refs, gid, opts)
    return idx if float(m.max()) > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default=str(OUT / "policy_all_best.pt"))
    ap.add_argument("--short", default=str(OUT / "policy_short_best.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "eval"))
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(CT / "scripts"))
    import build_mind_data as bd
    items = [r for r in bd.load_items() if r["split"] == args.split]
    if args.limit:
        items = items[:args.limit]
    all_items = bd.load_items()
    reads = bd.load_reads()
    built = {r["id"]: r for r in pickle.load(open(OUT / f"items_{args.split}.pkl", "rb"))}
    models = {}
    for name, path in (("all", args.all), ("short", args.short)):
        if Path(path).exists():
            models[name], rel_vocab = m1.load_policy(path)
    idf = idf_table(all_items)
    rng = random.Random(0)
    by_story = defaultdict(list)
    for it in items:
        by_story[(it["dataset"], it["story"], tuple(it["context"]))].append(it)
    stories = sorted({(it["dataset"], it["story"]) for it in all_items})
    first = {}
    for it in all_items:
        first.setdefault((it["dataset"], it["story"]), it)
    rows = []                              # one per item
    traces = []
    t0 = time.time()
    for si, (key, its) in enumerate(sorted(by_story.items(), key=lambda kv: kv[1][0]["id"])):
        key = key[:2]
        mem = m1.build_story_memory(its[0]["context"], reads)
        refs, _ = m1.node_table(mem)
        for j, it in enumerate(its):
            q = m1.tree_of(reads, it["question"])
            opts = [m1.tree_of(reads, o) for o in it["options"]]
            b = built.get(it["id"])
            row = {"id": it["id"], "dataset": it["dataset"], "story": it["story"], "answer_idx": it["answer_idx"],
                   "cheat": cheat_pick(it, idf), "in_candidates": bool(b["correct"].any()) if b else False,
                   "answer_sentence_words": b.get("answer_sentence_words") if b else None,
                   "answer_sentence_trusted": b.get("answer_sentence_trusted") if b else None,
                   "question_words": b["question_words"] if b else None}
            if q is None or any(o is None for o in opts):
                continue
            # --- normal / severed / question-shuffled / wrong-passage extractions (shared by both models)
            exs = {"normal": m1.extract(mem, q, rel_vocab)}
            sev = Memory()
            sev.store(q, "current", {"source": "question", "sentence_index": 0})
            exs["severed"] = m1.extract(sev, q, rel_vocab)
            sev_refs, _ = m1.node_table(sev)
            others = [o for o in its if o is not it]
            if others:
                oq = m1.tree_of(reads, others[0]["question"])
                exs["shuffled"] = m1.extract(mem, oq, rel_vocab)
            same = [x for x in stories if x[0] == key[0] and x != key]
            o_story = same[len(it["id"]) * 7 % len(same)]
            wmem = m1.build_story_memory(first[o_story]["context"], reads)
            exs["wrong_passage"] = m1.extract(wmem, q, rel_vocab)
            for mname, pol in models.items():
                res = {}
                for var, ex in exs.items():
                    k, p = m1.decide(pol, ex)
                    r = {"stated": k >= 0, "ok": None}
                    if k >= 0:
                        rf = {"normal": refs, "severed": sev_refs, "shuffled": refs, "wrong_passage": None}[var]
                        if rf is not None:
                            po = pred_option(rf, int(ex.cand_gid[k]), opts)
                            r["ok"] = (po == it["answer_idx"])
                            r["pred"] = po
                    res[var] = r
                row[mname] = res
            if "all" in models:
                tr = m1.answer_from_memory(models["all"], rel_vocab, mem, q, ex=exs["normal"])
                tr.update(id=it["id"], dataset=it["dataset"], options=it["options"], correct_option=it["options"][it["answer_idx"]],
                          predicted_option=(it["options"][row["all"]["normal"]["pred"]] if row["all"]["normal"].get("pred") is not None else None),
                          answer_in_candidates=row["in_candidates"], story_sentences=its[0]["context"])
                traces.append(tr)
            rows.append(row)
        print(f"story {si + 1}/{len(by_story)} {time.time() - t0:.0f}s", flush=True)
    json.dump(rows, open(str(args.out) + "_rows.json", "w"))
    with open(str(args.out) + "_traces_all.jsonl", "w") as f:
        for tr in traces:
            f.write(json.dumps(tr) + "\n")
    forced = forced_answer(built, items, models)
    summarise(rows, models, args, forced)
    pick_traces(rows, traces, args)


def ok_rows(rows, model, var="normal"):
    out = []
    for r in rows:
        if model in r and var in r[model]:
            x = r[model][var]
            out.append({"stated": x["stated"], "ok": bool(x["ok"]), "row": r})
    return out


def forced_answer(built, items, models):
    """Diagnostic: the best-scored candidate without the ABSTAIN option (answer always), mapped to an option with the
    precomputed scorer matrices. Separates 'which node' from 'whether to answer'."""
    import torch
    out = {}
    rnd = defaultdict(list)           # a uniformly random candidate, mapped to an option: the mapping's own floor
    for it in items:
        b = built.get(it["id"])
        if b is not None and len(b["ex"].cand_gid):
            m = b["meaning"] * 1000.0 + b["overall"]
            hit = [(b["meaning"][k].max() > 0 and int(np.argmax(m[k])) == b["answer_idx"]) for k in range(len(m))]
            rnd[it["dataset"]].append(float(np.mean(hit)))
            rnd["pooled"].append(float(np.mean(hit)))
        else:
            rnd[it["dataset"]].append(0.0)
            rnd["pooled"].append(0.0)
    out["random_candidate"] = {d: round(float(np.mean(v)), 4) for d, v in rnd.items()}
    for mname, pol in models.items():
        acc = defaultdict(list)
        for it in items:
            b = built.get(it["id"])
            if b is None:
                continue
            ex = b["ex"]
            ok = False
            if len(ex.cand_gid):
                with torch.no_grad():
                    lg = pol(m1.collate([ex]))[0][:len(ex.cand_gid)]
                k = int(lg.argmax())
                ok = b["meaning"][k].max() > 0 and int(np.argmax(b["meaning"][k] * 1000.0 + b["overall"][k])) == b["answer_idx"]
            acc[it["dataset"]].append(bool(ok))
            acc["pooled"].append(bool(ok))
        out[mname] = {d: round(float(np.mean(v)), 4) for d, v in acc.items()}
    return out


def summarise(rows, models, args, forced=None):
    res = {"n_items": len(rows), "forced_answer_acc": forced}
    for ds in ("mctest", "fairytaleqa", "pooled"):
        sel = [r for r in rows if ds == "pooled" or r["dataset"] == ds]
        d = {"n": len(sel), "ceiling_answer_in_candidates": round(float(np.mean([r["in_candidates"] for r in sel])), 4) if sel else None,
             "cheat_word_overlap": round(float(np.mean([r["cheat"] for r in sel])), 4) if sel else None, "random": 0.25}
        for mname in models:
            d[mname] = {"normal": rates(ok_rows(sel, mname)),
                        "severed": rates(ok_rows(sel, mname, "severed")),
                        "question_shuffled": rates(ok_rows([r for r in sel if "shuffled" in r.get(mname, {})], mname, "shuffled")),
                        "wrong_passage_abstain": round(1 - float(np.mean([r[mname]["wrong_passage"]["stated"] for r in sel])), 4) if sel else None}
            by = {}
            for fn, label in ((lambda r: band(r["answer_sentence_words"]), "answer_sentence_words"),
                              (lambda r: qband(r["question_words"]) if r["question_words"] else "unknown", "question_words"),
                              (lambda r: {True: "trusted", False: "untrusted", None: "unknown"}[r["answer_sentence_trusted"]], "answer_sentence_reading")):
                groups = defaultdict(list)
                for r in sel:
                    groups[fn(r)].append(r)
                by[label] = {g: dict(rates(ok_rows(v, mname)), cheat=round(float(np.mean([x["cheat"] for x in v])), 4),
                                     ceiling=round(float(np.mean([x["in_candidates"] for x in v])), 4))
                             for g, v in sorted(groups.items())}
            d[mname]["by"] = by
        res[ds] = d
    # short-to-long: both models on the same held-out items, by answer-sentence band
    if "all" in models and "short" in models:
        s2l = {}
        for scope in ("mctest", "fairytaleqa", "pooled"):
            sel = [r for r in rows if scope == "pooled" or r["dataset"] == scope]
            s2l[scope] = {}
            for bname in ("<=10", "11-20", "21+"):
                g = [r for r in sel if band(r["answer_sentence_words"]) == bname]
                s2l[scope][bname] = {m: rates(ok_rows(g, m)) for m in ("all", "short")}
        res["short_to_long"] = s2l
    json.dump(res, open(str(args.out) + "_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def pick_traces(rows, traces, args):
    """30 held-out traces, seeded, 10 right / 10 wrong / 10 abstain where available (pooled across datasets)."""
    by = {"right": [], "wrong": [], "abstain": []}
    ids = {r["id"]: r for r in rows}
    for t in traces:
        r = ids[t["id"]]["all"]["normal"]
        by["abstain" if not r["stated"] else ("right" if r["ok"] else "wrong")].append(t)
    rng = random.Random(1)
    pick = []
    for k in by:
        rng.shuffle(by[k])
        pick += [dict(t, outcome=k) for t in by[k][:10]]
    json.dump(pick, open(str(args.out) + "_traces_30.json", "w"), indent=1)
    print({k: len(v) for k, v in by.items()}, "-> picked", len(pick))


if __name__ == "__main__":
    main()
