#!/usr/bin/env python
"""NAOMI end-to-end runner: teach-ask demo, resumable benchmark, report."""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))

from naomi.memory import Memory  # noqa: E402
from naomi.mind import Mind, render_clause, split_sentences  # noqa: E402
from naomi.reader import Reader  # noqa: E402
from naomi.recall import option_index  # noqa: E402

DATASETS = {"mctest": "runs/mctest_mc160_episodes.jsonl",
            "ftqa": "runs/fairytaleqa_episodes_samepassage.jsonl"}
STOPW = frozenset("a an the of to in on at by for with and or is was were are be been it he she they "
                  "his her their its that this what who where when why how which did do does had has have "
                  "not from as but so then there them him i you we me my our your up out".split())


def _p(path):
    p = Path(path)
    return p if p.is_absolute() else CT / p


def cwords(s):
    out = set()
    for w in "".join(ch if ch.isalnum() else " " for ch in s.lower()).split():
        if w not in STOPW:
            out.add(w)
            if w.endswith("s") and len(w) > 3:
                out.add(w[:-1])
    return out


# ----------------------------------------------------------------------------- teach-ask
def cmd_teach_ask(a):
    text = a.text
    if os.path.exists(text):
        text = Path(text).read_text()
    m = Mind(cache_path=a.cache)
    print("taught:", m.teach(text, source="cli"))
    for q in a.ask or []:
        ans = m.ask(q)
        print(f"Q: {q}\nA: {m.say(ans)}  (score={ans.score:.2f})")
        if ans.support is not None:
            print("   support:", render_clause(ans.support), f"[{ans.support.source}]")
        if a.trace:
            for t in ans.trace:
                print("   |", t)
    m.reader.save_cache()


# ----------------------------------------------------------------------------- bench
def load_items(dataset, episodes=None):
    items = [json.loads(l) for l in open(_p(episodes or DATASETS[dataset]))]
    return [it for it in items if it.get("options")]


def memory_lemmas(mem):
    out = set()
    for c in mem.clauses():
        for f in [c.tree.predicate] + c.tree.args:
            out.add(f.lemma)
            out.add(f.token)
    return out


def cmd_bench(a):
    items = load_items(a.dataset, a.episodes)
    if a.split:
        items = [i for i in items if i["meta"]["split"] == a.split]
    if a.max_items_per_story:   # cap first, so --subset samples the capped list
        seen, kept = defaultdict(int), []
        for it in items:
            seen[it["meta"]["story_id"]] += 1
            if seen[it["meta"]["story_id"]] <= a.max_items_per_story:
                kept.append(it)
        items = kept
    if a.subset and a.subset < len(items):  # fixed, evenly spaced subset for quick iteration
        step = len(items) / a.subset
        items = [items[int(i * step)] for i in range(a.subset)]
    by_story = defaultdict(list)
    for it in items:
        by_story[it["meta"]["story_id"]].append(it)
    stories = sorted(by_story)
    ctx_of = {s: by_story[s][0]["context"] for s in stories}
    sel = stories[a.offset:a.offset + a.max_stories] if a.max_stories else stories[a.offset:]
    out_path = _p(a.out or f"runs/naomi_bench_{a.dataset}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in open(out_path):
            try:
                r = json.loads(l)
                done.add((r["story_id"], r["question"], r.get("ctx_hash")))
            except Exception:
                pass
    reader = Reader(cache_path=a.cache)
    if a.shuffle_questions:
        shuffle_questions(by_story)
    t_start, n_new = time.time(), 0
    with open(out_path, "a") as fo:
        for s in sel:
            nxt = stories[(stories.index(s) + 1) % len(stories)]
            k = 0
            for it in by_story[s]:
                if a.max_items_per_story and k >= a.max_items_per_story:
                    break
                k += 1
                ctx_hash = hash_ctx(it["context"])
                if (s, it["question"], ctx_hash) in done:
                    continue
                if time.time() - t_start > a.time_limit:
                    print(f"time limit hit; {n_new} new items written")
                    return
                t0 = time.time()
                rec = run_item(reader, it, s, ctx_hash)
                if a.shuffle_questions:
                    rec["shuffled"] = True
                    rec["orig_question"] = it["_orig_question"]
                if a.wrong_passage and nxt != s:
                    rec["wrong_abstained"] = run_wrong(reader, it, ctx_of[nxt])
                rec["seconds"] = round(time.time() - t0, 2)
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n_new += 1
    print(f"done; {n_new} new items written")


def shuffle_questions(by_story):
    """Control: give each item the question of the NEXT item of the same story (cyclic,
    deterministic); context, options and gold index stay the original ones. Stories with
    a single item keep their question (nothing to swap with)."""
    for grp in by_story.values():
        qs = [g["question"] for g in grp]
        for j, g in enumerate(grp):
            g["_orig_question"] = g["question"]
            g["question"] = qs[(j + 1) % len(qs)]


def hash_ctx(ctx):
    import hashlib
    return hashlib.md5("\n".join(ctx).encode()).hexdigest()[:8]


def run_item(reader, it, story, ctx_hash):
    mind = Mind(reader=reader, memory=Memory())
    counts = mind.teach(it["context"], source=story)
    q, opts = it["question"], it["options"]
    rr = reader.read_question(q)
    qc = rr.clauses[0] if rr.clauses else None
    hole = qc.hole() if qc else None
    ans = mind.ask(q, options=opts)
    idx = option_index(ans) if ans.text is not None else None
    sup = ans.support
    gold_words = cwords(it["answer_text"]) if it.get("answer_text") else cwords(opts[it["answer_idx"]])
    mem = memory_lemmas(mind.memory)
    return {
        "story_id": story, "ctx_hash": ctx_hash, "question": q, "options": opts,
        "gold": it["answer_idx"], "pred": idx, "correct": idx == it["answer_idx"],
        "answered": idx is not None,
        "hole_role": hole.role if hole else None,
        "question_tree": render_clause(qc) if qc else None,
        "question_source": qc.source if qc else None,
        "support": render_clause(sup) if sup else None,
        "support_source": sup.source if sup else None,
        "support_sentence": sup.sentence if sup else None,
        "negated": bool(ans.negated),
        "trace": ans.trace[:5],
        "gold_in_memory": bool(gold_words & mem),
        "teach": counts, "n_context": len(it["context"]),
    }


def run_wrong(reader, it, other_ctx):
    mind = Mind(reader=reader, memory=Memory())
    mind.teach(other_ctx, source="wrong")
    ans = mind.ask(it["question"], options=it["options"])
    return option_index(ans) is None


# ----------------------------------------------------------------------------- report
def overlap_baseline(it_rec, ctx):
    qw = cwords(it_rec["question"])
    best, bs = None, -1
    for s in ctx:
        sc = len(cwords(s) & qw)
        if sc > bs:
            best, bs = s, sc
    sw = cwords(best or "")
    scores = [len(cwords(o) & sw) for o in it_rec["options"]]
    m = max(scores)
    return scores.index(m)  # ties -> first


def pct(x, n):
    return f"{100.0 * x / n:.1f}%" if n else "n/a"


def categorize(r):
    if r["hole_role"] is None:
        return "no_hole"
    if not r["gold_in_memory"]:
        return "gold_not_in_memory"
    if not r["answered"]:
        return "abstain_gold_in_memory"
    if r["negated"]:
        return "negation"
    # answered wrong: did the support clause mention the gold answer?
    gw = cwords(r["options"][r["gold"]])
    sup_w = cwords(r["support_sentence"] or "")
    if gw & sup_w:
        return "right_clause_wrong_option"
    if r["support"]:
        return "wrong_support_clause"
    return "other"


def cmd_report(a):
    inp = _p(a.inp)
    recs = [json.loads(l) for l in open(inp)]
    ds = "ftqa" if "ftqa" in inp.name else "mctest"
    n = len(recs)
    ans = [r for r in recs if r["answered"]]
    cor = sum(r["correct"] for r in ans)
    print(f"n={n}")
    print(f"answered: {pct(len(ans), n)} ({len(ans)})")
    print(f"accuracy when answered: {pct(cor, len(ans))} ({cor}/{len(ans)})")
    print(f"overall accuracy (abstain=wrong): {pct(cor, n)}")
    print(f"accuracy w/ random guess on abstain: {pct(cor + 0.25 * (n - len(ans)), n)}")
    print(f"gold_in_memory: {pct(sum(r['gold_in_memory'] for r in recs), n)}")
    for key in ("hole_role", "support_source"):
        d = defaultdict(lambda: [0, 0])
        for r in ans:
            d[r[key]][0] += r["correct"]
            d[r[key]][1] += 1
        print(f"accuracy when answered by {key}:")
        for k, (c, t) in sorted(d.items(), key=lambda kv: -kv[1][1]):
            print(f"   {k}: {pct(c, t)} ({c}/{t})")
    wp = [r for r in recs if "wrong_abstained" in r]
    if wp:
        print(f"wrong-passage abstain: {pct(sum(r['wrong_abstained'] for r in wp), len(wp))} (n={len(wp)})")
    secs = [r["seconds"] for r in recs if "seconds" in r]
    if secs:
        print(f"mean seconds/item (incl. wrong-passage, as run): {sum(secs) / len(secs):.2f}")
    # cheat baselines on the same items
    items = {(i["meta"]["story_id"], i["question"], hash_ctx(i["context"])): i for i in load_items(ds)}
    ov, k = 0, 0
    for r in recs:
        it = items.get((r["story_id"], r.get("orig_question", r["question"]), r["ctx_hash"]))
        if it is None:
            continue
        k += 1
        ov += overlap_baseline(r, it["context"]) == r["gold"]
    if recs and recs[0].get("shuffled"):
        print("(question-shuffled control: chance is 25%; the word-overlap number below is NOT "
              "meaningful here)")
    print(f"cheat baselines: random=25.0%  word-overlap={pct(ov, k)} (n={k})")
    fails, cats = [], defaultdict(int)
    for r in recs:
        if (r["answered"] and not r["correct"]) or (not r["answered"] and r["gold_in_memory"]):
            r2 = dict(r)
            r2["category"] = categorize(r)
            cats[r2["category"]] += 1
            fails.append(r2)
    fp = _p(a.failures or f"runs/naomi_failures_{ds}.jsonl")
    with open(fp, "w") as f:
        for r in fails:
            f.write(json.dumps(r) + "\n")
    print(f"failures: {len(fails)} -> {fp}; categories: {dict(cats)}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("teach-ask")
    t.add_argument("--text", required=True)
    t.add_argument("--ask", action="append")
    t.add_argument("--trace", action="store_true")
    t.add_argument("--cache", default=None)
    t.set_defaults(fn=cmd_teach_ask)
    b = sp.add_parser("bench")
    b.add_argument("--dataset", choices=list(DATASETS), required=True)
    b.add_argument("--episodes", help="episodes jsonl overriding the dataset default")
    b.add_argument("--split")
    b.add_argument("--max-stories", type=int)
    b.add_argument("--max-items-per-story", type=int)
    b.add_argument("--offset", type=int, default=0)
    b.add_argument("--out")
    b.add_argument("--cache", default="runs/naomi_read_cache.jsonl")
    b.add_argument("--wrong-passage", action="store_true")
    b.add_argument("--subset", type=int, help="evenly spaced fixed subset of N items")
    b.add_argument("--shuffle-questions", action="store_true",
                   help="control: pair each item's context+options with another item's question "
                        "(same story, deterministic), scored against the original gold index")
    b.add_argument("--time-limit", type=float, default=110.0, help="stop starting new items after N s")
    b.set_defaults(fn=cmd_bench)
    r = sp.add_parser("report")
    r.add_argument("--in", dest="inp", required=True)
    r.add_argument("--failures", help="where to write the failures list")
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
