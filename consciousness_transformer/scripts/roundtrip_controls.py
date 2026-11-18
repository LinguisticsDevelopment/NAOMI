"""Round-trip check controls and application (Mind v2b step 1).
    python scripts/roundtrip_controls.py --dir runs/mind_v2 --split val  --model s2b_all --fit      # choose thresholds on val controls
    python scripts/roundtrip_controls.py --dir runs/mind_v2 --split test --model s2b_all --thr runs/mind_v2/roundtrip_val.json
For every answered item of an eval-rows file the thinker's tree gives the true-order control; negative controls are
built from it (shuffled order, adjacent swap, one word replaced by a random word of the story, one word dropped).
Formulations (src/naomi/roundtrip.py): A reread (fragment vs same words in true order), B order discord (text level),
C substitution (decoded words back into the source sentence).  Writes <dir>/roundtrip_<split>.json (+ _rows.jsonl)."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import random
import sys
from pathlib import Path

import numpy as np

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_v1 as m1  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402
from naomi import roundtrip as rt  # noqa: E402
from naomi.tree import tree_from_dict  # noqa: E402
import build_mind_data as B  # noqa: E402

GRID_A = [50, 60, 70, 80, 90, 95, 99, 100]
GRID_B = [0.0, 0.05, 0.1, 0.2, 0.25, 0.34, 0.5]
NEG = ["shuffle", "adjswap", "wordswap", "drop"]


def read_all(texts, cache_path, workers, extra=()):
    have = {}
    for p in list(extra) + [cache_path]:
        if Path(p).exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                have[r["k"]] = r
    todo = sorted({t for t in texts if t and t not in have})
    if todo:
        print(f"reading {len(todo)} strings", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with mp.Pool(workers, initializer=B._init_reader) as pool, open(cache_path, "a", encoding="utf-8") as f:
            for rec in pool.imap_unordered(B.read_one, todo, chunksize=2):
                have[rec["k"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return have


def find_sentence(arr, nodes):
    """Index of the story sentence whose tokens at the nodes' positions equal the nodes' tokens."""
    for s, text in enumerate(arr["sent_text"]):
        t = text.split()
        if all(n.token_index < len(t) and t[n.token_index] == n.token for n in nodes):
            return s
    return None


def build_cases(rows, model, stories, meta, rng, dec):
    """One case per answered item: tree, source words, decoded orders (a)/(b) and the negative controls."""
    cases = []
    for r in rows:
        if not r.get(f"{model}_answered"):
            continue
        tree = tree_from_dict(r[f"{model}_thinker_tree"])
        src = rt.source_nodes(tree)
        words = [n.token for n in src]
        arr = stories[meta[r["id"]]["context_key"]]
        c = {"id": r["id"], "n": len(words), "src": " ".join(words), "tree": tree, "src_nodes": src,
             "text_a": r[f"{model}_text_a"], "text_b": r[f"{model}_text_b"], "arr": arr}
        s = find_sentence(arr, src)
        c["sent"] = None if (s is None or len(cases) % 3) else arr["sent_text"][s]       # substitution check on every third item (long reads are slow)
        pool = [n.token for n in arr["nodes"]]
        neg = {}
        if len(words) >= 2:
            p = list(range(len(words)))
            for _ in range(20):
                rng.shuffle(p)
                if p != sorted(p):
                    break
            neg["shuffle"] = [words[i] for i in p]
            i = rng.randrange(len(words) - 1)
            w = list(words)
            w[i], w[i + 1] = w[i + 1], w[i]
            neg["adjswap"] = w
            i = rng.randrange(len(words))
            cands = [x for x in pool if x != words[i]]
            if cands:
                w = list(words)
                w[i] = rng.choice(cands)
                neg["wordswap"] = w
            i = rng.randrange(len(words))
            neg["drop"] = words[:i] + words[i + 1:]
            c["drop_pos"] = src[i].token_index
        c["neg_words"] = neg
        cases.append(c)
    # decoded node orders (a)/(b) for the order check: re-run the orderer on the thinker trees
    trees_a = [c["tree"] for c in cases]
    for c, ns in zip(cases, dec.orr.order_trees(trees_a) if trees_a else []):
        c["order_a_nodes"] = ns
    return cases


def sentence_variants(c, words, drop=False):
    """Source sentence with the kept slots filled by ``words`` (equal length), or with the dropped token removed."""
    if c["sent"] is None:
        return None
    if drop:
        toks = c["sent"].split()
        del toks[c["drop_pos"]]
        return " ".join(toks)
    return rt.substitute(c["sent"], [n.token_index for n in c["src_nodes"]], words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="s2b_all")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--thr", default=None)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    D = Path(args.dir) if Path(args.dir).is_absolute() else CT / args.dir
    rng = random.Random(0)
    rows = [json.loads(l) for l in open(D / f"eval_rows_{args.split}.jsonl")]
    stories = pickle.load(open(D / "stories.pkl", "rb"))
    meta = json.load(open(D / "items_meta.json"))
    ctx = m1.Ctx.get()
    dec = m2.Decoder(str(CT / "runs/naomi_orderer/large_best.pt"))
    cases = build_cases(rows, args.model, stories, meta, rng, dec)
    # ---- every string the checks need
    texts = []
    for c in cases:
        texts += [c["src"], c["text_a"], c["text_b"]]
        for k, w in c["neg_words"].items():
            texts.append(" ".join(w))
        if c["sent"] is not None:
            texts.append(c["sent"])
            for words in [c["text_a"].split(), c["text_b"].split()]:
                texts.append(sentence_variants(c, words))
            for k, words in c["neg_words"].items():
                texts.append(sentence_variants(c, words, drop=(k == "drop")))
    extra = [B.CACHE, CT / "runs/mind_v2/read_cache/reads_v2.jsonl", CT / "runs/mind_v2/read_cache/readback.jsonl"]
    extra += [p for p in (D / "read_cache" / "reads_v2b.jsonl",) if p.exists()]
    rb = read_all(texts, D / "read_cache" / "roundtrip_reads.jsonl", args.workers, extra)

    def read(s):
        r = rb.get(s)
        return None if r is None or r["tree"] is None else tree_from_dict(r["tree"])
    score = lambda ref, cand: rt.reread_score(read, ctx.score, ctx.usvs, cand, ref)
    # ---- scores per case
    for c in cases:
        c["A"], c["B"], c["C"] = {}, {}, {}
        c["A"]["true"], c["C"]["true"] = 100.0, 100.0
        c["B"]["true"] = 0.0 if c["n"] >= 2 else None
        for nm, key in (("a", "text_a"), ("b", "text_b")):
            c["A"][nm] = score(c["src"], c[key])
            if c["sent"] is not None:
                c["C"][nm] = score(c["sent"], sentence_variants(c, c[key].split()) or c["sent"])
        c["B"]["a"] = rt.order_discord(c["tree"], c["order_a_nodes"])
        c["B"]["b"] = None                 # the order check is evaluated on setting (a) only (needs the orderer's node order)
        for k, w in c["neg_words"].items():
            c["A"][k] = score(c["src"], " ".join(w))
            if c["sent"] is not None:
                c["C"][k] = score(c["sent"], sentence_variants(c, w, drop=(k == "drop")))
            if k in ("shuffle", "adjswap"):
                srcw = [n.token for n in c["src_nodes"]]
                # order of the permuted words as positions of the source (words may repeat: use first unused match)
                used, order = set(), []
                for x in w:
                    j = next(i for i, y in enumerate(srcw) if y == x and i not in used)
                    used.add(j)
                    order.append(c["src_nodes"][j])
                c["B"][k] = rt.order_discord(c["tree"], order)
    multi = [c for c in cases if c["n"] >= 2]

    def rate(vals, thr, higher_better=True):
        v = [x for x in vals if x is not None]
        if not v:
            return float("nan")
        return float(np.mean([(x >= thr) if higher_better else (x <= thr) for x in v]))
    out = {"split": args.split, "model": args.model, "n_answered": len(cases), "n_multiword": len(multi)}
    if args.fit:
        fit = {}
        for form, grid, hb in (("A", GRID_A, True), ("C", GRID_A, True), ("B", GRID_B, False)):
            rows_t = []
            for t in grid:
                ctrl = rate([c[form]["true"] for c in cases], t, hb)
                negs = {k: rate([c[form].get(k) for c in multi], t, hb) for k in NEG if any(c[form].get(k) is not None for c in multi)}
                rows_t.append({"thr": t, "true_pass": ctrl, "neg_pass": negs, "neg_mean": float(np.mean(list(negs.values()))) if negs else float("nan")})
            ok = [r for r in rows_t if r["true_pass"] >= 0.95 and not np.isnan(r["neg_mean"])]
            lo = min(r["neg_mean"] for r in ok)
            near = [r for r in ok if r["neg_mean"] <= lo + 0.02]
            chosen = (min(r["thr"] for r in near) if hb else max(r["thr"] for r in near))
            fit[form] = {"grid": rows_t, "chosen": chosen, "rule": "most lenient threshold whose mean negative-control pass rate is within 0.02 of the minimum, with true-order pass >= 0.95"}
        out["fit"] = fit
        thr = {f: fit[f]["chosen"] for f in fit}
    else:
        thr = json.load(open(args.thr))["thresholds"]
    out["thresholds"] = thr

    def passfn(form, v):
        return None if v is None else (v <= thr[form] if form == "B" else v >= thr[form])
    res = {}
    for kind in ["true", "a", "b"] + NEG:
        pop = cases if kind in ("true", "a", "b") else multi
        res[kind] = {"n": len(pop)}
        for form in ("A", "B", "C"):
            v = [passfn(form, c[form].get(kind)) for c in pop]
            v = [x for x in v if x is not None]
            res[kind][form] = {"pass_rate": float(np.mean(v)) if v else None, "n": len(v)}
        v = []
        for c in pop:
            pa, pb = passfn("A", c["A"].get(kind)), passfn("B", c["B"].get(kind))
            if pa is None:
                continue
            v.append(pa and (pb is None or pb))
        res[kind]["A_and_B"] = {"pass_rate": float(np.mean(v)) if v else None, "n": len(v)}
    out["pass_rates"] = res
    old = [r for r in rows if r.get(f"{args.model}_answered") and "roundtrip_a" in r]
    out["old_check"] = {"share_lt80_a": float(np.mean([r["roundtrip_a"] < 80 for r in old])) if old else None,
                        "share_lt80_true_order": float(np.mean([r["roundtrip_true"] < 80 for r in old])) if old else None, "n": len(old)}
    ids_fail = [c["id"] for c in cases if passfn("A", c["A"].get("a")) is False]
    out["fail_ids_a"] = ids_fail[:50]
    json.dump(out, open(D / f"roundtrip_{args.split}.json", "w"), indent=1, default=float)
    with open(D / f"roundtrip_{args.split}_rows.jsonl", "w") as f:
        for c in cases:
            f.write(json.dumps({"id": c["id"], "n": c["n"], "src": c["src"], "text_a": c["text_a"], "text_b": c["text_b"], "A": c["A"], "B": c["B"], "C": c["C"]}, default=float) + "\n")
    print(json.dumps({k: out[k] for k in out if k not in ("fit", "fail_ids_a")}, indent=1, default=float))
    if args.fit:
        for form, v in out["fit"].items():
            print(form, "chosen", v["chosen"])
            for r in v["grid"]:
                print("  ", r["thr"], round(r["true_pass"], 3), {k: round(x, 3) for k, x in r["neg_pass"].items()}, round(r["neg_mean"], 3))


if __name__ == "__main__":
    main()
