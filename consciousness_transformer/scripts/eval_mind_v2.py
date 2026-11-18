"""Evaluate Mind v2 on one split (run on the test split ONCE, after every choice is made on val).
    python scripts/eval_mind_v2.py --split val  --models s1_all=runs/mind_v2/thinker_s1_all_best.pt s2_all=... --main s2_all
Writes runs/mind_v2/eval_<split>.json (summary tables) and eval_rows_<split>.jsonl (one row per item)."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
sys.path.insert(0, str(CT / "scripts"))
from naomi import mind_v1 as m1  # noqa: E402
from naomi import mind_v2 as m2  # noqa: E402
from naomi.tree import tree_from_dict, tree_to_dict, walk  # noqa: E402
import train_mind_v2 as T  # noqa: E402
import build_mind_data as B  # noqa: E402

OUT = CT / "runs" / "mind_v2"
RB = OUT / "read_cache" / "readback.jsonl"
BANDS = ["<=10", "11-20", "21+", "none"]


def mean(x):
    return float(np.mean(x)) if len(x) else float("nan")


# ----------------------------------------------------------------------------- baselines (text only)
def idf_pick(question: str, sents: list) -> int:
    sets = [set(m2.toks(s)) for s in sents]
    df = Counter(w for s in sets for w in s)
    q = set(m2.toks(question))
    sc = [sum(math.log((len(sents) + 1.0) / (df[w] + 1.0)) for w in (q & s)) for s in sets]
    return int(np.argmax(sc)) if sc else -1


def best_full_subtree_text(arr, golds) -> str:
    best, bt = -1.0, ""
    for g in range(len(arr["parent"])):
        kept = {g} | set(arr["desc"][g].tolist())
        t = m2.true_order_text(arr, kept)
        r = m2.rouge_multi(t, golds)
        if r > best:
            best, bt = r, t
    return bt


# ----------------------------------------------------------------------------- model decisions
@torch.no_grad()
def run_model(model, stories, pairs, tab, bs=48):
    """Per pair: (anchor or -1, kept set or None, p_abstain)."""
    out = {}
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        b = m2.collate(ch, stories, tab, with_targets=False)
        e, anch, res = m2.decide(model, b)
        kept, _ = m2.keep_sets(model, e, b, stories, ch, anch)
        for p, a, k, r in zip(ch, anch, kept, res):
            out[p["id"]] = (a, k, r["p_abstain"])
    return out


def _read_texts(texts):
    """Read decoded strings with the same reader as the references (cached)."""
    have = {}
    for p in (CT / "runs/mind_v2/read_cache/readback.jsonl", CT / "runs/mind_v2/read_cache/roundtrip_reads.jsonl", RB):
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                have[r["k"]] = r
    RB.parent.mkdir(parents=True, exist_ok=True)
    todo = sorted({t for t in texts if t and t not in have})
    if todo:
        print(f"reading back {len(todo)} decoded strings", flush=True)
        with mp.Pool(4, initializer=B._init_reader) as pool, open(RB, "a", encoding="utf-8") as f:
            for rec in pool.imap_unordered(B.read_one, todo, chunksize=2):
                have[rec["k"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--alllen", default=None, help="stage-1 all-lengths model name for the short-to-long table (default: main)")
    ap.add_argument("--short", default=None, help="name of the short-trained model (for the short-to-long table)")
    ap.add_argument("--no_readback", action="store_true")
    ap.add_argument("--dir", default="mind_v2", help="run directory under runs/ (mind_v2b: the full FairytaleQA)")
    ap.add_argument("--rt_thr", default=None, help="roundtrip_<split>.json whose thresholds (declared on val controls) the new round-trip check uses")
    args = ap.parse_args()
    global OUT, RB
    OUT = CT / "runs" / args.dir
    RB = OUT / "read_cache" / "readback.jsonl"
    torch.set_num_threads(4)
    rng = random.Random(0)
    stories = pickle.load(open(OUT / "stories.pkl", "rb"))
    pairs = pickle.load(open(OUT / f"pairs_{args.split}.pkl", "rb"))
    meta = json.load(open(OUT / "items_meta.json"))
    reads = m2.load_reads([B.CACHE, CT / "runs/mind_v2/read_cache/reads_v2.jsonl", OUT / "read_cache" / "reads_v2.jsonl", OUT / "read_cache" / "reads_v2b.jsonl"])
    rt_thr = json.load(open(args.rt_thr))["thresholds"]["A"] if args.rt_thr else 99
    vj = json.load(open(OUT / "vocab.json"))
    vocab = m2.Vocab(vj["rels"], vj["poss"])
    tab = m2.SenseTable()
    ctx = m1.Ctx.get()
    dec = m2.Decoder(str(CT / "runs/naomi_orderer/large_best.pt"))
    normal = [p for p in pairs if p["kind"] == "normal"]
    wrong = [p for p in pairs if p["kind"] == "wrong"]
    shuf = [p for p in pairs if p["kind"] == "shuf"]
    models = {}
    for spec in args.models:
        name, path = spec.split("=")
        models[name] = T.load_model(path)[0]

    # ---- per-item static info
    rows = {}
    for p in normal:
        m = meta[p["item"]]
        t = p["tgt"]
        rows[p["item"]] = {"id": p["item"], "question": m["question"], "reference": m["answer"], "golds": m["golds"],
                           "band": m2.band_of(t["ans_len"]) if t["ans_len"] is not None else "none",
                           "best_f1": t["best_f1"], "reachable": t["best_f1"] >= m2.ACCEPT_F1,
                           "needs_new": t["ref_missing"] >= 1.0 / 3.0, "ref_missing": t["ref_missing"],
                           "story": m["story"], "local": m["local"], "explicit": m["explicit"], "old3000": m.get("old3000"), "n_golds": len(m["golds"])}
    skeys = {p["item"]: p["skey"] for p in normal}

    # ---- baselines and oracles (text)
    for p in normal:
        r = rows[p["item"]]
        arr = stories[p["skey"]]
        sents = arr["sent_text"]
        k = idf_pick(r["question"], sents)
        r["cheat_text"] = sents[k] if k >= 0 else ""
        r["echo_text"] = r["question"]
        r["randsent_text"] = rng.choice(sents) if sents else ""
        n = len(arr["parent"])
        g = rng.randrange(n) if n else -1
        r["randanchor_text"] = m2.true_order_text(arr, {g} | set(arr["desc"][g].tolist())) if g >= 0 else ""
        t = p["tgt"]
        if t["anchors"]:
            a, kept, _f = t["anchors"][0]
            r["oracle_pruned_text"] = m2.true_order_text(arr, set(kept))
        else:
            r["oracle_pruned_text"] = ""
        r["oracle_subtree_text"] = best_full_subtree_text(arr, r["golds"]) if n else ""
    # ---- models
    stats_wc = {}
    for name, model in models.items():
        dm = run_model(model, stories, normal, tab)
        trees = {}
        for p in normal:
            a, kept, pab = dm[p["id"]]
            r = rows[p["item"]]
            r[f"{name}_p_abstain"] = pab
            r[f"{name}_answered"] = a >= 0
            if a >= 0:
                arr = stories[p["skey"]]
                trees[p["item"]] = (m2.answer_tree(arr, a, kept), arr, p)
                r[f"{name}_true_order_text"] = m2.true_order_text(arr, kept)
                r[f"{name}_tree_compact"] = None
        ids = list(trees)
        texts_a = dec.texts([trees[i][0] for i in ids])
        # decoder (b): meaning only
        trees_b, logs = [], {}
        for i in ids:
            tr, arr, p = trees[i]
            qt = m1.tree_of(reads, rows[i]["question"])
            qn = m2.flat_tree(qt)[0] if qt is not None else []
            pool = m2.Pool(arr["nodes"], qn)
            tb, log = m2.meaning_only(tr, pool)
            trees_b.append(tb)
            logs[i] = log
        texts_b = dec.texts(trees_b)
        for i, ta, tb in zip(ids, texts_a, texts_b):
            rows[i][f"{name}_text_a"] = ta
            rows[i][f"{name}_text_b"] = tb
            rows[i][f"{name}_thinker_tree"] = tree_to_dict(trees[i][0])
            lg = logs[i]
            rows[i][f"{name}_wc"] = [(c == o, hp) for c, o, hp in lg]
            rows[i][f"{name}_wc_pairs"] = [(c, o) for c, o, hp in lg]
        # controls
        dw = run_model(model, stories, wrong, tab)
        ds = run_model(model, stories, shuf, tab) if shuf else {}
        sh_scores = []
        for p in shuf:
            a, kept, _ = ds[p["id"]]
            gold = meta[p["item"]]["golds"]
            txt = ""
            if a >= 0:
                arr = stories[p["skey"]]
                txt = dec.texts([m2.answer_tree(arr, a, kept)])[0]
            sh_scores.append((a >= 0, m2.rouge_multi(txt, gold)))
        # question-only memory (v1's severed control) and empty memory
        sev, emp = [], []
        sev_pairs, emp_pairs, sev_st = [], [], {}
        for j, p in enumerate(normal):
            q = meta[p["item"]]["question"]
            qt = m1.tree_of(reads, q)
            if qt is None:
                continue
            s1 = m2.Story([q], reads, vocab)
            sev_st[("s", j)] = s1.arrays()
            pf = m2.pair_features(s1, qt, vocab)
            if pf is not None:
                pf.update(id=p["id"], kind="sev", item=p["item"], skey=("s", j))
                sev_pairs.append(pf)
            s0 = m2.Story([], reads, vocab)
            sev_st[("e", j)] = s0.arrays()
            pf0 = m2.pair_features(s0, qt, vocab)
            if pf0 is not None:
                pf0.update(id=p["id"], kind="emp", item=p["item"], skey=("e", j))
                emp_pairs.append(pf0)
        dsv = run_model(model, sev_st, sev_pairs, tab)
        dem = run_model(model, sev_st, emp_pairs, tab)
        stats_wc[name] = {"wrong_abstain": mean([float(dw[p["id"]][0] < 0) for p in wrong]), "n_wrong": len(wrong),
                          "shuf_answer_rate": mean([a for a, _ in sh_scores]), "shuf_rouge": mean([s for _, s in sh_scores]), "n_shuf": len(sh_scores),
                          "severed_abstain": mean([float(v[0] < 0) for v in dsv.values()]), "empty_abstain": mean([float(v[0] < 0) for v in dem.values()]),
                          "n_severed": len(dsv), "n_empty": len(dem)}

    # ---- readback for the main model (+ cheat) -> tree score, round trip
    M = args.main
    ids_m = [i for i in rows if rows[i].get(f"{M}_answered")]
    texts = []
    for i in ids_m:
        texts += [rows[i][f"{M}_text_a"], rows[i][f"{M}_text_b"], rows[i][f"{M}_true_order_text"]]
    rb = {} if args.no_readback else _read_texts(texts)
    score = ctx.score
    usvs = ctx.usvs

    def read_tree(s):
        r = rb.get(s) or reads.get(s)
        return None if r is None or r["tree"] is None else tree_from_dict(r["tree"])
    reason_counts = {"a": Counter(), "b": Counter(), "true": Counter()}
    for i in rows:
        r = rows[i]
        ref_ts = [t for t in (m1.tree_of(reads, g) for g in r["golds"]) if t is not None]
        for tag, key in (("a", f"{M}_text_a"), ("b", f"{M}_text_b"), ("true", f"{M}_true_order_text")):
            if not r.get(f"{M}_answered"):
                continue
            tt = read_tree(r[key])
            if tt is None or not ref_ts:
                continue
            th = tree_from_dict(r[f"{M}_thinker_tree"])
            r[f"roundtrip_{tag}"] = score.score_pair(th, tt, usvs).overall                       # the OLD check (vs the thinker's tree cut from its sentence)
            if tag == "true":
                continue
            src = read_tree(r[f"{M}_true_order_text"])                                            # NEW check: same words in true order, read as a fragment
            if src is not None:
                r[f"rt2_{tag}"] = float(score.score_pair(src, tt, usvs).overall)
            pss = [score.score_pair(g, tt, usvs) for g in ref_ts]                                  # tree score: max over ALL gold answers
            ps = max(pss, key=lambda x: x.overall)
            r[f"tree_{tag}"] = ps.overall
            r[f"tree_{tag}_meaning"], r[f"tree_{tag}_shape"] = ps.meaning, ps.shape
            r[f"tree_{tag}_reasons"] = [x["kind"] for x in ps.reasons]
            r[f"tree_{tag}_reason_detail"] = [{k: v for k, v in x.items() if k in ("kind", "reference_token", "candidate_token", "side", "token")} for x in ps.reasons][:8]
            reason_counts[tag].update(x["kind"] for x in ps.reasons)
        ct = m1.tree_of(reads, r["cheat_text"])
        if ct is not None and ref_ts:
            r["tree_cheat"] = max(score.score_pair(g, ct, usvs).overall for g in ref_ts)

    # ---- tables
    def rouge_of(r, key, answered_key=None):
        if answered_key is not None and not r.get(answered_key):
            return 0.0
        return m2.rouge_multi(r.get(key) or "", r["golds"])
    systems = {"cheat: best-overlap sentence": "cheat_text", "baseline: question echo": "echo_text",
               "baseline: random story sentence": "randsent_text", "baseline: random-anchor full subtree": "randanchor_text",
               "oracle: best full subtree (by ROUGE, true order)": "oracle_subtree_text", "oracle: lemma-F1 pruned subtree (true order)": "oracle_pruned_text"}
    sysfn = {k: (lambda r, v=v: m2.rouge_multi(r.get(v) or "", r["golds"])) for k, v in systems.items()}
    for name in models:
        sysfn[f"{name}: (a) order-only decoder"] = (lambda r, n=name: rouge_of(r, f"{n}_text_a", f"{n}_answered"))
        sysfn[f"{name}: (b) meaning-only decoder"] = (lambda r, n=name: rouge_of(r, f"{n}_text_b", f"{n}_answered"))
        sysfn[f"{name}: true word order (no decoder)"] = (lambda r, n=name: rouge_of(r, f"{n}_true_order_text", f"{n}_answered"))
    R = list(rows.values())

    def table(select):
        out = {}
        for k, f in sysfn.items():
            vals = [f(r) for r in select]
            row = {"n": len(vals), "rouge_abstain0": mean(vals)}
            if k.split(":")[0] in models:
                nm = k.split(":")[0]
                ans = [f(r) for r in select if r.get(f"{nm}_answered")]
                row.update(answer_rate=mean([float(bool(r.get(f"{nm}_answered"))) for r in select]), rouge_answered_only=mean(ans))
            out[k] = row
        return out
    summary = {"split": args.split, "n_items": len(R), "n_reachable": sum(r["reachable"] for r in R), "n_needs_new": sum(r["needs_new"] for r in R),
               "overall": table(R), "by_band": {b: table([r for r in R if r["band"] == b]) for b in BANDS},
               "needs_new_words": table([r for r in R if r["needs_new"]]), "reachable_only": table([r for r in R if r["reachable"]]),
               "old3000_subset": table([r for r in R if r.get("old3000")]), "n_old3000": sum(bool(r.get("old3000")) for r in R),
               "new_items_only": table([r for r in R if not r.get("old3000")]),
               "multi_gold_items": sum(r["n_golds"] > 1 for r in R),
               "band_counts": {b: sum(r["band"] == b for r in R) for b in BANDS}, "controls": stats_wc,
               "oracle_ceiling_note": "oracle rows use the reference to pick the subtree; they are ceilings, not systems"}
    # tree scores / round trip for the main model
    ans = [r for r in R if r.get(f"{M}_answered")]
    tsum = {}
    for tag in ("a", "b", "true"):
        v = [r[f"tree_{tag}"] for r in ans if f"tree_{tag}" in r]
        rt = [r[f"roundtrip_{tag}"] for r in ans if f"roundtrip_{tag}" in r]
        tsum[tag] = {"tree_score_answered_mean": mean(v), "tree_score_abstain0_mean": sum(v) / max(len(R), 1), "n_scored": len(v),
                     "roundtrip_share_lt50": mean([float(x < 50.0) for x in rt]), "roundtrip_mean": mean(rt), "roundtrip_fail_share_lt80": mean([float(x < 80.0) for x in rt]), "n_roundtrip": len(rt),
                     "rt2_pass_rate": mean([float(r[f"rt2_{tag}"] >= rt_thr) for r in ans if f"rt2_{tag}" in r]) if tag != "true" else None, "rt2_threshold": rt_thr,
                     "reason_counts": dict(reason_counts[tag].most_common())}
        if tag == "true":
            tsum[tag] = {k: v_ for k, v_ in tsum[tag].items() if k.startswith("roundtrip") or k == "n_roundtrip"}
    for tag in ("a", "b"):
        tsum[tag]["rt2_by_len"] = {nm: {"n": len(sel), "pass_rate": mean([float(r[f"rt2_{tag}"] >= rt_thr) for r in sel])} for nm, sel in
                                   (("1-3 words", [r for r in ans if f"rt2_{tag}" in r and len(r[f"{M}_true_order_text"].split()) <= 3]),
                                    ("4-6 words", [r for r in ans if f"rt2_{tag}" in r and 4 <= len(r[f"{M}_true_order_text"].split()) <= 6]),
                                    ("7+ words", [r for r in ans if f"rt2_{tag}" in r and len(r[f"{M}_true_order_text"].split()) >= 7]))}
    cheat_t = [r["tree_cheat"] for r in R if "tree_cheat" in r]
    tsum["cheat_tree_score_mean"] = mean(cheat_t)
    summary["tree_and_roundtrip_main"] = tsum
    summary["main"] = M
    # word choice accuracy (decoder b)
    wc = {}
    for name in models:
        allw = [x for r in R if r.get(f"{name}_answered") for x in r[f"{name}_wc"]]
        wp = [c for c, hp in allw if hp]
        np_ = [c for c, hp in allw if not hp]
        wc[name] = {"nodes": len(allw), "accuracy": mean([c for c, _ in allw]), "with_meaning_point": {"n": len(wp), "acc": mean(wp)},
                    "no_point_names_or_unlexiconed": {"n": len(np_), "acc": mean(np_)}}
    summary["word_choice_accuracy"] = wc
    # short-to-long
    if args.short:
        AL = args.alllen or M
        summary["short_to_long"] = {b: {"n": sum(r["band"] == b for r in R),
                                        "all_lengths": mean([sysfn[f"{AL}: (a) order-only decoder"](r) for r in R if r["band"] == b]),
                                        "short_only": mean([sysfn[f"{args.short}: (a) order-only decoder"](r) for r in R if r["band"] == b])}
                                    for b in BANDS}
    json.dump(summary, open(OUT / f"eval_{args.split}.json", "w"), indent=1)
    with open(OUT / f"eval_rows_{args.split}.jsonl", "w") as f:
        for r in R:
            f.write(json.dumps(r, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("split", "n_items", "n_reachable", "n_needs_new", "band_counts")}))
    for k, v in summary["overall"].items():
        print(f"{k:55s} {v}")
    print(json.dumps(summary["controls"], indent=1))
    print(json.dumps(summary["tree_and_roundtrip_main"], indent=1))
    print(json.dumps(summary["word_choice_accuracy"], indent=1))


if __name__ == "__main__":
    main()
