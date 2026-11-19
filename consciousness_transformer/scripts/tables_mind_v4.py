"""v4-only tables for dev/reports/MIND_V4.md (no model is run): walk diagnostics by band, chosen vs oracle depth, tree adequacy,
failure attribution, and the hand-read dump.
    python scripts/tables_mind_v4.py test v4_all v4_short --alllen v4_all --handread 20   (hand-read only on val)
Reads runs/mind_v4/eval_<split>.json / eval_rows_<split>.jsonl / steps_<split>.json and the pair / story pickles."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CT / "src"))
from naomi import mind_v2 as m2  # noqa: E402

BANDS = ["<=10", "11-20", "21+", "none"]
D = CT / "runs" / "mind_v4"


def outline(arr, s, mark=()):
    idx = [g for g in range(len(arr["parent"])) if arr["sent"][g] == s]
    lines = []
    for g in idx:
        lines.append("  " * int(arr["depth"][g]) + f"{arr['nodes'][g].token}/{arr['nodes'][g].rel}" + (" <== answer" if g in mark else "") + (" <== v4" if g in mark and False else ""))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--main", default=None)
    ap.add_argument("--handread", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    main_m = args.main or args.models[0]
    rows = {r["id"]: r for r in (json.loads(l) for l in open(D / f"eval_rows_{args.split}{args.tag}.jsonl"))}
    steps = json.load(open(D / f"steps_{args.split}{args.tag}.json"))
    pairs = {p["item"]: p for p in pickle.load(open(D / f"pairs_{args.split}.pkl", "rb")) if p["kind"] == "normal"}
    stories = pickle.load(open(D / "stories.pkl", "rb"))
    train = pickle.load(open(D / "pairs_train.pkl", "rb"))
    short_depths = [stories[p["skey"]]["depth"][p["tgt"]["anchors"][0][0]] for p in train if p["kind"] == "normal" and p.get("tgt") and p["tgt"]["anchors"]
                    and p["tgt"]["best_f1"] >= m2.ACCEPT_F1 and p["tgt"]["ans_len"] is not None and p["tgt"]["ans_len"] <= 10]
    maxd_short = int(max(short_depths))
    print(f"## v4 walk diagnostics ({args.split})\n\nShort-trained items: oracle anchor depth max {maxd_short}, 95th pct {int(np.percentile(short_depths, 95))}, mean {np.mean(short_depths):.2f}.\n")
    print("### Steps, budget exhaustion, chosen anchor depth vs oracle depth, by answer-sentence band\n")
    print("| model | band | n | answer rate | mean steps | budget exhausted | chosen depth (answered) | oracle depth | chosen deeper than short-train max | chosen anchor is acceptable | chosen sentence is the oracle sentence |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for nm in args.models:
        for b in BANDS:
            ids = [i for i, r in rows.items() if r["band"] == b and i in steps.get(nm, {})]
            if not ids:
                continue
            st = [steps[nm][i] for i in ids]
            ans = [s for s in st if s["answered"]]
            od = [stories[pairs[i]["skey"]]["depth"][pairs[i]["tgt"]["anchors"][0][0]] for i in ids if pairs[i]["tgt"]["anchors"] and rows[i]["reachable"]]
            ok_a, ok_s, nr = [], [], 0
            for i in ids:
                if not (pairs[i]["tgt"]["anchors"] and rows[i]["reachable"]):
                    continue
                nr += 1
                s = steps[nm][i]
                acc = {a for a, _k, _f in pairs[i]["tgt"]["anchors"]}
                sents = {int(stories[pairs[i]["skey"]]["sent"][a]) for a in acc}
                ok_a.append(s["answered"] and s["anchor"] in acc)
                ok_s.append(s["answered"] and s["sent"] in sents)
            print(f"| {nm} | {b} | {len(ids)} | {len(ans) / len(ids):.3f} | {np.mean([s['steps'] for s in st]):.2f} | {np.mean([s['exhausted'] for s in st]):.3f} | "
                  f"{(np.mean([s['depth'] for s in ans]) if ans else float('nan')):.2f} | {(np.mean(od) if od else float('nan')):.2f} | "
                  f"{(np.mean([s['depth'] > maxd_short for s in ans]) if ans else float('nan')):.3f} | {np.mean(ok_a) if ok_a else float('nan'):.3f} | {np.mean(ok_s) if ok_s else float('nan'):.3f} |")
    # tree adequacy
    print("\n### Tree adequacy: share of items whose reference answer equals some subtree at ROUGE-L >= 0.8 (any node as anchor, true word order)\n")
    print("| band | n | share >= 0.8 | share >= 0.5 | mean best-subtree ROUGE |\n|---|---|---|---|---|")
    ad = {}
    for b in BANDS:
        sel = [r for r in rows.values() if r["band"] == b]
        if not sel:
            continue
        v = np.array([m2.rouge_multi(r["oracle_subtree_text"], r["golds"]) for r in sel])
        for r, x in zip(sel, v):
            ad[r["id"]] = float(x)
        print(f"| {b} | {len(sel)} | {np.mean(v >= 0.8):.3f} | {np.mean(v >= 0.5):.3f} | {v.mean():.3f} |")
    v = np.array(list(ad.values()))
    print(f"| all | {len(v)} | {np.mean(v >= 0.8):.3f} | {np.mean(v >= 0.5):.3f} | {v.mean():.3f} |")
    # failure attribution of the main model by band
    print(f"\n### Failure attribution, {main_m} (items with a reachable oracle answer; ROUGE-L true order; fail = ROUGE < 0.5 x oracle subtree ROUGE... see columns)\n")
    print("| band | n | right (within 0.05 of oracle subtree) | abstained / budget out | wrong sentence | right sentence, wrong anchor, TREE adequate (>=0.8) | right sentence, wrong anchor, tree NOT adequate |\n|---|---|---|---|---|---|---|")
    for b in ("<=10", "11-20", "21+"):
        ids = [i for i, r in rows.items() if r["band"] == b and r["reachable"] and i in steps.get(main_m, {}) and pairs[i]["tgt"]["anchors"]]
        cnt = defaultdict(int)
        for i in ids:
            s = steps[main_m][i]
            r = rows[i]
            acc = {a for a, _k, _f in pairs[i]["tgt"]["anchors"]}
            sents = {int(stories[pairs[i]["skey"]]["sent"][a]) for a in acc}
            sc = m2.rouge_multi(r.get(f"{main_m}_true_order_text") or "", r["golds"]) if s["answered"] else 0.0
            if s["answered"] and sc >= ad[i] - 0.05:
                cnt["right"] += 1
            elif not s["answered"]:
                cnt["abst"] += 1
            elif s["sent"] not in sents:
                cnt["wrong_sent"] += 1
            elif ad[i] >= 0.8:
                cnt["wa_tree_ok"] += 1
            else:
                cnt["wa_tree_bad"] += 1
        n = max(len(ids), 1)
        print(f"| {b} | {len(ids)} | " + " | ".join(f"{cnt[k] / n:.3f} ({cnt[k]})" for k in ("right", "abst", "wrong_sent", "wa_tree_ok", "wa_tree_bad")) + " |")
    if args.handread:
        rng = random.Random(0)
        fails = [i for i, r in rows.items() if r["band"] == "21+" and i in steps.get(main_m, {}) and m2.rouge_multi(r.get(f"{main_m}_true_order_text") or "", r["golds"]) < 0.3
                 and pairs[i]["tgt"]["anchors"]]
        pick = rng.sample(sorted(fails), min(args.handread, len(fails)))
        out = []
        for i in pick:
            p, r, s = pairs[i], rows[i], steps[main_m][i]
            arr = stories[p["skey"]]
            a0 = p["tgt"]["anchors"][0][0]
            sent = int(arr["sent"][a0])
            path_tok = [arr["nodes"][g].token for g in s["path"]]
            out.append({"id": i, "question": r["question"], "reference": r["reference"], "oracle_sentence_text": arr["sent_text"][sent], "oracle_sentence_tree": outline(arr, sent, {a0}),
                        "oracle_best_subtree_text": r["oracle_subtree_text"], "oracle_best_subtree_rouge": ad.get(i), "oracle_anchor": arr["nodes"][a0].token,
                        "v4_path_tokens": path_tok, "v4_path_sentences": [int(arr["sent"][g]) for g in s["path"]], "oracle_sentence_index": sent, "v4_answered": s["answered"],
                        "v4_text": r.get(f"{main_m}_true_order_text"), "v4_rouge": m2.rouge_multi(r.get(f"{main_m}_true_order_text") or "", r["golds"]) if s["answered"] else 0.0,
                        "v4_exhausted": s["exhausted"]})
        json.dump({"n_fail_pool": len(fails), "items": out}, open(D / f"handread_{args.split}_21plus.json", "w"), indent=1)
        print(f"\nhand-read dump: {len(out)} of {len(fails)} failing 21+ items -> runs/mind_v4/handread_{args.split}_21plus.json")


if __name__ == "__main__":
    main()
