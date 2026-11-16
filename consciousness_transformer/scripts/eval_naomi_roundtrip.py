"""Stage-A referee + round-trip evaluation (no training).

  python scripts/eval_naomi_roundtrip.py attested          # attested-configuration table + choice of N on dev + (a)/(b) agreement
  python scripts/eval_naomi_roundtrip.py part3             # do referee signals separate good from bad added links? (173 hand-judged)
  python scripts/eval_naomi_roundtrip.py roundtrip [--max-sents N]   # first round-trip numbers on TEST-source sentences
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from naomi import encoder as E  # noqa: E402
from naomi import orderer as O  # noqa: E402
from naomi import referee as R  # noqa: E402
from naomi.reader import _sense_space  # noqa: E402
from naomi.score import score_pair  # noqa: E402
from naomi.tree import tree_from_dict  # noqa: E402
from train_naomi_encoder import GOLD, load_jsonl  # noqa: E402
from train_naomi_orderer import load_items  # noqa: E402

OUT = ROOT / "runs" / "naomi_orderer"
ENC = ROOT / "runs" / "naomi_encoder"
SWEEP_N = (1, 2, 3, 5, 10, 20, 30, 40, 50, 100)


def classes_from_encoder():
    ck = torch.load(ENC / "main_best.pt", map_location="cpu", weights_only=False)
    return set(ck["sub_vocab"])


# ----------------------------------------------------------------------------- attested table + N on dev
def stage_attested(a):
    classes = classes_from_encoder()
    by = load_items()
    cnt = R.attested_from_items(by["train"], classes)
    att = R.Attested(cnt, classes, 1)
    gram = R.GrammarLicense()
    rng = random.Random(0)
    rels = sorted({r[2] for r in cnt})

    def links(items):
        pos, neg = [], []
        for it in items:
            sv, infos = it["sv"], it["infos"]
            nodes = [i for i, t in enumerate(infos) if t.is_node]
            for i, ok in enumerate(sv.sup):
                if not (ok and sv.head[i] > 0 and infos[sv.head[i] - 1].is_node):
                    continue
                j = sv.head[i] - 1
                pos.append(R.link_row(infos[j].pos, infos[j].subtypes, sv.rel[i], infos[i].pos, infos[i].subtypes, i < j, classes))
                if rng.random() < 0.5 and len(nodes) > 2:        # corrupted head
                    k = rng.choice([x for x in nodes if x not in (i, j)])
                    neg.append(R.link_row(infos[k].pos, infos[k].subtypes, sv.rel[i], infos[i].pos, infos[i].subtypes, i < k, classes))
                else:                                            # corrupted relation
                    r2 = rng.choice([r for r in rels if r != sv.rel[i]])
                    neg.append(R.link_row(infos[j].pos, infos[j].subtypes, r2, infos[i].pos, infos[i].subtypes, i < j, classes))
        return pos, neg

    dpos, dneg = links(by["dev"])
    sweep = []
    for n in SWEEP_N:
        tpr = np.mean([cnt.get(r, 0) >= n for r in dpos])
        fpr = np.mean([cnt.get(r, 0) >= n for r in dneg])
        sweep.append({"N": n, "dev_true_links_pass": round(float(tpr), 4), "dev_corrupted_links_pass": round(float(fpr), 4),
                      "youden_j": round(float(tpr - fpr), 4)})
    # declared rule: the largest N that still licenses >= 95% of the true dev links
    best = max((s for s in sweep if s["dev_true_links_pass"] >= 0.95), key=lambda s: s["N"])
    att.min_count = best["N"]
    json.dump(att.to_json(), open(OUT / "attested_configs.json", "w"))
    # (a) vs (b) agreement on the dev true links and on the corrupted ones
    agree = {}
    for name, rows in (("dev_true_links", dpos), ("dev_corrupted_links", dneg)):
        c = Counter()
        for r in rows:
            ga = gram.check(r[0], r[2], r[3], r[5])[0]
            gb = cnt.get(r, 0) >= best["N"]
            c[("a" if ga else "-") + ("b" if gb else "-")] += 1
        n = len(rows)
        agree[name] = {"n": n, "licensed_by_a": round(sum(v for k, v in c.items() if k[0] == "a") / n, 4),
                       "licensed_by_b": round(sum(v for k, v in c.items() if k[1] == "b") / n, 4),
                       "both": round(c["ab"] / n, 4), "a_only": round(c["a-"] / n, 4), "b_only": round(c["-b"] / n, 4),
                       "neither": round(c["--"] / n, 4)}
    res = {"min_count_chosen_on_dev": best["N"], "n_train_links": int(sum(cnt.values())), "n_distinct_configs": len(cnt),
           "classes": sorted(classes), "sweep": sweep, "agreement_a_vs_b": agree,
           "grammar_entries": len(gram.table.entries)}
    json.dump(res, open(OUT / "referee_attested_selection.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- shared context
class Ctx:
    def __init__(self):
        t0 = time.time()
        self.usvs = _sense_space()
        self.enc = E.NaomiEncoder.load(str(ENC / "main_best.pt"), self.usvs)
        self.orr = O.Orderer.load(str(OUT / "main_best.pt"), self.usvs)
        self.att = R.Attested.load(OUT / "attested_configs.json")
        self.ref = R.Referee(self.att, orderer=self.orr, encoder=self.enc, usvs=self.usvs)
        self.by = load_items()
        self.gold = {r["id"]: r for r in load_jsonl(GOLD)}
        print(f"context ready in {time.time() - t0:.0f}s", flush=True)

    def score(self, ref_tree, cand_tree):
        try:
            p = score_pair(ref_tree, cand_tree, self.usvs)
            return {"meaning": float(p.meaning), "shape": float(p.shape), "overall": float(p.overall)}
        except Exception:
            return None

    def roundtrip(self, tree, tail=()):
        """tree -> decoder -> text' -> encoder -> tree'. Two texts: the decoded words alone, and with the original's
        trailing punctuation tokens restored. Control: the TRUE-order words (same two variants) through the encoder."""
        nodes = self.orr.order_tree(tree)
        pos = [n.token_index for n in nodes]
        words = [n.token for n in nodes]
        real = sorted((n for n, _d, _p in O.walk(tree.root) if n.token_index is not None), key=lambda n: n.token_index)
        true_words = [n.token for n in real]
        contig = O.contiguous(O.tree_in(tree, self.orr.builder, self.orr.rel_ids))
        r = {"n": len(real), "contiguous": contig, "exact_order": pos == sorted(pos), "kendall": O.kendall_distance(pos),
             "text": " ".join(words + list(tail))}
        for name, ws in (("decoded", words), ("control", true_words)):
            r[name + "_nopunct"] = self.score(tree, self.enc.encode(" ".join(ws)))
            r[name + "_tail"] = self.score(tree, self.enc.encode(" ".join(list(ws) + list(tail))))
        return r


def tail_of(infos):
    k = len(infos)
    while k > 0 and not infos[k - 1].is_node:
        k -= 1
    return [t.token for t in infos[k:]]


def agg(rows, key):
    v = [r[key] for r in rows if r.get(key)]
    if not v:
        return None
    return {"n": len(v), "meaning": round(float(np.mean([x["meaning"] for x in v])), 2),
            "shape": round(float(np.mean([x["shape"] for x in v])), 2),
            "overall": round(float(np.mean([x["overall"] for x in v])), 2)}


def summarise(rows):
    return {"sentences": len(rows), "start_tree_subtrees_all_contiguous": round(float(np.mean([r["contiguous"] for r in rows])), 4) if rows else None,
            "exact_word_order_recovery": round(float(np.mean([r["exact_order"] for r in rows])), 4) if rows else None,
            "kendall_distance_mean": round(float(np.mean([r["kendall"] for r in rows])), 4) if rows else None,
            "decoded_text_no_punctuation": agg(rows, "decoded_nopunct"), "decoded_text_final_punctuation_restored": agg(rows, "decoded_tail"),
            "control_true_order_no_punctuation": agg(rows, "control_nopunct"), "control_true_order_final_punctuation_restored": agg(rows, "control_tail")}


# ----------------------------------------------------------------------------- part 4
def stage_roundtrip(a):
    cx = Ctx()
    test = cx.by["test"]
    if a.max_sents:
        test = test[::max(1, len(test) // a.max_sents)][:a.max_sents]
    rows = {"parser_connected": [], "encoder_on_connected": [], "encoder_on_fragmented": []}
    t0 = time.time()
    per = open(OUT / "roundtrip_per_sentence.jsonl", "w")
    for k, it in enumerate(test):
        rec = cx.gold[it["id"]]
        n = int(sum(t.is_node for t in it["infos"]))
        band = O.band_of(n)
        prop, _ = cx.enc.decode_infos(rec["text"], it["infos"])
        et = prop.tree
        if it["sv"].connected:
            r = cx.roundtrip(tree_from_dict(rec["tree"]), tail_of(it["infos"]))
            r["band"] = band
            rows["parser_connected"].append(r)
            r2 = cx.roundtrip(et, tail_of(it["infos"]))
            r2["band"] = band
            rows["encoder_on_connected"].append(r2)
            kind = ("parser_connected", "encoder_on_connected")
            per.write(json.dumps({"id": it["id"], "set": "connected", "band": band, "parser_tree": r, "encoder_tree": r2}) + "\n")
        else:
            r = cx.roundtrip(et, tail_of(it["infos"]))
            r["band"] = band
            rows["encoder_on_fragmented"].append(r)
            per.write(json.dumps({"id": it["id"], "set": "fragmented", "band": band, "encoder_tree": r}) + "\n")
        if k % 100 == 0:
            print(f"  {k}/{len(test)} {time.time() - t0:.0f}s", flush=True)
            per.flush()
    per.close()
    out = {}
    for name, rs in rows.items():
        out[name] = {"all": summarise(rs), "by_band": {b: summarise([r for r in rs if r["band"] == b])
                                                       for b in ("<=10", "11-20", "21-35", "36+") if any(r["band"] == b for r in rs)}}
    json.dump(out, open(OUT / ("eval_roundtrip.json" if not a.max_sents else "eval_roundtrip_small.json"), "w"), indent=1)
    print(json.dumps({k: v["all"] for k, v in out.items()}, indent=1))
    print(f"done {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- part 3
DECL_RT = 80.0


def quantiles_added(cx):
    """Fixed, judgment-free thresholds: quantiles of the encoder's confidence over ALL added links of test fragmented sentences."""
    ps, ms = [], []
    for it in cx.by["test"]:
        if it["sv"].connected:
            continue
        prop, logp = cx.enc.decode_infos("", it["infos"])
        for i in prop.heads:
            if it["sv"].unknown[i]:
                ps.append(float(np.exp(logp[i, prop.heads[i]])))
                ms.append(margin(logp, i))
    return {"p_q50": float(np.quantile(ps, .5)), "p_q75": float(np.quantile(ps, .75)),
            "m_q50": float(np.quantile(ms, .5)), "m_q75": float(np.quantile(ms, .75)), "n_added_links": len(ps)}


def margin(logp, i):
    row = np.sort(logp[i][logp[i] > E.NEG / 2])[::-1]
    return float(row[0] - row[1]) if len(row) > 1 else float(row[0])


def stage_part3(a):
    cx = Ctx()
    q = quantiles_added(cx)
    print("thresholds", q, flush=True)
    judg = json.load(open(ENC / "handcheck_judgments.json"))["judgments"]
    dump = {d["id"]: d for d in json.load(open(ENC / "handcheck_dump.json"))}
    items = {it["id"]: it for it in cx.by["test"]}
    links, sents = [], {}
    for sid, js in judg.items():
        it, rec = items[sid], cx.gold[sid]
        prop, logp = cx.enc.decode_infos(rec["text"], it["infos"])
        TL = {L["dep"]: L for L in R.tree_links(prop.tree, cx.att.classes)}
        rt = cx.roundtrip(prop.tree, tail_of(it["infos"]))
        sents[sid] = {"band": dump[sid]["band"], "rt_overall": rt["decoded_tail"]["overall"] if rt["decoded_tail"] else None,
                      "rt_exact": rt["exact_order"], "rt_kendall": rt["kendall"]}
        for dep_s, v in js.items():
            dep = int(dep_s)
            L = TL.get(dep)
            if L is None:                                   # the dep is the encoder's root: no link finding
                continue
            ok_a, why = cx.ref.grammar.check(L["row"][0], L["rel"], L["row"][3], L["side"])
            h = prop.heads[dep]
            links.append({"id": sid, "dep": dep, "judgment": v, "rel": L["rel"], "licensed_a": bool(ok_a), "why_a": why,
                          "count_b": cx.att.count(L["row"]), "p": float(np.exp(logp[dep, h])), "margin": margin(logp, dep),
                          "rt": sents[sid]["rt_overall"]})
    n_total = sum(len(j) for j in judg.values())
    print(f"links {len(links)} of {n_total} judged (rest are roots)", flush=True)
    N = cx.att.min_count
    S = {"a": lambda L: L["licensed_a"], "b": lambda L: L["count_b"] >= N,
         "a&b": lambda L: L["licensed_a"] and L["count_b"] >= N,
         "b&p>=q50": lambda L: L["count_b"] >= N and L["p"] >= q["p_q50"],
         "b&p>=q75": lambda L: L["count_b"] >= N and L["p"] >= q["p_q75"],
         "b&margin>=q50": lambda L: L["count_b"] >= N and L["margin"] >= q["m_q50"],
         "a&b&p>=q50": lambda L: L["licensed_a"] and L["count_b"] >= N and L["p"] >= q["p_q50"],
         "p>=q75": lambda L: L["p"] >= q["p_q75"],
         "p>=q50": lambda L: L["p"] >= q["p_q50"],
         "b&rt>=80": lambda L: L["count_b"] >= N and (L["rt"] or 0) >= DECL_RT,
         "b&p>=q50&rt>=80": lambda L: L["count_b"] >= N and L["p"] >= q["p_q50"] and (L["rt"] or 0) >= DECL_RT,
         "rt>=80": lambda L: (L["rt"] or 0) >= DECL_RT}
    mix = lambda ls: {"n": len(ls), **{k: sum(1 for L in ls if L["judgment"] == k) for k in "rdw"},  # noqa: E731
                      "wrong_rate": round(sum(1 for L in ls if L["judgment"] == "w") / len(ls), 3) if ls else None,
                      "wrong_or_debatable_rate": round(sum(1 for L in ls if L["judgment"] in "wd") / len(ls), 3) if ls else None}
    res = {"thresholds": q, "min_count_N": N, "declared_rt_threshold": DECL_RT, "n_links": len(links), "all_links": mix(links),
           "settings": {}, "sentences": sents}
    for name, f in S.items():
        ps = [L for L in links if f(L)]
        fl = [L for L in links if not f(L)]
        byS = defaultdict(list)
        for L in links:
            byS[L["id"]].append(L)
        acc = [sid for sid, ls in byS.items() if all(f(L) for L in ls)]
        inacc = [L for sid in acc for L in byS[sid]]
        res["settings"][name] = {"pass": mix(ps), "fail": mix(fl), "share_links_passing": round(len(ps) / len(links), 3),
                                 "sentences_accepted_all_links_pass": len(acc), "sentences_total": len(byS),
                                 "links_in_accepted_sentences": mix(inacc)}
    # descriptive quartiles of continuous signals (not thresholds)
    for key in ("p", "margin", "rt", "count_b"):
        vs = np.array([(L[key] if L[key] is not None else -1) for L in links], dtype=float)
        cuts = np.quantile(vs, [.25, .5, .75])
        bins = np.digitize(vs, cuts, right=True)
        res.setdefault("quartiles", {})[key] = {f"Q{b + 1}": mix([L for L, bb in zip(links, bins) if bb == b]) | {"range_upper": round(float(cuts[b]) if b < 3 else float(vs.max()), 3)}
                                                  for b in range(4)}
    res["by_relation_a_b"] = {}
    for rel in sorted({L["rel"] for L in links}):
        ls = [L for L in links if L["rel"] == rel]
        res["by_relation_a_b"][rel] = {"n": len(ls), "a_pass": sum(L["licensed_a"] for L in ls), "b_pass": sum(L["count_b"] >= N for L in ls)}
    res["links"] = links
    json.dump(res, open(OUT / "part3_referee_signals.json", "w"), indent=1)
    for name, s in res["settings"].items():
        print(f"{name:18s} pass {s['pass']} | fail {s['fail']['n']} (w {s['fail']['wrong_rate']}) | share {s['share_links_passing']} | "
              f"sent acc {s['sentences_accepted_all_links_pass']}/{s['sentences_total']} wrong-in-acc {s['links_in_accepted_sentences']['wrong_rate']} "
              f"(n {s['links_in_accepted_sentences']['n']})")
    print("all", res["all_links"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["attested", "part3", "roundtrip"])
    ap.add_argument("--max-sents", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    {"attested": stage_attested, "part3": stage_part3, "roundtrip": stage_roundtrip}[a.stage](a)


if __name__ == "__main__":
    main()
