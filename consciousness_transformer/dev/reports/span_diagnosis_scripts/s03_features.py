"""Q4: thinker input features by answer-sentence length. Reads the stored stories/pairs only (no model).  python s03_features.py"""
import pickle, collections
import numpy as np
from common import *

stories = pickle.load(open(RUN / "stories.pkl", "rb"))
BAND = ["<=10", "11-20", "21+"]
SB = ["sb0 same-lemma-as-q", "sb1 same-lemma (content q)", "sb2 USVS sim to q", "sb3 idf-weighted ident", "sb4 sentence q-coverage", "sb5 FIND sim of sentence", "sb6 FIND rank of sentence (0=best)"]


def per_item(pairs, meta, want_best=True):
    rows = []
    for p in pairs:
        if p["kind"] != "normal":
            continue
        t = p["tgt"]
        if t["ans_sent"] is None or not t["anchors"] or t["best_f1"] < m2.ACCEPT_F1:
            continue
        arr = stories[p["skey"]]
        s = t["ans_sent"]
        idx = np.where(arr["sent"] == s)[0]
        a0, kept, _ = t["anchors"][0]
        sb = p["sb"]
        # oracle (lemma-F1) anchor = the largest-size accepted anchor in the answer sentence if any, else first
        accs = [a for a, k, f in t["anchors"]]
        a = a0
        sl = len(idx)
        cov = sb[:, 4]
        row = {"band": m2.band_of(sl), "ans_len": sl, "N": arr["parent"].size, "n_sent": arr["n_sent"],
               "sent_pos": s / max(arr["n_sent"] - 1, 1), "nodes_per_sent": arr["parent"].size / arr["n_sent"],
               # answer-sentence level (what the sentence choice sees)
               "ans_sb4_cov": float(cov[idx[0]]), "ans_sb4_cov_over_max": float(cov[idx[0]] / max(cov.max(), 1e-8)),
               "ans_sb5_find": float(sb[idx[0], 5]), "ans_sb6_findrank": float(sb[idx[0], 6]),
               "ans_is_top_cov": float(cov[idx[0]] >= cov.max() - 1e-9), "ans_is_top_find": float(sb[idx[0], 6] == sb[:, 6].min()),
               # node level in the answer sentence (what the anchor choice sees)
               "ans_mean_depth": float(arr["depth"][idx].mean()), "ans_max_depth": float(arr["depth"][idx].max()),
               "ans_max_logsize": float(np.log1p(arr["size"][idx]).max()),
               "ans_n_ident_nodes": float((sb[idx, 0] > 0).sum()), "ans_share_ident_nodes": float((sb[idx, 0] > 0).mean()),
               "ans_n_ident_content": float((sb[idx, 1] > 0).sum()),
               "ans_max_sb2_sim": float(sb[idx, 2].max()),
               # the target anchor / kept set
               "anc_depth": float(arr["depth"][a]), "anc_logsize": float(np.log1p(arr["size"][a])), "anc_size": float(arr["size"][a]),
               "anc_size_over_sent": float(arr["size"][a] / sl), "anc_is_head": float(arr["depth"][a] <= 1),
               "anc_relpos": float(arr["scal"][a][2]), "anc_sb1": float(sb[a, 1]), "anc_sb2": float(sb[a, 2]),
               "kept_n": float(len(kept)), "kept_over_sent": float(len(kept) / sl),
               "kept_max_reldepth": float(max(arr["depth"][g] - arr["depth"][a] for g in kept)),
               "kept_max_logsize": float(max(np.log1p(arr["size"][g]) for g in kept)),
               "n_anchors_accepted": float(len(accs)),
               # distractors: how many other-sentence nodes look as matching
               "ident_nodes_elsewhere": float((sb[:, 1] > 0).sum() - (sb[idx, 1] > 0).sum())}
        rows.append(row)
    return rows


meta = json.load(open(RUN / "items_meta.json"))
tr = pickle.load(open(RUN / "pairs_train.pkl", "rb"))
va = pickle.load(open(RUN / "pairs_val.pkl", "rb"))
rtr, rva = per_item(tr, meta), per_item(va, meta)
print("reachable items with an answer sentence: train", len(rtr), "val", len(rva))
print("train counts by band", collections.Counter(r["band"] for r in rtr), "val", collections.Counter(r["band"] for r in rva))
keys = [k for k in rtr[0] if k not in ("band",)]
print("\n## Feature means by answer-sentence band (reachable items; train+val pooled), standardized shift 21+ vs <=10 (Cohen d, pooled sd), share of 21+ values outside the range seen in <=10 TRAIN items")
allr = rtr + rva
short_tr = [r for r in rtr if r["band"] == "<=10"]
print("| feature | <=10 | 11-20 | 21+ | d(21+ vs <=10) | 21+ outside short-train [min,max] | 21+ above short-train p95 |")
res = []
for k in keys:
    g = {b: np.array([r[k] for r in allr if r["band"] == b], dtype=float) for b in BAND}
    sd = np.sqrt((g["<=10"].var() * len(g["<=10"]) + g["21+"].var() * len(g["21+"])) / (len(g["<=10"]) + len(g["21+"])))
    d = (g["21+"].mean() - g["<=10"].mean()) / max(sd, 1e-9)
    st = np.array([r[k] for r in short_tr], dtype=float)
    lo, hi, p95 = st.min(), st.max(), np.percentile(st, 95)
    out_ = float(((g["21+"] < lo) | (g["21+"] > hi)).mean())
    above = float((g["21+"] > p95).mean())
    res.append((k, g["<=10"].mean(), g["11-20"].mean(), g["21+"].mean(), d, out_, above))
for k, a, b, c, d, o, ab in sorted(res, key=lambda x: -abs(x[4])):
    print(f"| {k} | {a:.3f} | {b:.3f} | {c:.3f} | {d:+.2f} | {100*o:.1f}% | {100*ab:.1f}% |")
# node-level feature ranges for the keep head and anchor head, all nodes of answer sentences
print("\n## Keep-head inputs for nodes inside the target kept subtree (relative depth, log size): distribution by band")
for b in BAND:
    rd, ls = [], []
    for p in tr + va:
        if p["kind"] != "normal": continue
        t = p["tgt"]
        if t["ans_sent"] is None or not t["anchors"] or t["best_f1"] < m2.ACCEPT_F1: continue
        arr = stories[p["skey"]]
        sl = int((arr["sent"] == t["ans_sent"]).sum())
        if m2.band_of(sl) != b: continue
        a, kept, _ = t["anchors"][0]
        for g in kept:
            rd.append((arr["depth"][g] - arr["depth"][a]) / 5.0); ls.append(np.log1p(arr["size"][g]) / 4.0)
    rd, ls = np.array(rd), np.array(ls)
    print(f"| {b} | kept nodes {len(rd)} | rel-depth mean {rd.mean():.2f} p95 {np.percentile(rd,95):.2f} max {rd.max():.2f} | log-size/4 mean {ls.mean():.2f} p95 {np.percentile(ls,95):.2f} max {ls.max():.2f} |")
# scalars of ALL nodes in answer sentence (anchor head's x input): depth/8, log size/4
print("\n## Anchor-head candidate nodes (all nodes of the answer sentence), by band: depth/8 and log1p(size)/4")
for b in BAND:
    dd, ss, nn = [], [], []
    for p in tr + va:
        if p["kind"] != "normal": continue
        t = p["tgt"]
        if t["ans_sent"] is None or t["best_f1"] < m2.ACCEPT_F1: continue
        arr = stories[p["skey"]]
        idx = np.where(arr["sent"] == t["ans_sent"])[0]
        if m2.band_of(len(idx)) != b: continue
        dd += list(arr["scal"][idx, 0]); ss += list(arr["scal"][idx, 1]); nn.append(len(idx))
    dd, ss = np.array(dd), np.array(ss)
    print(f"| {b} | items {len(nn)} | candidate nodes/sentence {np.mean(nn):.1f} | depth/8 mean {dd.mean():.2f} p95 {np.percentile(dd,95):.2f} max {dd.max():.2f} | logsize/4 mean {ss.mean():.2f} max {ss.max():.2f} |")
print("\n## Story-level (same for all bands because the passage is the whole story section): nodes per story / sentences per story")
for b in BAND:
    print(b, "N mean %.0f, n_sent mean %.1f" % (np.mean([r["N"] for r in allr if r["band"] == b]), np.mean([r["n_sent"] for r in allr if r["band"] == b])))
pickle.dump({"train": rtr, "val": rva}, open(OUTD / "features_items.pkl", "wb"))
