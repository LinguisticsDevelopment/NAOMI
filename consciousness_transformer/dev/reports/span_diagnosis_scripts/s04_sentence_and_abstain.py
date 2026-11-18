"""Sentence selection (Q2/Q4 support) and abstentions on answerable items (Q6).   python s04_sentence_and_abstain.py val"""
import sys, pickle, collections
import numpy as np, torch
from common import *

split = sys.argv[1] if len(sys.argv) > 1 else "val"
stories, pairs, meta = load_split(split)
normal = {p["item"]: p for p in pairs if p["kind"] == "normal"}
dec = pickle.load(open(OUTD / f"decomp_{split}_s2b_all.pkl", "rb"))
colls = {m: {r["id"]: r for r in pickle.load(open(OUTD / f"collect_{split}_{m}.pkl", "rb"))} for m in ("s2b_all", "s1_all", "s1_short")}
# --- cheat sentence (IDF) from the stored eval rows
rows = {}
for line in open(RUN / f"eval_rows_{split}.jsonl"):
    r = json.loads(line); rows[r["id"]] = r


def sent_mass(rec, arr):
    S = arr["n_sent"]
    m = np.zeros(S)
    p = np.exp(rec["alog"] - rec["alog"].max()); p = p / p.sum()
    np.add.at(m, arr["sent"].astype(int), p)
    return m


BANDS = ["<=10", "11-20", "21+"]
acc = collections.defaultdict(lambda: collections.defaultdict(list))
len_rows = []
for d in dec:
    i = d["id"]
    if d["oracle"] < 0.5:
        continue
    p = normal[i]; arr = stories[p["skey"]]
    nr = colls["s2b_all"][i]["node_rouge"]
    best = nr.max(); accn = np.where(nr >= best - SLACK)[0]
    osents = {int(arr["sent"][g]) for g in accn}
    b = m2.band_of(d["ans_len"]) if d["ans_len"] is not None else "none"
    if b == "none": continue
    sb = p["sb"]
    S = arr["n_sent"]
    slen = np.bincount(arr["sent"].astype(int), minlength=S)
    cov = np.array([sb[arr["sent"] == s, 4][0] if (arr["sent"] == s).any() else 0 for s in range(S)])
    rank = np.array([sb[arr["sent"] == s, 6][0] if (arr["sent"] == s).any() else 9 for s in range(S)])
    cheat = rows[i]["cheat_text"]
    cheat_s = [k for k, t in enumerate(arr["sent_text"]) if t == cheat]
    sel = {"max coverage feature (sb4)": int(cov.argmax()), "best FIND rank feature (sb6)": int(rank.argmin()),
           "IDF best-overlap sentence (cheat)": cheat_s[0] if cheat_s else -1}
    for m, c in colls.items():
        sel[f"model {m}: argmax sentence mass"] = int(sent_mass(c[i], arr).argmax())
        if c[i]["anchor"] >= 0:
            sel[f"model {m}: chosen anchor's sentence"] = int(arr["sent"][c[i]["anchor"]])
    sel["random sentence (expected)"] = None
    for k, s in sel.items():
        if k.startswith("random"):
            acc[k][b].append(len(osents) / S)
        else:
            acc[k][b].append(float(s in osents))
    # length prior
    for m, c in colls.items():
        if c[i]["anchor"] >= 0:
            len_rows.append((m, b, int(slen[int(arr["sent"][c[i]["anchor"]])]), float(slen[list(osents)[0]]), float(slen.mean()),
                             float(np.mean(slen[:] <= slen[int(arr["sent"][c[i]["anchor"]])]))))
print("## Sentence-selection accuracy (oracle>=0.5 items; hit = the chosen sentence holds an oracle-acceptable anchor), by answer-sentence band")
print("| selector | " + " | ".join(BANDS) + " | all |")
for k, v in acc.items():
    allv = sum((v[b] for b in BANDS), [])
    print(f"| {k} | " + " | ".join(f"{100*mean(v[b]):.1f}% (n={len(v[b])})" for b in BANDS) + f" | {100*mean(allv):.1f}% |")
print("\n## Length prior: length (nodes) of the sentence the model's anchor lies in, vs the oracle sentence and the story mean (answered items, oracle>=0.5)")
print("| model | band | chosen len | oracle-sentence len | story mean sentence len | mean percentile of chosen length within story |")
for m in colls:
    for b in BANDS:
        rs = [r for r in len_rows if r[0] == m and r[1] == b]
        if rs:
            print(f"| {m} | {b} | {np.mean([r[2] for r in rs]):.1f} | {np.mean([r[3] for r in rs]):.1f} | {np.mean([r[4] for r in rs]):.1f} | {np.mean([r[5] for r in rs]):.2f} |")

# --- Q6 abstentions
print("\n## Q6: abstentions on answerable items (oracle>=0.5)")
model = T.load_model(str(RUN / "thinker_s2b_all_best.pt"))[0]
tab = m2.SenseTable()
ab = [d for d in dec if d["oracle"] >= 0.5]
n_ab = [d for d in ab if d["anchor"] < 0]
print("answerable items", len(ab), "abstained", len(n_ab), "share %.3f" % (len(n_ab) / len(ab)))
# forced answer for the abstained items: anchor = argmax alog, keep from the model
ids = [d["id"] for d in n_ab]
forced = {}
with torch.no_grad():
    for k in range(0, len(ids), 32):
        ch = [normal[i] for i in ids[k:k + 32]]
        b = m2.collate(ch, stories, tab, with_targets=False)
        e = model.encode(b)
        anch = []
        for j in range(len(ch)):
            o0, o1 = int(b["s_off"][j]), int(b["s_off"][j + 1])
            anch.append(int(e["alog"][o0:o1].argmax()) if o1 > o0 else -1)
        kept, _ = m2.keep_sets(model, e, b, stories, ch, anch)
        for p, a, kp in zip(ch, anch, kept):
            txt = m2.true_order_text(stories[p["skey"]], kp) if a >= 0 else ""
            forced[p["item"]] = m2.rouge_multi(txt, meta[p["item"]]["golds"])
print("forced-answer ROUGE on abstained answerable items: mean %.3f (abstain scores 0; oracle mean %.3f); abstain-reward in training 0.10" % (
    mean(forced.values()), mean(d["oracle"] for d in n_ab)))
print("  of those, forced answer > 0.10: %.2f, > 0.3: %.2f" % (mean(v > 0.10 for v in forced.values()), mean(v > 0.3 for v in forced.values())))
# features
def feats(d):
    p = normal[d["id"]]; arr = stories[p["skey"]]; c = colls["s2b_all"][d["id"]]
    itf = p["itf"]
    S = arr["n_sent"]
    slen = np.bincount(arr["sent"].astype(int), minlength=S)
    al = c["alog"]
    top = np.sort(al)[::-1]
    return {"p_abstain": c["p_abstain"], "N nodes": c["N"], "n_sent": S, "mean sentence len": float(slen.mean()),
            "answer-sentence len": d["ans_len"] if d["ans_len"] is not None else np.nan,
            "itf cov.max (best sentence q-coverage)": float(itf[1]), "itf find.max": float(itf[2]), "itf n question content words/10": float(itf[3]),
            "top anchor logit": float(top[0]) if len(top) else np.nan, "top1-top2 logit gap": float(top[0] - top[1]) if len(top) > 1 else np.nan,
            "logsumexp anchor logits": float(np.log(np.exp(al - al.max()).sum()) + al.max()) if len(al) else np.nan,
            "question tokens": len(d["q_parent"]), "wh found (1/0)": float(d["wh"] >= 0), "oracle ROUGE": d["oracle"],
            "max node sb1 (same-lemma content)": float(p["sb"][:, 1].max()) if p["sb"].size else 0.0,
            "n story nodes with same-lemma-as-q (sb0)": float((p["sb"][:, 0] > 0).sum())}
F = {d["id"]: feats(d) for d in ab}
def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort() + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
print("\n| feature | mean abstained | mean answered | AUC (abstained higher) |")
for k in next(iter(F.values())):
    a = [F[d["id"]][k] for d in n_ab]; b = [F[d["id"]][k] for d in ab if d["anchor"] >= 0]
    print(f"| {k} | {np.nanmean(a):.3f} | {np.nanmean(b):.3f} | {auc(a, b):.2f} |")
print("\nabstained answerable items by group:")
for key in ("attribute", "explicit", "local"):
    c1 = collections.Counter(d[key] for d in n_ab); c2 = collections.Counter(d[key] for d in ab)
    print(key, {k: f"{c1[k]}/{c2[k]} ({100*c1[k]/c2[k]:.0f}%)" for k in c2})
print("by band:", {b: f"{sum(1 for d in n_ab if m2.band_of(d['ans_len'])==b)}/{sum(1 for d in ab if d['ans_len'] is not None and m2.band_of(d['ans_len'])==b)}" for b in BANDS})
# calibration: p_abstain deciles vs forced-answer value on ALL answerable items? forced for all is costly; use answered items' actual f
print("\nall items (incl. oracle<0.5): abstain rate by p_abstain; answered ROUGE by p_abstain bin (answered items only)")
bins = [0, 1e-6, 1e-3, 0.05, 0.2, 0.5, 1.0001]
for lo, hi in zip(bins[:-1], bins[1:]):
    rs = [d for d in dec if lo <= d["p_abstain"] < hi]
    ans = [d for d in rs if d["anchor"] >= 0]
    print(f"| p_abstain [{lo:g},{hi:g}) | n {len(rs)} | answered {len(ans)} | mean ROUGE of answered {mean(d['f'] for d in ans):.3f} | mean oracle {mean(d['oracle'] for d in rs):.3f} |")
