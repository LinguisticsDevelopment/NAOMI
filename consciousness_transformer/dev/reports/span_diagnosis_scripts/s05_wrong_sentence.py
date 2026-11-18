"""Why is the sentence wrong? Is there lexical/USVS signal toward the oracle sentence that the model ignores?  python s05_wrong_sentence.py val"""
import sys, pickle, collections
import numpy as np
from common import *
split = sys.argv[1] if len(sys.argv) > 1 else "val"
stories, pairs, meta = load_split(split)
normal = {p["item"]: p for p in pairs if p["kind"] == "normal"}
dec = pickle.load(open(OUTD / f"decomp_{split}_s2b_all.pkl", "rb"))
col = {r["id"]: r for r in pickle.load(open(OUTD / f"collect_{split}_s2b_all.pkl", "rb"))}
rowsE = {}
for line in open(RUN / f"eval_rows_{split}.jsonl"):
    r = json.loads(line); rowsE[r["id"]] = r
res = collections.defaultdict(list)
rk = collections.Counter()
for d in dec:
    if d["oracle"] < 0.5 or d["anchor"] < 0:
        continue
    i = d["id"]; p = normal[i]; arr = stories[p["skey"]]; c = col[i]
    nr = c["node_rouge"]; acc = np.where(nr >= nr.max() - SLACK)[0]
    osents = {int(arr["sent"][g]) for g in acc}
    S = arr["n_sent"]; sb = p["sb"]
    cov = np.array([sb[arr["sent"] == s, 4].max() if (arr["sent"] == s).any() else 0 for s in range(S)])
    sim = np.array([sb[arr["sent"] == s, 2].max() if (arr["sent"] == s).any() else 0 for s in range(S)])
    m = np.zeros(S); pr = np.exp(c["alog"] - c["alog"].max()); pr /= pr.sum(); np.add.at(m, arr["sent"].astype(int), pr)
    order = list(np.argsort(-m))
    orank = min(order.index(s) for s in osents)
    ok = int(arr["sent"][d["anchor"]]) in osents
    ms = int(arr["sent"][d["anchor"]])
    ocov = max(cov[s] for s in osents)
    res["items"].append(1)
    res["ok"].append(ok)
    res["orank"].append(orank)
    res["oracle_sent_cov0"].append(ocov == 0)
    res["oracle_sent_cov_ge_chosen"].append(ocov >= cov[ms])
    res["oracle_is_maxcov"].append(ocov >= cov.max() - 1e-9)
    res["chosen_is_maxcov"].append(cov[ms] >= cov.max() - 1e-9)
    res["ref_missing"].append(rowsE[i]["ref_missing"] if i in rowsE else np.nan)
    res["wrong"].append(not ok)
    res["maxcov_tied"].append(int((cov >= cov.max() - 1e-9).sum()))
n = len(res["ok"])
ok = np.array(res["ok"]); wr = ~ok
print("answered answerable items", n, "wrong-sentence share", wr.mean())
for k in ("oracle_sent_cov0", "oracle_sent_cov_ge_chosen", "oracle_is_maxcov", "chosen_is_maxcov"):
    a = np.array(res[k], dtype=float)
    print(f"{k}: among wrong-sentence {a[wr].mean():.3f}  among right-sentence {a[ok].mean():.3f}")
o = np.array(res["orank"])
print("rank of oracle sentence in model sentence mass: wrong-sentence items: rank1 (2nd) %.2f, rank<=2 %.2f, rank<=4 %.2f; mean %.2f" % ((o[wr] == 1).mean(), (o[wr] <= 2).mean(), (o[wr] <= 4).mean(), o[wr].mean()))
rm = np.array(res["ref_missing"], dtype=float)
print("share of gold content words NOT in the story or question (needs new words): wrong %.3f right %.3f" % (np.nanmean(rm[wr]), np.nanmean(rm[ok])))
print("median # sentences tied at max coverage: wrong", np.median(np.array(res["maxcov_tied"])[wr]), "right", np.median(np.array(res["maxcov_tied"])[ok]))
# how is the oracle sentence related to the question (adjacent to the max-coverage sentence?)
print("\nwrong-sentence items: signed offset (oracle sentence - model sentence) histogram")
off = collections.Counter()
for d in dec:
    if d["oracle"] < 0.5 or d["anchor"] < 0 or d["cat"] != "wrong_sentence": continue
    i = d["id"]; arr = stories[normal[i]["skey"]]; c = col[i]
    nr = c["node_rouge"]; acc = np.where(nr >= nr.max() - SLACK)[0]
    osents = sorted({int(arr["sent"][g]) for g in acc}); ms = int(arr["sent"][d["anchor"]])
    o_ = min(osents, key=lambda s: abs(s - ms)) - ms
    off[max(-4, min(4, o_))] += 1
print(sorted(off.items()))
# by question group: sentence accuracy
