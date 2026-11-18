"""Within-item correlation of the model's per-sentence anchor mass with sentence length and with question coverage.  python s06_length_prior.py val"""
import sys, pickle
import numpy as np
from common import *
split = sys.argv[1] if len(sys.argv) > 1 else "val"
stories, pairs, meta = load_split(split)
normal = {p["item"]: p for p in pairs if p["kind"] == "normal"}
def sp(a, b):
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])
print("| model | spearman(mass, sentence length) | spearman(mass, q-coverage sb4) | spearman(mass, sentence index) | n items |")
for m in ("s1_short", "s1_all", "s2b_all"):
    col = pickle.load(open(OUTD / f"collect_{split}_{m}.pkl", "rb"))
    a, b, c = [], [], []
    for r in col:
        if r["best_f1"] < 0.5: continue
        p = normal[r["id"]]; arr = stories[p["skey"]]; S = arr["n_sent"]
        if S < 4: continue
        mass = np.zeros(S); pr = np.exp(r["alog"] - r["alog"].max()); pr /= pr.sum(); np.add.at(mass, arr["sent"].astype(int), pr)
        slen = np.bincount(arr["sent"].astype(int), minlength=S).astype(float)
        cov = np.array([p["sb"][arr["sent"] == s, 4].max() if (arr["sent"] == s).any() else 0 for s in range(S)])
        a.append(sp(mass, slen)); b.append(sp(mass, cov)); c.append(sp(mass, np.arange(S, dtype=float)))
    print(f"| {m} | {np.nanmean(a):+.2f} | {np.nanmean(b):+.2f} | {np.nanmean(c):+.2f} | {len(a)} |")
# the feature set has no sentence index / neighbour input: confirm from the batch tensors
import inspect
src = inspect.getsource(m2.Thinker.encode)
print("\nThinker.encode mentions s_sent only for pooling:", src.count("s_sent"), "uses; any sentence-index scalar in NS scalars?", "no (NS=6: depth, log size, token position within sentence, is_function, is_name, has key)")
