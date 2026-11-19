"""Sentence-choice accuracy from collected decisions (works for v2b and v3 alike: sentence mass = summed exp(alog) per sentence;
for v3 alog is log p(sentence) + log p(node | sentence), so the mass is the sentence softmax).
    python s07_sentence_choice.py val s2b_all v3_s1"""
import sys, pickle
import numpy as np
from common import *

split, names = sys.argv[1], sys.argv[2:]
stories, _, meta = load_split(split)
for name in names:
    recs = pickle.load(open(OUTD / f"collect_{split}_{name}.pkl", "rb"))
    ok = {k: [] for k in (1, 2, 4)}
    ok_ans = []
    adj = {"+1": 0, "-1": 0, "other": 0}
    for r in recs:
        arr = stories[r["skey"]]
        best, acc = oracle_of(r)
        if best < 0.5 or len(r["alog"]) == 0:
            continue
        osents = {int(arr["sent"][g]) for g in acc}
        p = np.exp(r["alog"] - r["alog"].max())
        p = p / p.sum()
        m = np.zeros(arr["n_sent"])
        for g in range(len(p)):
            m[int(arr["sent"][g])] += p[g]
        order = np.argsort(-m, kind="stable")
        for k in ok:
            ok[k].append(bool(osents & set(order[:k].tolist())))
        if r["anchor"] >= 0:
            sa = int(arr["sent"][r["anchor"]])
            ok_ans.append(sa in osents)
            if sa not in osents:
                d = min(osents, key=lambda s: abs(s - sa)) - sa
                adj["+1" if d == 1 else "-1" if d == -1 else "other"] += 1
    print(f"{split} {name}: oracle>=0.5 items {len(ok[1])}; oracle sentence in top-1/2/4 sentences: " + " / ".join(f"{100*np.mean(ok[k]):.1f}%" for k in (1, 2, 4)) +
          f"; answered anchor in oracle sentence {100*np.mean(ok_ans):.1f}%; wrong sentence offset to nearest oracle sentence {adj}")
