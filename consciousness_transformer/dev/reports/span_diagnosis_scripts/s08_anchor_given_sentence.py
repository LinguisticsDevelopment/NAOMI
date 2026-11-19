"""Step 2 alone: with the ORACLE sentence forced, does the model's best node in it hit an acceptable anchor?  By answer-sentence band.
This is the part the shape features are meant to carry; it isolates it from the sentence choice (which the short-only training biases to short sentences).
    python s08_anchor_given_sentence.py val s1_short s1_all v3_short v3_s1"""
import sys, pickle
import numpy as np
from common import *

split, names = sys.argv[1], sys.argv[2:]
stories, _, meta = load_split(split)
print("| model | <=10 top1 acceptable / node ROUGE (n) | 11-20 | 21+ |")
for name in names:
    recs = pickle.load(open(OUTD / f"collect_{split}_{name}.pkl", "rb"))
    by = {b: [] for b in ("<=10", "11-20", "21+")}
    for r in recs:
        arr = stories[r["skey"]]
        best, acc = oracle_of(r)
        if best < 0.5 or len(r["alog"]) == 0 or r["ans_len"] is None:
            continue
        s_star = int(arr["sent"][int(r["node_rouge"].argmax())])
        idx = np.where(arr["sent"] == s_star)[0]
        top = int(idx[np.argmax(r["alog"][idx])])
        by[m2.band_of(r["ans_len"])].append((float(r["node_rouge"][top] >= best - SLACK), float(r["node_rouge"][top])))
    print(f"| {name} | " + " | ".join(f"{100*np.mean([x for x, _ in by[b]]):.1f}% / {np.mean([y for _, y in by[b]]):.3f} ({len(by[b])})" for b in by) + " |")
