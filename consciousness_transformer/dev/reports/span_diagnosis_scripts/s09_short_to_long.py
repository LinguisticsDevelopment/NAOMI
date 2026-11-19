"""Why short-trained models fail on long answer sentences, after length-invariant features.  python s09_short_to_long.py val v3_short v3_s1 ...
(1) sentence choice by band + the length of the sentence chosen vs the gold sentence + within-item correlation of sentence mass with sentence length;
(2) with the oracle sentence forced: depth and share-of-sentence size of the model's best node vs the oracle anchor, by band."""
import sys, pickle
import numpy as np
from common import *

split, names = sys.argv[1], sys.argv[2:]
stories, _, meta = load_split(split)


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


for name in names:
    recs = pickle.load(open(OUTD / f"collect_{split}_{name}.pkl", "rb"))
    acc_b = {}; chosen_len = {}; gold_len = {}; corr = []; fdep = {}; fsh = {}; odep = {}; osh = {}
    for r in recs:
        arr = stories[r["skey"]]
        best, acc = oracle_of(r)
        if best < 0.5 or len(r["alog"]) == 0:
            continue
        sent = np.asarray(arr["sent"]); ns = arr["n_sent"]
        slen = np.bincount(sent, minlength=ns).astype(float)
        p = np.exp(r["alog"] - r["alog"].max()); p /= p.sum()
        m = np.zeros(ns)
        np.add.at(m, sent, p)
        osents = {int(sent[g]) for g in acc}
        band = m2.band_of(r["ans_len"]) if r["ans_len"] is not None else "none"
        top = int(np.argmax(m))
        acc_b.setdefault(band, []).append(top in osents)
        chosen_len.setdefault(band, []).append(slen[top])
        gold_len.setdefault(band, []).append(np.mean([slen[s] for s in osents]))
        if ns > 3 and slen.std() > 0 and m.std() > 0:
            corr.append(np.corrcoef(rank(np.log(m + 1e-12)), rank(slen))[0, 1])
        # step 2 with the oracle sentence forced
        sel = np.where(np.isin(sent, list(osents)))[0]
        if len(sel):
            j = sel[np.argmax(r["alog"][sel])]
            sz = np.asarray(arr["size"], float)
            for d_, s_, g in ((fdep, fsh, j), (odep, osh, int(acc[np.argmax(np.asarray(arr["size"])[acc])]))):
                d_.setdefault(band, []).append(arr["depth"][g])
                s_.setdefault(band, []).append(sz[g] / max(slen[sent[g]], 1))
    print(f"\n== {split} {name}  within-item Spearman(sentence mass, sentence length) mean {np.mean(corr):+.3f} (n={len(corr)})")
    print("| band | n | sentence top-1 hit | chosen sentence nodes | gold sentence nodes | forced-sentence: model depth / oracle depth | model size share / oracle size share |")
    for b in ("<=10", "11-20", "21+"):
        if b in acc_b:
            print(f"| {b} | {len(acc_b[b])} | {100*np.mean(acc_b[b]):.1f}% | {np.mean(chosen_len[b]):.1f} | {np.mean(gold_len[b]):.1f} | {np.mean(fdep[b]):.2f} / {np.mean(odep[b]):.2f} | {np.mean(fsh[b]):.2f} / {np.mean(osh[b]):.2f} |")
