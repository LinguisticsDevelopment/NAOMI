from glib import *
import json, sys
from nltk.corpus import wordnet as wn
sids = B["sids"]
Mf = np.asarray(M, dtype=np.float32); W = Mf * np.sqrt(IDF).astype(np.float32)
Wu = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
def lex(i):
    try: return wn.synset(sids[i]).lexname()
    except Exception: return "?"
def short(i): return sids[i].rsplit(".", 2)[0] + "." + sids[i].rsplit(".", 2)[1] + sids[i].rsplit(".", 2)[2].lstrip("0")
def analyse(name, G, dims, min_abs=0.0):
    placed = np.abs(G).sum(1) > 0; idx = np.where(placed)[0]; rng = np.random.default_rng(0)
    out = []
    for d in dims:
        col = G[idx, d]; order = np.argsort(col, kind="stable")
        lo = idx[order[:12]]; hi = idx[order[::-1][:12]]
        rec = dict(dim=int(d), hi=[short(i) for i in hi], lo=[short(i) for i in lo])
        for tag, grp in (("hi", hi), ("lo", lo)):
            lx = [lex(i) for i in grp]; c = max(set(lx), key=lx.count)
            rec[f"{tag}_lex"] = f"{c} {lx.count(c)}/12"
            X = Wu[grp]; sim = X @ X.T; rec[f"{tag}_coh"] = round(float((sim.sum() - 12) / (12 * 11)), 3)
            # intruder test in USVS space: 10 top words + 1 random placed word (n=100 trials: random 10-subset of top 12)
            hit = 0
            for t in range(100):
                sub = rng.choice(grp, 10, replace=False); intr = idx[rng.integers(len(idx))]
                cand = np.append(sub, intr); S2 = Wu[cand] @ Wu[cand].T; m = (S2.sum(1) - 1) / 10
                hit += int(np.argmin(m) == 10)
            rec[f"{tag}_intruder_hit"] = hit / 100
        out.append(rec)
    return out
res = {}
G = np.load(P + "spec_hyp0.05.npy")[:, :200]
res["spec_hyp0.05_first15"] = analyse("spec", G, range(15))
Z = np.load(P + "fit_k50.npy"); pl = np.abs(Z).sum(1) > 0
var = Z[pl].var(0); dims = np.argsort(-var)[:15]
res["fit_k50_top15_by_var"] = analyse("fit", Z, dims)
json.dump(res, open(P + "p3names.json", "w"), indent=1)
for k, v in res.items():
    print("==", k, "mean coh hi/lo", np.mean([r['hi_coh'] for r in v]).round(3), np.mean([r['lo_coh'] for r in v]).round(3), "intruder hit", np.mean([r['hi_intruder_hit'] for r in v] + [r['lo_intruder_hit'] for r in v]).round(3))
    for r in v:
        print(r["dim"], "HI", r["hi_lex"], r["hi_coh"], r["hi_intruder_hit"], " | LO", r["lo_lex"], r["lo_coh"], r["lo_intruder_hit"])
        print("   HI:", " ".join(r["hi"])); print("   LO:", " ".join(r["lo"]))
