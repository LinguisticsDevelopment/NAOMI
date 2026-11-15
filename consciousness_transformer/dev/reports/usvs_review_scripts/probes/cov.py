from evalib import *
import pickle, json
ok = pos != "?"; nwn = int(ok.sum()); gold = B["gold"][0]; gpos = B["gold"][1]
out = {}
def rep(name, flag):
    out[name] = dict(senses=int((flag & ok).sum()), share=round(float((flag & ok).sum() / nwn), 4), gold=round(float(flag[gold].mean()), 4),
                     by_pos={p: round(float(flag[pos == p].mean()), 3) for p in "nvar"}, gold_by_upos={u: round(float(flag[gold[gpos == u]].mean()), 3) for u in ("NOUN", "VERB", "ADJ", "ADV")})
# lookup coverage (senses that can be flagged as one half of a recorded opposite pair)
sim_tr = E["similar"][~split["similar"]]
head = np.arange(N)
for s, h in sim_tr: head[s] = h
for nm, ant in (("P1 lookup, train 80% antonym edges", E["ant"][~split["ant"]]), ("P1 lookup, all antonym edges", E["ant"])):
    f = np.zeros(N, bool); f[ant.ravel()] = True
    hs = set(ant.ravel().tolist()); f2 = f.copy()
    for s in range(N):
        if head[s] != s and head[s] in hs: f2[s] = True
    rep(nm + " (direct only)", f); rep(nm + " (+satellites via head)", f2)
# probe 2A valued senses
import p2lib
from p2lib import *
Wm, V, ch = build_A(W_IDF, 1, 1, nolex=True); rep("P2A senses with a non-default value (m=1)", (V != 1).any(1)); del Wm, V
Wm, V, ch = build_A(W_IDF, 3, 1, nolex=True); rep("P2A senses with a non-default value (m=3)", (V != 1).any(1)); del Wm, V
for nm in ("B1_per_pair", "B2_merged"):
    ent = pickle.load(open(P + f"entries_{nm}.pkl", "rb")); f = np.zeros(N, bool); f[list(ent.keys())] = True; rep(f"P2{nm} senses with >=1 pole entry (after L3)", f)
for nm, fn in (("P3 spectral hyp0.3 (giant comp)", "spec_hyp0.3.npy"), ("P3 spectral hyp0.05 (giant comp)", "spec_hyp0.05.npy"), ("P3 spectral hyp0.0 (giant comp)", "spec_hyp0.0.npy"), ("P3 fitted k=100 (train-degree>0)", "fit_k100.npy")):
    G = np.load(P + fn, mmap_mode="r"); f = np.zeros(N, bool)
    for i in range(0, N, 20000): f[i:i + 20000] = np.abs(np.asarray(G[i:i + 20000])).sum(1) > 0
    rep(nm, f)
# everyday flags
sids = B["sids"]; sidx = {s: i for i, s in enumerate(sids)}
antset_tr = set(map(tuple, E["ant"][~split["ant"]].tolist())); antset_te = set(map(tuple, E["ant"][split["ant"]].tolist()))
fl = []
for (a, b, sa, sb) in EV:
    i, j = sorted((sidx[sa], sidx[sb])); t = (i, j)
    fl.append("train" if t in antset_tr else "HELD-OUT" if t in antset_te else "not recorded")
out["everyday_flags"] = dict(zip(EVN, fl))
json.dump(out, open(P + "coverage.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
