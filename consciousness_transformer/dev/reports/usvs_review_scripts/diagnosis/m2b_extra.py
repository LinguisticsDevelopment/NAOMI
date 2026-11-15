from scorecard_lib import *
from nltk.corpus import wordnet as wn
u = load(); C = u.core_coords.astype(np.float32); M = dense_senses(u); w = idf_from(M)
P = pickle.load(open(S + "/pairs.pkl", "rb")); sets = build_sets(u, C, M, P)
sidx = {s: i for i, s in enumerate(u.sense_ids)}
out = {}
# sense-level similar_to (near-synonym) pairs
sim = []
for s in wn.all_synsets("s"):
    for h in s.similar_tos(): sim.append((sidx[s.name()], sidx[h.name()]))
A, B = M[[a for a, b in sim]], M[[b for a, b in sim]]
for nm, ww in (("raw", None), ("idf", w)):
    cs = wcos(A, B, ww); rn = cos_sets({"r": sets["sense"]["rand_a"]}, ww)["r"]
    ca = cos_sets({"x": sets["sense"]["ant"]}, ww)["x"]
    out[f"simto|{nm}"] = {"n": len(sim), "mean": float(cs.mean()), "median": float(np.median(cs)), "AUC_vs_rand_adj": auc(cs, rn), "AUC_simto_vs_ant(P simto>ant)": auc(cs, ca), "rand_adj_mean": float(rn.mean()), "ant_mean": float(ca.mean())}
# hyp vs cohyp by POS & level
for lvl in ("core", "sense"):
    for nm, ww in (("raw", None), ("idf", w)):
        cs = cos_sets({k: sets[lvl][k] for k in ("hyp", "cohyp")}, ww)
        d = {}
        for k in ("hyp", "cohyp"):
            for p in "nv":
                sel = np.array([q == p for q in sets[lvl][k][2]]); d[f"{k}_{p}_mean"] = float(cs[k][sel].mean()); d[f"{k}_{p}_n"] = int(sel.sum())
        out[f"hyp_cohyp_by_pos|{lvl}|{nm}"] = d
# sparse core words: pos of MFS
nc = (np.abs(C) > 1e-6).sum(1)
import collections
cnt = collections.Counter(); tot = collections.Counter()
for i, wd in enumerate(u.core_words):
    s = u.senses_of(wd)
    p = (s[0].rsplit(".", 2)[1] if s else "?"); p = "a" if p == "s" else p
    tot[p] += 1; cnt[p] += int(nc[i] <= 10)
out["sparse_core_by_pos"] = {p: {"n": tot[p], "frac_le10_axes": cnt[p] / tot[p]} for p in tot}
# cosine of sparse words to everything
sp = nc <= 10
out["core_sparse_vs_dense_random_pair_mean_cos"] = {}
rng = np.random.default_rng(0)
i = rng.integers(0, len(nc), 20000); j = rng.integers(0, len(nc), 20000)
cs = wcos(C[i], C[j])
for nm, sel in (("both_dense", ~sp[i] & ~sp[j]), ("both_sparse", sp[i] & sp[j]), ("mixed", sp[i] ^ sp[j])):
    out["core_sparse_vs_dense_random_pair_mean_cos"][nm] = {"n": int(sel.sum()), "mean": float(cs[sel].mean())}
# pair coverage: antonym pairs where both are dense/sparse
ant = P["ant"]; core = {x: k for k, x in enumerate(u.core_words)}
d = collections.Counter((bool(sp[core[a]]), bool(sp[core[b]])) for a, b, *_ in ant)
out["ant737_sparse_flags"] = {str(k): v for k, v in d.items()}
# top-1 nearest neighbours check: for 200 random dense noun words: do nearest neighbours (core cosine) include WordNet sister/synonym? skip
json.dump(out, open(S + "/m2b_extra.json", "w"), indent=1)
print(json.dumps(out, indent=1))
