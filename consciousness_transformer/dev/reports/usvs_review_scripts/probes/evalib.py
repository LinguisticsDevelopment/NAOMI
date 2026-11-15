import sys, pickle, json, os
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_diag")
from common import auc, _rank
P = "/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_probes/"
B = pickle.load(open(P + "base.pkl", "rb"))
M = np.load(P + "M.npy", mmap_mode="r")
M = np.asarray(M)
pos, E, split, S, EV, axes = B["pos"], B["E"], B["split"], B["S"], B["EV"], B["axes"]
N, D = M.shape
df = (M != 0).sum(0); IDF = np.log((N + 1.0) / (df + 1.0))
ORDER = ["ant_te", "sim_te", "syn_mfs", "hyp_te", "sis_te", "rand", "rand_n", "rand_v", "rand_a", "rand_r", "everyday", "simlex"]
OFF = {}; _o = 0
for k in ORDER: OFF[k] = (_o, _o + len(S[k])); _o += len(S[k])
ALLP = np.concatenate([S[k] for k in ORDER])
def spearman(x, y):
    return float(np.corrcoef(_rank(np.asarray(x, float)), _rank(np.asarray(y, float)))[0, 1])
f4 = lambda x: round(float(x), 4)
def evaluate(simfn, placed=None):
    """simfn: (Nx2 int array) -> N similarities. placed: optional bool[N]; if given only pairs with both ends placed are scored
    (adds cov_<set> = share of pairs kept)."""
    Sx = {}; keepmask = {}
    for k in ORDER:
        a = S[k]
        km = np.ones(len(a), bool) if (placed is None or k == 'everyday') else (placed[a[:, 0]] & placed[a[:, 1]])
        Sx[k] = a[km]; keepmask[k] = km
    allp = np.concatenate([Sx[k] for k in ORDER]); off = {}; o = 0
    for k in ORDER: off[k] = (o, o + len(Sx[k])); o += len(Sx[k])
    s = np.asarray(simfn(allp), dtype=np.float64)
    g = {k: s[off[k][0]:off[k][1]] for k in ORDER}
    r = {}
    if placed is not None:
        r["cov"] = {k: f4(keepmask[k].mean()) for k in ("ant_te", "sim_te", "syn_mfs", "hyp_te", "sis_te", "rand", "simlex")}
    r["rand_mean"] = f4(g["rand"].mean())
    r["ant_mean"] = f4(g["ant_te"].mean()) if len(g["ant_te"]) else None; r["sim_mean"] = f4(g["sim_te"].mean()) if len(g["sim_te"]) else None
    A_ = lambda x, y: f4(auc(x, y)) if len(x) and len(y) else None
    r["AUC_sim_v_rand"] = A_(g["sim_te"], g["rand_a"])
    r["AUC_mfs_v_rand"] = A_(g["syn_mfs"], g["rand"])
    r["P_sim>ant"] = A_(g["sim_te"], g["ant_te"])
    r["P_mfs>ant"] = A_(g["syn_mfs"], g["ant_te"])
    r["AUC_ant<rand"] = A_(-g["ant_te"], -g["rand"])
    r["AUC_hyp>rand"] = A_(g["hyp_te"], g["rand"])
    r["AUC_sis>rand"] = A_(g["sis_te"], g["rand"])
    pa = pos[Sx["ant_te"][:, 0]]; pm = pos[Sx["syn_mfs"][:, 0]]
    for p in "nvar":
        a = g["ant_te"][pa == p]; m = g["syn_mfs"][pm == p]; rp = g["rand_" + p]
        r[f"n_ant_{p}"] = int((pa == p).sum())
        r[f"P_mfs>ant_{p}"] = A_(m, a)
        r[f"AUC_ant<rand_{p}"] = A_(-a, -rp)
        r[f"ant_mean_{p}"] = f4(a.mean()) if len(a) else None
        r[f"rand_mean_{p}"] = f4(rp.mean()) if len(rp) else None
    a = g["ant_te"][pa == "a"]; r["P_sim>ant_a"] = A_(g["sim_te"], a)
    gold = B["simlex_gold"][keepmask["simlex"]]; sl = g["simlex"]; spp = B["simlex_pos"][keepmask["simlex"]]
    r["simlex_n"] = int(len(sl))
    r["simlex_rho"] = f4(spearman(sl, gold)) if len(sl) > 10 else None
    for p in "nva":
        mk = spp == p
        r[f"simlex_rho_{p}"] = f4(spearman(sl[mk], gold[mk])) if mk.sum() > 10 else None
    r["everyday"] = [f4(x) for x in g["everyday"]]
    return r
EVN = ["happy/unhappy","happy/sad","hot/cold","good/bad","big/small","love/hate","buy/sell","alive/dead","dog/cat","dog/bone","bone/stone","eat/chew","eat/drink","king/queen","red/blue","big/large","quick/fast"]
RES = P + "results.json"
def save(name, r, extra=None):
    d = json.load(open(RES)) if os.path.exists(RES) else {}
    if extra: r = dict(r, **extra)
    d[name] = r; json.dump(d, open(RES, "w"), indent=1)
def wcos_pairs(X, w=None, chunk=20000):
    """returns simfn: weighted cosine over columns of X with axis weights w"""
    def fn(pr):
        out = np.empty(len(pr))
        for i in range(0, len(pr), chunk):
            p = pr[i:i + chunk]; A = X[p[:, 0]].astype(np.float64); Bm = X[p[:, 1]].astype(np.float64)
            ww = 1.0 if w is None else w
            num = (A * Bm * ww).sum(1); den = np.sqrt((A * A * ww).sum(1) * (Bm * Bm * ww).sum(1))
            out[i:i + chunk] = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
        return out
    return fn
