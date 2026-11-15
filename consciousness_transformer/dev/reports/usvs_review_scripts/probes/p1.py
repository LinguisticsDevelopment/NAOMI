from evalib import *
import time
t = time.time()
# --- recorded-opposite lookup built from TRAIN antonym edges (+ train similar_to clusters)
ant_tr = E["ant"][~split["ant"]]; ant_all = E["ant"]
sim_tr = E["similar"][~split["similar"]]
head = np.arange(N)
for sat, h in sim_tr: head[sat] = h       # satellite -> head via train similar_to
def mk_lookup(ants, use_cluster):
    keys = set()
    for a, b in ants:
        if use_cluster: a, b = head[a], head[b]
        keys.add((min(a, b), max(a, b)))
    def flag(pr):
        a, b = pr[:, 0], pr[:, 1]
        if use_cluster: a, b = head[a], head[b]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        return np.array([(x, y) in keys for x, y in zip(lo, hi)])
    return flag
LK = {"direct_train": mk_lookup(ant_tr, False), "cluster_train": mk_lookup(ant_tr, True),
      "direct_all(not held out)": mk_lookup(ant_all, False), "cluster_all(not held out)": mk_lookup(ant_all, True)}
# catch rates on test antonyms / false flags on random + sim
cov = {}
for k, fl in LK.items():
    cov[k] = dict(ant_te=f4(fl(S["ant_te"]).mean()), sim_te_false=f4(fl(S["sim_te"]).mean()), rand_false=f4(fl(S["rand"]).mean()), mfs_false=f4(fl(S["syn_mfs"]).mean()),
                  by_pos={p: f4(fl(S["ant_te"][pos[S["ant_te"][:, 0]] == p]).mean()) for p in "nvar"})
    print(k, cov[k])
# --- weights
kd = np.array(["attr" if a.startswith("attr:") else "lex" if a.startswith("lex:") else "prime" for a in axes])
dead = df == 0
# near-duplicate axes (sense-level column corr > 0.9): greedy, keep lowest index
Mf = np.asarray(M, dtype=np.float32)
live = np.where(~dead)[0]
C = np.corrcoef(Mf[:, live].T.astype(np.float64)) if False else None
mu = Mf[:, live].mean(0); sd = Mf[:, live].std(0) + 1e-9
Z = (Mf[:, live] - mu) / sd
Cm = (Z.T @ Z) / N
dup = np.zeros(D, bool)
for a in range(len(live)):
    if dup[live[a]]: continue
    for b in range(a + 1, len(live)):
        if Cm[a, b] > 0.9 and not dup[live[b]]: dup[live[b]] = True
print("dead", dead.sum(), "dups", dup.sum(), time.time() - t)
rank_ubiq = np.argsort(-df, kind="stable")
energy = IDF * (Mf.astype(np.float64) ** 2).sum(0)
rank_en = np.argsort(-energy, kind="stable")
W = {}
W["a_raw"] = np.ones(D)
W["b_idf"] = IDF.copy()
for k in (3, 12, 30):
    w = IDF.copy(); w[rank_ubiq[:k]] = 0; W[f"c_idf_drop{k}"] = w
    w = np.ones(D); w[rank_ubiq[:k]] = 0; W[f"c_raw_drop{k}"] = w
w = IDF.copy(); w[dead | dup] = 0; W["c_idf_dead+dup"] = w
w = IDF.copy(); w[dead | dup] = 0; w[rank_ubiq[:12]] = 0; W["c_idf_dead+dup+drop12"] = w
for K in (50, 100, 200):
    w = np.zeros(D); w[rank_en[:K]] = IDF[rank_en[:K]]; W[f"d_top{K}_idf"] = w
json.dump({"dead": int(dead.sum()), "dup": int(dup.sum()), "dropped_k12": [axes[i] for i in rank_ubiq[:12]], "dropped_k3": [axes[i] for i in rank_ubiq[:3]]}, open(P + "p1_axes.json", "w"))
for wn_, w in W.items():
    base = wcos_pairs(Mf, w)
    r = evaluate(base); save(f"P1|{wn_}", r)
    print(wn_, {k: r[k] for k in ("rand_mean", "AUC_sim_v_rand", "P_sim>ant", "P_mfs>ant", "AUC_ant<rand", "AUC_hyp>rand", "simlex_rho")}, round(time.time() - t))
    for lk in ("direct_train", "cluster_train", "cluster_all(not held out)"):
        fl = LK[lk]
        def fn(pr, base=base, fl=fl):
            s = base(pr); f = fl(pr); s = s.copy(); s[f] = 0.0; return s
        r2 = evaluate(fn); save(f"P1|{wn_}+lookup[{lk}]", r2)
        print("   +", lk, {k: r2[k] for k in ("P_sim>ant", "P_mfs>ant", "AUC_ant<rand", "simlex_rho")})
json.dump(cov, open(P + "p1_cov.json", "w"), indent=1)
