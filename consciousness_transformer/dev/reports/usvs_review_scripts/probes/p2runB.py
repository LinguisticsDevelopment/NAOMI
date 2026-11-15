import p2lib
p2lib.W_for_orient = p2lib.W_IDF
from p2lib import *
import time, json, pickle
t0 = time.time(); gold_idx = B["gold"][0]
info = {}
res = {}
for name, builder in (("B1_per_pair", lambda: build_B1(gold_idx)), ("B2_merged", lambda: build_B2(gold_idx, W_IDF))):
    entries, log, K = builder()
    print(name, "dims", K, round(time.time() - t0), flush=True)
    pad = to_padded(entries)
    # separation of held-out test antonym pairs by construction
    te = S["ant_te"]; sep = same = onedim = none = 0
    for a, b in te:
        ea, eb = entries.get(int(a), {}), entries.get(int(b), {})
        com = set(ea) & set(eb)
        if not ea and not eb: none += 1
        elif com:
            d = min(com)
            if ea[d][0] != eb[d][0]: sep += 1
            else: same += 1
        else: onedim += 1
    tr_ = S["ant_tr"]; sep_tr = sum(1 for a, b in tr_ if (set(entries.get(int(a), {})) & set(entries.get(int(b), {}))) and any(entries[int(a)][d][0] != entries[int(b)][d][0] for d in set(entries[int(a)]) & set(entries[int(b)])))
    # sim_te pairs (satellite-head) sharing pole: sanity
    st = sum(1 for a, b in S["sim_te"] if set(entries.get(int(a), {})) & set(entries.get(int(b), {})))
    info[name] = dict(K=K, log=log, test_ant=dict(n=len(te), separated=sep, same_pole=same, one_side_only=onedim, neither_placed=none),
                      train_ant_separated=sep_tr, n_train_ant=len(tr_), sim_te_sharing_dim=st, n_sim_te=len(S["sim_te"]),
                      mean_entries_per_placed=float(np.mean([len(v) for v in entries.values()])), max_entries=max(len(v) for v in entries.values()))
    print(json.dumps(info[name], default=float)[:900], flush=True)
    json.dump(info, open(P + "p2B_info.json", "w"), indent=1, default=float)
    pickle.dump(entries, open(P + f"entries_{name}.pkl", "wb"))
    for wn_, W in (("idf", W_IDF), ("raw", W_RAW)):
        wnorm = np.sqrt((W.astype(np.float64) ** 2).sum(1)).astype(np.float32)
        for lam in (0.3, 0.9, 3.0):
            for meas in (("cos", "gower_max", "gower") if wn_ == "idf" else ("cos", "gower_max")):
                fn = simB(W, wnorm, pad, lam, meas)
                r = evaluate(fn); trs = fn(S["ant_tr"]); r["ant_tr_mean"] = f4(trs.mean()); r["ant_tr_below_randmean"] = f4((trs < r["rand_mean"]).mean())
                nm = f"P2{name}|{wn_}|lam{lam}|{meas}"; save(nm, r)
                print(nm, {k: r[k] for k in ("rand_mean", "ant_mean", "ant_tr_mean", "P_sim>ant", "P_mfs>ant", "AUC_ant<rand", "AUC_hyp>rand", "AUC_sim_v_rand")}, round(time.time() - t0), flush=True)
