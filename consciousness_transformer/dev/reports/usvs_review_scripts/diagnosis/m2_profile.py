"""Task 2a/2b + 5 + 4(top-K): profile at core & sense level, +/- IDF, kinds masks, top-K axes."""
from scorecard_lib import *
import time
t0 = time.time()
u = load(); C = u.core_coords.astype(np.float32); M = dense_senses(u)
P = pickle.load(open(S + "/pairs.pkl", "rb"))
sets = build_sets(u, C, M, P)
print({l: {k: len(v[0]) for k, v in d.items()} for l, d in sets.items()}, f"{time.time()-t0:.0f}s", flush=True)
w_idf = idf_from(M); kd = kinds(u.axes)
res = {"meta": {"axes": len(u.axes), "n_senses": len(u.sense_ids)}}
res["idf_check_vs_score_axis_weights"] = None
res["base"] = {}
for lvl in ("core", "sense"):
    for wn_, w in (("raw", None), ("idf", w_idf)):
        cs = cos_sets(sets[lvl], w)
        res["base"][f"{lvl}|{wn_}"] = summarize(cs, sets[lvl], by_pos=True)
        if lvl == "sense" and "ant_all" in cs:
            res["base"][f"{lvl}|{wn_}"]["ant_all_n"] = int(len(cs["ant_all"])); res["base"][f"{lvl}|{wn_}"]["ant_all_mean"] = round(float(cs["ant_all"].mean()), 4)
            res["base"][f"{lvl}|{wn_}"]["AUC_syn_vs_ant_all"] = round(auc(cs["syn"], cs["ant_all"]), 4)
        if lvl == "core":
            res["base"][f"{lvl}|{wn_}"]["syn_proj_mean"] = round(float(cs["syn_proj"].mean()), 4)
            res["base"][f"{lvl}|{wn_}"]["AUC_syn_proj_vs_rand"] = round(auc(cs["syn_proj"], cs["rand"]), 4)
            res["base"][f"{lvl}|{wn_}"]["AUC_syn_proj_vs_ant"] = round(auc(cs["syn_proj"], cs["ant"]), 4)
    print("base", lvl, f"{time.time()-t0:.0f}s", flush=True)
# kinds masks
res["kinds"] = {}
kmasks = {"prime": kd == "prime", "attr": kd == "attr", "lex": kd == "lex",
          "prime+attr": kd != "lex", "prime+lex": kd != "attr", "attr+lex": kd != "prime"}
for lvl in ("core", "sense"):
    for wn_, w in (("raw", None), ("idf", w_idf)):
        for kn, m in kmasks.items():
            cs = cos_sets(sets[lvl], w, m.astype(float))
            res["kinds"][f"{lvl}|{wn_}|{kn}"] = summarize(cs, sets[lvl])
    print("kinds", lvl, f"{time.time()-t0:.0f}s", flush=True)
# top-K axes by idf-weighted energy
res["topk"] = {}
for lvl, X in (("core", C), ("sense", M)):
    energy = w_idf * (X.astype(np.float64) ** 2).sum(0)
    order = np.argsort(-energy)
    res["topk"][f"{lvl}_order_first30"] = [u.axes[j] for j in order[:30]]
    for K in (20, 50, 100, 200, 400, 607):
        m = np.zeros(len(u.axes)); m[order[:K]] = 1
        for wn_, w in (("raw", None), ("idf", w_idf)):
            cs = cos_sets(sets[lvl], w, m)
            s = summarize(cs, sets[lvl])
            s["kind_composition"] = {k: int(((kd[order[:K]]) == k).sum()) for k in ("prime", "attr", "lex")}
            res["topk"][f"{lvl}|{wn_}|K{K}"] = s
    print("topk", lvl, f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open(S + "/m2_profile.json", "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
