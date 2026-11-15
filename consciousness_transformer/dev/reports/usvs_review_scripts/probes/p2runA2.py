from p2lib import *
import time, json
t0=time.time(); sh={}
for wn_, W in (("raw",W_RAW),("idf",W_IDF)):
    for nm, arr in (("train", E["ant"][~split["ant"]]), ("test", E["ant"][split["ant"]])):
        a, b = arr[:, 0], arr[:, 1]
        score = np.minimum(W[a], W[b]) * IDF * NOLEX
        j = score.argmax(1)
        wa = W[a, j] ** 2 / np.maximum((W[a] ** 2).sum(1), 1e-12); wb = W[b, j] ** 2 / np.maximum((W[b] ** 2).sum(1), 1e-12)
        sh[f"{wn_}|{nm}"] = dict(mean_share=float(np.mean((wa + wb) / 2)), median=float(np.median((wa + wb) / 2)), p90=float(np.percentile((wa + wb) / 2, 90)), axis_top=[axes[k] for k, _ in collections.Counter(j.tolist()).most_common(8)])
json.dump(sh, open(P+"p2_share_nolex.json","w"), indent=1)
for wn_, W in (("idf",W_IDF),):
    for m in (1,3):
        for f in (1,3,10):
            Wm,V,chosen = build_A(W,m,f,nolex=True)
            for meas in ("cos","gower_max"):
                fn=simA(Wm,V,meas); r=evaluate(fn); trs=fn(S["ant_tr"]); r["ant_tr_mean"]=f4(trs.mean()); r["ant_tr_below_randmean"]=f4((trs<r["rand_mean"]).mean())
                nm=f"P2A-nolex|{wn_}|m{m}|x{f}|{meas}"; save(nm,r)
                print(nm,{k:r[k] for k in ("rand_mean","ant_mean","ant_tr_mean","P_sim>ant","P_mfs>ant","AUC_ant<rand","AUC_hyp>rand","AUC_sim_v_rand")},round(time.time()-t0),flush=True)
            if f==1: print(collections.Counter(axes[j] for js in chosen for j in js).most_common(8), int((V!=1).any(1).sum()))
