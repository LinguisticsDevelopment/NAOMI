"""Hybrid: flat vector [ w_idf/|w_idf| ; gamma * g/|g| ]  (g = graph coordinates, zero if the sense has no graph position); cosine on the concatenation."""
from glib import *
import json
Mf = np.asarray(M, dtype=np.float32); W = Mf * np.sqrt(IDF).astype(np.float32)
Wu = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
def run(name, G, placed):
    gn = np.linalg.norm(G, axis=1, keepdims=True); Gu = np.where(gn > 0, G / np.maximum(gn, 1e-12), 0).astype(np.float32)
    has = (gn[:, 0] > 0).astype(np.float32)
    for gam in (0.5, 1.0, 2.0):
        def fn(pr, gam=gam):
            a, b = pr[:, 0], pr[:, 1]; out = np.empty(len(pr))
            for i in range(0, len(pr), 20000):
                x, y = a[i:i + 20000], b[i:i + 20000]
                num = (Wu[x] * Wu[y]).sum(1) + gam ** 2 * (Gu[x] * Gu[y]).sum(1)
                den = np.sqrt((1 + gam ** 2 * has[x]) * (1 + gam ** 2 * has[y]))
                out[i:i + 20000] = num / den
            return out
        r = evaluate(fn); save(f"P3c|{name}|gamma{gam}|all", r)
        r2 = evaluate(fn, placed=placed); save(f"P3c|{name}|gamma{gam}|placed", r2)
        print(name, gam, {x: r[x] for x in ("rand_mean", "ant_mean", "P_sim>ant", "P_mfs>ant", "AUC_ant<rand", "AUC_hyp>rand", "AUC_sim_v_rand", "simlex_rho")}, flush=True)
for nm, f_, k in (("spec_hyp0.05_k50", "spec_hyp0.05.npy", 50), ("spec_hyp0.05_k200", "spec_hyp0.05.npy", 200), ("spec_hyp0.3_k200", "spec_hyp0.3.npy", 200)):
    G = np.load(P + f_)[:, :k]; run(nm, G, (np.abs(G).sum(1) > 0))
for k in (20, 50, 100):
    G = np.load(P + f"fit_k{k}.npy"); run(f"fit_k{k}", G, (np.abs(G).sum(1) > 0))
