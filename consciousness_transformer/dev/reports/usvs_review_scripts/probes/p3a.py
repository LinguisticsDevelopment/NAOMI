from glib import *
import json, sys
t0 = time.time()
KS = (10, 20, 50, 100, 200)
for hypw in (0.3, 0.05, 0.0):
    I, J, Wt = graph_edges(hypw); lab, g, deg = giant(I, J)
    ok = pos != "?"; nodes = np.where((lab == g) & ok)[0]
    S_, dis, d = sym_op(I, J, Wt, nodes)
    torch.manual_seed(0); X = torch.randn(len(nodes), 200, dtype=torch.float64)
    ev, V = torch.lobpcg(S_, k=200, X=X, niter=120, tol=1e-5, largest=True)
    order = torch.argsort(ev, descending=True); ev = ev[order]; V = V[:, order]
    resid = (S_ @ V - V * ev).norm(dim=0)
    print(f"hyp{hypw} nodes {len(nodes)} eig time {round(time.time()-t0)}s ev[0,1,9,19,49,99,199]={[round(float(ev[i]),5) for i in (0,1,9,19,49,99,199)]} resid med {float(resid.median()):.4f} max {float(resid.max()):.4f}", flush=True)
    emb = np.zeros((N, 200), np.float32); emb[nodes] = (V.numpy() * dis[:, None]).astype(np.float32)
    np.save(P + f"spec_hyp{hypw}.npy", emb); np.save(P + f"spec_hyp{hypw}_ev.npy", ev.numpy())
    placed = np.zeros(N, bool); placed[nodes] = True
    cnt = dict(nodes=int(len(nodes)), outside=int((ok & ~placed).sum()), outside_by_pos={p: int(((pos == p) & ~placed).sum()) for p in "nvar"}, deg0=int(((deg == 0) & ok).sum()),
               n_components=int(len(np.unique(lab[ok]))), edges=int(len(I)), neg=int((Wt < 0).sum()), gold_placed=float(placed[B["gold"][0]].mean()))
    json.dump(cnt, open(P + f"p3a_cnt_hyp{hypw}.json", "w"))
    for k in KS:
        e = emb[:, :k]; nrm = np.linalg.norm(e, axis=1, keepdims=True); en = np.where(nrm > 0, e / np.maximum(nrm, 1e-12), 0)
        fn = lambda pr: (en[pr[:, 0]] * en[pr[:, 1]]).sum(1)
        r = evaluate(fn); save(f"P3a|spec|hyp{hypw}|k{k}|all", r)
        r2 = evaluate(fn, placed=placed); save(f"P3a|spec|hyp{hypw}|k{k}|placed", r2)
        print(f"  k{k}", {x: r[x] for x in ("rand_mean", "ant_mean", "P_sim>ant", "P_mfs>ant", "AUC_ant<rand", "AUC_hyp>rand", "simlex_rho")}, "| placed-only", {x: r2[x] for x in ("ant_mean","P_sim>ant", "P_mfs>ant", "AUC_ant<rand", "AUC_hyp>rand","AUC_sim_v_rand")}, r2["cov"]["ant_te"], round(time.time() - t0), flush=True)
