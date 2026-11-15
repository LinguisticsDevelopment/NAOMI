"""Signed-graph embedding by Adam gradient descent. Objective (cosine geometry, rows unit-normalised z):
 L = sum_{(i,j) in E+} w_ij (1 - z_i.z_j) / sum w  +  3 * mean_{(i,j) in E-} relu(z_i.z_j + 0.5)  +  1 * mean_{random pairs} (z_i.z_j)^2
 E+ : train positive edges (similar_to, also_see, derivational, pertainym, attribute, verb_group, entailment, cause = 1.0; hypernym = 0.3); E- : train antonym edges
 init N(0,1)/sqrt(k), torch seed 0; Adam lr 0.02; 400 full-batch iterations; 100k random pairs per iteration from a seeded generator."""
from glib import *
import json, sys, hashlib
ITERS = 400; LR = 0.02
def fit(k, seed=0, log=None):
    I, J, Wt = graph_edges(0.3)
    I_t, J_t = torch.tensor(I), torch.tensor(J); w = torch.tensor(Wt, dtype=torch.float32)
    pm = w > 0; nm = w < 0
    deg = np.bincount(np.concatenate([I, J]), minlength=N)
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed + 1)
    X = (torch.randn(N, k) / np.sqrt(k)).requires_grad_(True)
    opt = torch.optim.Adam([X], lr=LR)
    ok_idx = torch.tensor(np.where(pos != "?")[0])
    curve = []
    for it in range(ITERS + 1):
        z = X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)
        c = (z[I_t] * z[J_t]).sum(1)
        lp = (w[pm] * (1 - c[pm])).sum() / w[pm].sum()
        ln = torch.relu(c[nm] + 0.5).mean()
        ri = ok_idx[torch.randint(len(ok_idx), (100000,), generator=gen)]; rj = ok_idx[torch.randint(len(ok_idx), (100000,), generator=gen)]
        lr_ = ((z[ri] * z[rj]).sum(1) ** 2).mean()
        loss = lp + 3 * ln + lr_
        if it % 25 == 0:
            curve.append(dict(it=it, loss=float(loss), pos=float(lp), neg=float(ln), rand=float(lr_), neg_cos_mean=float(c[nm].mean())))
            if log: print(f"   k{k} it{it} loss {float(loss):.4f} pos {float(lp):.4f} neg {float(ln):.4f} rand {float(lr_):.4f}", flush=True)
        if it == ITERS: break
        opt.zero_grad(); loss.backward(); opt.step()
    Z = (X / X.norm(dim=1, keepdim=True).clamp_min(1e-9)).detach().numpy().astype(np.float32)
    placed = deg > 0
    Z[~placed] = 0
    return Z, placed, curve
if __name__ == "__main__":
    t0 = time.time(); allc = {}
    for k in (20, 50, 100):
        Z, placed, curve = fit(k, 0, log=True); np.save(P + f"fit_k{k}.npy", Z)
        allc[k] = curve
        fn = lambda pr: (Z[pr[:, 0]] * Z[pr[:, 1]]).sum(1)
        r = evaluate(fn); save(f"P3b|fit|k{k}|all", r); r2 = evaluate(fn, placed=placed); save(f"P3b|fit|k{k}|placed", r2)
        print(f"k{k}", {x: r[x] for x in ("rand_mean", "ant_mean", "P_sim>ant", "P_mfs>ant", "AUC_ant<rand", "AUC_hyp>rand", "simlex_rho")}, "placed", {x: r2[x] for x in ("ant_mean", "P_sim>ant", "P_mfs>ant", "AUC_ant<rand", "AUC_hyp>rand")}, r2["cov"]["ant_te"], round(time.time() - t0), flush=True)
        json.dump(dict(curves=allc, placed=int(placed.sum())), open(P + "p3b_curves.json", "w"))
    # reproducibility: rerun k=20 seed 0
    Z2, _, _ = fit(20, 0)
    Z1 = np.load(P + "fit_k20.npy")
    print("repro k20: max abs diff", float(np.abs(Z1 - Z2).max()), "sha1", hashlib.sha1(Z1.tobytes()).hexdigest()[:12], hashlib.sha1(Z2.tobytes()).hexdigest()[:12], flush=True)
    json.dump(dict(curves=allc, placed=int(placed.sum()), repro_maxdiff=float(np.abs(Z1 - Z2).max()), sha1=[hashlib.sha1(Z1.tobytes()).hexdigest(), hashlib.sha1(Z2.tobytes()).hexdigest()]), open(P + "p3b_curves.json", "w"))
