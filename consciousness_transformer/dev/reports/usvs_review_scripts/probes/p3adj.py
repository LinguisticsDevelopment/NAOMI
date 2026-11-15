"""Adjective-only signed spectral (reproduces the style of the alternatives report's 2b.7 probe, on our 80/20 split):
 nodes = adjective synsets; edges = train similar_to, also_see (+1), train antonym (-1); giant component; dense eigh;
 unnormalised signed Laplacian L = D~ - A (D~ = sum|w|) and normalised variant; cosine of the first k eigenvectors."""
from glib import *
import json
t0 = time.time()
adjm = (pos == "a")
I, J, Wt = [], [], []
for k_, w_ in (("similar", 1.0), ("also", 1.0), ("ant", -1.0)):
    e = E[k_][~split[k_]]; ok = adjm[e[:, 0]] & adjm[e[:, 1]]
    I.append(e[ok, 0]); J.append(e[ok, 1]); Wt.append(np.full(int(ok.sum()), w_))
I, J, Wt = np.concatenate(I), np.concatenate(J), np.concatenate(Wt)
lab = components(I, J); deg = np.bincount(np.concatenate([I, J]), minlength=N)
cnt = np.bincount(lab[adjm]); g = cnt.argmax(); nodes = np.where((lab == g) & adjm)[0]
print("adj nodes", int(adjm.sum()), "giant", len(nodes), "components(>=2)", int((np.bincount(lab[adjm & (deg > 0)]) > 1).sum()))
m = -np.ones(N, np.int64); m[nodes] = np.arange(len(nodes)); a, b = m[I], m[J]; ok = (a >= 0) & (b >= 0); a, b, w = a[ok], b[ok], Wt[ok]
n = len(nodes); A = np.zeros((n, n)); np.add.at(A, (a, b), w); np.add.at(A, (b, a), w)
d = np.abs(A).sum(1)
placed = np.zeros(N, bool); placed[nodes] = True
for name, L in (("unnorm", np.diag(d) - A), ("norm", np.eye(n) - A / np.sqrt(np.outer(d, d)))):
    ev, V = torch.linalg.eigh(torch.tensor(L)); ev = ev.numpy(); V = V.numpy()
    print(name, "eig", ev[:8].round(4), round(time.time() - t0), flush=True)
    for k in (5, 10, 20, 50, 100, 200):
        emb = np.zeros((N, k), np.float32); emb[nodes] = V[:, :k]
        nr = np.linalg.norm(emb, axis=1, keepdims=True); en = np.where(nr > 0, emb / np.maximum(nr, 1e-12), 0)
        fn = lambda pr: (en[pr[:, 0]] * en[pr[:, 1]]).sum(1)
        r = evaluate(fn, placed=placed); save(f"P3a-adj|{name}|k{k}|placed", r)
        print(f"  {name} k{k} cov_ant {r['cov']['ant_te']} n_ant {r['n_ant_a']} ant_mean {r['ant_mean']} sim_mean {r['sim_mean']} P_sim>ant {r['P_sim>ant']} AUC_ant<rand {r['AUC_ant<rand_a']} rand_a {r['rand_mean_a']}", flush=True)
