from glib import *
I, J, Wt = graph_edges(0.3); lab, g, deg = giant(I, J)
nodes = np.where((lab == g) & (pos != "?"))[0]
S, dis, d = sym_op(I, J, Wt, nodes)
print("n", len(nodes), "nnz", S._nnz())
for k, it in ((20, 60),):
    torch.manual_seed(0); X = torch.randn(len(nodes), k, dtype=torch.float64)
    t = time.time()
    ev, V = torch.lobpcg(S, k=k, X=X, niter=it, tol=1e-6, largest=True)
    print(k, it, "time", round(time.time() - t), ev[:6].tolist(), ev[-3:].tolist())
    R = S @ V - V * ev; print("resid", float(R.norm(dim=0).max()), float(R.norm(dim=0).median()))
