from evalib import *
import torch, time
torch.set_num_threads(4)
POSW = {"similar": 1.0, "also": 1.0, "deriv": 1.0, "pert": 1.0, "attr": 1.0, "vgroup": 1.0, "entail": 1.0, "cause": 1.0}
def graph_edges(hypw, train=True, negw=1.0):
    I, J, Wt = [], [], []
    for k in list(POSW) + ["hyp", "ant"]:
        e = E[k][~split[k]] if train else E[k]
        w = {"hyp": hypw, "ant": -negw}.get(k, POSW.get(k))
        if w == 0 or len(e) == 0: continue
        I.append(e[:, 0]); J.append(e[:, 1]); Wt.append(np.full(len(e), w))
    I, J, Wt = np.concatenate(I), np.concatenate(J), np.concatenate(Wt)
    # merge duplicate undirected pairs (sum weights)
    lo, hi = np.minimum(I, J), np.maximum(I, J)
    key = lo * N + hi; order = np.argsort(key, kind="stable"); key = key[order]; Wt = Wt[order]
    uk, start = np.unique(key, return_index=True)
    wsum = np.add.reduceat(Wt, start)
    keep = wsum != 0
    return (uk // N)[keep], (uk % N)[keep], wsum[keep]
def components(I, J, n=N):
    lab = np.arange(n)
    while True:
        m = np.minimum(lab[I], lab[J])
        new = lab.copy(); np.minimum.at(new, I, m); np.minimum.at(new, J, m)
        # pointer jumping
        new = new[new]
        if (new == lab).all(): break
        lab = new
    return lab
def giant(I, J):
    lab = components(I, J)
    deg = np.bincount(np.concatenate([I, J]), minlength=N)
    ok = pos != "?"
    cnt = np.bincount(lab[ok]); g = cnt.argmax()
    return lab, g, deg
def sym_op(I, J, Wt, nodes):
    """normalised signed adjacency S = D^-1/2 A D^-1/2 restricted to nodes (index map), as torch sparse csr; D = sum |w|"""
    m = -np.ones(N, np.int64); m[nodes] = np.arange(len(nodes))
    a, b = m[I], m[J]; ok = (a >= 0) & (b >= 0); a, b, w = a[ok], b[ok], Wt[ok]
    n = len(nodes); d = np.bincount(a, weights=np.abs(w), minlength=n) + np.bincount(b, weights=np.abs(w), minlength=n)
    dis = 1 / np.sqrt(np.maximum(d, 1e-12))
    r = np.concatenate([a, b]); c = np.concatenate([b, a]); v = np.concatenate([w, w]) * dis[r] * dis[c]
    S = torch.sparse_coo_tensor(torch.tensor(np.stack([r, c])), torch.tensor(v, dtype=torch.float64), (n, n)).coalesce()
    return S, dis, d
