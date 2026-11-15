from evalib import *
import collections
tr = {k: E[k][~split[k]] for k in E}
Mf = np.asarray(M, dtype=np.float32)
W_RAW = Mf
W_IDF = (Mf * np.sqrt(IDF).astype(np.float32))
head = np.arange(N)
for sat, h in tr["similar"]: head[sat] = h
sats = collections.defaultdict(list)
for sat, h in tr["similar"]: sats[h].append(sat)
def members(h): return [h] + sorted(sats.get(h, []))
def adj(edges, both=True):
    d = collections.defaultdict(list)
    for a, b in edges:
        d[int(a)].append(int(b))
        if both: d[int(b)].append(int(a))
    return d
ADJ = {k: adj(tr[k]) for k in ("deriv", "pert", "hyp", "also", "attr")}
NOLEX = np.array([0.0 if a.startswith('lex:') else 1.0 for a in axes])
# ---------------------------------------------------------------- variant A
def build_A(W, m, f, nolex=False):
    V = np.ones((N, D), np.int8); Wm = W.copy(); mark = np.zeros((N, D), bool)
    chosen = []
    for a, b in tr["ant"]:
        S1, S2 = members(int(a)), members(int(b))
        m1, m2 = W[S1].mean(0), W[S2].mean(0)
        score = np.minimum(m1, m2) * IDF
        if nolex: score = score * NOLEX
        js = [j for j in np.argsort(-score, kind="stable")[:m] if score[j] > 0]
        chosen.append(js)
        for j in js:
            for s in S1:
                if not mark[s, j]: mark[s, j] = True; V[s, j] = 1; Wm[s, j] *= f
            for s in S2:
                if not mark[s, j]: mark[s, j] = True; V[s, j] = -1; Wm[s, j] *= f
    return Wm, V, chosen
def simA(W, V, measure):
    def fn(pr, chunk=8000):
        out = np.empty(len(pr))
        for i in range(0, len(pr), chunk):
            p = pr[i:i + chunk]; A = W[p[:, 0]]; Bm = W[p[:, 1]]; Va = V[p[:, 0]]; Vb = V[p[:, 1]]
            if measure == "cos":
                num = (A * Bm).sum(1) + (A * Va * Bm * Vb).sum(1)
                den = 2 * np.sqrt((A * A).sum(1) * (Bm * Bm).sum(1))
                out[i:i + chunk] = np.where(den > 0, num / np.where(den > 0, den, 1), 0)
            else:
                mn = np.minimum(A, Bm); ag = 1 - np.abs(Va.astype(np.float32) - Vb) / 2
                num = (mn * ag).sum(1)
                den = mn.sum(1) if measure == "gower" else np.maximum(A, Bm).sum(1)
                out[i:i + chunk] = np.where(den > 0, num / np.where(den > 0, den, 1), 0)
        return out
    return fn
# ---------------------------------------------------------------- variant B
LEVELS = [1.0, 0.5, 0.25, 0.125]
def propagate(entries, levels_log, gold_idx, hypo=True):
    """entries: dict s -> dict d -> (sign, g). Adds satellites (g .5), deriv/pert (.25), hyp one step (.125). returns log of coverage"""
    def cov(name):
        keys = np.fromiter(entries.keys(), dtype=np.int64)
        flag = np.zeros(N, bool); flag[keys] = True
        levels_log[name] = dict(senses=int(flag.sum()), share=float(flag.sum() / (pos != "?").sum()),
            gold=float(flag[gold_idx[gold_idx >= 0]].mean()), by_pos={p: float(flag[pos == p].sum() / (pos == p).sum()) for p in "nvar"})
    cov("L0_train_ant_heads")
    # L1 satellites
    new = {}
    for s, dd in list(entries.items()):
        for t in sats.get(s, []):
            for d, (sg, g) in dd.items():
                if d not in entries.get(t, {}): new.setdefault(t, {})[d] = (sg, 0.5)
    for t, dd in new.items(): entries.setdefault(t, {}).update({d: v for d, v in dd.items() if d not in entries.get(t, {})})
    cov("L1_+satellites")
    src = sorted(entries)
    new = {}
    for s in src:
        for kind in ("deriv", "pert"):
            for t in ADJ[kind].get(s, []):
                for d, (sg, g) in entries[s].items():
                    if d not in entries.get(t, {}) and (t not in new or d not in new[t]): new.setdefault(t, {})[d] = (sg, 0.25)
    for t, dd in new.items(): entries.setdefault(t, {}).update(dd)
    cov("L2_+derivation/pertainym")
    if hypo:
        src = sorted(entries); new = {}
        for s in src:
            for t in ADJ["hyp"].get(s, []):
                for d, (sg, g) in entries[s].items():
                    if d not in entries.get(t, {}) and (t not in new or d not in new[t]): new.setdefault(t, {})[d] = (sg, 0.125)
        for t, dd in new.items(): entries.setdefault(t, {}).update(dd)
        cov("L3_+hypernym/hyponym_step")
    return entries
def build_B1(gold_idx):
    entries = {}; log = {}
    for d, (a, b) in enumerate(tr["ant"]):
        entries.setdefault(int(a), {})[d] = (1, 1.0); entries.setdefault(int(b), {})[d] = (-1, 1.0)
    return propagate(entries, log, gold_idx), log, len(tr["ant"])
def build_B2(gold_idx, Wv):
    """merged dimensions: antonym pairs merged via shared attribute noun / also_see between members; poles by signed colouring,
    orientation by vector similarity where only the attribute links them; plus attribute-based assignment of other adjective heads."""
    pairs = [(int(a), int(b)) for a, b in tr["ant"]]
    P = len(pairs); par = list(range(P))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    mem = [(members(a), members(b)) for a, b in pairs]
    owner = collections.defaultdict(list)       # member synset -> [(pair idx, side)]
    for p, (m1, m2) in enumerate(mem):
        for s in m1: owner[s].append((p, 0))
        for s in m2: owner[s].append((p, 1))
    attr_owner = collections.defaultdict(set)   # noun -> pairs
    for s, T in tr["attr"]:
        for x, n in ((int(s), int(T)), (int(T), int(s))):
            for p, side in owner.get(x, []): attr_owner[n].add(p)
    link_attr = []; link_also = []
    for n, ps in attr_owner.items():
        ps = sorted(ps)
        for q in ps[1:]: link_attr.append((ps[0], q))
    for a, b in tr["also"]:
        for (p, s1) in owner.get(int(a), []):
            for (q, s2) in owner.get(int(b), []):
                if p != q: link_also.append((p, s1, q, s2))
    for p, q in link_attr: par[find(p)] = find(q)
    for p, s1, q, s2 in link_also: par[find(p)] = find(q)
    comp = [find(p) for p in range(P)]
    cid = {c: i for i, c in enumerate(sorted(set(comp)))}
    # signed colouring
    sign_side = {}  # (pair, side) -> sign
    nb = collections.defaultdict(list)
    for p, s1, q, s2 in link_also: nb[p].append((q, s1, s2)); nb[q].append((p, s2, s1))
    for p, q in link_attr: nb[p].append((q, None, None)); nb[q].append((p, None, None))
    vec = lambda lst: W_for_orient[lst].mean(0)
    seen = set()
    for root in range(P):
        if root in seen: continue
        seen.add(root); sign_side[(root, 0)] = 1; sign_side[(root, 1)] = -1
        queue = [root]; plus = [mem[root][0]]; minus = [mem[root][1]]
        while queue:
            p = queue.pop(0)
            for (q, s1, s2) in sorted(nb[p], key=lambda t: (t[0], str(t[1]))):
                if q in seen: continue
                seen.add(q)
                if s1 is not None:   # also_see: same sign
                    sg = sign_side[(p, s1)]
                    sign_side[(q, s2)] = sg; sign_side[(q, 1 - s2)] = -sg
                else:                # attribute-share: orient by vector similarity to the poles collected so far
                    Rp = W_for_orient[sum(plus, [])].mean(0); Rm = W_for_orient[sum(minus, [])].mean(0)
                    q1, q2 = W_for_orient[mem[q][0]].mean(0), W_for_orient[mem[q][1]].mean(0)
                    cs = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
                    sc = cs(q1, Rp) - cs(q1, Rm) + cs(q2, Rm) - cs(q2, Rp)
                    sg = 1 if sc >= 0 else -1
                    sign_side[(q, 0)] = sg; sign_side[(q, 1)] = -sg
                for side in (0, 1):
                    (plus if sign_side[(q, side)] > 0 else minus).append(mem[q][side])
                queue.append(q)
    entries = {}
    for p, (m1, m2) in enumerate(mem):
        d = cid[comp[p]]
        for side, mm in ((0, m1), (1, m2)):
            for s in mm:
                if d not in entries.get(s, {}): entries.setdefault(s, {})[d] = (sign_side[(p, side)], 1.0 if s in (pairs[p][0], pairs[p][1]) else 0.5)
    # attribute assignment of adjectives sharing an attribute noun with a component (by nearest pole)
    comp_poles = collections.defaultdict(lambda: ([], []))
    for s, dd in entries.items():
        for d, (sg, g) in dd.items(): comp_poles[d][0 if sg > 0 else 1].append(s)
    Rv = {d: (W_for_orient[pp[0]].mean(0), W_for_orient[pp[1]].mean(0)) for d, pp in comp_poles.items() if pp[0] and pp[1]}
    noun_comp = collections.defaultdict(set)
    for n, ps in attr_owner.items():
        for p in ps: noun_comp[n].add(cid[comp[p]])
    added = 0
    for s, T in tr["attr"]:
        for x, n in ((int(s), int(T)), (int(T), int(s))):
            if x in entries or n not in noun_comp: continue
            for d in sorted(noun_comp[n]):
                if d not in Rv: continue
                cs = lambda a_, b_: float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-12))
                v = W_for_orient[x]; diff = cs(v, Rv[d][0]) - cs(v, Rv[d][1])
                entries.setdefault(x, {})[d] = (1 if diff >= 0 else -1, 0.5); added += 1
    log = {"n_components": len(cid), "n_pairs": P, "attr_assigned": added}
    return propagate(entries, log, gold_idx), log, len(cid)
def to_padded(entries, KMAX=6):
    keys = sorted(entries); R = len(keys)
    rowof = -np.ones(N, np.int64); rowof[keys] = np.arange(R)
    dims = -np.ones((R, KMAX), np.int32); g = np.zeros((R, KMAX), np.float32); v = np.zeros((R, KMAX), np.float32)
    for i, s in enumerate(keys):
        items = sorted(entries[s].items(), key=lambda t: (-t[1][1], t[0]))[:KMAX]
        for k, (d, (sg, gg)) in enumerate(items): dims[i, k] = d; g[i, k] = gg; v[i, k] = sg
    return rowof, dims, g, v
def simB(W, wnorm, padded, lam, measure, chunk=6000):
    rowof, dims, g, v = padded
    zero = (-np.ones((1, dims.shape[1]), np.int32), np.zeros((1, dims.shape[1]), np.float32), np.zeros((1, dims.shape[1]), np.float32))
    dims2 = np.vstack([dims, zero[0]]); g2 = np.vstack([g, zero[1]]); v2 = np.vstack([v, zero[2]])
    ro = np.where(rowof >= 0, rowof, len(dims))
    cfull = lambda idx, rr: lam * wnorm[idx][:, None] * g2[rr]
    def fn(pr):
        out = np.empty(len(pr))
        for i in range(0, len(pr), chunk):
            p = pr[i:i + chunk]; ia, ib = p[:, 0], p[:, 1]; ra, rb = ro[ia], ro[ib]
            A = W[ia]; Bm = W[ib]
            ca, cb = cfull(ia, ra), cfull(ib, rb)
            da, db = dims2[ra], dims2[rb]; va, vb = v2[ra], v2[rb]
            match = (da[:, :, None] == db[:, None, :]) & (da[:, :, None] >= 0)     # n,K,K
            prod = ca[:, :, None] * cb[:, None, :]
            same = (va[:, :, None] * vb[:, None, :]) > 0
            mn = np.minimum(ca[:, :, None], cb[:, None, :])
            if measure == "cos":
                dotB = (prod * match * same).sum((1, 2))
                num = (A * Bm).sum(1) + dotB
                den = np.sqrt(((A * A).sum(1) + (ca ** 2).sum(1)) * ((Bm * Bm).sum(1) + (cb ** 2).sum(1)))
            else:
                mnE = np.minimum(A, Bm).sum(1); mnB = (mn * match).sum((1, 2)); mnBs = (mn * match * same).sum((1, 2))
                num = mnE + mnBs
                if measure == "gower": den = mnE + mnB
                else: den = np.maximum(A, Bm).sum(1) + ca.sum(1) + cb.sum(1) - mnB
            out[i:i + chunk] = np.where(den > 0, num / np.where(den > 0, den, 1), 0)
        return out
    return fn
