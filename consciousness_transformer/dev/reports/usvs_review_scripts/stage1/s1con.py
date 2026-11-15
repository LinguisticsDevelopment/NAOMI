"""Traceable-scale construction, B2 relation baseline, metrics.  No language rules: only dictionary edges + recorded opposite pairs."""
from s1lib import *
import collections, math
PS = pickle.load(open(SC + "s1_pairs.pkl", "rb")); SETS, NAMED = PS["SETS"], PS["NAMED"]
fold = np.load(os.path.join(HERE, "fold.npy")); ANT = E["ant"]
ALLNB = None
def allnb(extra_ant):
    """all-dictionary-link neighbour sets of each sense (no antonym edges except the recorded `extra_ant` pairs)"""
    global ALLNB
    if ALLNB is None:
        ALLNB = [set() for _ in range(N)]
        for t in ("sim", "deriv", "pert", "also", "attr", "vgroup", "entail", "cause", "hypd", "hypu"):
            for a, b in EDGE[t]: ALLNB[a].add(int(b))
    out = {}
    for a, b in extra_ant: out.setdefault(int(a), set()).add(int(b)); out.setdefault(int(b), set()).add(int(a))
    return out

# ---------------- scales ----------------
def make_scales(pairs, merge, nb_attr=None):
    """one scale per recorded pair (plus member a, minus member b). merge=True: pairs whose members share an explicit `attribute`
    noun are merged into one scale; orientation = which assignment makes the members more alike in the named block (B1 cosine)."""
    sc = [[{int(a)}, {int(b)}, [i]] for i, (a, b) in enumerate(pairs)]
    if not merge: return sc
    import evalib
    attr = collections.defaultdict(set)
    for a, b in E["attr"]:
        for x, n in ((a, b), (b, a)): attr[int(n)].add(int(x))
    pa = collections.defaultdict(list)       # attribute noun -> scale ids
    for si, (P_, Q_, _) in enumerate(sc):
        for n, adjs in attr.items():
            if (P_ | Q_) & adjs: pa[n].append(si)
    par = list(range(len(sc)))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    flip = {}   # scale -> bool (stored relative to its root's orientation, resolved greedily)
    for n, l in pa.items():
        for si in l[1:]:
            r0, r1 = f(l[0]), f(si)
            if r0 == r1: continue
            A0, B0_ = list(sc[r0][0]), list(sc[r0][1]); A1, B1_ = list(sc[r1][0]), list(sc[r1][1])
            pr = lambda X, Y: evalib.wcos_pairs(evalib.M, evalib.IDF)(np.array([(x, y) for x in X for y in Y])).mean()
            same = pr(A0, A1) + pr(B0_, B1_); opp = pr(A0, B1_) + pr(B0_, A1)
            if opp > same: sc[r1][0], sc[r1][1] = sc[r1][1], sc[r1][0]
            sc[r0][0] |= sc[r1][0]; sc[r0][1] |= sc[r1][1]; sc[r0][2] += sc[r1][2]; par[r1] = r0
    return [sc[i] for i in range(len(sc)) if par[i] == i]

# ---------------- BFS with parents ----------------
def bfs(srcs, A, T, hyp, cap, Ahyp):
    d = {s: (0, None, None) for s in srcs}; fr = list(srcs)
    for k in range(1, T + 1):
        nf = []
        for x in fr:
            if cap and k > 1 and len(A[x]) > cap: continue
            for n, t in A[x]:
                if n not in d: d[n] = (k, x, t); nf.append(n)
        fr = nf
    if hyp:
        for x, (k, _, _) in list(d.items()):
            if k < T and not (cap and k > 0 and len(Ahyp[x]) > cap):
                for n, t in Ahyp[x]:
                    if n not in d: d[n] = (k + 1, x, t)
    return d
_bfs_cache = {}
def scale_bfs(scales, etypes, T, hyp, cap, key):
    k = (key, tuple(etypes), T, hyp, cap)
    if k not in _bfs_cache:
        A = adj(etypes); Ah = adj(("hypd", "hypu"))
        _bfs_cache.clear() if len(_bfs_cache) > 12 else None
        _bfs_cache[k] = [(bfs(P_, A, T, hyp, cap, Ah), bfs(Q_, A, T, hyp, cap, Ah)) for P_, Q_, _ in scales]
    return _bfs_cache[k]

# ---------------- placement ----------------
def place(scales, bf, d, steps, imp, extra_ant=(), hyp_step_cost=1):
    """returns dict sense -> {scale: (pos, imp)}   imp in {'a','b'}"""
    out = collections.defaultdict(dict); members = {}
    for si, ((P_, Q_, _), (dp, dq)) in enumerate(zip(scales, bf)):
        nodes = {}
        for n in set(dp) | set(dq):
            kp = dp[n][0] if n in dp and dp[n][0] <= steps else None
            kq = dq[n][0] if n in dq and dq[n][0] <= steps else None
            if kp is None and kq is None: continue
            if n in P_: p = 1.0; kmin = 0
            elif n in Q_: p = -1.0; kmin = 0
            else:
                p = (d ** kp if kp is not None else 0.0) - (d ** kq if kq is not None else 0.0)
                kmin = min(k for k in (kp, kq) if k is not None)
            nodes[n] = (max(-1.0, min(1.0, p)), d ** kmin, kp is not None and kq is not None)
        if imp == "a":
            for n, (p, ia, both) in nodes.items(): out[n][si] = (p, ia)
        else:
            M_ = set(nodes)
            for n, (p, ia, both) in nodes.items():
                nb = ALLNB[n] | extra_ant.get(n, set()) if extra_ant else ALLNB[n]
                out[n][si] = (p, (len(nb & M_) / len(nb)) if nb else 1.0)
        members[si] = nodes
    return out, members

def pair_scale(pl, pairs):
    """-> (agree, g, shared) arrays; exact formula in the report"""
    n = len(pairs); ag = np.zeros(n); g = np.zeros(n); sh = np.zeros(n, bool)
    for i, (a, b) in enumerate(pairs):
        da, db = pl.get(int(a)), pl.get(int(b))
        if not da or not db: continue
        s = da.keys() & db.keys() if len(da) < len(db) else db.keys() & da.keys()
        if not s: continue
        num = den = 0.0
        for k in s:
            m = min(da[k][1], db[k][1]); den += m; num += m * (1 - abs(da[k][0] - db[k][0]) / 2)
        if den <= 0: continue
        ag[i] = num / den; g[i] = den / len(s); sh[i] = True
    return ag, g, sh
def combine(named, ag, g): return (1 - g) * named + g * ag

# ---------------- B2: opposition as a relation ----------------
def b2_sims(build_pairs, r, d, pairs, named):
    A = adj(SPREAD); opp = collections.defaultdict(list)
    for a, b in build_pairs: opp[int(a)].append(int(b)); opp[int(b)].append(int(a))
    cache = {}
    def reach(x):
        if x not in cache:
            dd = {x: 0}; fr = [x]
            for k in range(1, r + 1):
                nf = []
                for y in fr:
                    for n, _ in A[y]:
                        if n not in dd: dd[n] = k; nf.append(n)
                fr = nf
            cache[x] = dd
        return cache[x]
    st = np.zeros(len(pairs))
    for i, (a, b) in enumerate(pairs):
        ra, rb = reach(int(a)), reach(int(b)); best = 0.0
        for p, sa in ra.items():
            for q in opp.get(p, ()):
                if q in rb: best = max(best, d ** (sa + rb[q]))
        st[i] = best
    return named * (1 - st), st

# ---------------- metrics ----------------
def metrics(sims, keep_ant=None, ant_key="ant_test"):
    """sims: set name -> similarity array. keep_ant: bool mask over the antonym set (None = all pairs)."""
    g = dict(sims); a = g[ant_key] if keep_ant is None else g[ant_key][keep_ant]
    ap = pos[SETS[ant_key][:, 0]]; ap = ap if keep_ant is None else ap[keep_ant]
    pm = pos[SETS["syn_mfs"][:, 0]]
    A_ = lambda x, y: f4(auc(x, y)) if len(x) and len(y) else None
    r = {"n_ant": int(len(a)), "ant_mean": f4(a.mean()) if len(a) else None, "rand_mean": f4(g["rand"].mean()),
         "P_sim>ant": A_(g["sim_te"], a), "P_mfs>ant": A_(g["syn_mfs"], a), "AUC_ant<rand": A_(-a, -g["rand"]),
         "AUC_hyp": A_(g["hyp_te"], g["rand"]), "AUC_sim_v_rand": A_(g["sim_te"], g["rand_a"]), "AUC_mfs_v_rand": A_(g["syn_mfs"], g["rand"])}
    for p in "nvar":
        x = a[ap == p]; r[f"n_{p}"] = int(len(x))
        r[f"ant<rand_{p}"] = A_(-x, -g["rand_" + p]); r[f"mfs>ant_{p}"] = A_(g["syn_mfs"][pm == p], x); r[f"ant_mean_{p}"] = f4(x.mean()) if len(x) else None
    gold = B["simlex_gold"]; r["simlex"] = f4(spearman(g["simlex"], gold))
    return r
def unrel_report(sims, named, thr=0.15):
    o = {}
    for k in ("rand_n", "rand_v", "rand_a", "unrel30"):
        drop = named[k] - sims[k]; o[k] = {"mean_drop": f4(drop.mean()), "frac_pushed_apart": f4((drop > thr).mean()), "frac_pulled_together": f4((drop < -thr).mean()), "frac_changed_at_all": f4((np.abs(drop) > 1e-9).mean()), "mean_sim": f4(sims[k].mean())}
    return o

# ---------------- explanation helpers ----------------
TYPE = {"sim": "similar_to", "deriv": "derivational", "pert": "pertainym", "also": "also_see", "hypd": "has-hypernym", "hypu": "has-hyponym"}
def chain(node, si, bf, scales):
    dp, dq = bf[si]; P_, Q_, _ = scales[si]
    sides = [(dd[node][0], dd) for dd in (dp, dq) if node in dd]
    k, d_ = min(sides, key=lambda x: x[0]); path = [node]; ety = []; x = node
    while d_[x][1] is not None: ety.append(TYPE[d_[x][2]]); x = d_[x][1]; path.append(x)
    other = sorted(Q_ if x in P_ else P_)[0]
    s = sids[path[-1]]
    for i in range(len(path) - 2, -1, -1): s += f" -[{ety[i]}]-> {sids[path[i]]}"
    return s + f"   [{sids[x]} is the recorded opposite of {sids[other]}]", x, other
def agree_dicts(da, db):
    s = da.keys() & db.keys()
    if not s: return 0.0, 0.0
    num = den = 0.0
    for k in s:
        m = min(da[k][1], db[k][1]); den += m; num += m * (1 - abs(da[k][0] - db[k][0]) / 2)
    return (num / den, den / len(s)) if den > 0 else (0.0, 0.0)
def op_very(d): return {k: (p + 0.5 * ((1.0 if p > 0 else -1.0 if p < 0 else p) - p), i) for k, (p, i) in d.items()}
def op_slightly(d): return {k: (p * 0.4, i) for k, (p, i) in d.items()}
def op_not(d): return {k: (p * -0.6, i) for k, (p, i) in d.items()}
