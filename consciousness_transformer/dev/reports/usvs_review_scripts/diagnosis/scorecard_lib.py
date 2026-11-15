"""Shared pair-set / profile machinery for the USVS content-word scorecard (read-only on data/usvs)."""
from common import *
import pickle, random, collections

def build_sets(u, C, M, P, seed=0, n_cap=20000, n_rand=20000, n_rand_pos=4000):
    """Return {level: {setname: (A_rows, B_rows, pos_list)}} gathered row matrices (float32)."""
    rng = random.Random(seed)
    core = {w: i for i, w in enumerate(u.core_words)}
    sidx = {s: i for i, s in enumerate(u.sense_ids)}
    posc = lambda sid: "a" if sid.rsplit(".", 2)[1] == "s" else sid.rsplit(".", 2)[1]
    mfs = {}
    for w in u.core_words:
        s = u.senses_of(w); mfs[w] = s[0] if s else None
    out = {"core": {}, "sense": {}}
    def add(level, name, ia, ib, pos):
        X = C if level == "core" else M
        out[level][name] = (X[np.asarray(ia)], X[np.asarray(ib)], list(pos), (np.asarray(ia), np.asarray(ib)))
    # synonyms (shared synset, strict)
    syn = [(a, b, p) for a, b, p in P["syn"]]
    add("core", "syn", [core[a] for a, b, p in syn], [core[b] for a, b, p in syn], [p for *_, p in syn])
    sp = [(sidx[mfs[a]], sidx[mfs[b]], p) for a, b, p in syn if mfs[a] and mfs[b] and mfs[a] != mfs[b]]
    add("sense", "syn", [x[0] for x in sp], [x[1] for x in sp], [x[2] for x in sp])
    # project-style synonyms (core only; morphy-expanded; = harness edge set)
    sp2 = P["syn_proj"]
    add("core", "syn_proj", [core[a] for a, b in sp2], [core[b] for a, b in sp2], ["?"] * len(sp2))
    # antonyms direct
    an = P["ant"]
    add("core", "ant", [core[t[0]] for t in an], [core[t[1]] for t in an], [t[4] for t in an])
    add("sense", "ant", [sidx[t[2]] for t in an], [sidx[t[3]] for t in an], [t[4] for t in an])
    ana = [t for t in P["ant_all"] if t[2] in sidx and t[3] in sidx]
    add("sense", "ant_all", [sidx[t[2]] for t in ana], [sidx[t[3]] for t in ana], [t[4] for t in ana])
    # hypernym
    h = P["hyp"]
    hc = [t for t in h if t[2] and t[3] and t[2] != t[3]]
    add("core", "hyp", [core[t[2]] for t in hc], [core[t[3]] for t in hc], [t[4] for t in hc])
    hs = [t for t in h if t[0] in sidx and t[1] in sidx]; rng.shuffle(hs); hs = hs[:n_cap]
    add("sense", "hyp", [sidx[t[0]] for t in hs], [sidx[t[1]] for t in hs], [t[4] for t in hs])
    # co-hyponym
    c = P["cohyp"]
    cc = [t for t in c if t[2] and t[3] and t[2] != t[3]]
    add("core", "cohyp", [core[t[2]] for t in cc], [core[t[3]] for t in cc], [t[4] for t in cc])
    cs = [t for t in c if t[0] in sidx and t[1] in sidx]; rng.shuffle(cs); cs = cs[:n_cap]
    add("sense", "cohyp", [sidx[t[0]] for t in cs], [sidx[t[1]] for t in cs], [t[4] for t in cs])
    # random (global), excluding nothing (collisions with related pairs negligible)
    W = len(u.core_words); N = len(u.sense_ids)
    ra = [rng.randrange(W) for _ in range(n_rand)]; rb = [rng.randrange(W) for _ in range(n_rand)]
    keep = [i for i in range(n_rand) if ra[i] != rb[i]]
    add("core", "rand", [ra[i] for i in keep], [rb[i] for i in keep], ["?"] * len(keep))
    ra = [rng.randrange(N) for _ in range(n_rand)]; rb = [rng.randrange(N) for _ in range(n_rand)]
    keep = [i for i in range(n_rand) if ra[i] != rb[i]]
    add("sense", "rand", [ra[i] for i in keep], [rb[i] for i in keep], ["?"] * len(keep))
    # POS-matched random pools
    cpos = collections.defaultdict(list)
    for w, i in core.items():
        if mfs[w]: cpos[posc(mfs[w])].append(i)
    spos = collections.defaultdict(list)
    for s, i in sidx.items():
        if not s.startswith(("interj.", "fn.")): spos[posc(s)].append(i)
    for p in "nvar":
        for lvl, pool in (("core", cpos[p]), ("sense", spos[p])):
            ia = [pool[rng.randrange(len(pool))] for _ in range(n_rand_pos)]
            ib = [pool[rng.randrange(len(pool))] for _ in range(n_rand_pos)]
            k = [i for i in range(n_rand_pos) if ia[i] != ib[i]]
            add(lvl, "rand_" + p, [ia[i] for i in k], [ib[i] for i in k], [p] * len(k))
    return out

def cos_sets(sets, w, mask=None, chunk=20000):
    r = {}
    for name, (A, B, pos, _) in sets.items():
        r[name] = np.concatenate([wcos(A[i:i + chunk], B[i:i + chunk], w, mask) for i in range(0, len(A), chunk)])
    return r

def summarize(cs, sets, by_pos=False):
    """metrics from cosine dict"""
    f = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
    o = {}
    for k in ("syn", "ant", "hyp", "cohyp", "rand"):
        if k in cs: o[f"{k}_n"] = int(len(cs[k])); o[f"{k}_mean"] = f(cs[k].mean()); o[f"{k}_median"] = f(np.median(cs[k]))
    o["AUC_syn_vs_rand"] = f(auc(cs["syn"], cs["rand"]))
    o["AUC_syn_vs_ant"] = f(auc(cs["syn"], cs["ant"]))        # P(cos_syn > cos_ant)
    o["AUC_ant_below_rand"] = f(auc(-cs["ant"], -cs["rand"]))  # P(cos_ant < cos_rand)
    o["P_ant_closer_than_syn"] = f(1 - auc(cs["syn"], cs["ant"]))
    o["AUC_hyp_vs_rand"] = f(auc(cs["hyp"], cs["rand"]))
    o["AUC_cohyp_vs_rand"] = f(auc(cs["cohyp"], cs["rand"]))
    o["AUC_hyp_vs_cohyp"] = f(auc(cs["hyp"], cs["cohyp"]))
    if by_pos:
        bp = {}
        for p in "nvar":
            d = {}
            sy = np.array([c for c, q in zip(cs["syn"], sets["syn"][2]) if q == p])
            an = np.array([c for c, q in zip(cs["ant"], sets["ant"][2]) if q == p])
            rp = cs["rand_" + p]
            d["syn_n"] = len(sy); d["ant_n"] = len(an)
            if len(sy): d["syn_mean"] = f(sy.mean()); d["syn_median"] = f(np.median(sy)); d["AUC_syn_vs_rand"] = f(auc(sy, rp))
            if len(an):
                d["ant_mean"] = f(an.mean()); d["ant_median"] = f(np.median(an)); d["rand_mean"] = f(rp.mean())
                d["AUC_ant_below_rand"] = f(auc(-an, -rp))
                if len(sy): d["AUC_syn_vs_ant"] = f(auc(sy, an)); d["P_ant_closer_than_syn"] = f(1 - auc(sy, an))
            hy = np.array([c for c, q in zip(cs["hyp"], sets["hyp"][2]) if q == p])
            if len(hy): d["hyp_n"] = len(hy); d["hyp_mean"] = f(hy.mean()); d["AUC_hyp_vs_rand"] = f(auc(hy, rp))
            bp[p] = d
        o["by_pos"] = bp
    return o
