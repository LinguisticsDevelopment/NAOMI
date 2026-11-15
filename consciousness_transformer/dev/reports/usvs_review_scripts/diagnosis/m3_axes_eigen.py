"""Task 3 (axis usage, near-dup, shares, axes/sense) + Task 4 (eigenspectra). Read-only."""
from common import *
import time
t0=time.time()
u = load(); C = u.core_coords.astype(np.float64); M = dense_senses(u).astype(np.float64)
ax = u.axes; D = len(ax); kd = kinds(ax); w = idf_from(M.astype(np.float32))
res = {}
# ---- IDF equality check against score.axis_weights on the extended artifact
from nsm_ct.ground.usvs import load_usvs
from naomi.score import axis_weights
ue = load_usvs(DATA, extended=True)
wd = axis_weights(ue)
res["idf_check"] = {"max_abs_diff_first607": float(np.abs(wd[:D] - w).max()), "n_axes_extended": len(ue.axes), "n_senses_extended": len(ue.sense_ids), "idf_ext_axes": {a: float(x) for a, x in zip(ue.axes[D:], wd[D:])}}
# ---- axis usage
nz_s = (M != 0); nz_c = (np.abs(C) > 1e-6)
fs = nz_s.mean(0); fc = nz_c.mean(0)
mean_s = np.where(nz_s.sum(0) > 0, M.sum(0) / np.maximum(nz_s.sum(0), 1), 0)
mean_c = np.where(nz_c.sum(0) > 0, np.abs(C).sum(0) / np.maximum(nz_c.sum(0), 1), 0)
neg_c = (C < -1e-6).mean(0)
rows = []
for j in range(D):
    rows.append(dict(axis=ax[j], kind=str(kd[j]), frac_senses=round(float(fs[j]),5), frac_core=round(float(fc[j]),5),
        mean_w_senses=round(float(mean_s[j]),4), mean_abs_core=round(float(mean_c[j]),4), frac_core_neg=round(float(neg_c[j]),5), idf=round(float(w[j]),4),
        n_senses=int(nz_s[:,j].sum()), n_core=int(nz_c[:,j].sum())))
res["axis_rows"] = rows
res["near_constant_senses_gt50pct"] = [(r["axis"], r["frac_senses"]) for r in sorted(rows, key=lambda r: -r["frac_senses"]) if r["frac_senses"] > 0.5]
res["frac_gt25pct_senses"] = [(r["axis"], r["frac_senses"]) for r in sorted(rows, key=lambda r: -r["frac_senses"]) if r["frac_senses"] > 0.25]
res["frac_gt50pct_core"] = [(r["axis"], r["frac_core"]) for r in sorted(rows, key=lambda r: -r["frac_core"]) if r["frac_core"] > 0.5]
res["empty_senses"] = [r["axis"] for r in rows if r["n_senses"] == 0]
res["empty_core"] = [r["axis"] for r in rows if r["n_core"] == 0]
res["empty_both"] = [r["axis"] for r in rows if r["n_core"] == 0 and r["n_senses"] == 0]
res["rare_senses_le50"] = [(r["axis"], r["n_senses"]) for r in rows if 0 < r["n_senses"] <= 50]
res["by_kind_summary"] = {k: dict(n_axes=int((kd == k).sum()), empty_senses=int(((kd == k) & (nz_s.sum(0) == 0)).sum()), empty_core=int(((kd == k) & (nz_c.sum(0) == 0)).sum()),
     mean_frac_senses=float(fs[kd == k].mean()), mean_frac_core=float(fc[kd == k].mean())) for k in ("prime", "attr", "lex")}
# attr provenance: attribute-relation (gradable dimension) vs domain (topic/region/usage)
from nsm_ct.wordnet import attributes, domains
attr_rel, dom = set(), set()
for wd_ in u.core_words:
    for a in attributes(wd_): attr_rel.add("attr:" + a.lower())
    for d in domains(wd_): dom.add("attr:" + d.lower())
ak = [a for a in ax if a.startswith("attr:")]
res["attr_provenance"] = {"attribute_relation_only": sum(1 for a in ak if a in attr_rel and a not in dom),
    "domain_only": sum(1 for a in ak if a in dom and a not in attr_rel), "both": sum(1 for a in ak if a in dom and a in attr_rel), "neither_found": [a for a in ak if a not in dom and a not in attr_rel][:20], "n_neither": sum(1 for a in ak if a not in dom and a not in attr_rel)}
res["attr_relation_axes_sample"] = sorted([a for a in ak if a in attr_rel and a not in dom])[:80]
res["attr_relation_axes_stats"] = {"n": sum(1 for a in ak if a in attr_rel and a not in dom),
    "mean_frac_senses": float(np.mean([fs[ax.index(a)] for a in ak if a in attr_rel and a not in dom])), "mean_frac_core": float(np.mean([fc[ax.index(a)] for a in ak if a in attr_rel and a not in dom]))}
res["domain_axes_stats"] = {"n": sum(1 for a in ak if a in dom and a not in attr_rel),
    "mean_frac_senses": float(np.mean([fs[ax.index(a)] for a in ak if a in dom and a not in attr_rel])), "mean_frac_core": float(np.mean([fc[ax.index(a)] for a in ak if a in dom and a not in attr_rel]))}
isdom = np.array([a in dom and a not in attr_rel for a in ax]); isrel = np.array([a in attr_rel and a not in dom for a in ax])
# ---- axes per sense / per core word
ns = nz_s.sum(1); nc = nz_c.sum(1)
q = lambda x: {p: float(np.percentile(x, p)) for p in (0, 5, 25, 50, 75, 95, 100)} | {"mean": float(x.mean())}
res["axes_per_sense"] = q(ns); res["axes_per_core_word"] = q(nc)
res["axes_per_sense_by_kind"] = {k: q(nz_s[:, kd == k].sum(1)) for k in ("prime", "attr", "lex")}
res["senses_with_0_prime"] = float((nz_s[:, kd == "prime"].sum(1) == 0).mean())
# ---- share of squared norm by kind (per sense, then averaged; also pooled), unweighted and IDF-weighted
def share(X, ww):
    E = (X ** 2) * ww
    tot = E.sum(1); ok = tot > 0
    out = {}
    for k in ("prime", "attr", "lex"):
        e = E[:, kd == k].sum(1)
        out[k] = dict(mean_of_ratios=float((e[ok] / tot[ok]).mean()), median=float(np.median(e[ok] / tot[ok])), pooled=float(e.sum() / tot.sum()))
    return out
res["norm_share"] = {"sense|raw": share(M, np.ones(D)), "sense|idf": share(M, w), "core|raw": share(C, np.ones(D)), "core|idf": share(C, w)}
# domain-vs-relation split of attr share
def share2(X, ww):
    E = (X ** 2) * ww; tot = E.sum(1); ok = tot > 0
    return {"attr_domain": float((E[:, isdom].sum(1)[ok] / tot[ok]).mean()), "attr_relation": float((E[:, isrel].sum(1)[ok] / tot[ok]).mean())}
res["attr_split_share"] = {"sense|raw": share2(M, np.ones(D)), "sense|idf": share2(M, w), "core|raw": share2(C, np.ones(D)), "core|idf": share2(C, w)}
# ---- near-duplicate axes
def topcorr(X, label, thr=0.9, top=60):
    sd = X.std(0); keep = np.where(sd > 1e-9)[0]
    Z = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    R = (Z.T @ Z) / X.shape[0]
    iu = np.triu_indices(len(keep), 1); v = R[iu]
    order = np.argsort(-v)
    pairs = [(ax[keep[iu[0][o]]], ax[keep[iu[1][o]]], round(float(v[o]), 4)) for o in order[:top]]
    return {"n_columns_with_variance": int(len(keep)), "n_pairs_gt_0.9": int((v > thr).sum()), "n_pairs_gt_0.8": int((v > 0.8).sum()), "top": pairs}
res["dup_core"] = topcorr(C, "core")
res["dup_sense"] = topcorr(M, "sense")
print("axes done", time.time()-t0, flush=True)
# ---- eigenspectra
def spec(X, ww=None, center=False):
    Xw = X * (np.sqrt(ww) if ww is not None else 1.0)
    n = Xw.shape[0]
    if center: Xw = Xw - Xw.mean(0)
    G = np.zeros((D, D))
    for i in range(0, n, 20000): B = Xw[i:i+20000]; G += B.T @ B
    lam = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    cum = np.cumsum(lam) / lam.sum()
    return {"PR": float(lam.sum() ** 2 / (lam ** 2).sum()), "n90": int(np.searchsorted(cum, 0.90) + 1), "n95": int(np.searchsorted(cum, 0.95) + 1), "n99": int(np.searchsorted(cum, 0.99) + 1),
            "top5_share": float(cum[4]), "top1_share": float(cum[0]), "rank": int((lam > lam[0] * 1e-10).sum())}
res["spectra"] = {}
for lvl, X in (("core", C), ("sense", M)):
    for wn_, ww in (("raw", None), ("idf", w)):
        for c in (False, True):
            res["spectra"][f"{lvl}|{wn_}|{'centered' if c else 'uncentered'}"] = spec(X, ww, c)
# normalised-row variant (cosine geometry: unit rows), IDF-weighted
def unit(X, ww):
    Xw = X * (np.sqrt(ww) if ww is not None else 1.0); n = np.linalg.norm(Xw, axis=1, keepdims=True); return np.where(n > 0, Xw / np.maximum(n, 1e-12), 0)
for lvl, X in (("core", C), ("sense", M)):
    for wn_, ww in (("raw", None), ("idf", w)):
        res["spectra"][f"{lvl}|{wn_}|unit_rows_uncentered"] = spec(unit(X, ww), None, False)
        res["spectra"][f"{lvl}|{wn_}|unit_rows_centered"] = spec(unit(X, ww), None, True)
json.dump(res, open(S + "/m3_axes_eigen.json", "w"), indent=1)
print("done", time.time()-t0)
