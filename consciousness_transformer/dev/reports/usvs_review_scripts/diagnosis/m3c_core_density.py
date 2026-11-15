from common import *
u = load(); C = u.core_coords.astype(np.float64); kd = kinds(u.axes)
nz = np.abs(C) > 1e-6; dense = nz.sum(1) >= 400
out = {"n_dense_core": int(dense.sum())}
for t in (0.01, 0.05, 0.1, 0.3):
    c = (np.abs(C[dense]) > t).sum(1); out[f"axes_per_dense_core_word_abs>{t}"] = {"mean": float(c.mean()), "median": float(np.median(c)), "p5": float(np.percentile(c, 5)), "p95": float(np.percentile(c, 95))}
    out[f"axes_with_>{t}_for_ge_50pct_of_dense_words"] = int(((np.abs(C[dense]) > t).mean(0) >= 0.5).sum())
a = np.abs(C[dense]); out["dense_value_quantiles_nonzero"] = {str(q): float(np.percentile(a[a > 1e-6], q)) for q in (10, 50, 90, 99)}
neg = C[dense] < -1e-6
out["frac_dense_words_with_neg_entry"] = float(neg.any(1).mean()); out["mean_neg_entries_per_dense_word"] = float(neg.sum(1).mean())
negmag = np.abs(np.minimum(C[dense], 0)).sum(1) / np.abs(C[dense]).sum(1); out["neg_mass_share_per_dense_word_mean"] = float(negmag.mean())
out["neg_axis_count_ge_50pct_dense_words"] = int((neg.mean(0) >= 0.5).sum())
tanh_like = float((np.abs(C) == 1.0).mean()); out["frac_entries_abs_eq_1"] = tanh_like
print(json.dumps(out, indent=1)); json.dump(out, open(S + "/m3c_core_density.json", "w"), indent=1)
