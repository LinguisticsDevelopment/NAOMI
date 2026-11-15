import torch, hashlib, numpy as np
import p3b
from p3b import *
torch.set_num_threads(1)
Za, pl, _ = fit(20, 0); Zb, _, _ = fit(20, 0)
print("1-thread repro: maxdiff", float(np.abs(Za - Zb).max()), hashlib.sha1(Za.tobytes()).hexdigest()[:12], hashlib.sha1(Zb.tobytes()).hexdigest()[:12])
Z1 = np.load(P + "fit_k20.npy")
for nm, Z in (("run4thr", Z1), ("run1thrA", Za)):
    fn = lambda pr: (Z[pr[:, 0]] * Z[pr[:, 1]]).sum(1); r = evaluate(fn, placed=pl)
    print(nm, {x: r[x] for x in ("ant_mean", "P_sim>ant", "P_mfs>ant", "AUC_ant<rand", "AUC_hyp>rand")})
# seed sensitivity: seed 1
Zs, _, _ = fit(20, 1); fn = lambda pr: (Zs[pr[:, 0]] * Zs[pr[:, 1]]).sum(1); r = evaluate(fn, placed=pl)
print("seed1 (1thr)", {x: r[x] for x in ("ant_mean", "P_sim>ant", "P_mfs>ant", "AUC_ant<rand", "AUC_hyp>rand")}, "simlex", r["simlex_rho"])
