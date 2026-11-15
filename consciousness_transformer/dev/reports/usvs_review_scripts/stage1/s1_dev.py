"""DEV grid: choose decay per step, steps, edge types, hyp step, degree cap, attribute merge, importance definition.
Selection rule (fixed before looking at any test number): among configs with no damage (dev-subset hyp AUC, sim-vs-random AUC and SimLex each
>= B1 - 0.01), maximise AUC(dev opposite pair LESS similar than random pair), ties -> fewer steps, then larger decay. Same rule for B2."""
import time, itertools, sys
from s1con import *
rng = random.Random(5)
sub = lambda k, n: SETS[k][rng.sample(range(len(SETS[k])), n)] if len(SETS[k]) > n else SETS[k]
DEVSETS = {"ant_dev": SETS["ant_dev"], "sim_te": SETS["sim_te"], "syn_mfs": sub("syn_mfs", 3000), "hyp_te": sub("hyp_te", 3000), "rand": sub("rand", 6000),
           "rand_a": sub("rand_a", 2000), "simlex": SETS["simlex"]}
# named B1 for the subsets (recompute from the keyed table by pair lookup)
import evalib
w1 = None
def named_for(arr, which="B1"):
    drop = json.load(open(SC + "p1_axes.json"))["dropped_k12"]; w = evalib.IDF.copy()
    if which == "B1":
        for a in drop: w[axes.index(a)] = 0
    return evalib.wcos_pairs(evalib.M, w)(arr)
DN = {k: named_for(v) for k, v in DEVSETS.items()}
def dev_metrics(sims):
    A_ = lambda x, y: auc(x, y)
    return {"ant<rand": A_(-sims["ant_dev"], -sims["rand"]), "hyp": None, "sim_v_rand": A_(sims["sim_te"], sims["rand_a"]),
            "hyp": A_(sims["hyp_te"], sims["rand"]), "simlex": spearman(sims["simlex"], B["simlex_gold"]), "ant_mean": float(sims["ant_dev"].mean()),
            "mfs>ant": A_(sims["syn_mfs"], sims["ant_dev"])}
BASE = dev_metrics(DN)
def run_one(cfg):
    scales = make_scales(SETS["ant_build"], cfg["merge"]) if cfg["merge"] else make_scales(SETS["ant_build"], False)
    bf = scale_bfs(scales, cfg["et"], 4, cfg["hyp"], cfg["cap"], ("m" if cfg["merge"] else "n"))
    ea = allnb(SETS["ant_build"])
    out = []
    for d, steps, imp in itertools.product((0.5, 0.7, 0.85), (1, 2, 3, 4), cfg["imps"]):
        pl, _ = place(scales, bf, d, steps, imp, ea)
        sims = {}; cov = None
        for k, arr in DEVSETS.items():
            ag, g, sh = pair_scale(pl, arr); sims[k] = combine(DN[k], ag, g)
            if k == "ant_dev": cov = float(sh.mean())
        m = dev_metrics(sims); m.update(cfg={**cfg, "et": "+".join(cfg["et"]), "imps": None}, d=d, steps=steps, imp=imp, cov_shared=cov, n_scales=len(scales), n_placed=len(pl))
        out.append(m)
    return out
if __name__ == "__main__":
    which = sys.argv[1]; res = []
    ETS = [("sim",), ("sim", "deriv"), ("sim", "deriv", "pert"), ("sim", "deriv", "pert", "also")]
    grid = [dict(et=e, hyp=h, cap=c, merge=mg, imps=("a", "b")) for e in ETS for h in (False, True) for c in (None, 30) for mg in (False, True)]
    part = int(which); chunk = grid[part::4]
    t0 = time.time()
    for cfg in chunk:
        res += run_one(cfg); print(part, cfg["et"], cfg["hyp"], cfg["cap"], cfg["merge"], round(time.time() - t0), flush=True)
        json.dump({"base": BASE, "res": res}, open(os.path.join(HERE, f"dev_part{part}.json"), "w"))
