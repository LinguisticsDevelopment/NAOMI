from s1_dev import *
out = []
for r in (0, 1, 2):
    for d in ((0.5,) if r == 0 else (0.3, 0.5, 0.7, 0.9)):
        sims = {}; st_ant = None
        for k, arr in DEVSETS.items():
            s, st = b2_sims(SETS["ant_build"], r, d, arr, DN[k]); sims[k] = s
            if k == "ant_dev": st_ant = st
        m = dev_metrics(sims); m.update(r=r, d=d, frac_ant_flagged=float((st_ant > 0).mean())); out.append(m); print(m, flush=True)
ok = [m for m in out if m["hyp"] >= BASE["hyp"] - .01 and m["sim_v_rand"] >= BASE["sim_v_rand"] - .01 and m["simlex"] >= BASE["simlex"] - .01]
best = sorted(ok, key=lambda m: (-round(m["ant<rand"], 4), m["r"], -m["d"]))[0]; print("B2 frozen", best)
json.dump({"all": out, "frozen": best, "base": BASE}, open(os.path.join(HERE, "b2_dev.json"), "w"), indent=1)
