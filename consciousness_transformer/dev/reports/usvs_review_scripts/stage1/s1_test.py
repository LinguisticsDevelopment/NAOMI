"""TEST: frozen dev settings, looked at once. B0, B1, B2, construction (importance a, b), null control (5 seeds)."""
from s1con import *
fz = json.load(open(os.path.join(HERE, "frozen_settings.json")))["frozen"]; b2f = json.load(open(os.path.join(HERE, "b2_dev.json")))["frozen"]
BUILD = SETS["ant_build"]; KEYS = list(SETS)
def build_placement(imp, build_pairs, cfg=None, d=None, steps=None, merge=None):
    c = fz[imp]["cfg"]; cfg = cfg or c
    et = tuple(cfg["et"].split("+")) if isinstance(cfg["et"], str) else cfg["et"]
    d = d if d is not None else fz[imp]["d"]; steps = steps if steps is not None else fz[imp]["steps"]
    scales = make_scales(build_pairs, cfg["merge"]); bf = [(bfs(P_, adj(et), 4, cfg["hyp"], cfg["cap"], adj(("hypd", "hypu"))), bfs(Q_, adj(et), 4, cfg["hyp"], cfg["cap"], adj(("hypd", "hypu")))) for P_, Q_, _ in scales]
    ea = allnb(build_pairs); pl, mem = place(scales, bf, d, steps, imp, ea)
    return pl, scales, bf, mem
def construct_sims(pl, base="B1"):
    sims, shared, g_, ag_ = {}, {}, {}, {}
    for k in KEYS:
        ag, g, sh = pair_scale(pl, SETS[k]); sims[k] = combine(NAMED[base][k], ag, g); shared[k] = sh; g_[k] = g; ag_[k] = ag
    return sims, shared, g_, ag_
def has_scale(pl, arr): return np.array([(int(a) in pl) and (int(b) in pl) for a, b in arr])
if __name__ == "__main__":
    R = {"frozen_construction": {k: {kk: (v if kk != "cfg" else v) for kk, v in fz[k].items()} for k in fz}, "frozen_b2": b2f}
    t0 = time.time() if "time" in globals() else None
    import time; t0 = time.time()
    PL = {}; SIM = {}
    for imp in "ab":
        pl, scales, bf, mem = build_placement(imp, BUILD); PL[imp] = pl
        SIM[imp], sh, g_, ag_ = construct_sims(pl); SIM[imp + "_sh"] = sh; SIM[imp + "_g"] = g_; SIM[imp + "_ag"] = ag_
        print(imp, "placed senses", len(pl), "scales", len(scales), round(time.time() - t0), flush=True)
    SIM["B0"] = NAMED["B0"]; SIM["B1"] = NAMED["B1"]
    b2 = {}; b2st = {}
    for k in KEYS: b2[k], b2st[k] = b2_sims(BUILD, b2f["r"], b2f["d"], SETS[k], NAMED["B1"][k])
    SIM["B2"] = b2
    placedmask = has_scale(PL["a"], SETS["ant_test"]); sharedmask = SIM["a_sh"]["ant_test"]
    b2mask = b2st["ant_test"] > 0
    R["coverage_test_ant"] = {"both_ends_have_a_scale": f4(placedmask.mean()), "share_a_scale": f4(sharedmask.mean()), "b2_flagged": f4(b2mask.mean()),
                              "placed_same_as_b": bool((has_scale(PL["b"], SETS["ant_test"]) == placedmask).all())}
    print(R["coverage_test_ant"], flush=True)
    views = {"all": None, "placed": placedmask, "shared": sharedmask, "b2flagged": b2mask}
    R["methods"] = {}
    for name, sims in (("B0", SIM["B0"]), ("B1", SIM["B1"]), ("B2", SIM["B2"]), ("construction_a", SIM["a"]), ("construction_b", SIM["b"])):
        R["methods"][name] = {v: metrics(sims, m, "ant_test") for v, m in views.items()}
        R["methods"][name]["unrelated"] = unrel_report(sims, NAMED["B1"])
        R["methods"][name]["in_sample_build_ant_mean"] = f4(sims["ant_build"].mean()); R["methods"][name]["dev_ant_mean"] = f4(sims["ant_dev"].mean())
        R["methods"][name]["everyday17"] = [f4(x) for x in sims["everyday"]]; R["methods"][name]["unrel30"] = [f4(x) for x in sims["unrel30"]]
    print("methods done", round(time.time() - t0), flush=True)
    # ---------- null control: right members permuted within POS ----------
    nulls = []
    for seed in range(5):
        r = random.Random(100 + seed); right = list(BUILD[:, 1]); byp = {}
        for i, x in enumerate(right): byp.setdefault(pos[x], []).append(i)
        npairs = BUILD.copy()
        for p, idxs in byp.items():
            vals = [right[i] for i in idxs]; r.shuffle(vals)
            for i, v in zip(idxs, vals): npairs[i, 1] = v
        real = set(map(tuple, np.sort(ANT, 1).tolist()))
        ok = [i for i in range(len(npairs)) if npairs[i, 0] != npairs[i, 1]]; npairs = npairs[ok]
        out = {}
        for imp in "ab":
            pl, *_ = build_placement(imp, npairs)
            sims, sh, _, _ = construct_sims(pl)
            out[imp] = {"all": metrics(sims, None, "ant_test"), "shared": metrics(sims, sh["ant_test"], "ant_test"), "shared_cov": f4(sh["ant_test"].mean()),
                        "unrelated": unrel_report(sims, NAMED["B1"]), "pushed_apart_frac_all_pairs_ant_test": None}
        nulls.append(out); print("null", seed, round(time.time() - t0), flush=True)
    R["null"] = nulls
    json.dump(R, open(os.path.join(HERE, "results_main.json"), "w"), indent=1)
    pickle.dump({"b2st_test": b2st["ant_test"]}, open(os.path.join(HERE, "b2st.pkl"), "wb"))
    print("done", round(time.time() - t0))
