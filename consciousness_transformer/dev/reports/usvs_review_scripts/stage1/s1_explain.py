"""Coverage, traceability, degree/negation, unrelated-pair false positives, everyday pairs. FROZEN = dev-chosen; WIDE = illustrative (not dev-chosen)."""
from s1con import *
from s1_test import build_placement, fz, BUILD
import evalib, time
from nltk.corpus import wordnet as wn
t0 = time.time()
WIDE = dict(et="sim+deriv+pert+also", hyp=True, cap=None, merge=False)
CFG = {"frozen_a": dict(imp="a"), "frozen_b": dict(imp="b"), "wide_a": dict(imp="a", cfg=WIDE, d=0.7, steps=3), "wide_b": dict(imp="b", cfg=WIDE, d=0.7, steps=3)}
ALLPAIRS = ANT
nm = lambda i: sids[i]
gloss = lambda i: wn.synset(sids[i]).definition()[:70]
TYPE = {"sim": "similar_to", "deriv": "derivational", "pert": "pertainym", "also": "also_see", "hypd": "has-hypernym", "hypu": "has-hyponym"}
def chain(node, si, bf, scales):
    dp, dq = bf[si]; P_, Q_, _ = scales[si]
    sides = [(k, d_) for k, d_ in ((dp[node][0], dp) if node in dp else (None, None), (dq[node][0], dq) if node in dq else (None, None)) if k is not None]
    k, d_ = min(sides, key=lambda x: x[0]); path = [node]; ety = []
    x = node
    while d_[x][1] is not None: ety.append(TYPE[d_[x][2]]); x = d_[x][1]; path.append(x)
    src = x; other = sorted(Q_ if src in P_ else P_)[0]
    s = nm(path[-1]); 
    for i in range(len(path) - 2, -1, -1): s += f" -[{ety[i]}]-> {nm(path[i])}"
    return s + f"   [{nm(src)} is the recorded opposite of {nm(other)}]", src, other
res = {}
# ---------- coverage ----------
gold, gpos = B["gold"]; ok = pos != "?"; nwn = int(ok.sum())
def cov_of(pl):
    flag = np.zeros(N, bool); flag[list(pl)] = True
    return {"senses": int((flag & ok).sum()), "share": f4((flag & ok).sum() / nwn), "by_pos": {p: f4(flag[pos == p].mean()) for p in "nvar"},
            "gold_tokens": f4(flag[gold].mean()), "gold_by_upos": {u: f4(flag[gold[gpos == u]].mean()) for u in ("NOUN", "VERB", "ADJ", "ADV")}}
res["coverage"] = {}
for tag, pairs in (("build_only(60%)", BUILD), ("all_3802_recorded", ALLPAIRS)):
    for steps in range(0, 5):
        for cn, cfg in (("frozen_edges(sim+hyp)", fz["a"]["cfg"]), ("wide_edges(sim+deriv+pert+also+hyp)", WIDE)):
            pl, scales, bf, mem = build_placement("a", pairs, cfg=cfg, d=0.7, steps=steps)
            res["coverage"][f"{tag}|{cn}|steps={steps}"] = cov_of(pl)
print("coverage done", round(time.time() - t0), flush=True)
# ---------- scale stats + multi-scale + both ends ----------
res["scales"] = {}; PLS = {}
for cn, kw in CFG.items():
    pl, scales, bf, mem = build_placement(**kw, build_pairs=BUILD); PLS[cn] = (pl, scales, bf, mem)
    sz = np.array([len(m) for m in mem.values()]); multi = sum(1 for v in pl.values() if len(v) > 1)
    both = [(si, n) for si, m in mem.items() for n, v in m.items() if v[2] and abs(v[0]) < 1]
    mid = sum(1 for v in pl.values() for (p, i) in v.values() if abs(p) < 0.2 and i > 0)
    ex = [f"{nm(n)} on scale {nm(min(scales[si][0]))}|{nm(min(scales[si][1]))} pos={mem[si][n][0]:.2f}" for si, n in both[:6]]
    res["scales"][cn] = {"n_scales": len(scales), "placed_senses": len(pl), "scale_size_placed_nodes": {"min": int(sz.min()), "median": float(np.median(sz)), "p90": float(np.percentile(sz, 90)), "max": int(sz.max()),
                         "share_size_2": f4((sz == 2).mean())}, "senses_on_multiple_scales": multi, "max_scales_one_sense": max(len(v) for v in pl.values()),
                         "reached_from_both_ends(non-members)": len(both), "examples_both_ends": ex, "placements_with_|pos|<0.2": mid}
    print(cn, res["scales"][cn], flush=True)
# ---------- test pairs under each config (frozen = the tested one; wide = exploratory) ----------
res["test_pairs_by_config"] = {}
for cn, (pl, scales, bf, mem) in PLS.items():
    arr = SETS["ant_test"]; ag, g, sh = pair_scale(pl, arr)
    pole = []
    for i in np.where(sh)[0]:
        a, b = arr[i]; common = pl[int(a)].keys() & pl[int(b)].keys()
        pole.append(max(((min(pl[int(a)][k][1], pl[int(b)][k][1])), "opposite" if pl[int(a)][k][0] * pl[int(b)][k][0] < 0 else "same", abs(pl[int(a)][k][0]) , abs(pl[int(b)][k][0])) for k in common)[1])
    sims = {k: combine(NAMED["B1"][k], *pair_scale(pl, SETS[k])[:2]) for k in ("ant_test", "rand", "sim_te", "hyp_te", "syn_mfs", "rand_a", "simlex")}
    cp = collections.Counter(pole)
    res["test_pairs_by_config"][cn] = {"share_pairs_sharing_a_scale": f4(sh.mean()), "n_sharing": int(sh.sum()), "poles_of_sharing_pairs": dict(cp),
        "ant<rand_all": f4(auc(-sims["ant_test"], -sims["rand"])), "ant_mean_all": f4(sims["ant_test"].mean()), "hyp": f4(auc(sims["hyp_te"], sims["rand"])),
        "sim_v_rand": f4(auc(sims["sim_te"], sims["rand_a"])), "simlex": f4(spearman(sims["simlex"], B["simlex_gold"]))}
    print("test", cn, res["test_pairs_by_config"][cn], flush=True)
# ---------- traceability ----------
tr = []
rg = random.Random(3); members = {}
for a, b in SETS["ant_test"]:
    for x in (int(a), int(b)): members[x] = pos[x]
buildmem = set(BUILD.ravel().tolist())
out = []
for pcode, label in (("a", "adjective"), ("n", "noun"), ("v", "verb")):
    chosen = []
    for cn in ("frozen_a", "wide_a"):
        pl, scales, bf, mem = PLS[cn]
        cands = sorted(x for x, p in members.items() if p == pcode and x in pl and x not in buildmem and x not in [c[0] for c in chosen])
        rg.shuffle(cands)
        for x in cands:
            if len(chosen) >= 5: break
            chosen.append((x, cn))
    for x, cn in chosen:
        pl, scales, bf, mem = PLS[cn]; plb = PLS[cn.replace("_a", "_b")][0]
        si = max(pl[x], key=lambda k: pl[x][k][1]); txt, src, other = chain(x, si, bf, scales)
        out.append(dict(pos=label, config=cn, sense=nm(x), gloss=gloss(x), scale=f"{nm(min(scales[si][0]))} ({gloss(min(scales[si][0]))}) <-> {nm(min(scales[si][1]))} ({gloss(min(scales[si][1]))})",
                        position=f4(pl[x][si][0]), imp_a=f4(pl[x][si][1]), imp_b=f4(plb[x][si][1]), n_scales=len(pl[x]), chain=txt))
res["trace"] = out
for o in out: print(o, flush=True)
json.dump(res, open(os.path.join(HERE, "results_explain1.json"), "w"), indent=1, default=str)
print("part1 done", round(time.time() - t0))
