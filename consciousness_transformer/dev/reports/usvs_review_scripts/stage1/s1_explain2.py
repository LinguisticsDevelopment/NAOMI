"""Degree / negation arithmetic, unrelated-pair false positives with chains, everyday pairs."""
from s1con import *
from s1_test import build_placement, fz, BUILD
import evalib, time
from nltk.corpus import wordnet as wn
WIDE = dict(et="sim+deriv+pert+also", hyp=True, cap=None, merge=False)
CFG = {"frozen_a": dict(imp="a"), "frozen_b": dict(imp="b"), "wide_a": dict(imp="a", cfg=WIDE, d=0.7, steps=3), "wide_b": dict(imp="b", cfg=WIDE, d=0.7, steps=3)}
PLS = {cn: build_placement(**kw, build_pairs=BUILD) for cn, kw in CFG.items()}
nm = lambda i: sids[i]
B1 = lambda a, b: float(evalib.wcos_pairs(evalib.M, NAMED_W1)(np.array([(a, b)]))[0])
drop = json.load(open(SC + "p1_axes.json"))["dropped_k12"]; NAMED_W1 = evalib.IDF.copy()
for a in drop: NAMED_W1[axes.index(a)] = 0
fold = np.load(os.path.join(HERE, "fold.npy")); foldof = {}
for (a, b), f in zip(ANT, fold): foldof[int(a)] = f; foldof[int(b)] = f
partner = {}
for a, b in ANT: partner.setdefault(int(a), int(b)); partner.setdefault(int(b), int(a))
def full(named, da, db):
    ag, g = agree_dicts(da, db); return (1 - g) * named + g * ag
res = {"degree": {}, "orderings": {}, "unrel": {}}
WORDS = ["happy", "hot", "big", "good", "fast", "rich", "clean", "strong", "easy", "safe"]
A_ADJ = adj(SPREAD)
def pick_syn(x, pl, si):
    # synonym of X: a SPREAD-edge neighbour (similar_to/also_see/deriv/pertainym) placed on the same scale with same sign; else best B1 on that scale
    cands = [n for n, _ in A_ADJ[x] if pos[n] == pos[x] and n in pl and si in pl[n] and pl[n][si][0] * pl[x][si][0] > 0]
    if cands: return max(cands, key=lambda n: B1(x, n))
    return None
def row(x, pl, scales, si, opp):
    dx = {si: pl[x][si]}                    # this scale only
    do = {si: pl[opp][si]} if opp in pl and si in pl[opp] else {si: (-1.0 if pl[x][si][0] > 0 else 1.0, 1.0)}
    named_xo = B1(x, opp); r = {}
    r["X~very"] = full(1.0, dx, op_very(dx)); r["X~slightly"] = full(1.0, dx, op_slightly(dx)); r["X~not"] = full(1.0, dx, op_not(dx))
    r["not~opp"] = full(named_xo, op_not(dx), do); r["X~opp"] = full(named_xo, dx, do)
    s = pick_syn(x, pl, si); r["X~syn"] = full(B1(x, s), dx, {si: pl[s][si]}) if s is not None else None; r["syn"] = nm(s) if s is not None else None
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
for cn in ("frozen_a", "frozen_b", "wide_a", "wide_b"):
    pl, scales, bf, mem = PLS[cn]; res["degree"][cn] = {}
    for w in WORDS:
        x = sidx[wn.synsets(w, "a")[0].name()]; f = foldof.get(x, "no recorded antonym")
        if x not in pl: res["degree"][cn][w] = dict(sense=nm(x), fold=f, placed=False); continue
        own = [k for k in pl[x] if x in scales[k][0] or x in scales[k][1]]
        si = max(own, key=lambda k: pl[x][k][1]) if own else max(pl[x], key=lambda k: pl[x][k][1]); P_, Q_, _ = scales[si]
        is_member = bool(own)
        other = Q_ if (x in P_ or (not is_member and pl[x][si][0] > 0)) else P_
        opp = partner.get(x) if partner.get(x) in other else sorted(other)[0]
        res["degree"][cn][w] = dict(sense=nm(x), fold=f, placed=True, recorded_member_in_build=is_member, pos=round(pl[x][si][0], 3), imp=round(pl[x][si][1], 3),
                                    opp=nm(opp), n_scales=len(pl[x]), **row(x, pl, scales, si, opp))
    print(cn, flush=True); [print("  ", w, v) for w, v in res["degree"][cn].items()]
# aggregate ordering test over placed NON-member adjectives (|pos|>=0.2), seeded sample of 2000
rg = random.Random(11)
for cn in ("frozen_a", "frozen_b", "wide_a", "wide_b"):
    pl, scales, bf, mem = PLS[cn]; mset = set()
    for P_, Q_, _ in scales: mset |= P_ | Q_
    cand = [x for x in pl if pos[x] == "a" and x not in mset and any(abs(p) >= 0.2 for p, i in pl[x].values())]
    smp = rg.sample(cand, min(2000, len(cand))); cnt = collections.Counter(); n = 0; sums = collections.defaultdict(float)
    for x in smp:
        si = max(pl[x], key=lambda k: pl[x][k][1] if abs(pl[x][k][0]) >= 0.2 else -1)
        if abs(pl[x][si][0]) < 0.2: continue
        P_, Q_, _ = scales[si]; opp = sorted(Q_ if pl[x][si][0] > 0 else P_)[0]
        r = row(x, pl, scales, si, opp); n += 1
        for k in ("X~very", "X~slightly", "X~not", "not~opp", "X~opp"): sums[k] += r[k]
        cnt["O1 very>slightly>not"] += r["X~very"] > r["X~slightly"] > r["X~not"]
        cnt["O2 X~not > X~opp"] += r["X~not"] > r["X~opp"]
        cnt["O3 not~opp > X~opp"] += r["not~opp"] > r["X~opp"]
        cnt["O4 not~opp < 1 (not on the opposite)"] += r["not~opp"] < 0.999
        cnt["O6 not~opp > X~not (near the opposite rather than near X)"] += r["not~opp"] > r["X~not"]
        if r["X~syn"] is not None: cnt["O5 X~syn > X~not (n with syn)"] += r["X~syn"] > r["X~not"]; cnt["n_syn"] += 1
    res["orderings"][cn] = {"n_words": n, "n_candidates": len(cand), "frac": {k: round(v / (cnt["n_syn"] if k.startswith("O5") else n), 3) for k, v in cnt.items() if k != "n_syn"}, "n_with_syn": cnt["n_syn"],
                            "mean_sim": {k: round(v / n, 3) for k, v in sums.items()}}
    print(cn, res["orderings"][cn], flush=True)
# unrelated control: worst changes with chains
for cn in ("frozen_a", "wide_a"):
    pl, scales, bf, mem = PLS[cn]; rows = []
    for k in ("rand_n", "rand_v", "rand_a", "unrel30"):
        arr = SETS[k]; ag, g, sh = pair_scale(pl, arr); s = combine(NAMED["B1"][k], ag, g); d = NAMED["B1"][k] - s
        for i in range(len(arr)):
            if sh[i] and abs(d[i]) > 0.02: rows.append((float(d[i]), k, int(arr[i, 0]), int(arr[i, 1]), float(NAMED["B1"][k][i]), float(s[i]), float(g[i])))
    rows.sort(key=lambda r: -r[0]); out = {"n_changed_gt_0.02": len(rows), "pushed_apart_gt_0.15": sum(1 for r in rows if r[0] > 0.15), "pulled_together_gt_0.15": sum(1 for r in rows if r[0] < -0.15)}
    def expl(r):
        d_, k, a, b, nmd, sc, g = r; a_, b_ = pl[a], pl[b]; si = max(a_.keys() & b_.keys(), key=lambda s_: min(a_[s_][1], b_[s_][1]))
        return dict(set=k, pair=f"{nm(a)} / {nm(b)}", B1=round(nmd, 3), construction=round(sc, 3), g=round(g, 3), pos=(round(a_[si][0], 2), round(b_[si][0], 2)),
                    chain_a=chain(a, si, bf, scales)[0], chain_b=chain(b, si, bf, scales)[0])
    out["worst_pushed_apart"] = [expl(r) for r in rows[:5]]; out["worst_pulled_together"] = [expl(r) for r in rows[::-1][:5]]
    res["unrel"][cn] = out; print(cn, json.dumps(out)[:2500], flush=True)
# everyday 17 + unrel30 placements (frozen_a, frozen_b, wide_a)
ev = {}
Rm = json.load(open(os.path.join(HERE, "results_main.json")))["methods"]
names17 = [f"{a}/{b}" for a, b, _, _ in B["EV"]]
ev["names17"] = names17
for m in ("B1", "B2", "construction_a", "construction_b"): ev[m] = Rm[m]["everyday17"]
rec = []
antt = {tuple(sorted(map(int, p))): f for p, f in zip(ANT, fold)}
for a, b, sa, sb in B["EV"]: rec.append(antt.get(tuple(sorted((sidx[sa], sidx[sb]))), "not recorded"))
ev["recorded_fold"] = rec
for cn in ("wide_a",):
    pl = PLS[cn][0]; ag, g, sh = pair_scale(pl, SETS["everyday"]); ev[cn] = [round(float(x), 3) for x in combine(NAMED["B1"]["everyday"], ag, g)]
ev["shared_scale_frozen_a"] = [bool(x) for x in pair_scale(PLS["frozen_a"][0], SETS["everyday"])[2]]
res["everyday"] = ev
json.dump(res, open(os.path.join(HERE, "results_explain2.json"), "w"), indent=1, default=str)
print("done")
