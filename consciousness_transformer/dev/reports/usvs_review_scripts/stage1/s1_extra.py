"""Dev-grid summary + how the frozen construction reaches the test pairs it separates."""
from s1con import *
from s1_test import build_placement, BUILD
import collections
out = {}
res = []
for p in range(4): d = json.load(open(os.path.join(HERE, f"dev_part{p}.json"))); res += d["res"]; base = d["base"]
tab = collections.defaultdict(lambda: [0, 0, 0])
for r in res:
    k = (r["imp"], r["cfg"]["et"], r["cfg"]["hyp"]); 
    if r["ant<rand"] > tab[k][0]: tab[k] = [r["ant<rand"], r["cov_shared"], r["steps"]]
out["dev_best_by_edges"] = {f"{k[0]}|{k[1]}|hyp={k[2]}": [round(v[0], 4), round(v[1], 3), v[2]] for k, v in sorted(tab.items())}
out["dev_ant<rand_range_over_all_768"] = [round(min(r["ant<rand"] for r in res), 4), round(max(r["ant<rand"] for r in res), 4), round(base["ant<rand"], 4)]
out["dev_simv_hyp_simlex_range"] = {k: [round(min(r[k] for r in res), 3), round(max(r[k] for r in res), 3), round(base[k], 3)] for k in ("hyp", "sim_v_rand", "simlex")}
pl, scales, bf, mem = build_placement("a", BUILD)
arr = SETS["ant_test"]; ag, g, sh = pair_scale(pl, arr); how = collections.Counter(); chains = []
for i in np.where(sh)[0]:
    a, b = int(arr[i, 0]), int(arr[i, 1]); common = pl[a].keys() & pl[b].keys(); si = max(common, key=lambda s: min(pl[a][s][1], pl[b][s][1]))
    ty = []
    for x in (a, b):
        dd = bf[si][0] if x in bf[si][0] and bf[si][0][x][0] <= 1 else bf[si][1]
        ty.append(TYPE.get(dd[x][2], "member") if dd[x][2] else "member")
    how[tuple(sorted(ty))] += 1
    if len(chains) < 6 and pl[a][si][0] * pl[b][si][0] < 0: chains.append(f"{sids[a]} / {sids[b]}: " + chain(a, si, bf, scales)[0] + "  ||  " + chain(b, si, bf, scales)[0])
out["frozen_test_sharing_pairs_reach_types"] = {"+".join(k): v for k, v in how.items()}; out["examples_separated_test_pairs"] = chains
json.dump(out, open(os.path.join(HERE, "results_extra.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:4000])
