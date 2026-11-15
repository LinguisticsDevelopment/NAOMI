from s1lib import *
res = []; 
for p in range(4):
    d = json.load(open(os.path.join(HERE, f"dev_part{p}.json"))); res += d["res"]; base = d["base"]
print("base", base, len(res))
ok = [r for r in res if r["hyp"] >= base["hyp"] - .01 and r["sim_v_rand"] >= base["sim_v_rand"] - .01 and r["simlex"] >= base["simlex"] - .01]
print("feasible", len(ok))
frozen = {}
for imp in "ab":
    c = sorted([r for r in ok if r["imp"] == imp], key=lambda r: (-round(r["ant<rand"], 4), r["steps"], -r["d"]))
    frozen[imp] = c[0]
    print(imp, "top5"); [print("  ", round(r["ant<rand"], 4), r["cfg"], r["d"], r["steps"], "cov", round(r["cov_shared"], 3), "hyp", round(r["hyp"], 3), "sl", round(r["simlex"], 3)) for r in c[:5]]
    allr = sorted([r for r in res if r["imp"] == imp], key=lambda r: -r["ant<rand"])[:2]; print(" best-unconstrained", [(round(r["ant<rand"], 4), r["cfg"], r["d"], r["steps"], round(r["hyp"], 3)) for r in allr])
json.dump({"base_dev_B1": base, "frozen": frozen}, open(os.path.join(HERE, "frozen_settings.json"), "w"), indent=1)
