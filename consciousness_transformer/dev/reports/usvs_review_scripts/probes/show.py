import json, sys
R = json.load(open("results.json"))
cols = ["rand_mean","AUC_mfs_v_rand","AUC_sim_v_rand","P_sim>ant","P_mfs>ant","AUC_ant<rand","AUC_hyp>rand","AUC_sis>rand","simlex_rho"]
def row(k, extra=()):
    r = R[k]; return "| " + k + " | " + " | ".join(str(r.get(c)) for c in cols) + " | " + " | ".join(str(r.get(c)) for c in extra) + " |"
pat = sys.argv[1:]
for k in R:
    if any(p in k for p in pat): print(row(k))
