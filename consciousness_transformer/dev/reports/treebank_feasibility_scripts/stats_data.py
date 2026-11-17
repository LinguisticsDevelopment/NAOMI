import sys, json, glob, os, collections
sys.path.insert(0, os.path.dirname(__file__))
import conllu
ROOT = "/home/user/NAOMI/consciousness_transformer/data/ud"
out = {}
for d in ["EWT", "GUM", "LinES", "ParTUT", "es_gsd", "es_ancora"]:
    r = {}
    for f in sorted(glob.glob(f"{ROOT}/{d}/*.conllu")):
        sp = f.split("-ud-")[1].split(".")[0]
        ns = nt = n20 = n35 = nm = 0; lens = []
        pun = 0
        for s in conllu.read(f):
            toks = [t for t in s["toks"] if t["upos"] != "PUNCT"]
            L = len(toks); ns += 1; nt += len(s["toks"]); lens.append(L)
            n20 += L > 20; n35 += L > 35; nm += len(s["mwts"])
        lens.sort()
        r[sp] = {"sent": ns, "tok": nt, "gt20": round(n20 / ns, 3), "gt35": round(n35 / ns, 3), "mwt": nm,
                 "median_len": lens[len(lens) // 2]}
    out[d] = r
    print(d, json.dumps(r))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "data_stats.json"), "w"), indent=1)
