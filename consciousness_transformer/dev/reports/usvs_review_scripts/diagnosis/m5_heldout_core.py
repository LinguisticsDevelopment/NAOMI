"""Core-level profile on GENUINELY held-out pairs (placement re-run by h2_heldout_core.py, M24 rule), +/- IDF, kinds masks, topK."""
from common import *
from nsm_ct.ground.closeness import discrimination
H = pickle.load(open(S + "/heldout_core.pkl", "rb"))
u = load(); M = dense_senses(u); w = idf_from(M); kd = kinds(u.axes)
assert H["axes"] == list(u.axes), "axes differ"
words = H["words"]; idx = {x: i for i, x in enumerate(words)}; C = H["C"].astype(np.float64)
rng = random.Random(0)
excl = set(H["all_syn"]) | set(H["all_ant"]) | {tuple(sorted(p)) for p in H["all_isa"]}
rand = []
while len(rand) < 4000:
    a, b = rng.choice(words), rng.choice(words)
    if a != b and tuple(sorted((a, b))) not in excl: rand.append((a, b))
def cosv(prs, ww=None, mask=None):
    A = C[[idx[a] for a, b in prs]]; B = C[[idx[b] for a, b in prs]]; return wcos(A, B, ww, mask)
out = {}
sets = {"syn": H["test_syn"], "ant": H["test_ant"], "isa": H["test_isa"], "sim": H["test_sim"], "rand": rand}
def prof(ww, mask):
    c = {k: cosv(v, ww, mask) for k, v in sets.items()}
    return {"n": {k: len(v) for k, v in sets.items()}, "means": {k: round(float(v.mean()), 4) for k, v in c.items()},
            "AUC_syn_vs_rand": round(auc(c["syn"], c["rand"]), 4), "syn_gt_ant (P cos_syn>cos_ant)": round(auc(c["syn"], c["ant"]), 4),
            "AUC_ant_below_rand": round(auc(-c["ant"], -c["rand"]), 4), "AUC_isa_vs_rand": round(auc(c["isa"], c["rand"]), 4), "AUC_sim_vs_rand": round(auc(c["sim"], c["rand"]), 4)}
out["raw"] = prof(None, None); out["idf"] = prof(w, None)
for kn, m in {"prime": kd == "prime", "attr": kd == "attr", "lex": kd == "lex", "prime+attr": kd != "lex", "prime+lex": kd != "attr", "attr+lex": kd != "prime"}.items():
    out[f"raw|{kn}"] = prof(None, m.astype(float)); out[f"idf|{kn}"] = prof(w, m.astype(float))
energy = w * (C ** 2).sum(0); order = np.argsort(-energy)
for K in (20, 50, 100, 200):
    m = np.zeros(len(w)); m[order[:K]] = 1
    out[f"idf|top{K}"] = prof(w, m); out[f"raw|top{K}"] = prof(None, m)
# project-style discrimination (harness metric) cross-check: syn>ant via discrimination() raw
from nsm_ct.ground.meaning_value import cosine
placed = {x: H["C"][i] for x, i in idx.items()}
out["harness_discrimination_syn_ant_raw"] = discrimination(H["test_syn"], H["test_ant"], lambda a, b: cosine(placed[a], placed[b]), placed)
json.dump(out, open(S + "/m5_heldout_core.json", "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict): print(k, v["AUC_syn_vs_rand"], v["syn_gt_ant (P cos_syn>cos_ant)"], v["AUC_ant_below_rand"], v["means"])
    else: print(k, v)
