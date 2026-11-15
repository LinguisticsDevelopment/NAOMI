"""Task 2: sign opposition counts, counterfactual signed-blend, and the 40 concrete pairs at both levels."""
from scorecard_lib import *
from nsm_ct.ground.usvs import _mean_unit_coord
from nsm_ct.ground.definition_graph import content_words
from nsm_ct.ground.usvs import SENSE_W_GENUS, SENSE_W_GLOSS
from nsm_ct.wordnet import _wn
u = load(); C = u.core_coords.astype(np.float64); M = dense_senses(u).astype(np.float64)
P = pickle.load(open(S + "/pairs.pkl", "rb")); w = idf_from(M.astype(np.float32)); D = len(u.axes); kd = kinds(u.axes)
core = {x: i for i, x in enumerate(u.core_words)}; sidx = {s: i for i, s in enumerate(u.sense_ids)}
res = {"sense_min": float(M.min()), "core_min": float(C.min()), "core_max": float(C.max())}
def signopp(A, B, thr=0.0):
    pr = A * B
    neg = (pr < -thr)
    return neg
sets_core = {
 "direct_ant_737": [(core[t[0]], core[t[1]]) for t in P["ant"]],
 "tiered_default_ant": [(core[a], core[b]) for a, b, t in P["ant_tiered"] if t in ("direct", "satellite_head") and a in core and b in core],
 "tiered_all_ant": [(core[a], core[b]) for a, b, t in P["ant_tiered"] if a in core and b in core],
 "syn": [(core[a], core[b]) for a, b, p in P["syn"]],
}
rng = random.Random(1); W = len(u.core_words)
sets_core["rand"] = [(rng.randrange(W), rng.randrange(W)) for _ in range(10000)]
res["core_signopp"] = {}
for name, prs in sets_core.items():
    A = C[[a for a, b in prs]]; B = C[[b for a, b in prs]]
    pr = A * B
    anyneg = (pr < -1e-6).any(1); strong = (pr < -0.01).any(1)
    nneg = (pr < -1e-6).sum(1)
    attr_neg = (pr[:, kd == "attr"] < -1e-6).any(1)
    cs = wcos(A, B)
    res["core_signopp"][name] = {"n": len(prs), "frac_any_axis_product_lt_0": float(anyneg.mean()), "frac_any_axis_product_lt_-0.01": float(strong.mean()),
        "mean_n_opposed_axes": float(nneg.mean()), "frac_opposed_on_attr_axis": float(attr_neg.mean()), "frac_cos_lt_0": float((cs < 0).mean()),
        "mean_cos": float(cs.mean()), "frac_both_have_any_negative_entry": float((((A < -1e-6).any(1)) & ((B < -1e-6).any(1))).mean()),
        "frac_either_has_any_negative_entry": float((((A < -1e-6).any(1)) | ((B < -1e-6).any(1))).mean())}
res["sense_signopp_direct_ant"] = {"n": len(P["ant"]), "frac_any_axis_product_lt_0": float(((M[[sidx[t[2]] for t in P["ant"]]] * M[[sidx[t[3]] for t in P["ant"]]]) < 0).any(1).mean())}
# --- sign loss: reconstruct pre-sparsification blend (copy of sense_usvs_weights up to the top-k step) for sampled senses
wn = _wn(); placed = {x: u.core_coords[i] for x, i in core.items()}
from nsm_ct.wordnet import all_senses
gl = {sid: (g, lem) for sid, g, lx, lem in all_senses()}
def blend(sid):
    gloss = gl[sid][0]; syn = wn.synset(sid)
    hl = [l.name().lower() for h in (syn.hypernyms() + syn.instance_hypernyms()) for l in h.lemmas() if "_" not in l.name()]
    parts = []
    r = _mean_unit_coord(hl, placed)
    if r is not None: parts.append((SENSE_W_GENUS * r[1], r[0]))
    r = _mean_unit_coord(content_words(gloss or ""), placed)
    if r is not None: parts.append((SENSE_W_GLOSS * r[1], r[0]))
    if not parts: return None
    b = np.zeros(D)
    for ww_, v in parts: b += ww_ * v
    return b
rs = random.Random(2); sample = rs.sample(u.sense_ids[:117659], 4000)
neg_mass, tot_mass, has_neg, n_ok, negattr, dropped_attr_neg_axes = 0.0, 0.0, 0, 0, 0.0, 0
for sid in sample:
    try: b = blend(sid)
    except Exception: continue
    if b is None: continue
    n_ok += 1; ng = np.clip(-b, 0, None); neg_mass += ng.sum(); tot_mass += np.abs(b).sum(); has_neg += int((b < -1e-6).any())
    negattr += ng[kd == "attr"].sum()
res["sign_loss_blend"] = {"n_sampled": n_ok, "frac_senses_blend_has_negative_entry": has_neg / n_ok, "negative_mass_share_of_abs_mass": neg_mass / tot_mass, "share_of_negative_mass_on_attr_axes": negattr / neg_mass}
# counterfactual: signed blend (no top-k/clip) cosines for direct antonym synset pairs vs syn-through-hypernym pairs vs random
def cf(prs):
    A, B = [], []
    for a, b in prs:
        try: x, y = blend(a), blend(b)
        except Exception: continue
        if x is None or y is None: continue
        A.append(x); B.append(y)
    A, B = np.array(A), np.array(B)
    return A, B
antp = [(t[2], t[3]) for t in P["ant"]][:737]
Aa, Ba = cf(antp)
rr = [(rs.choice(u.sense_ids), rs.choice(u.sense_ids)) for _ in range(1500)]
Ar, Br = cf(rr)
res["counterfactual_signed_blend_unsparsified"] = {"note": "reconstruction of sense_usvs_weights BEFORE the top-24 / clip step, signs kept; NOT the shipped artifact",
   "ant_n": len(Aa), "ant_mean_cos": float(wcos(Aa, Ba).mean()), "rand_n": len(Ar), "rand_mean_cos": float(wcos(Ar, Br).mean()),
   "ant_frac_opposed_axis": float(((Aa * Ba) < -1e-6).any(1).mean()), "rand_frac_opposed_axis": float(((Ar * Br) < -1e-6).any(1).mean()),
   "ant_AUC_below_rand": float(auc(-wcos(Aa, Ba), -wcos(Ar, Br)))}
# --- 40 pairs
PAIRS = [("happy","unhappy","a"),("happy","sad","a"),("hot","cold","a"),("big","small","a"),("good","bad","a"),("love","hate","v"),("up","down","r"),
 ("open","closed","a"),("alive","dead","a"),("rich","poor","a"),("win","lose","v"),("buy","sell","v"),("give","take","v"),("dog","cat","n"),("dog","bone","n"),
 ("bone","stone","n"),("eat","chew","v"),("eat","drink","v"),("run","walk","v"),("king","queen","n"),("man","woman","n"),("boy","girl","n"),("day","night","n"),
 ("red","blue","a"),("red","color","n"),("apple","fruit","n"),("apple","pear","n"),("car","vehicle","n"),("happy","joyful","a"),("big","large","a"),("quick","fast","a"),
 ("dog","puppy","n"),("hot","warm","a"),("tall","short","a"),("old","young","a"),("fast","slow","a"),("true","false","a"),("begin","end","v"),
 ("dog","justice","n"),("table","run","n")]
def pick(word, pos):
    ss = u.senses_of(word)
    for s in ss:
        p = s.rsplit(".", 2)[1]; p = "a" if p == "s" else p
        if p == pos: return s
    return ss[0] if ss else None
rows = []
for a, b, p in PAIRS:
    sa, sb = pick(a, p), pick(b, p)
    r = {"a": a, "b": b, "pos": p, "sa": sa, "sb": sb}
    if a in core and b in core:
        ca, cb = C[core[a]][None], C[core[b]][None]
        r["core_raw"] = float(wcos(ca, cb)[0]); r["core_idf"] = float(wcos(ca, cb, w)[0])
    else: r["core_raw"] = r["core_idf"] = None
    if sa and sb:
        va, vb = M[sidx[sa]][None], M[sidx[sb]][None]
        r["sense_raw"] = float(wcos(va, vb)[0]); r["sense_idf"] = float(wcos(va, vb, w)[0])
    else: r["sense_raw"] = r["sense_idf"] = None
    rows.append(r)
res["pairs40"] = rows
json.dump(res, open(S + "/m4_signs_pairs.json", "w"), indent=1)
for r in rows: print(f'{r["a"]:7}{r["b"]:9}{r["pos"]} core {r["core_raw"] if r["core_raw"] is None else round(r["core_raw"],2)}/{r["core_idf"] if r["core_idf"] is None else round(r["core_idf"],2)}  sense {r["sense_raw"] if r["sense_raw"] is None else round(r["sense_raw"],2)}/{r["sense_idf"] if r["sense_idf"] is None else round(r["sense_idf"],2)}  {r["sa"]} {r["sb"]}')
print(json.dumps({k: v for k, v in res.items() if k != "pairs40"}, indent=1))
