"""USVS content-word BASELINE SCORECARD (read-only).  python scorecard.py  [~2 min]
Part A (production artifact, data/usvs, 607 base axes, no writes): core- and sense-level cosine metrics +/- IDF,
        sign-opposition counts, axis usage, eigenspectra, 5 concrete pairs.
Part B (live scorer): the RESEARCH_NOTES tree cases.
Held-out core numbers and SimLex come from separate commands listed in the report (h1_harness.py, probe_simlex.py)."""
import sys, json, random
import numpy as np
sys.path.insert(0, "/home/user/NAOMI/consciousness_transformer/src")
from nltk.corpus import wordnet as wn
from nsm_ct.ground.usvs import load_usvs
DATA = "/home/user/NAOMI/consciousness_transformer/data/usvs"
u = load_usvs(DATA, extended=False); D = len(u.axes)
core = {w: i for i, w in enumerate(u.core_words)}; sidx = {s: i for i, s in enumerate(u.sense_ids)}
C = u.core_coords.astype(np.float64)
M = np.zeros((len(u.sense_ids), D)); M[np.repeat(np.arange(len(M)), np.diff(u.sense_indptr)), u.sense_axis_idx] = u.sense_axis_val
w_idf = np.log((len(M) + 1.0) / ((M != 0).sum(0) + 1.0))              # == naomi.score.axis_weights on the base artifact
kd = np.array(["attr" if a.startswith("attr:") else "lex" if a.startswith("lex:") else "prime" for a in u.axes])

def wcos(A, B, w=None):
    w = np.ones(A.shape[1]) if w is None else w
    num = (A * B * w).sum(1); den = np.sqrt((A * A * w).sum(1) * (B * B * w).sum(1))
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
def auc(pos, neg):                                                   # P(pos > neg) + .5 P(tie)
    x = np.concatenate([pos, neg]); uq, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

# ---- pair sets (WordNet) ----
rng = random.Random(0)
syn = {}                                                             # lemma pairs sharing a synset, both in core
for ss in wn.all_synsets():
    ls = sorted({l.name().lower() for l in ss.lemmas() if l.name().lower() in core})
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)): syn.setdefault((ls[i], ls[j]), 1)
syn = list(syn)
mfs = {x: (u.senses_of(x) or [None])[0] for x in u.core_words}
ant = {}                                                             # direct lemma antonym pointers, both lemmas in core
for ss in wn.all_synsets():
    for l in ss.lemmas():
        for a in l.antonyms():
            x, y = l.name().lower(), a.name().lower()
            if x in core and y in core and x != y: ant[(frozenset((x, y)), frozenset((ss.name(), a.synset().name())))] = (x, y, ss.name(), a.synset().name())
ant = list(ant.values())
W, N = len(u.core_words), len(u.sense_ids)
rc = [(rng.randrange(W), rng.randrange(W)) for _ in range(20000)]; rs = [(rng.randrange(N), rng.randrange(N)) for _ in range(20000)]
synS = [(sidx[mfs[a]], sidx[mfs[b]]) for a, b in syn if mfs[a] and mfs[b] and mfs[a] != mfs[b]]
def cosl(X, prs, w): return wcos(X[[a for a, b in prs]], X[[b for a, b in prs]], w)
R = {}
for lvl, X, S_, A_, Rn in (("core", C, [(core[a], core[b]) for a, b in syn], [(core[a], core[b]) for a, b, *_ in ant], rc),
                           ("sense", M, synS, [(sidx[s1], sidx[s2]) for _, _, s1, s2 in ant], rs)):
    for wn_, w in (("raw", None), ("idf", w_idf)):
        cs, ca, cr = cosl(X, S_, w), cosl(X, A_, w), cosl(X, Rn, w)
        R[f"{lvl}|{wn_}"] = dict(syn_n=len(cs), ant_n=len(ca), syn_mean=cs.mean(), ant_mean=ca.mean(), rand_mean=cr.mean(),
            AUC_syn_vs_rand=auc(cs, cr), P_ant_less_similar_than_rand=auc(-ca, -cr), P_ant_closer_than_random_synonym=1 - auc(cs, ca))
# sign opposition (core): antonym pairs with a strongly sign-opposed axis (product < -0.01) vs random pairs
A_ = np.array([(core[a], core[b]) for a, b, *_ in ant]); pr = C[A_[:, 0]] * C[A_[:, 1]]
R["sign"] = dict(core_ant_strongly_opposed=float((pr < -0.01).any(1).mean()), core_ant_any_opposed=float((pr < -1e-6).any(1).mean()),
                 core_rand_strongly_opposed=float(((C[[a for a, b in rc]] * C[[b for a, b in rc]]) < -0.01).any(1).mean()),
                 sense_min_value=float(M.min()), sense_ant_opposed=float(((M[A_[:, 0] * 0 + [sidx[s1] for *_, s1, s2 in ant]] * M[[sidx[s2] for *_, s1, s2 in ant]]) < 0).any(1).mean()))
# axis usage
fs = (M != 0).mean(0); E = (M ** 2)
R["axes"] = dict(SOMETHING=float(fs[u.axes.index("SOMETHING")]), ONE=float(fs[u.axes.index("ONE")]), PART=float(fs[u.axes.index("PART")]), WHEN=float(fs[u.axes.index("WHEN")]),
    n_empty_sense_axes=int(((M != 0).sum(0) == 0).sum()), prime_share_sqnorm_raw=float(E[:, kd == "prime"].sum() / E.sum()),
    prime_share_sqnorm_idf=float((E * w_idf)[:, kd == "prime"].sum() / (E * w_idf).sum()), lex_share_sqnorm_idf=float((E * w_idf)[:, kd == "lex"].sum() / (E * w_idf).sum()),
    mean_axes_per_sense=float((M != 0).sum(1).mean()))
def spec(X, center):
    n = X.shape[0]; G = np.zeros((D, D)); mu = X.mean(0) if center else 0
    for i in range(0, n, 20000): B = X[i:i + 20000] - mu; G += B.T @ B
    lam = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None); cum = np.cumsum(lam) / lam.sum()
    return dict(PR=float(lam.sum() ** 2 / (lam ** 2).sum()), n90=int(np.searchsorted(cum, .9) + 1), n99=int(np.searchsorted(cum, .99) + 1))
R["spectra"] = {f"{l}|{c}": spec(X, c == "centered") for l, X in (("core", C), ("sense", M)) for c in ("uncentered", "centered")}
R["spectra_idf_sense_centered"] = spec(M * np.sqrt(w_idf), True)
# concrete pairs (first sense of the given POS)
def first(word, pos):
    ss = u.senses_of(word); return next((s for s in ss if s.rsplit(".", 2)[1].replace("s", "a") == pos), ss[0] if ss else None)
for a, b, p in (("happy", "unhappy", "a"), ("happy", "sad", "a"), ("dog", "cat", "n"), ("eat", "chew", "v"), ("bone", "stone", "n"), ("dog", "bone", "n"), ("hot", "cold", "a")):
    va, vb = M[[sidx[first(a, p)]]], M[[sidx[first(b, p)]]]
    R[f"pair {a}/{b} sense raw|idf"] = [float(wcos(va, vb)[0]), float(wcos(va, vb, w_idf)[0])]
# Part B: tree cases
from naomi.reader import make_node
from naomi.score import score_pair
from naomi.types import REL_ROOT, Tree
ue = load_usvs(DATA)
def Nn(rel, tok, idx, pos, *k): n = make_node(rel, tok, idx, pos); n.children = list(k); return n
ate = lambda v="ate", s="dog", o="bone": Tree(sentence="", root=Nn(REL_ROOT, v, 2, "VERB", Nn("SUBJECT", s, 1, "NOUN"), Nn("OBJECT", o, 4, "NOUN")))
she = lambda a: Tree(sentence="", root=Nn(REL_ROOT, "was", 1, "AUX", Nn("SUBJECT", "she", 0, "PRON"), Nn("COMPLEMENT", a, 3, "ADJ")))
for k, (x, y) in {"tree dog->cat": (ate(), ate(s="cat")), "tree ate->chewed": (ate(), ate(v="chewed")), "tree bone->stone": (ate(), ate(o="stone")),
                  "tree happy->sad": (she("happy"), she("sad")), "tree happy->unhappy": (she("happy"), she("unhappy"))}.items():
    ps = score_pair(x, y, ue); R[k] = dict(meaning=ps.meaning, overall=ps.overall, reasons=[r["kind"] for r in ps.reasons])
print(json.dumps(R, indent=1, default=float))
