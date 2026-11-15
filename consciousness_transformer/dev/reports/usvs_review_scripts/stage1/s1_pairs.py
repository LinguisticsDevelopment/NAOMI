"""Eval pair sets for stage 1 + named-block similarities B0 (IDF) and B1 (IDF, 12 ubiquitous axes dropped)."""
from s1lib import *
import evalib
from nltk.corpus import wordnet as wn
fold = np.load(os.path.join(HERE, "fold.npy")); ant = E["ant"]
SETS = {f"ant_{t}": ant[fold == t] for t in ("build", "dev", "test")}
for k in ("sim_te", "syn_mfs", "hyp_te", "rand", "rand_n", "rand_v", "rand_a", "rand_r", "simlex", "everyday"): SETS[k] = S0[k]
UN = [("dog","bone","n"),("king","queen","n"),("dog","cat","n"),("bone","stone","n"),("table","river","n"),("car","tree","n"),("doctor","hammer","n"),
 ("apple","mountain","n"),("book","rain","n"),("house","fish","n"),("bread","window","n"),("horse","moon","n"),("eat","drink","v"),("eat","chew","v"),
 ("run","sleep","v"),("cut","walk","v"),("sing","climb","v"),("throw","think","v"),("break","smile","v"),("read","swim","v"),("red","blue","a"),
 ("tall","green","a"),("loud","wet","a"),("round","hungry","a"),("sweet","heavy","a"),("young","yellow","a"),("proud","cheap","a"),("sharp","empty","a"),
 ("quiet","square","a"),("brave","sticky","a")]
rows, names = [], []
for a, b, p in UN:
    f = lambda w: next(s for s in wn.synsets(w, p) if s.name() in sidx).name()
    rows.append((sidx[f(a)], sidx[f(b)])); names.append(f"{a}/{b}")
SETS["unrel30"] = np.array(rows)
M, IDF = evalib.M, evalib.IDF
drop = json.load(open(SC + "p1_axes.json"))["dropped_k12"]; w1 = IDF.copy()
for a in drop: w1[axes.index(a)] = 0
NAMED = {"B0": {k: evalib.wcos_pairs(M, IDF)(v) for k, v in SETS.items()}, "B1": {k: evalib.wcos_pairs(M, w1)(v) for k, v in SETS.items()}}
pickle.dump(dict(SETS=SETS, NAMED=NAMED, unrel_names=names, EV=B["EV"]), open(SC + "s1_pairs.pkl", "wb"))
print({k: len(v) for k, v in SETS.items()}, {b: {k: round(float(v.mean()), 3) for k, v in d.items() if k.startswith(("ant", "rand", "unrel"))} for b, d in NAMED.items()})
