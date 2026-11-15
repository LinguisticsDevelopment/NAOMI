"""Stage-1 shared library. Reads the cached base (edges, eval pair sets, 607-axis sense matrix) built by ../probes/mkbase.py."""
import sys, os, json, pickle, random
import numpy as np
PR = "/home/user/NAOMI/consciousness_transformer/dev/reports/usvs_review_scripts/probes"
SC = "/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_probes/"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PR)
sys.path.insert(0, "/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_diag")
from common import auc, _rank
B = pickle.load(open(SC + "base.pkl", "rb"))
sids, pos, E, S0 = B["sids"], B["pos"], B["E"], B["S"]
sidx = {s: i for i, s in enumerate(sids)}
N = len(sids)
axes = B["axes"]

def spearman(x, y):
    return float(np.corrcoef(_rank(np.asarray(x, float)), _rank(np.asarray(y, float)))[0, 1])
f4 = lambda x: None if x is None else round(float(x), 4)

# ---------- adjacency over explicit dictionary edges ----------
SPREAD = ("sim", "deriv", "pert", "also")          # the spec's spread edge types
NAME = {"sim": "similar_to", "deriv": "derivational", "pert": "pertainym", "also": "also_see", "hypd": "hypernym", "hypu": "hyponym",
        "attr": "attribute", "vgroup": "verb_group", "entail": "entailment", "cause": "cause", "ant": "antonym(recorded)"}
def _und(e):
    return np.concatenate([e, e[:, ::-1]]) if len(e) else e
EDGE = {"sim": _und(E["similar"]), "deriv": _und(E["deriv"]), "pert": _und(E["pert"]), "also": _und(E["also"]),
        "attr": _und(E["attr"]), "vgroup": _und(E["vgroup"]), "entail": _und(E["entail"]), "cause": _und(E["cause"]),
        "hypd": E["hyp"], "hypu": E["hyp"][:, ::-1]}      # hypd: child->parent (go up) ; hypu: parent->child
def adj_of(types):
    A = [[] for _ in range(N)]
    for t in types:
        for a, b in EDGE[t]: A[a].append((int(b), t))
    return A
_adj_cache = {}
def adj(types):
    k = tuple(types)
    if k not in _adj_cache: _adj_cache[k] = adj_of(types)
    return _adj_cache[k]

def head_of():
    h = np.arange(N)
    for sat, hd in E["similar"]: h[sat] = min(h[sat], hd) if h[sat] != sat else hd
    return h
