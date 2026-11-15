"""Reproduce the RESEARCH_NOTES tree-scorer cases with the live scorer (read-only)."""
import sys
sys.path.insert(0, "/home/user/NAOMI/consciousness_transformer/src")
from naomi.reader import make_node, CT_DIR
from naomi.score import score_pair, axis_weights
from naomi.types import REL_ROOT, Tree
from nsm_ct.ground.usvs import load_usvs
u = load_usvs(str(CT_DIR / "data" / "usvs"))
def N(rel, tok, idx, pos, *kids):
    n = make_node(rel, tok, idx, pos); n.children = list(kids); return n
T = lambda r: Tree(sentence="", root=r)
def ate(verb="ate", subj="dog", obj="bone", extra=()):
    return T(N(REL_ROOT, verb, 2, "VERB", N("SUBJECT", subj, 1, "NOUN"), *( [N("OBJECT", obj, 4, "NOUN")] if obj else [] ), *extra))
def she(adj, *k): return T(N(REL_ROOT, "was", 1, "AUX", N("SUBJECT", "she", 0, "PRON"), N("COMPLEMENT", adj, 3, "ADJ", *k)))
ref = ate()
cases = {
 "dog->cat": (ref, ate(subj="cat")), "ate->chewed": (ref, ate(verb="chewed")), "bone->stone": (ref, ate(obj="stone")),
 "ate->drank": (ref, ate(verb="drank")), "dog->justice": (ref, ate(subj="justice")), "bone->idea": (ref, ate(obj="idea")),
 "ate->ran (obj dropped)": (ref, ate(verb="ran", obj=None)),
 "she was happy vs sad": (she("happy"), she("sad")), "happy vs unhappy": (she("happy"), she("unhappy")),
 "happy vs not happy": (she("happy"), she("happy", N("SPECIFICATION", "not", 2, "PART"))),
 "hot vs cold": (she("hot"), she("cold")), "good vs bad": (she("good"), she("bad")), "happy vs joyful": (she("happy"), she("joyful")),
}
for idf in (True, False):
    print("IDF" if idf else "NO-IDF")
    for k, (a, b) in cases.items():
        ps = score_pair(a, b, u, idf=idf)
        print(f"  {k:26} overall {ps.overall:6.1f} meaning {ps.meaning:6.1f} shape {ps.shape:6.1f}  reasons {[r['kind'] for r in ps.reasons]}")
