import sys, os, random, json
sys.path.insert(0, os.path.dirname(__file__))
import conllu
from nsm_ct.corpus import _tokenize_words
R = "/home/user/NAOMI/consciousness_transformer/data/ud"
def ud_forms(s):  # surface tokens (MWT surface split excluded): UD syntactic words, lowercased
    return [t["form"].lower() for t in s["toks"]]
def proj(s):
    return _tokenize_words(s["meta"]["text"])
res = {}
for name, path in [("EWT", "EWT/en_ewt-ud-dev.conllu"), ("GUM", "GUM/en_gum-ud-dev.conllu"), ("es_gsd", "es_gsd/es_gsd-ud-dev.conllu")]:
    sents = list(conllu.read(f"{R}/{path}"))
    same = sum(ud_forms(s) == proj(s) for s in sents)
    same_n = sum(len(ud_forms(s)) == len(proj(s)) for s in sents)
    res[name] = {"n": len(sents), "identical": round(same / len(sents), 3), "same_count": round(same_n / len(sents), 3)}
print(json.dumps(res))
random.seed(1)
sents = list(conllu.read(f"{R}/EWT/en_ewt-ud-dev.conllu"))
ex = [s for s in sents if (s["mwts"] or any(t["form"] in ("'s", "n't", "'m") or "'" in t["form"] or "-" in t["form"] for t in s["toks"])) and len(s["toks"]) < 22]
random.shuffle(ex)
for s in ex[:20]:
    u, p = ud_forms(s), proj(s)
    if u != p:
        a = [x for x in u if x not in p]; b = [x for x in p if x not in u]
        print("DIFF", s["meta"]["text"][:90], "| UD only:", a[:4], "| proj only:", b[:4])
    else:
        print("SAME", s["meta"]["text"][:90])
# class-level differences over EWT dev
import collections
c = collections.Counter()
for s in sents:
    u, p = ud_forms(s), proj(s)
    if u != p:
        for x in set(u) - set(p): c["UD:" + x] += 1
        for x in set(p) - set(u): c["PROJ:" + x] += 1
print(c.most_common(30))
