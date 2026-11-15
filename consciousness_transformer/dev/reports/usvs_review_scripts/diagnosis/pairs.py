"""Build WordNet pair sets (word-level for core, synset-level for senses). Saves pairs.pkl"""
from common import *
import collections
from nltk.corpus import wordnet as wn
u = load()
core = {w: i for i, w in enumerate(u.core_words)}
sidx = {s: i for i, s in enumerate(u.sense_ids)}
pos_of = lambda p: "a" if p == "s" else p
rng = random.Random(0)
def mfs(word):
    s = u.senses_of(word); return s[0] if s else None
P = {}
# synonyms: lemma pairs sharing a synset, both in core
syn = {}
for ss in wn.all_synsets():
    ls = sorted({l.name().lower() for l in ss.lemmas() if l.name().lower() in core})
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)):
            syn.setdefault((ls[i], ls[j]), pos_of(ss.pos()))
P["syn"] = [(a, b, p) for (a, b), p in syn.items()]
# direct antonyms (lemma-level pointers), both lemmas in core
ant = {}
for ss in wn.all_synsets():
    for l in ss.lemmas():
        for a in l.antonyms():
            x, y = l.name().lower(), a.name().lower()
            if x == y: continue
            k = tuple(sorted((x, y)))
            sk = tuple(sorted(((x, ss.name()), (y, a.synset().name()))))
            ant.setdefault(k + (ss.name() if x <= y else a.synset().name(), a.synset().name() if x <= y else ss.name()), pos_of(ss.pos()))
# store: (a, b, synA, synB, pos), unique by synset pair
antl = []
seen = set()
for ss in wn.all_synsets():
    for l in ss.lemmas():
        for a in l.antonyms():
            k = tuple(sorted((ss.name(), a.synset().name())))
            kk = k + tuple(sorted((l.name().lower(), a.name().lower())))
            if kk in seen: continue
            seen.add(kk)
            antl.append((l.name().lower(), a.name().lower(), ss.name(), a.synset().name(), pos_of(ss.pos())))
P["ant_all"] = antl  # all direct lemma antonym pointers (deduped), any vocab
P["ant"] = [t for t in antl if t[0] in core and t[1] in core]
# tiered antonym edges from artifact (word level)
P["ant_tiered"] = [(a, b, t) for a, b, t in u.antonyms]
# hypernym pairs (direct), synset level + core lemma level
def first_core_lemma(ss):
    for l in ss.lemmas():
        n = l.name().lower()
        if n in core: return n
    return None
hyp = []; cohyp = []
for ss in wn.all_synsets():
    if ss.pos() not in "nv": continue
    hs = ss.hypernyms() + ss.instance_hypernyms()
    for h in hs:
        hyp.append((ss.name(), h.name(), first_core_lemma(ss), first_core_lemma(h), ss.pos()))
P["hyp"] = hyp
hypo = collections.defaultdict(list)
for ss in wn.all_synsets():
    if ss.pos() not in "nv": continue
    for h in ss.hypernyms():
        hypo[h.name()].append(ss.name())
for h, kids in hypo.items():
    if len(kids) < 2: continue
    pr = [(a, b) for i, a in enumerate(kids) for b in kids[i + 1:]]
    rng.shuffle(pr)
    for a, b in pr[:3]:
        sa, sb = wn.synset(a), wn.synset(b)
        cohyp.append((a, b, first_core_lemma(sa), first_core_lemma(sb), sa.pos()))
P["cohyp"] = cohyp
pickle.dump(P, open(S + "/pairs.pkl", "wb"))
print({k: len(v) for k, v in P.items()})
c = collections.Counter(t[2] for t in P["syn"]); print("syn pos", c)
print("ant pos", collections.Counter(t[4] for t in P["ant"]))
print("hyp core-core", sum(1 for t in hyp if t[2] and t[3]), "cohyp core-core", sum(1 for t in cohyp if t[2] and t[3]))
# --- project-style synonyms (wn.synsets(word) incl. morphy), for comparability with harness counts
proj = set()
for w in u.core_words:
    for ss in wn.synsets(w):
        for l in ss.lemmas():
            n = l.name().lower()
            if n != w and n in core: proj.add(tuple(sorted((w, n))))
P["syn_proj"] = sorted(proj)
pickle.dump(P, open(S + "/pairs.pkl", "wb"))
print("syn_proj", len(P["syn_proj"]))
