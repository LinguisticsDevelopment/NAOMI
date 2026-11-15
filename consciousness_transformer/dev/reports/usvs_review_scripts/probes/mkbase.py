"""Build cached base: sense matrix, edge lists by type, 80/20 split, eval pair sets, gold tokens, simlex."""
import sys, gzip, json, pickle, random
import numpy as np
sys.path.insert(0, "/home/user/NAOMI/consciousness_transformer/src")
sys.path.insert(0, "/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_diag")
from common import load, dense_senses
from nltk.corpus import wordnet as wn
P = "/tmp/claude-0/-home-user-NAOMI/664d9faf-b57c-55b6-97de-b37fd7e015b5/scratchpad/usvs_probes/"
u = load()
M = dense_senses(u); np.save(P + "M.npy", M)
sids = list(u.sense_ids); sidx = {s: i for i, s in enumerate(sids)}
axes = list(u.axes)
posc = lambda p: "a" if p == "s" else p
pos = np.array([("?" if s.startswith(("interj.", "fn.")) else posc(s.rsplit(".", 2)[1])) for s in sids])
E = {k: set() for k in "similar also deriv pert attr hyp vgroup entail cause ant".split()}
def add(k, a, b):
    if a == b: return
    if a in sidx and b in sidx:
        x, y = sidx[a], sidx[b]
        E[k].add((x, y) if k in ("hyp", "similar") else (min(x, y), max(x, y)))  # hyp: (child,parent); similar:(sat,head)
for ss in wn.all_synsets():
    n = ss.name()
    for h in ss.hypernyms() + ss.instance_hypernyms(): add("hyp", n, h.name())
    for h in ss.similar_tos(): add("similar", n, h.name()) if ss.pos() == "s" else None
    for h in ss.also_sees(): add("also", n, h.name())
    for h in ss.attributes(): add("attr", n, h.name())
    for h in ss.verb_groups(): add("vgroup", n, h.name())
    for h in ss.entailments(): add("entail", n, h.name())
    for h in ss.causes(): add("cause", n, h.name())
    for l in ss.lemmas():
        for d in l.derivationally_related_forms(): add("deriv", n, d.synset().name())
        for d in l.pertainyms(): add("pert", n, d.synset().name())
        for d in l.antonyms(): add("ant", n, d.synset().name())
E = {k: sorted(v) for k, v in E.items()}
print({k: len(v) for k, v in E.items()})
rng = random.Random(0)
split = {}
for k, v in E.items():
    idx = list(range(len(v))); rng.shuffle(idx); nte = int(round(0.2 * len(v)))
    te = set(idx[:nte]); split[k] = np.array([i in te for i in range(len(v))])
E = {k: np.array(v, dtype=np.int64).reshape(-1, 2) for k, v in E.items()}
# eval pair sets (indices)
rng = random.Random(1)
wnidx = np.where(pos != "?")[0]
S = {}
S["ant_te"] = E["ant"][split["ant"]]
S["ant_tr"] = E["ant"][~split["ant"]]
S["sim_te"] = E["similar"][split["similar"]]
core = u.core_words
mfs = {w: (u.senses_of(w) or [None])[0] for w in core}
syn = {}
for ss in wn.all_synsets():
    ls = sorted({l.name().lower() for l in ss.lemmas() if l.name().lower() in mfs})
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)): syn[(ls[i], ls[j])] = 1
sp = [(sidx[mfs[a]], sidx[mfs[b]]) for a, b in syn if mfs[a] and mfs[b] and mfs[a] != mfs[b] and mfs[a] in sidx and mfs[b] in sidx]
S["syn_mfs"] = np.array(sp)
h = E["hyp"][split["hyp"]]
if len(h) > 10000: h = h[rng.sample(range(len(h)), 10000)]
S["hyp_te"] = h
# sisters: children of same parent; keep pairs with >=1 of two child edges in test
par = {}
htr = {(int(a), int(b)): bool(t) for (a, b), t in zip(E["hyp"], split["hyp"])}
kids = {}
for (a, b) in htr: kids.setdefault(b, []).append(a)
sis = []
for p_, ks in kids.items():
    if len(ks) < 2: continue
    prs = [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]]
    rng.shuffle(prs)
    for a, b in prs[:3]:
        if htr[(a, p_)] or htr[(b, p_)]: sis.append((a, b))
sis = np.array(sis); S["sis_te"] = sis[rng.sample(range(len(sis)), min(10000, len(sis)))]
S["rand"] = np.array([(int(wnidx[rng.randrange(len(wnidx))]), int(wnidx[rng.randrange(len(wnidx))])) for _ in range(20000)])
for p_ in "nvar":
    pool = np.where(pos == p_)[0]
    S["rand_" + p_] = np.array([(int(pool[rng.randrange(len(pool))]), int(pool[rng.randrange(len(pool))])) for _ in range(4000)])
for k in S: S[k] = S[k][S[k][:, 0] != S[k][:, 1]]
# everyday pairs (senses as in diagnosis 2c)
EV = [("happy","unhappy","happy.a.01","unhappy.a.01"),("happy","sad","happy.a.01","sad.a.01"),("hot","cold","hot.a.01","cold.a.01"),
 ("good","bad","good.a.01","bad.a.01"),("big","small","large.a.01","small.a.01"),("love","hate","love.v.01","hate.v.01"),
 ("buy","sell","buy.v.01","sell.v.01"),("alive","dead","alive.a.01","dead.a.01"),("dog","cat","dog.n.01","cat.n.01"),
 ("dog","bone","dog.n.01","bone.n.01"),("bone","stone","bone.n.01","rock.n.01"),("eat","chew","eat.v.01","chew.v.01"),
 ("eat","drink","eat.v.01","drink.v.01"),("king","queen","king.n.01","queen.n.01"),("red","blue","red.s.01","blue.s.01"),
 ("big","large","large.a.01","large.a.01"),("quick","fast","quick.s.01","fast.a.01")]
S["everyday"] = np.array([(sidx[a], sidx[b]) for _, _, a, b in EV])
# simlex at sense level: MFS sense of given POS
sl = []
for line in open("/home/user/NAOMI/consciousness_transformer/data/SimLex-999.txt").read().split("\n")[1:]:
    p = line.split("\t")
    if len(p) < 4: continue
    w1, w2, pp, g = p[0].lower(), p[1].lower(), p[2].lower(), float(p[3])
    pp = {"a": "a", "n": "n", "v": "v"}[pp]
    def f(w):
        ss = [s for s in wn.synsets(w) if posc(s.pos()) == pp]
        return sidx.get(ss[0].name()) if ss else None
    a, b = f(w1), f(w2)
    if a is not None and b is not None: sl.append((a, b, g, w1, w2, pp))
S["simlex"] = np.array([(a, b) for a, b, *_ in sl]); simlex_gold = np.array([g for _, _, g, *_ in sl])
simlex_pos = np.array([x[5] for x in sl])
# gold v6 content tokens
gold = []
with gzip.open("/home/user/NAOMI/consciousness_transformer/runs/encoder_gold_v6.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line)
        ps = d["pos"]
        for t in d["token_sense_candidates"]:
            i = t["index"]
            if i < len(ps) and ps[i] in ("NOUN", "VERB", "ADJ", "ADV"):
                s = t.get("chosen_sense") or (t["sense_candidates"][0] if t.get("sense_candidates") else None)
                gold.append((sidx.get(s, -1), ps[i]))
gold = np.array([g[0] for g in gold]), np.array([g[1] for g in gold])
print("gold content tokens", len(gold[0]), "unmapped", int((gold[0] < 0).sum()))
print({k: len(v) for k, v in S.items()}, "simlex", len(sl))
pickle.dump(dict(sids=sids, axes=axes, pos=pos, E=E, split=split, S=S, EV=EV, simlex_gold=simlex_gold, simlex_pos=simlex_pos, gold=gold), open(P + "base.pkl", "wb"))
