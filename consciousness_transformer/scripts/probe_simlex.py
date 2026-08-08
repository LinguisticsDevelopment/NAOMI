import sys; sys.path.insert(0, "src")
import numpy as np
from nsm_ct.ground.usvs import load_usvs
from nsm_ct.ground.meaning_value import cosine

u = load_usvs("data/usvs")

def spearman(x, y):
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v)); r[order] = np.arange(len(v))
        # average ties
        v = np.asarray(v)
        for val in np.unique(v):
            m = v == val
            r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])

def usvs_sim(a, b):
    ca, cb = u.word_coord(a), u.word_coord(b)
    if ca is not None and cb is not None:
        return cosine(ca, cb)
    sa, sb = u.senses_of(a), u.senses_of(b)
    if sa and sb:
        da, db = u.sense_dense(sa[0]), u.sense_dense(sb[0])
        if da is not None and db is not None and da.any() and db.any():
            return cosine(da, db)
    return None

# antonym-aware variant: signed edge store pushes known opposites down
def usvs_sim_ant(a, b):
    s = usvs_sim(a, b)
    if s is None:
        return None
    if b in u.antonyms_of(a) or a in u.antonyms_of(b):
        s = s - 1.0     # signed relational correction (the M18.3/M29 design)
    return s

rows = []
with open("data/SimLex-999.txt") as f:
    next(f)
    for line in f:
        p = line.split("\t")
        rows.append((p[0].lower(), p[1].lower(), float(p[3])))

for name, fn in (("plain cosine", usvs_sim), ("with antonym edges", usvs_sim_ant)):
    gold, pred, core_pairs = [], [], 0
    for a, b, g in rows:
        s = fn(a, b)
        if s is not None:
            gold.append(g); pred.append(s)
            if u.word_coord(a) is not None and u.word_coord(b) is not None:
                core_pairs += 1
    print(f"{name:20} rho={spearman(np.array(gold), np.array(pred)):.3f}  "
          f"coverage={len(gold)}/999 (core-core {core_pairs})")

# core-only subset (the propagation layer alone)
gold, pred = [], []
for a, b, g in rows:
    ca, cb = u.word_coord(a), u.word_coord(b)
    if ca is not None and cb is not None:
        s = cosine(ca, cb)
        if b in u.antonyms_of(a) or a in u.antonyms_of(b):
            s -= 1.0
        gold.append(g); pred.append(s)
print(f"{'core-only + edges':20} rho={spearman(np.array(gold), np.array(pred)):.3f}  n={len(gold)}")
