"""Three-way split (60/20/20, seed 20261115) of the 3,802 recorded antonym pairs by FAMILY.
Family = union-find over antonym pairs joined by: shared member / shared cluster head, or a derivational, pertainym or
attribute edge between members of two pairs. also_see is NOT used for grouping (with it one family holds 1,263 of 3,802 pairs)."""
from s1lib import *
import collections
ant = E["ant"]; P = len(ant); head = head_of(); nb = adj(("deriv", "pert", "attr"))
par = list(range(P))
def f(i):
    while par[i] != i: par[i] = par[par[i]]; i = par[i]
    return i
def uni(a, b):
    ra, rb = f(a), f(b)
    if ra != rb: par[ra] = rb
owner = collections.defaultdict(list)
for pid, (a, b) in enumerate(ant):
    for x in (int(a), int(b)): owner[x].append(pid); owner[int(head[x])].append(pid)
for k, l in owner.items():
    for q in l[1:]: uni(l[0], q)
for pid, (a, b) in enumerate(ant):
    for x in (int(a), int(b)):
        for n, _ in nb[x]:
            for q in owner.get(n, []): uni(pid, q)
fam = np.array([f(i) for i in range(P)])
fams = collections.defaultdict(list)
for i, r in enumerate(fam): fams[int(r)].append(i)
keys = sorted(fams); rng = random.Random(20261115); rng.shuffle(keys)
target = {"build": 0.6 * P, "dev": 0.2 * P, "test": 0.2 * P}; cnt = {k: 0 for k in target}; fold = np.empty(P, object)
for k in keys:
    fk = max(target, key=lambda t: (target[t] - cnt[t]) / target[t])
    for i in fams[k]: fold[i] = fk
    cnt[fk] += len(fams[k])
pp = pos[ant[:, 0]]
stats = {"n_pairs": P, "n_families": len(fams), "largest_families": sorted(map(len, fams.values()), reverse=True)[:6],
         "folds": {t: {"all": int((fold == t).sum()), **{p: int(((fold == t) & (pp == p)).sum()) for p in "nvar"}} for t in target}}
print(json.dumps(stats))
np.save(os.path.join(HERE, "fold.npy"), fold.astype(str)); np.save(os.path.join(HERE, "family.npy"), fam)
json.dump(stats, open(os.path.join(HERE, "folds.json"), "w"), indent=1)
