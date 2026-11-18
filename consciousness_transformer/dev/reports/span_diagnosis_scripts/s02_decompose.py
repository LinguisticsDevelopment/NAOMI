"""Q1 (decomposition), Q2 (anchor geometry), Q3 (question type), Q5 (wh attachment).  python s02_decompose.py val s2b_all"""
import sys, pickle, collections
import numpy as np
from common import *

split, name = sys.argv[1], sys.argv[2]
recs = pickle.load(open(OUTD / f"collect_{split}_{name}.pkl", "rb"))
stories, _, meta = load_split(split)
vj = json.load(open(RUN / "vocab.json"))
rels, poss = ["?"] + vj["rels"], ["?"] + vj["poss"]
ctx = m1.Ctx.get()
axes = list(ctx.usvs.axes)
out = {}


def tbl(title, groups, key_fn):
    """groups: dict label -> list of records (with cat). Prints n, oracle, model, gap, shares of total loss per category."""
    rows = []
    for lab, rs in groups.items():
        n = len(rs)
        orc = mean(r["oracle"] for r in rs)
        mod = mean(r["f_eff"] for r in rs)
        loss = sum(r["loss"] for r in rs)
        shares = {c: sum(r["loss"] for r in rs if r["cat"] == c) / max(loss, 1e-9) for c in CATS}
        cnt = {c: sum(r["cat"] == c for r in rs) / n for c in CATS}
        rows.append((lab, n, orc, mod, orc - mod, shares, cnt))
    print("\n##", title)
    print("| group | n | oracle | model | gap | " + " | ".join(f"{c} items% / loss%" for c in CATS) + " |")
    for lab, n, orc, mod, gap, sh, cnt in rows:
        print(f"| {lab} | {n} | {orc:.3f} | {mod:.3f} | {gap:.3f} | " + " | ".join(f"{100*cnt[c]:.1f} / {100*sh[c]:.1f}" for c in CATS) + " |")
    return rows


R = []
for r in recs:
    arr = stories[r["skey"]]
    cat, loss, best, f, acc = categorize(r, arr)
    r["cat"], r["loss"], r["oracle"], r["f_eff"], r["acc"] = cat, loss, best, f, acc
    R.append(r)
print("n", len(R), "oracle", mean(r["oracle"] for r in R), "model", mean(r["f_eff"] for r in R))
out["q1_overall"] = tbl("Q1 overall", {"all": R}, None)
out["q1_reachable"] = tbl("Q1 oracle>=0.5 (reachable-ish)", {"oracle>=0.5": [r for r in R if r["oracle"] >= 0.5], "oracle<0.5": [r for r in R if r["oracle"] < 0.5]}, None)
bands = {b: [r for r in R if m2.band_of(r["ans_len"]) == b or (r["ans_len"] is None and b == "none")] for b in ["<=10", "11-20", "21+", "none"]}
out["q1_band"] = tbl("Q1 by answer-sentence length band", bands, None)
# model-answered-only loss when answered: avg ROUGE
print("answered share", mean(r["anchor"] >= 0 for r in R))
# --- Q2
print("\n## Q2: oracle anchors vs the model's scores")
Ns = [len(r["alog"]) == r["N"] for r in R]
print("alog covers all N nodes (every node is a candidate):", all(Ns))
ranks = []
for r in R:
    if len(r["alog"]) == 0:
        continue
    order = np.argsort(-r["alog"], kind="stable")
    pos = {int(g): i for i, g in enumerate(order)}
    r["orank"] = min(pos[int(g)] for g in r["acc"])
    r["orank_best"] = pos[int(r["node_rouge"].argmax())]
c = [r for r in R if "orank" in r and r["oracle"] >= 0.5]
print("items oracle>=0.5:", len(c), " best-acceptable-anchor rank in model scores: top1 %.3f top3 %.3f top5 %.3f top10 %.3f median %.1f" % tuple(
    [mean(r["orank"] < k for r in c) for k in (1, 3, 5, 10)] + [float(np.median([r["orank"] for r in c]))]))
print("  oracle anchor in model top1 / top3 when model answered:", mean(r["orank"] < 1 for r in c if r["anchor"] >= 0))
# sentence-level: is the oracle sentence the one carrying the argmax score mass?
def sent_mass(r, arr):
    S = arr["n_sent"]
    m = np.zeros(S)
    p = np.exp(r["alog"] - r["alog"].max())
    p = p / p.sum()
    for g in range(len(p)):
        m[int(arr["sent"][g])] += p[g]
    return m
sm_ok = []
for r in c:
    arr = stories[r["skey"]]
    m = sent_mass(r, arr)
    osents = {int(arr["sent"][g]) for g in r["acc"]}
    sm_ok.append((int(m.argmax()) in osents, int(arr["sent"][r["anchor"]]) in osents if r["anchor"] >= 0 else False))
print("oracle sentence = argmax sentence mass:", mean(x for x, _ in sm_ok), " model anchor in oracle sentence:", mean(y for _, y in sm_ok))

def clause_head(arr, g):
    while g >= 0:
        if poss[arr["pos"][g]] in ("VERB", "AUX"):
            return g
        g = int(arr["parent"][g])
    return -1

def relation(arr, a, o):
    """Relation of model anchor a to oracle anchor o, structure only."""
    if a == o:
        return "same"
    if arr["sent"][a] != arr["sent"][o]:
        return "other sentence"
    anc = []
    x = int(o)
    while x >= 0:
        anc.append(x); x = int(arr["parent"][x])
    anc_a = []
    x = int(a)
    while x >= 0:
        anc_a.append(x); x = int(arr["parent"][x])
    if a in anc:
        d = anc.index(a)
        return "model is parent of oracle" if d == 1 else "model is higher ancestor of oracle"
    if o in anc_a:
        d = anc_a.index(o)
        return "model is child of oracle" if d == 1 else "model is deeper descendant of oracle"
    if arr["parent"][a] == arr["parent"][o]:
        return "sibling"
    return "same sentence, other branch (same clause head)" if clause_head(arr, a) == clause_head(arr, o) else "same sentence, other clause"

def tdist(arr, a, o):
    if arr["sent"][a] != arr["sent"][o]:
        return None
    A = {}
    x, d = int(a), 0
    while x >= 0:
        A[x] = d; x = int(arr["parent"][x]); d += 1
    x, d = int(o), 0
    while x >= 0:
        if x in A:
            return A[x] + d
        x = int(arr["parent"][x]); d += 1
    return None

rel_c = collections.Counter(); dist_c = collections.Counter()
rel_all = collections.Counter()
for r in R:
    if r["cat"] != "right_sentence_wrong_anchor":
        continue
    arr = stories[r["skey"]]
    a = r["anchor"]
    # nearest acceptable anchor in the same sentence
    best_o, bd = None, 99
    for o in r["acc"]:
        d = tdist(arr, a, int(o))
        if d is not None and d < bd:
            best_o, bd = int(o), d
    rel = relation(arr, a, best_o)
    rel_c[rel] += 1
    dist_c[bd] += 1
    r["rel"], r["tdist"] = rel, bd
n = sum(rel_c.values())
print("\n### (c) right sentence, wrong anchor: relation of model anchor to the nearest oracle anchor (n=%d)" % n)
for k, v in rel_c.most_common():
    ls = sum(r["loss"] for r in R if r.get("rel") == k) / max(sum(r["loss"] for r in R if r["cat"] == "right_sentence_wrong_anchor"), 1e-9)
    print(f"| {k} | {v} | {100*v/n:.1f}% | loss share within (c) {100*ls:.1f}% |")
print("(c) precision/recall of the model's text by relation (broad = high recall low precision):")
for k in rel_c:
    rs = [r for r in R if r.get("rel") == k]
    print(f"| {k} | prec {mean(r['prec'] for r in rs):.2f} | rec {mean(r['rec'] for r in rs):.2f} | f {mean(r['f'] for r in rs):.2f} | oracle {mean(r['oracle'] for r in rs):.2f} |")
cc = [r for r in R if r["cat"] == "right_sentence_wrong_anchor"]
print("(c) overall: prec %.2f rec %.2f; too broad (rec>prec) %.2f" % (mean(r["prec"] for r in cc), mean(r["rec"] for r in cc), mean(r["rec"] > r["prec"] for r in cc)))
print("tree distance histogram:", sorted(dist_c.items()))
print("mean tdist", np.mean([r["tdist"] for r in R if "tdist" in r]))
# among wrong sentence: nearest sentence distance
sd = collections.Counter()
for r in R:
    if r["cat"] == "wrong_sentence":
        arr = stories[r["skey"]]
        osents = sorted({int(arr["sent"][g]) for g in r["acc"]})
        d = min(abs(int(arr["sent"][r["anchor"]]) - s) for s in osents) if osents else None
        sd[d if d is None or d < 5 else "5+"] += 1
print("\n### (b) wrong sentence: sentence-index distance to the nearest oracle sentence:", dict(sd))
# oracle anchor position in the tree (the geometry of the target): depth, size share
print("\n### oracle anchor geometry by band (the best acceptable anchor with the largest size)")
for b, rs in bands.items():
    dd, sz, root = [], [], []
    for r in rs:
        if r["oracle"] < 0.5: continue
        arr = stories[r["skey"]]
        o = int(r["acc"][np.argmax([arr["size"][g] for g in r["acc"]])])
        s = int(arr["sent"][o])
        sl = int((arr["sent"] == s).sum())
        dd.append(int(arr["depth"][o])); sz.append(arr["size"][o] / sl); root.append(float(arr["parent"][o] < 0 or arr["depth"][o] <= 1))
    print(f"| {b} | n={len(dd)} | mean depth {np.mean(dd):.2f} | mean subtree/sentence size {np.mean(sz):.2f} | anchor is sentence head (depth<=1) {np.mean(root):.2f} |")

# --- Q3 question type
def wh_group(r):
    wh = r["wh"]
    if wh < 0:
        return "no wh node"
    v = ctx.raw(r["q_sid"][wh])
    if v is None:
        return "wh: no point"
    top = [axes[i] for i in np.argsort(-v) if axes[i] != "UNKNOWN" and v[i] > 0][:2]
    g = "+".join(top) if top else "UNKNOWN only"
    # a child of the wh node that carries MUCH_MANY (a dimension test, not a word list)
    for j, p in enumerate(r["q_parent"]):
        if p == wh and r["q_sid"][j]:
            vv = ctx.raw(r["q_sid"][j])
            if vv is not None and "MUCH_MANY" in axes and vv[axes.index("MUCH_MANY")] > 0:
                g += " [+MUCH_MANY child]"
    return g

grp = collections.defaultdict(list)
for r in R:
    r["wh_group"] = wh_group(r)
    grp[r["wh_group"]].append(r)
out["q3_wh"] = tbl("Q3a: question type by the wh node's lexicon dimensions (other than UNKNOWN)", {k: v for k, v in sorted(grp.items(), key=lambda kv: -len(kv[1])) if len(v) >= 10}, None)
print("small groups:", {k: len(v) for k, v in grp.items() if len(v) < 10})
ag = collections.defaultdict(list)
for r in R:
    ag[r["attribute"]].append(r)
out["q3_attr"] = tbl("Q3b: FairytaleQA attribute", dict(sorted(ag.items(), key=lambda kv: -len(kv[1]))), None)
out["q3_ex"] = tbl("Q3c: explicit/implicit", {k: [r for r in R if r["explicit"] == k] for k in ("explicit", "implicit")}, None)
out["q3_loc"] = tbl("Q3d: local/summary", {k: [r for r in R if r["local"] == k] for k in ("local", "summary")}, None)
# attribute x wh group cross
print("\nattribute x wh-group counts (top):")
cross = collections.Counter((r["attribute"], r["wh_group"]) for r in R)
for k, v in cross.most_common(14):
    print(k, v)

# --- Q5 wh attachment
def wh_attach(r):
    wh = r["wh"]
    if wh < 0:
        return "no wh"
    qp = r["q_parent"]; qpos = [poss[i] for i in r["q_pos"]]
    depth = []
    for j in range(len(qp)):
        d, x = 0, j
        while qp[x] >= 0:
            x = qp[x]; d += 1
        depth.append(d)
    verbs = [j for j in range(len(qp)) if qpos[j] == "VERB"]
    mv = min(verbs, key=lambda j: (depth[j], j)) if verbs else -1          # the shallowest VERB-tagged node = the question's main verb
    p = qp[wh]
    if p < 0:
        return "wh is the root (verb below it)" if mv >= 0 else "wh is the root (no verb)"
    if mv >= 0 and p == mv:
        return "wh directly under main verb"
    if qpos[p] == "AUX":
        # does the aux itself sit above/beside the verb?
        return "wh under AUX"
    if qpos[p] == "VERB":
        return "wh under a non-main VERB"
    return "wh under other (" + qpos[p] + ")"
at = collections.defaultdict(list)
for r in R:
    r["wh_att"] = wh_attach(r)
    at[r["wh_att"]].append(r)
out["q5"] = tbl("Q5: attachment of the question's wh node (structure only: tags and relations)", dict(sorted(at.items(), key=lambda kv: -len(kv[1]))), None)
# main-verb-reachable: is the main verb a descendant-or-self-of-ancestor... additional coarse split
coarse = {"wh under main verb or is root above verb": [r for r in R if r["wh_att"] in ("wh directly under main verb", "wh is the root (verb below it)")],
          "wh under AUX / other (not under the main verb)": [r for r in R if r["wh_att"] not in ("wh directly under main verb", "wh is the root (verb below it)", "no wh")],
          "no wh": at.get("no wh", [])}
out["q5_coarse"] = tbl("Q5 coarse", {k: v for k, v in coarse.items() if v}, None)
# which of these has the model rank-ed oracle anchor lower
for k, v in coarse.items():
    c2 = [r for r in v if "orank" in r and r["oracle"] >= 0.5]
    if c2: print(k, "oracle anchor top1/top3:", mean(r["orank"] < 1 for r in c2), mean(r["orank"] < 3 for r in c2), "n", len(c2))
pickle.dump([{k: v for k, v in r.items() if k not in ("alog", "node_rouge", "acc")} for r in R], open(OUTD / f"decomp_{split}_{name}.pkl", "wb"))
