"""Task 6: WordNet relation coverage counts (no design)."""
from common import *
import collections, gzip, time
from nltk.corpus import wordnet as wn
t0 = time.time()
u = load(); sidx = {s: i for i, s in enumerate(u.sense_ids)}
res = {}
allsyn = list(wn.all_synsets()); res["n_synsets_wordnet"] = len(allsyn); res["n_synsets_usvs"] = len(u.sense_ids)
pc = collections.Counter(s.pos() for s in allsyn); res["synsets_by_pos"] = dict(pc)
# direct antonyms
ptrs = collections.Counter(); pairs_lemma = collections.defaultdict(set); syn_in = collections.defaultdict(set); synpairs = collections.defaultdict(set)
for s in allsyn:
    for l in s.lemmas():
        for a in l.antonyms():
            p = s.pos()
            ptrs[p] += 1
            pairs_lemma[p].add(frozenset(((s.name(), l.name()), (a.synset().name(), a.name()))))
            syn_in[p].add(s.name()); syn_in[p].add(a.synset().name())
            synpairs[p].add(frozenset((s.name(), a.synset().name())))
res["antonym_pointers_directed_by_pos"] = dict(ptrs)
res["antonym_lemma_pairs_unordered_by_pos"] = {p: len(v) for p, v in pairs_lemma.items()}; res["antonym_lemma_pairs_total"] = sum(len(v) for v in pairs_lemma.values())
res["antonym_synset_pairs_by_pos"] = {p: len(v) for p, v in synpairs.items()}
res["antonym_distinct_synsets_by_pos_of_synset"] = {}
allsyn_ant = set()
for p, v in syn_in.items(): allsyn_ant |= v
cnt = collections.Counter(wn.synset(x).pos() for x in allsyn_ant); res["antonym_distinct_synsets_by_pos_of_synset"] = dict(cnt); res["antonym_distinct_synsets_total"] = len(allsyn_ant)
res["share_of_synsets_with_direct_antonym_by_pos"] = {p: cnt[p] / pc[p] for p in cnt}
res["share_of_synsets_with_direct_antonym_total"] = len(allsyn_ant) / len(allsyn)
# adjective clusters
heads = [s for s in allsyn if s.pos() == "a"]; sats = [s for s in allsyn if s.pos() == "s"]
res["adj_heads"] = len(heads); res["adj_satellites"] = len(sats)
head_has_ant = [h for h in heads if any(l.antonyms() for l in h.lemmas())]
res["adj_heads_with_direct_antonym"] = len(head_has_ant)
ant_head_pairs = set()
for h in head_has_ant:
    for l in h.lemmas():
        for a in l.antonyms():
            if a.synset().pos() == "a": ant_head_pairs.add(frozenset((h.name(), a.synset().name())))
res["bipolar_head_pairs_a_a"] = len(ant_head_pairs)
sat_heads = collections.defaultdict(list)
for s in sats:
    for h in s.similar_tos(): sat_heads[h.name()].append(s.name())
res["heads_with_satellites"] = len(sat_heads); res["mean_satellites_per_head_with_sats"] = float(np.mean([len(v) for v in sat_heads.values()]))
res["median_satellites_per_head_with_sats"] = float(np.median([len(v) for v in sat_heads.values()])); res["max_satellites_per_head"] = max(len(v) for v in sat_heads.values())
res["mean_satellites_per_head_all_heads"] = len(sats) / len(heads)
sat_nohead = [s for s in sats if not s.similar_tos()]; res["satellites_without_similar_to"] = len(sat_nohead)
res["satellites_multi_head"] = sum(1 for s in sats if len(s.similar_tos()) > 1)
in_bip = set()
for pr in ant_head_pairs:
    for h in pr: in_bip.add(h); in_bip.update(sat_heads.get(h, []))
res["adj_synsets_in_bipolar_clusters(head_pair_plus_satellites)"] = len(in_bip); res["share_of_adj_synsets_in_bipolar_clusters"] = len(in_bip) / (len(heads) + len(sats))
res["heads_without_antonym"] = len(heads) - len(head_has_ant)
res["heads_without_antonym_but_with_satellites"] = sum(1 for h in heads if h not in set(x.name() for x in head_has_ant) and h.name() in sat_heads)
# attribute relation
attr_pairs = set(); adj_att = set(); noun_att = set()
for s in allsyn:
    for a in s.attributes():
        attr_pairs.add((s.name(), a.name()));
        (adj_att if s.pos() in "as" else noun_att).add(s.name())
        (noun_att if a.pos() == "n" else adj_att).add(a.name())
res["attribute_links_directed"] = len(attr_pairs)
res["attribute_adj_synsets"] = len({x for x in [p[0] for p in attr_pairs] if wn.synset(x).pos() in "as"}); res["attribute_noun_synsets"] = len({p[1] for p in attr_pairs if wn.synset(p[1]).pos() == "n"} | {p[0] for p in attr_pairs if wn.synset(p[0]).pos()=="n"})
res["attribute_adj_synsets_by_pos"] = dict(collections.Counter(wn.synset(p[0]).pos() for p in attr_pairs if wn.synset(p[0]).pos() in "as"))
# graph for BFS
names = [s.name() for s in allsyn]; nid = {n: i for i, n in enumerate(names)}
def edges(kinds):
    E = []
    for s in allsyn:
        i = nid[s.name()]
        if "sim" in kinds:
            for t in s.similar_tos(): E.append((i, nid[t.name()]))
        if "also" in kinds:
            for t in s.also_sees(): E.append((i, nid[t.name()]))
        if "hyper" in kinds:
            for t in s.hypernyms() + s.instance_hypernyms(): E.append((i, nid[t.name()]))
        if "deriv" in kinds:
            for l in s.lemmas():
                for d in l.derivationally_related_forms(): E.append((i, nid[d.synset().name()]))
        if "pert" in kinds:
            for l in s.lemmas():
                for d in l.pertainyms(): E.append((i, nid[d.synset().name()]))
        if "attr" in kinds:
            for t in s.attributes(): E.append((i, nid[t.name()]))
        if "verbgroup" in kinds:
            for t in s.verb_groups(): E.append((i, nid[t.name()]))
    return E
def bfs(kinds, maxd=3):
    E = edges(kinds); adj = [[] for _ in names]
    for a, b in E: adj[a].append(b); adj[b].append(a)
    dist = np.full(len(names), 99, dtype=np.int16)
    fr = [nid[x] for x in allsyn_ant]
    for x in fr: dist[x] = 0
    for d in range(1, maxd + 1):
        nf = []
        for x in fr:
            for y in adj[x]:
                if dist[y] == 99: dist[y] = d; nf.append(y)
        fr = nf
    return dist, len(E)
sense_pos = {s: s.rsplit(".", 2)[1] for s in u.sense_ids if not s.startswith(("interj.", "fn."))}
res["bfs"] = {}
DIST = {}
for label, kinds in (("sim+also+deriv+pert+hyper", ("sim", "also", "deriv", "pert", "hyper")), ("sim+also+deriv+pert (no hypernym)", ("sim", "also", "deriv", "pert")),
                     ("sim+also+deriv+pert+hyper+attr+verbgroup", ("sim", "also", "deriv", "pert", "hyper", "attr", "verbgroup")), ("sim only", ("sim",)), ("deriv only", ("deriv",)), ("hyper only", ("hyper",))):
    dist, ne = bfs(kinds); DIST[label] = dist
    row = {"n_edges_directed_listed": ne}
    for tag, sel in (("all", [nid[s] for s in sense_pos if s in nid]),):
        d = dist[sel]; row["all"] = {f"<={k}": float((d <= k).mean()) for k in (0, 1, 2, 3)}
    for p in "nvasr":
        sel = [nid[s] for s, q in sense_pos.items() if q == p and s in nid]
        d = dist[sel]; row[p] = {f"<={k}": float((d <= k).mean()) for k in (0, 1, 2, 3)} | {"n": len(sel)}
    res["bfs"][label] = row
    print(label, time.time() - t0, flush=True)
# gold tokens
cont = {"NOUN", "VERB", "ADJ", "ADV"}
tot = 0; withsense = 0; dcount = collections.defaultdict(lambda: collections.Counter()); bypos = collections.Counter(); sent = 0
lab = "sim+also+deriv+pert+hyper"; lab2 = "sim+also+deriv+pert (no hypernym)"
cnt_by = {lab: collections.Counter(), lab2: collections.Counter()}; poscount = collections.Counter(); pos_with = collections.Counter()
with gzip.open("/home/user/NAOMI/consciousness_transformer/runs/encoder_gold_v6.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line); sent += 1
        for t in d.get("token_sense_candidates", []):
            i = t["index"]; p = d["pos"][i]
            if p not in cont: continue
            tot += 1; poscount[p] += 1
            s = t.get("chosen_sense") or (t["sense_candidates"][0] if t.get("sense_candidates") else None)
            if not s or s not in nid: continue
            withsense += 1; pos_with[p] += 1
            for L in (lab, lab2): cnt_by[L][int(DIST[L][nid[s]])] += 1
res["gold_v6"] = {"sentences": sent, "content_tokens_with_candidate_entry": tot, "content_tokens_with_sense": withsense, "by_upos": dict(poscount), "with_sense_by_upos": dict(pos_with)}
for L in (lab, lab2):
    res["gold_v6"][L] = {f"<={k}": sum(c for dd, c in cnt_by[L].items() if dd <= k) / withsense for k in (0, 1, 2, 3)}
# hypernym depth
dep = {"n": [], "v": []}
for s in allsyn:
    if s.pos() in dep: dep[s.pos()].append((s.min_depth(), s.max_depth()))
res["hypernym_depth"] = {}
for p, v in dep.items():
    mn = np.array([a for a, b in v]); mx = np.array([b for a, b in v])
    res["hypernym_depth"][p] = {"n": len(v), "min_depth_mean": float(mn.mean()), "min_depth_median": float(np.median(mn)), "min_depth_hist": {int(k): int(c) for k, c in zip(*np.unique(mn, return_counts=True))},
        "max_depth_mean": float(mx.mean()), "max_depth_median": float(np.median(mx)), "max_depth_hist": {int(k): int(c) for k, c in zip(*np.unique(mx, return_counts=True))}, "n_roots(depth0)": int((mn == 0).sum())}
res["verb_synsets_without_hypernym"] = sum(1 for s in allsyn if s.pos() == "v" and not s.hypernyms())
json.dump(res, open(S + "/m6_relations.json", "w"), indent=1)
print("done", time.time() - t0)
