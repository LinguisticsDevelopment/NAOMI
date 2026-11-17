"""Part 2: read three sets with parser + (forced) encoder decode; one JSON per set. usage: part2_read.py SET"""
import sys, json, random, time
from pathlib import Path
CT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CT / "src"))
import numpy as np
from naomi import encoder as E
from naomi.reader import Reader, _sense_space, normalize
from naomi.referee import Attested, link_row
from naomi.types import REL_ROOT
R = CT / "runs"; HERE = Path(__file__).parent
which = sys.argv[1]
def episodes(p):
    return [json.loads(l) for l in open(p)]
def pick(which):
    if which == "stories":
        seen, out = set(), []
        for e in episodes(R / "mctest_mc160_episodes.jsonl"):
            for s in e["context"]:
                if s not in seen: seen.add(s); out.append(s)
        return out[:400]
    if which == "questions":
        seen, out = set(), []
        for e in episodes(R / "mctest_mc160_episodes.jsonl"):
            q = e["question"]
            if q not in seen: seen.add(q); out.append(q)
        return out[:400]
    if which == "fairytale":
        seen, out = set(), []
        for e in episodes(R / "fairytaleqa_episodes_samepassage.jsonl"):
            if e["meta"].get("split") != "test": continue
            for s in e["context"]:
                if s not in seen: seen.add(s); out.append(s)
        random.Random(0).shuffle(out)
        return out[:400]
sents = pick(which)
reader = Reader(use_encoder=False, cache_path=None)
usvs = _sense_space()
encs = {n: E.NaomiEncoder.load(str(R / p), usvs) for n, p in
        (("A", "naomi_encoder/main_best.pt"), ("A2", "naomi_encoder_a2/a2_best.pt"), ("C", "naomi_encoder_a2/a2ctrl_best.pt"))}
att = Attested.load(R / "naomi_orderer/attested_configs.json")
def band(n): return "<=10" if n <= 10 else "11-20" if n <= 20 else "21-35" if n <= 35 else "36+"
res = []
t0 = time.time()
for si, s in enumerate(sents):
    infos = E.infos_from_sentence(reader, s)
    nodes = [i for i, t in enumerate(infos) if t.is_node]
    rec = {"i": si, "text": s, "n_words": len(nodes), "band": band(len(nodes)), "parser": "ok"}
    if len(nodes) < 2:
        rec["parser"] = "trivial"; res.append(rec); continue
    rr = None
    try: rr = reader.read_parser(s)
    except Exception as ex: rec["error"] = str(ex)[:80]
    if rr is None or rr.tree is None:
        rec["parser"] = "failed"; sv = None; forced = {}
        unknown = list(nodes); sup = {}
    else:
        sv = E.supervision_from_tree(rr.tree, len(infos))
        if max([n.token_index for n, _d, _p in __import__("naomi.tree", fromlist=["walk"]).walk(rr.tree.root) if n.token_index is not None] + [0]) >= len(infos):
            rec["parser"] = "misaligned"; res.append(rec); continue
        forced = {i: sv.head[i] for i in nodes if sv.sup[i] and (sv.head[i] == 0 or infos[sv.head[i] - 1].is_node)}
        unknown = [i for i in nodes if i not in forced]
        rec["connected"] = bool(sv.connected)
    rec["n_parser_links"] = len([i for i in forced if forced[i] != 0])
    rec["n_added"] = len(unknown)
    outs = {}
    for n, enc in encs.items():
        H, logp, _ = enc.arc_logprobs(infos)
        _n, W, h, heads = enc._decode(infos, H, logp, forced)
        ids, _ = enc._rels(H, [(i, heads[i]) for i in unknown])
        outs[n] = (H, np.exp(logp), heads, {i: enc.rel_labels[k] for i, k in zip(unknown, ids)})
    added = []
    for i in unknown:
        h2 = outs["A2"][2][i]
        d = {"dep": i, "head": h2, "rel": outs["A2"][3][i]}
        for n in encs:
            P = outs[n][1]
            d[f"p_{n}"] = float(P[i, h2]); d[f"chosen_{n}"] = outs[n][2][i] == h2
            d[f"rel_{n}"] = outs[n][3][i]
        d["S1"] = d["chosen_A"] and d["rel_A"] == d["rel"]
        mp = min(d["p_A"], d["p_A2"], d["p_C"])
        d["S2"] = d["chosen_A"] and d["chosen_C"]
        d["S3"] = d["S2"] and mp >= 0.8; d["S4"] = d["S2"] and mp >= 0.9
        if h2 > 0:
            hh, tt = infos[h2 - 1], infos[i]
            cnt = att.count(link_row(hh.pos, hh.subtypes, d["rel"], tt.pos, tt.subtypes, i < h2 - 1, att.classes))
        else: cnt = None
        d["attested_count"] = cnt
        d["S5"] = d["S3"] and cnt is not None and cnt >= 40
        added.append(d)
    rec["added"] = added
    # final tree links (A2 constrained decode, parser links keep parser relation)
    heads = outs["A2"][2]
    links = []
    for i in nodes:
        if i in forced: rel, src = (REL_ROOT if heads[i] == 0 else sv.rel[i]), "parser"
        else: rel, src = outs["A2"][3][i], "encoder"
        links.append({"dep": i, "head": heads[i], "rel": rel, "src": src})
    rec["links"] = links
    rec["tokens"] = [t.token for t in infos]
    res.append(rec)
    if si % 50 == 0:
        print(which, si, round(time.time() - t0), flush=True)
        json.dump(res, open(HERE / f"part2_{which}.json", "w"))
json.dump(res, open(HERE / f"part2_{which}.json", "w"))
print("done", which, len(res), round(time.time() - t0))
