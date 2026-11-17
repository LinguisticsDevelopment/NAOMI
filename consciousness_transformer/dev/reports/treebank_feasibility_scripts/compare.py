"""Agreement of converted UD trees with the parser trees. Usage: compare.py TABLE.json HALF(dev|test|all) [--detail out.json]"""
import os, sys, json, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import conllu, convert as C, align as A
UD = "/home/user/NAOMI/consciousness_transformer/data/ud"
def load(table_path, half):
    table = json.load(open(table_path))
    sample = {x["id"]: x for x in json.load(open(f"{HERE}/sample_ewt_1500.json"))}
    ud = {}
    for sp in ("train", "dev", "test"):
        for s in conllu.read(f"{UD}/EWT/en_ewt-ud-{sp}.conllu"):
            if s["meta"]["sent_id"] in sample: ud[s["meta"]["sent_id"]] = s
    recs = []
    for f in sorted(glob.glob(f"{HERE}/out/parser_ewt.*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if half in ("all", sample[r["id"]]["half"]): r["band"] = sample[r["id"]]["band"]; recs.append(r)
    return table, ud, recs

def analyse(table, ud, recs):
    S = collections.Counter(); out = []
    for r in recs:
        S["sent"] += 1
        if r["tree"] is None: S["parser_fail"] += 1; continue
        s = ud[r["id"]]
        groups, st = A.align(s, r["tokens"])
        if groups is None: S["tok_" + st] += 1; continue
        if st == "ok_masked": S["sent_with_masked_tokens"] += 1
        conv = C.convert(s, table)
        proj = A.project(s, conv, groups, r["pos"])
        edges, root, connected, n, unatt = A.parser_edges(r["tree"])
        out.append({"rec": r, "proj": proj, "edges": edges, "root": root, "connected": connected, "conv": conv, "s": s})
        S["usable"] += 1; S["connected" if connected else "pieces"] += 1
    return S, out

def cmp_edges(item, only_committed=True):
    """list of (child, parser_head, parser_rel, ud_head, ud_label, orig, flags) for tokens with both sides defined"""
    rows = []
    proj, edges = item["proj"], item["edges"]
    for p, v in proj.items():
        if v is None: continue
        if p in edges: ph, pr = edges[p]
        elif p == item["root"] and item["connected"]: ph, pr = -1, "ROOT"
        elif item["connected"]: continue
        else: continue
        rows.append((p, ph, pr, v[0], v[1], v[2], v[3], v[4]))
    return rows

def main():
    table_path, half = sys.argv[1], sys.argv[2]
    table, ud, recs = load(table_path, half)
    S, items = analyse(table, ud, recs)
    print("sentences", dict(S))
    res = {}
    for kind in ("connected", "pieces"):
        T = collections.Counter(); byrel = collections.defaultdict(collections.Counter); byud = collections.defaultdict(collections.Counter)
        band = collections.defaultdict(collections.Counter)
        for it in items:
            if it["connected"] != (kind == "connected"): continue
            for p, ph, pr, uh, ul, orig, fl, up in cmp_edges(it):
                T["n"] += 1; hit = ph == uh; hl = hit and pr == ul
                T["head"] += hit; T["headrel"] += hl
                byrel[pr]["n"] += 1; byrel[pr]["head"] += hit; byrel[pr]["headrel"] += hl
                b = C.base(orig); byud[b]["n"] += 1; byud[b]["head"] += hit; byud[b]["headrel"] += hl
                bb = it["rec"]["band"]; band[bb]["n"] += 1; band[bb]["head"] += hit; band[bb]["headrel"] += hl
        res[kind] = {"T": T, "byrel": byrel, "byud": byud, "band": band}
        n = T["n"] or 1
        print(f"== {kind}: links {T['n']}  head agree {T['head']/n:.3f}  head+rel {T['headrel']/n:.3f}")
        for k, v in sorted(band.items()): print("   band", k, v["n"], f"{v['head']/v['n']:.3f} {v['headrel']/v['n']:.3f}")
        if kind == "connected":
            print("  by parser relation:")
            for k, v in sorted(byrel.items(), key=lambda x: -x[1]["n"]): print(f"   {k:16s} n={v['n']:5d} head {v['head']/v['n']:.3f} h+r {v['headrel']/v['n']:.3f}")
            print("  by UD deprel (base):")
            for k, v in sorted(byud.items(), key=lambda x: -x[1]["n"])[:22]: print(f"   {k:12s} n={v['n']:5d} head {v['head']/v['n']:.3f} h+r {v['headrel']/v['n']:.3f}")
    # root agreement among connected
    rc = sum(1 for it in items if it["connected"] and it["proj"].get(it["root"]) and it["proj"][it["root"]][0] == -1)
    nc = sum(1 for it in items if it["connected"])
    print("root agree (connected):", rc, "/", nc)
    # disagreement list (connected + pieces committed)
    dis = collections.defaultdict(list)
    for it in items:
        for p, ph, pr, uh, ul, orig, fl, up in cmp_edges(it):
            if ph == uh and pr == ul: continue
            kind = "head" if ph != uh else "label"
            key = (kind, C.base(orig), up, pr, ul if kind == "label" else "-")
            toks = it["rec"]["tokens"]
            dis[key].append({"id": it["rec"]["id"], "tok": toks[p], "ph": toks[ph] if ph >= 0 else "ROOT", "pr": pr,
                             "uh": toks[uh] if uh >= 0 else "ROOT", "ul": ul, "text": " ".join(toks)[:140], "connected": it["connected"]})
    ranked = sorted(dis.items(), key=lambda x: -len(x[1]))
    if "--detail" in sys.argv:
        json.dump([{"key": k, "n": len(v), "examples": v[:6]} for k, v in ranked[:60]], open(sys.argv[sys.argv.index("--detail") + 1], "w"), indent=1)
    # label confusion on links where both sides agree on the head
    conf = collections.defaultdict(collections.Counter); cur = {}
    for it in items:
        for p, ph, pr, uh, ul, orig, fl, up in cmp_edges(it):
            if ph == uh:
                k = (orig if ":" in orig and orig.split(":")[0] != "root" else C.base(orig), up, ul); conf[k][pr] += 1
    print("label confusion (head agrees): UD deprel, dep upos, CONV label -> parser labels")
    rows = [(sum(c.values()), k, c) for k, c in conf.items() if c.most_common(1)[0][0] != k[2]]
    for n, k, c in sorted(rows, key=lambda x: -x[0])[:22]: print("  ", n, k, c.most_common(4))
    print("top disagreement keys (kind, UD deprel, dep upos, parser rel, conv label):")
    for k, v in ranked[:25]: print("  ", len(v), k)
if __name__ == "__main__":
    main()
