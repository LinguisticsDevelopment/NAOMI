import gzip, json, collections
G = "/home/user/NAOMI/consciousness_transformer/runs/gold_v7.jsonl.gz"
def band(n): return "<=10" if n <= 10 else "11-20" if n <= 20 else "21-35" if n <= 35 else "36+"
tot = collections.Counter(); bands = collections.Counter(); cl = collections.Counter(); rel = collections.Counter()
def walk(n, top, st):
    if n["token_index"] is not None:
        st["n"] += 1
        if not top and n["pos"] == "VERB": st["cl"] += 1; st["rel"][n["rel"]] += 1
        if n["rel"] == "UNATTACHED": st["un"] += 1
    for c in n["children"]: walk(c, n["token_index"] is None and False, st)
with gzip.open(G, "rt") as f:
    for line in f:
        r = json.loads(line); root = r["tree"]["root"]
        st = {"n": 0, "cl": 0, "un": 0, "rel": collections.Counter()}
        if root["token_index"] is None: continue   # synthetic root => several pieces
        walk(root, True, st)
        if st["un"]: continue
        b = band(st["n"]); tot["sent"] += 1; tot["tok"] += st["n"]; bands[b] += 1; cl[b] += st["cl"]; tot["cl"] += st["cl"]
        rel.update(st["rel"]); tot["gt20"] += st["n"] > 20; tot["gt35"] += st["n"] > 35
print(dict(tot), dict(bands), dict(cl)); print(rel.most_common())
