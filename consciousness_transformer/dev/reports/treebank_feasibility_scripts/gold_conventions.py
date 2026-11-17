import gzip, json, collections, sys
sys.path.insert(0, "/home/user/NAOMI/consciousness_transformer/src")
G = "/home/user/NAOMI/consciousness_transformer/runs/gold_v7.jsonl.gz"
cnt = collections.defaultdict(collections.Counter)
tot = collections.Counter()
def walk(n, parent):
    for c in n["children"]:
        if parent is not None and c["rel"] not in ("UNATTACHED", "ROOT") and n["token_index"] is not None:
            d = "dep_after_head" if c["token_index"] > n["token_index"] else "dep_before_head"
            cnt[c["rel"]][(n["pos"], c["pos"], d)] += 1
            tot[c["rel"]] += 1
        walk(c, n)
with gzip.open(G, "rt") as f:
    for line in f:
        r = json.loads(line)
        walk(r["tree"]["root"], None)
out = {}
for rel, c in cnt.items():
    out[rel] = {"n": tot[rel], "top": [[list(k), v] for k, v in c.most_common(8)]}
json.dump(out, open("gold_conventions.json", "w"), indent=1)
for rel in sorted(out, key=lambda r: -out[r]["n"]):
    print(rel, out[rel]["n"])
    for k, v in out[rel]["top"][:6]:
        print("   ", k, round(v / out[rel]["n"] * 100, 1))
