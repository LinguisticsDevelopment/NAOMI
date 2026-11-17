"""Pick 40 story + 20 question sentences stratified by length, shuffle, hide provenance. Writes blind file (no provenance) and key."""
import json, random
from pathlib import Path
H = Path(__file__).parent
def strata(n):  # word-count strata for short text
    return 0 if n <= 6 else 1 if n <= 10 else 2 if n <= 15 else 3
blind, key = [], {}
cnt = 0
for s, k in (("stories", 40), ("questions", 20)):
    R = [r for r in json.load(open(H / f"part2_{s}.json")) if r["parser"] in ("ok",) and "links" in r and r["n_words"] >= 3]
    rng = random.Random(1)
    by = {i: [r for r in R if strata(r["n_words"]) == i] for i in range(4)}
    per = k // 4
    chosen = []
    for i in range(4):
        rng.shuffle(by[i]); chosen += by[i][:per]
    rng.shuffle(chosen)
    for r in chosen:
        cnt += 1; sid = f"J{cnt:02d}"
        links = list(r["links"]); rng.shuffle(links)
        out = []
        for j, l in enumerate(links):
            lid = f"{sid}.{j + 1}"
            out.append({"link": lid, "dependent": f"{l['dep']}:{r['tokens'][l['dep']]}",
                        "head": "ROOT" if l["head"] == 0 else f"{l['head'] - 1}:{r['tokens'][l['head'] - 1]}", "relation": l["rel"]})
            key[lid] = {"src": l["src"], "set": s}
        blind.append({"sentence": sid, "text": r["text"], "tokens": " ".join(f"{i}:{t}" for i, t in enumerate(r["tokens"])), "links": out})
json.dump(blind, open(H / "part2_blind.json", "w"), indent=1)
json.dump(key, open(H / "part2_blind_key.json", "w"), indent=1)
print(len(blind), len(key))
