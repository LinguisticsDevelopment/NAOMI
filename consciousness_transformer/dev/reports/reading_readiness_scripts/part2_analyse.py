import json
from pathlib import Path
from collections import Counter, defaultdict
H = Path(__file__).parent
BANDS = ["<=10", "11-20", "21-35", "36+"]
out = {}
for s in ["stories", "questions", "fairytale"]:
    R = json.load(open(H / f"part2_{s}.json"))
    o = {"n": len(R), "status": dict(Counter(r["parser"] for r in R))}
    ns = [r["n_words"] for r in R]
    o["len_mean"] = round(sum(ns) / len(ns), 1); o["len_median"] = sorted(ns)[len(ns) // 2]; o["len_max"] = max(ns)
    o["band_counts"] = dict(Counter(r["band"] for r in R))
    ok = [r for r in R if r["parser"] in ("ok", "failed")]
    o["n_assessed"] = len(ok)
    full = [r for r in ok if r["parser"] == "ok" and r.get("connected")]
    o["parser_fully_connected"] = len(full)
    rest = [r for r in ok if r not in full]
    o["fragmented_or_failed"] = len(rest)
    add = [a for r in rest for a in r["added"]]
    o["links_to_add"] = len(add)
    o["links_per_rest_sentence"] = round(len(add) / max(len(rest), 1), 2)
    for k in ["S1", "S2", "S3", "S4", "S5"]: o[f"added_pass_{k}"] = sum(a[k] for a in add)
    o["root_links_added"] = sum(a["head"] == 0 for a in add)
    def trusted(r, k):
        return r["parser"] != "failed" and all(a[k] for a in r["added"])
    o["trusted"] = {}
    for k in ["S3", "S4", "S5"]:
        o["trusted"][k] = {"all": sum(trusted(r, k) for r in ok), "by_band": {b: [sum(trusted(r, k) for r in ok if r["band"] == b), sum(r["band"] == b for r in ok)] for b in BANDS}}
    o["parser_failed"] = sum(r["parser"] == "failed" for r in ok)
    # trusted trees counting parser-failed sentences with all links passing too (all-encoder)
    o["by_band_connected"] = {b: [sum(1 for r in full if r["band"] == b), sum(r["band"] == b for r in ok)] for b in BANDS}
    o["by_band_frag_added"] = {b: [sum(len(r["added"]) for r in rest if r["band"] == b), sum(a["S3"] for r in rest if r["band"] == b for a in r["added"])] for b in BANDS}
    out[s] = o
json.dump(out, open(H / "part2_results.json", "w"), indent=1)
for s, o in out.items(): print(s, json.dumps(o))
