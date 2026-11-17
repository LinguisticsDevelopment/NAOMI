"""Part 1 analysis. Settings declared before computing (also written at the top of READING_READINESS.md).
S1 both A and A2 propose the link; S2 A, A2 and control all choose this head in their decoded tree;
S3 S2 and min head probability over the three models >= 0.8; S4 same with 0.9;
S5 S3 and the link's configuration attested in train gold with count >= 40 (referee min_count; ROOT links: not attested)."""
import json
from pathlib import Path
from collections import defaultdict
rows = json.load(open(Path(__file__).parent / "part1_links.json"))
def minp(r): return min(r["p_A"], r["p_A2"], r["p_C"])
def s2(r): return r["chosen_A"] and r["chosen_A2"] and r["chosen_C"]
def att(r): return r["attested_count"] is not None and r["attested_count"] >= 40
S = {"S1": lambda r: "A" in r["models"] and "A2" in r["models"],
     "S2": s2, "S3": lambda r: s2(r) and minp(r) >= 0.8, "S4": lambda r: s2(r) and minp(r) >= 0.9,
     "S5": lambda r: s2(r) and minp(r) >= 0.8 and att(r)}
def tally(rs):
    c = {"n": len(rs)}
    for j in "rdw": c[j] = sum(r["judgment"] == j for r in rs)
    c["wrong_pct"] = round(100 * c["w"] / len(rs), 1) if rs else None
    return c
out = {"settings": {"S1": "A and A2 both propose", "S2": "A, A2, control all choose this head",
                    "S3": "S2 and min p>=0.8", "S4": "S2 and min p>=0.9", "S5": "S3 and attested (count>=40)"}, "results": {}}
N = len(rows)
for k, f in S.items():
    P = [r for r in rows if f(r)]; F = [r for r in rows if not f(r)]
    res = {"pass": tally(P), "fail": tally(F), "share_pass_pct": round(100 * len(P) / N, 1), "by_band": {}}
    for b in ["<=10", "11-20", "21-35", "36+"]:
        res["by_band"][b] = {"pass": tally([r for r in P if r["band"] == b]), "all": tally([r for r in rows if r["band"] == b])}
    # sentence acceptance on A2-added links
    bysent = defaultdict(list)
    for r in rows:
        if "A2" in r["models"]: bysent[r["sid"]].append(r)
    acc = {s: v for s, v in bysent.items() if all(f(r) for r in v)}
    al = [r for v in acc.values() for r in v]
    res["sentence_accept"] = {"sentences_with_A2_links": len(bysent), "accepted": len(acc), "links_in_accepted": tally(al),
                              "accepted_by_band": {b: sum(1 for s in acc if next(x["band"] for x in bysent[s]) == b) for b in ["<=10", "11-20", "21-35", "36+"]}}
    out["results"][k] = res
# extra: attested alone, A2 alone-probability >=0.8 for context
out["context"] = {"attested_only": tally([r for r in rows if att(r)]), "none": tally(rows),
                  "A2_p>=0.8_any": tally([r for r in rows if "A2" in r["models"] and r["p_A2"] >= 0.8])}
json.dump(out, open(Path(__file__).parent / "part1_results.json", "w"), indent=1)
for k, v in out["results"].items():
    print(k, "pass", v["pass"], v["share_pass_pct"], "| fail", v["fail"])
    print("   bands", {b: (x["pass"]["n"], x["pass"]["w"], x["all"]["n"]) for b, x in v["by_band"].items()})
    print("   sent", v["sentence_accept"])
print(out["context"])
