"""Part 1: per-link features for the 270 judged links (A / A2 / control probabilities, margins, attested)."""
import sys, json, gzip, math
from pathlib import Path
CT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CT / "src")); sys.path.insert(0, str(CT / "scripts"))
import numpy as np
from naomi import encoder as E
from naomi.reader import Reader, _sense_space
from naomi.referee import Attested, link_row
from eval_naomi_encoder import items_from_records

R = CT / "runs"
key = json.load(open(R / "naomi_encoder_a2/handcheck_blind_key.json"))
judg = json.load(open(R / "naomi_encoder_a2/handcheck_blind_judgments.json"))["judgments"]
ids = {v["id"] for v in key["sentences"].values()}
recs = [r for r in map(json.loads, gzip.open(R / "gold_v7.jsonl.gz", "rt")) if r["id"] in ids]
assert len(recs) == 40
reader = Reader(use_encoder=False, cache_path=None)
items = {it["id"]: it for it in items_from_records(recs, reader)}
usvs = _sense_space()
encs = {n: E.NaomiEncoder.load(str(R / p), usvs) for n, p in
        (("A", "naomi_encoder/main_best.pt"), ("A2", "naomi_encoder_a2/a2_best.pt"), ("C", "naomi_encoder_a2/a2ctrl_best.pt"))}
att = Attested.load(R / "naomi_orderer/attested_configs.json")
cache = {}
def model_out(n, sid):
    if (n, sid) not in cache:
        it = items[sid]
        prop, logp = encs[n].decode_infos(it["text"], it["infos"])
        cache[(n, sid)] = (prop.heads, np.exp(logp))
    return cache[(n, sid)]
bands = {v["id"]: v["band"] for v in key["sentences"].values()}
rows = []
for lid, L in key["links"].items():
    sid, dep, head, rel = L["key"].split("|"); dep, head = int(dep), int(head)
    it = items[sid]; infos = it["infos"]
    d = {"link": lid, "sid": sid, "band": bands[sid], "dep": dep, "head": head, "rel": rel, "models": L["models"],
         "judgment": judg[lid]}
    for n in encs:
        heads, P = model_out(n, sid)
        p = P[dep]; valid = np.sort(p[p > 1e-30])[::-1]
        d[f"p_{n}"] = float(P[dep, head]); d[f"margin_{n}"] = float(valid[0] - valid[1]) if len(valid) > 1 else float(valid[0])
        d[f"chosen_{n}"] = heads[dep] == head
        d[f"argmax_{n}"] = int(np.argmax(p)) == head
    if head > 0:
        h = infos[head - 1]; t = infos[dep]
        row = link_row(h.pos, h.subtypes, rel, t.pos, t.subtypes, dep < head - 1, att.classes)
        d["attested_count"] = att.count(row)
    else:
        d["attested_count"] = None   # ROOT link: no configuration in the table
    rows.append(d)
json.dump(rows, open(Path(__file__).parent / "part1_links.json", "w"), indent=1)
print(len(rows), sum(r["judgment"] == "w" for r in rows), "none-attested", sum(r["attested_count"] is None for r in rows))
