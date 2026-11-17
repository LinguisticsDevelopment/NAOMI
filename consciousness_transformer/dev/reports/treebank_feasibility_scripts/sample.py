"""Stratified sample of UD EWT (all splits pooled), seed 0: 375 per length band (non-punct words), then dev/test halves."""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(__file__))
import conllu
R = "/home/user/NAOMI/consciousness_transformer/data/ud"
bands = {"<=10": (3, 10), "11-20": (11, 20), "21-35": (21, 35), "36+": (36, 10**9)}
pool = {b: [] for b in bands}
for sp in ("train", "dev", "test"):
    for s in conllu.read(f"{R}/EWT/en_ewt-ud-{sp}.conllu"):
        L = sum(t["upos"] != "PUNCT" for t in s["toks"])
        if "\n" in s["meta"].get("text", "") or not s["meta"].get("text"):
            continue
        for b, (lo, hi) in bands.items():
            if lo <= L <= hi:
                pool[b].append(s["meta"]["sent_id"])
random.seed(0)
out = []
for b in bands:
    ids = sorted(set(pool[b])); random.shuffle(ids)
    for k, i in enumerate(ids[:375]):
        out.append({"id": i, "band": b, "half": "dev" if k % 2 == 0 else "test"})
print({b: len(pool[b]) for b in pool}, len(out))
json.dump(out, open("sample_ewt_1500.json", "w"))
