"""Step 5: for UD sentences the parser leaves in pieces, how do the links the ENCODER ADDS agree with the converted UD tree?
Usage: encoder_vs_ud.py TABLE.json HALF"""
import os, sys, json, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import compare as K  # compare.main() is guarded below
import align as A, convert as C
from naomi.encoder import NaomiEncoder
CK = "/home/user/NAOMI/consciousness_transformer/runs/naomi_encoder/main_best.pt"
table_path, half = sys.argv[1], sys.argv[2]
table, ud, recs = K.load(table_path, half)
S, items = K.analyse(table, ud, recs)
enc = NaomiEncoder.load(CK)
def enc_edges(tree):
    ed, root = {}, None
    def walk(n, par):
        nonlocal root
        if par is None: root = n.token_index
        else: ed[n.token_index] = (par.token_index, n.rel)
        for c in n.children: walk(c, n)
    walk(tree.root, None)
    return ed, root
T = collections.Counter(); byband = collections.defaultdict(collections.Counter); byrel = collections.defaultdict(collections.Counter)
com = collections.Counter(); rows = []
for it in items:
    if it["connected"]: continue
    r = it["rec"]; t = enc.encode(r["text"])
    ed, eroot = enc_edges(t)
    unk = set(A.parser_edges(r["tree"])[4])
    for p in range(len(r["tokens"])):
        v = it["proj"].get(p)
        if v is None or p not in ed: continue
        eh, er = ed[p]
        if p in unk:   # link the encoder ADDS beyond the parser
            T["n"] += 1; hit = (eh == v[0]); hl = hit and er == v[1]
            T["head"] += hit; T["hl"] += hl
            # trivial baseline: attach to the preceding project token that is a node
            prev = next((q for q in range(p - 1, -1, -1) if r["pos"][q] != "PUNCT"), -1)
            T["base_prev"] += (prev == v[0])
            b = r["band"]; byband[b]["n"] += 1; byband[b]["head"] += hit; byband[b]["hl"] += hl
            byrel[er]["n"] += 1; byrel[er]["head"] += hit; byrel[er]["hl"] += hl
            rows.append({"id": r["id"], "tok": r["tokens"][p], "enc_head": r["tokens"][eh], "enc_rel": er, "ud_head": r["tokens"][v[0]] if v[0] >= 0 else "ROOT", "ud_label": v[1], "orig": v[2], "band": b, "text": " ".join(r["tokens"])[:120]})
        elif p in it["edges"]:  # committed link: encoder vs parser vs UD
            com["n"] += 1; ph = it["edges"][p][0]
            com["enc_eq_parser"] += (eh == ph); com["parser_eq_ud"] += (ph == v[0]); com["enc_eq_ud"] += (eh == v[0])
n = T["n"]
print("usable sentences", dict(S))
print(f"added links {n}: head agree with UD {T['head']/n:.3f}, head+rel {T['hl']/n:.3f}; baseline 'previous token' head agree {T['base_prev']/n:.3f}")
for b, v in sorted(byband.items()): print("  band", b, v["n"], f"{v['head']/v['n']:.3f} {v['hl']/v['n']:.3f}")
for k, v in sorted(byrel.items(), key=lambda x: -x[1]["n"]): print(f"  enc rel {k:16s} n={v['n']:4d} head {v['head']/v['n']:.3f} h+r {v['hl']/v['n']:.3f}")
m = com["n"]
print(f"committed links {m}: enc==parser {com['enc_eq_parser']/m:.3f}  parser==UD {com['parser_eq_ud']/m:.3f}  enc==UD {com['enc_eq_ud']/m:.3f}")
json.dump(rows, open(f"{HERE}/out/added_links_{half}.json", "w"))
