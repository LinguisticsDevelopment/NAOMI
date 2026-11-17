"""Draw 60 converted BETWEEN-CLAUSE links (10 per type) from UD sentences over 20 tokens (EWT dev/test + GUM dev), print them for hand judgement."""
import os, sys, json, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import conllu, convert as C
UD = "/home/user/NAOMI/consciousness_transformer/data/ud"
table = json.load(open(sys.argv[1]))
pool = {t: [] for t in ("clause_coord", "subordinate", "relative", "complement", "reported_speech", "purpose")}
sents = []
for p in ("EWT/en_ewt-ud-dev.conllu", "EWT/en_ewt-ud-test.conllu", "GUM/en_gum-ud-dev.conllu"):
    sents += list(conllu.read(f"{UD}/{p}"))
random.seed(7)
random.shuffle(sents)
def frag(conv, i, s):
    """local converted view: parent --label--> node ( child rels )"""
    toks = {t["id"]: t["form"] for t in s["toks"]}
    kids = {}
    for j, h in conv["heads"].items(): kids.setdefault(h, []).append(j)
    def sub(j, d):
        c = sorted(kids.get(j, []))
        inner = " ".join(f"{conv['labels'][k]}:{sub(k, d - 1)}" for k in c) if d > 0 and c else ("..." if c else "")
        return toks[j] + (f"({inner})" if inner else "")
    return sub(i, 2)
for s in sents:
    L = sum(t["upos"] != "PUNCT" for t in s["toks"])
    if L <= 20 or L > 45: continue
    conv = C.convert(s, table)
    if not conv["heads"]: continue
    T = {t["id"]: t for t in s["toks"]}
    kidud = {}
    for t in s["toks"]: kidud.setdefault(t["head"], []).append(t)
    for t in s["toks"]:
        b = C.base(t["deprel"]); i = t["id"]
        if i not in conv["heads"]: continue
        kids = kidud.get(i, [])
        marks = [k for k in kids if C.base(k["deprel"]) == "mark"]
        has_subj = any(C.base(k["deprel"]) in ("nsubj", "csubj") for k in kids)
        typ = None
        if b == "conj" and t["upos"] in ("VERB", "AUX") and (has_subj or any(C.base(k["deprel"]) == "cop" for k in kids)): typ = "clause_coord"
        elif b == "advcl" and any(m["upos"] == "SCONJ" for m in marks): typ = "subordinate"
        elif b == "acl" and t["deprel"] == "acl:relcl": typ = "relative"
        elif b == "ccomp" and marks: typ = "complement"
        elif b in ("parataxis", "ccomp") and not marks and '"' in s["meta"]["text"]: typ = "reported_speech"
        elif b in ("advcl", "xcomp") and any(m["upos"] == "PART" for m in marks): typ = "purpose"
        if typ:
            h = T.get(t["head"])
            # find the converted node under which the link sits: the node on top of this clause (marker/coordinator) -> use clause node's ancestors
            top = i
            while conv["heads"][top] != 0:
                p = conv["heads"][top]
                if "cop" in conv["flags"][p] or conv["upos"][p] in ("SCONJ", "CCONJ", "PART") or (conv["upos"][p] == "ADP" and conv["labels"][top] in ("MODIFICATION", "SUBORDINATION", "PREPOSITION")):
                    top = p
                else: break
            pool[typ].append({"type": typ, "id": s["meta"]["sent_id"], "text": s["meta"]["text"],
                              "ud": f"{t['form']}({t['upos']}) -{t['deprel']}-> {h['form'] if h else 'ROOT'}", "marks": [m["form"] for m in marks],
                              "conv_up": (f"{conv['form'][conv['heads'][top]] if conv['heads'][top] else 'ROOT'} <-{conv['labels'][top]}- " ),
                              "conv_frag": frag(conv, top, s)})
out = []
for typ, lst in pool.items():
    random.shuffle(lst)
    seen = set(); k = 0
    for x in lst:
        if x["id"] in seen: continue
        seen.add(x["id"]); out.append(x); k += 1
        if k == 10: break
    print(typ, len(lst), file=sys.stderr)
json.dump(out, open(f"{HERE}/out/handjudge_60.json", "w"), indent=1)
for n, x in enumerate(out):
    print(f"[{n}] {x['type']} | {x['text'][:230]}\n    UD: {x['ud']} marks={x['marks']}\n    CONV: {x['conv_up']}{x['conv_frag'][:260]}")
