"""Step 7: apply the frozen table, unchanged, to 300 Spanish UD (GSD) sentences. Structural well-formedness only (no accuracy claim)."""
import os, sys, json, random, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import conllu, convert as C
UD = "/home/user/NAOMI/consciousness_transformer/data/ud"
table = json.load(open(sys.argv[1]))
for name, path in [("es_gsd", "es_gsd/es_gsd-ud-train.conllu"), ("es_ancora(sanity)", "es_ancora/es_ancora-ud-dev.conllu")]:
    S = list(conllu.read(f"{UD}/{path}")); random.seed(0); random.shuffle(S); S = [s for s in S if sum(t["upos"] != "PUNCT" for t in s["toks"]) >= 3][:300]
    tot = collections.Counter(); lab = collections.Counter(); unm = collections.Counter(); ud_rel = collections.Counter(); depth = []
    ex = []
    for s in S:
        conv = C.convert(s, table); n = len(conv["heads"])
        tot["sent"] += 1; tot["tok"] += n; tot["one_root"] += conv["n_roots"] == 1; tot["wellformed"] += conv["well_formed"]
        for i in conv["heads"]:
            lab[conv["labels"][i]] += 1; ud_rel[C.base(conv["orig"][i])] += 1
            if "UNMAPPED" in conv["flags"][i]: unm[conv["flags"][i].split("UNMAPPED:")[1]] += 1
        # depth + every token has a head (guaranteed by construction; verify)
        d = 0
        for i in conv["heads"]:
            x, k = i, 0
            while x != 0: x = conv["heads"][x]; k += 1
            d = max(d, k)
        depth.append(d)
        tot["every_token_attached"] += all(i in conv["heads"] for i in conv["heads"].values() if i != 0)
        tot["mwt_sent"] += bool(s["mwts"])
        if len(ex) < 3 and 8 <= n <= 14:
            from naomi.tree import compact, tree_from_dict
            ex.append((s["meta"]["text"], compact(tree_from_dict(C.to_project_tree(s, conv)))))
    print("==", name, dict(tot), "max depth p50/p90:", sorted(depth)[len(depth)//2], sorted(depth)[int(len(depth)*.9)])
    print("  labels:", {k: round(v / tot["tok"], 3) for k, v in lab.most_common()})
    print("  UD relations seen:", len(ud_rel), " unmapped (default label):", dict(unm))
    for t, c in ex: print("  EX:", t[:100], "\n     ", c[:220])
