"""Step 6 (+7): convert whole treebanks with the frozen table; counts, well-formedness, label distribution, clause links."""
import os, sys, json, glob, collections, random, gzip
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import conllu, convert as C
UD = "/home/user/NAOMI/consciousness_transformer/data/ud"
table = json.load(open(sys.argv[1]))
which = sys.argv[2:]
CLAUSE_UD = ["conj", "advcl", "acl:relcl", "acl", "ccomp", "xcomp", "parataxis", "csubj"]
def clause_link(conv, i):
    """Between-clause link, one definition for UD and gold: a VERB (not AUX) that is not the root hangs from something."""
    return conv["upos"][i] == "VERB" and conv["heads"][i] != 0
def band(n): return "<=10" if n <= 10 else "11-20" if n <= 20 else "21-35" if n <= 35 else "36+"
for d in which:
    tot = collections.Counter(); lab = collections.Counter(); unm = collections.Counter(); ctype = collections.Counter()
    bands = collections.Counter(); cl_by_band = collections.Counter()
    for f in sorted(glob.glob(f"{UD}/{d}/*.conllu")):
        for s in conllu.read(f):
            conv = C.convert(s, table)
            if not conv["heads"]: tot["empty"] += 1; continue
            n = len(conv["heads"]); b = band(n)
            tot["sent"] += 1; tot["tok"] += n; tot["wf"] += conv["well_formed"]; tot["multi_root"] += (conv["n_roots"] != 1)
            bands[b] += 1; tot["gt20"] += n > 20; tot["gt35"] += n > 35
            for i in conv["heads"]:
                lab[conv["labels"][i]] += 1
                if "UNMAPPED" in conv["flags"][i]: unm[conv["flags"][i].split("UNMAPPED:")[1]] += 1
                if clause_link(conv, i):
                    cl_by_band[b] += 1; tot["clause_links"] += 1
                    ctype[C.base(conv["orig"][i])] += 1
    print("==", d, dict(tot))
    print("  length bands (sentences):", dict(bands))
    print("  clause links by UD relation of the verb:", ctype.most_common(10), "per band:", dict(cl_by_band))
    print("  labels:", {k: round(v / tot["tok"], 4) for k, v in lab.most_common()})
    print("  unmapped UD relations (default label used):", unm.most_common(10), sum(unm.values()))
