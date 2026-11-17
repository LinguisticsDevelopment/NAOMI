"""Minimal CoNLL-U reader (throwaway). Yields dicts with text, sent_id, tokens (syntactic words only), mwts."""
def read(path):
    meta, toks, mwts = {}, [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if toks:
                    yield {"meta": meta, "toks": toks, "mwts": mwts}
                meta, toks, mwts = {}, [], []
                continue
            if line.startswith("#"):
                if "=" in line:
                    k, v = line[1:].split("=", 1)
                    meta[k.strip()] = v.strip()
                continue
            c = line.split("\t")
            if "." in c[0]:
                continue  # empty nodes
            if "-" in c[0]:
                a, b = c[0].split("-")
                mwts.append((int(a), int(b), c[1]))
                continue
            toks.append({"id": int(c[0]), "form": c[1], "lemma": c[2], "upos": c[3], "xpos": c[4],
                         "feats": c[5], "head": int(c[6]), "deprel": c[7], "misc": c[9]})
