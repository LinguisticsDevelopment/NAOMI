"""Throwaway converter: UD CoNLL-U sentence -> project-format tree, driven by conversion_table.json.
No language-specific logic: relation names and UPOS/FEATS conditions come from the table (data)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = json.load(open(os.path.join(HERE, "conversion_table.json")))

class N:
    __slots__ = ("id", "form", "lemma", "upos", "feats", "ud", "kids", "par", "label", "flag", "orig_ud")
    def __init__(s, t):
        s.id, s.form, s.lemma, s.upos = t["id"], t["form"], t["lemma"], t["upos"]
        s.feats = dict(kv.split("=", 1) for kv in t["feats"].split("|")) if t["feats"] not in ("_", "") else {}
        s.ud = s.orig_ud = t["deprel"]; s.kids = []; s.par = None; s.label = None; s.flag = ""
def base(r): return r.split(":", 1)[0]

def _detach(n):
    n.par.kids.remove(n); n.par = None
def _attach(n, p, at=None):
    n.par = p; p.kids.append(n)

def _replace(old, new):
    """new takes old's place under old's parent (new must be detached)."""
    p = old.par; i = p.kids.index(old); p.kids[i] = new; new.par = p; old.par = None

def build(sent, table=TABLE):
    toks = sent["toks"]
    nodes = {t["id"]: N(t) for t in toks}
    root = N({"id": 0, "form": "ROOT", "lemma": "ROOT", "upos": "ROOT", "feats": "_", "deprel": "ROOT"})
    nodes[0] = root
    drop = set(table["ops"]["drop_punct"]["upos"])
    for t in toks:
        n = nodes[t["id"]]
        if t["head"] in nodes: _attach(n, nodes[t["head"]])
    # drop punctuation (leaf by UD validity; any children re-parented to its head)
    for t in toks:
        n = nodes[t["id"]]
        if n.upos in drop or base(n.ud) == "punct":
            for k in list(n.kids): _detach(k); _attach(k, n.par)
            _detach(n); del nodes[n.id]
    return nodes, root

def copula_swap(nodes, root, cfg):
    for i in sorted(nodes):
        P = nodes.get(i)
        if P is None or P is root: continue
        cops = [k for k in P.kids if base(k.ud) == cfg["marker_rel"]]
        if not cops: continue
        c = cops[0]
        mv = set(cfg["move_to_copula_rels"])
        needs = set(cfg["move_conj_if_conjunct_has_rel"])
        move = []
        for k in P.kids:
            if k is c: continue
            if base(k.ud) in mv: move.append(k)
            elif base(k.ud) == "conj" and any(base(g.ud) in needs for g in k.kids): move.append(k)
        _detach(c)
        old_ud = P.ud
        _replace(P, c); c.ud = old_ud
        for k in move: _detach(k); _attach(k, c)
        P.ud = cfg["pseudo_rel"]; _attach(P, c)
        c.flag = "cop"

def phrase_head_swap(nodes, root, cfg):
    mrel = cfg["marker_rel"]
    for i in sorted(nodes):
        N0 = nodes.get(i)
        if N0 is None or N0 is root: continue
        ms = sorted([k for k in N0.kids if base(k.ud) == mrel], key=lambda x: x.id)
        if not ms: continue
        for m in ms: _detach(m)
        m1 = ms[0]
        for a, b in zip(ms, ms[1:]):  # chain m1 -> m2 ...; tail -> N0
            _attach(b, a); b.ud = cfg["pseudo_rel"]
        tail = ms[-1]
        up = [k for k in N0.kids if base(k.ud) in cfg["bring_up_rels"] and k.ud != "cc:preconj"]
        if cfg.get("bring_up_conj_if_conjunct_has_marker"):
            up += [k for k in N0.kids if base(k.ud) == "conj" and any(base(g.ud) == mrel for g in k.kids)]
        for k in up: _detach(k)
        old_ud = N0.ud
        _replace(N0, m1); m1.ud = old_ud
        N0.ud = cfg["pseudo_rel"]; _attach(N0, tail)
        for k in up: _attach(k, m1)

def coord_lift(nodes, root, cfg):
    for i in sorted(nodes):
        X = nodes.get(i)
        if X is None: continue
        K = sorted([k for k in X.kids if base(k.ud) == cfg["conj_rel"]], key=lambda x: x.id)
        if not K: continue
        ccs = [g for k in K for g in k.kids if base(g.ud) == cfg["cc_rel"] and g.ud != "cc:preconj"]
        ccs += [g for g in X.kids if base(g.ud) == cfg["cc_rel"]]
        ccs = sorted(ccs, key=lambda x: x.id)
        if not ccs:
            for k in K: k.ud = cfg["pseudo_asyndeton"]
            continue
        C = ccs[0]
        _detach(C)
        for k in K: _detach(k)
        for e in ccs[1:]: _detach(e)
        pre = [g for g in X.kids if g.ud == "cc:preconj"]
        for g in pre: _detach(g)
        old_ud = X.ud
        _replace(X, C); C.ud = old_ud
        X.ud = cfg["pseudo_item"]; _attach(X, C)
        for k in K: k.ud = cfg["pseudo_item"]; _attach(k, C)
        for e in ccs[1:] + pre: _attach(e, C)
        C.flag = "coord"

def _match(rule, n, head):
    u = rule["ud"]; us = [u] if isinstance(u, str) else u
    if not (n.ud in us or base(n.ud) in us):
        return False
    if "dep_upos" in rule and n.upos not in rule["dep_upos"]: return False
    if "head_upos" in rule and head.upos not in rule["head_upos"]: return False
    if "dep_feat" in rule:
        k, v = rule["dep_feat"].split("=")
        if n.feats.get(k) != v: return False
    return True

def label_all(nodes, root, table):
    unm = 0
    for i, n in nodes.items():
        if n is root: continue
        if n.par is root: n.label = "ROOT"; continue
        for ri, r in enumerate(table["rules"]):
            if _match(r, n, n.par):
                n.label = r["label"]; n.flag = n.flag or ""; n.flag += f"r{ri}"; break
        else:
            n.label = table["default_label"]; n.flag += "UNMAPPED:" + n.ud

def convert(sent, table=TABLE):
    nodes, root = build(sent, table)
    ops = table["ops"]
    for name in table["op_order"]:
        if name == "copula_swap": copula_swap(nodes, root, ops[name])
        elif name == "phrase_head_swap":
            for cfg in ops[name]: phrase_head_swap(nodes, root, cfg)
        elif name == "coord_lift": coord_lift(nodes, root, ops[name])
    label_all(nodes, root, table)
    heads = {i: (n.par.id if n.par is not root else 0) for i, n in nodes.items() if n is not root}
    labels = {i: n.label for i, n in nodes.items() if n is not root}
    # well-formedness: exactly one root and a path to it from every node
    roots = [i for i, h in heads.items() if h == 0]
    ok = len(roots) == 1
    for i in heads:
        seen, x = set(), i
        while x != 0 and x not in seen: seen.add(x); x = heads[x]
        if x != 0: ok = False
    return {"heads": heads, "labels": labels, "orig": {i: n.orig_ud for i, n in nodes.items() if n is not root},
            "upos": {i: n.upos for i, n in nodes.items() if n is not root},
            "form": {i: n.form for i, n in nodes.items() if n is not root},
            "lemma": {i: n.lemma for i, n in nodes.items() if n is not root},
            "flags": {i: n.flag for i, n in nodes.items() if n is not root}, "well_formed": ok, "n_roots": len(roots)}

def to_project_tree(sent, conv, fill_senses=False):
    """CoNLL-U sentence + conversion -> naomi Tree (nested JSON contract). Lemma/POS from the UD columns."""
    from naomi.types import Node, Tree, KIND_NAME, KIND_WORD, REL_ROOT
    from naomi.tree import tree_to_dict
    if not conv["heads"]: return None
    ix = {t["id"]: k for k, t in enumerate(sent["toks"])}
    nm = {}
    for i in conv["heads"]:
        pos = conv["upos"][i]; kind = KIND_NAME if pos in ("PROPN", "NUM") else KIND_WORD
        nm[i] = Node(rel=conv["labels"][i], token=conv["form"][i].lower(), token_index=ix[i], lemma=conv["lemma"][i].lower(), pos=pos, kind=kind)
    root = None
    for i, h in conv["heads"].items():
        if h == 0: nm[i].rel = REL_ROOT; root = nm[i]
        elif h in nm: nm[h].children.append(nm[i])
    for n in nm.values(): n.children.sort(key=lambda c: c.token_index)
    t = Tree(sentence=sent["meta"].get("text", ""), root=root, source="ud_converted")
    return tree_to_dict(t)

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    import conllu
    from naomi.tree import tree_from_dict, compact
    for k, s in enumerate(conllu.read(sys.argv[1])):
        if k >= int(sys.argv[2]): break
        print(s["meta"]["text"])
        d = to_project_tree(s, convert(s))
        print("  ", compact(tree_from_dict(d)) if d else None)
