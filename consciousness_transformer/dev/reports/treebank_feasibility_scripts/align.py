"""Alignment of UD words to the project's tokens (character streams) + projection of converted trees onto project tokens."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert as C

_Q = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
def _n(s): return "".join(s.lower().translate(_Q).split())

def ud_units(sent):
    """[(surface, [word ids])]: an MWT whose word forms do not concatenate to its surface stays one unit."""
    mw = {a: (b, f) for a, b, f in sent["mwts"]}
    out, i, toks = [], 0, sent["toks"]
    while i < len(toks):
        t = toks[i]
        if t["id"] in mw:
            b, f = mw[t["id"]]
            ids = [x["id"] for x in toks if t["id"] <= x["id"] <= b]
            if _n("".join(x["form"] for x in toks if x["id"] in ids)) == _n(f):
                out += [(x["form"], [x["id"]]) for x in toks if x["id"] in ids]
            else:
                out.append((f, ids))
            i += len(ids)
        else:
            out.append((t["form"], [t["id"]])); i += 1
    return out

def align(sent, ptoks):
    """Return (groups, status). groups[p] = list of UD word ids for project token p (empty if none). status: ok | stream | split."""
    units = ud_units(sent)
    us = "".join(_n(s) for s, _ in units); ps = "".join(_n(t) for t in ptoks)
    if us != ps: return None, "stream"
    cu, cp = [], []
    for k, (s, _) in enumerate(units): cu += [k] * len(_n(s))
    for k, t in enumerate(ptoks): cp += [k] * len(_n(t))
    groups = [[] for _ in ptoks]
    ustart = {}
    for pos_, k in enumerate(cu): ustart.setdefault(k, pos_)
    # a unit must lie fully inside one project token, or the project token inside whole units
    p_units = [set() for _ in ptoks]; u_ptoks = {}
    for ch in range(len(cu)):
        p_units[cp[ch]].add(cu[ch]); u_ptoks.setdefault(cu[ch], set()).add(cp[ch])
    split_units = {k for k, ps_ in u_ptoks.items() if len(ps_) > 1}
    for p, us_ in enumerate(p_units):
        for k in sorted(us_):
            if k not in split_units: groups[p] += units[k][1]
    # project tokens carved out of a UD word keep no UD information (masked); counted via the status
    return groups, ("ok" if not split_units else "ok_masked")

def project(sent, conv, groups, ppos):
    """Project converted tree to project tokens. Returns dict p -> (head p or -1, label, orig_ud, flags) for project NON-PUNCT tokens
    that have a UD node; None entries for tokens without one."""
    w2p = {}
    for p, g in enumerate(groups):
        for w in g: w2p[w] = p
    res = {}
    for p, g in enumerate(groups):
        if ppos[p] == "PUNCT": continue
        ws = [w for w in g if w in conv["heads"]]
        if not ws: res[p] = None; continue
        gs = set(ws)
        rep = next((w for w in ws if conv["heads"][w] not in gs), ws[0])
        h = conv["heads"][rep]
        if h == 0: hp = -1
        else:
            hp = w2p.get(h)
            if hp is None or hp == p: res[p] = None; continue
        res[p] = (hp, conv["labels"][rep], conv["orig"][rep], conv["flags"][rep], conv["upos"][rep], rep)
    return res

def parser_edges(tree):
    """Parser tree dict -> (committed {child: (parent, rel)}, root token or None, connected bool, n_nodes, unknown set)."""
    edges, root, unattached, n = {}, None, [], 0
    def walk(nd, par):
        nonlocal root, n
        ti = nd["token_index"]
        if ti is not None:
            n += 1
            if nd["rel"] == "ROOT" and par is None: root = ti
            elif nd["rel"] == "UNATTACHED": unattached.append(ti)
            elif par is not None and par["token_index"] is not None: edges[ti] = (par["token_index"], nd["rel"])
        for c in nd["children"]: walk(c, nd if ti is not None else None)
    r = tree["root"]
    walk(r, None)
    # for a synthetic ROOT (token_index None) its children are tops: ROOT-rel or UNATTACHED
    connected = (r["token_index"] is not None) and not unattached
    return edges, (r["token_index"] if r["token_index"] is not None else None), connected, n, unattached
