"""Human-annotated treebank sentences (Universal Dependencies, CoNLL-U) as a second teacher for the encoder.

Three pieces, all language-neutral code (no words of any language; every relation name, part-of-speech
name and structural convention of the foreign scheme lives in ``data/treebank/conversion_table.json``):

1. ``read_conllu`` / ``convert``: translate one annotated sentence into the project's format (head and relation
   label per word), driven by the table. Every converted link carries the names of the table operations and
   the table rule that produced it (``Link.flags``).
2. ``align`` / ``project_links``: the treebank's words and the project's tokens can be split differently; the two
   character streams are aligned, and a token that does not line up with whole words is masked (unknown).
3. ``compose``: the supervision rule. The parser's committed links always win; a converted link is used only for
   a token the parser left as the top of an unconnected piece (or loose); it is dropped if it would make a cycle.
"""
from __future__ import annotations

import json
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TABLE_PATH = Path(__file__).resolve().parents[2] / "data" / "treebank" / "conversion_table.json"
SRC_PARSER, SRC_TREEBANK, SRC_UNKNOWN = "parser", "treebank", "unknown"


def load_table(path=TABLE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fetch(dest: Path, table: Optional[dict] = None, splits: Sequence[str] = ("train", "dev", "test")) -> List[Path]:
    """Download the treebank files named in the table (no-op for files already present)."""
    table = table or load_table()
    cfg = table["fetch"]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for s in splits:
        f = dest / cfg["files"][s]
        if not f.exists():
            urllib.request.urlretrieve(cfg["base_url"] + cfg["files"][s], f)
        out.append(f)
    return out


# ----------------------------------------------------------------------------- CoNLL-U
@dataclass
class Tok:
    id: int
    form: str
    lemma: str
    upos: str
    feats: str
    head: int
    deprel: str


@dataclass
class Sent:
    sent_id: str
    text: str
    toks: List[Tok]
    mwts: List[Tuple[int, int, str]]


def parse_conllu(lines) -> List[Sent]:
    out, meta, toks, mwts = [], {}, [], []
    for line in list(lines) + [""]:
        line = line.rstrip("\n")
        if not line:
            if toks:
                out.append(Sent(meta.get("sent_id", ""), meta.get("text", ""), toks, mwts))
            meta, toks, mwts = {}, [], []
        elif line.startswith("#"):
            if "=" in line:
                k, v = line[1:].split("=", 1)
                meta[k.strip()] = v.strip()
        else:
            c = line.split("\t")
            if "." in c[0]:
                continue                                  # empty node
            if "-" in c[0]:
                a, b = c[0].split("-")
                mwts.append((int(a), int(b), c[1]))
                continue
            toks.append(Tok(int(c[0]), c[1], c[2], c[3], c[5], int(c[6]), c[7]))
    return out


def read_conllu(path) -> List[Sent]:
    with open(path, encoding="utf-8") as f:
        return parse_conllu(f)


# ----------------------------------------------------------------------------- converter
class _N:
    __slots__ = ("id", "upos", "feats", "ud", "orig", "kids", "par", "label", "ops", "rule")

    def __init__(self, t: Tok):
        self.id, self.upos, self.ud, self.orig = t.id, t.upos, t.deprel, t.deprel
        self.feats = dict(kv.split("=", 1) for kv in t.feats.split("|")) if t.feats not in ("_", "") else {}
        self.kids: List["_N"] = []
        self.par: Optional["_N"] = None
        self.label: Optional[str] = None
        self.ops: List[str] = []
        self.rule = ""


_CUR: List[Optional[str]] = [None]      # name of the structural operation currently re-attaching nodes


def base(r: str) -> str:
    return r.split(":", 1)[0]


def _detach(n: _N):
    n.par.kids.remove(n)
    n.par = None


def _attach(n: _N, p: _N):
    n.par = p
    p.kids.append(n)
    if _CUR[0] and _CUR[0] not in n.ops:
        n.ops.append(_CUR[0])


def _replace(old: _N, new: _N):
    p = old.par
    p.kids[p.kids.index(old)] = new
    new.par, old.par = p, None
    if _CUR[0] and _CUR[0] not in new.ops:
        new.ops.append(_CUR[0])


def _build(sent: Sent, table: dict):
    nodes = {t.id: _N(t) for t in sent.toks}
    root = _N(Tok(0, "", "", "", "_", -1, table["root_rel"]))
    nodes[0] = root
    drop = set(table["ops"]["drop_punct"]["upos"])
    for t in sent.toks:
        if t.head in nodes:
            _attach(nodes[t.id], nodes[t.head])
    for t in sent.toks:
        n = nodes[t.id]
        if n.upos in drop:
            for k in list(n.kids):
                _detach(k)
                _attach(k, n.par)
            _detach(n)
            del nodes[n.id]
    return nodes, root


def _copula_swap(nodes, root, cfg):
    for i in sorted(nodes):
        P = nodes.get(i)
        if P is None or P is root:
            continue
        cops = [k for k in P.kids if base(k.ud) == cfg["marker_rel"]]
        if not cops:
            continue
        c = cops[0]
        mv, needs = set(cfg["move_to_copula_rels"]), set(cfg["move_conj_if_conjunct_has_rel"])
        move = [k for k in P.kids if k is not c and (base(k.ud) in mv or (
            base(k.ud) == cfg["conj_rel"] and any(base(g.ud) in needs for g in k.kids)))]
        _detach(c)
        old = P.ud
        _replace(P, c)
        c.ud = old
        for k in move:
            _detach(k)
            _attach(k, c)
        P.ud = cfg["pseudo_rel"]
        _attach(P, c)


def _phrase_head_swap(nodes, root, cfg):
    mrel = cfg["marker_rel"]
    for i in sorted(nodes):
        n0 = nodes.get(i)
        if n0 is None or n0 is root:
            continue
        ms = sorted([k for k in n0.kids if base(k.ud) == mrel], key=lambda x: x.id)
        if not ms:
            continue
        for m in ms:
            _detach(m)
        m1 = ms[0]
        for a, b in zip(ms, ms[1:]):
            _attach(b, a)
            b.ud = cfg["pseudo_rel"]
        tail = ms[-1]
        up = [k for k in n0.kids if base(k.ud) in cfg["bring_up_rels"] and k.ud != cfg["preconj_rel"]]
        if cfg.get("bring_up_conj_if_conjunct_has_marker"):
            up += [k for k in n0.kids if base(k.ud) == cfg["conj_rel"] and any(base(g.ud) == mrel for g in k.kids)]
        for k in up:
            _detach(k)
        old = n0.ud
        _replace(n0, m1)
        m1.ud = old
        n0.ud = cfg["pseudo_rel"]
        _attach(n0, tail)
        for k in up:
            _attach(k, m1)


def _coord_lift(nodes, root, cfg):
    cc, pre = cfg["cc_rel"], cfg["preconj_rel"]
    for i in sorted(nodes):
        x = nodes.get(i)
        if x is None:
            continue
        ks = sorted([k for k in x.kids if base(k.ud) == cfg["conj_rel"]], key=lambda y: y.id)
        if not ks:
            continue
        ccs = [g for k in ks for g in k.kids if base(g.ud) == cc and g.ud != pre]
        ccs += [g for g in x.kids if base(g.ud) == cc and g.ud != pre
                and (g.id > x.id or not cfg["coordinator_must_follow_first_conjunct"])]
        ccs = sorted(ccs, key=lambda y: y.id)
        if not ccs:
            for k in ks:
                k.ud = cfg["pseudo_asyndeton"]
            continue
        c = ccs[0]
        _detach(c)
        for k in ks:
            _detach(k)
        for e in ccs[1:]:
            _detach(e)
        pres = [g for g in x.kids if g.ud == pre]
        for g in pres:
            _detach(g)
        old = x.ud
        _replace(x, c)
        c.ud = old
        x.ud = cfg["pseudo_item"]
        _attach(x, c)
        for k in ks:
            k.ud = cfg["pseudo_item"]
            _attach(k, c)
        for e in ccs[1:] + pres:
            _attach(e, c)


def _match(rule: dict, n: _N, head: _N) -> bool:
    u = rule["ud"]
    us = [u] if isinstance(u, str) else u
    if not (n.ud in us or base(n.ud) in us):
        return False
    if "dep_upos" in rule and n.upos not in rule["dep_upos"]:
        return False
    if "head_upos" in rule and head.upos not in rule["head_upos"]:
        return False
    if "dep_feat" in rule:
        k, v = rule["dep_feat"].split("=")
        if n.feats.get(k) != v:
            return False
    return True


@dataclass
class Conv:
    heads: Dict[int, int]            # word id -> head word id (0 = root)
    labels: Dict[int, str]
    orig: Dict[int, str]
    upos: Dict[int, str]
    flags: Dict[int, str]            # operations that moved the word + the table rule that labelled it
    well_formed: bool


def convert(sent: Sent, table: dict) -> Conv:
    nodes, root = _build(sent, table)
    ops = table["ops"]
    for name in table["op_order"]:
        if name == "copula_swap":
            _CUR[0] = name
            _copula_swap(nodes, root, ops[name])
        elif name == "phrase_head_swap":
            for cfg in ops[name]:
                _CUR[0] = cfg["name"]
                _phrase_head_swap(nodes, root, cfg)
        elif name == "coord_lift":
            _CUR[0] = name
            _coord_lift(nodes, root, ops[name])
    _CUR[0] = None
    for n in nodes.values():
        if n is root:
            continue
        if n.par is root:
            n.label, n.rule = table["root_label"], "root"
            continue
        for r in table["rules"]:
            if _match(r, n, n.par):
                n.label, n.rule = r["label"], r["name"]
                break
        else:
            n.label, n.rule = table["default_label"], "UNMAPPED:" + n.ud
    ids = [i for i, n in nodes.items() if n is not root]
    heads = {i: (nodes[i].par.id if nodes[i].par is not root else 0) for i in ids}
    ok = sum(h == 0 for h in heads.values()) == 1
    for i in heads:
        seen, x = set(), i
        while x != 0 and x not in seen:
            seen.add(x)
            x = heads[x]
        ok = ok and x == 0
    return Conv(heads, {i: nodes[i].label for i in ids}, {i: nodes[i].orig for i in ids},
                {i: nodes[i].upos for i in ids}, {i: "+".join(nodes[i].ops + [nodes[i].rule]) for i in ids}, ok)


# ----------------------------------------------------------------------------- alignment to the project's tokens
_Q = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _norm(s: str) -> str:
    return "".join(s.lower().translate(_Q).split())


def ud_units(sent: Sent) -> List[Tuple[str, List[int]]]:
    """[(surface, [word ids])]: a multi-word token whose word forms do not concatenate to its surface stays one unit."""
    mw = {a: (b, f) for a, b, f in sent.mwts}
    out, i, toks = [], 0, sent.toks
    while i < len(toks):
        t = toks[i]
        if t.id in mw:
            b, f = mw[t.id]
            part = [x for x in toks if t.id <= x.id <= b]
            if _norm("".join(x.form for x in part)) == _norm(f):
                out += [(x.form, [x.id]) for x in part]
            else:
                out.append((f, [x.id for x in part]))
            i += len(part)
        else:
            out.append((t.form, [t.id]))
            i += 1
    return out


ALIGN_FAIL, ALIGN_OK, ALIGN_MASKED = "stream", "ok", "ok_masked"


def align(sent: Sent, ptoks: Sequence[str]):
    """-> (groups, status); groups[p] = treebank word ids that project token p consists of (empty = masked)."""
    units = ud_units(sent)
    us = "".join(_norm(s) for s, _ in units)
    ps = "".join(_norm(t) for t in ptoks)
    if us != ps:
        return None, ALIGN_FAIL
    cu, cp = [], []
    for k, (s, _) in enumerate(units):
        cu += [k] * len(_norm(s))
    for k, t in enumerate(ptoks):
        cp += [k] * len(_norm(t))
    p_units = [set() for _ in ptoks]
    u_ptoks: Dict[int, set] = {}
    for ch in range(len(cu)):
        p_units[cp[ch]].add(cu[ch])
        u_ptoks.setdefault(cu[ch], set()).add(cp[ch])
    split = {k for k, v in u_ptoks.items() if len(v) > 1}
    groups = [[] for _ in ptoks]
    for p, us_ in enumerate(p_units):
        for k in sorted(us_):
            if k not in split:
                groups[p] += units[k][1]
    return groups, (ALIGN_OK if not split else ALIGN_MASKED)


@dataclass
class Link:
    head: int      # project token index, -1 = root
    label: str
    orig: str      # the treebank's own relation name
    flags: str     # operations + table rule that produced this link


def project_links(conv: Conv, groups, is_node: Sequence[bool]) -> Dict[int, Link]:
    """Converted tree -> links between project tokens (only for node tokens with an aligned word and head)."""
    w2p = {w: p for p, g in enumerate(groups) for w in g}
    res: Dict[int, Link] = {}
    for p, g in enumerate(groups):
        ws = [w for w in g if w in conv.heads]
        if not is_node[p] or not ws:
            continue
        gs = set(ws)
        rep = next((w for w in ws if conv.heads[w] not in gs), ws[0])
        h = conv.heads[rep]
        if h == 0:
            hp = -1
        else:
            hp = w2p.get(h)
            if hp is None or hp == p:
                continue
        res[p] = Link(hp, conv.labels[rep], conv.orig[rep], conv.flags[rep])
    return res


# ----------------------------------------------------------------------------- supervision composition
@dataclass
class Composed:
    head: List[int]                 # 0 = root, j+1 = token j, -1 = none
    rel: List[Optional[str]]
    source: List[str]
    flags: List[str]
    counts: Counter = field(default_factory=Counter)


def compose(sv_head, sv_rel, sv_sup, sv_unknown, links: Dict[int, Link], is_node) -> Composed:
    """Parser links win everywhere. A treebank link is added only for a parser-unknown token, if it exists, its head
    is a node, it makes no cycle with the links accepted so far, and (for a root claim) no root exists yet."""
    n = len(sv_head)
    head, rel = list(sv_head), list(sv_rel)
    source = [SRC_PARSER if s else SRC_UNKNOWN for s in sv_sup]
    flags = [""] * n
    cnt: Counter = Counter()
    cur = {i: sv_head[i] for i in range(n) if sv_sup[i]}
    has_root = any(h == 0 for h in cur.values())
    for i in range(n):
        if not sv_unknown[i]:
            continue
        L = links.get(i)
        if L is None:
            cnt["no_link"] += 1
            continue
        if L.head == -1:
            if has_root:
                cnt["root_conflict"] += 1
                continue
            new = 0
        else:
            if not is_node[L.head]:
                cnt["head_not_node"] += 1
                continue
            x, cyc = L.head, False
            while True:
                if x == i:
                    cyc = True
                    break
                h = cur.get(x)
                if h is None or h == 0:
                    break
                x = h - 1
            if cyc:
                cnt["cycle_dropped"] += 1
                continue
            new = L.head + 1
        cur[i] = new
        has_root = has_root or new == 0
        head[i], rel[i], source[i], flags[i] = new, L.label, SRC_TREEBANK, L.flags
        cnt["accepted"] += 1
    return Composed(head, rel, source, flags, cnt)
