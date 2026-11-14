"""Grammar-backwards decoder prototype: meaning tree -> candidate token sequences.

CONSTRAINT: this file holds NO language patterns.  Word order is read, as data,
from the parser's grammar file (``quantum_parser/grammars/<lang>.json``); every
function-word slot is reported as a slot, never filled with a word from a list
here.  Selection among candidates is done by the grammar's own judge, the
deterministic parser (re-parse, keep the candidate whose tree agrees best).

Pipeline
  derive_order_table(grammar)       grammar JSON -> OrderTable (entries + lift closure)
  linearize(tree, table, pos)       tree -> candidate token sequences (own tokens only)
  decode(tree, table, parser)       candidates -> re-parse -> best by edge-F1

A *tree* here is the gold-v6 clause shape (``{"clauses": [{"predicate": str,
"roles": [{"relation", "word"}]}]}``); token indices are IGNORED (they would
leak the original order).  ``pos`` maps a token string to its parser POS tag
name (the grounded sense provides this in the real pipeline).
"""
from __future__ import annotations

import itertools
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

CT_DIR = Path(__file__).resolve().parents[2]
REPO = CT_DIR.parent
DEFAULT_GRAMMAR = REPO / "quantum_parser" / "grammars" / "english.json"
SLOT = "<%s>"  # slot token, e.g. "<PREP>": a function-word position the grammar requires

_REF = re.compile(r"^(anchor|before\[(\d+|\*)\]|after\[(\d+|\*)\])$")


# --------------------------------------------------------------------------- grammar as data
@dataclass(frozen=True)
class Entry:
    """One licensed (head, relation, dependent) adjacency read from a grammar rule."""
    rule: str
    head: str          # anchor category (NodeType name)
    dep: str           # dependent slot category
    rel: str           # connection type (SUBJECT, DESCRIPTION, MODIFICATION, PREPOSITION...)
    side: str          # "before" | "after" the anchor
    idx: int           # position in that side's slot list (-1 = quantifier-many)
    result: str        # category the rule builds
    extra: Tuple[str, ...] = ()   # other slots the rule needs (function words / punctuation)


@dataclass
class OrderTable:
    entries: List[Entry] = field(default_factory=list)
    lift: Dict[str, set] = field(default_factory=dict)   # cat -> cats it can serve as (closure)
    conn_types: set = field(default_factory=set)         # all connection names in the grammar
    tag_to_cat: Dict[str, str] = field(default_factory=dict)

    def serves(self, cat: str, wanted: str) -> bool:
        return cat == wanted or wanted in self.lift.get(cat, ())

    def find(self, rel: str, head_cat: str, dep_cat: str, allow_extra=False) -> List[Entry]:
        return [e for e in self.entries if e.rel == rel and self.serves(head_cat, e.head)
                and self.serves(dep_cat, e.dep) and (allow_extra or not e.extra)]


def load_grammar(path=DEFAULT_GRAMMAR) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ref(s: str):
    m = _REF.match(s)
    if not m:
        return None
    if s == "anchor":
        return ("anchor", 0)
    side = "before" if s.startswith("before") else "after"
    k = s[s.index("[") + 1:-1]
    return (side, -1 if k == "*" else int(k))


def derive_order_table(grammar: dict, tag_to_cat: Optional[Dict[str, str]] = None) -> OrderTable:
    """Read adjacency/ordering facts out of the rule patterns.  Pure data transform."""
    if tag_to_cat is None:  # the parser's own POS-tag -> node-category map (data in enums.py)
        qp = str(REPO / "quantum_parser")
        if qp not in sys.path:
            sys.path.insert(0, qp)
        from src.parser.enums import TAG_TO_NODE_TYPE
        tag_to_cat = {t.name: n.name for t, n in TAG_TO_NODE_TYPE.items()}
    tb = OrderTable(tag_to_cat=dict(tag_to_cat))
    for name in grammar["order"]:
        for r in grammar["rulesets"][name]["rules"]:
            pat = r["pattern"]
            anchor = pat["anchor"]
            slots = {("before", i): s for i, s in enumerate(pat.get("before", []))}
            slots.update({("after", i): s for i, s in enumerate(pat.get("after", []))})
            conns = r.get("connections", [])
            for c in conns:
                tb.conn_types.add(c["type"])
            if not conns and not slots:       # unit lift: anchor X builds Y  (e.g. NOUN -> NOMINAL)
                if r["result"] != anchor["type"] and not anchor.get("subtypes"):
                    tb.lift.setdefault(anchor["type"], set()).add(r["result"])
                continue
            touched, new = set(), []
            for c in conns:
                a, b = _ref(c["from"]), _ref(c["to"])
                if a is None or b is None or (a[0] == "anchor") == (b[0] == "anchor"):
                    continue
                dep = b if a[0] == "anchor" else a
                key = (dep[0], max(dep[1], 0))
                if key not in slots:
                    continue
                touched.add(key)
                if dep[1] < 0:
                    touched.update(k for k in slots if k[0] == dep[0])
                s = slots[key]
                if anchor.get("subtypes") or s.get("subtypes") or s.get("subcategories"):
                    continue
                new.append((c["type"], s["type"], dep[0], dep[1]))
            extra = tuple(s["type"] + ("/" + "|".join(s.get("subtypes", [])) if s.get("subtypes") else "")
                          for k, s in slots.items() if k not in touched)
            for rel, dcat, side, idx in new:
                tb.entries.append(Entry(name, anchor["type"], dcat, rel, side, idx, r["result"], extra))
    # transitive closure of lift
    changed = True
    while changed:
        changed = False
        for k, vs in list(tb.lift.items()):
            for v in list(vs):
                for w in tb.lift.get(v, ()):
                    if w not in vs:
                        vs.add(w)
                        changed = True
    return tb


def required_function_slots(table: OrderTable) -> Dict[str, List[str]]:
    """Which categories the grammar needs as their own tokens (anchors that are not content
    categories' lifts): reported for the feasibility study, not used to pick words."""
    out: Dict[str, List[str]] = {}
    for e in table.entries:
        if e.rel in ("PREPOSITION", "COORDINATION"):
            out.setdefault(e.head, []).append(f"{e.rule}:{e.side}[{e.idx}]->{e.dep}")
    return out


# --------------------------------------------------------------------------- tree -> candidates
@dataclass
class Node:
    rel: str            # "PREDICATE" for the head, else the role label
    tok: str
    cat: str            # grammar category of the token (via POS tag)


@dataclass
class Option:
    head: int           # index of the head node
    side: str
    idx: int
    prep: bool = False  # realised as a PP: needs a PREP slot before the dependent
    fallback: bool = False


def _cat(table: OrderTable, tok: str, pos: Dict[str, str], node_pos: Optional[str]) -> str:
    tag = node_pos or pos.get(tok) or pos.get(tok.lower()) or "X"
    return table.tag_to_cat.get(tag, "NIL")


def clause_nodes(clause: dict, table: OrderTable, pos: Dict[str, str]) -> List[Node]:
    nodes = [Node("PREDICATE", clause.get("predicate") or "", _cat(table, clause.get("predicate") or "", pos, None))]
    for r in clause["roles"]:
        w = r.get("word")
        if w:
            nodes.append(Node(r["relation"], w, _cat(table, w, pos, r.get("pos"))))
    return nodes


def attach_options(table: OrderTable, nodes: List[Node], i: int) -> List[Option]:
    """Where may role ``i`` hang?  Every (head, side, index) the grammar licenses; if the grammar
    licenses none, a flagged fallback (any side of the predicate)."""
    r, opts = nodes[i], []
    for h, hn in enumerate(nodes):
        if h == i:
            continue
        if r.rel in table.conn_types:                      # direct relation (SUBJECT, DESCRIPTION...)
            for e in table.find(r.rel, hn.cat, r.cat):
                opts.append(Option(h, e.side, max(e.idx, 0)))
        else:                                              # PP-type relation: PREPOSITION rule + MODIFICATION rule
            for pe in table.find("PREPOSITION", "PREP", r.cat) + [
                    e for e in table.entries if e.rel == "PREPOSITION" and table.serves(r.cat, e.dep) and not e.extra]:
                for me in table.find("MODIFICATION", hn.cat, pe.result):
                    opts.append(Option(h, me.side, max(me.idx, 0), prep=True))
    if not opts:
        opts = [Option(0, "after", 9, fallback=True), Option(0, "before", 9, fallback=True)]
    seen, out = set(), []
    for o in opts:
        k = (o.head, o.side, o.idx, o.prep)
        if k not in seen:
            seen.add(k)
            out.append(o)
    return out


def _orderings(items: list, cap: int = 6):
    return list(itertools.islice(itertools.permutations(items), cap)) if len(items) > 1 else [tuple(items)]


def _realise(nodes, children, h, slot_fill, rel_of, cap) -> List[List[str]]:
    """All token orderings of the subtree at h (bounded)."""
    outs = [[nodes[h].tok]]
    for side in ("before", "after"):
        kids = sorted((c for c in children.get(h, []) if c[1] == side), key=lambda c: c[2])
        if not kids:
            continue
        groups = [list(g) for _, g in itertools.groupby(kids, key=lambda c: c[2])]
        side_variants: List[List[str]] = [[]]
        for g in groups:
            new = []
            for perm in _orderings(g, cap):
                parts = [[]]
                for (d, _s, _i, prep) in perm:
                    subs = _realise(nodes, children, d, slot_fill, rel_of, cap)
                    pre = []
                    if prep:
                        w = slot_fill("PREP", nodes[d].rel, nodes[d].tok)
                        pre = [w] if w else []
                    parts = [p + pre + s for p in parts for s in subs[:cap]][:cap * 4]
                new.extend(sv + p for sv in side_variants for p in parts)
            side_variants = new[:cap * 8]
        outs = ([sv + o for o in outs for sv in side_variants] if side == "before"
                else [o + sv for o in outs for sv in side_variants])
    return outs[:cap * 8]


def default_slot_fill(slot: str, rel: str, tok: str) -> Optional[str]:
    """Function-word slot filler.  The grammar says a PREP is required before a PP filler but not
    WHICH word; the only word-bearing information in a tree is the relation label itself when it
    is an open PP role (OF, WITH, ...: the label *is* the preposition token, uppercased).  Closed
    labels (PLACE, SOURCE...) collapse many prepositions -> None (slot left unfilled)."""
    if slot == "PREP" and rel not in ("PLACE", "SOURCE", "AGENT", "RECIPIENT", "TIME", "REASON", "MANNER"):
        return rel.lower()
    return None


def linearize_clause(clause: dict, table: OrderTable, pos: Dict[str, str],
                     slot_fill: Callable = default_slot_fill, max_candidates: int = 60,
                     max_assign: int = 200) -> List[List[str]]:
    nodes = clause_nodes(clause, table, pos)
    opts = [attach_options(table, nodes, i) for i in range(1, len(nodes))]
    cands: List[List[str]] = []
    seen = set()
    for assign in itertools.islice(itertools.product(*opts), max_assign):
        children: Dict[int, list] = {}
        ok = True
        for i, o in enumerate(assign, start=1):
            children.setdefault(o.head, []).append((i, o.side, o.idx, o.prep))
        # reject cycles / unreachable nodes: every node must be reachable from the predicate (0)
        reach, stack = set(), [0]
        while stack:
            n = stack.pop()
            if n in reach:
                continue
            reach.add(n)
            stack.extend(c[0] for c in children.get(n, []))
        if len(reach) != len(nodes):
            continue
        for toks in _realise(nodes, children, 0, slot_fill, None, 4):
            k = tuple(toks)
            if k not in seen:
                seen.add(k)
                cands.append(list(toks))
            if len(cands) >= max_candidates:
                return cands
    return cands


def linearize(tree: dict, table: OrderTable, pos: Dict[str, str],
              slot_fill: Callable = default_slot_fill, max_candidates: int = 60,
              coordinators: Optional[Sequence[Optional[str]]] = None) -> List[List[str]]:
    """Candidate token sequences for a whole tree (own tokens + required slots only)."""
    per = [linearize_clause(c, table, pos, slot_fill, max_candidates) or [[c.get("predicate") or ""]]
           for c in tree["clauses"]]
    if len(per) == 1:
        return per[0][:max_candidates]
    out, seen = [], set()
    for order in itertools.islice(itertools.permutations(range(len(per))), 6):
        for combo in itertools.islice(itertools.product(*[per[i][:6] for i in order]), max_candidates):
            toks: List[str] = []
            for j, part in enumerate(combo):
                if j and coordinators and j - 1 < len(coordinators) and coordinators[j - 1]:
                    toks.append(coordinators[j - 1])
                toks += part
            if tuple(toks) not in seen:
                seen.add(tuple(toks))
                out.append(toks)
            if len(out) >= max_candidates:
                return out
    return out


# --------------------------------------------------------------------------- the parser as judge
def tree_edges(tree: dict) -> Counter:
    """Order-free edge multiset of a gold-shaped tree: (predicate, role, filler), by token string."""
    out: Counter = Counter()
    for c in tree["clauses"]:
        p = (c.get("predicate") or "").lower()
        out[(p, "PREDICATE", "")] += 1
        for r in c["roles"]:
            if r.get("word"):
                out[(p, r["relation"], r["word"].lower())] += 1
    return out


def edge_f1(gold: Counter, pred: Counter) -> Tuple[float, float, float]:
    inter = sum((gold & pred).values())
    p = inter / max(sum(pred.values()), 1)
    r = inter / max(sum(gold.values()), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


class Judge:
    """Wraps the deterministic parser: tokens -> order-free edge multiset (or None if no parse)."""

    def __init__(self, max_hypotheses: int = 1500, max_seconds: float = 1.0):
        for d in (CT_DIR / "src", ):
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
        from nsm_ct.clause import extract_discourse
        from nsm_ct.input_encoder import ParserInputEncoder
        from nsm_ct.nsm_primes import PRIME_NAMES
        from nsm_ct.structure import PARSE_LABELS
        from nsm_ct.tokenizer import SimpleTokenizer
        self._extract = extract_discourse
        self.parser = ParserInputEncoder(SimpleTokenizer.build([], extra_tokens=list(PRIME_NAMES) + PARSE_LABELS))
        self.max_h, self.max_s = max_hypotheses, max_seconds
        self.cache: Dict[str, Optional[Counter]] = {}
        self.n_calls = 0

    def parse_edges(self, tokens: Sequence[str]) -> Optional[Counter]:
        text = " ".join(tokens)
        if text in self.cache:
            return self.cache[text]
        self.n_calls += 1
        res: Optional[Counter] = None
        try:
            graphs, _s, _m = self.parser._parse_topk_one(text, k=1, max_hypotheses=self.max_h,
                                                         max_seconds=self.max_s)
            if graphs:
                clauses, _links = self._extract(graphs[0])
                if clauses:
                    res = tree_edges({"clauses": [
                        {"predicate": c.predicate, "roles": [{"relation": r, "word": a.token} for r, a in c.args]}
                        for c in clauses]})
        except Exception:
            res = None
        self.cache[text] = res
        return res


@dataclass
class Decoded:
    tokens: List[str]
    f1: float
    parsed: bool
    n_candidates: int
    n_parsed: int
    seconds: float


def decode(tree: dict, table: OrderTable, judge: Judge, pos: Dict[str, str],
           slot_fill: Callable = default_slot_fill, max_candidates: int = 60,
           coordinators: Optional[Sequence[Optional[str]]] = None) -> Decoded:
    """tree -> best candidate by re-parse agreement (search-by-parser over grammar-derived candidates)."""
    t0 = time.perf_counter()
    gold = tree_edges(tree)
    cands = linearize(tree, table, pos, slot_fill, max_candidates, coordinators)
    best, best_f, n_ok = (cands[0] if cands else []), -1.0, 0
    for c in cands:
        e = judge.parse_edges(c)
        if e is None:
            continue
        n_ok += 1
        f = edge_f1(gold, e)[2]
        if f > best_f:
            best, best_f = c, f
            if f >= 1.0:
                break
    return Decoded(best, max(best_f, 0.0), n_ok > 0, len(cands), n_ok, time.perf_counter() - t0)


def bleu(ref: Sequence[str], hyp: Sequence[str], n: int = 4) -> float:
    """Smoothed sentence BLEU-n (lowercased), for the report only."""
    import math
    ref, hyp = [t.lower() for t in ref], [t.lower() for t in hyp]
    if not hyp:
        return 0.0
    logs = []
    for k in range(1, n + 1):
        h = Counter(tuple(hyp[i:i + k]) for i in range(len(hyp) - k + 1))
        r = Counter(tuple(ref[i:i + k]) for i in range(len(ref) - k + 1))
        tot = max(sum(h.values()), 1)
        logs.append(math.log((sum((h & r).values()) + 1) / (tot + 1)))
    bp = min(1.0, math.exp(1 - len(ref) / len(hyp)))
    return bp * math.exp(sum(logs) / n)


# --------------------------------------------------------------------------- measurement harness
CONTENT_TAGS = ("NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM")   # POS tag names (parser's), not words


def tree_token_indices(rec: dict) -> set:
    """Surface positions of every token that is a node of the record's tree."""
    toks, idx = rec["tokens"], set()
    for c in rec["lattice"]["trees"][0]["clauses"]:
        for r in c["roles"]:
            if r.get("token_index") is not None:
                idx.add(r["token_index"])
        pi = c.get("predicate_token_index")
        if pi is None and c.get("predicate"):
            pi = next((i for i, t in enumerate(toks) if t == c["predicate"] and i not in idx), None)
        if pi is not None:
            idx.add(pi)
    return idx


def strip_positions(tree: dict, seed: int = 0) -> dict:
    """What the decoder may see: drop token_index and shuffle role order (both would leak order)."""
    import random
    rng = random.Random(seed)
    out = {"clauses": []}
    for c in tree["clauses"]:
        roles = [{"relation": r["relation"], "word": r.get("word")} for r in c["roles"]]
        rng.shuffle(roles)
        out["clauses"].append({"predicate": c.get("predicate"), "roles": roles})
    return out


def _metrics(rec: dict, tokens: List[str], edges: Optional[Counter], gold: Counter) -> dict:
    orig, pos = rec["tokens"], rec["pos"]
    content = Counter(t.lower() for t, p in zip(orig, pos) if p in CONTENT_TAGS)
    got = Counter(t.lower() for t in tokens)
    p, r, f = edge_f1(gold, edges) if edges is not None else (0.0, 0.0, 0.0)
    return {"parsed": edges is not None, "f1": f, "exact": edges is not None and edges == gold,
            "content_recall": sum((content & got).values()) / max(sum(content.values()), 1),
            "bleu": bleu(orig, tokens), "n_out": len(tokens), "text": " ".join(tokens)}


def evaluate_record(rec: dict, variant: str, table: OrderTable, judge: Judge) -> dict:
    tree = rec["lattice"]["trees"][0]
    gold = tree_edges(tree)
    toks, pos = rec["tokens"], rec["pos"]
    posmap = {t: p for t, p in zip(toks, pos)}
    t0 = time.perf_counter()
    info = {}
    if variant == "orig":                       # control 0: parser self-consistency on the real sentence
        out = list(toks)
    elif variant == "upper":                    # control 1: delete non-tree tokens, keep original order
        keep = tree_token_indices(rec)
        out = [t for i, t in enumerate(toks) if i in keep]
    else:
        fill = default_slot_fill
        if variant == "decoder_oracle_prep":    # what-if: caller supplies the PP preposition (from the original)
            at = {(r["relation"], r["word"]): r["token_index"] for c in tree["clauses"] for r in c["roles"]
                  if r.get("token_index") is not None}

            def fill(slot, rel, tok, _at=at):   # noqa: E306
                i = _at.get((rel, tok))
                if slot != "PREP" or i is None:
                    return None
                for j in range(i - 1, max(i - 4, -1), -1):
                    if pos[j] == "ADP":
                        return toks[j]
                return None
        links = rec["lattice"]["discourse_links_per_tree"][0]
        coords = [lk.get("coordinator") for lk in sorted(links, key=lambda l: l["clause_i"])] if links else None
        d = decode(strip_positions(tree), table, judge, posmap, fill, coordinators=coords)
        out = d.tokens
        info = {"n_cand": d.n_candidates, "n_parsed_cand": d.n_parsed}
    edges = judge.parse_edges(out)
    m = _metrics(rec, out, edges, gold)
    m.update(info, seconds=time.perf_counter() - t0, n_tok=len(toks), n_clauses=len(tree["clauses"]),
             id=rec["text"], orig=rec["text"])
    return m


_W: dict = {}


def _worker_init():
    _W["table"] = derive_order_table(load_grammar())
    _W["judge"] = Judge()


def _worker(args):
    rec, variant = args
    try:
        return evaluate_record(rec, variant, _W["table"], _W["judge"])
    except Exception as e:  # keep the sweep alive
        return {"id": rec["text"], "error": repr(e), "parsed": False, "f1": 0.0, "exact": False}


def select_records(path: Path, n: int, max_tok: int, min_tok: int = 0, max_clauses: int = 2, seed: int = 0):
    import gzip
    import random
    recs = [json.loads(l) for l in gzip.open(path, "rt")]
    pool = [r for r in recs if min_tok < len(r["tokens"]) <= max_tok
            and len(r["lattice"]["trees"][0]["clauses"]) <= max_clauses]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def main(argv=None):
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="decoder", choices=["decoder", "decoder_oracle_prep", "upper", "orig"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--min-tok", type=int, default=0)
    ap.add_argument("--max-tok", type=int, default=12)
    ap.add_argument("--max-clauses", type=int, default=2)
    ap.add_argument("--skip", type=int, default=0, help="records already done (resume)")
    ap.add_argument("--budget", type=float, default=100.0, help="wall seconds before stopping")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = select_records(CT_DIR / "runs" / "encoder_gold_v6.jsonl.gz", a.n, a.max_tok, a.min_tok, a.max_clauses)[a.skip:]
    t0, done = time.time(), 0
    with mp.Pool(a.workers, initializer=_worker_init) as pool, open(a.out, "a") as f:
        for res in pool.imap(_worker, [(r, a.variant) for r in recs], chunksize=1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            done += 1
            if time.time() - t0 > a.budget:
                pool.terminate()
                break
    print(f"done {done}/{len(recs)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
