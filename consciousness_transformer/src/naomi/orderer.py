"""Learned decoder v1: word ORDER from a tree (the tree's nodes carry their own words).

Input: nodes with the encoder's per-token features (tagger POS, structural subtypes, USVS meaning vector,
name flag) plus the relation to their head. NO position goes in: the nodes are put in a deterministic
shuffled order and the model is permutation-invariant (information passing along tree edges by pooling,
a pairwise "does a come before b" scorer over {head} + children, antisymmetric by construction).
Output: the linear order, built recursively so every subtree is contiguous. A tie in the scorer is broken by
the surface token string only (a canonical, language-neutral tie-break). No word list lives here.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .encoder import FeatureBuilder, Feats, TokenInfo
from .tree import walk
from .types import REL_ROOT, Node, Tree

UNK_REL = "<unk>"


# ----------------------------------------------------------------------------- tree input (no positions)
@dataclass
class TreeIn:
    """One tree, positions removed. ``true_pos`` and ``nodes`` are bookkeeping for training / evaluation
    and are never read by the network."""

    feats: Feats
    rel: np.ndarray                   # [N] relation id
    parent: np.ndarray                # [N] -1 for the root
    tokens: List[str]
    virtual: np.ndarray               # [N] bool: synthetic anchor (no token), never emitted
    groups: List[Tuple[int, List[int], int]]   # (head, items = [head] + children, offset into the pair arrays)
    pa: np.ndarray                    # pair first item  (node index)
    pb: np.ndarray                    # pair second item
    ph: np.ndarray                    # pair group head
    true_pos: Optional[np.ndarray] = None
    nodes: Optional[List[Node]] = None

    @property
    def n(self) -> int:
        return len(self.tokens)

    @property
    def n_tokens(self) -> int:
        return int((~self.virtual).sum())


def make_tree_in(infos: Sequence[TokenInfo], rels: Sequence[str], parent: Sequence[int], builder: FeatureBuilder,
                 rel_ids: Dict[str, int], true_pos: Optional[Sequence[int]] = None, virtual: Optional[Sequence[bool]] = None,
                 nodes: Optional[List[Node]] = None, seed: int = 0) -> TreeIn:
    """Array-level constructor: nodes are re-indexed by a seeded shuffle so that input order carries nothing."""
    n = len(infos)
    perm = list(range(n))
    random.Random(seed).shuffle(perm)             # new index k holds old node perm[k]
    inv = {old: k for k, old in enumerate(perm)}
    infos_s = [infos[o] for o in perm]
    par = np.array([(-1 if parent[o] < 0 else inv[parent[o]]) for o in perm], dtype=np.int64)
    rel = np.array([rel_ids.get(rels[o], rel_ids[UNK_REL]) for o in perm], dtype=np.int64)
    virt = np.array([bool(virtual[o]) if virtual is not None else False for o in perm], dtype=bool)
    tp = None if true_pos is None else np.array([true_pos[o] for o in perm], dtype=np.int64)
    feats = builder.build(infos_s)
    kids: Dict[int, List[int]] = {}
    for i, p in enumerate(par):
        if p >= 0:
            kids.setdefault(int(p), []).append(i)
    groups, pa, pb, ph, off = [], [], [], [], 0
    for h in sorted(kids):
        items = [h] + sorted(kids[h])
        groups.append((h, items, off))
        for a, b in itertools.combinations(items, 2):
            pa.append(a), pb.append(b), ph.append(h)
        off += len(items) * (len(items) - 1) // 2
    return TreeIn(feats, rel, par, [t.token for t in infos_s], virt, groups, np.array(pa, dtype=np.int64),
                  np.array(pb, dtype=np.int64), np.array(ph, dtype=np.int64), tp,
                  None if nodes is None else [nodes[o] for o in perm])


def tree_in(tree: Tree, builder: FeatureBuilder, rel_ids: Dict[str, int], seed: int = 0) -> TreeIn:
    """From a Node tree. ``token_index`` is kept only as ``true_pos`` bookkeeping (never fed to the model)."""
    ns, parent, idx = [], [], {}
    for n, _d, p in walk(tree.root):
        idx[id(n)] = len(ns)
        ns.append(n)
        parent.append(-1 if p is None else idx[id(p)])
    infos = [TokenInfo(n.token, n.pos, list(n.subtypes), n.kind, list(n.senses), n.lemma) for n in ns]
    virtual = [n.token_index is None for n in ns]
    tp = [(-1 if n.token_index is None else n.token_index) for n in ns]
    return make_tree_in(infos, [n.rel for n in ns], parent, builder, rel_ids, tp, virtual, ns, seed)


# ----------------------------------------------------------------------------- batching
def collate_trees(tins: Sequence[TreeIn], table: np.ndarray, n_sub: int) -> dict:
    N = sum(t.n for t in tins)
    pos = np.concatenate([t.feats.pos for t in tins])
    name = np.concatenate([t.feats.name for t in tins])
    meaning = np.concatenate([t.feats.meaning for t in tins])
    rel = np.concatenate([t.rel for t in tins])
    subs = np.zeros((N, max(n_sub, 1)), dtype=np.float32)
    parent, pa, pb, ph = [], [], [], []
    o = 0
    for t in tins:
        for k, ids in enumerate(t.feats.subs):
            subs[o + k, ids] = 1.0
        parent.append(np.where(t.parent >= 0, t.parent + o, -1))
        pa.append(t.pa + o), pb.append(t.pb + o), ph.append(t.ph + o)
        o += t.n
    return {"pos": torch.from_numpy(pos), "subs": torch.from_numpy(subs),
            "meaning": torch.from_numpy(table[meaning]), "name": torch.from_numpy(name),
            "rel": torch.from_numpy(rel), "parent": torch.from_numpy(np.concatenate(parent)),
            "pa": torch.from_numpy(np.concatenate(pa)), "pb": torch.from_numpy(np.concatenate(pb)),
            "ph": torch.from_numpy(np.concatenate(ph))}


# ----------------------------------------------------------------------------- the network
class OrdererNet(nn.Module):
    def __init__(self, n_pos: int, n_sub: int, n_rel: int, d_axes: int, d: int = 128, rounds: int = 2,
                 d_pair: int = 192, dropout: float = 0.2):
        super().__init__()
        self.emb_pos = nn.Embedding(n_pos, 32)
        self.sub = nn.Linear(max(n_sub, 1), 32, bias=False)
        self.mean = nn.Linear(d_axes, 96)
        self.emb_rel = nn.Embedding(n_rel, 32)
        self.inp = nn.Linear(32 + 32 + 96 + 32 + 1, d)
        self.mp = nn.ModuleList([nn.Linear(3 * d + 1, d) for _ in range(rounds)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(rounds)])
        self.drop = nn.Dropout(dropout)
        self.pair = nn.Sequential(nn.Linear(3 * d + 2, d_pair), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(d_pair, d_pair), nn.ReLU(), nn.Linear(d_pair, 1))

    def node_states(self, b: dict) -> torch.Tensor:
        x = torch.cat([self.emb_pos(b["pos"]), self.sub(b["subs"]), self.mean(b["meaning"]),
                       self.emb_rel(b["rel"]), b["name"].unsqueeze(-1)], -1)
        h = self.drop(torch.relu(self.inp(x)))
        par = b["parent"]
        has = par >= 0
        N = h.shape[0]
        cnt = torch.zeros(N).index_add_(0, par[has], torch.ones(int(has.sum())))
        cf = torch.log1p(cnt).unsqueeze(-1)
        for lin, ln in zip(self.mp, self.norm):
            agg = torch.zeros_like(h).index_add_(0, par[has], h[has]) / cnt.clamp(min=1).unsqueeze(-1)
            hp = torch.where(has.unsqueeze(-1), h[par.clamp(min=0)], torch.zeros_like(h))
            h = ln(h + self.drop(torch.relu(lin(torch.cat([h, agg, hp, cf], -1)))))
        return h

    def _g(self, h, a, b, head):
        ia = torch.cat([h[a], (a == head).float().unsqueeze(-1)], -1)
        ib = torch.cat([h[b], (b == head).float().unsqueeze(-1)], -1)
        return self.pair(torch.cat([ia, ib, h[head]], -1)).squeeze(-1)

    def forward(self, b: dict) -> torch.Tensor:
        """-> s [P]: antisymmetric logit that pair item ``pa`` comes before ``pb``."""
        h = self.node_states(b)
        a, c, head = b["pa"], b["pb"], b["ph"]
        return self._g(h, a, c, head) - self._g(h, c, a, head)


# ----------------------------------------------------------------------------- ordering from per-group orders
def emit(tin: TreeIn, group_order: Dict[int, List[int]]) -> List[int]:
    """Recursive emission: each head's sorted items; the head item emits the head token, a child item emits its
    whole subtree. Every subtree is contiguous by construction. Returns node indices (virtual anchors dropped)."""
    out: List[int] = []
    roots = [i for i in range(tin.n) if tin.parent[i] < 0]
    stack = [("n", r) for r in reversed(roots)]
    while stack:
        kind, i = stack.pop()
        if kind == "t":
            if not tin.virtual[i]:
                out.append(i)
            continue
        items = group_order.get(i)
        if items is None:
            stack.append(("t", i))
            continue
        for it in reversed(items):
            stack.append(("t", i) if it == i else ("n", it))
    return out


def sort_groups_by_scores(tin: TreeIn, s: np.ndarray) -> Dict[int, List[int]]:
    """Deterministic: wins (number of items the item is scored before), ties by total score, then token string."""
    res = {}
    for h, items, off in tin.groups:
        k = len(items)
        iu = np.triu_indices(k, 1)
        S = np.zeros((k, k))
        S[iu] = s[off: off + len(iu[0])]
        S = S - S.T
        wins = (S > 0).sum(1)
        tot = np.round(S.sum(1), 4)
        order = sorted(range(k), key=lambda j: (-int(wins[j]), -float(tot[j]), tin.tokens[items[j]]))
        res[h] = [items[j] for j in order]
    return res


def true_group_order(tin: TreeIn) -> Dict[int, List[int]]:
    return {h: sorted(items, key=lambda i: tin.true_pos[i]) for h, items, _ in tin.groups}


def true_order(tin: TreeIn) -> List[int]:
    idx = [i for i in range(tin.n) if not tin.virtual[i]]
    return sorted(idx, key=lambda i: tin.true_pos[i])


def contiguous(tin: TreeIn) -> bool:
    """CEILING test: every subtree occupies a contiguous block of the (true) order, so the block-wise
    formulation can reproduce it (non-projective trees cannot be reproduced)."""
    real = [i for i in range(tin.n) if not tin.virtual[i]]
    rank = {i: r for r, i in enumerate(sorted(real, key=lambda i: tin.true_pos[i]))}
    kids: Dict[int, List[int]] = {}
    for i, p in enumerate(tin.parent):
        if p >= 0:
            kids.setdefault(int(p), []).append(i)
    ok = True
    order, st = [], [i for i in range(tin.n) if tin.parent[i] < 0]
    while st:
        i = st.pop()
        order.append(i)
        st.extend(kids.get(i, []))
    lo, hi, sz = {}, {}, {}
    for i in reversed(order):
        rs = [] if tin.virtual[i] else [rank[i]]
        l = min(rs, default=10 ** 9)
        h = max(rs, default=-1)
        s = len(rs)
        for c in kids.get(i, []):
            l, h, s = min(l, lo[c]), max(h, hi[c]), s + sz[c]
        lo[i], hi[i], sz[i] = l, h, s
        if s and not tin.virtual[i] and h - l + 1 != s:
            ok = False
    return ok


# ----------------------------------------------------------------------------- metrics
def kendall_distance(order_pos: Sequence[int]) -> float:
    """Normalised Kendall tau distance of a predicted order given as the true positions in predicted order."""
    a = np.asarray(order_pos)
    n = len(a)
    if n < 2:
        return 0.0
    lt = a[:, None] > a[None, :]
    return float(np.triu(lt, 1).sum()) / (n * (n - 1) / 2)


def band_of(n: int) -> str:
    return "<=10" if n <= 10 else "11-20" if n <= 20 else "21-35" if n <= 35 else "36+"


class Metrics:
    def __init__(self):
        self.sent = self.exact = self.heads = self.heads_ok = self.pairs = self.pairs_ok = 0
        self.kt: List[float] = []

    def add(self, tin: TreeIn, group_order: Dict[int, List[int]], pred: Optional[List[int]] = None):
        truth = true_group_order(tin)
        for h, items, _ in tin.groups:
            self.heads += 1
            self.heads_ok += int(group_order[h] == truth[h])
            pos = {it: r for r, it in enumerate(group_order[h])}
            for a, b in itertools.combinations(items, 2):
                self.pairs += 1
                self.pairs_ok += int((pos[a] < pos[b]) == (tin.true_pos[a] < tin.true_pos[b]))
        if pred is None:
            pred = emit(tin, group_order)
        self.sent += 1
        self.exact += int(pred == true_order(tin))
        self.kt.append(kendall_distance([int(tin.true_pos[i]) for i in pred]))

    def out(self) -> dict:
        d = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
        return {"sentences": self.sent, "exact_order": d(self.exact, self.sent), "heads": self.heads,
                "heads_fully_right": d(self.heads_ok, self.heads), "pairwise_acc": d(self.pairs_ok, self.pairs),
                "kendall_distance": round(float(np.mean(self.kt)), 4) if self.kt else None}


def evaluate(tins: Sequence[TreeIn], order_fn: Callable[[Sequence[TreeIn]], List[Dict[int, List[int]]]],
             by_band: bool = True) -> dict:
    allm, bands = Metrics(), {}
    gos = order_fn(tins)
    for t, go in zip(tins, gos):
        allm.add(t, go)
        if by_band:
            bands.setdefault(band_of(t.n_tokens), Metrics()).add(t, go)
    out = {"all": allm.out()}
    if by_band:
        out["by_band"] = {k: bands[k].out() for k in ("<=10", "11-20", "21-35", "36+") if k in bands}
    return out


# ----------------------------------------------------------------------------- cheat baseline (lookup table)
class LookupBaseline:
    """Per (relation, dependent tag, head tag): the majority side of the head and the mean distance rank on that
    side, learned from training groups; backoffs (relation, dependent tag) -> relation -> global. No model."""

    def __init__(self, pos_names: Sequence[str], rel_names: Sequence[str]):
        self.pos_names, self.rel_names = pos_names, rel_names
        self.stat: Dict[tuple, List[list]] = {}

    def _keys(self, tin: TreeIn, i: int, h: int):
        r, d, hp = int(tin.rel[i]), int(tin.feats.pos[i]), int(tin.feats.pos[h])
        return [(r, d, hp), (r, d), (r,), ()]

    def fit(self, tins: Sequence[TreeIn]):
        acc: Dict[tuple, list] = {}
        for t in tins:
            for h, items, _ in t.groups:
                srt = sorted(items, key=lambda i: t.true_pos[i])
                hi = srt.index(h)
                for j, i in enumerate(srt):
                    if i == h:
                        continue
                    for k in self._keys(t, i, h):
                        a = acc.setdefault(k, [0, 0, 0.0, 0.0])      # n_before, n_after, sum_rank_before, sum_rank_after
                        if j < hi:
                            a[0] += 1
                            a[2] += hi - j
                        else:
                            a[1] += 1
                            a[3] += j - hi
        self.stat = {k: (0 if a[0] > a[1] else 1, (a[2] / a[0]) if a[0] > a[1] else (a[3] / max(a[1], 1)))
                     for k, a in acc.items()}

    def group_order(self, t: TreeIn) -> Dict[int, List[int]]:
        res = {}
        for h, items, _ in t.groups:
            before, after = [], []
            for i in items:
                if i == h:
                    continue
                side, rank = next(self.stat[k] for k in self._keys(t, i, h) if k in self.stat)
                (before if side == 0 else after).append((rank, t.tokens[i], i))
            before.sort(key=lambda x: (-x[0], x[1]))
            after.sort(key=lambda x: (x[0], x[1]))
            res[h] = [i for _, _, i in before] + [h] + [i for _, _, i in after]
        return res


# ----------------------------------------------------------------------------- the orderer (inference)
class Orderer:
    def __init__(self, net: OrdererNet, builder: FeatureBuilder, rel_labels: List[str], config: dict):
        self.net, self.builder, self.rel_labels, self.config = net.eval(), builder, rel_labels, config
        self.rel_ids = {r: i for i, r in enumerate(rel_labels)}
        self.n_sub = len(builder.sub_vocab)

    @classmethod
    def load(cls, path: str, usvs=None) -> "Orderer":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if usvs is None:
            from .reader import _sense_space
            usvs = _sense_space()
        net = OrdererNet(len(ck["pos_vocab"]), len(ck["sub_vocab"]), len(ck["rel_labels"]), len(usvs.axes),
                         **ck["config"])
        net.load_state_dict(ck["state"])
        return cls(net, FeatureBuilder(usvs, ck["pos_vocab"], ck["sub_vocab"], 0), ck["rel_labels"], ck["config"])

    @torch.no_grad()
    def scores(self, tins: Sequence[TreeIn], chunk: int = 4000) -> List[np.ndarray]:
        out, cur, n = [], [], 0
        for t in list(tins) + [None]:
            if t is None or (cur and n + t.n > chunk):
                if cur:
                    s = self.net(collate_trees(cur, self.builder.table, self.n_sub)).numpy()
                    o = 0
                    for c in cur:
                        k = len(c.pa)
                        out.append(s[o:o + k])
                        o += k
                cur, n = [], 0
            if t is not None:
                cur.append(t)
                n += t.n
        return out

    def group_orders(self, tins: Sequence[TreeIn]) -> List[Dict[int, List[int]]]:
        return [sort_groups_by_scores(t, s) for t, s in zip(tins, self.scores(tins))]

    def order_inputs(self, tins: Sequence[TreeIn]) -> List[List[int]]:
        return [emit(t, g) for t, g in zip(tins, self.group_orders(tins))]

    def order_tree(self, tree: Tree, seed: int = 0) -> List[Node]:
        """The tree's real nodes in the predicted word order."""
        t = tree_in(tree, self.builder, self.rel_ids, seed)
        return [t.nodes[i] for i in self.order_inputs([t])[0]]

    def order_trees(self, trees: Sequence[Tree], seed: int = 0) -> List[List[Node]]:
        ts = [tree_in(tr, self.builder, self.rel_ids, seed) for tr in trees]
        return [[t.nodes[i] for i in o] for t, o in zip(ts, self.order_inputs(ts))]
