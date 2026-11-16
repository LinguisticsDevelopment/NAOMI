"""Learned encoder (stage A): text -> tree, as "for every word, its head and its relation".

The encoder's whole output is, for every non-punctuation token, one head (another token or ROOT)
and one structural relation label. Every token is a node by contract, so it cannot drop or invent
content. It never picks a word sense and never answers anything. Input features are language-neutral:
the tagger's part of speech, the tagger's structural subtypes (multi-hot), the USVS vector of the
token's candidate senses (mean of the top 3), a name flag; an optional hashed word identity
(off by default). No word list or pattern lives here.

Training signal is "unknown, not wrong": only tokens the parser really attached are supervised
(``supervision_from_tree``); tops of unconnected pieces and loose tokens carry no loss.

Index convention: positions are indices into the FULL token sequence (punctuation included). In score
matrices column 0 is the ROOT slot and column j+1 is token j.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .reader import _PUNCT_POS, Reader, make_node, normalize
from .tree import walk
from .types import KIND_NAME, REL_ROOT, REL_UNATTACHED, Node, Tree

NEG = -1.0e9
TOP_SENSES = 3
SOURCE_ENCODER = "encoder"


# ----------------------------------------------------------------------------- token info and features
@dataclass
class TokenInfo:
    token: str
    pos: str
    subtypes: List[str] = field(default_factory=list)
    kind: str = "word"
    senses: List[str] = field(default_factory=list)
    lemma: str = ""

    @property
    def is_node(self) -> bool:
        return self.pos != _PUNCT_POS


def infos_from_sentence(reader: Reader, sentence: str) -> List[TokenInfo]:
    """Tag a raw sentence and look up lemmas / candidate senses exactly like the parser path does."""
    toks, pos, subs = reader.tag_full(normalize(sentence))
    out = []
    for i, t in enumerate(toks):
        if pos[i] == _PUNCT_POS:
            out.append(TokenInfo(t, pos[i], list(subs[i]), "word", [], t))
        else:
            n = make_node(REL_UNATTACHED, t, i, pos[i], None, subs[i], reader.lang)
            out.append(TokenInfo(t, pos[i], list(subs[i]), n.kind, list(n.senses), n.lemma))
    return out


def infos_from_record(reader: Reader, rec: dict) -> Optional[List[TokenInfo]]:
    """TokenInfo from a gold record (senses / lemma / kind from its tree, subtypes from the tagger)."""
    toks, pos, subs = reader.tag_full(normalize(rec["text"]))
    if [t for t, _ in rec["tokens"]] != toks:
        return None
    by_idx = {n.token_index: n for n, _, _ in walk(_tree(rec).root) if n.token_index is not None}
    out = []
    for i, t in enumerate(toks):
        n = by_idx.get(i)
        if n is None:
            out.append(TokenInfo(t, pos[i], list(subs[i]), "word", [], t))
        else:
            out.append(TokenInfo(t, pos[i], list(subs[i]), n.kind, list(n.senses), n.lemma))
    return out


def _tree(rec: dict) -> Tree:
    from .tree import tree_from_dict
    return tree_from_dict(rec["tree"])


@dataclass
class Feats:
    pos: np.ndarray        # [T] int
    subs: List[List[int]]  # per token, subtype ids
    meaning: np.ndarray    # [T] int row into the builder's table (row 0 = zeros)
    name: np.ndarray       # [T] float 0/1
    node: np.ndarray       # [T] bool (non-punctuation)
    hashed: np.ndarray     # [T] int word-identity bucket (used only if the model has hashing on)


def hash_bucket(tok: str, buckets: int) -> int:
    return zlib.crc32(tok.encode("utf-8")) % buckets if buckets else 0


class FeatureBuilder:
    """Turns TokenInfo lists into Feats. Holds the vocabularies and the meaning-vector table."""

    def __init__(self, usvs, pos_vocab: Dict[str, int], sub_vocab: Dict[str, int], hash_buckets: int = 0):
        self.usvs, self.pos_vocab, self.sub_vocab, self.hash_buckets = usvs, pos_vocab, sub_vocab, hash_buckets
        self.d_axes = len(usvs.axes)
        self._rows: Dict[tuple, int] = {(): 0}
        self._vecs: List[np.ndarray] = [np.zeros(self.d_axes, dtype=np.float32)]
        self._table: Optional[np.ndarray] = None

    def meaning_row(self, senses: Sequence[str]) -> int:
        key = tuple(senses[:TOP_SENSES])
        r = self._rows.get(key)
        if r is None:
            vs = [v for v in (self.usvs.sense_dense(s) for s in key) if v is not None]
            if not vs:
                r = 0
            else:
                self._vecs.append(np.mean(vs, axis=0).astype(np.float32))
                r = len(self._vecs) - 1
            self._rows[key] = r
            self._table = None
        return r

    @property
    def table(self) -> np.ndarray:
        if self._table is None or len(self._table) != len(self._vecs):
            self._table = np.stack(self._vecs)
        return self._table

    def build(self, infos: Sequence[TokenInfo]) -> Feats:
        unk_p = self.pos_vocab.get(None, 0)
        return Feats(
            pos=np.array([self.pos_vocab.get(t.pos, unk_p) for t in infos], dtype=np.int64),
            subs=[[self.sub_vocab[s] for s in t.subtypes if s in self.sub_vocab] for t in infos],
            meaning=np.array([self.meaning_row(t.senses) if t.is_node and t.kind != KIND_NAME else 0 for t in infos],
                             dtype=np.int64),
            name=np.array([float(t.kind == KIND_NAME) for t in infos], dtype=np.float32),
            node=np.array([t.is_node for t in infos], dtype=bool),
            hashed=np.array([hash_bucket(t.token, self.hash_buckets) for t in infos], dtype=np.int64))


# ----------------------------------------------------------------------------- supervision ("unknown, not wrong")
@dataclass
class Supervision:
    head: List[int]       # per token: 0 = ROOT, j+1 = token j, -1 = no target
    rel: List[Optional[str]]
    sup: List[bool]       # True where the token is supervised (head and relation)
    unknown: List[bool]   # True for node tokens the parser did not attach (tops of pieces, loose tokens)
    connected: bool       # the parser produced one fully connected tree


def supervision_from_tree(tree: Tree, n_tokens: int) -> Supervision:
    head = [-1] * n_tokens
    rel: List[Optional[str]] = [None] * n_tokens
    sup = [False] * n_tokens
    unknown = [False] * n_tokens
    connected = tree.root.token_index is not None
    for n, _d, parent in walk(tree.root):
        i = n.token_index
        if i is None:
            continue
        if n.rel == REL_UNATTACHED:
            unknown[i] = True
            connected = False
        elif n.rel == REL_ROOT:                     # the main component's root (synthetic root's child or the root)
            head[i], rel[i], sup[i] = 0, REL_ROOT, True
        elif parent is not None and parent.token_index is not None:
            head[i], rel[i], sup[i] = parent.token_index + 1, n.rel, True
        else:                                       # no usable head: treat as unknown, never as a target
            unknown[i] = True
    if tree.root.token_index is None:
        connected = False
    return Supervision(head, rel, sup, unknown, connected)


# ----------------------------------------------------------------------------- the network
class EncoderNet(nn.Module):
    def __init__(self, n_pos: int, n_sub: int, n_rel: int, d_axes: int, d_in: int = 192, hidden: int = 128,
                 layers: int = 2, hash_buckets: int = 0, d_arc: int = 128, d_rel: int = 64, dropout: float = 0.3):
        super().__init__()
        self.hash_buckets = hash_buckets
        self.emb_pos = nn.Embedding(n_pos, 32)
        self.sub = nn.Linear(max(n_sub, 1), 32, bias=False)
        self.mean = nn.Linear(d_axes, 96)
        self.emb_hash = nn.Embedding(hash_buckets, 32) if hash_buckets else None
        d_cat = 32 + 32 + 96 + 1 + (32 if hash_buckets else 0)
        self.inp = nn.Linear(d_cat, d_in)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(d_in, hidden, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        h2 = 2 * hidden
        self.arc_dep = nn.Sequential(nn.Linear(h2, d_arc), nn.ReLU(), nn.Dropout(dropout))
        self.arc_head = nn.Sequential(nn.Linear(h2, d_arc), nn.ReLU(), nn.Dropout(dropout))
        self.root_arc = nn.Parameter(torch.zeros(d_arc))
        self.U = nn.Parameter(torch.empty(d_arc, d_arc))
        nn.init.xavier_uniform_(self.U)
        self.head_bias = nn.Linear(d_arc, 1)
        self.rel_dep = nn.Sequential(nn.Linear(h2, d_rel), nn.ReLU(), nn.Dropout(dropout))
        self.rel_head = nn.Sequential(nn.Linear(h2, d_rel), nn.ReLU(), nn.Dropout(dropout))
        self.root_rel = nn.Parameter(torch.zeros(d_rel))
        self.rel_out = nn.Sequential(nn.Linear(3 * d_rel, d_rel), nn.ReLU(), nn.Linear(d_rel, n_rel))

    def encode(self, pos, subs, meaning, name, hashed, lengths):
        x = [self.emb_pos(pos), self.sub(subs), self.mean(meaning), name.unsqueeze(-1)]
        if self.emb_hash is not None:
            x.append(self.emb_hash(hashed))
        x = self.drop(torch.relu(self.inp(torch.cat(x, -1))))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return out

    def arc_scores(self, H):
        """[B, T, T+1]: score of token i taking slot j (0 = ROOT, j+1 = token j) as its head."""
        d = self.arc_dep(H)
        h = self.arc_head(H)
        h = torch.cat([self.root_arc.expand(h.shape[0], 1, -1), h], 1)
        return torch.einsum("bid,de,bje->bij", d, self.U, h) + self.head_bias(h).transpose(1, 2)

    def rel_logits(self, H, b, dep, head):
        """Relation logits for (batch b, dependent position dep, head slot head) triples (1-D index tensors)."""
        d = self.rel_dep(H)[b, dep]
        hs = torch.cat([self.root_rel.expand(H.shape[0], 1, -1), self.rel_head(H)], 1)[b, head]
        return self.rel_out(torch.cat([d, hs, d * hs], -1))


def collate(items: Sequence[Feats], table: np.ndarray, n_sub: int):
    B, T = len(items), max(len(f.pos) for f in items)
    pos = np.zeros((B, T), dtype=np.int64)
    subs = np.zeros((B, T, max(n_sub, 1)), dtype=np.float32)
    mean = np.zeros((B, T, table.shape[1]), dtype=np.float32)
    name = np.zeros((B, T), dtype=np.float32)
    node = np.zeros((B, T), dtype=bool)
    hashed = np.zeros((B, T), dtype=np.int64)
    lens = np.zeros(B, dtype=np.int64)
    for b, f in enumerate(items):
        n = len(f.pos)
        pos[b, :n], name[b, :n], node[b, :n], hashed[b, :n], lens[b] = f.pos, f.name, f.node, f.hashed, n
        mean[b, :n] = table[f.meaning]
        for t, ids in enumerate(f.subs):
            subs[b, t, ids] = 1.0
    return {"pos": torch.from_numpy(pos), "subs": torch.from_numpy(subs), "meaning": torch.from_numpy(mean),
            "name": torch.from_numpy(name), "hashed": torch.from_numpy(hashed), "lengths": torch.from_numpy(lens),
            "node": torch.from_numpy(node)}


def head_mask(node: torch.Tensor) -> torch.Tensor:
    """[B, T, T+1] bool: True where slot j is an allowed head of token i (ROOT, or another node token)."""
    B, T = node.shape
    m = torch.zeros(B, T, T + 1, dtype=torch.bool)
    m[:, :, 0] = True
    m[:, :, 1:] = node.unsqueeze(1)
    m[:, torch.arange(T), torch.arange(T) + 1] = False
    return m


# ----------------------------------------------------------------------------- decoding (Chu-Liu/Edmonds)
def _find_cycle(heads: np.ndarray) -> Optional[List[int]]:
    n = len(heads)
    state = np.zeros(n, dtype=np.int8)
    for s in range(1, n):
        if state[s]:
            continue
        path, v = [], s
        while v != 0 and state[v] == 0:
            state[v] = 1
            path.append(v)
            v = int(heads[v])
        if v != 0 and state[v] == 1:
            return path[path.index(v):]
        for p in path:
            state[p] = 2
    return None


def edmonds(W: np.ndarray) -> np.ndarray:
    """Maximum spanning arborescence rooted at node 0. W[h, d] = score of head h for dependent d.
    Returns heads[d] (heads[0] = 0). Deterministic: ties resolve to the lowest index."""
    n = W.shape[0]
    W = W.copy()
    W[:, 0] = NEG
    np.fill_diagonal(W, NEG)
    best = W.argmax(0)
    best[0] = 0
    cyc = _find_cycle(best)
    if cyc is None:
        return best
    in_c = np.zeros(n, dtype=bool)
    in_c[cyc] = True
    others = [v for v in range(n) if not in_c[v]]
    m = len(others)
    c = m                                           # index of the contracted node
    W2 = np.full((m + 1, m + 1), NEG)
    W2[:m, :m] = W[np.ix_(others, others)]
    enter = np.zeros(m, dtype=np.int64)
    leave = np.zeros(m, dtype=np.int64)
    cyc_arr = np.array(cyc)
    base = W[best[cyc_arr], cyc_arr]
    for k, o in enumerate(others):
        gain = W[o, cyc_arr] - base
        a = int(gain.argmax())
        W2[k, c], enter[k] = gain[a], cyc_arr[a]
        lv = W[cyc_arr, o]
        a = int(lv.argmax())
        W2[c, k], leave[k] = lv[a], cyc_arr[a]
    sub = edmonds(W2)
    heads = best.copy()
    for k, o in enumerate(others):
        if k == 0:
            continue
        h2 = int(sub[k])
        heads[o] = leave[k] if h2 == c else others[h2]
    ho = int(sub[c])
    heads[enter[ho]] = others[ho]
    return heads


def tree_score(W: np.ndarray, heads: np.ndarray) -> float:
    return float(sum(W[heads[d], d] for d in range(1, len(heads))))


def decode_heads(W: np.ndarray) -> np.ndarray:
    """Single-root maximum arborescence. W is (n+1)x(n+1), node 0 = ROOT. Returns heads (heads[0] = 0).
    If the unconstrained solution has several ROOT children, each is tried as the only root."""
    n = W.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int64)
    heads = edmonds(W)
    roots = [d for d in range(1, n) if heads[d] == 0]
    if len(roots) <= 1:
        return heads
    best, best_s = None, None
    for r in roots:
        W2 = W.copy()
        W2[0, :] = NEG
        W2[0, r] = W[0, r]
        h = edmonds(W2)
        s = tree_score(W2, h)
        if best is None or s > best_s:
            best, best_s = h, s
    return best


# ----------------------------------------------------------------------------- tree construction
def build_tree(sentence: str, infos: Sequence[TokenInfo], heads: Dict[int, int], rels: Dict[int, str],
               source: str = SOURCE_ENCODER) -> Tree:
    """heads/rels keyed by token position; head 0 = ROOT else token position + 1. Exactly one root."""
    nodes = {i: Node(rel=rels[i], token=infos[i].token, token_index=i, lemma=infos[i].lemma or infos[i].token,
                     pos=infos[i].pos, kind=infos[i].kind, senses=list(infos[i].senses),
                     subtypes=list(infos[i].subtypes)) for i in heads}
    root = None
    for i, h in heads.items():
        if h == 0:
            nodes[i].rel = REL_ROOT
            root = nodes[i]
        else:
            nodes[h - 1].children.append(nodes[i])
    for n in nodes.values():
        n.children.sort(key=lambda c: c.token_index)
    if root is None:
        raise ValueError("no root")
    return Tree(sentence=sentence, root=root, source=source, diagnostics=[])


@dataclass
class Proposal:
    tree: Tree
    score: float
    heads: Dict[int, int]
    rels: Dict[int, str]


# ----------------------------------------------------------------------------- the encoder
class NaomiEncoder:
    """Checkpoint-backed encoder. ``usvs`` is the extended 614-axis space (``naomi.reader._sense_space()``)."""

    def __init__(self, net: EncoderNet, rel_labels: List[str], builder: FeatureBuilder, config: dict):
        self.net, self.rel_labels, self.builder, self.config = net.eval(), rel_labels, builder, config
        self.n_sub = len(builder.sub_vocab)
        self._root_rel = rel_labels.index(REL_ROOT)
        self._reader: Optional[Reader] = None

    @classmethod
    def load(cls, path: str, usvs=None) -> "NaomiEncoder":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        if usvs is None:
            from .reader import _sense_space
            usvs = _sense_space()
        net = EncoderNet(len(ck["pos_vocab"]), len(ck["sub_vocab"]), len(ck["rel_labels"]), len(usvs.axes), **cfg)
        net.load_state_dict(ck["state"])
        return cls(net, ck["rel_labels"], FeatureBuilder(usvs, ck["pos_vocab"], ck["sub_vocab"],
                                                         cfg.get("hash_buckets", 0)), cfg)

    # -- scores ---------------------------------------------------------------
    @torch.no_grad()
    def arc_logprobs(self, infos: Sequence[TokenInfo]):
        """-> (H, logp [T, T+1] numpy with disallowed slots at NEG, feats). Rows of punctuation are meaningless."""
        f = self.builder.build(infos)
        b = collate([f], self.builder.table, self.n_sub)
        H = self.net.encode(b["pos"], b["subs"], b["meaning"], b["name"], b["hashed"], b["lengths"])
        s = self.net.arc_scores(H)
        s = s.masked_fill(~head_mask(b["node"]), NEG)
        return H, F.log_softmax(s, -1)[0].numpy(), f

    @torch.no_grad()
    def _rels(self, H, pairs: List[Tuple[int, int]]):
        """pairs (dep position, head slot) -> (best non-ROOT label ids or ROOT for slot 0, log-prob of that)."""
        if not pairs:
            return [], []
        dep = torch.tensor([p[0] for p in pairs])
        hd = torch.tensor([p[1] for p in pairs])
        lg = self.net.rel_logits(H, torch.zeros_like(dep), dep, hd)
        lp = F.log_softmax(lg, -1)
        roots = hd == 0
        lg2 = lg.clone()
        lg2[:, self._root_rel] = NEG
        lg2[roots] = NEG
        lg2[roots, self._root_rel] = 0.0
        ids = lg2.argmax(-1)
        return ids.tolist(), lp[torch.arange(len(pairs)), ids].tolist()

    # -- decoding -------------------------------------------------------------
    def _decode(self, infos, H, logp, forced: Optional[Dict[int, int]] = None, root_only: Optional[int] = None):
        nodes = [i for i, t in enumerate(infos) if t.is_node]
        W = np.full((len(nodes) + 1, len(nodes) + 1), NEG)
        slot = {i: k + 1 for k, i in enumerate(nodes)}
        for d_i, i in enumerate(nodes):
            W[0, d_i + 1] = logp[i, 0]
            for j in nodes:
                W[slot[j], d_i + 1] = logp[i, j + 1]
        for i, col in (forced or {}).items():     # col in head convention: 0 = ROOT, j+1 = token j
            k, hslot = slot[i], (0 if col == 0 else slot[col - 1])
            keep = W[hslot, k]
            W[:, k] = NEG
            W[hslot, k] = keep
        if root_only is not None:
            r = slot[root_only]
            W[0, :] = NEG
            W[0, r] = logp[root_only, 0]
        h = decode_heads(W)
        heads = {i: (0 if h[slot[i]] == 0 else nodes[h[slot[i]] - 1] + 1) for i in nodes}
        return nodes, W, h, heads

    def _finish(self, sentence, infos, H, heads, headscore) -> Proposal:
        order = sorted(heads)
        ids, lps = self._rels(H, [(i, heads[i]) for i in order])
        rels = {i: self.rel_labels[k] for i, k in zip(order, ids)}
        return Proposal(build_tree(sentence, infos, heads, rels), headscore + float(sum(lps)), heads, rels)

    def decode_infos(self, sentence: str, infos: Sequence[TokenInfo]) -> Tuple[Proposal, np.ndarray]:
        H, logp, _ = self.arc_logprobs(infos)
        nodes, W, h, heads = self._decode(infos, H, logp)
        return self._finish(sentence, infos, H, heads, tree_score(W, h)), logp

    def propose_infos(self, sentence: str, infos: Sequence[TokenInfo], k: int = 4) -> List[Proposal]:
        """Up to k distinct complete single-root trees, best first: the best tree plus variants that
        force the next-best head for the least confident tokens, or re-root at the next-best roots."""
        H, logp, _ = self.arc_logprobs(infos)
        nodes, W, h, heads = self._decode(infos, H, logp)
        seen = {tuple(sorted(heads.items())): self._finish(sentence, infos, H, heads, tree_score(W, h))}
        if k > 1 and len(nodes) > 1:
            margin = []
            for i in nodes:
                row = np.sort(logp[i][logp[i] > NEG / 2])[::-1]
                margin.append((row[0] - row[1] if len(row) > 1 else np.inf, i))
            margin.sort()
            variants: List[Tuple[Optional[Dict[int, int]], Optional[int]]] = []
            for _m, i in margin[:2 * k + 2]:
                order = np.argsort(-logp[i], kind="stable")
                cur = heads[i]
                alt = next((int(j) for j in order if j != cur and logp[i, j] > NEG / 2), None)
                if alt is not None:
                    variants.append(({i: alt}, None))
            roots = sorted((i for i in nodes if heads[i] != 0), key=lambda i: -logp[i, 0])
            variants += [(None, i) for i in roots[:k]]
            for forced, ro in variants:
                _n, W2, h2, hd2 = self._decode(infos, H, logp, forced, ro)
                key = tuple(sorted(hd2.items()))
                if key not in seen:
                    seen[key] = self._finish(sentence, infos, H, hd2, tree_score(W2, h2))
        return sorted(seen.values(), key=lambda p: -p.score)[:k]

    # -- public ---------------------------------------------------------------
    def _infos(self, sentence: str) -> List[TokenInfo]:
        if self._reader is None:
            self._reader = Reader(use_encoder=False, cache_path=None)
        return infos_from_sentence(self._reader, sentence)

    def encode(self, sentence: str) -> Tree:
        return self.decode_infos(sentence, self._infos(sentence))[0].tree

    def propose(self, sentence: str, k: int = 4) -> List[Proposal]:
        return self.propose_infos(sentence, self._infos(sentence), k)
