"""Mind v1: the one-step learned answering policy (consciousness v1, CONSCIOUSNESS_V1_PROPOSAL.md).

FIND proposes candidate memory nodes (tensor similarity to the question, plus one tree link), a small
learned policy scores each candidate (and ABSTAIN) from language-neutral numeric features, and the
answer is a memory node with its subtree. No word identity reaches the policy; weights hold how to
look things up, never a fact. Answering is learned from the answer alone.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------- data
def load_items(mc_path, ft_path) -> List[dict]:
    """Multiple-choice items of both datasets with a by-story split: MCTest dev = test, a validation set
    carved from MCTest train stories (every fifth story by a stable hash); FairytaleQA its own splits
    (val -> val). Only entity items with options are scorable."""
    out = []
    for path, ds in ((mc_path, "mctest"), (ft_path, "fairytaleqa")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("options") or r.get("answer_idx") is None:
                continue
            m = r["meta"]
            sp = m["split"]
            if ds == "mctest":
                sp = "test" if sp == "dev" else ("val" if zlib.crc32(m["story_id"].encode()) % 5 == 0 else "train")
            else:
                sp = {"val": "val", "test": "test"}.get(sp, "train")
            out.append({"id": f"{ds}:{m['story_id']}:{len(out)}", "dataset": ds, "story": m["story_id"],
                        "split": sp, "context": list(r["context"]), "question": r["question"],
                        "options": list(r["options"]), "answer_idx": int(r["answer_idx"])})
    return out


# ----------------------------------------------------------------------------- shared resources
UNKNOWN_AXIS = "UNKNOWN"          # the lexicon's own dimension for "wh" meaning (a USVS axis name, not a word)
SEED_K = 30                        # FIND budget (top memory nodes, split over the question's content nodes)
SIM_HARD = 0.8                     # COMPARE similarity counted as "the same meaning"
CORRECT_THRESHOLD = 20.0           # meaning% (tree scorer) a candidate subtree needs vs the correct option (the scorer folds determiners into the noun, so a right bare noun phrase scores ~35-50)
EXTRA_LABELS = ("ROOT", "UNATTACHED")


class Ctx:
    """USVS access shared by FIND/COMPARE/labelling. One per process."""
    _inst = None

    @classmethod
    def get(cls) -> "Ctx":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        from functools import lru_cache
        from . import score
        from .reader import FN_PREFIX, _sense_space
        if not getattr(score._wn_synset, "cache_info", None):      # the scorer re-reads WordNet per pair: cache it
            score._wn_synset = lru_cache(maxsize=None)(score._wn_synset)
        self.score = score
        self.usvs = _sense_space()
        self.fn_prefix = FN_PREFIX
        self.D = len(self.usvs.axes)
        self.sw = np.sqrt(score.axis_weights(self.usvs)).astype(np.float32)
        self.unit = float(self.sw.max())
        self.unknown_idx = list(self.usvs.axes).index(UNKNOWN_AXIS) if UNKNOWN_AXIS in list(self.usvs.axes) else -1
        self._raw: Dict[str, Optional[np.ndarray]] = {}

    def raw(self, sid: Optional[str]) -> Optional[np.ndarray]:
        if not sid:
            return None
        if sid not in self._raw:
            try:
                v = self.usvs.sense_dense(sid)
            except Exception:
                v = None
            self._raw[sid] = None if v is None else np.asarray(v, dtype=np.float32)
        return self._raw[sid]

    @staticmethod
    def chosen(n) -> Optional[str]:
        return n.sense or (n.senses[0] if n.senses else None)

    def is_function(self, n) -> bool:
        s = self.chosen(n)
        return bool(s) and s.startswith(self.fn_prefix)

    def key(self, n) -> Optional[str]:
        """Unit-dimension key of a node with no meaning point (a name, or a lemma the lexicon lacks)."""
        if n.kind == "name":
            return "n|" + n.token
        return None if self.raw(self.chosen(n)) is not None else "w|" + n.lemma


# ----------------------------------------------------------------------------- node table
@dataclass
class NodeRef:
    gid: int
    sent: int
    nid: int
    node: object
    parent: int
    rel: str
    depth: int
    children: List[int] = field(default_factory=list)


def node_table(memory, layers=None) -> Tuple[List[NodeRef], List[int]]:
    """Every real token node of every stored tree (insertion order = sentence order)."""
    from .tree import walk
    refs: List[NodeRef] = []
    sent_len: List[int] = []
    for si, st in enumerate(sorted((t for t in memory.trees() if layers is None or t.layer in layers), key=lambda s: s.nid)):
        idx: Dict[int, int] = {}
        n0 = len(refs)
        for n, d, p in walk(st.tree.root):
            if n.token_index is None:
                continue
            anc = p
            while anc is not None and anc.token_index is None:
                anc = None
            g = len(refs)
            idx[id(n)] = g
            par = idx.get(id(anc), -1) if anc is not None else -1
            refs.append(NodeRef(g, si, st.nid, n, par, n.rel, d))
            if par >= 0:
                refs[par].children.append(g)
        sent_len.append(len(refs) - n0)
    return refs, sent_len


def _sub_matrix(memory, node, cache) -> np.ndarray:
    """TPR flattening of the subtree rooted at ``node`` over RELATIVE paths (the node itself at the empty path),
    the memory's own scheme (``Memory._matrix``): sum of (1/depth) * bind(role(path), filler(lemma))."""
    m = np.zeros((memory.dim, memory.dim), dtype=np.float32)
    stack = [(node, "", 1)]
    while stack:
        n, path, depth = stack.pop()
        if n.token_index is not None:
            r = cache.get(("r", path))
            if r is None:
                r = cache[("r", path)] = memory.codec.role_vec(zlib.crc32(path.encode()), path)
            f = cache.get(("f", n.lemma))
            if f is None:
                f = cache[("f", n.lemma)] = memory.codec.filler_vec("w:" + n.lemma)
            m += (1.0 / depth) * memory.codec.bind(r, f)
        for c in n.children:
            stack.append((c, (path + "/" + c.rel) if path else c.rel, depth + 1))
    return m.reshape(-1)


def _unit_rows(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-8)


def sim_matrix(ctx: Ctx, A: Sequence, B: Sequence) -> np.ndarray:
    """COMPARE at node level: cosine of IDF-scaled USVS vectors; 1 for the same unit key (name / unlexiconed lemma)."""
    def mats(X):
        V = np.zeros((len(X), ctx.D), dtype=np.float32)
        keys = []
        for i, n in enumerate(X):
            k = ctx.key(n)
            keys.append(k)
            if k is None:
                V[i] = ctx.raw(ctx.chosen(n)) * ctx.sw
        return V, keys
    Va, ka = mats(A)
    Vb, kb = mats(B)
    S = _unit_rows(Va) @ _unit_rows(Vb).T
    for i, x in enumerate(ka):
        if x is not None:
            for j, y in enumerate(kb):
                S[i, j] = 1.0 if x == y else 0.0
    return np.clip(S, 0.0, 1.0)


def subtree_nodes(n) -> list:
    from .tree import walk
    return [m for m, _d, _p in walk(n) if m.token_index is not None]


def subtree_tree(n, sentence: str = ""):
    """The candidate's subtree as a Tree rooted at that node (its own relation reset to ROOT)."""
    from .types import REL_ROOT, Tree
    return Tree(sentence=sentence, root=replace(n, rel=REL_ROOT))


# ----------------------------------------------------------------------------- FIND / candidates / features
@dataclass
class Extracted:
    """Everything the policy sees for one (memory, question) pair. Numeric only: no token strings."""
    cand_gid: np.ndarray              # [C] index into the memory node table
    cand_sent: np.ndarray             # [C] sentence index of each candidate
    fs: np.ndarray                    # [C, nf] float32 scalar + relation features
    cv: np.ndarray                    # [C, D] float16 USVS vector of the candidate node (unit norm, 0 if none)
    sv: np.ndarray                    # [C, D] float16 USVS mean of the candidate subtree
    wv: np.ndarray                    # [D] float16 USVS vector of the question's wh node (0 if none)
    qf: np.ndarray                    # [nq] question structure features
    itf: np.ndarray                   # [ni] item-level features (for the abstain score)
    n_sent: int = 0
    n_nodes: int = 0
    meta: dict = field(default_factory=dict)


NS = 40                            # scalar candidate features before the relation one-hots


def n_cand_features(R: int) -> int:
    return NS + 4 * R


def n_q_features(R: int) -> int:
    return 6 + R


N_ITEM_FEATURES = 9


def _onehot(vocab: Dict[str, int], label: Optional[str], R: int) -> np.ndarray:
    v = np.zeros(R, dtype=np.float32)
    if label is not None and label in vocab:
        v[vocab[label]] = 1.0
    return v


def extract(memory, qtree, rel_vocab: Sequence[str], k: int = SEED_K, layers=None) -> Extracted:
    """FIND (balanced over the question's content nodes) -> one-link expansion -> numeric features."""
    ctx = Ctx.get()
    from .tree import walk
    R = len(rel_vocab)
    vocab = {r: i for i, r in enumerate(rel_vocab)}
    refs, sent_len = node_table(memory, layers)
    N = len(refs)
    qnodes = [n for n, _d, _p in walk(qtree.root) if n.token_index is not None]
    qpar = {}
    for n, _d, p in walk(qtree.root):
        qpar[id(n)] = p if (p is not None and p.token_index is not None) else None
    qdepth = {id(n): d for n, d, _p in walk(qtree.root)}
    # wh node: the question node whose own meaning point carries the UNKNOWN dimension
    wh, whval = None, 0.0
    for n in sorted(qnodes, key=lambda x: x.token_index):
        v = ctx.raw(ctx.chosen(n))
        if v is not None and ctx.unknown_idx >= 0 and v[ctx.unknown_idx] > whval:
            wh, whval = n, float(v[ctx.unknown_idx])
    qcontent = [n for n in qnodes if n is not wh and not ctx.is_function(n)]
    D = ctx.D
    wv = np.zeros(D, dtype=np.float16)
    if wh is not None:
        r = ctx.raw(ctx.chosen(wh))
        wv = (r / max(float(np.linalg.norm(r)), 1e-8)).astype(np.float16)
    ext_meta = {"wh_found": wh is not None}
    if N == 0 or not qcontent:
        return _empty(R, D, wv, wh, qnodes, qdepth, vocab, ext_meta)
    # ---- tensors (FIND) and meaning similarities (COMPARE)
    cache: dict = {}
    T = _unit_rows(np.stack([_sub_matrix(memory, r.node, cache) for r in refs]))
    TQ = _unit_rows(np.stack([_sub_matrix(memory, n, cache) for n in qcontent]))
    FIND = T @ TQ.T                                              # [N, Qc]
    nodes_all = [r.node for r in refs]
    S = sim_matrix(ctx, nodes_all, qcontent)                     # [N, Qc]
    ident = np.array([[1.0 if (r.node.lemma == qn.lemma and r.node.kind == qn.kind) else 0.0 for qn in qcontent]
                      for r in refs], dtype=np.float32)               # same concept (lemma identity), content nodes only
    wq = np.array([ctx.unit if ctx.key(n) is not None else
                   float(np.linalg.norm(ctx.raw(ctx.chosen(n)) * ctx.sw)) for n in qcontent], dtype=np.float32)
    wq = wq / max(float(wq.sum()), 1e-8)
    content = np.array([not ctx.is_function(r.node) for r in refs])
    # ---- FIND: per question node its top-k_q memory nodes (content only), then one tree link
    k_q = max(4, int(np.ceil(k / len(qcontent))))
    seeds: List[int] = []
    for j in range(len(qcontent)):
        order = [i for i in np.argsort(-FIND[:, j], kind="stable") if content[i] and FIND[i, j] > 0][:k_q]
        seeds += [i for i in order if i not in seeds]
    seed_set = set(seeds)
    cand_order: List[int] = list(seeds)
    seen = set(seeds)
    for g in seeds:
        for h in ([refs[g].parent] if refs[g].parent >= 0 else []) + refs[g].children:
            if h not in seen and content[h]:
                seen.add(h)
                cand_order.append(h)
    if not cand_order:
        return _empty(R, D, wv, wh, qnodes, qdepth, vocab, ext_meta)
    find_node = FIND.max(1)
    # ---- sentence level: tensor handle similarity and coverage
    hq = memory.handle_of(qtree)
    sims = {s.nid: c for s, c in memory.similar(hq, k=10 ** 6, layers=layers)}
    nids = sorted({r.nid for r in refs})
    sent_find = np.array([sims.get(n, 0.0) for n in nids], dtype=np.float32)
    sent_rank = np.argsort(np.argsort(-sent_find, kind="stable"), kind="stable") / max(len(nids), 1)
    n_s = len(nids)
    soft = np.zeros(n_s, dtype=np.float32)
    hard = np.zeros(n_s, dtype=np.float32)
    matched_node = (ident > 0).any(1) & content
    n_sentences = max(1, len({r.sent for r in refs}))
    df = np.array([len({r.sent for r in refs if ident[r.gid, j] > 0}) for j in range(len(qcontent))], dtype=np.float32)
    idw = np.log((n_sentences + 1.0) / (df + 1.0))
    idw = idw / max(float(idw.sum()), 1e-8)
    id_cov = np.zeros(len(set(r.sent for r in refs)), dtype=np.float32)
    for s in range(n_s):
        rows = [r.gid for r in refs if r.sent == s]
        if rows:
            m = S[rows].max(0)
            soft[s] = float((wq * m).sum())
            hard[s] = float((wq * (m >= SIM_HARD)).sum())
            id_cov[s] = float((idw * ident[rows].max(0)).sum())
    id_rank = np.argsort(np.argsort(-id_cov, kind="stable"), kind="stable") / max(n_s, 1)
    # ---- wh context in the question
    whrel = wh.rel if wh is not None else None
    whpar = qpar.get(id(wh)) if wh is not None else None
    qrefs, qg = _question_refs(qtree)
    story_lemmas = {(r.node.kind, r.node.lemma) for r in refs if content[r.gid]}
    qmatched = np.array([(qr.node is not wh and not ctx.is_function(qr.node) and (qr.node.kind, qr.node.lemma) in story_lemmas)
                         for qr in qrefs])
    wh_paths = []                         # [(target lemma key, steps)] nearest two matched question nodes from the wh node
    if wh is not None and qmatched.any():
        excl = set()
        for _ in range(2):
            dd, st, tx = _nearest_node(qrefs, qg[id(wh)], qmatched & np.array([i not in excl for i in range(len(qrefs))]))
            if tx is None:
                break
            excl.add(tx)
            wh_paths.append(((qrefs[tx].node.kind, qrefs[tx].node.lemma), st))
    wh_raw = ctx.raw(ctx.chosen(wh)) if wh is not None else None
    wh_axes = None
    if wh_raw is not None:
        wh_axes = wh_raw.copy()
        if ctx.unknown_idx >= 0:
            wh_axes[ctx.unknown_idx] = 0.0
        if float(np.linalg.norm(wh_axes)) < 1e-8:
            wh_axes = None
    par_sim = np.zeros(N, dtype=np.float32)
    if whpar is not None and not ctx.is_function(whpar):
        ps = sim_matrix(ctx, [refs[r.parent].node if r.parent >= 0 else r.node for r in refs], [whpar])[:, 0]
        par_sim = np.where([r.parent >= 0 for r in refs], ps, 0.0).astype(np.float32)
    mention: Dict[str, int] = {}
    for r in refs:
        key = (r.node.kind[:1], r.node.lemma)
        mention[key] = mention.get(key, 0) + 1
    # ---- per-candidate rows
    C = len(cand_order)
    nf = n_cand_features(R)
    fs = np.zeros((C, nf), dtype=np.float32)
    cv = np.zeros((C, D), dtype=np.float16)
    sv = np.zeros((C, D), dtype=np.float16)
    find_rank = np.argsort(np.argsort(-find_node[cand_order], kind="stable"), kind="stable") / max(C, 1)
    for ci, g in enumerate(cand_order):
        r = refs[g]
        n = r.node
        sub = subtree_nodes(n)
        raw = ctx.raw(ctx.chosen(n))
        has = raw is not None and ctx.key(n) is None
        if has:
            cv[ci] = raw / max(float(np.linalg.norm(raw)), 1e-8)
        acc = np.zeros(D, dtype=np.float32)
        for m in sub:
            rr = ctx.raw(ctx.chosen(m))
            if rr is not None and ctx.key(m) is None and not ctx.is_function(m):
                acc += rr / max(float(np.linalg.norm(rr)), 1e-8)
        if float(np.linalg.norm(acc)) > 0:
            sv[ci] = acc / float(np.linalg.norm(acc))
        # nearest other matched node in the same sentence, by tree links
        dist, steps = _nearest(refs, g, matched_node)
        pe = fe = le = 0.0
        for (tk, wsteps) in wh_paths:
            d2, st2 = _nearest(refs, g, matched_node, lambda x, tk=tk: (refs[x].node.kind, refs[x].node.lemma) == tk and refs[x].sent == r.sent)
            if d2 is not None and wsteps and st2:
                pe = max(pe, 1.0 if st2 == wsteps else 0.0)
                fe = max(fe, 1.0 if st2[0] == wsteps[0] else 0.0)
                le = max(le, 1.0 if st2[-1] == wsteps[-1] else 0.0)
        s = r.sent
        row = fs[ci]
        row[0] = find_node[g]
        row[1] = find_rank[ci]
        row[2] = sent_find[s]
        row[3] = sent_rank[s]
        row[4] = float(S[g].max())
        row[5] = soft[s]
        row[6] = hard[s]
        row[7] = 1.0 if g in seed_set else 0.0
        row[8] = 0.0 if g in seed_set else 1.0
        row[9] = 0.0
        if row[4] >= 0.3:
            jbest = int(S[g].argmax())
            if ctx.score.sense_opposite(ctx.chosen(n), ctx.chosen(qcontent[jbest]), ctx.score.DEFAULT_ANTONYM_TIERS):
                row[9] = 1.0
        row[10] = (dist / 10.0) if dist is not None else 0.0
        row[11] = 1.0 if dist is None else 0.0
        row[12] = sum(1 for d_, _r in steps if d_ == "up") / 5.0
        row[13] = sum(1 for d_, _r in steps if d_ == "down") / 5.0
        row[14] = r.depth / 10.0
        row[15] = len(r.children) / 5.0
        row[16] = len(sub) / 10.0
        row[17] = n.token_index / max(sent_len[s], 1)
        row[18] = s / max(n_s - 1, 1)
        row[19] = sent_len[s] / 40.0
        row[20] = 1.0 if n.kind == "name" else 0.0
        row[21] = mention[(n.kind[:1], n.lemma)] / 5.0
        row[22] = 1.0 if row[4] >= 0.99 else 0.0
        row[23] = 1.0 if (whrel is not None and r.rel == whrel) else 0.0
        row[24] = par_sim[g]
        row[25] = row[23] * row[24]
        row[26] = float(np.linalg.norm(raw * ctx.sw)) / 10.0 if has else 0.0
        row[27] = 1.0 if has else 0.0
        row[28] = 1.0 if (steps and steps[0][0] == "up") else 0.0
        row[29] = 1.0 if (steps and steps[0][0] == "down") else 0.0
        row[30] = id_cov[s]
        row[31] = id_rank[s]
        row[32] = 1.0 if id_cov[s] >= id_cov.max() - 1e-9 else 0.0
        row[33] = id_cov[s] / max(float(id_cov.max()), 1e-8)
        row[34] = pe
        row[35] = fe
        row[36] = le
        if wh_axes is not None and has:
            row[37] = float(raw @ wh_axes) / max(float(np.linalg.norm(raw) * np.linalg.norm(wh_axes)), 1e-8)
        if wh_axes is not None and float(np.linalg.norm(sv[ci].astype(np.float32))) > 0:
            row[38] = float(sv[ci].astype(np.float32) @ wh_axes) / max(float(np.linalg.norm(wh_axes)), 1e-8)
        row[39] = 1.0 if wh_axes is not None else 0.0
        o = NS
        row[o:o + R] = _onehot(vocab, r.rel, R)
        row[o + R:o + 2 * R] = _onehot(vocab, refs[r.parent].rel if r.parent >= 0 else None, R)
        row[o + 2 * R:o + 3 * R] = _onehot(vocab, steps[0][1] if steps else None, R)
        row[o + 3 * R:o + 4 * R] = _onehot(vocab, steps[-1][1] if steps else None, R)
    qf = _qfeat(R, vocab, wh, qnodes, qdepth, len(qcontent), float((S.max(0) >= SIM_HARD).mean()) if len(qcontent) else 0.0)
    top = np.sort(find_node[cand_order])[::-1]
    itf = np.array([top[0], top[:3].mean(), id_cov.max(), soft.max(), hard.max(),
                    float((S.max(0) >= SIM_HARD).mean()), C / 100.0, n_s / 30.0, float(len(seeds)) / SEED_K],
                   dtype=np.float32)
    return Extracted(np.array(cand_order), np.array([refs[g].sent for g in cand_order]), fs, cv, sv, wv, qf, itf,
                     n_s, N, ext_meta)


def _nearest(refs, g, matched, want=None) -> Tuple[Optional[int], list]:
    """Nearest OTHER matched node in g's tree (undirected BFS): (hops, [(direction, relation), ...]).
    ``want`` (optional) restricts the target to nodes for which want(x) is true."""
    d, steps, _x = _nearest_node(refs, g, matched, want)
    return d, steps


def _nearest_node(refs, g, matched, want=None):
    from collections import deque
    seen = {g}
    q = deque([(g, [])])
    while q:
        x, path = q.popleft()
        if x != g and matched[x] and (want is None or want(x)):
            return len(path), path, x
        r = refs[x]
        nxt = ([(r.parent, ("up", r.rel))] if r.parent >= 0 else []) + [(c, ("down", refs[c].rel)) for c in r.children]
        for y, step in nxt:
            if y not in seen:
                seen.add(y)
                q.append((y, path + [step]))
    return None, [], None


def _question_refs(qtree):
    """NodeRef table of the question tree (same structure as the memory's) and node-id -> index."""
    from .tree import walk
    refs, idx = [], {}
    for n, d, p in walk(qtree.root):
        if n.token_index is None:
            continue
        g = len(refs)
        idx[id(n)] = g
        par = idx.get(id(p), -1) if (p is not None and p.token_index is not None) else -1
        refs.append(NodeRef(g, 0, 0, n, par, n.rel, d))
        if par >= 0:
            refs[par].children.append(g)
    return refs, idx


def _qfeat(R, vocab, wh, qnodes, qdepth, n_content, matched_frac) -> np.ndarray:
    qf = np.zeros(n_q_features(R), dtype=np.float32)
    qf[0] = 1.0 if wh is not None else 0.0
    if wh is not None:
        qf[1] = qdepth[id(wh)] / 5.0
        qf[2] = wh.token_index / max(len(qnodes), 1)
        qf[6:] = _onehot(vocab, wh.rel, R)
    qf[3] = len(qnodes) / 20.0
    qf[4] = n_content / 10.0
    qf[5] = matched_frac
    return qf


def _empty(R, D, wv, wh, qnodes, qdepth, vocab, meta) -> Extracted:
    nf = n_cand_features(R)
    return Extracted(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros((0, nf), np.float32),
                     np.zeros((0, D), np.float16), np.zeros((0, D), np.float16), wv,
                     _qfeat(R, vocab, wh, qnodes, qdepth, 0, 0.0), np.zeros(N_ITEM_FEATURES, np.float32), 0, 0, meta)


# ----------------------------------------------------------------------------- labels (training/eval only)
def label_candidates(refs, cand_gid, option_trees, answer_idx, thr: float = CORRECT_THRESHOLD):
    """Per candidate subtree vs each option tree, by the tree scorer: meaning% and overall%. A candidate is correct
    when the correct option is its strictly best match (meaning, then overall) and the meaning is >= thr."""
    ctx = Ctx.get()
    C, O = len(cand_gid), len(option_trees)
    meaning = np.zeros((C, O), dtype=np.float32)
    overall = np.zeros((C, O), dtype=np.float32)
    opt_nodes = [subtree_nodes(t.root) for t in option_trees]
    for ci, g in enumerate(cand_gid):
        n = refs[g].node
        sub = subtree_nodes(n)
        cand_tree = subtree_tree(n)
        sm = sim_matrix(ctx, sub, [m for on in opt_nodes for m in on]) if sub else None
        off = 0
        for oi, t in enumerate(option_trees):
            k = len(opt_nodes[oi])
            if sm is not None and k and sm[:, off:off + k].max() >= 0.3:
                ps = ctx.score.score_pair(t, cand_tree, ctx.usvs)
                meaning[ci, oi], overall[ci, oi] = ps.meaning, ps.overall
            off += k
    key = meaning * 1000.0 + overall
    correct = np.zeros(C, dtype=bool)
    for ci in range(C):
        o = int(np.argmax(key[ci]))
        correct[ci] = (o == answer_idx and meaning[ci, o] >= thr and (key[ci] == key[ci, o]).sum() == 1)
    return meaning, overall, correct


def answer_sentence(refs, n_sent: int, option_tree) -> Optional[int]:
    """Index of the story sentence that best holds the option's content: mean over the option's content nodes of the
    best node similarity inside the sentence (measurement of where the answer is, not part of the system)."""
    ctx = Ctx.get()
    on = [m for m in subtree_nodes(option_tree.root) if not ctx.is_function(m)] or subtree_nodes(option_tree.root)
    if not refs or not on:
        return None
    S = sim_matrix(ctx, [r.node for r in refs], on)
    best, bi = -1.0, None
    for s in range(n_sent):
        rows = [r.gid for r in refs if r.sent == s]
        if rows:
            v = float(S[rows].max(0).mean())
            if v > best + 1e-9:
                best, bi = v, s
    return bi


def to_option(refs, gid: int, option_trees) -> Tuple[int, np.ndarray]:
    """Closest option to a chosen node (multiple-choice scoring only): (index, meaning% per option)."""
    ctx = Ctx.get()
    cand = subtree_tree(refs[gid].node)
    pairs = [ctx.score.score_pair(t, cand, ctx.usvs) for t in option_trees]
    ms = np.array([[p.meaning, p.overall] for p in pairs])
    return int(np.argmax(ms[:, 0] * 1000.0 + ms[:, 1])), ms[:, 0]


# ----------------------------------------------------------------------------- the policy (the learned part)
import torch
import torch.nn as nn


class Policy(nn.Module):
    """Scores each candidate (and ABSTAIN) from numeric features only. A shared linear projection maps the USVS vectors
    (candidate, subtree, the question's wh node) to a small space where the candidate/wh product is a learned
    compatibility; everything else is scalar or relation-label one-hot."""

    def __init__(self, D: int, nf: int, nq: int, ni: int = N_ITEM_FEATURES, P: int = 32, H: int = 64, drop: float = 0.2,
                 use_vec: bool = True):
        super().__init__()
        self.cfg = dict(D=D, nf=nf, nq=nq, ni=ni, P=P, H=H, drop=drop, use_vec=use_vec)
        self.use_vec = use_vec
        self.proj = nn.Linear(D, P, bias=False)
        self.score = nn.Sequential(nn.Linear(5 * P + nf + nq, H), nn.ReLU(), nn.Dropout(drop), nn.Linear(H, H // 2),
                                   nn.ReLU(), nn.Linear(H // 2, 1))
        self.abst = nn.Sequential(nn.Linear(ni + nq + P, H // 2), nn.ReLU(), nn.Linear(H // 2, 1))
        self.dropin = nn.Dropout(drop)

    def forward(self, b: dict) -> torch.Tensor:
        """b: cv, sv [B,C,D]; wv [B,D]; fs [B,C,nf]; qf [B,nq]; itf [B,ni]; mask [B,C] -> logits [B, C+1] (last = abstain)."""
        pc, ps, pw = self.proj(self.dropin(b["cv"])), self.proj(self.dropin(b["sv"])), self.proj(b["wv"])
        if not self.use_vec:
            pc, ps, pw = pc * 0, ps * 0, pw * 0
        B, C, _ = pc.shape
        w = pw.unsqueeze(1).expand(B, C, -1)
        q = b["qf"].unsqueeze(1).expand(B, C, -1)
        x = torch.cat([pc, ps, pc * w, ps * w, w, b["fs"], q], -1)
        s = self.score(x).squeeze(-1).masked_fill(~b["mask"], -1e9)
        a = self.abst(torch.cat([b["itf"], b["qf"], pw], -1))
        return torch.cat([s, a], -1)


def collate(exs: Sequence[Extracted]) -> dict:
    B = len(exs)
    C = max(1, max(len(e.cand_gid) for e in exs))
    D = exs[0].wv.shape[0]
    nf = exs[0].fs.shape[1]
    t = lambda sh, dt=np.float32: np.zeros(sh, dtype=dt)  # noqa: E731
    cv, sv, fs, mask = t((B, C, D)), t((B, C, D)), t((B, C, nf)), np.zeros((B, C), dtype=bool)
    for i, e in enumerate(exs):
        c = len(e.cand_gid)
        if c:
            cv[i, :c], sv[i, :c], fs[i, :c], mask[i, :c] = e.cv, e.sv, e.fs, True
    return {"cv": torch.from_numpy(cv), "sv": torch.from_numpy(sv), "fs": torch.from_numpy(fs),
            "wv": torch.from_numpy(np.stack([e.wv.astype(np.float32) for e in exs])),
            "qf": torch.from_numpy(np.stack([e.qf for e in exs])),
            "itf": torch.from_numpy(np.stack([e.itf for e in exs])), "mask": torch.from_numpy(mask)}


def save_policy(policy: Policy, rel_vocab, path, extra=None):
    torch.save({"cfg": policy.cfg, "state": policy.state_dict(), "rel_vocab": list(rel_vocab), "extra": extra or {}}, str(path))


def load_policy(path) -> Tuple[Policy, List[str]]:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    p = Policy(**ck["cfg"])
    p.load_state_dict(ck["state"])
    return p.eval(), ck["rel_vocab"]


@torch.no_grad()
def decide(policy: Policy, ex: Extracted) -> Tuple[int, np.ndarray]:
    """-> (candidate index, or -1 for ABSTAIN; probabilities over [candidates..., abstain])."""
    if len(ex.cand_gid) == 0:
        return -1, np.array([1.0])
    lg = policy(collate([ex]))[0]
    lg = torch.cat([lg[:len(ex.cand_gid)], lg[-1:]])
    p = torch.softmax(lg, -1).numpy()
    k = int(p.argmax())
    return (-1 if k == len(ex.cand_gid) else k), p


def render_subtree(node) -> str:
    return " ".join(m.token for m in sorted(subtree_nodes(node), key=lambda x: x.token_index))


def answer_from_memory(policy: Policy, rel_vocab, memory, qtree, layers=None, top: int = 5, ex: Optional[Extracted] = None) -> dict:
    """The answer and its JSON-serialisable trace. ``text`` is None for 'not stated'."""
    from .tree import compact
    ex = ex if ex is not None else extract(memory, qtree, rel_vocab, layers=layers)
    k, p = decide(policy, ex)
    refs, _ = node_table(memory, layers)
    stored = {s.nid: s for s in memory.trees()}
    order = list(np.argsort(-p[:len(ex.cand_gid)])[:top]) if len(ex.cand_gid) else []
    cands = []
    for ci in order:
        r = refs[int(ex.cand_gid[ci])]
        st = stored[r.nid]
        cands.append({"candidate": int(ci), "node": r.node.token, "subtree": render_subtree(r.node), "rel": r.rel,
                      "sentence_index": int(st.provenance.get("sentence_index", r.sent)), "sentence": st.tree.sentence,
                      "prob": round(float(p[ci]), 4), "find": round(float(ex.fs[ci, 0]), 4),
                      "compare": round(float(ex.fs[ci, 4]), 4)})
    out = {"question": qtree.sentence, "question_tree": compact(qtree), "candidates_considered": int(len(ex.cand_gid)),
           "top": cands, "abstain_prob": round(float(p[-1]), 4), "stated": k >= 0, "text": None, "sentence_ids": [],
           "facts": [], "node_gid": None}
    if k >= 0:
        r = refs[int(ex.cand_gid[k])]
        st = stored[r.nid]
        out.update(text=render_subtree(r.node), node_gid=int(ex.cand_gid[k]), sentence_ids=[int(st.provenance.get("sentence_index", r.sent))],
                   facts=[{"sentence_index": int(st.provenance.get("sentence_index", r.sent)), "sentence": st.tree.sentence,
                           "tree": compact(st.tree), "provenance": {a: b for a, b in st.provenance.items() if a != "sense_rules"}}],
                   prob=round(float(p[k]), 4))
    return out


# ----------------------------------------------------------------------------- building items from the read cache
def tree_of(reads: dict, s: str):
    from .tree import tree_from_dict
    r = reads.get(s)
    return None if r is None or r["tree"] is None else tree_from_dict(r["tree"])


def build_story_memory(context: Sequence[str], reads: dict, source: str = "story"):
    from .memory import Memory
    mem = Memory()
    for i, s in enumerate(context):
        t = tree_of(reads, s)
        if t is not None:
            mem.store(t, "current", {"source": source, "sentence_index": i, "reader": t.source})
    return mem


def make_rel_vocab(reads: dict) -> List[str]:
    from .tree import walk
    from .tree import tree_from_dict
    labs = set(EXTRA_LABELS)
    for r in reads.values():
        if r["tree"] is not None:
            labs |= {n.rel for n, _d, _p in walk(tree_from_dict(r["tree"]).root)}
    return sorted(labs)


_WORK: dict = {}


def _story_job(job):
    items, wp_items, reads, rel_vocab = job["items"], job["wp"], _WORK["reads"], _WORK["rel_vocab"]
    out = []
    mems: dict = {}
    def mem_for(key, context):
        if key not in mems:
            mems[key] = build_story_memory(context, reads)
        return mems[key]
    for it in items:
        mem = mem_for(tuple(it["context"]), it["context"])
        q = tree_of(reads, it["question"])
        opts = [tree_of(reads, o) for o in it["options"]]
        if q is None or any(o is None for o in opts):
            continue
        ex = extract(mem, q, rel_vocab)
        refs, _ = node_table(mem)
        meaning, overall, correct = label_candidates(refs, ex.cand_gid, opts, it["answer_idx"])
        n_s = len({r.sent for r in refs})
        asent = answer_sentence(refs, n_s, opts[it["answer_idx"]])
        rec = {"id": it["id"], "dataset": it["dataset"], "story": it["story"], "split": it["split"], "ex": ex,
               "meaning": meaning, "overall": overall, "correct": correct, "answer_idx": it["answer_idx"],
               "n_story_sentences": n_s, "question_words": len(subtree_nodes(q.root)),
               "question_trusted": bool(reads[it["question"]]["trusted"])}
        if asent is not None:
            ctxs = [s for s in it["context"] if tree_of(reads, s) is not None]
            sent_text = ctxs[asent]
            rec.update(answer_sentence=asent, answer_sentence_words=reads[sent_text]["n_words"],
                       answer_sentence_trusted=bool(reads[sent_text]["trusted"]))
        wp = wp_items.get(it["id"])
        if wp is not None:
            wmem = mem_for(tuple(wp["context"]), wp["context"])
            rec["ex_wp"] = extract(wmem, q, rel_vocab)
        out.append(rec)
    return out


def _init_items(reads, rel_vocab):
    import torch
    torch.set_num_threads(1)
    _WORK["reads"], _WORK["rel_vocab"] = reads, rel_vocab
    Ctx.get()


def build_items_stage(args, load_items_fn, load_reads_fn, out_dir: Path):
    import multiprocessing as mp
    import pickle
    import time
    items = load_items_fn()
    reads = load_reads_fn()
    rel_vocab = make_rel_vocab(reads)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rel_vocab, open(out_dir / "rel_vocab.json", "w"))
    by_story: Dict[Tuple[str, str], List[dict]] = {}
    for it in items:
        by_story.setdefault((it["dataset"], it["story"]), []).append(it)
    stories = sorted(by_story)
    first = {k: v[0] for k, v in by_story.items()}
    jobs = []
    for k in stories:
        wp = {}
        same = [s for s in stories if s[0] == k[0] and s != k]
        for it in by_story[k]:
            o = same[zlib.crc32(it["id"].encode()) % len(same)]       # a different story of the same dataset
            wp[it["id"]] = {"story": o[1], "context": first[o]["context"]}
        jobs.append({"items": by_story[k], "wp": wp})
    if args.limit:
        jobs = jobs[:args.limit]
    t0 = time.time()
    recs = []
    with mp.Pool(args.workers, initializer=_init_items, initargs=(reads, rel_vocab)) as pool:
        for n, out in enumerate(pool.imap_unordered(_story_job, jobs), 1):
            recs += out
            print(f"stories {n}/{len(jobs)} items {len(recs)} {time.time() - t0:.0f}s", flush=True)
    for sp in ("train", "val", "test"):
        part = sorted((r for r in recs if r["split"] == sp), key=lambda r: r["id"])
        with open(out_dir / f"items_{sp}.pkl", "wb") as f:
            pickle.dump(part, f)
        print(sp, len(part), "answer-in-candidates", round(float(np.mean([r["correct"].any() for r in part])), 3) if part else None)
