"""Mind v2: the text-in / text-out chain (CONSCIOUSNESS_V2_PROPOSAL.md).

story -> memory (trees); question -> tree; THINKER (learned) picks an ANCHOR node in memory and keeps/drops each node of
the anchor's subtree (or abstains); the kept subtree is the answer TREE; the DECODER turns it into text; the text is scored
against the reference text. No word identity reaches the thinker: nodes carry a USVS point (shared projection), relation
label, tag, depth, subtree size and per-node bits ("same lemma as a question node" -- an equality test -- and the best USVS
similarity to a question node). Everything language-specific lives in the reader's grammar and the lexicon.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import mind_v1 as m1
from .tree import tree_from_dict, walk
from .types import REL_ROOT, Tree

NS, NQ, NB = 6, 3, 7                 # story-node scalars, question-node scalars, story-node pair bits
N_ITEM = 4
ACCEPT_F1 = 0.5                      # an item is "reachable" when some kept subtree reaches this lemma F1
ANCHOR_SLACK = 0.05                  # anchors within this F1 of the best are all accepted targets
MAX_ANCHORS = 3


# ----------------------------------------------------------------------------- data and metrics
def load_items(path) -> List[dict]:
    """FairytaleQA free-form items, the file's own split (whole stories; no story is in two splits)."""
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = r["meta"]
        golds = [g for g in (m.get("all_gold_answers") or [r.get("answer_text")]) if g]
        if not r.get("answer_text") or not golds:
            continue
        out.append({"id": f"ft{len(out)}", "story": m["story_id"], "split": m["split"], "context": list(r["context"]),
                    "question": r["question"], "answer": r["answer_text"], "golds": golds,
                    "local": m.get("local_or_sum"), "explicit": m.get("ex_or_im")})
    return out


def toks(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def rouge_l(pred: str, ref: str) -> float:
    """Plain LCS ROUGE-L F1 on lowercased word tokens."""
    a, b = toks(pred), toks(ref)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    l = prev[-1]
    if l == 0:
        return 0.0
    p, r = l / len(a), l / len(b)
    return 2 * p * r / (p + r)


def rouge_multi(pred: str, refs: Sequence[str]) -> float:
    return max((rouge_l(pred, r) for r in refs), default=0.0)


def band_of(n: Optional[int]) -> str:
    if n is None:
        return "none"
    return "<=10" if n <= 10 else ("11-20" if n <= 20 else "21+")


# ----------------------------------------------------------------------------- vocab / sense table
class Vocab:
    def __init__(self, rels: Sequence[str], poss: Sequence[str]):
        self.rels, self.poss = list(rels), list(poss)
        self.rel_id = {r: i + 1 for i, r in enumerate(self.rels)}     # 0 = unknown
        self.pos_id = {p: i + 1 for i, p in enumerate(self.poss)}

    @classmethod
    def from_reads(cls, reads: dict) -> "Vocab":
        rels, poss = set(m1.EXTRA_LABELS), set()
        for r in reads.values():
            if r["tree"] is not None:
                for n, _d, _p in walk(tree_from_dict(r["tree"]).root):
                    rels.add(n.rel)
                    poss.add(n.pos)
        return cls(sorted(rels), sorted(poss))

    def to_json(self):
        return {"rels": self.rels, "poss": self.poss}


class SenseTable:
    """Row per sense id: the unit-normalised, axis-weighted USVS point (row 0 = no point)."""

    def __init__(self):
        self.ctx = m1.Ctx.get()
        self.rows: Dict[Optional[str], int] = {None: 0}
        self.vecs: List[np.ndarray] = [np.zeros(self.ctx.D, dtype=np.float32)]

    def idx(self, sid: Optional[str]) -> int:
        if sid in self.rows:
            return self.rows[sid]
        v = self.ctx.raw(sid)
        if v is None:
            self.rows[sid] = 0
            return 0
        w = v * self.ctx.sw
        w = w / max(float(np.linalg.norm(w)), 1e-8)
        self.rows[sid] = len(self.vecs)
        self.vecs.append(w.astype(np.float32))
        return self.rows[sid]

    def matrix(self) -> torch.Tensor:
        if getattr(self, "_m", None) is None or self._m.size(0) != len(self.vecs):
            self._m = torch.from_numpy(np.stack(self.vecs))
        return self._m


# ----------------------------------------------------------------------------- stories (memory + node arrays)
def flat_tree(tree: Tree):
    """Token nodes of one tree: (nodes, parent idx, depth)."""
    nodes, parent, depth, idx = [], [], [], {}
    for n, d, p in walk(tree.root):
        if n.token_index is None:
            continue
        idx[id(n)] = len(nodes)
        nodes.append(n)
        parent.append(idx.get(id(p), -1) if p is not None else -1)
        depth.append(d)
    return nodes, parent, depth


def node_scalars(ctx, nodes, parent, depth, tindex_max) -> np.ndarray:
    N = len(nodes)
    size = np.ones(N, dtype=np.float32)
    for i in range(N - 1, -1, -1):          # pre-order: children come after parents
        if parent[i] >= 0:
            size[parent[i]] += size[i]
    sc = np.zeros((N, NS), dtype=np.float32)
    for i, n in enumerate(nodes):
        sc[i] = (depth[i] / 8.0, math.log1p(size[i]) / 4.0, n.token_index / max(tindex_max + 1, 1),
                 float(ctx.is_function(n)), float(n.kind == "name"), float(ctx.key(n) is not None and n.kind != "name"))
    return sc


class Story:
    """One passage in a Memory plus numeric node arrays. ``refs`` are the memory's node table (real Nodes with senses)."""

    def __init__(self, context: Sequence[str], reads: dict, vocab: Vocab):
        ctx = m1.Ctx.get()
        self.context = list(context)
        self.memory = m1.build_story_memory(context, reads)
        self.refs, _ = m1.node_table(self.memory)
        stored = sorted(self.memory.trees(), key=lambda s: s.nid)
        self.sent_text = [s.tree.sentence for s in stored]
        N = len(self.refs)
        self.N = N
        self.parent = np.array([r.parent for r in self.refs], dtype=np.int64)
        self.depth = np.array([r.depth for r in self.refs], dtype=np.int64)
        self.sent = np.array([r.sent for r in self.refs], dtype=np.int64)
        self.rel = np.array([vocab.rel_id.get(r.rel, 0) for r in self.refs], dtype=np.int64)
        self.pos = np.array([vocab.pos_id.get(r.node.pos, 0) for r in self.refs], dtype=np.int64)
        self.sid = [ctx.chosen(r.node) for r in self.refs]
        tmax = {}
        for r in self.refs:
            tmax[r.sent] = max(tmax.get(r.sent, 0), r.node.token_index)
        scal = np.zeros((N, NS), dtype=np.float32)
        size = np.ones(N, dtype=np.float32)
        for g in range(N - 1, -1, -1):
            if self.parent[g] >= 0:
                size[self.parent[g]] += size[g]
        self.size = size
        for g, r in enumerate(self.refs):
            n = r.node
            scal[g] = (r.depth / 8.0, math.log1p(size[g]) / 4.0, n.token_index / (tmax[r.sent] + 1.0),
                       float(ctx.is_function(n)), float(n.kind == "name"), float(ctx.key(n) is not None and n.kind != "name"))
        self.scal = scal
        self.desc: List[np.ndarray] = [np.zeros(0, dtype=np.int64)] * N
        for g in range(N - 1, -1, -1):
            kids = self.refs[g].children
            if kids:
                self.desc[g] = np.array(kids + [d for k in kids for d in self.desc[k].tolist()], dtype=np.int64)
        self.lemkey = [(r.node.kind, r.node.lemma) for r in self.refs]

    def arrays(self) -> dict:
        return {"parent": self.parent, "depth": self.depth, "sent": self.sent, "rel": self.rel, "pos": self.pos,
                "sid": self.sid, "scal": self.scal, "desc": self.desc, "size": self.size, "n_sent": len(self.sent_text),
                "sent_text": self.sent_text, "kids": [list(r.children) for r in self.refs],
                "nodes": [replace(r.node, children=[]) for r in self.refs],
                "tidx": np.array([r.node.token_index for r in self.refs], dtype=np.int64)}


def question_arrays(qtree: Tree, vocab: Vocab) -> dict:
    ctx = m1.Ctx.get()
    nodes, parent, depth = flat_tree(qtree)
    tmax = max([n.token_index for n in nodes] + [0])
    return {"nodes": nodes, "parent": np.array(parent, dtype=np.int64), "rel": np.array([vocab.rel_id.get(n.rel, 0) for n in nodes], dtype=np.int64),
            "pos": np.array([vocab.pos_id.get(n.pos, 0) for n in nodes], dtype=np.int64),
            "sid": [ctx.chosen(n) for n in nodes], "scal": node_scalars(ctx, nodes, parent, depth, tmax)}


def pair_features(story: Story, qtree: Tree, vocab: Vocab) -> Optional[dict]:
    """Question-dependent numeric features for one (story, question). No word identity is stored."""
    ctx = m1.Ctx.get()
    q = question_arrays(qtree, vocab)
    qn = q["nodes"]
    N, Q = story.N, len(qn)
    if Q == 0:
        return None
    sb = np.zeros((N, NB), dtype=np.float32)
    qb = np.zeros((Q, NQ), dtype=np.float32)
    wh, whv = -1, 0.0
    for j, n in enumerate(qn):
        r = ctx.raw(ctx.chosen(n))
        if r is not None and ctx.unknown_idx >= 0 and r[ctx.unknown_idx] > whv:
            wh, whv = j, float(r[ctx.unknown_idx])
        qb[j, 2] = float(r[ctx.unknown_idx]) if (r is not None and ctx.unknown_idx >= 0) else 0.0
    if N > 0:
        qkeys = [(n.kind, n.lemma) for n in qn]
        qcont = [j for j, n in enumerate(qn) if j != wh and not ctx.is_function(n)]
        S = m1.sim_matrix(ctx, [r.node for r in story.refs], qn)               # [N, Q]
        ident = np.array([[1.0 if k == qk else 0.0 for qk in qkeys] for k in story.lemkey], dtype=np.float32)
        sb[:, 0] = ident.max(1)
        if qcont:
            sb[:, 1] = ident[:, qcont].max(1)
            sb[:, 2] = S[:, qcont].max(1)
        n_sent = max(1, len(story.sent_text))
        df = np.array([len({story.sent[g] for g in range(N) if ident[g, j] > 0}) for j in range(Q)], dtype=np.float32)
        idw = np.log((n_sent + 1.0) / (df + 1.0))
        sb[:, 3] = (ident * idw[None]).max(1) / max(float(idw.max()), 1e-8) if Q else 0.0
        cov = np.zeros(n_sent, dtype=np.float32)
        wq = np.zeros(Q, dtype=np.float32)
        wq[qcont] = idw[qcont] if qcont else 0
        wq = wq / max(float(wq.sum()), 1e-8)
        for s in range(n_sent):
            rows = np.where(story.sent == s)[0]
            if len(rows):
                cov[s] = float((wq * ident[rows].max(0)).sum())
        hq = story.memory.handle_of(qtree)
        sims = {st.nid: c for st, c in story.memory.similar(hq, k=10 ** 6)}
        nids = sorted({r.nid for r in story.refs})
        find = np.array([sims.get(n, 0.0) for n in nids], dtype=np.float32)
        rank = np.argsort(np.argsort(-find, kind="stable"), kind="stable") / max(len(nids), 1)
        sb[:, 4] = cov[story.sent]
        sb[:, 5] = find[story.sent]
        sb[:, 6] = rank[story.sent]
        qb[:, 0] = [float(k in set(story.lemkey)) for k in qkeys]
        qb[:, 1] = S.max(0)
        itf = np.array([n_sent / 20.0, float(cov.max()), float(find.max()), len(qcont) / 10.0], dtype=np.float32)
    else:
        itf = np.zeros(N_ITEM, dtype=np.float32)
    return {"q": {k: v for k, v in q.items() if k != "nodes"}, "qb": qb, "sb": sb, "wh": wh, "itf": itf}


# ----------------------------------------------------------------------------- targets from the reference tree
def ref_lemmas(tree: Tree) -> Counter:
    return Counter((n.kind, n.lemma) for n, _d, _p in walk(tree.root) if n.token_index is not None)


def _consume(keys, ref: Counter):
    left = Counter(ref)
    hit = []
    for k in keys:
        if left[k] > 0:
            left[k] -= 1
            hit.append(True)
        else:
            hit.append(False)
    return hit


def best_subtrees(story: Story, ref: Counter):
    """For every node as anchor: the kept set maximising lemma F1 under two rules (strict: matched nodes connected to the
    anchor; closed: also the connectors between matched nodes). Returns [(f1, anchor, kept gids)] sorted best first."""
    nref = sum(ref.values())
    out = []
    if nref == 0:
        return out
    for a in range(story.N):
        order = [a] + story.desc[a].tolist()
        keys = [story.lemkey[g] for g in order]
        hit = _consume(keys, ref)
        matched = {g for g, h in zip(order, hit) if h}
        if not matched:
            continue
        # strict: matched nodes whose parent chain (inside the subtree) is all matched
        strict = set()
        for g in order:
            if g in matched and (g == a or story.parent[g] in strict):
                strict.add(g)
        # closed: ancestors-or-self of any matched node, up to the anchor
        closed = {a}
        for g in matched:
            x = g
            while x != a and x >= 0:
                closed.add(x)
                x = story.parent[x]
        cands = []
        for kept in (strict, closed):
            if a not in kept:
                continue
            kk = [story.lemkey[g] for g in sorted(kept)]
            ov = sum(_consume(kk, ref))
            if ov:
                p, r = ov / len(kk), ov / nref
                cands.append((2 * p * r / (p + r), -len(kept), kept))
        if cands:
            f1, _nl, kept = max(cands, key=lambda c: (c[0], c[1]))
            out.append((f1, a, kept))
    out.sort(key=lambda t: (-t[0], t[1]))
    return out


def answer_sentence_of(story: Story, ref: Counter) -> Optional[int]:
    best, bs = 0, None
    for s in range(len(story.sent_text)):
        keys = [story.lemkey[g] for g in np.where(story.sent == s)[0]]
        ov = sum(_consume(keys, ref))
        if ov > best:
            best, bs = ov, s
    return bs


def build_targets(story: Story, ref_tree: Tree, qkeys=()) -> dict:
    ctx = m1.Ctx.get()
    ref = ref_lemmas(ref_tree)
    subs = best_subtrees(story, ref)
    best = subs[0][0] if subs else 0.0
    acc = [(a, sorted(k), f1) for f1, a, k in subs if f1 >= best - ANCHOR_SLACK][:MAX_ANCHORS] if subs else []
    s_ans = answer_sentence_of(story, ref)
    known = set(story.lemkey) | set(qkeys)
    cont = [(n.kind, n.lemma) for n, _d, _p in walk(ref_tree.root) if n.token_index is not None and not ctx.is_function(n)]
    return {"best_f1": float(best), "anchors": acc, "ans_sent": s_ans,
            "ans_len": None if s_ans is None else int((story.sent == s_ans).sum()),
            "ref_n": int(sum(ref.values())), "ref_missing": float(np.mean([k not in known for k in cont])) if cont else 0.0}


# ----------------------------------------------------------------------------- the thinker
class Thinker(nn.Module):
    def __init__(self, D: int, n_rel: int, n_pos: int, P: int = 24, H: int = 64, drop: float = 0.2):
        super().__init__()
        self.H = H
        self.proj = nn.Linear(D, P, bias=False)
        self.rel = nn.Embedding(n_rel + 1, 8)
        self.pos = nn.Embedding(n_pos + 1, 8)
        self.s_in = nn.Sequential(nn.Linear(P + 16 + NS + NB, H), nn.GELU(), nn.Linear(H, H))
        self.q_in = nn.Sequential(nn.Linear(P + 16 + NS + NQ, H), nn.GELU(), nn.Linear(H, H))
        self.tree = nn.ModuleList([nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, H)) for _ in range(4)])
        self.tree_q = nn.ModuleList([nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, H)) for _ in range(2)])
        self.ln = nn.ModuleList([nn.LayerNorm(H) for _ in range(6)])
        self.wq, self.wk, self.wv = nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, H)
        self.cross = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, H))
        self.anchor = nn.Sequential(nn.Linear(2 * H + 2 * H + 1, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, 1))
        self.abst = nn.Sequential(nn.Linear(2 * H + 2 * H + N_ITEM + 1, H), nn.GELU(), nn.Linear(H, 1))
        self.keep = nn.Sequential(nn.Linear(3 * H + 2 * H + 2, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, 1))
        self.drop = nn.Dropout(drop)

    def _round(self, h, parent, mlp, ln):
        has = (parent >= 0)
        p = parent.clamp(min=0)
        par = h[p] * has.unsqueeze(1)
        agg = torch.zeros_like(h).index_add(0, p, h * has.unsqueeze(1))
        cnt = torch.zeros(h.size(0), device=h.device).index_add(0, p, has.float()).clamp(min=1).unsqueeze(1)
        return ln(h + mlp(torch.cat([h, agg / cnt, par], 1)))

    def encode(self, b: dict) -> dict:
        T = b["table"]
        xs = torch.cat([self.proj(T[b["s_sense"]]), self.rel(b["s_rel"]), self.pos(b["s_pos"]), b["s_scal"], b["s_bits"]], 1)
        xq = torch.cat([self.proj(T[b["q_sense"]]), self.rel(b["q_rel"]), self.pos(b["q_pos"]), b["q_scal"], b["q_bits"]], 1)
        hs, hq = self.drop(self.s_in(xs)), self.drop(self.q_in(xq))
        for i in range(2):
            hq = self._round(hq, b["q_parent"], self.tree_q[i], self.ln[4 + i])
        hs = self._round(hs, b["s_parent"], self.tree[0], self.ln[0])
        att = (self.wq(hs) @ self.wk(hq).T) / math.sqrt(self.H)
        att = att.masked_fill(b["s_item"][:, None] != b["q_item"][None, :], -1e9)
        ctx = torch.softmax(att, 1) @ self.wv(hq)
        hs = hs + self.cross(torch.cat([hs, ctx], 1))
        hs = self._round(hs, b["s_parent"], self.tree[1], self.ln[1])
        hs = self._round(hs, b["s_parent"], self.tree[2], self.ln[2])
        B = b["B"]
        qm = torch.zeros(B, self.H).index_add(0, b["q_item"], hq)
        qm = qm / torch.bincount(b["q_item"], minlength=B).clamp(min=1).unsqueeze(1)
        wh = torch.where((b["q_wh"] >= 0).unsqueeze(1), hq[b["q_wh"].clamp(min=0)], torch.zeros(B, self.H))
        qvec = torch.cat([qm, wh], 1)
        Sg = int(b["s_sent"].max().item()) + 1 if hs.size(0) else 0
        sp = torch.zeros(Sg, self.H).index_add(0, b["s_sent"], hs) / torch.bincount(b["s_sent"], minlength=Sg).clamp(min=1).unsqueeze(1) if Sg else hs
        spn = sp[b["s_sent"]] if Sg else hs
        a_in = torch.cat([hs, spn, qvec[b["s_item"]], b["s_bits"][:, 4:5]], 1)
        alog = self.anchor(a_in).squeeze(1)
        ab = []
        for i in range(B):
            o0, o1 = int(b["s_off"][i]), int(b["s_off"][i + 1])
            if o1 > o0:
                x = hs[o0:o1]
                ab.append(torch.cat([x.max(0).values, x.mean(0), qvec[i], b["itf"][i], torch.logsumexp(alog[o0:o1], 0, keepdim=True)]))
            else:
                ab.append(torch.cat([torch.zeros(2 * self.H), qvec[i], b["itf"][i], torch.full((1,), -10.0)]))
        ablog = self.abst(torch.stack(ab)).squeeze(1)
        return {"hs": hs, "qvec": qvec, "alog": alog, "ablog": ablog}

    def keep_logits(self, e: dict, b: dict, a_idx: torch.Tensor, n_idx: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        hs = e["hs"]
        ha, hn = hs[a_idx], hs[n_idx]
        rel = torch.stack([(b["s_depth"][n_idx] - b["s_depth"][a_idx]) / 5.0, torch.log1p(b["s_size"][n_idx]) / 4.0], 1)
        return self.keep(torch.cat([hn, ha, hn * ha, e["qvec"][item], rel], 1)).squeeze(1)


def collate(pairs: Sequence[dict], stories: Dict, tab: SenseTable, with_targets: bool = True) -> dict:
    """Concatenate pairs into one forest batch. ``stories`` maps skey -> arrays dict (or Story)."""
    sa = lambda p: stories[p["skey"]]
    B = len(pairs)
    s_off, q_off = [0], [0]
    cols = {k: [] for k in ("s_parent", "s_rel", "s_pos", "s_sent", "s_scal", "s_bits", "s_sense", "s_depth", "s_size",
                            "s_item", "q_parent", "q_rel", "q_pos", "q_scal", "q_bits", "q_sense", "q_item")}
    sg = 0
    q_wh, itf = [], []
    for i, p in enumerate(pairs):
        s = sa(p)
        arr = s if isinstance(s, dict) else s.arrays()
        N = len(arr["parent"])
        o = s_off[-1]
        cols["s_parent"].append(np.where(arr["parent"] >= 0, arr["parent"] + o, -1))
        cols["s_rel"].append(arr["rel"])
        cols["s_pos"].append(arr["pos"])
        cols["s_sent"].append(arr["sent"] + sg)
        cols["s_scal"].append(arr["scal"])
        cols["s_bits"].append(p["sb"] if N else np.zeros((0, NB), dtype=np.float32))
        if "sidx" not in arr:
            arr["sidx"] = np.array([tab.idx(x) for x in arr["sid"]], dtype=np.int64)
        cols["s_sense"].append(arr["sidx"])
        cols["s_depth"].append(arr["depth"].astype(np.float32))
        cols["s_size"].append(arr["size"])
        cols["s_item"].append(np.full(N, i, dtype=np.int64))
        sg += arr["n_sent"]
        s_off.append(o + N)
        q = p["q"]
        Q = len(q["parent"])
        oq = q_off[-1]
        cols["q_parent"].append(np.where(q["parent"] >= 0, q["parent"] + oq, -1))
        cols["q_rel"].append(q["rel"])
        cols["q_pos"].append(q["pos"])
        cols["q_scal"].append(q["scal"])
        cols["q_bits"].append(p["qb"])
        cols["q_sense"].append(np.array([tab.idx(x) for x in q["sid"]], dtype=np.int64))
        cols["q_item"].append(np.full(Q, i, dtype=np.int64))
        q_wh.append(-1 if p["wh"] < 0 else p["wh"] + oq)
        q_off.append(oq + Q)
        itf.append(p["itf"])
    cat = lambda k: np.concatenate(cols[k]) if cols[k] else np.zeros(0)
    b = {"B": B, "table": tab.matrix(), "s_off": torch.tensor(s_off), "itf": torch.from_numpy(np.stack(itf)),
         "q_wh": torch.tensor(q_wh, dtype=torch.long)}
    for k in cols:
        a = cat(k)
        if k in ("s_scal", "s_bits", "q_scal", "q_bits"):
            a = a.reshape(-1, {"s_scal": NS, "s_bits": NB, "q_scal": NS, "q_bits": NQ}[k]).astype(np.float32)
        b[k] = torch.from_numpy(a.astype(np.float32 if a.dtype.kind == "f" else np.int64))
    if with_targets:
        b["pairs"] = list(pairs)
    return b


# ----------------------------------------------------------------------------- decisions, answer trees
@torch.no_grad()
def decide(model: Thinker, b: dict, thr: float = 0.5):
    """Greedy: per item (anchor gid within the story or -1 = abstain, kept gids)."""
    model.eval()
    e = model.encode(b)
    res, a_list, n_list, it_list, anchors = [], [], [], [], []
    for i in range(b["B"]):
        o0, o1 = int(b["s_off"][i]), int(b["s_off"][i + 1])
        lg = torch.cat([e["alog"][o0:o1], e["ablog"][i:i + 1]])
        k = int(lg.argmax())
        p = torch.softmax(lg, 0)
        anchors.append(-1 if k == o1 - o0 else k)
        res.append({"p_abstain": float(p[-1]), "p_anchor": float(p[k]) if k < o1 - o0 else 0.0})
    return e, anchors, res


def keep_sets(model: Thinker, e: dict, b: dict, stories: Dict, pairs, anchors, thr: float = 0.5, sample: bool = False):
    """For each item with an anchor, the kept gids (anchor always kept; a node needs its parent kept). Returns
    (kept list, log-prob tensors per item when sample)."""
    A, Nn, It = [], [], []
    for i, (p, a) in enumerate(zip(pairs, anchors)):
        if a < 0:
            continue
        arr = stories[p["skey"]]
        d = arr["desc"][a]
        off = int(b["s_off"][i])
        A += [off + a] * len(d)
        Nn += (off + d).tolist()
        It += [i] * len(d)
    kept = [None] * len(pairs)
    logp = [torch.zeros(()) for _ in pairs]
    if A:
        lg = model.keep_logits(e, b, torch.tensor(A), torch.tensor(Nn), torch.tensor(It))
        pr = torch.sigmoid(lg)
        dec = (torch.bernoulli(pr.detach()) if sample else (pr > thr).float())
        lp = torch.where(dec > 0, F.logsigmoid(lg), F.logsigmoid(-lg))
        pos = 0
        for i, (p, a) in enumerate(zip(pairs, anchors)):
            if a < 0:
                continue
            arr = stories[p["skey"]]
            d = arr["desc"][a]
            dd = dec[pos:pos + len(d)]
            logp[i] = lp[pos:pos + len(d)].sum()
            pos += len(d)
            keep = {a}
            for g, v in zip(d.tolist(), dd.tolist()):          # desc is parent-before-child
                if v > 0 and arr["parent"][g] in keep:
                    keep.add(g)
            kept[i] = keep
    else:
        for i, a in enumerate(anchors):
            if a >= 0:
                kept[i] = {a}
    return kept, logp


def answer_tree(arr: dict, anchor: int, kept) -> Tree:
    """The kept subtree as a Tree (anchor's relation reset to ROOT). ``arr`` is a story's arrays dict."""
    def build(g, root):
        n = arr["nodes"][g]
        return replace(n, rel=REL_ROOT if root else n.rel, children=[build(c, False) for c in arr["kids"][g] if c in kept])
    return Tree(sentence="", root=build(anchor, True))


def true_order_text(arr: dict, kept) -> str:
    return " ".join(arr["nodes"][g].token for g in sorted(kept, key=lambda g: (int(arr["sent"][g]), int(arr["tidx"][g]))))


# ----------------------------------------------------------------------------- decoder (b): word from meaning
class Pool:
    """Words (token occurrences) of the story and the question, each with its USVS point and tag."""

    def __init__(self, story_nodes, qnodes):
        ctx = m1.Ctx.get()
        self.ctx = ctx
        nodes = list(story_nodes) + list(qnodes)
        seen, ents = {}, []
        for n in nodes:
            sid = ctx.chosen(n)
            key = (n.token, n.lemma, sid, n.pos, tuple(n.subtypes), n.kind, n.rel)
            if key in seen:
                ents[seen[key]]["count"] += 1
                continue
            seen[key] = len(ents)
            ents.append({"node": n, "count": 1, "vec": self.vec(n)})
        self.ents = ents
        self.V = np.stack([e["vec"] if e["vec"] is not None else np.zeros(ctx.D, np.float32) for e in ents]) if ents else np.zeros((0, ctx.D), np.float32)
        self.has = np.array([e["vec"] is not None for e in ents], dtype=bool)
        self.kind = [e["node"].kind for e in ents]

    def vec(self, n):
        r = self.ctx.raw(self.ctx.chosen(n))
        if r is None or n.kind == "name":
            return None
        w = r * self.ctx.sw
        return (w / max(float(np.linalg.norm(w)), 1e-8)).astype(np.float32)

    @staticmethod
    def tag_agreement(a, b) -> float:
        sa, sb = set(a.subtypes), set(b.subtypes)
        jac = len(sa & sb) / max(len(sa | sb), 1)
        return float(a.pos == b.pos) + 0.5 * jac + 0.5 * float(a.rel == b.rel)

    def choose(self, n):
        """Word for a node that carries only its meaning point, relation and tags: the pool entry with the nearest USVS
        point; ties (and nodes with no point: names, unlexiconed lemmas) go to tag agreement, then pool frequency, then
        the token string (canonical tie-break). Returns the pool Node."""
        v = self.vec(n)
        if v is not None and self.has.any():
            sims = np.where(self.has, self.V @ v, -9.0)
            cand = np.where(sims >= sims.max() - 1e-4)[0]
        else:
            cand = np.array([i for i in range(len(self.ents)) if (not self.has[i]) and self.kind[i] == n.kind], dtype=int)
            if len(cand) == 0:
                cand = np.array([i for i in range(len(self.ents)) if self.kind[i] == n.kind] or list(range(len(self.ents))), dtype=int)
        best = min(cand, key=lambda i: (-self.tag_agreement(n, self.ents[i]["node"]), -self.ents[i]["count"], self.ents[i]["node"].token))
        return self.ents[int(best)]["node"], v is not None


def meaning_only(tree: Tree, pool: Pool):
    """Replace every node's token/lemma by the word the pool chooses from its meaning. Returns (tree, [(chosen, original, had_point)])."""
    log = []

    def go(n):
        w, hp = pool.choose(n)
        log.append((w.token, n.token, hp))
        return replace(n, token=w.token, lemma=w.lemma, children=[go(c) for c in n.children])
    return Tree(sentence=tree.sentence, root=go(tree.root)), log


class Decoder:
    def __init__(self, path: str):
        from .orderer import Orderer
        self.orr = Orderer.load(path)

    def texts(self, trees: Sequence[Tree]) -> List[str]:
        if not trees:
            return []
        return [" ".join(n.token for n in ns) for ns in self.orr.order_trees(trees)]


# ----------------------------------------------------------------------------- reads and the items build
def load_reads(paths) -> dict:
    out = {}
    for p in paths:
        try:
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                out[r["k"]] = r
        except FileNotFoundError:
            pass
    return out


_W: dict = {}


def _init_build(reads_paths, vocab_json):
    import torch as _t
    _t.set_num_threads(1)
    _W["reads"] = load_reads(reads_paths)
    v = json.load(open(vocab_json))
    _W["vocab"] = Vocab(v["rels"], v["poss"])


def _job(job):
    reads, vocab = _W["reads"], _W["vocab"]
    story = Story(job["context"], reads, vocab)
    arr = story.arrays()
    out = []
    for sp in job["specs"]:
        qt = m1.tree_of(reads, sp["question"])
        if qt is None:
            continue
        pf = pair_features(story, qt, vocab)
        if pf is None:
            continue
        pf.update(id=sp["id"], kind=sp["kind"], item=sp["item"], skey=job["skey"])
        if sp.get("answer") is not None:
            at = m1.tree_of(reads, sp["answer"])
            if at is not None:
                qk = [(n.kind, n.lemma) for n, _d, _p in walk(qt.root) if n.token_index is not None]
                pf["tgt"] = build_targets(story, at, qk)
        out.append(pf)
    return job["skey"], arr, out


def build_items_stage(args, B, out_dir):
    import multiprocessing as mp
    import pickle
    import random
    import time
    from pathlib import Path
    cache2 = Path(out_dir) / "read_cache" / "reads_v2.jsonl"
    paths = [B.CACHE, cache2]
    reads = load_reads(paths)
    vocab = Vocab.from_reads(reads)
    vj = Path(out_dir) / "vocab.json"
    json.dump(vocab.to_json(), open(vj, "w"))
    items = load_items(B.FT)
    rng = random.Random(0)
    by_split = {}
    for it in items:
        by_split.setdefault(it["split"], []).append(it)
    ctx_ids, jobs = {}, {}

    def add(context, spec):
        key = tuple(context)
        if key not in ctx_ids:
            ctx_ids[key] = len(ctx_ids)
            jobs[key] = {"context": list(context), "skey": ctx_ids[key], "specs": []}
        jobs[key]["specs"].append(spec)

    meta = {}
    for sp, its in by_split.items():
        for it in its:
            add(it["context"], {"id": it["id"] + "|normal", "kind": "normal", "item": it["id"], "question": it["question"], "answer": it["answer"]})
            meta[it["id"]] = {"split": sp}
            if sp == "train" and rng.random() > 0.3:
                pass
            else:
                other = [o for o in its if o["story"] != it["story"]]
                o = rng.choice(other)
                add(o["context"], {"id": it["id"] + "|wrong", "kind": "wrong", "item": it["id"], "question": it["question"]})
            if sp != "train":
                same = [o for o in its if o["story"] == it["story"] and o["question"] != it["question"]]
                if same:
                    o = rng.choice(same)
                    add(it["context"], {"id": it["id"] + "|shuf", "kind": "shuf", "item": it["id"], "question": o["question"]})
    print(f"contexts {len(jobs)} pairs {sum(len(j['specs']) for j in jobs.values())}", flush=True)
    t0 = time.time()
    stories, pairs = {}, []
    with mp.Pool(args.workers, initializer=_init_build, initargs=(paths, str(vj))) as pool:
        for n, (sk, arr, out) in enumerate(pool.imap_unordered(_job, list(jobs.values()), chunksize=2), 1):
            stories[sk] = arr
            pairs += out
            if n % 50 == 0:
                print(f"contexts {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    id2split = {it["id"]: it["split"] for it in items}
    pickle.dump(stories, open(Path(out_dir) / "stories.pkl", "wb"))
    for sp in ("train", "val", "test"):
        ps = [p for p in pairs if id2split[p["item"]] == sp]
        pickle.dump(ps, open(Path(out_dir) / f"pairs_{sp}.pkl", "wb"))
        print(sp, len(ps), Counter(p["kind"] for p in ps), flush=True)
    json.dump({it["id"]: {k: it[k] for k in ("split", "story", "question", "answer", "golds", "local", "explicit")} | {"context_key": ctx_ids[tuple(it["context"])]} for it in items},
              open(Path(out_dir) / "items_meta.json", "w"))
    print("done", time.time() - t0, flush=True)
