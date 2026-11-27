"""Mind v3: the thinker with a sentence step (dev/reports/SPAN_DIAGNOSIS.md, dev/reports/MIND_V3.md).

Method, data, signal and scoring are Mind v2's (``mind_v2``: reads, memory, targets, keep sets, decoders, ROUGE).
Only the learned thinker changes:
  1. two-step choice: a softmax over the story's SENTENCES (plus ABSTAIN), then a softmax over the nodes inside the chosen sentence;
  2. sentence context: sentence states see their neighbours (relative-offset attention + previous/next message passing),
     their relative position in the story and their signed offset to the best question-matching sentences (by FIND);
  3. length-invariant shape: node depth / size are relative to their own sentence, pooling is a mean, clause structure is read
     off the tree shape (a node with >= 2 distinct relation kinds below it), nothing gives the sentence scorer a node count;
  4. abstain is computed from the sentence / anchor scores (and the chosen sentence's state), not from lexical coverage.
No word identity reaches the model; every feature is positional, structural, an equality bit or a USVS point.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import mind_v2 as m2
from .mind_v2 import NB, NQ, NS

NSI = 10                 # length-invariant node scalars
NSENT = 8                # sentence feature numbers (position, 3 FIND offsets, 1 coverage offset, 3 pair scalars)
NEG = -1e4
RMAX = 6                 # sentence attention bias covers signed offsets -RMAX..RMAX
TOPK_OFF = 3


def _seg_max(x: torch.Tensor, seg: torch.Tensor, n: int, fill: float = -1e30) -> torch.Tensor:
    shape = (n,) + tuple(x.shape[1:])
    idx = seg.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    return torch.full(shape, fill, dtype=x.dtype).scatter_reduce(0, idx, x, "amax", include_self=True)


def _seg_lse(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    mx = _seg_max(x.detach(), seg, n)
    s = torch.zeros(n).index_add(0, seg, torch.exp(x - mx[seg]))
    return mx + torch.log(s.clamp(min=1e-30))


class Thinker3(nn.Module):
    def __init__(self, D: int, n_rel: int, n_pos: int, P: int = 24, H: int = 64, drop: float = 0.2, inv: bool = True, ctx: bool = True):
        super().__init__()
        self.H, self.inv, self.use_ctx, self.n_rel = H, inv, ctx, n_rel
        ns = NSI if inv else NS
        self.proj = nn.Linear(D, P, bias=False)
        self.rel = nn.Embedding(n_rel + 1, 8)
        self.pos = nn.Embedding(n_pos + 1, 8)
        self.s_in = nn.Sequential(nn.Linear(P + 16 + ns + NB, H), nn.GELU(), nn.Linear(H, H))
        self.q_in = nn.Sequential(nn.Linear(P + 16 + NS + NQ, H), nn.GELU(), nn.Linear(H, H))
        self.tree = nn.ModuleList([nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, H)) for _ in range(4)])
        self.tree_q = nn.ModuleList([nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, H)) for _ in range(2)])
        self.ln = nn.ModuleList([nn.LayerNorm(H) for _ in range(6)])
        self.wq, self.wk, self.wv = nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, H)
        self.cross = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, H))
        # sentence layer
        n_sf = NSENT if ctx else 3
        pool = 2 * H if inv else 2 * H + 1
        self.sent_in = nn.Sequential(nn.Linear(pool + 2 * H + n_sf, H), nn.GELU(), nn.Linear(H, H))
        self.sq, self.sk, self.sv = nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, H)
        self.bias = nn.Embedding(2 * RMAX + 1, 1)
        nn.init.zeros_(self.bias.weight)
        self.sround = nn.ModuleList([nn.Sequential(nn.Linear(4 * H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, H)) for _ in range(2)])
        self.sln = nn.ModuleList([nn.LayerNorm(H) for _ in range(2)])
        self.sscore = nn.Sequential(nn.Linear(H + 2 * H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, 1))
        self.anchor = nn.Sequential(nn.Linear(H + H + 2 * H + 1, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, 1))
        self.abst = nn.Sequential(nn.Linear(6 + H + 2 * H, H), nn.GELU(), nn.Linear(H, 1))
        self.keep = nn.Sequential(nn.Linear(3 * H + 2 * H + 3, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, 1))
        self.drop = nn.Dropout(drop)

    # ------------------------------------------------------------------ pieces
    def _round(self, h, parent, mlp, ln):
        has = (parent >= 0)
        p = parent.clamp(min=0)
        par = h[p] * has.unsqueeze(1)
        agg = torch.zeros_like(h).index_add(0, p, h * has.unsqueeze(1))
        cnt = torch.zeros(h.size(0)).index_add(0, p, has.float()).clamp(min=1).unsqueeze(1)
        return ln(h + mlp(torch.cat([h, agg / cnt, par], 1)))

    def shape(self, b: dict, gs: torch.Tensor, Sg: int):
        """Per-node structural scalars, all relative to the node's own sentence. Returns (scalars [N, NSI], maxdepth [N], size of sentence [N])."""
        depth, size, par = b["s_depth"], b["s_size"], b["s_parent"]
        N = depth.size(0)
        cnt = torch.bincount(gs, minlength=Sg).float()
        maxd = _seg_max(depth, gs, Sg)
        has = par >= 0
        p = par.clamp(min=0)
        nchild = torch.zeros(N).index_add(0, p, has.float())
        key = p[has] * (self.n_rel + 1) + b["s_rel"][has]
        ndist = torch.zeros(N)
        if key.numel():
            uk = torch.unique(key)
            ndist = ndist.index_add(0, uk // (self.n_rel + 1), torch.ones(uk.numel()))
        clause = (ndist >= 2).float()
        cd = clause.clone()
        for _ in range(int(depth.max().item()) + 1):
            cd = clause + torch.where(has, cd[p], torch.zeros(()))
        rel_cd = cd / _seg_max(cd, gs, Sg).clamp(min=1)[gs]
        # size: inv=1 -> share of the sentence (a leaf then reads 1/N, which gives N away); inv=2 -> share of the parent's subtree (local)
        rsize = size / cnt[gs] if self.inv == 1 else torch.where(has, size / size[p], torch.ones(()))
        sc = torch.stack([depth / maxd[gs].clamp(min=1), rsize, b["s_scal"][:, 2], b["s_scal"][:, 3], b["s_scal"][:, 4], b["s_scal"][:, 5],
                          (nchild == 0).float(), nchild.clamp(max=6) / 6.0, clause, rel_cd], 1)
        return sc, maxd[gs], cnt[gs]

    def sentence_features(self, b, gs, Sg, sent_item, rank, n_per):
        """Per-sentence numbers: relative position, signed offsets to the top-FIND and top-coverage sentences, the pair scalars."""
        B = b["B"]
        first = torch.zeros(Sg, dtype=torch.long).scatter(0, gs, torch.arange(gs.numel()))
        sb = b["s_bits"][first]                       # [Sg, NB]; bits 4-6 are constant within a sentence
        cov, find, frank = sb[:, 4], sb[:, 5], sb[:, 6]
        npi = n_per[sent_item].float()
        pos = rank.float() / (npi - 1).clamp(min=1)
        feats = [pos, cov, find, frank]
        if self.use_ctx:
            offs = torch.zeros(Sg, TOPK_OFF + 1)
            start = torch.cumsum(n_per, 0) - n_per
            for i in range(B):
                n = int(n_per[i])
                if n == 0:
                    continue
                sl = slice(int(start[i]), int(start[i]) + n)
                r = torch.arange(n).float()
                for k, v in enumerate((find[sl], cov[sl])):
                    kk = min(TOPK_OFF if k == 0 else 1, n)
                    top = torch.topk(v, kk).indices.float()
                    for j in range(kk):
                        offs[sl, (j if k == 0 else TOPK_OFF)] = ((top[j] - r) / 5.0).clamp(-2, 2)
            feats = [pos, cov, find, frank] + [offs[:, j] for j in range(TOPK_OFF + 1)]
        else:
            feats = [cov, find, frank]                # no position, no offsets, no neighbours: the pair scalars only
        return torch.stack(feats, 1)

    def encode(self, b: dict) -> dict:
        T = b["table"]
        B = b["B"]
        N = b["s_sense"].size(0)
        H = self.H
        xq = torch.cat([self.proj(T[b["q_sense"]]), self.rel(b["q_rel"]), self.pos(b["q_pos"]), b["q_scal"], b["q_bits"]], 1)
        hq = self.drop(self.q_in(xq))
        for i in range(2):
            hq = self._round(hq, b["q_parent"], self.tree_q[i], self.ln[4 + i])
        qm = torch.zeros(B, H).index_add(0, b["q_item"], hq)
        qm = qm / torch.bincount(b["q_item"], minlength=B).clamp(min=1).unsqueeze(1)
        wh = torch.where((b["q_wh"] >= 0).unsqueeze(1), hq[b["q_wh"].clamp(min=0)], torch.zeros(B, H))
        qvec = torch.cat([qm, wh], 1)
        if N == 0:
            ab = self._abst_in(torch.zeros(B, 1, H), torch.full((B, 6), -10.0), qvec)
            return {"hs": torch.zeros(0, H), "qvec": qvec, "alog": torch.zeros(0), "anode": torch.zeros(0), "gs": torch.zeros(0, dtype=torch.long),
                    "sl_pad": torch.full((B, 1), NEG), "n_per": torch.zeros(B, dtype=torch.long), "ablog": self.abst(ab).squeeze(1), "rank": torch.zeros(0, dtype=torch.long),
                    "sent_item": torch.zeros(0, dtype=torch.long), "maxd": torch.zeros(0), "ssz": torch.zeros(0), "start": torch.zeros(B, dtype=torch.long), "hq": hq, "hsent": torch.zeros(0, H)}
        uniq, gs = torch.unique(b["s_sent"], return_inverse=True)
        Sg = uniq.numel()
        cnt = torch.bincount(gs, minlength=Sg).float()
        sent_item = torch.zeros(Sg, dtype=torch.long).scatter(0, gs, b["s_item"])
        n_per = torch.bincount(sent_item, minlength=B)
        start = torch.cumsum(n_per, 0) - n_per
        rank = torch.arange(Sg) - start[sent_item]
        Smax = max(int(n_per.max()), 1)
        if self.inv:
            sc, maxd, ssz = self.shape(b, gs, Sg)
        else:
            sc, maxd, ssz = b["s_scal"], b["s_depth"], cnt[gs]
        xs = torch.cat([self.proj(T[b["s_sense"]]), self.rel(b["s_rel"]), self.pos(b["s_pos"]), sc, b["s_bits"]], 1)
        hs = self.drop(self.s_in(xs))
        hs = self._round(hs, b["s_parent"], self.tree[0], self.ln[0])
        att = (self.wq(hs) @ self.wk(hq).T) / math.sqrt(H)
        att = att.masked_fill(b["s_item"][:, None] != b["q_item"][None, :], -1e9)
        hs = hs + self.cross(torch.cat([hs, torch.softmax(att, 1) @ self.wv(hq)], 1))
        hs = self._round(hs, b["s_parent"], self.tree[1], self.ln[1])
        hs = self._round(hs, b["s_parent"], self.tree[2], self.ln[2])
        # ---- sentence layer
        hmean = torch.zeros(Sg, H).index_add(0, gs, hs) / cnt.unsqueeze(1)
        root = (b["s_parent"] < 0).float().unsqueeze(1)
        hroot = torch.zeros(Sg, H).index_add(0, gs, hs * root) / torch.zeros(Sg).index_add(0, gs, root.squeeze(1)).clamp(min=1).unsqueeze(1)
        if self.inv:
            pooled = torch.cat([hmean, hroot], 1)
        else:
            pooled = torch.cat([_seg_max(hs, gs, Sg), hmean, torch.log1p(cnt).unsqueeze(1) / 4.0], 1)
        sf = self.sentence_features(b, gs, Sg, sent_item, rank, n_per)
        h = self.sent_in(torch.cat([pooled, qvec[sent_item], sf], 1))
        if self.use_ctx:
            valid = (torch.arange(Smax)[None, :] < n_per[:, None])
            hp = torch.zeros(B, Smax, H).index_put((sent_item, rank), h)
            off = torch.arange(Smax)[None, :] - torch.arange(Smax)[:, None]
            bias = self.bias(off.clamp(-RMAX, RMAX) + RMAX).squeeze(-1)                       # [Smax, Smax]
            for r in range(2):
                a = (self.sq(hp) @ self.sk(hp).transpose(1, 2)) / math.sqrt(H) + bias[None]
                a = a.masked_fill(~valid[:, None, :], -1e9)
                cv = torch.softmax(a, 2) @ self.sv(hp)
                prev = torch.cat([torch.zeros(B, 1, H), hp[:, :-1]], 1)
                nxt = torch.cat([hp[:, 1:], torch.zeros(B, 1, H)], 1)
                hp = self.sln[r](hp + self.sround[r](torch.cat([hp, prev, nxt, cv], 2)) * valid.unsqueeze(2))
            h = hp[sent_item, rank]
        else:
            hp = torch.zeros(B, Smax, H).index_put((sent_item, rank), h)
        slog = self.sscore(torch.cat([h, qvec[sent_item]], 1)).squeeze(1)
        sl_pad = torch.full((B, Smax), NEG).index_put((sent_item, rank), slog)
        # ---- nodes inside a sentence
        a_in = torch.cat([hs, h[gs], qvec[b["s_item"]], b["s_bits"][:, 4:5]], 1)
        anode = self.anchor(a_in).squeeze(1)
        within = anode - _seg_lse(anode, gs, Sg)[gs]
        # joint score: log p(sentence | abstain-normalised) is added later; alog here is log p(sentence) + log p(node | sentence) over sentences only
        sl_lp = torch.log_softmax(sl_pad, 1)
        alog = sl_lp[b["s_item"], rank[gs]] + within
        # ---- abstain from the sentence / anchor scores
        top = sl_pad.argmax(1)
        s2 = torch.topk(sl_pad, min(2, Smax), 1).values
        gap = (s2[:, 0] - s2[:, 1]) if Smax > 1 else torch.zeros(B)
        gtop = (start + top).clamp(max=Sg - 1)
        best_in = _seg_max(anode, gs, Sg)
        wtop = _seg_max(within, gs, Sg)
        ent = -(torch.softmax(sl_pad, 1) * sl_lp).sum(1)
        st = torch.stack([s2[:, 0], gap, torch.logsumexp(sl_pad, 1), ent, best_in[gtop], wtop[gtop]], 1)
        has_item = (n_per > 0).float().unsqueeze(1)
        st = (st * has_item + (-10.0) * (1 - has_item)).clamp(-30, 30)
        ablog = self.abst(self._abst_in(hp, st, qvec, top)).squeeze(1)
        return {"hs": hs, "qvec": qvec, "alog": alog, "anode": anode, "within": within, "gs": gs, "sl_pad": sl_pad, "n_per": n_per, "ablog": ablog,
                "rank": rank, "sent_item": sent_item, "maxd": maxd, "ssz": ssz, "start": start, "hq": hq, "hsent": h[gs]}

    def _abst_in(self, hp, st, qvec, top=None):
        B = qvec.size(0)
        htop = hp[torch.arange(B), top] if top is not None else torch.zeros(B, self.H)
        return torch.cat([st, htop, qvec], 1)

    def keep_logits(self, e: dict, b: dict, a_idx: torch.Tensor, n_idx: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        hs = e["hs"]
        ha, hn = hs[a_idx], hs[n_idx]
        d, sz = b["s_depth"], b["s_size"]
        if self.inv:
            ssz = e["ssz"]
            rel = torch.stack([(d[n_idx] - d[a_idx]) / e["maxd"][a_idx].clamp(min=1), sz[n_idx] / ssz[a_idx], sz[n_idx] / sz[a_idx]], 1)
        else:
            rel = torch.stack([(d[n_idx] - d[a_idx]) / 5.0, torch.log1p(sz[n_idx]) / 4.0, torch.zeros(a_idx.numel())], 1)
        return self.keep(torch.cat([hn, ha, hn * ha, e["qvec"][item], rel], 1)).squeeze(1)


# ----------------------------------------------------------------------------- decisions
def _item_view(e, b, i):
    """Item i: node slice, node -> local sentence rank, sentence logits."""
    o0, o1 = int(b["s_off"][i]), int(b["s_off"][i + 1])
    n = int(e["n_per"][i])
    st = int(e["start"][i])
    gl = e["gs"][o0:o1] - st if o1 > o0 else e["gs"][o0:o1]
    return o0, o1, n, gl


@torch.no_grad()
def decide(model: Thinker3, b: dict, topk: int = 1):
    """Per item: anchor gid within the story or -1 (abstain). The abstain decision is the argmax over [sentences, abstain]; the
    anchor is the best node of the top sentence (topk > 1: the best joint sentence x node score among the top-k sentences)."""
    model.eval()
    e = model.encode(b)
    anchors, res = [], []
    for i in range(b["B"]):
        o0, o1, n, gl = _item_view(e, b, i)
        lg = torch.cat([e["sl_pad"][i, :n], e["ablog"][i:i + 1]])
        p = torch.softmax(lg, 0)
        k = int(lg.argmax())
        if n == 0 or k == n:
            anchors.append(-1)
            res.append({"p_abstain": float(p[-1]), "p_anchor": 0.0})
            continue
        cands = torch.topk(p[:n], min(topk, n)).indices.tolist() if topk > 1 else [k]
        best, ba = -1e30, -1
        for s in cands:
            idx = torch.where(gl == s)[0]
            if idx.numel() == 0:
                continue
            w = e["within"][o0:o1][idx]
            j = int(w.argmax())
            sc = float(torch.log(p[s]) + w[j])
            if sc > best:
                best, ba = sc, int(idx[j])
        anchors.append(ba)
        res.append({"p_abstain": float(p[-1]), "p_anchor": math.exp(best) if ba >= 0 else 0.0})
    return e, anchors, res


def sample(e: dict, b: dict):
    """Sample [sentence | abstain] then a node inside the sentence. Returns anchors and log-prob tensors (one per item)."""
    anchors, lps = [], []
    for i in range(b["B"]):
        o0, o1, n, gl = _item_view(e, b, i)
        lg = torch.cat([e["sl_pad"][i, :n], e["ablog"][i:i + 1]])
        lp = torch.log_softmax(lg, 0)
        c = int(torch.multinomial(lp.exp().detach(), 1))
        if n == 0 or c == n:
            anchors.append(-1)
            lps.append(lp[-1])
            continue
        idx = torch.where(gl == c)[0]
        w = e["within"][o0:o1][idx]
        j = int(torch.multinomial(torch.softmax(w, 0).detach(), 1))
        anchors.append(int(idx[j]))
        lps.append(lp[c] + torch.log_softmax(w, 0)[j])
    return anchors, lps


def stage1_loss(model: Thinker3, b: dict, pairs: Sequence[dict], stories: Dict, weights: Sequence[float]):
    """(sentence-and-abstain loss, node-in-sentence loss, keep loss). Sentence targets: every sentence holding an accepted anchor
    (soft, logsumexp); node target: the accepted anchors of the sentence of the best anchor."""
    e = model.encode(b)
    A, Nn, It, Y = [], [], [], []
    ls, la = [], []
    for i, p in enumerate(pairs):
        o0, o1, n, gl = _item_view(e, b, i)
        lg = torch.cat([e["sl_pad"][i, :n], e["ablog"][i:i + 1]])
        lse = torch.logsumexp(lg, 0)
        if p["kind"] == "wrong":
            ls.append(lse - e["ablog"][i])
            la.append(torch.zeros(()))
            continue
        acc = [a for a, _k, _f in p["tgt"]["anchors"]]
        sents = sorted({int(gl[a]) for a in acc})
        ls.append(lse - torch.logsumexp(lg[torch.tensor(sents)], 0))
        s0 = int(gl[acc[0]])
        in_s = [a for a in acc if int(gl[a]) == s0]
        w = e["within"][o0:o1]
        la.append(-torch.logsumexp(w[torch.tensor(in_s)], 0))
        arr = stories[p["skey"]]
        for a, kept, _f in p["tgt"]["anchors"]:
            d = arr["desc"][a]
            if len(d) == 0:
                continue
            ks = set(kept)
            A += [o0 + a] * len(d)
            Nn += (o0 + d).tolist()
            It += [i] * len(d)
            Y += [1.0 if g in ks else 0.0 for g in d.tolist()]
    w = torch.tensor(weights, dtype=torch.float32)
    den = w.sum().clamp(min=1e-6)
    loss_s = (torch.stack(ls) * w).sum() / den
    loss_a = (torch.stack(la) * w).sum() / den
    loss_k = torch.zeros(())
    if A:
        lg = model.keep_logits(e, b, torch.tensor(A), torch.tensor(Nn), torch.tensor(It))
        bce = F.binary_cross_entropy_with_logits(lg, torch.tensor(Y), reduction="none")
        wi = w[torch.tensor(It)]
        loss_k = (bce * wi).sum() / wi.sum().clamp(min=1e-6)
    return loss_s, loss_a, loss_k


# ----------------------------------------------------------------------------- v2 / v3 dispatch (scripts reuse the v2 evaluation code)
class Mind:
    """``mind_v2`` with ``decide`` dispatching on the model type, so the v2 evaluation / diagnosis scripts run v2 and v3 thinkers alike."""

    def __init__(self, topk: int = 1):
        self.topk = topk

    def __getattr__(self, name):
        return getattr(m2, name)

    def decide(self, model, b, thr: float = 0.5):
        if isinstance(model, Thinker3):
            return decide(model, b, topk=self.topk)
        return m2.decide(model, b, thr)
