"""Mind v4: the thinker answers by stepping down the tree (dev/reports/MIND_V4.md).

Method, data, signal and scoring are Mind v2/v3's (``mind_v2``: reads, memory, targets, keep sets, decoders, ROUGE). Only the
learned thinker changes. It is ONE small policy applied at every step; the state is the current node:
    HERE      this node's subtree is the answer (the keep/drop head may then prune, per node, as before)
    DOWN(k)   move to child k
    NEXT/PREV move to the next / previous sentence's root (only a root can do this)
    ABSTAIN   no answer
A step first chooses a MOVE (stay / NEXT / PREV) and, when it stays, an in-sentence action (HERE / ABSTAIN / DOWN(k)). NEXT and PREV
are hill-climbing steps on a learned sentence value (move iff the neighbour's value beats this one by a learned margin), so the
walk cannot ping-pong between two sentences.
The policy sees the question tree's representation, the node, its immediate children (relation, USVS point, subtree summary,
match bits, child order) and the parent's relation. No absolute depth, no sentence length, no sentence-wide pooling, no node
position in the sentence (only child order). The node states come from ONE recursive operation (a node merges its children's
states, bottom up, the same weights at every level), so a long sentence is the same problem as a short one, only more steps.
At a sentence root the policy also sees how promising the neighbouring sentences are (a learned value per root, +-1 and +-2
roots away), the signed offsets to the best question-matching sentences (from the memory's FIND) and, for ABSTAIN only, the
story's best match scores. No word identity reaches the model: every input is a USVS point, a relation / tag, an equality bit or
a structural number.

Because the policy is Markov in the node, its action logits are computed for all nodes at once; training is cross-entropy on the
states along the gold paths (start sentence -> oracle sentence by NEXT/PREV, then DOWN to the oracle anchor, then HERE); walking
is a loop over those logits.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import mind_v2 as m2
from . import mind_v3 as m3
from .mind_v2 import NB, NQ, NS

NEG = -1e4
NLOC = 5                      # local node scalars: is_function, is_name, has_key, is_leaf, number of children
BUDGET = 20                   # step budget of the walk (12 at training time; 20 chosen on val: same answers, no budget exhaustion)
N_HERE, N_NEXT, N_PREV, N_ABST = 0, 1, 2, 3
NOFF = 4                      # signed offsets to the top-3 FIND roots and the top coverage root


def _seg_lse(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    return m3._seg_lse(x, seg, n)


class Thinker4(nn.Module):
    def __init__(self, D: int, n_rel: int, n_pos: int, P: int = 24, H: int = 64, drop: float = 0.2):
        super().__init__()
        self.H, self.n_rel = H, n_rel
        self.proj = nn.Linear(D, P, bias=False)
        self.rel = nn.Embedding(n_rel + 1, 8)
        self.pos = nn.Embedding(n_pos + 1, 8)
        self.sib = nn.Embedding(6, 8)                                      # child order among siblings (clipped)
        self.s_in = nn.Sequential(nn.Linear(P + 16 + NLOC + NB, H), nn.GELU(), nn.Linear(H, H))
        self.q_in = nn.Sequential(nn.Linear(P + 16 + NS + NQ, H), nn.GELU(), nn.Linear(H, H))
        self.tree_q = nn.ModuleList([nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, H)) for _ in range(2)])
        self.ln_q = nn.ModuleList([nn.LayerNorm(H) for _ in range(2)])
        self.wq, self.wk, self.wv = nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, H)
        self.wq2, self.wk2, self.wv2 = nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, H)
        self.cross = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, H))
        self.cross2 = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, H))
        self.rec = nn.Sequential(nn.Linear(2 * H + 5, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, H))   # the one recursive merge
        self.ln_rec = nn.LayerNorm(H)
        self.st_in = nn.Sequential(nn.Linear(H + 2 * H + 16 + 4, H), nn.GELU(), nn.Linear(H, H))
        self.here = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, 1))
        self.down = nn.Sequential(nn.Linear(3 * H + 20, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, 1))
        self.sval = nn.Sequential(nn.Linear(H + NOFF, H), nn.GELU(), nn.Linear(H, 1))
        self.margin = nn.Parameter(torch.zeros(2))                       # NEXT / PREV need the neighbour to win by softplus(margin) > 0
        self.abst = nn.Sequential(nn.Linear(H + 2 + 1, H), nn.GELU(), nn.Linear(H, 1))
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

    def _attend(self, h, hq, b, wq, wk, wv):
        att = (wq(h) @ wk(hq).T) / math.sqrt(self.H)
        att = att.masked_fill(b["s_item"][:, None] != b["q_item"][None, :], -1e9)
        return torch.softmax(att, 1) @ wv(hq)

    def _recurse(self, h, bits, parent, depth):
        """Bottom-up: every node merges its children's states with the SAME function (depth only schedules the order).
        Returns the node states and the subtree max of the four match bits."""
        N = h.size(0)
        has = parent >= 0
        p = parent.clamp(min=0)
        sm = bits.clone()
        for d in range(int(depth.max().item()), -1, -1):
            ch = (depth == d + 1).nonzero().squeeze(1)
            tgt = (depth == d).nonzero().squeeze(1)
            par = p[ch]
            agg = torch.zeros(N, h.size(1)).index_add(0, par, h[ch])
            cnt = torch.zeros(N).index_add(0, par, torch.ones(ch.numel()))
            cmax = torch.zeros(N, bits.size(1)).scatter_reduce(0, par.unsqueeze(1).expand(-1, bits.size(1)), sm[ch], "amax", include_self=True)
            new = self.ln_rec(h[tgt] + self.rec(torch.cat([h[tgt], agg[tgt] / cnt[tgt].clamp(min=1).unsqueeze(1), cmax[tgt], (cnt[tgt] > 0).float().unsqueeze(1)], 1)))
            h = h.index_copy(0, tgt, new)
            sm = sm.index_copy(0, tgt, torch.maximum(sm[tgt], cmax[tgt]))
        return h, sm

    def encode(self, b: dict) -> dict:
        T = b["table"]
        B = b["B"]
        H = self.H
        xq = torch.cat([self.proj(T[b["q_sense"]]), self.rel(b["q_rel"]), self.pos(b["q_pos"]), b["q_scal"], b["q_bits"]], 1)
        hq = self.drop(self.q_in(xq))
        for i in range(2):
            hq = self._round(hq, b["q_parent"], self.tree_q[i], self.ln_q[i])
        qm = torch.zeros(B, H).index_add(0, b["q_item"], hq)
        qm = qm / torch.bincount(b["q_item"], minlength=B).clamp(min=1).unsqueeze(1)
        wh = torch.where((b["q_wh"] >= 0).unsqueeze(1), hq[b["q_wh"].clamp(min=0)], torch.zeros(B, H))
        qvec = torch.cat([qm, wh], 1)
        N = b["s_sense"].size(0)
        if N == 0:
            return {"N": 0, "qvec": qvec, "hs": torch.zeros(0, H)}
        parent, depth = b["s_parent"], b["s_depth"].long()
        has = parent >= 0
        p = parent.clamp(min=0)
        nchild = torch.zeros(N).index_add(0, p, has.float())
        loc = torch.stack([b["s_scal"][:, 3], b["s_scal"][:, 4], b["s_scal"][:, 5], (nchild == 0).float(), nchild.clamp(max=6) / 6.0], 1)
        par_np = parent.numpy()
        sib_np = np.zeros(N, dtype=np.int64)
        seen: Dict[int, int] = {}
        for g in range(N):
            pp = int(par_np[g])
            if pp >= 0:
                sib_np[g] = seen.get(pp, 0)
                seen[pp] = int(sib_np[g]) + 1
        sibe = self.sib(torch.from_numpy(np.minimum(sib_np, 5)))
        x = torch.cat([self.proj(T[b["s_sense"]]), self.rel(b["s_rel"]), self.pos(b["s_pos"]), loc, b["s_bits"]], 1)
        h = self.drop(self.s_in(x))
        h = h + self.cross(torch.cat([h, self._attend(h, hq, b, self.wq, self.wk, self.wv)], 1))
        h, sm = self._recurse(h, b["s_bits"][:, :4], parent, depth)
        hs = h + self.cross2(torch.cat([h, self._attend(h, hq, b, self.wq2, self.wk2, self.wv2)], 1))
        rel_n = self.rel(b["s_rel"])
        rel_p = self.rel(b["s_rel"][p]) * has.unsqueeze(1)
        st = self.st_in(torch.cat([hs, qvec[b["s_item"]], rel_n, rel_p, sm], 1))
        here = self.here(st).squeeze(1)
        # ---- children: one score per (parent, child) edge, stored at the child. The score of going down into child c is the
        # log of the mass of 'here' scores in c's subtree (the same bottom-up merge at every node: sub[n] = lse(here[n], sub[children]))
        # plus a learned local correction, so a descent is exactly the chain rule of a softmax over the nodes below.
        ch = has.nonzero().squeeze(1)
        down = torch.full((N,), NEG)
        sub = here
        for d in range(int(depth.max().item()) - 1, -1, -1):
            cd = (depth == d + 1).nonzero().squeeze(1)
            if cd.numel() == 0:
                continue
            tg = (depth == d).nonzero().squeeze(1)
            cl = _seg_lse(sub[cd], p[cd], N)
            sub = sub.index_copy(0, tg, torch.logaddexp(here[tg], cl[tg]))
        if ch.numel():
            pc = p[ch]
            dv = sub[ch] + self.down(torch.cat([st[pc], hs[ch], rel_n[ch], sibe[ch], sm[ch], hs[pc] * hs[ch]], 1)).squeeze(1)
            down = down.index_put((ch,), dv)
        # ---- sentence roots: signed offsets to the best question-matching roots, a learned value, hill-climbing moves
        R = (~has).nonzero().squeeze(1)
        nR = R.numel()
        ritem = b["s_item"][R]
        rcount = torch.bincount(ritem, minlength=B)
        rstart = torch.cumsum(rcount, 0) - rcount
        find, cov = b["s_bits"][R, 5], b["s_bits"][R, 4]
        offs = torch.zeros(nR, NOFF)
        start = torch.full((B,), -1, dtype=torch.long)
        for i in range(B):
            n = int(rcount[i])
            if n == 0:
                continue
            sl = slice(int(rstart[i]), int(rstart[i]) + n)
            r = torch.arange(n).float()
            kk = min(3, n)
            top = torch.topk(find[sl], kk).indices
            for j in range(kk):
                offs[sl, j] = ((top[j].float() - r) / 5.0).clamp(-2, 2)
            offs[sl, 3] = ((torch.argmax(cov[sl]).float() - r) / 5.0).clamp(-2, 2)
            start[i] = R[int(rstart[i]) + int(top[0])]
        sv = self.sval(torch.cat([st[R], offs], 1)).squeeze(1)
        start_v = start.clone()
        for i in range(B):
            if int(rcount[i]):
                start_v[i] = R[int(rstart[i]) + int(torch.argmax(sv[int(rstart[i]):int(rstart[i]) + int(rcount[i])]))]

        def shift(k):
            ix = torch.arange(nR) + k
            ixc = ix.clamp(0, nR - 1)
            return ixc, (ix >= 0) & (ix < nR) & (ritem[ixc] == ritem)
        i1, ok1 = shift(1)
        im, okm = shift(-1)
        negs = torch.full((nR,), NEG)
        nxt = torch.full((N,), NEG).index_put((R,), torch.where(ok1, sv[i1] - sv - F.softplus(self.margin[0]), negs))
        prv = torch.full((N,), NEG).index_put((R,), torch.where(okm, sv[im] - sv - F.softplus(self.margin[1]), negs))
        mneg = torch.full((nR,), -1, dtype=torch.long)
        next_node = torch.full((N,), -1, dtype=torch.long).index_put((R,), torch.where(ok1, R[i1], mneg))
        prev_node = torch.full((N,), -1, dtype=torch.long).index_put((R,), torch.where(okm, R[im], mneg))
        svn = torch.zeros(N).index_put((R,), sv)
        abst = self.abst(torch.cat([st, b["itf"][b["s_item"]][:, 1:3], svn.unsqueeze(1)], 1)).squeeze(1)
        return {"N": N, "hs": hs, "qvec": qvec, "move": torch.stack([torch.zeros(N), nxt, prv], 1), "in2": torch.stack([here, abst], 1), "down": down,
                "next_node": next_node, "prev_node": prev_node, "start": start, "start_v": start_v, "R": R, "rcount": rcount, "rstart": rstart, "find": find, "sv": sv}

    def keep_logits(self, e: dict, b: dict, a_idx: torch.Tensor, n_idx: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        hs = e["hs"]
        ha, hn = hs[a_idx], hs[n_idx]
        d, sz, par = b["s_depth"], b["s_size"], b["s_parent"]
        pn = par[n_idx].clamp(min=0)
        rel = torch.stack([((d[n_idx] - d[a_idx]) / 8.0).clamp(max=1.0), sz[n_idx] / sz[a_idx], sz[n_idx] / sz[pn]], 1)
        return self.keep(torch.cat([hn, ha, hn * ha, e["qvec"][item], rel], 1)).squeeze(1)


# ----------------------------------------------------------------------------- walking
def _kids_of(parent: torch.Tensor):
    kids: Dict[int, List[int]] = {}
    for c, p in enumerate(parent.tolist()):
        if p >= 0:
            kids.setdefault(p, []).append(c)
    return kids


def _walk(e, kids, s0: int, budget: int, here_bias: float = 0.0):
    """Greedy walk from node s0. Returns (final node or -1, steps, path nodes, sum log-prob, p_abstain at the first state)."""
    mv, in2, down = e["move"], e["in2"], e["down"]
    s, path, lp, pab = s0, [s0], 0.0, None
    for step in range(budget):
        ks = kids.get(s, [])
        lm = torch.log_softmax(mv[s], 0)
        lin = torch.log_softmax(torch.cat([in2[s], down[ks]]) if ks else in2[s], 0)
        if here_bias:
            lin = torch.log_softmax(lin + here_bias * (torch.arange(lin.numel()) == 0).float(), 0)       # decision-time calibration, chosen on val
        if pab is None:
            pab = float(torch.exp(lm[0] + lin[1]))
        m = int(lm.argmax())
        if m == 1 or m == 2:
            lp += float(lm[m])
            s = int(e["next_node"][s]) if m == 1 else int(e["prev_node"][s])
            path.append(s)
            continue
        k = int(lin.argmax())
        lp += float(lm[0] + lin[k])
        if k == 0:
            return s, step + 1, path, lp, pab
        if k == 1:
            return -1, step + 1, path, lp, pab
        s = ks[k - 2]
        path.append(s)
    return -1, budget, path, lp, pab               # budget exhausted = no decision


@torch.no_grad()
def decide(model: Thinker4, b: dict, nstart: int = 1, budget: int = BUDGET, start: str = "value", here_bias: float = 0.0):
    """Per item: anchor gid within the story or -1. The walk starts at the root with the best learned sentence value (start="value")
    or the best FIND root (start="find"); nstart > 1: walk from the top-nstart roots and keep the answering walk with the best summed
    log-prob (selected on val)."""
    model.eval()
    e = model.encode(b)
    anchors, res = [], []
    kids = _kids_of(b["s_parent"]) if e["N"] else {}
    for i in range(b["B"]):
        o0, o1 = int(b["s_off"][i]), int(b["s_off"][i + 1])
        if o1 == o0:
            anchors.append(-1)
            res.append({"p_abstain": 1.0, "p_anchor": 0.0, "steps": 0, "exhausted": False, "path": []})
            continue
        rs, rc = int(e["rstart"][i]), int(e["rcount"][i])
        key = e["sv"] if start == "value" else e["find"]
        order = torch.argsort(-key[rs:rs + rc], stable=True)[:max(nstart, 1)]
        best = None
        for j in order.tolist():
            s0 = int(e["R"][rs + j])
            fin, steps, path, lp, pab = _walk(e, kids, s0, budget, here_bias)
            cand = (fin >= 0, lp, fin, steps, path, pab, steps >= budget and fin < 0)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
        _ans, _lp, fin, steps, path, pab, exh = best
        anchors.append(fin - o0 if fin >= 0 else -1)
        res.append({"p_abstain": pab, "p_anchor": 0.0, "steps": steps, "exhausted": bool(exh), "path": [g - o0 for g in path]})
    return e, anchors, res


# ----------------------------------------------------------------------------- training targets (answer-derived only)
def _roots_local(arr):
    if "roots_l" not in arr:
        arr["roots_l"] = [g for g in range(len(arr["parent"])) if arr["parent"][g] < 0]
    return arr["roots_l"]


def build_states(pairs, b, stories, e, rng=None, rand_start: bool = True):
    """Gold states and their acceptable actions. Returns (node idx, weight, acc4 [S,4] bool, acc_down [N] bool, item idx)."""
    N = b["s_sense"].size(0)
    nodes, ws, acc4, its = [], [], [], []
    acc_down = torch.zeros(N, dtype=torch.bool)
    for i, p in enumerate(pairs):
        o0 = int(b["s_off"][i])
        arr = stories[p["skey"]]
        RL = _roots_local(arr)
        if not RL:
            continue
        rank = {int(arr["sent"][g]): j for j, g in enumerate(RL)}
        j0 = RL.index(int(e["start"][i]) - o0)
        if p["kind"] == "wrong":
            for j, g in enumerate(RL):
                nodes.append(o0 + g)
                ws.append(1.0 if j == j0 else 1.0 / max(len(RL) - 1, 1))
                acc4.append([False, False, False, True])
                its.append(i)
            continue
        anchors = {a for a, _k, _f in p["tgt"]["anchors"]}
        sacc = sorted({rank[int(arr["sent"][a])] for a in anchors})
        anc = set()
        for a in anchors:
            x = a
            while x >= 0:
                anc.add(x)
                x = int(arr["parent"][x])
        starts = [j0]
        if rand_start and rng is not None and len(RL) > 1:
            starts.append(rng.randrange(len(RL)))
        states = {}                                      # node -> acc4
        for g in anc:
            kids = arr["kids"][g]
            ds = [c for c in kids if c in anc]
            states[g] = ([g in anchors, False, False, False], ds)
        for j in starts:
            t = min(sacc, key=lambda s: (abs(s - j), s))
            for r in range(min(j, t), max(j, t) + 1):
                g = RL[r]
                if r in sacc:
                    continue                             # an accepted root is already a tree state
                dmin = min(abs(s - r) for s in sacc)
                near = [s for s in sacc if abs(s - r) == dmin]
                states[g] = ([False, any(s > r for s in near), any(s < r for s in near), False], [])
        w = 1.0 / len(states)
        for g, (a4, ds) in states.items():
            nodes.append(o0 + g)
            ws.append(w)
            acc4.append(a4)
            its.append(i)
            for c in ds:
                acc_down[o0 + c] = True
    return torch.tensor(nodes, dtype=torch.long), torch.tensor(ws), torch.tensor(acc4, dtype=torch.bool), acc_down, torch.tensor(its, dtype=torch.long)


def stage1_loss(model: Thinker4, b: dict, pairs: Sequence[dict], stories: Dict, rng=None, rand_start: bool = True):
    """(path loss, sentence-value loss, keep loss). Path loss: -log of the probability mass on the acceptable actions of each gold state, weighted so
    every item counts once; the keep loss as v3 (accepted anchors, per descendant BCE)."""
    e = model.encode(b)
    nodes, ws, acc4, acc_down, its = build_states(pairs, b, stories, e, rng, rand_start)
    N = e["N"]
    parent = b["s_parent"]
    has = parent >= 0
    pc = parent.clamp(min=0)
    dn = e["down"]
    big = torch.full((N,), -1e30)
    if has.any():
        cl_all = _seg_lse(dn[has], pc[has], N)
        cl_acc = _seg_lse(torch.where(acc_down[has], dn[has], torch.full((int(has.sum()),), -1e30)), pc[has], N)
    else:
        cl_all = cl_acc = big
    I2 = e["in2"][nodes]
    in_all = torch.logaddexp(torch.logsumexp(I2, 1), cl_all[nodes])
    in_acc = torch.logaddexp(torch.logsumexp(torch.where(acc4[:, [N_HERE, N_ABST]], I2, torch.full_like(I2, -1e30)), 1), cl_acc[nodes])
    lm = torch.log_softmax(e["move"][nodes], 1)
    stay_ok = acc4[:, N_HERE] | acc4[:, N_ABST] | (cl_acc[nodes] > -1e29)
    neg = torch.full_like(lm[:, 0], -1e30)
    acc = torch.logsumexp(torch.stack([torch.where(stay_ok, lm[:, 0] + in_acc - in_all, neg), torch.where(acc4[:, N_NEXT], lm[:, 1], neg),
                                       torch.where(acc4[:, N_PREV], lm[:, 2], neg)], 1), 1)
    loss_p = (-acc * ws).sum() / max(len(pairs), 1)
    ls = []                                              # sentence value: softmax over the story's roots toward the sentence(s) holding an accepted anchor
    for i, p in enumerate(pairs):
        if p["kind"] != "normal":
            continue
        arr = stories[p["skey"]]
        rs, rc = int(e["rstart"][i]), int(e["rcount"][i])
        if rc == 0:
            continue
        rank = {int(arr["sent"][g]): j for j, g in enumerate(_roots_local(arr))}
        tg = sorted({rank[int(arr["sent"][a])] for a, _k, _f in p["tgt"]["anchors"]})
        v = e["sv"][rs:rs + rc]
        ls.append(torch.logsumexp(v, 0) - torch.logsumexp(v[torch.tensor(tg)], 0))
    loss_s = torch.stack(ls).sum() / max(len(pairs), 1) if ls else torch.zeros(())
    A, Nn, It, Y = [], [], [], []
    for i, p in enumerate(pairs):
        if p["kind"] != "normal":
            continue
        o0 = int(b["s_off"][i])
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
    loss_k = torch.zeros(())
    if A:
        lg = model.keep_logits(e, b, torch.tensor(A), torch.tensor(Nn), torch.tensor(It))
        loss_k = F.binary_cross_entropy_with_logits(lg, torch.tensor(Y))
    return loss_p, loss_s, loss_k


# ----------------------------------------------------------------------------- dispatch (the v2 / v3 evaluation code runs v2, v3 and v4 thinkers alike)
class Mind(m3.Mind):
    def __init__(self, topk: int = 1, nstart: int = 1, start: str = "value", budget: int = BUDGET, here_bias: float = 0.0):
        super().__init__(topk)
        self.nstart, self.start, self.budget, self.here_bias = nstart, start, budget, here_bias

    def decide(self, model, b, thr: float = 0.5):
        if isinstance(model, Thinker4):
            return decide(model, b, nstart=self.nstart, start=self.start, budget=self.budget, here_bias=self.here_bias)
        return super().decide(model, b, thr)
