"""Mind v5: the thinker completes the question (dev/design/CONSCIOUSNESS_V4_PROPOSAL.md option A, owner amendment binding).

Output = the question tree with one of its nodes replaced by a FILLER tree built from memory. The thinker learns
  * the filler: node by node, each node a COPY of a memory node of the whole story (across sentences) or of a question node,
    attached under an earlier filler node; STOP ends it; ABSTAIN at step 0 leaves the question unfilled;
  * which question node to replace. NOTHING in this file, in data prep or in the targets names, finds or scores the
    question word / unknown slot: the thinker sees every question node's USVS point (one of its dimensions is whatever
    the lexicon says) plus tag / relation features and chooses; the choice is trained by an expected-reward loss over the
    placements (reward = placement-free frame share, see ``frame_rewards``). No code here reads the lexicon's unknown axis.
Data, memory, encoder pieces, decoders and ROUGE are Mind v2/v3's (``mind_v2``, ``mind_v3``).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import mind_v1 as m1
from . import mind_v2 as m2
from . import mind_v3 as m3
from .mind_v2 import NB, NQ
from .tree import walk
from .types import REL_ROOT, Tree

TMAX = 16                # longest filler (imitation targets are cut to this many nodes)
NEGINF = -1e9


# ----------------------------------------------------------------------------- features without any slot knowledge
def refeature(pf: dict, arr: dict, qn: Sequence, ctx) -> dict:
    """Question-dependent numeric features for one (story, question), recomputed from the story arrays and the question's
    node list. Same quantities as ``mind_v2.pair_features`` but nothing is excluded or flagged for any question node: the
    "content" set is every question node whose sense is not a function-class sense (a lexicon property), the wh index is -1
    and the column that used to carry the unknown-axis value is zero. FIND (columns 5, 6) is copied from the old pair."""
    N, Q = len(arr["parent"]), len(qn)
    sb = np.zeros((N, NB), dtype=np.float32)
    qb = np.zeros((Q, NQ), dtype=np.float32)
    qkeys = [(n.kind, n.lemma) for n in qn]
    qcont = [j for j, n in enumerate(qn) if not ctx.is_function(n)]
    itf = np.zeros(m2.N_ITEM, dtype=np.float32)
    if N > 0 and Q > 0:
        lem = [(n.kind, n.lemma) for n in arr["nodes"]]
        S = m1.sim_matrix(ctx, list(arr["nodes"]), list(qn))
        cols: Dict = {}
        for j, k in enumerate(qkeys):
            cols.setdefault(k, []).append(j)
        ident = np.zeros((N, Q), dtype=np.float32)
        for g, k in enumerate(lem):
            for j in cols.get(k, ()):
                ident[g, j] = 1.0
        sb[:, 0] = ident.max(1)
        if qcont:
            sb[:, 1] = ident[:, qcont].max(1)
            sb[:, 2] = S[:, qcont].max(1)
        n_sent = max(1, int(arr["n_sent"]))
        sent = arr["sent"]
        df = np.array([len(set(sent[ident[:, j] > 0].tolist())) for j in range(Q)], dtype=np.float32)
        idw = np.log((n_sent + 1.0) / (df + 1.0))
        sb[:, 3] = (ident * idw[None]).max(1) / max(float(idw.max()), 1e-8)
        wq = np.zeros(Q, dtype=np.float32)
        if qcont:
            wq[qcont] = idw[qcont]
        wq = wq / max(float(wq.sum()), 1e-8)
        cov = np.zeros(n_sent, dtype=np.float32)
        for s in range(n_sent):
            rows = np.where(sent == s)[0]
            if len(rows):
                cov[s] = float((wq * ident[rows].max(0)).sum())
        sb[:, 4] = cov[sent]
        sb[:, 5:7] = pf["sb"][:, 5:7]
        inmem = set(lem)
        qb[:, 0] = [float(k in inmem) for k in qkeys]
        qb[:, 1] = S.max(0)
        itf = np.array([n_sent / 20.0, float(cov.max()), float(pf["sb"][:, 5].max()), len(qcont) / 10.0], dtype=np.float32)
    out = {k: v for k, v in pf.items() if k not in ("tgt",)}
    out.update(sb=sb, qb=qb, itf=itf, wh=-1)
    return out


# ----------------------------------------------------------------------------- targets
def preorder(arr: dict, a: int, kept: set) -> List[int]:
    out, stack = [], [a]
    while stack:
        g = stack.pop()
        out.append(g)
        stack += [c for c in reversed(arr["kids"][g]) if c in kept]
    return out


def frame_rewards(parent: np.ndarray, supp: np.ndarray) -> np.ndarray:
    """Placement-free frame reward for replacing question node j (with its subtree): 1 minus the share of SUPPORTED question
    nodes (supp = the node's lemma occurs in the answer-bearing memory sentence, i.e. "memory already says it") that the
    replacement removes. Uses only the question tree shape and the support vector; no slot label, no word, no axis."""
    Q = len(parent)
    tot = float(supp.sum())
    sub = np.eye(Q, dtype=np.float32)
    for i in range(Q - 1, -1, -1):          # pre-order: parents come before children
        if parent[i] >= 0:
            sub[parent[i]] += sub[i]
    return (1.0 - (sub @ supp) / max(tot, 1e-6)).astype(np.float32)


def support_vector(arr: dict, ans_sent: Optional[int], qn: Sequence) -> Optional[np.ndarray]:
    """1 for every question node whose (kind, lemma) occurs in the answer-bearing memory sentence (the sentence that holds the
    most reference lemmas): "memory already says it". None when no sentence holds any reference lemma."""
    if ans_sent is None or not len(qn):
        return None
    keys = {(n.kind, n.lemma) for n, sn in zip(arr["nodes"], arr["sent"]) if int(sn) == int(ans_sent)}
    return np.array([float((n.kind, n.lemma) in keys) for n in qn], dtype=np.float32)


def build_targets5(arr: dict, tgt: Optional[dict], qn: Sequence, qparent: np.ndarray) -> Optional[dict]:
    """Imitation target for the filler (the reference nodes aligned to memory by lemma, as v2-v4) and the frame rewards.
    None for unreachable items (counted by the caller, not tuned on)."""
    if not tgt or not tgt["anchors"] or tgt["best_f1"] < m2.ACCEPT_F1:
        return None
    a, kept, _f = tgt["anchors"][0]
    order = preorder(arr, int(a), {int(x) for x in kept})[:TMAX]
    pos = {g: i for i, g in enumerate(order)}
    par = [pos.get(int(arr["parent"][g]), -1) for g in order]
    acc = []
    for t in range(len(order) + 1):
        chosen = set(order[:t])
        if t == 0:
            acc.append(sorted({int(x[0]) for x in tgt["anchors"]}))
        elif t < len(order):
            acc.append([g for g in order if g not in chosen and int(arr["parent"][g]) in chosen])
        else:
            acc.append([])
    supp = support_vector(arr, tgt["ans_sent"], qn)
    rew = frame_rewards(qparent, supp) if (supp is not None and supp.sum() > 0) else None
    return {"seq": order, "par": par, "acc": acc, "rew": rew, "ref_n": tgt["ref_n"]}


# ----------------------------------------------------------------------------- the model
class Thinker5(m3.Thinker3):
    def __init__(self, D: int, n_rel: int, n_pos: int, P: int = 24, H: int = 64, drop: float = 0.2):
        super().__init__(D, n_rel, n_pos, P=P, H=H, drop=drop, inv=True, ctx=True)
        self.keyS = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, H))
        self.keyQ = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))
        self.wq2, self.wk2, self.wv2 = nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, H)
        self.q_up = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, H))
        self.start = nn.Parameter(torch.zeros(H))
        self.init_h = nn.Linear(H, H)
        self.cell = nn.GRUCell(H, H)
        self.qry = nn.Linear(H, H)
        self.feat = nn.Linear(3, 1)
        self.abst_head = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, 1))
        self.stop_head = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, 1))
        self.par_q, self.par_k = nn.Linear(H, H), nn.Linear(H, H)
        self.w_pm = nn.Parameter(torch.tensor(2.0))
        self.rep = nn.Sequential(nn.Linear(3 * H, H), nn.GELU(), nn.Linear(H, 1))

    # ---- keys over [story nodes | question nodes] for every item
    def pack(self, b: dict, e: dict) -> dict:
        assert bool((b["q_wh"] < 0).all()), "v5 never receives a slot index"
        B, H = b["B"], self.H
        so = b["s_off"]
        nper = so[1:] - so[:-1]
        Nmax = max(int(nper.max()), 1)
        qper = torch.bincount(b["q_item"], minlength=B)
        Qmax = max(int(qper.max()), 1)
        qoff = torch.cumsum(qper, 0) - qper
        hq = e["hq"]
        hs, hsent = e["hs"], e["hsent"]
        Ntot = hs.size(0)
        K = torch.zeros(B, Nmax + Qmax, H)
        parpad = torch.full((B, Nmax), -1, dtype=torch.long)
        if Ntot:
            sl = torch.arange(Ntot) - so[:-1][b["s_item"]]
            ks = self.keyS(torch.cat([hs, hsent], 1))
            K = K.index_put((b["s_item"], sl), ks)
            gp = b["s_parent"]
            lp = torch.where(gp >= 0, gp - so[:-1][b["s_item"]], torch.full_like(gp, -1))
            parpad = parpad.index_put((b["s_item"], sl), lp)
            att = (self.wq2(hq) @ self.wk2(hs).T) / math.sqrt(H)
            att = att.masked_fill(b["q_item"][:, None] != b["s_item"][None, :], NEGINF)
            has = (nper[b["q_item"]] > 0).float().unsqueeze(1)
            cq = (torch.softmax(att, 1) @ self.wv2(hs)) * has
        else:
            cq = torch.zeros_like(hq)
        hq2 = hq + self.q_up(torch.cat([hq, cq], 1))
        ql = torch.arange(hq.size(0)) - qoff[b["q_item"]]
        kq = self.keyQ(hq2)
        K = K.index_put((b["q_item"], Nmax + ql), kq)
        validS = torch.arange(Nmax)[None, :] < nper[:, None]
        validQ = torch.arange(Qmax)[None, :] < qper[:, None]
        valid = torch.cat([validS, validQ], 1)
        isq = torch.cat([torch.zeros(B, Nmax), torch.ones(B, Qmax)], 1)
        isroot = torch.cat([(parpad < 0).float(), torch.zeros(B, Qmax)], 1)
        qm = e["qvec"][:, :H]
        return {"K": K, "valid": valid, "Nmax": Nmax, "Qmax": Qmax, "nper": nper, "qper": qper, "parpad": parpad, "isq": isq,
                "isroot": isroot, "qm": qm, "M": Nmax + Qmax, "kq": K[:, Nmax:], "validQ": validQ}

    def _logits(self, P: dict, h: torch.Tensor, chosen: torch.Tensor, t: int):
        B, M = P["K"].shape[:2]
        q = self.qry(h)
        lg = torch.einsum("bmh,bh->bm", P["K"], q) / math.sqrt(self.H)
        pc = torch.zeros(B, M)
        N = P["Nmax"]
        par = P["parpad"]
        pc[:, :N] = (chosen.gather(1, par.clamp(min=0)[:, :N]) & (par >= 0)).float()
        feats = torch.stack([pc, P["isroot"], P["isq"]], 2)
        lg = lg + self.feat(feats).squeeze(-1)
        lg = lg.masked_fill(~P["valid"] | chosen, NEGINF)
        spec = (self.abst_head if t == 0 else self.stop_head)(torch.cat([h, P["qm"]], 1))
        return torch.cat([lg, spec], 1)

    def _parent_logits(self, P: dict, prev: torch.Tensor, Kseq: torch.Tensor, seqpos: torch.Tensor, newpos: torch.Tensor) -> torch.Tensor:
        """Logits over earlier filler steps for attaching the node at ``newpos`` (feature: memory parent edge)."""
        lg = torch.einsum("bsh,bh->bs", self.par_k(Kseq), self.par_q(prev)) / math.sqrt(self.H)
        par_new = P["parpad"].gather(1, newpos.clamp(max=P["Nmax"] - 1).unsqueeze(1)).squeeze(1)
        par_new = torch.where(newpos < P["Nmax"], par_new, torch.full_like(par_new, -2))
        return lg + self.w_pm * (seqpos == par_new.unsqueeze(1)).float()

    def _replace_logits(self, P: dict, chosen: torch.Tensor) -> torch.Tensor:
        cnt = chosen.float().sum(1, keepdim=True).clamp(min=1)
        fill = (P["K"] * chosen.unsqueeze(2).float()).sum(1) / cnt
        kq = P["kq"]
        lg = self.rep(torch.cat([kq, fill.unsqueeze(1).expand_as(kq), P["qm"].unsqueeze(1).expand_as(kq)], 2)).squeeze(-1)
        return lg.masked_fill(~P["validQ"], NEGINF)

    # ---- teacher-forced training loss
    def tf_loss(self, b: dict, pairs: Sequence[dict], w_rep: float = 1.0):
        """pairs carry ``tg`` (imitation target; None = abstain target). Returns (pointer loss, parent loss, replace loss, stats)."""
        e = self.encode(b)
        P = self.pack(b, e)
        B, M, H = len(pairs), P["M"], self.H
        lens = [len(p["tg"]["seq"]) if p.get("tg") else 0 for p in pairs]
        T = max(lens)
        h = torch.tanh(self.init_h(P["qm"]))
        prev = self.start.unsqueeze(0).expand(B, H)
        chosen = torch.zeros(B, M, dtype=torch.bool)
        ar = torch.arange(B)
        lp_tot, n_ptr = torch.zeros(()), 0
        par_tot, n_par = torch.zeros(()), 0
        Kseq, seqpos_list = [], []
        for t in range(T + 1):
            h = self.cell(prev, h)
            full = self._logits(P, h, chosen, t)
            tm = torch.zeros(B, M + 1, dtype=torch.bool)
            act = torch.zeros(B, dtype=torch.bool)
            for i, p in enumerate(pairs):
                tg = p.get("tg")
                if t > lens[i]:
                    continue
                act[i] = True
                if tg is None or t == lens[i]:
                    tm[i, M] = True
                else:
                    tm[i, tg["acc"][t]] = True
            if act.any():
                lse = torch.logsumexp(full, 1)
                tgt = torch.logsumexp(full.masked_fill(~tm, NEGINF), 1)
                lp_tot = lp_tot + ((lse - tgt) * act.float()).sum()
                n_ptr += int(act.sum())
            if t < T:
                idx = torch.tensor([p["tg"]["seq"][t] if (p.get("tg") and t < lens[i]) else 0 for i, p in enumerate(pairs)])
                adv = torch.tensor([t < lens[i] for i in range(B)])
                prev_new = P["K"][ar, idx]
                if t >= 1 and adv.any():
                    Ks = torch.stack(Kseq, 1)
                    sp = torch.stack(seqpos_list, 1)
                    pl = self._parent_logits(P, prev_new, Ks, sp, idx)
                    tpar = torch.tensor([p["tg"]["par"][t] if (p.get("tg") and t < lens[i]) else 0 for i, p in enumerate(pairs)])
                    ce = F.cross_entropy(pl, tpar.clamp(min=0), reduction="none")
                    par_tot = par_tot + (ce * adv.float()).sum()
                    n_par += int(adv.sum())
                chosen = chosen.clone()
                chosen[ar[adv], idx[adv]] = True
                prev = torch.where(adv.unsqueeze(1), prev_new, prev)
                Kseq.append(prev_new)
                seqpos_list.append(idx)
        # ---- replace-choice: expected frame reward over the placements (never a slot label)
        rl = self._replace_logits(P, chosen)
        pj = torch.softmax(rl, 1)
        R = torch.zeros(B, P["Qmax"])
        has = torch.zeros(B)
        for i, p in enumerate(pairs):
            tg = p.get("tg")
            if tg is not None and tg["rew"] is not None:
                r = torch.from_numpy(tg["rew"])
                R[i, :len(r)] = r
                has[i] = 1.0
        loss_rep = -((pj * R).sum(1) * has).sum() / has.sum().clamp(min=1)
        ent = -(pj * torch.log(pj.clamp(min=1e-9))).sum(1)
        loss_rep = loss_rep - 0.01 * (ent * has).sum() / has.sum().clamp(min=1)
        return lp_tot / max(n_ptr, 1), par_tot / max(n_par, 1), loss_rep, {"exp_frame": float(((pj * R).sum(1) * has).sum() / has.sum().clamp(min=1))}

    # ---- greedy decode
    @torch.no_grad()
    def decode(self, b: dict, max_steps: int = TMAX) -> List[dict]:
        self.eval()
        e = self.encode(b)
        P = self.pack(b, e)
        B, M, H = b["B"], P["M"], self.H
        h = torch.tanh(self.init_h(P["qm"]))
        prev = self.start.unsqueeze(0).expand(B, H)
        chosen = torch.zeros(B, M, dtype=torch.bool)
        alive = torch.ones(B, dtype=torch.bool)
        ar = torch.arange(B)
        res = [{"abstain": False, "pos": [], "par": [], "p_abs": 0.0} for _ in range(B)]
        Kseq, seqpos_list = [], []
        for t in range(max_steps + 1):
            h = self.cell(prev, h)
            full = self._logits(P, h, chosen, t)
            pr = torch.softmax(full, 1)
            if t == 0:
                for i in range(B):
                    res[i]["p_abs"] = float(pr[i, M])
            pred = full.argmax(1)
            stop = (pred == M) | (t == max_steps) | (P["valid"].sum(1) == 0)
            for i in range(B):
                if alive[i] and stop[i]:
                    res[i]["abstain"] = (t == 0)
            alive = alive & ~stop
            if not alive.any():
                break
            idx = torch.where(alive, pred.clamp(max=M - 1), torch.zeros_like(pred))
            prev_new = P["K"][ar, idx]
            pidx = torch.full((B,), -1, dtype=torch.long)
            if t >= 1:
                pl = self._parent_logits(P, prev_new, torch.stack(Kseq, 1), torch.stack(seqpos_list, 1), idx)
                pidx = pl.argmax(1)
            for i in range(B):
                if alive[i]:
                    res[i]["pos"].append(int(idx[i]))
                    res[i]["par"].append(int(pidx[i]))
            chosen = chosen.clone()
            chosen[ar[alive], idx[alive]] = True
            prev = torch.where(alive.unsqueeze(1), prev_new, prev)
            Kseq.append(prev_new)
            seqpos_list.append(idx)
        rl = self._replace_logits(P, chosen)
        for i in range(B):
            q = int(P["qper"][i])
            res[i]["j_logits"] = rl[i, :q].tolist()
            res[i]["j"] = int(rl[i, :q].argmax()) if q else -1
            res[i]["Nmax"] = P["Nmax"]
        return res


# ----------------------------------------------------------------------------- output trees (with provenance)
def source_of(pos: int, Nmax: int):
    return ("m", pos) if pos < Nmax else ("q", pos - Nmax)


def filler_tree(arr: dict, qn: Sequence, srcs: Sequence, par: Sequence[int]):
    """The filler as a Tree (root relation ROOT) plus its provenance list. ``srcs`` = [("m", gid) | ("q", j)], ``par`` = index of
    the earlier filler node each attaches to (-1 for the first). Nodes are COPIES (token, lemma, senses, tags); token indices are
    renumbered 0.. in memory order (sentence, position), question copies last."""
    if not srcs:
        return None, []
    copies, keys = [], []
    for kind, i in srcs:
        n = arr["nodes"][i] if kind == "m" else qn[i]
        copies.append(replace(n, children=[]))
        keys.append((0, int(arr["sent"][i]), int(arr["tidx"][i])) if kind == "m" else (1, 0, int(qn[i].token_index)))
    order = sorted(range(len(copies)), key=lambda k: keys[k])
    for r, k in enumerate(order):
        copies[k].token_index = r
    for k in range(1, len(copies)):
        p = par[k] if 0 <= par[k] < k else 0
        copies[p].children.append(copies[k])
    for n in copies:
        n.children.sort(key=lambda c: c.token_index)
    copies[0].rel = REL_ROOT
    return Tree(sentence="", root=copies[0]), [tuple(s) for s in srcs]


def assemble(qtree: Tree, qn: Sequence, j: int, filler: Tree, base: int = 1000) -> Tree:
    """The question tree with node j (and its subtree) replaced by the filler (the filler's root takes j's relation)."""
    target = qn[j]

    def renum(n, off):
        return replace(n, token_index=None if n.token_index is None else n.token_index + off, children=[renum(c, off) for c in n.children])

    def go(n):
        if n is target:
            f = renum(filler.root, base)
            return replace(f, rel=n.rel)
        return replace(n, children=[go(c) for c in n.children])
    return Tree(sentence=qtree.sentence, root=go(qtree.root))


def frame_parts(qparent: np.ndarray, j: int, supp: Optional[np.ndarray]):
    """(share of question nodes kept, share of supported question nodes kept) when node j's subtree is replaced."""
    Q = len(qparent)
    rem = {j}
    for i in range(Q):
        a = i
        while a >= 0:
            if a == j:
                rem.add(i)
                break
            a = int(qparent[a])
    fa = 1.0 - len(rem) / Q
    fs = None
    if supp is not None and supp.sum() > 0:
        fs = 1.0 - float(sum(supp[i] for i in rem)) / float(supp.sum())
    return fa, fs


def lemma_f1(pred_keys: Sequence, ref: Counter) -> float:
    left = Counter(ref)
    ov = 0
    for k in pred_keys:
        if left[k] > 0:
            left[k] -= 1
            ov += 1
    n = sum(ref.values())
    if not pred_keys or n == 0 or ov == 0:
        return 0.0
    p, r = ov / len(pred_keys), ov / n
    return 2 * p * r / (p + r)
