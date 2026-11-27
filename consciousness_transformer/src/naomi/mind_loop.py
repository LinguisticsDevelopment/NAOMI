"""Mind loop: the thinker as ONE learned transition applied in a loop (dev/design/CONSCIOUSNESS_LOOP_PROPOSAL.md, approved).

    (tree_out, state', memory') = T(tree_in, state, memory)

* Tree object (same form in and out): K node slots; per slot a USVS point (unit vector in the sense-table space), a has-point
  probability, an identity code (for nodes with no meaning point: names / lexicon-less words, a per-story random code = a variable),
  a relation distribution, a part-of-speech distribution, a soft parent pointer over the K slots (+ ROOT) and an existence probability.
* Memory: M slots of H numbers, EMPTY (zeros) at the start of every story. Only T writes it: every token of the step (the state token and
  the K tree slots) emits a content-based write (key vs slot address+content, strength, erase vector, add vector). T reads it by
  cross-attention in every layer. No other code touches the memory.
* State: one vector; the halting logit is read from it (PonderNet: halting probability per step, loss = expectation over the halting
  step of the output-tree loss + KL to a geometric prior).
* There is no action menu: no head picks a sentence, a node, keep/drop, or copies anything. Output words come from the decoder
  (nearest meaning among the words of the story and the question, setting (b)); the model only produces trees.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from . import mind_v1 as m1
from . import mind_v2 as m2
from .types import REL_ROOT, Tree

K = 24                 # node slots per tree
MEM = 128              # memory slots
CODE = 24              # identity-code size
NSTEP = 6              # ask-loop cap (safety limit; halting is learned)
EPS = 1e-6


# ----------------------------------------------------------------------------- node forms (numpy, from encoder trees)
def node_key(n) -> Optional[str]:
    """Identity key of a node with no meaning point (a name, or a lemma the lexicon lacks); None for a node with a point.
    Same rule as ``mind_v2.Pool`` (names never use their sense point)."""
    ctx = m1.Ctx.get()
    if n.kind != "name" and ctx.raw(ctx.chosen(n)) is not None:
        return None
    return ("n|" + n.token) if n.kind == "name" else ("w|" + n.lemma)


def node_row(n, table: m2.SenseTable) -> int:
    return 0 if n.kind == "name" else table.idx(m1.Ctx.chosen(n))


def form_of_nodes(nodes: Sequence, parent: Sequence[int], rel_ids: Sequence[int], pos_ids: Sequence[int], table: m2.SenseTable) -> dict:
    """One tree as slot arrays, cut at K nodes (pre-order prefix, so parents precede children). Parent index K = ROOT."""
    n = min(len(nodes), K)
    par = [int(parent[i]) if 0 <= parent[i] < n else K for i in range(n)]
    return {"row": np.array([node_row(nodes[i], table) for i in range(n)], dtype=np.int64),
            "key": [node_key(nodes[i]) for i in range(n)],
            "rel": np.array(rel_ids[:n], dtype=np.int64), "pos": np.array(pos_ids[:n], dtype=np.int64),
            "par": np.array(par, dtype=np.int64), "n": n}


def story_chunks(arr: dict, table: m2.SenseTable) -> List[dict]:
    """The encoder's sentence trees of one story as a list of chunk forms, in reading order; a sentence longer than K nodes is
    read in several consecutive chunks (one T step each); a parent outside the chunk becomes ROOT."""
    out = []
    sent = arr["sent"]
    for s in range(int(arr["n_sent"])):
        idx = np.where(sent == s)[0]
        for a in range(0, len(idx), K):
            ch = idx[a:a + K]
            lo, hi = int(ch[0]), int(ch[-1])
            par = [int(arr["parent"][g]) - lo if lo <= int(arr["parent"][g]) <= hi else -1 for g in ch]
            out.append(form_of_nodes([arr["nodes"][g] for g in ch], par, arr["rel"][ch].tolist(), arr["pos"][ch].tolist(), table))
    return out


# ----------------------------------------------------------------------------- tensors
class CodeBook:
    """Random identity code per key, fixed within one story (salt): a variable with no meaning point."""

    def __init__(self, salt: int):
        self.salt, self.cache = int(salt), {}

    def __call__(self, key: str) -> np.ndarray:
        v = self.cache.get(key)
        if v is None:
            v = np.random.default_rng([self.salt & 0x7FFFFFFF, zlib.crc32(key.encode())]).standard_normal(CODE).astype(np.float32)
            v /= np.linalg.norm(v)
            self.cache[key] = v
        return v


def tree_tensors(forms: Sequence[Optional[dict]], books: Sequence[CodeBook], SM: torch.Tensor, R: int, P: int, known: Optional[Sequence[set]] = None):
    """Dense tensors for a batch of tree forms (None = empty tree). Returns (input form with one-hots, target ids dict)."""
    B = len(forms)
    row = torch.zeros(B, K, dtype=torch.long)
    rel = torch.zeros(B, K, dtype=torch.long)
    pos = torch.zeros(B, K, dtype=torch.long)
    par = torch.full((B, K), K, dtype=torch.long)
    ex = torch.zeros(B, K)
    idc = torch.zeros(B, K, CODE)
    idm = torch.zeros(B, K)
    for b, f in enumerate(forms):
        if f is None or f["n"] == 0:
            continue
        n = f["n"]
        row[b, :n] = torch.from_numpy(f["row"])
        rel[b, :n] = torch.from_numpy(f["rel"])
        pos[b, :n] = torch.from_numpy(f["pos"])
        par[b, :n] = torch.from_numpy(f["par"])
        ex[b, :n] = 1.0
        for i, k in enumerate(f["key"]):
            if k is not None:
                idc[b, i] = torch.from_numpy(books[b](k))
                idm[b, i] = 1.0 if (known is None or k in known[b]) else 0.0
    vec = SM[row]
    has = (row > 0).float()
    tin = {"vec": vec, "has": has, "idc": idc * (1 - has).unsqueeze(-1), "rel": F.one_hot(rel, R).float(), "pos": F.one_hot(pos, P).float(),
           "par": F.one_hot(par, K + 1).float(), "ex": ex}
    tgt = {"vec": vec, "has": has, "idc": idc, "idm": idm, "rel": rel, "pos": pos, "par": par, "ex": ex}
    return tin, tgt


# ----------------------------------------------------------------------------- the model
class Block(nn.Module):
    def __init__(self, H: int, nh: int):
        super().__init__()
        self.nh = nh
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(H), nn.LayerNorm(H), nn.LayerNorm(H)
        self.sa = nn.MultiheadAttention(H, nh, batch_first=True)
        self.rq, self.rk, self.rv, self.ro, self.ra = (nn.Linear(H, H) for _ in range(5))
        self.ff = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(), nn.Linear(2 * H, H))

    def forward(self, x: torch.Tensor, mem: torch.Tensor, addr: torch.Tensor, block_read: bool):
        h = self.ln1(x)
        x = x + self.sa(h, h, h, need_weights=False)[0]
        if not block_read:
            h = self.ln2(x)
            B, T, H = h.shape
            nh, dh = self.nh, H // self.nh
            q = self.rq(h).view(B, T, nh, dh).transpose(1, 2)
            k = (self.rk(mem) + self.ra(addr)[None]).view(B, -1, nh, dh).transpose(1, 2)
            v = self.rv(mem).view(B, -1, nh, dh).transpose(1, 2)
            att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), -1)
            x = x + self.ro((att @ v).transpose(1, 2).reshape(B, T, H))
        return x + self.ff(self.ln3(x))


class Loop(nn.Module):
    """T. One set of weights for reading a sentence and for asking a question."""

    def __init__(self, D: int, R: int, P: int, H: int = 128, nh: int = 4, L: int = 2, mem: int = MEM):
        super().__init__()
        self.D, self.R, self.P, self.H, self.M = D, R, P, H, mem
        self.inp_vec = nn.Linear(D, H, bias=False)
        self.inp_id = nn.Linear(CODE, H, bias=False)
        self.e_rel, self.e_pos = nn.Linear(R, H, bias=False), nn.Linear(P, H, bias=False)
        self.e_par = nn.Linear(H, H)
        self.e_feat = nn.Linear(3, H)
        self.slot = nn.Parameter(torch.randn(K, H) * 0.02)
        self.s0 = nn.Parameter(torch.zeros(H))
        self.addr = nn.Parameter(torch.randn(mem, H) * 0.1)
        self.blocks = nn.ModuleList(Block(H, nh) for _ in range(L))
        self.lnf = nn.LayerNorm(H)
        # output tree heads (per slot)
        self.o_vec, self.o_has, self.o_id = nn.Linear(H, D), nn.Linear(H, 1), nn.Linear(H, CODE)
        self.o_rel, self.o_pos, self.o_ex = nn.Linear(H, R), nn.Linear(H, P), nn.Linear(H, 1)
        self.o_pq, self.o_pk, self.o_root = nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, 1)
        self.halt = nn.Linear(H, 1)
        nn.init.constant_(self.halt.bias, -1.0)
        nn.init.constant_(self.o_ex.bias, -1.0)
        # memory write (every token of the step)
        self.w_k, self.w_g, self.w_e, self.w_a, self.w_b = nn.Linear(H, H), nn.Linear(H, 1), nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, 1)
        self.k_addr, self.k_cont = nn.Linear(H, H), nn.Linear(H, H, bias=False)

    # ---- tree object -> tokens
    def embed(self, t: dict) -> torch.Tensor:
        x = self.inp_vec(t["vec"]) + self.inp_id(t["idc"]) + self.e_rel(t["rel"]) + self.e_pos(t["pos"]) + self.slot
        par = t["par"]
        x = x + self.e_par(par[..., :K] @ x) + self.e_feat(torch.stack([t["has"], par[..., K], t["ex"]], -1))
        return x

    def init_state(self, B: int) -> torch.Tensor:
        return self.s0.unsqueeze(0).expand(B, -1)

    def init_memory(self, B: int) -> torch.Tensor:
        return torch.zeros(B, self.M, self.H)

    def step(self, tin: dict, state: torch.Tensor, mem: torch.Tensor, block_read: bool = False):
        x = torch.cat([state.unsqueeze(1), self.embed(tin)], 1)
        for blk in self.blocks:
            x = blk(x, mem, self.addr, block_read)
        x = self.lnf(x)
        hs, hn = x[:, 0], x[:, 1:]
        q, k = self.o_pq(hn), self.o_pk(hn)
        pl = torch.einsum("bih,bjh->bij", q, k) / math.sqrt(self.H)
        pl = pl.masked_fill(torch.eye(K, dtype=torch.bool), -1e9)
        par = torch.softmax(torch.cat([pl, self.o_root(hn)], -1), -1)
        tout = {"vec": F.normalize(self.o_vec(hn), dim=-1), "has": torch.sigmoid(self.o_has(hn)).squeeze(-1), "idc": self.o_id(hn),
                "rel_logit": self.o_rel(hn), "pos_logit": self.o_pos(hn), "par": par, "ex": torch.sigmoid(self.o_ex(hn)).squeeze(-1)}
        tout["rel"], tout["pos"] = torch.softmax(tout["rel_logit"], -1), torch.softmax(tout["pos_logit"], -1)
        gate = torch.cat([torch.ones(x.size(0), 1), tin["ex"]], 1)
        return tout, hs, self.halt(hs).squeeze(-1), self.write(x, gate, mem)

    def write(self, x: torch.Tensor, gate_in: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        beta = 1 + F.softplus(self.w_b(x))
        slotk = self.k_addr(self.addr)[None] + self.k_cont(mem)
        w = torch.softmax(beta * torch.einsum("bth,bmh->btm", self.w_k(x), slotk) / math.sqrt(self.H), -1)
        g = torch.sigmoid(self.w_g(x)).squeeze(-1) * gate_in
        wg = w * g.unsqueeze(-1)
        erase = torch.einsum("btm,bth->bmh", wg, torch.sigmoid(self.w_e(x))).clamp(0, 1)
        return mem * (1 - erase) + torch.einsum("btm,bth->bmh", wg, torch.tanh(self.w_a(x)))

    # ---- reading a story: one T step per chunk, memory carries across steps
    def read(self, chunks: List[dict], active: Optional[List[torch.Tensor]] = None, state=None, mem=None):
        """chunks[s] = input-form dict for step s over a batch of B stories; active[s] = [B] bool (story has a chunk at step s)."""
        B = chunks[0]["vec"].size(0)
        state = self.init_state(B) if state is None else state
        mem = self.init_memory(B) if mem is None else mem
        for s, tin in enumerate(chunks):
            _o, st2, _h, m2_ = self.step(tin, state, mem)
            if active is None:
                state, mem = st2, m2_
            else:
                a = active[s]
                state, mem = torch.where(a[:, None], st2, state), torch.where(a[:, None, None], m2_, mem)
        return state, mem

    # ---- asking: loop until the learned halt (cap = safety limit)
    def ask(self, tin: dict, state: torch.Tensor, mem: torch.Tensor, nstep: int = NSTEP, block_read: bool = False):
        outs, hl = [], []
        for _ in range(nstep):
            tout, state, h, mem = self.step(tin, state, mem, block_read)
            outs.append(tout)
            hl.append(h)
            tin = {"vec": tout["vec"], "has": tout["has"], "idc": tout["idc"] * (1 - tout["has"]).unsqueeze(-1), "rel": tout["rel"],
                   "pos": tout["pos"], "par": tout["par"], "ex": tout["ex"]}
        return outs, torch.stack(hl, 1)


def halting(hl: torch.Tensor) -> torch.Tensor:
    """PonderNet halting distribution over steps from the halting logits [B, N]: p_n = lam_n prod_{k<n}(1-lam_k); the last step takes the rest."""
    lam = torch.sigmoid(hl)
    surv = torch.cumprod(torch.cat([torch.ones_like(lam[:, :1]), 1 - lam[:, :-1]], 1), 1)
    p = lam * surv
    return torch.cat([p[:, :-1], surv[:, -1:]], 1)


def geometric_prior(N: int, lam: float) -> torch.Tensor:
    q = torch.tensor([lam * (1 - lam) ** n for n in range(N - 1)])
    return torch.cat([q, (1 - lam) ** (N - 1) * torch.ones(1)])


# ----------------------------------------------------------------------------- the differentiable set-matching tree loss
W_VEC, W_REL, W_POS, W_HAS, W_PAR = 2.0, 0.5, 0.5, 0.5, 0.5


def tree_loss(out: dict, tg: dict) -> torch.Tensor:
    """Per-item loss [B] of an output tree against a reference tree (both in slot form).
    Hungarian matching of output slots to reference nodes on (1 - USVS cosine [identity-code cosine for nodes with no point]) + relation CE
    + pos CE + has-point BCE + existence; matched slots: those costs + existence-to-1; unmatched slots: existence-to-0; plus the parent
    pointer CE of matched slots (target parent mapped through the matching; ROOT stays ROOT). Normalised by max(#reference nodes, 3)."""
    B = out["ex"].size(0)
    th, tm = tg["has"], tg["idm"]
    ex = out["ex"].clamp(EPS, 1 - EPS)
    cos = torch.einsum("bid,bjd->bij", out["vec"], tg["vec"])
    idcos = torch.einsum("bic,bjc->bij", F.normalize(out["idc"], dim=-1), F.normalize(tg["idc"], dim=-1))
    cvec = torch.where(th[:, None, :] > 0.5, 1 - cos, tm[:, None, :] * (1 - idcos))
    has = out["has"].clamp(EPS, 1 - EPS)[:, :, None]
    hbce = -(th[:, None, :] * has.log() + (1 - th[:, None, :]) * (1 - has).log())
    lr, lp = F.log_softmax(out["rel_logit"], -1), F.log_softmax(out["pos_logit"], -1)
    rce = -torch.einsum("bir,bjr->bij", lr, F.one_hot(tg["rel"], lr.size(-1)).float())
    pce = -torch.einsum("bip,bjp->bij", lp, F.one_hot(tg["pos"], lp.size(-1)).float())
    cost = W_VEC * cvec + W_REL * rce + W_POS * pce + W_HAS * hbce - ex.log()[:, :, None]
    nt = tg["ex"].sum(1).long()
    Mt = torch.zeros(B, K, K)
    cd = cost.detach().numpy()
    inv, pb, pi, pt = [], [], [], []
    for b in range(B):
        T = int(nt[b])
        inv.append({})
        if T == 0:
            continue
        r, c = linear_sum_assignment(cd[b, :, :T])
        Mt[b, r, c] = 1.0
        mp = {int(j): int(i) for i, j in zip(r, c)}
        inv[b] = mp
        for i, j in zip(r, c):
            tp = int(tg["par"][b, j])
            if tp == K:
                pb.append(b), pi.append(int(i)), pt.append(K)
            elif tp in mp:
                pb.append(b), pi.append(int(i)), pt.append(mp[tp])
    matched = (cost * Mt).sum((1, 2))
    unm = (-(1 - ex).log() * (1 - Mt.sum(2))).sum(1)
    loss = matched + unm
    if pb:
        pb_, pi_, pt_ = torch.tensor(pb), torch.tensor(pi), torch.tensor(pt)
        lpar = -torch.log(out["par"][pb_, pi_, pt_].clamp(min=EPS))
        loss = loss + W_PAR * torch.zeros(B).index_add(0, pb_, lpar)
    return loss / nt.clamp(min=3).float()


def ponder_loss(outs: List[dict], hl: torch.Tensor, tg: dict, lam_prior: float = 0.3, beta: float = 0.05):
    """PonderNet: sum_n p_n L_n + beta KL(p || geometric). Returns (mean loss, per-step mean tree loss [N], halting distribution [B, N])."""
    p = halting(hl)
    Ls = torch.stack([tree_loss(o, tg) for o in outs], 1)
    q = geometric_prior(len(outs), lam_prior)
    kl = (p * (p.clamp(min=EPS).log() - q.log())).sum(1)
    return (p * Ls).sum(1).mean() + beta * kl.mean(), Ls.detach().mean(0), p.detach()


def halt_step(p: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Evaluation halting: the first step at which the cumulative halting probability reaches tau (the last step at the latest)."""
    return (torch.cumsum(p, 1) < tau).sum(1).clamp(max=p.size(1) - 1)


# ----------------------------------------------------------------------------- decoding: nodes -> words (setting (b)) -> tree for the orderer
def decode_tree(tout: dict, b: int, pool: m2.Pool, book: CodeBook, rel_names: Sequence[str], pos_names: Sequence[str], thr: float) -> Optional[Tree]:
    """Output slots with existence > thr become nodes; each node's word is the pool entry with the nearest USVS point (a node with no point:
    the pool entry with no point whose identity code is nearest); relation = the node's own relation distribution argmax; structure = the
    parent pointers (cycles broken at ROOT). Returns a Tree or None (empty = abstain)."""
    ex = tout["ex"][b].numpy()
    ids = [i for i in range(K) if ex[i] > thr]
    if not ids or not pool.ents:
        return None
    ctx = m1.Ctx.get()
    ent_key = [None if e["vec"] is not None else (("n|" + e["node"].token) if e["node"].kind == "name" else ("w|" + e["node"].lemma)) for e in pool.ents]
    nodes = {}
    for i in ids:
        pr = tout["rel"][b, i].numpy().copy()
        pr[0] = -1
        rel = rel_names[int(pr.argmax()) - 1] if int(pr.argmax()) >= 1 and int(pr.argmax()) - 1 < len(rel_names) else REL_ROOT
        pp = tout["pos"][b, i].numpy().copy()
        pp[0] = -1
        pos = pos_names[int(pp.argmax()) - 1] if int(pp.argmax()) >= 1 and int(pp.argmax()) - 1 < len(pos_names) else ""
        if float(tout["has"][b, i]) >= 0.5 and pool.has.any():
            sims = np.where(pool.has, pool.V @ tout["vec"][b, i].numpy(), -9.0)
        else:
            c = tout["idc"][b, i].numpy()
            c = c / max(float(np.linalg.norm(c)), 1e-8)
            sims = np.array([float(book(k) @ c) if k is not None else -9.0 for k in ent_key])
            if sims.max() <= -9.0:
                sims = np.where(pool.has, pool.V @ tout["vec"][b, i].numpy(), -9.0) if pool.has.any() else np.zeros(len(pool.ents))
        cand = np.where(sims >= sims.max() - 1e-4)[0]
        probe = replace(pool.ents[int(cand[0])]["node"], pos=pos, rel=rel, subtypes=[])
        best = min(cand, key=lambda e: (-pool.tag_agreement(probe, pool.ents[e]["node"]), -pool.ents[e]["count"], pool.ents[e]["node"].token))
        nodes[i] = replace(pool.ents[int(best)]["node"], rel=rel, token_index=i, children=[])
    par = {}
    for i in ids:
        pr = tout["par"][b, i].numpy()
        cand = [j for j in ids if j != i]
        par[i] = max(cand, key=lambda j: pr[j]) if cand and max(pr[j] for j in cand) > pr[K] else None
    for i in ids:                                          # break cycles at ROOT
        seen, cur = [], i
        while cur is not None and cur not in seen:
            seen.append(cur)
            cur = par[cur]
        if cur is not None:
            par[cur] = None
    roots = [i for i in ids if par[i] is None]
    root = max(roots, key=lambda i: ex[i])
    for i in roots:
        if i != root:
            par[i] = root
    for i in ids:
        if i != root:
            nodes[par[i]].children.append(nodes[i])
    for n in nodes.values():
        n.children.sort(key=lambda c: c.token_index)
    nodes[root].rel = REL_ROOT
    return Tree(sentence="", root=nodes[root])
