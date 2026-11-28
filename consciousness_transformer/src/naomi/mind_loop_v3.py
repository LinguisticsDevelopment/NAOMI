"""Mind loop v3 (dev/reports/MIND_LOOP_V3.md; owner amendment 2026-11-28 in dev/design/CONSCIOUSNESS_LOOP_PROPOSAL.md).

Same model (``mind_loop.Loop``; v1/v2 checkpoints load), same tree form, loop, halt schedule and counterfactual items as v2. ONLY the training grade changes:

* Candidate pool per item = the words of the story memory + the question (identity = USVS sense row, or identity key for names / lexicon-less words).
  A candidate with a USVS point is represented by that point, one without by its identity code (the same two representations the decoder uses).
* For each output slot i: choice distribution over the pool, s_i = softmax_c( cos(slot meaning, candidate) / TAU ). The slot's meaning is ``vec`` against
  point candidates and ``idc`` (normalised) against code candidates. TAU = 0.1 is FIXED (declared, not learned).
* "No node": the existence head. The no-node logit is  LSE_c(logit_c) - e  (e = existence pre-activation), which makes
  P(no node) = 1 - sigmoid(e) = 1 - ex exactly and P(candidate c) = ex * s_c, so the decoder's existence threshold is the same quantity that is trained.
* Asymmetric reward, fixed before training (owner): correct candidate +1, no node +0.25, any wrong candidate -1.
  Expected reward of a slot = sum_c P(c) r(c) + P(no node) 0.25. Loss = - expected reward (summed over slots) + the v1 structural terms
  (relation CE, pos CE, has-point BCE, parent CE, matched slots only). No meaning regression: nothing pulls a slot toward an average.
* Slots are matched to reference nodes by Hungarian on the SAME costs as v1. A matched slot's correct choice is its reference node's candidate; an
  unmatched slot's correct choice is "no node". A reference node whose identity is NOT in the pool cannot be output by the decoder: it is EXCLUDED
  from the reward (its matched slot gets the structural terms only) and counted (stats ``unreach`` / ``ref``).
* Items with an empty reference (wrong story / empty memory) have no reference nodes: every slot's correct choice is "no node".
* ``commit`` warm-up (same epochs as the halt warm-up by default): the existence head is bypassed (ex taken as 1 for matched reachable slots,
  unmatched slots contribute no reward term, no gradient to the existence head) so the choice is trained before silence can be learned.
  After the warm-up the loss is the pure expected reward.
* ``decouple`` (variant v3b, declared deviation, see report): the plain loss silenced the model (the choice gradient is scaled by ex, ex falls to 0 while the
  choice is still weak, then nothing can recover). Decoupled form: the choice heads get the expected reward GIVEN that the slot answers (the same gradient
  with the factor ex dropped, a per-slot rescale of the same direction), the existence head gets the true expected reward with the choice held fixed
  (s detached): it answers where P(correct | answers) > 0.625 and stays silent elsewhere. Validation always reports the pure expected-reward loss.
T never writes memory; there is no copy/pointer head: the pool is used only to grade (and by the decoder, unchanged).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from . import mind_loop as ML
from . import mind_loop_v2 as V2

TAU = 0.1                       # fixed similarity temperature
R_CORRECT, R_NONE, R_WRONG = 1.0, 0.25, -1.0
W_VEC, W_REL, W_POS, W_HAS, W_PAR = ML.W_VEC, ML.W_REL, ML.W_POS, ML.W_HAS, ML.W_PAR
EPS = ML.EPS


def reward_table(n_cand: int, correct: int = -1) -> torch.Tensor:
    """Reward of each choice [candidates..., no node]. correct = candidate index, or -1 when the right choice is 'no node'."""
    r = torch.full((n_cand + 1,), R_WRONG)
    r[n_cand] = R_NONE
    if correct >= 0:
        r[correct] = R_CORRECT
    return r


def expected_reward(p_cand: torch.Tensor, p_none: torch.Tensor, correct) -> torch.Tensor:
    """p_cand [..., C] (mass on candidates), p_none [...]; correct: LongTensor [...] candidate index of the correct choice (-1 = none correct)."""
    C = p_cand.size(-1)
    onehot = F.one_hot(correct.clamp(min=0), C).float() * (correct >= 0).unsqueeze(-1)
    r = R_WRONG + (R_CORRECT - R_WRONG) * onehot
    return (p_cand * r).sum(-1) + R_NONE * p_none


# ----------------------------------------------------------------------------- candidate pool
def build_pool(mems: Sequence[dict], qs: Sequence[dict], books: Sequence[ML.CodeBook], SM: torch.Tensor, refs: Sequence[dict] = ()):
    """Pool tensors for a batch. mems[b]/qs[b] are stored-memory / question forms (None = none); refs[b] reference form or None.
    -> dict(vec [B,C,D], code [B,C,CODE], pt [B,C] bool, valid [B,C] bool, tidx [B,K] long: pool index of each reference node, -1 = absent / unreachable)."""
    B = len(qs)
    rows, keys, index = [], [], []
    for b in range(B):
        ids, r, k = {}, [], []
        for f in (mems[b], qs[b]):
            if f is None:
                continue
            for row, key in zip(np.asarray(f["row"][:f["n"]]).tolist(), list(f["key"])[:f["n"]]):
                i = V2.identity(row, key)
                if i is not None and i not in ids:
                    ids[i] = len(r)
                    r.append(int(row))
                    k.append(key)
        rows.append(r), keys.append(k), index.append(ids)
    C = max([len(r) for r in rows] + [1])
    row = torch.zeros(B, C, dtype=torch.long)
    code = torch.zeros(B, C, ML.CODE)
    valid = torch.zeros(B, C, dtype=torch.bool)
    for b in range(B):
        n = len(rows[b])
        if n:
            row[b, :n] = torch.tensor(rows[b])
            valid[b, :n] = True
            for c, key in enumerate(keys[b]):
                if key is not None and rows[b][c] == 0:
                    code[b, c] = torch.from_numpy(books[b](key))
    tidx = torch.full((B, ML.K), -1, dtype=torch.long)
    for b in range(B):
        f = refs[b] if refs else None
        if f is None:
            continue
        for j in range(min(f["n"], ML.K)):
            i = V2.identity(f["row"][j], f["key"][j])
            tidx[b, j] = index[b].get(i, -1) if i is not None else -1
    return {"vec": SM[row], "code": code, "pt": row > 0, "valid": valid, "tidx": tidx}


def choice_logits(out: dict, pool: dict) -> torch.Tensor:
    """[B,K,C] similarity logits (cosine / TAU) of each output slot's meaning to each candidate; invalid candidates = -1e4."""
    cv = torch.einsum("bkd,bcd->bkc", out["vec"], pool["vec"])
    cc = torch.einsum("bkq,bcq->bkc", F.normalize(out["idc"], dim=-1), pool["code"])
    lg = torch.where(pool["pt"][:, None, :], cv, cc) / TAU
    return lg.masked_fill(~pool["valid"][:, None, :], -1e4)


# ----------------------------------------------------------------------------- the grade
def tree_loss_v3(out: dict, tg: dict, pool: dict, commit: bool = False, decouple: bool = False):
    """Per-item loss [B] and stats. loss = ( -sum_slots expected reward + structural terms of matched slots ) / max(#reference nodes, 3)."""
    B = out["ex"].size(0)
    K = ML.K
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
    # matching cost: identical to v1 (Hungarian is on detached values)
    cost = W_VEC * cvec + W_REL * rce + W_POS * pce + W_HAS * hbce - ex.log()[:, :, None]
    nt = tg["ex"].sum(1).long()
    Mt = torch.zeros(B, K, K)
    cd = cost.detach().numpy()
    pb, pi, pt = [], [], []
    for b in range(B):
        T = int(nt[b])
        if T == 0:
            continue
        r, c = linear_sum_assignment(cd[b, :, :T])
        Mt[b, r, c] = 1.0
        mp = {int(j): int(i) for i, j in zip(r, c)}
        for i, j in zip(r, c):
            tp = int(tg["par"][b, j])
            if tp == K:
                pb.append(b), pi.append(int(i)), pt.append(K)
            elif tp in mp:
                pb.append(b), pi.append(int(i)), pt.append(mp[tp])
    struct = ((W_REL * rce + W_POS * pce + W_HAS * hbce) * Mt).sum((1, 2))
    if pb:
        pb_, pi_, pt_ = torch.tensor(pb), torch.tensor(pi), torch.tensor(pt)
        lpar = -torch.log(out["par"][pb_, pi_, pt_].clamp(min=EPS))
        struct = struct + W_PAR * torch.zeros(B).index_add(0, pb_, lpar)
    # ---- the choice and its expected reward
    lg = choice_logits(out, pool)
    s = torch.softmax(lg, -1)                                               # [B,K,C] choice among candidates (given a node is output)
    slot_t = torch.full((B, K), -2, dtype=torch.long)                       # -2: matched but unreachable (excluded); -1: correct = no node; >=0 candidate
    slot_t[:] = -1
    mt_idx = Mt.argmax(2)                                                   # reference node matched to each slot
    matched = Mt.sum(2) > 0
    tcand = torch.gather(pool["tidx"], 1, mt_idx)                           # candidate index of the matched reference (-1 = unreachable)
    slot_t = torch.where(matched, torch.where(tcand >= 0, tcand, torch.full_like(tcand, -2)), slot_t)
    reach = slot_t >= 0
    excl = slot_t == -2
    s_corr = torch.gather(s, 2, slot_t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    if decouple and not commit:                                             # ex-normalised gradient of the SAME expected reward (see module doc, 'decoupled')
        r_choice = (2 * s_corr - 1) * reach                                 # choice heads: reward given that the slot answers (the ex factor of the gradient dropped)
        sd = s_corr.detach()
        r_match = ex * (R_CORRECT * sd + R_WRONG * (1 - sd)) + R_NONE * (1 - ex)
        r_none = ex * R_WRONG + R_NONE * (1 - ex)
        R = r_choice + torch.where(reach, r_match, torch.where(excl, torch.zeros_like(ex), r_none))   # existence head: true expected reward, choice held fixed
    elif commit:                                                            # existence bypassed: reward of committing, matched reachable slots only
        R = (2 * s_corr - 1) * reach
    else:
        r_match = ex * (R_CORRECT * s_corr + R_WRONG * (1 - s_corr)) + R_NONE * (1 - ex)
        r_none = ex * R_WRONG + R_NONE * (1 - ex)
        R = torch.where(reach, r_match, torch.where(excl, torch.zeros_like(ex), r_none))
    loss = (-R.sum(1) + struct) / nt.clamp(min=3).float()
    n_reach = reach.float().sum()
    stats = {"ref": nt.sum().float(), "unreach": excl.float().sum(), "reward_matched": (R * reach).sum() / n_reach.clamp(min=1), "s_corr": (s_corr * reach).sum() / n_reach.clamp(min=1),
             "ex_mean": ex.mean(), "ex_matched": (ex * matched).sum() / matched.float().sum().clamp(min=1), "ex_unmatched": (ex * ~matched).sum() / (~matched).float().sum().clamp(min=1),
             "n_slots_reach": n_reach}
    return loss, {k: v.detach() for k, v in stats.items()}


def ponder_loss_v3(outs: List[dict], hl: torch.Tensor, tg: dict, pool: dict, lam: float = V2.LAM_PRIOR, beta: float = 0.3, warm: bool = False, commit: bool = False, decouple: bool = False):
    """v2's halting (warm-up at the prior, then PonderNet + KL) around the reward loss. Returns (loss, per-step mean loss [N], halting dist [B,N], stats of the last step)."""
    res = [tree_loss_v3(o, tg, pool, commit, decouple) for o in outs]
    Ls = torch.stack([r[0] for r in res], 1)
    q = V2.halt_prior(len(outs), lam)
    if warm:
        p = q.unsqueeze(0).expand(Ls.size(0), -1)
        return (p * Ls).sum(1).mean(), Ls.detach().mean(0), p.detach(), res[-1][1]
    p = ML.halting(hl)
    kl = (p * (p.clamp(min=EPS).log() - q.log())).sum(1)
    return (p * Ls).sum(1).mean() + beta * kl.mean(), Ls.detach().mean(0), p.detach(), res[-1][1]
