"""Mind loop v2 (dev/reports/MIND_LOOP_V2.md). Same model (``mind_loop.Loop``, v1 checkpoints load); two training-side fixes, both language-neutral:

1. Halt: PonderNet with a geometric prior of mean ~3 steps (lambda 1/3) and a WARM-UP in which the halting distribution is held AT the
   prior (the halt head gets no gradient), so every step trains before the halt can specialise; then it is released (learned halt read from
   the state, KL weight ``beta`` to the same prior).
2. Counterfactual memory items: take a training item, find the memory slots aligned to the reference answer (same sense row / same identity key,
   low document frequency, not in the question), replace those slots' content (USVS point, identity key, node) with the content of another slot
   of the SAME tag (pos) from a different story, and replace the same identity in the reference tree. Same question, different memory ->
   different correct answer. No word lists: alignment is by identity, "content-like" is a document-frequency statistic of the data.
Memory is still pre-stored by code and T never writes it.
"""
from __future__ import annotations

import string
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Dict, List, Optional

import torch

from . import mind_loop as ML

LAM_PRIOR = 1.0 / 3.0          # geometric halting prior, mean ~3 steps (cap NSTEP = 6)


def halt_prior(n: int, lam: float = LAM_PRIOR) -> torch.Tensor:
    return ML.geometric_prior(n, lam)


def is_warm(epoch: int, warm_epochs: int) -> bool:
    """Schedule: epochs [0, warm_epochs) hold the halting distribution at the prior; from epoch warm_epochs the halt is learned."""
    return epoch < warm_epochs


def ponder_loss_v2(outs: List[dict], hl: torch.Tensor, tg: dict, lam: float = LAM_PRIOR, beta: float = 0.3, warm: bool = False):
    """warm: p = prior (hl unused, no gradient to the halt head); else p = PonderNet halting(hl) and loss += beta KL(p || prior).
    Returns (loss, per-step mean tree loss [N], halting distribution [B, N] (detached))."""
    Ls = torch.stack([ML.tree_loss(o, tg) for o in outs], 1)
    q = halt_prior(len(outs), lam)
    if warm:
        p = q.unsqueeze(0).expand(Ls.size(0), -1)
        return (p * Ls).sum(1).mean(), Ls.detach().mean(0), p.detach()
    p = ML.halting(hl)
    kl = (p * (p.clamp(min=ML.EPS).log() - q.log())).sum(1)
    return (p * Ls).sum(1).mean() + beta * kl.mean(), Ls.detach().mean(0), p.detach()


# ----------------------------------------------------------------------------- counterfactual memory
def identity(row, key):
    """What a node IS, with no language knowledge: its sense row (has a USVS point) or its identity key (name / lexicon-less word)."""
    if int(row) > 0:
        return ("r", int(row))
    return ("k", key) if key else None


def swap_text(text: str, pairs) -> str:
    """Replace surface tokens (whole token, case-insensitive, punctuation kept) of ``text``: pairs = [(old, new)]."""
    out = []
    for w in text.split():
        core = w.strip(string.punctuation)
        for o, d in pairs:
            if core and core.lower() == o.lower():
                w = w.replace(core, d, 1)
                break
        out.append(w)
    return " ".join(out)


class CF:
    """Builds counterfactual memories for one split (donors come from other stories of the same split)."""

    def __init__(self, sf: dict, stories: dict, skeys, max_df: float = 0.02, max_swaps: int = 3):
        self.sf, self.stories, self.skeys = sf, stories, sorted(skeys)
        self.max_swaps = max_swaps
        self.ids = {s: [identity(r, k) for r, k in zip(sf[s]["mem"]["row"].tolist(), sf[s]["mem"]["key"])] for s in self.skeys}
        df = Counter()
        for s in self.skeys:
            df.update({i for i in self.ids[s] if i is not None})
        self.df, self.lim = df, max_df * len(self.skeys)
        self.donors: Dict[int, list] = defaultdict(list)
        for s in self.skeys:
            mem = sf[s]["mem"]
            for g, i in enumerate(self.ids[s]):
                if i is not None and df[i] <= self.lim and g < len(stories[s]["nodes"]):
                    self.donors[int(mem["pos"][g])].append((s, g))

    def build(self, skey: str, q: dict, ref: dict, rng) -> Optional[dict]:
        """-> {mem, keys, nodes, ref, swaps:[(old token, new token)]} or None when no reference node is aligned to a content slot."""
        mem, ml, nodes0 = self.sf[skey]["mem"], self.ids[skey], self.stories[skey]["nodes"]
        qids = {identity(r, k) for r, k in zip(q["row"].tolist(), q["key"])}
        slots = defaultdict(list)
        for j, i in enumerate(ml):
            if i is not None:
                slots[i].append(j)
        cand = defaultdict(list)
        for ri in range(ref["n"]):
            i = identity(ref["row"][ri], ref["key"][ri])
            if i is not None and i not in qids and i in slots and self.df[i] <= self.lim:
                cand[i].append(ri)
        order = sorted(cand, key=str)
        rng.shuffle(order)
        used, swaps = set(ml), []
        for i in order[:self.max_swaps]:
            pool = self.donors.get(int(mem["pos"][slots[i][0]]))
            if not pool:
                continue
            for _ in range(20):
                ds, dg = pool[rng.randrange(len(pool))]
                did = self.ids[ds][dg]
                if ds != skey and did not in used:
                    break
            else:
                continue
            used.add(did)
            swaps.append((i, ds, dg))
        if not swaps:
            return None
        row, key, nodes = mem["row"].copy(), list(mem["key"]), list(nodes0)
        rrow, rkey = ref["row"].copy(), list(ref["key"])
        pairs = []
        for i, ds, dg in swaps:
            dm, dn = self.sf[ds]["mem"], self.stories[ds]["nodes"][dg]
            for j in slots[i]:
                row[j], key[j] = dm["row"][dg], dm["key"][dg]
                if j < len(nodes):
                    nodes[j] = replace(dn, rel=nodes[j].rel)
            for ri in cand[i]:
                rrow[ri], rkey[ri] = dm["row"][dg], dm["key"][dg]
            pairs.append((nodes0[slots[i][0]].token, dn.token))
        m2 = {**mem, "row": row, "key": key}
        return {"mem": m2, "keys": {k for k in key if k}, "nodes": nodes, "ref": {**ref, "row": rrow, "key": rkey}, "swaps": pairs}
