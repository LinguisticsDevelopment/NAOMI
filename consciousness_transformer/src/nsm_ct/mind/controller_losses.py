"""Teacher-supervision losses for the M3 controller — soft→discrete anneal.

On top of the existing answer signal (:func:`clause_psyche.compute_clause_psyche_losses`)
we add the ``gold_chain`` supervision the codebase deferred (§0k/§0n/§0p):

* **relation-to-follow CE** — each inference hop's relation, decoded against the
  relation-atom codebook, supervised toward the teacher's gold relation (padded
  hops ignored). This is also the op-trace-match signal.
* **halt CE** — the PonderNet produce-step distribution toward the gold depth
  (answerable) / "never produce" (abstain). Directly fixes §0n's collapsed
  when-to-stop.

Both are weighted strongly early and decayed (the anneal), and the relation
softmax temperature anneals high→low so the runtime trace becomes a clean one-hot.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ..clause_psyche import ClausePsyche, compute_clause_psyche_losses
from ..clause_reactor import ClauseBatch
from .controller import relation_logits


def supervision_loss(
    out: Dict[str, torch.Tensor], rel_targets: torch.Tensor, depth: torch.Tensor,
    answerable: torch.Tensor, codebook: torch.Tensor, *, temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Relation-to-follow CE + halt CE from the teacher's gold supervision."""
    device = rel_targets.device
    rel_loss = torch.zeros((), device=device)
    if "hop_rels" in out and (rel_targets >= 0).any():
        logits = relation_logits(out["hop_rels"], codebook, temperature)  # [B, K, R]
        B, K, R = logits.shape
        rel_loss = F.cross_entropy(
            logits.reshape(B * K, R), rel_targets.reshape(B * K), ignore_index=-1)

    halt_loss = torch.zeros((), device=device)
    if "halt_dist" in out:
        halt = out["halt_dist"].clamp(min=1e-8)        # [B, K]
        p_never = out["p_never"].clamp(min=1e-8)       # [B]
        K = halt.shape[1]
        gold_k = (depth - 1).clamp(min=0, max=K - 1)   # 0-indexed produce step
        produce_nll = -torch.log(halt.gather(1, gold_k.unsqueeze(1)).squeeze(1))
        abstain_nll = -torch.log(p_never)
        # answerable -> hit the gold produce step; unanswerable -> never produce.
        halt_loss = (answerable * produce_nll + (1.0 - answerable) * abstain_nll).mean()

    return {"rel": rel_loss, "halt": halt_loss,
            "supervision": rel_loss + halt_loss}


def combined_loss(
    out: Dict[str, torch.Tensor], batch: ClauseBatch, model: ClausePsyche,
    sup: Dict[str, torch.Tensor], codebook: torch.Tensor, *,
    temperature: float = 1.0, w_rel: float = 1.0, w_halt: float = 1.0,
    **answer_kw,
) -> Dict[str, torch.Tensor]:
    """Answer loss + weighted teacher supervision (the M3 training objective)."""
    ans = compute_clause_psyche_losses(out, batch, model, **answer_kw)
    s = supervision_loss(out, sup["rel_targets"], sup["depth"], sup["answerable"],
                         codebook, temperature=temperature)
    total = ans["total"] + w_rel * s["rel"] + w_halt * s["halt"]
    return {"total": total, "answer": ans["total"], "rel": s["rel"], "halt": s["halt"], **{
        k: v for k, v in ans.items() if k != "total"}}


__all__ = ["supervision_loss", "combined_loss"]
