"""The learned controller — reasons in vector space, emits a typed op-trace (M3).

``MindController`` is the proven ``ClausePsyche`` focus-chaining loop (GRU + the
§0n read-its-own-output traversal + PonderNet halting + abstain), surfaced in the
``mind.ops`` vocabulary so its per-tick `(op, relation-to-follow, halt)` choices are
a faithful, inspectable **op-trace** — the thing the M2 executor teaches and can
replay-validate. Learned: op-type, relation-to-follow, when-to-stop. Not learned:
the ops, the TPR binds, or any fact.

Relation selection is decoded against the fixed relation-atom codebook
(:func:`nsm_ct.mind.teacher.relation_codebook`); a ``temperature`` controls the
soft→discrete anneal (high early = smooth gradients; low late = a clean one-hot
runtime trace).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from ..clause_psyche import ClausePsyche
from ..tpr import TPRCodec
from . import ops, teacher


class MindController(ClausePsyche):
    """The focus-chaining controller, surfaced as a typed-op emitter.

    Identical learning/behaviour to :class:`ClausePsyche`; adds the relation
    codebook and op-trace decoding so the loop plugs into the ``mind`` framework.
    """

    def __init__(self, codec: TPRCodec, hidden: int = 128, hops: int = 5,
                 halting: bool = True, temperature: float = 1.0,
                 derive_chain: bool = False) -> None:
        super().__init__(codec, hidden=hidden, hops=hops, halting=halting)
        self.derive_chain = derive_chain
        self.temperature = float(temperature)
        self.register_buffer("relation_codebook",
                             torch.from_numpy(teacher.relation_codebook(codec)))


def relation_logits(hop_rels: torch.Tensor, codebook: torch.Tensor,
                    temperature: float = 1.0) -> torch.Tensor:
    """Cosine logits of each hop's relation vector against the codebook → ``[B, K, R]``.

    ``temperature`` sharpens (low) or softens (high) the selection — the soft→discrete
    anneal lever.
    """
    hr = F.normalize(hop_rels, dim=-1)               # [B, K, d]
    cb = F.normalize(codebook, dim=-1)               # [R, d]
    return torch.einsum("bkd,rd->bkr", hr, cb) / max(temperature, 1e-6)


def relation_match(out: dict, rel_targets: torch.Tensor, codebook: torch.Tensor) -> float:
    """Op-trace match: fraction of supervised hops whose argmax relation == gold."""
    if "hop_rels" not in out:
        return 0.0
    logits = relation_logits(out["hop_rels"], codebook, temperature=1e-3)  # ~argmax
    pred = logits.argmax(-1)                          # [B, K]
    mask = rel_targets >= 0
    if mask.sum() == 0:
        return 1.0
    return float(((pred == rel_targets) & mask).sum() / mask.sum())


def emit_op_trace(out: dict, codebook: torch.Tensor, episode_idx: int,
                  query: tuple) -> List[ops.Op]:
    """Decode the controller's emitted op-trace for one episode (inspection/replay).

    INFER hops with their decoded relation-to-follow, then RESPOND (or HALT/abstain).
    """
    trace: List[ops.Op] = []
    if "hop_rels" in out:
        logits = relation_logits(out["hop_rels"], codebook, temperature=1e-3)
        rels = logits.argmax(-1)[episode_idx].tolist()      # [K]
        stop = out["hop_rels"].shape[1]
        if "halt_dist" in out:
            stop = int(out["halt_dist"][episode_idx].argmax()) + 1
        for k in range(min(stop, len(rels))):
            r = teacher.REASONING_RELS[rels[k]]
            trace.append(ops.Op(ops.INFER, {"follow": r, "hop": k}))
    abstain = bool(out["abstain_prob"][episode_idx] >= 0.5) if "abstain_prob" in out else False
    if abstain:
        trace.append(ops.Op(ops.RESPOND, {"subject": query[0], "relation": query[1],
                                          "answer": ops.ABSTAIN}))
    else:
        trace.append(ops.Op(ops.RESPOND, {"subject": query[0], "relation": query[1]}))
    return trace


__all__ = ["MindController", "relation_logits", "relation_match", "emit_op_trace"]
