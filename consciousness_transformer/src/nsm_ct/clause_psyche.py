"""ClausePsyche — the learned controller that reacts and GENERATES a meaning-object.

A sibling of :class:`nsm_ct.clause_reactor.ClauseReactor` (kept as the baseline),
this one does not pick a multiple-choice option: it **generates a clause** —
factored fillers (predicate / subject / place) assembled by fixed TPR binds into
a d×d clause matrix — scored by **Frobenius** to the gold clause matrix and by an
exact **decode** of the generated place filler. The GRU hidden is carried as the
**consciousness state** (the spotlight), regularized by the consistency loss.

Perception is the same fixed grounded stream as the reactor (reuse
``build_clause_batch``); the only learned parameters are the GRU + heads.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import entity_memory as em
from .clause_reactor import ClauseBatch
from .losses import consciousness_consistency_loss
from .tpr import TPRCodec

# op-routing vocabulary (present + shape-tested; emergent, not hard-supervised here)
OPS = ("WRITE", "SUPERSEDE", "NEGATE", "CORROBORATE", "RESPOND")


class ClausePsyche(nn.Module):
    """GRU controller → op routing + a maintained consciousness state + a
    generated clause meaning-object (assembled d×d matrix)."""

    def __init__(self, codec: TPRCodec, hidden: int = 128) -> None:
        super().__init__()
        self.dim = d = codec.dim
        self.gru = nn.GRUCell(6 * d, hidden)        # (entity, rel, value, pred, coord, mem_read)
        self.write_gate = nn.Linear(hidden, 1)
        self.overwrite_gate = nn.Linear(hidden, 1)
        self.decide_truth = nn.Linear(hidden + d, 1)
        self.op_head = nn.Linear(hidden, len(OPS))  # routing (present; emergent)
        self.respond = nn.Linear(hidden, 1)
        # response generators (the meaning-object): factored fillers
        self.gen_pred = nn.Linear(hidden, d)
        self.gen_subject = nn.Linear(hidden + d, d)
        self.gen_place = nn.Linear(hidden + d, d)
        self.gen_negate = nn.Linear(hidden, 1)
        # fixed TPR roles for assembly (buffers; not learned)
        self.register_buffer("self_role", torch.from_numpy(codec.self_role.copy()))
        self.register_buffer("subj_role", torch.from_numpy(codec.role_vec(0, "SUBJECT").copy()))
        self.register_buffer("place_role", torch.from_numpy(codec.role_vec(1, "PLACE").copy()))
        self.register_buffer("pred_atom", torch.from_numpy(codec.filler_vec("pred:is").copy()))

    def assemble(self, pred: torch.Tensor, subj: torch.Tensor, place: torch.Tensor) -> torch.Tensor:
        """Bind factored fillers into a clause matrix ``[B, d, d]`` (fixed roles)."""
        b, d = pred.shape
        return (self.self_role.view(1, d, 1) * pred.view(b, 1, d)
                + self.subj_role.view(1, d, 1) * subj.view(b, 1, d)
                + self.place_role.view(1, d, 1) * place.view(b, 1, d))

    def forward(self, batch: ClauseBatch) -> Dict[str, torch.Tensor]:
        b, T, d = batch.entity.shape
        device = batch.entity.device
        state = torch.zeros(b, self.gru.hidden_size, device=device)
        memory = em.init_memory(b, d, device)
        coord = batch._coord()

        states, resp_logits, ops = [], [], []
        gen_pred, gen_subj, gen_place, gen_neg = [], [], [], []
        for t in range(T):
            e, r, v = batch.entity[:, t], batch.relation[:, t], batch.value[:, t]
            p, c = batch.pred[:, t], coord[:, t]
            real, isq = batch.mask[:, t], batch.is_q[:, t]
            mem_read = em.query(memory, e, r)
            state = self.gru(torch.cat([e, r, v, p, c, mem_read], dim=-1), state)
            states.append(state)

            stmt = real * (1.0 - isq)
            gate = torch.sigmoid(self.write_gate(state)).squeeze(-1) * stmt
            owr = torch.sigmoid(self.overwrite_gate(state)).squeeze(-1) * gate
            neg = torch.sigmoid(self.decide_truth(torch.cat([state, v], dim=-1))).squeeze(-1) * stmt
            memory = em.write(memory, e, r, v, gate - neg, overwrite=owr)

            ops.append(self.op_head(state))
            rl = self.respond(state).squeeze(-1).masked_fill(real <= 0, float("-inf"))
            resp_logits.append(rl)
            gen_pred.append(self.gen_pred(state))
            gen_subj.append(self.gen_subject(torch.cat([state, e], dim=-1)))
            gen_place.append(self.gen_place(torch.cat([state, mem_read], dim=-1)))
            gen_neg.append(self.gen_negate(state))

        RL = torch.stack(resp_logits, dim=1)                  # [B, T]
        w = torch.softmax(RL, dim=1).unsqueeze(-1)            # respond distribution
        pred_f = (w * torch.stack(gen_pred, dim=1)).sum(1)    # [B, d]
        subj_f = (w * torch.stack(gen_subj, dim=1)).sum(1)
        place_f = (w * torch.stack(gen_place, dim=1)).sum(1)
        negate = torch.sigmoid((w.squeeze(-1) * torch.stack(gen_neg, dim=1).squeeze(-1)).sum(1))

        matrix = self.assemble(pred_f, subj_f, place_f)       # the generated meaning-object

        # decode readout: cosine(generated place filler, option place vectors)
        pf = F.normalize(place_f, dim=-1)
        on = F.normalize(batch.options, dim=-1)
        answer_logits = torch.einsum("bd,bkd->bk", pf, on) * 10.0

        return {
            "matrix": matrix, "place_filler": place_f, "subject_filler": subj_f,
            "pred_filler": pred_f, "negate": negate,
            "op_logits": torch.stack(ops, dim=1), "states": torch.stack(states, dim=1),
            "answer_logits": answer_logits, "respond_gates": w.squeeze(-1),
        }


def gold_matrix(model: ClausePsyche, batch: ClauseBatch) -> torch.Tensor:
    """Assemble the gold clause matrix: question subject + answer place + 'is'."""
    b = batch.entity.shape[0]
    subj = (batch.is_q.unsqueeze(-1) * batch.entity).sum(1)          # var:<question entity>
    place = batch.options[torch.arange(b), batch.answer]            # the correct place vector
    pred = model.pred_atom.expand(b, -1)
    return model.assemble(pred, subj, place)


def compute_clause_psyche_losses(
    out: Dict[str, torch.Tensor], batch: ClauseBatch, model: ClausePsyche,
    *, w_decode: float = 1.0, w_consistency: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Frobenius(meaning-object, gold) + decode-CE(place) + consistency."""
    M_gold = gold_matrix(model, batch)
    frob = (out["matrix"] - M_gold).pow(2).sum(dim=(-1, -2)).mean()
    decode = F.cross_entropy(out["answer_logits"], batch.answer)
    consistency = consciousness_consistency_loss(out["states"])
    total = frob + w_decode * decode + w_consistency * consistency
    return {"total": total, "frobenius": frob, "decode": decode, "consistency": consistency}


def clause_decode_accuracy(out: Dict[str, torch.Tensor], batch: ClauseBatch) -> float:
    """Exact place decode: generated place filler nearest the correct option."""
    pred = out["answer_logits"].argmax(-1)
    return float((pred == batch.answer).float().mean())
