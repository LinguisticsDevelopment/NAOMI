"""ClausePsyche — the learned controller that reacts, REASONS, and responds.

A sibling of :class:`nsm_ct.clause_reactor.ClauseReactor` (kept as the baseline).
It generates a **clause meaning-object** (factored fillers assembled by fixed TPR
binds into a d×d matrix), scored by Frobenius + decode — not a 1-of-4 pick.

Reasoning is **emergent from the loop**: after ingesting the context (base facts
written into the order-3 ``entity_memory``, the differentiable STM), the controller
runs ``hops`` **inference ticks**. Each tick queries the STM with a state-derived
key, updates the **consciousness state** (the GRU hidden), and may **write a derived
``(entity,relation,value)`` back into the STM** — a materialized intermediate belief
the next tick re-reads. Multi-hop chaining (modus ponens, transitivity) thus emerges,
exactly as TPR-RNN does emergent multi-hop on bAbI. The state also drives an
**abstain** decision: respond only when it can derive, else emit "I don't know".
``hops=0`` reduces to the original single-pass behaviour (kept for the baseline).
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

OPS = ("WRITE", "SUPERSEDE", "NEGATE", "CORROBORATE", "RESPOND")


class ClausePsyche(nn.Module):
    """GRU controller → memory reaction + K inference hops (STM write-back) + a
    consciousness-state abstain gate + a generated clause meaning-object."""

    def __init__(self, codec: TPRCodec, hidden: int = 128, hops: int = 0,
                 halting: bool = False) -> None:
        super().__init__()
        self.dim = d = codec.dim
        self.hops = hops                            # with halting, this is the max-hop cap
        self.halting = halting
        self.gru = nn.GRUCell(6 * d, hidden)        # (entity, rel, value, pred, coord, mem_read)
        self.write_gate = nn.Linear(hidden, 1)
        self.overwrite_gate = nn.Linear(hidden, 1)
        self.decide_truth = nn.Linear(hidden + d, 1)
        self.op_head = nn.Linear(hidden, len(OPS))
        self.respond = nn.Linear(hidden, 1)
        # inference-hop head: only the RELATION to follow next is decided from the state.
        # The entity to look up is the value just read (the loop reads its own output), so
        # no head needs to re-conjure an arbitrary entity vector — that was the bottleneck.
        self.q_rel = nn.Linear(hidden, d)
        # the OUTPUT gate: reads the state AND the value just read, deciding "is this
        # completed thought my answer?" — seeing the value is what lets it learn to stop
        # once it has reached an answer (vs an intermediate entity).
        self.out_head = nn.Linear(hidden + d, 1)
        self.abstain = nn.Linear(hidden, 1)         # the consciousness state's whether-to-answer
        # response generators (the meaning-object): factored fillers
        self.gen_pred = nn.Linear(hidden, d)
        self.gen_subject = nn.Linear(hidden + d, d)
        self.gen_place = nn.Linear(hidden + d, d)
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
        zd = torch.zeros(b, d, device=device)

        states, resp_logits, ops = [], [], []
        gp, gs, gpl = [], [], []
        for t in range(T):                                   # Phase 1: ingest the stream
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
            gp.append(self.gen_pred(state))
            gs.append(self.gen_subject(torch.cat([state, e], dim=-1)))
            gpl.append(self.gen_place(torch.cat([state, mem_read], dim=-1)))

        qent = (batch.is_q.unsqueeze(-1) * batch.entity).sum(1)     # the question's entity
        ponder = torch.zeros(b, device=device)
        last_read = torch.zeros(b, d, device=device)
        focus = qent                                         # the working thought's current subject
        read_steps, hop_states = [], []
        for _k in range(self.hops):                          # Phase 2: think in meaning objects
            qr = self.q_rel(state)                           # the thought's relation (which to follow)
            mem_read = em.query(memory, focus, qr)           # fill the thought's value from STM
            state = self.gru(torch.cat([focus, qr, mem_read, zd, zd, mem_read], dim=-1), state)
            states.append(state)
            read_steps.append(mem_read)
            hop_states.append(state)
            last_read = mem_read
            focus = mem_read                                 # REINPUT: the read value becomes the
                                                             # next thought's subject (sourced, not conjured)

        if self.halting and self.hops > 0:                   # pick WHICH thought is the answer
            R = torch.stack(read_steps, dim=1)               # [b, K, d]  the per-step thoughts
            S = torch.stack(hop_states, dim=1)               # [b, K, h]
            score = self.out_head(torch.cat([S, R], dim=-1)).squeeze(-1)   # "is this thought the answer?"
            attn = torch.softmax(score, dim=1)               # attend over my own thoughts [b, K]
            acc_val = (attn.unsqueeze(-1) * R).sum(1)         # the selected answer value
            settled = (attn.unsqueeze(-1) * S).sum(1)
            ponder = (attn * torch.arange(1, self.hops + 1, device=device).float()).sum(1)  # depth reached
            pred_f = self.gen_pred(settled)
            subj_f = self.gen_subject(torch.cat([settled, qent], dim=-1))
            place_f = self.gen_place(torch.cat([settled, acc_val], dim=-1))
            abstain_state = settled
            w_stream = torch.softmax(torch.stack(resp_logits, dim=1), dim=1)
        elif self.hops > 0:                                  # fixed-hop: respond from final state
            pred_f = self.gen_pred(state)
            subj_f = self.gen_subject(torch.cat([state, qent], dim=-1))
            place_f = self.gen_place(torch.cat([state, last_read], dim=-1))
            abstain_state = state
            ponder = ponder + float(self.hops)
            w_stream = torch.softmax(torch.stack(resp_logits, dim=1), dim=1)
        else:                                                # single-pass aggregation (baseline)
            w = torch.softmax(torch.stack(resp_logits, dim=1), dim=1).unsqueeze(-1)
            pred_f = (w * torch.stack(gp, dim=1)).sum(1)
            subj_f = (w * torch.stack(gs, dim=1)).sum(1)
            place_f = (w * torch.stack(gpl, dim=1)).sum(1)
            abstain_state = state
            w_stream = w.squeeze(-1)

        matrix = self.assemble(pred_f, subj_f, place_f)
        abstain_logit = self.abstain(abstain_state).squeeze(-1)
        pf = F.normalize(place_f, dim=-1)
        on = F.normalize(batch.options, dim=-1)
        answer_logits = torch.einsum("bd,bkd->bk", pf, on) * 10.0

        return {
            "matrix": matrix, "place_filler": place_f, "subject_filler": subj_f,
            "pred_filler": pred_f, "op_logits": torch.stack(ops, dim=1),
            "states": torch.stack(states, dim=1), "answer_logits": answer_logits,
            "respond_gates": w_stream, "abstain_logit": abstain_logit,
            "abstain_prob": torch.sigmoid(abstain_logit), "ponder_steps": ponder,
        }


def gold_matrix(model: ClausePsyche, batch: ClauseBatch) -> torch.Tensor:
    """Assemble the gold clause matrix: question subject + answer value + 'is'."""
    b = batch.entity.shape[0]
    subj = (batch.is_q.unsqueeze(-1) * batch.entity).sum(1)
    value = batch.options[torch.arange(b), batch.answer]
    pred = model.pred_atom.expand(b, -1)
    return model.assemble(pred, subj, value)


def compute_clause_psyche_losses(
    out: Dict[str, torch.Tensor], batch: ClauseBatch, model: ClausePsyche,
    *, w_decode: float = 1.0, w_consistency: float = 0.01, w_abstain: float = 1.0,
    w_ponder: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Frobenius + decode-CE on **answerable** episodes; abstain BCE; consistency;
    + a small **ponder cost** (reward stopping as soon as it is sure).

    Generation is supervised only where the answer is derivable; the abstain head
    learns ``should_abstain = 1 - answerable`` — so the model is trained to respond
    only with what it can derive (and say "I don't know" otherwise)."""
    b = batch.entity.shape[0]
    ok = batch.answerable if batch.answerable is not None else batch.answer.new_ones(b).float()
    denom = ok.sum().clamp(min=1.0)
    M_gold = gold_matrix(model, batch)
    frob = ((out["matrix"] - M_gold).pow(2).sum(dim=(-1, -2)) * ok).sum() / denom
    decode = (F.cross_entropy(out["answer_logits"], batch.answer, reduction="none") * ok).sum() / denom
    abstain = F.binary_cross_entropy_with_logits(out["abstain_logit"], 1.0 - ok)
    consistency = consciousness_consistency_loss(out["states"])
    ponder = out["ponder_steps"].mean()
    total = (frob + w_decode * decode + w_abstain * abstain
             + w_consistency * consistency + w_ponder * ponder)
    return {"total": total, "frobenius": frob, "decode": decode,
            "abstain": abstain, "consistency": consistency, "ponder": ponder}


def clause_decode_accuracy(out: Dict[str, torch.Tensor], batch: ClauseBatch) -> float:
    """Exact decode on answerable episodes: generated place nearest the correct option."""
    b = batch.entity.shape[0]
    ok = batch.answerable if batch.answerable is not None else batch.answer.new_ones(b).float()
    correct = (out["answer_logits"].argmax(-1) == batch.answer).float() * ok
    return float(correct.sum() / ok.sum().clamp(min=1.0))


def abstain_prf(out: Dict[str, torch.Tensor], batch: ClauseBatch) -> Dict[str, float]:
    """Precision/recall of the abstain decision vs the unanswerable flag."""
    if batch.answerable is None:
        return {"precision": 1.0, "recall": 1.0, "n_abstain": 0}
    should = (1.0 - batch.answerable).bool()
    did = out["abstain_prob"] >= 0.5
    tp = float((did & should).sum())
    prec = tp / max(float(did.sum()), 1.0)
    rec = tp / max(float(should.sum()), 1.0)
    return {"precision": prec, "recall": rec, "n_abstain": int(should.sum())}
