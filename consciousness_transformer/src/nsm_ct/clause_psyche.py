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
        # derive-chain mode (M9): the loop chains on its own GENERATED derived value
        # (gen_derive) rather than the memory read — so proof-chain supervision sits
        # causally on the answer path (the M3 lesson: supervise what drives the answer).
        self.derive_chain = False
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
        # per-hop "show your work" head: the meaning-VALUE derived at each inference hop
        # (e.g. ProofWriter's furry→kind→smart). Separate from gen_place (the final
        # answer) so proof-chain supervision (M9) shapes the reasoning without colliding
        # with the answer readout. Unused unless the M9 value-supervision loss reads it.
        self.gen_derive = nn.Linear(hidden, d)
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
            new_state = self.gru(torch.cat([e, r, v, p, c, mem_read], dim=-1), state)
            # Padding-invariant: a pad step (mask=0) must NOT drift the consciousness
            # state, else the answer depends on trailing padding (batch T) — which breaks
            # single-question inference. Only real steps update the state.
            state = torch.where(real.unsqueeze(-1) > 0, new_state, state)
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
        read_steps, hop_states, hop_rels, derived_steps = [], [], [], []
        for _k in range(self.hops):                          # Phase 2: think in meaning objects
            qr = self.q_rel(state)                           # the thought's relation (which to follow)
            mem_read = em.query(memory, focus, qr)           # fill the thought's value from STM
            state = self.gru(torch.cat([focus, qr, mem_read, zd, zd, mem_read], dim=-1), state)
            states.append(state)
            read_steps.append(mem_read)
            hop_states.append(state)
            hop_rels.append(qr)                              # exposed for M3 relation-to-follow supervision
            if self.derive_chain:                            # M9: chain on the GENERATED derived value
                derived = self.gen_derive(state)             # (supervised toward the proof's k-th value)
                derived_steps.append(derived)
                last_read = focus = derived                  # the derivation drives the next hop + answer
            else:
                last_read = mem_read
                focus = mem_read                             # §0n: the read value becomes the next
                                                             # thought's subject (sourced, not conjured)

        halt_dist = p_never = step_answer_logits = None
        if self.halting and self.hops > 0:                   # PonderNet: decide WHEN to produce
            R = torch.stack(read_steps, dim=1)               # [b, K, d]  the per-step thoughts
            S = torch.stack(hop_states, dim=1)               # [b, K, h]
            qK = qent.unsqueeze(1).expand(-1, self.hops, -1)
            place_k = self.gen_place(torch.cat([S, R], dim=-1))           # [b,K,d] candidate answers
            subj_k = self.gen_subject(torch.cat([S, qK], dim=-1))
            pred_k = self.gen_pred(S)
            h = torch.sigmoid(self.out_head(torch.cat([S, R], dim=-1)).squeeze(-1))  # produce-now? [b,K]
            # first-output halting distribution: pi_k = h_k * prod_{j<k}(1-h_j); never = prod(1-h)
            keep = torch.cumprod(1.0 - h, dim=1)
            excl = torch.cat([torch.ones(b, 1, device=device), keep[:, :-1]], dim=1)
            halt_dist = h * excl                              # [b,K]  output-at-step distribution
            p_never = keep[:, -1]                             # [b]    abstain mass
            pred_f = (halt_dist.unsqueeze(-1) * pred_k).sum(1)            # answer = the committed step
            subj_f = (halt_dist.unsqueeze(-1) * subj_k).sum(1)
            place_f = (halt_dist.unsqueeze(-1) * place_k).sum(1)
            ponder = ((halt_dist * torch.arange(1, self.hops + 1, device=device).float()).sum(1)
                      + p_never * float(self.hops))
            on = F.normalize(batch.options, dim=-1)
            step_answer_logits = torch.einsum("bkd,bod->bko",
                                              F.normalize(place_k, dim=-1), on) * 10.0
            abstain_state = (halt_dist.unsqueeze(-1) * S).sum(1)
            w_stream = halt_dist
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
        pf = F.normalize(place_f, dim=-1)
        on = F.normalize(batch.options, dim=-1)
        answer_logits = torch.einsum("bd,bkd->bk", pf, on) * 10.0
        # abstain: for the halting model it IS the "never produced an answer" mass; else learned.
        if p_never is not None:
            abstain_prob = p_never
            abstain_logit = torch.log(p_never.clamp(1e-6, 1 - 1e-6)) \
                - torch.log((1 - p_never).clamp(1e-6, 1 - 1e-6))
        else:
            abstain_logit = self.abstain(abstain_state).squeeze(-1)
            abstain_prob = torch.sigmoid(abstain_logit)

        out = {
            "matrix": matrix, "place_filler": place_f, "subject_filler": subj_f,
            "pred_filler": pred_f, "op_logits": torch.stack(ops, dim=1),
            "states": torch.stack(states, dim=1), "answer_logits": answer_logits,
            "respond_gates": w_stream, "abstain_logit": abstain_logit,
            "abstain_prob": abstain_prob, "ponder_steps": ponder,
        }
        if halt_dist is not None:
            out["halt_dist"] = halt_dist
            out["p_never"] = p_never
            out["step_answer_logits"] = step_answer_logits
        if self.hops > 0:
            out["hop_rels"] = torch.stack(hop_rels, dim=1)   # [b, K, d] per-hop relation-to-follow
            S_all = torch.stack(hop_states, dim=1)            # [b, K, h]
            out["hop_states"] = S_all
            out["hop_reads"] = torch.stack(read_steps, dim=1)  # [b, K, d] per-hop memory read
            out["hop_derived"] = (torch.stack(derived_steps, dim=1) if self.derive_chain
                                  else self.gen_derive(S_all))  # [b, K, d] per-hop derived VALUE (M9)
        return out


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
    w_ponder: float = 0.01, r_correct: float = 1.0, r_wrong: float = 1.0,
    r_abstain: float = 0.25, w_prior: float = 0.0, prior_lambda: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """Two regimes. For the **halting** model (PonderNet): an asymmetric reward —
    greatly reward a correct produced answer, greatly punish a wrong one, mildly
    punish "I don't know" — so the model only commits when it is actually confident
    (and otherwise keeps thinking, or abstains). For non-halting models: the original
    Frobenius + decode-CE + abstain-BCE."""
    b = batch.entity.shape[0]
    ok = batch.answerable if batch.answerable is not None else batch.answer.new_ones(b).float()
    consistency = consciousness_consistency_loss(out["states"])

    if "halt_dist" in out:                                   # asymmetric-reward (the user's design)
        pi = out["halt_dist"]                                # [b,K] produce-at-step distribution
        p_never = out["p_never"]                             # [b]   abstain mass
        pc = torch.softmax(out["step_answer_logits"], dim=-1)[
            torch.arange(b), :, batch.answer]                # [b,K] P(correct) if produced at step k
        # per-step reward if it produces here: answerable -> ++correct/--wrong; else any answer wrong
        step_reward = ok.unsqueeze(1) * (r_correct * pc - r_wrong * (1.0 - pc)) \
            + (1.0 - ok).unsqueeze(1) * (-r_wrong)
        produce_reward = (pi * step_reward).sum(1)           # [b]
        abstain_reward = p_never * (ok * (-r_abstain) + (1.0 - ok) * r_abstain)
        reward = (produce_reward + abstain_reward).mean()
        ponder = out["ponder_steps"].mean()
        total = -reward + w_consistency * consistency + w_ponder * ponder
        # PonderNet exploration prior: nudge the produce-step distribution toward a geometric
        # so DEEP steps get gradient to learn the walk (escapes the step-1 cold-start collapse).
        prior_kl = pi.new_zeros(())
        if w_prior > 0.0:
            K = pi.shape[1]
            k = torch.arange(1, K + 1, device=pi.device).float()
            prior = (1 - prior_lambda) ** (k - 1)
            prior = prior / prior.sum()
            pi_c = pi / (1.0 - p_never).clamp(min=1e-6).unsqueeze(1)      # conditional on producing
            prior_kl = (pi_c * (torch.log(pi_c.clamp(min=1e-8)) - torch.log(prior))).sum(1).mean()
            total = total + w_prior * prior_kl
        return {"total": total, "reward": reward, "consistency": consistency,
                "ponder": ponder, "p_abstain": p_never.mean(), "prior_kl": prior_kl}

    denom = ok.sum().clamp(min=1.0)
    M_gold = gold_matrix(model, batch)
    frob = ((out["matrix"] - M_gold).pow(2).sum(dim=(-1, -2)) * ok).sum() / denom
    decode = (F.cross_entropy(out["answer_logits"], batch.answer, reduction="none") * ok).sum() / denom
    abstain = F.binary_cross_entropy_with_logits(out["abstain_logit"], 1.0 - ok)
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
