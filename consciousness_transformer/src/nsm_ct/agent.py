"""Psyche: the trained consciousness/reasoning entity.

Psyche drives an episode through the emergent loop. The episode is a **uniform
stream of items** (statements … question … maybe distractors); the model is never
told which item is the question.

Two loop modes share the same heads, trust, and memory:

* **controlled** (default) — a *self-driven* loop. Each tick Psyche emits a
  control distribution over ``{READ, THINK, RESPOND}`` and decides for itself
  whether to advance and ingest the next sentence (a soft read-pointer), reason
  internally over memory with no new input, or contribute to its answer. So
  responses are sparse and it can "process and wait". Differentiable
  approximation of true input-pull control (RL is the next rung — RESEARCH_NOTES).
* **sequential** — the legacy one-tick-per-item loop, kept behind
  ``loop_mode="sequential"`` as a baseline and for tests.

In both modes an emergent **trust** signal scales how strongly an item influences
memory, and writes are content-addressed (overwrite, never forget) by default.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .long_term_memory import LongTermMemory
from .memory import MemoryState, WorkingMemory
from .model import (
    ACTION_ABSORB,
    ACTION_APPEND,
    ACTION_NAMES,
    ACTION_RESPOND,
    ConsciousnessTransformer,
)


class Psyche(nn.Module):
    """The trained consciousness loop over an :class:`~nsm_ct.dataset.EpisodeBatch`.

    Args:
        model: The state-transition transformer.
        memory: The per-episode *local context* (working memory).
        answer_mode: ``"mc"`` (score options) or ``"open"`` (classify answer).
        reasoning_hops: Extra post-stream reasoning steps (sequential mode).
        long_term: Optional persistent :class:`LongTermMemory` (world facts).
        pad_id: Pad token id (for the no-input reasoning steps).
        loop_mode: ``"controlled"`` (self-driven) or ``"sequential"`` (legacy).
        control_slack: Extra ticks beyond #items for thinking/responding.
        memory_addressing: ``"content"`` (overwrite) or ``"slot"`` (by index).
    """

    def __init__(
        self,
        model: ConsciousnessTransformer,
        memory: WorkingMemory,
        answer_mode: str = "mc",
        reasoning_hops: int = 1,
        long_term: Optional[LongTermMemory] = None,
        pad_id: int = 0,
        loop_mode: str = "controlled",
        control_slack: int = 6,
        memory_addressing: str = "content",
    ) -> None:
        super().__init__()
        self.model = model
        self.memory = memory
        self.answer_mode = answer_mode
        self.reasoning_hops = max(1, reasoning_hops)
        self.long_term = long_term
        self.pad_id = pad_id
        self.loop_mode = loop_mode
        self.control_slack = max(0, control_slack)
        self.memory_addressing = memory_addressing

    # -- reads / writes ------------------------------------------------------
    def _read(self, mem: MemoryState, state: torch.Tensor) -> torch.Tensor:
        """Read local context, plus long-term memory when present."""
        read = self.memory.read(mem, state)
        if self.long_term is not None:
            read = read + self.long_term.read(state)
        return read

    def _write(self, mem: MemoryState, slot_idx: int, content: torch.Tensor,
               gate: torch.Tensor) -> MemoryState:
        """Write per the configured addressing (content-overwrite or by slot)."""
        if self.memory_addressing == "content":
            return self.memory.write_content(mem, content, gate)
        return self.memory.write(mem, slot_idx, content, gate)

    def _respond(self, state: torch.Tensor, mem_read: torch.Tensor, batch) -> torch.Tensor:
        if self.answer_mode == "mc":
            return self.model.respond_mc(state, mem_read, batch.opt_ids, batch.opt_mask)
        return self.model.respond_open(state, mem_read)

    # -- forward dispatch ----------------------------------------------------
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        if self.loop_mode == "sequential":
            return self._forward_sequential(batch)
        return self._forward_controlled(batch)

    # -- controlled (self-driven) loop --------------------------------------
    def _forward_controlled(self, batch) -> Dict[str, torch.Tensor]:
        """Self-driven read/think/respond loop with a soft read-pointer.

        Tick quantities are attributed back to items via the read weights, so the
        returned tensors are item-aligned and the loss/metrics/consolidate paths
        are identical to the sequential mode.
        """
        device = batch.item_ids.device
        b, num_items, _ = batch.item_ids.shape
        state = self.model.initial_state(b, device)
        mem = self.memory.init_state(b, max(num_items, 1), device)
        n_real = batch.step_mask.sum(dim=1)                                  # [B]
        rows = torch.arange(b, device=device)
        eye = torch.eye(num_items, device=device)                           # for one-hot read
        p = torch.zeros(b, device=device)                                   # read pointer
        ticks = num_items + self.control_slack

        states = [state]
        rec_w, rec_cread, rec_crespond = [], [], []
        rec_trust, rec_append, rec_write, rec_action, rec_rlog, rec_pointer = [], [], [], [], [], []
        rec_rlog_mem, rec_wgate = [], []
        answer_num: torch.Tensor = torch.zeros(b, batch.opt_ids.shape[1], device=device)
        answer_den: torch.Tensor = torch.zeros(b, 1, device=device)
        mem_num: torch.Tensor = torch.zeros(b, batch.opt_ids.shape[1], device=device)

        for _ in range(ticks):
            c = self.model.control_gate(state)                  # [B, 3] READ/THINK/RESPOND
            c_read, c_respond = c[:, 0], c[:, 2]
            # The pointer selects which sentence to read; feed its FULL token
            # sequence (preserving word-level detail). Past the end → no input.
            idx = p.floor().long().clamp(max=num_items - 1)     # [B]
            past = (p >= n_real).float()                        # [B] 1 = nothing left
            cur_ids = batch.item_ids[rows, idx]                 # [B, L]
            cur_mask = batch.item_mask[rows, idx] * (1.0 - past).unsqueeze(-1)
            cur_roles = batch.item_roles[rows, idx]
            cur_depths = batch.item_depths[rows, idx]

            mem_read = self._read(mem, state)
            out = self.model.step(state, cur_ids, cur_mask, mem_read, cur_roles, cur_depths)
            new_state = out.new_state
            trust = self.model.trust_gate(new_state, mem_read)  # [B]
            probs = F.softmax(out.action_logits, dim=-1)
            write_gate = probs[:, ACTION_ABSORB] * c_read * trust * (1.0 - past)
            mem = self.memory.write_content(mem, out.write_vector, write_gate)

            resp_read = self._read(mem, new_state)
            rlog = self._respond(new_state, resp_read, batch)   # [B, K_opts]
            # Memory-bottleneck readout: the same response head with the state
            # zeroed, so the answer must be recoverable from MEMORY alone. Its
            # loss makes the trust-gated writes load-bearing (the state can't
            # smuggle the answer past memory).
            rlog_mem = self._respond(torch.zeros_like(new_state), resp_read, batch)
            answer_num = answer_num + c_respond.unsqueeze(-1) * rlog
            answer_den = answer_den + c_respond.unsqueeze(-1)
            mem_num = mem_num + c_respond.unsqueeze(-1) * rlog_mem

            w = eye[idx] * (1.0 - past).unsqueeze(-1)           # [B, N] one-hot read
            rec_w.append(w); rec_cread.append(c_read); rec_crespond.append(c_respond)
            rec_trust.append(trust); rec_append.append(probs[:, ACTION_APPEND])
            rec_wgate.append(write_gate)
            rec_write.append(out.write_vector); rec_action.append(out.action_logits)
            rec_rlog.append(rlog); rec_rlog_mem.append(rlog_mem); rec_pointer.append(p)

            p = torch.min(p + c_read, n_real)                   # advance only when reading
            state = new_state
            states.append(state)

        answer_logits = answer_num / answer_den.clamp(min=1e-6)
        answer_logits_mem = mem_num / answer_den.clamp(min=1e-6)

        # Per-question readout for multi-question streams: question j's answer is
        # the RESPOND-weighted aggregate over the ticks between reading it and
        # reading the next question (a soft pointer window) — several questions
        # answered in ONE unreset run, differentiably.
        P = torch.stack(rec_pointer, dim=1)                     # [B, K]
        CRESP = torch.stack(rec_crespond, dim=1)                # [B, K]
        RL = torch.stack(rec_rlog, dim=1)                       # [B, K, opts]
        qf = batch.q_positions.float()                          # [B, Q] (-1 pad)
        big = torch.full_like(qf, 1e4)
        nxt = torch.cat([qf[:, 1:], big[:, :1]], dim=1)
        nxt = torch.where(nxt < 0, big, nxt)
        sharp = 8.0
        in_window = (
            torch.sigmoid(sharp * (P.unsqueeze(1) - qf.unsqueeze(2) + 0.25))
            * (1.0 - torch.sigmoid(sharp * (P.unsqueeze(1) - nxt.unsqueeze(2) + 0.25)))
        )                                                       # [B, Q, K]
        qw = in_window * CRESP.unsqueeze(1)                     # [B, Q, K]
        qden = qw.sum(-1, keepdim=True).clamp(min=1e-4)
        question_logits = (qw.unsqueeze(-1) * RL.unsqueeze(1)).sum(2) / qden  # [B, Q, opts]

        W = torch.stack(rec_w, dim=1)                           # [B, K, N]
        CR = torch.stack(rec_cread, dim=1)                      # [B, K]
        TR = torch.stack(rec_trust, dim=1)                      # [B, K]
        AP = torch.stack(rec_append, dim=1)                     # [B, K]
        WV = torch.stack(rec_write, dim=1)                      # [B, K, mem]
        ACT = torch.stack(rec_action, dim=1)                    # [B, K, A]

        attn = W * CR.unsqueeze(-1)                             # tick→item attribution
        denom = attn.sum(dim=1).clamp(min=1e-6)                 # [B, N]
        trust_item = (attn * TR.unsqueeze(-1)).sum(dim=1) / denom
        append_item = (attn * (AP * TR).unsqueeze(-1)).sum(dim=1) / denom * batch.step_mask
        WG = torch.stack(rec_wgate, dim=1)                      # [B, K]
        write_gate_item = (attn * WG.unsqueeze(-1)).sum(dim=1) / denom * batch.step_mask
        action_item = torch.einsum("bkn,bka->bna", attn, ACT) / denom.unsqueeze(-1)
        # Per-item write vector = the content written at the tick that read that
        # item most (peak read), so consolidated facts stay distinctive rather
        # than blurring into one averaged vector.
        peak = W.argmax(dim=1)                                  # [B, N] tick per item
        write_item = torch.gather(WV, 1, peak.unsqueeze(-1).expand(-1, -1, WV.shape[-1]))

        return {
            "answer_logits": answer_logits,
            "answer_logits_mem": answer_logits_mem,             # memory-only readout
            "question_logits": question_logits,                 # [B, Q, opts]
            "action_logits": action_item,                       # [B, N, A]
            "respond_gates": CRESP,                             # [B, K] (per tick)
            "append_gates": append_item,                        # [B, N]
            "write_vecs": write_item,                           # [B, N, mem]
            "trust": trust_item,                                # [B, N]
            "write_gates": write_gate_item,                     # [B, N] effective ABSORB×trust
            "states": torch.stack(states, dim=1),               # [B, K+1, dim]
            "memory_occupancy": self.memory.occupancy(mem),
            "pointer": P,                                       # [B, K] (probe)
            "respond_logits_ticks": RL,                         # [B, K, opts] (probe)
        }

    # -- sequential (legacy) loop -------------------------------------------
    def _forward_sequential(self, batch) -> Dict[str, torch.Tensor]:
        """One tick per item, with a fixed number of post-stream reasoning steps."""
        device = batch.item_ids.device
        b, num_items, _ = batch.item_ids.shape
        state = self.model.initial_state(b, device)
        mem = self.memory.init_state(b, max(num_items, 1), device)

        states = [state]
        action_logits_list: List[torch.Tensor] = []
        respond_logits_list: List[torch.Tensor] = []
        respond_logits_mem_list: List[torch.Tensor] = []
        respond_gate_list: List[torch.Tensor] = []
        item_respond_gate_list: List[torch.Tensor] = []
        write_vecs_list: List[torch.Tensor] = []
        append_gate_list: List[torch.Tensor] = []
        trust_list: List[torch.Tensor] = []
        write_gate_list: List[torch.Tensor] = []

        for t in range(num_items):
            real = batch.step_mask[:, t]
            mem_read = self._read(mem, state)
            out = self.model.step(
                state, batch.item_ids[:, t], batch.item_mask[:, t], mem_read,
                batch.item_roles[:, t], batch.item_depths[:, t],
            )
            probs = F.softmax(out.action_logits, dim=-1)
            new_state = out.new_state
            trust = self.model.trust_gate(new_state, mem_read)
            absorb = probs[:, ACTION_ABSORB] * real * trust
            mem = self._write(mem, t, out.write_vector, absorb)

            resp_read = self._read(mem, new_state)
            respond_gate = probs[:, ACTION_RESPOND] * real

            action_logits_list.append(out.action_logits)
            respond_logits_list.append(self._respond(new_state, resp_read, batch))
            respond_logits_mem_list.append(
                self._respond(torch.zeros_like(new_state), resp_read, batch)
            )
            respond_gate_list.append(respond_gate)
            item_respond_gate_list.append(respond_gate)
            write_vecs_list.append(out.write_vector)
            append_gate_list.append(probs[:, ACTION_APPEND] * real * trust)
            trust_list.append(trust * real)
            write_gate_list.append(absorb)

            state = new_state
            states.append(state)

        pad_ids = torch.full((b, 1), self.pad_id, dtype=torch.long, device=device)
        pad_mask = torch.zeros((b, 1), dtype=torch.float32, device=device)
        for _ in range(self.reasoning_hops - 1):
            mem_read = self._read(mem, state)
            out = self.model.step(state, pad_ids, pad_mask, mem_read)
            probs = F.softmax(out.action_logits, dim=-1)
            new_state = out.new_state
            resp_read = self._read(mem, new_state)
            respond_logits_list.append(self._respond(new_state, resp_read, batch))
            respond_logits_mem_list.append(
                self._respond(torch.zeros_like(new_state), resp_read, batch)
            )
            respond_gate_list.append(probs[:, ACTION_RESPOND])
            state = new_state
            states.append(state)

        gates = torch.stack(respond_gate_list, dim=1)
        rlog = torch.stack(respond_logits_list, dim=1)
        rlog_mem = torch.stack(respond_logits_mem_list, dim=1)
        w = gates / gates.sum(dim=1, keepdim=True).clamp(min=1e-6)
        answer_logits = (w.unsqueeze(-1) * rlog).sum(dim=1)
        answer_logits_mem = (w.unsqueeze(-1) * rlog_mem).sum(dim=1)

        return {
            "answer_logits": answer_logits,
            "answer_logits_mem": answer_logits_mem,
            "question_logits": None,  # multi-question readout is controlled-mode only
            "action_logits": torch.stack(action_logits_list, dim=1),
            "respond_gates": torch.stack(item_respond_gate_list, dim=1),
            "append_gates": torch.stack(append_gate_list, dim=1),
            "write_vecs": torch.stack(write_vecs_list, dim=1),
            "trust": torch.stack(trust_list, dim=1),
            "write_gates": torch.stack(write_gate_list, dim=1),
            "states": torch.stack(states, dim=1),
            "memory_occupancy": self.memory.occupancy(mem),
        }

    # -- chained-question consistency probe ---------------------------------
    @torch.no_grad()
    def answer_at_positions(self, batch, positions: torch.Tensor) -> torch.Tensor:
        """Answer several questions within ONE unreset run (controlled loop).

        For each question item index, reads out the answer at the first tick the
        pointer passes that item — so multiple questions are answered without
        resetting state/memory between them.

        Args:
            batch: An episode batch whose stream contains the questions.
            positions: ``[B, Q]`` item indices of the questions (-1 = padding).

        Returns:
            ``[B, Q]`` predicted answer indices (-1 where padded).
        """
        out = self._forward_controlled(batch)
        pointer = out["pointer"]                    # [B, K]
        rlog = out["respond_logits_ticks"]          # [B, K, opts]
        b, q = positions.shape
        answers = torch.full((b, q), -1, dtype=torch.long, device=positions.device)
        last = pointer.shape[1] - 1
        for i in range(b):
            for j in range(q):
                qi = int(positions[i, j])
                if qi < 0:
                    continue
                passed = (pointer[i] >= qi + 0.5).nonzero(as_tuple=False)
                k = int(passed[0]) if passed.numel() > 0 else last
                answers[i, j] = int(rlog[i, k].argmax())
        return answers

    # -- long-term consolidation --------------------------------------------
    @torch.no_grad()
    def consolidate(self, out: Dict[str, torch.Tensor], batch) -> int:
        """Commit trusted items the model chose to APPEND into long-term memory.

        Consolidation strength = APPEND gate × trust, so only trustworthy items
        the model elects to keep enter (or update) the world-fact base. Returns
        the number of entries touched (added or overwritten in place).
        """
        if self.long_term is None:
            return 0
        write_vecs = out["write_vecs"]
        append_gates = out["append_gates"]
        step_mask = batch.step_mask
        touched = 0
        for bi in range(write_vecs.shape[0]):
            idxs_real = [t for t in range(step_mask.shape[1]) if step_mask[bi, t] > 0]
            if not idxs_real:
                continue
            vecs = write_vecs[bi, idxs_real]
            gates = append_gates[bi, idxs_real]
            texts = batch.item_texts[bi] if batch.item_texts is not None else None
            metas = (
                [{"text": texts[t], "kind": "learned"} for t in idxs_real]
                if texts is not None else None
            )
            touched += len(self.long_term.consolidate(vecs, gates=gates, metas=metas))
        return touched

    @torch.no_grad()
    def seed_world_facts(self, fact_texts, encoder, max_len: int) -> int:
        """Seed long-term memory with base 'facts we know about the world'."""
        if self.long_term is None or not fact_texts:
            return 0
        device = next(self.model.parameters()).device
        ids = [encoder.encode(f)[:max_len] or [self.pad_id] for f in fact_texts]
        length = max(len(x) for x in ids)
        item_ids = torch.full((len(ids), length), self.pad_id, dtype=torch.long, device=device)
        item_mask = torch.zeros((len(ids), length), dtype=torch.float32, device=device)
        for i, seq in enumerate(ids):
            item_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            item_mask[i, : len(seq)] = 1.0
        state = self.model.initial_state(len(ids), device)
        mem_read = torch.zeros(len(ids), self.model.cfg.memory_dim, device=device)
        out = self.model.step(state, item_ids, item_mask, mem_read)
        metas = [{"text": f, "kind": "base"} for f in fact_texts]
        return len(self.long_term.consolidate(out.write_vector, metas=metas))

    # -- introspection -------------------------------------------------------
    @torch.no_grad()
    def trace(self, batch) -> Dict[str, object]:
        """Per-item chosen action, trust, the chosen RESPOND step, and the answer."""
        out = self.forward(batch)
        item_actions = out["action_logits"].argmax(-1)
        respond_step = out["respond_gates"].argmax(-1)
        answers = out["answer_logits"].argmax(-1)
        actions, trust = [], []
        for i in range(item_actions.shape[0]):
            n = int(batch.step_mask[i].sum().item())
            actions.append([ACTION_NAMES[a] for a in item_actions[i, :n].tolist()])
            trust.append([round(x, 2) for x in out["trust"][i, :n].tolist()])
        return {
            "actions": actions,
            "trust": trust,
            "respond_step": respond_step.tolist(),
            "answers": answers.tolist(),
        }


# Backwards-compatible alias (the loop used to be called Mind).
Mind = Psyche
