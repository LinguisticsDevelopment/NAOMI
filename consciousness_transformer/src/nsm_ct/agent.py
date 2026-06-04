"""The Mind: drives an episode through the emergent action loop.

The episode is a **uniform stream of items** (statements … question … maybe
distractors). The model is never told which item is the question. At each item
the action head emits a 4-way distribution over
``{ABSORB, APPEND, RESPOND, SKIP}`` whose probabilities act as **soft gates**:

* ``ABSORB`` → gate the write into local (working) memory;
* ``APPEND`` → gate consolidation of the item into long-term memory;
* ``RESPOND`` → weight this step's response in a cross-step aggregate;
* ``SKIP`` → the residual (do nothing).

The episode's answer is the RESPOND-probability-weighted average of the per-step
responses, so the model learns *when* to answer (after the question) purely from
answer correctness — nothing supervises the action choice by position. See
RESEARCH_NOTES for the one exception (APPEND's label-free novelty signal).
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


class Mind(nn.Module):
    """Stateful driver over an :class:`~nsm_ct.dataset.EpisodeBatch`.

    Args:
        model: The state-transition transformer.
        memory: The per-episode *local context* (working memory).
        answer_mode: ``"mc"`` (score options) or ``"open"`` (classify answer).
        reasoning_hops: Extra post-stream reasoning steps (also response chances).
        long_term: Optional persistent :class:`LongTermMemory` (world facts).
        pad_id: Pad token id (for the no-input reasoning steps).
    """

    def __init__(
        self,
        model: ConsciousnessTransformer,
        memory: WorkingMemory,
        answer_mode: str = "mc",
        reasoning_hops: int = 1,
        long_term: Optional[LongTermMemory] = None,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.memory = memory
        self.answer_mode = answer_mode
        self.reasoning_hops = max(1, reasoning_hops)
        self.long_term = long_term
        self.pad_id = pad_id

    # -- reads ---------------------------------------------------------------
    def _read(self, mem: MemoryState, state: torch.Tensor) -> torch.Tensor:
        """Read local context, plus long-term memory when present."""
        read = self.memory.read(mem, state)
        if self.long_term is not None:
            read = read + self.long_term.read(state)
        return read

    def _respond(self, state: torch.Tensor, mem_read: torch.Tensor, batch) -> torch.Tensor:
        if self.answer_mode == "mc":
            return self.model.respond_mc(state, mem_read, batch.opt_ids, batch.opt_mask)
        return self.model.respond_open(state, mem_read)

    def _novelty(self, write_vecs: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:
        """Label-free novelty target for APPEND: 1 - max cosine sim to LTM.

        Returns ``[B, T]`` in [0, 1] (detached). All-novel (1.0) when the
        long-term repo is empty.
        """
        b, t, _ = write_vecs.shape
        if self.long_term is None or len(self.long_term) == 0:
            return torch.ones(b, t, device=write_vecs.device) * step_mask
        ltm = self.long_term._slots.to(write_vecs.device)          # [N, mem_dim]
        wv = F.normalize(write_vecs.detach(), dim=-1)
        lt = F.normalize(ltm, dim=-1)
        sims = wv @ lt.t()                                         # [B, T, N]
        novelty = (1.0 - sims.max(dim=-1).values).clamp(0.0, 1.0)  # [B, T]
        return novelty * step_mask

    # -- forward -------------------------------------------------------------
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Unroll the uniform item stream and return outputs for the loss."""
        device = batch.item_ids.device
        b, num_items, _ = batch.item_ids.shape
        state = self.model.initial_state(b, device)
        mem = self.memory.init_state(b, max(num_items, 1), device)

        states = [state]
        action_logits_list: List[torch.Tensor] = []   # item steps only
        respond_logits_list: List[torch.Tensor] = []   # all response chances
        respond_gate_list: List[torch.Tensor] = []      # all response chances
        item_respond_gate_list: List[torch.Tensor] = []  # item steps only (metric)
        write_vecs_list: List[torch.Tensor] = []
        append_gate_list: List[torch.Tensor] = []

        for t in range(num_items):
            real = batch.step_mask[:, t]                            # [B]
            mem_read = self._read(mem, state)
            out = self.model.step(state, batch.item_ids[:, t], batch.item_mask[:, t], mem_read)
            probs = F.softmax(out.action_logits, dim=-1)            # [B, A]

            absorb = probs[:, ACTION_ABSORB] * real
            mem = self.memory.write(mem, t, out.write_vector, absorb)
            new_state = out.new_state

            # Response opportunity after absorbing this item.
            resp_read = self._read(mem, new_state)
            resp_logits = self._respond(new_state, resp_read, batch)
            respond_gate = probs[:, ACTION_RESPOND] * real

            action_logits_list.append(out.action_logits)
            respond_logits_list.append(resp_logits)
            respond_gate_list.append(respond_gate)
            item_respond_gate_list.append(respond_gate)
            write_vecs_list.append(out.write_vector)
            append_gate_list.append(probs[:, ACTION_APPEND] * real)

            state = new_state
            states.append(state)

        # Post-stream reasoning steps (no new input) — also response chances.
        pad_ids = torch.full((b, 1), self.pad_id, dtype=torch.long, device=device)
        pad_mask = torch.zeros((b, 1), dtype=torch.float32, device=device)
        for _ in range(self.reasoning_hops - 1):
            mem_read = self._read(mem, state)
            out = self.model.step(state, pad_ids, pad_mask, mem_read)
            probs = F.softmax(out.action_logits, dim=-1)
            new_state = out.new_state
            resp_read = self._read(mem, new_state)
            respond_logits_list.append(self._respond(new_state, resp_read, batch))
            respond_gate_list.append(probs[:, ACTION_RESPOND])
            state = new_state
            states.append(state)

        # Aggregate the answer over all response chances, weighted by P(RESPOND).
        gates = torch.stack(respond_gate_list, dim=1)              # [B, S]
        rlog = torch.stack(respond_logits_list, dim=1)            # [B, S, K]
        w = gates / gates.sum(dim=1, keepdim=True).clamp(min=1e-6)
        answer_logits = (w.unsqueeze(-1) * rlog).sum(dim=1)        # [B, K]

        write_vecs = torch.stack(write_vecs_list, dim=1)          # [B, T, mem_dim]
        append_gates = torch.stack(append_gate_list, dim=1)       # [B, T]
        item_respond_gates = torch.stack(item_respond_gate_list, dim=1)  # [B, T]

        return {
            "answer_logits": answer_logits,
            "action_logits": torch.stack(action_logits_list, dim=1),  # [B, T, A]
            "respond_gates": item_respond_gates,                      # [B, T]
            "append_gates": append_gates,                            # [B, T]
            "write_vecs": write_vecs,                                # [B, T, mem_dim]
            "novelty_target": self._novelty(write_vecs, batch.step_mask),  # [B, T]
            "states": torch.stack(states, dim=1),                    # [B, *, state_dim]
            "memory_occupancy": self.memory.occupancy(mem),          # [B]
        }

    # -- long-term consolidation --------------------------------------------
    @torch.no_grad()
    def consolidate(self, out: Dict[str, torch.Tensor], batch) -> int:
        """Commit items the model chose to APPEND into long-term memory.

        Each item's write vector is consolidated with strength = its APPEND gate,
        carrying the item's text as provenance; facts from one episode are linked.
        Returns the number of entries added.
        """
        if self.long_term is None:
            return 0
        write_vecs = out["write_vecs"]      # [B, T, mem_dim]
        append_gates = out["append_gates"]  # [B, T]
        step_mask = batch.step_mask         # [B, T]
        added = 0
        for bi in range(write_vecs.shape[0]):
            idxs_real = [t for t in range(step_mask.shape[1]) if step_mask[bi, t] > 0]
            if not idxs_real:
                continue
            vecs = write_vecs[bi, idxs_real]                 # [k, mem_dim]
            gates = append_gates[bi, idxs_real]              # [k]
            texts = batch.item_texts[bi] if batch.item_texts is not None else None
            metas = (
                [{"text": texts[t], "kind": "learned"} for t in idxs_real]
                if texts is not None else None
            )
            added += len(self.long_term.consolidate(vecs, gates=gates, metas=metas))
        return added

    @torch.no_grad()
    def seed_world_facts(self, fact_texts, encoder, max_len: int) -> int:
        """Seed long-term memory with base 'facts we know about the world'.

        Each fact sentence is read through the model to produce a write vector,
        then stored with its text and ``kind="base"``. Returns entries added.
        """
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
        """Per-item chosen action, the chosen RESPOND step, and the answer."""
        out = self.forward(batch)
        item_actions = out["action_logits"].argmax(-1)          # [B, T]
        respond_step = out["respond_gates"].argmax(-1)          # [B]
        answers = out["answer_logits"].argmax(-1)               # [B]
        actions = []
        for i in range(item_actions.shape[0]):
            n = int(batch.step_mask[i].sum().item())
            actions.append([ACTION_NAMES[a] for a in item_actions[i, :n].tolist()])
        return {
            "actions": actions,
            "respond_step": respond_step.tolist(),
            "answers": answers.tolist(),
        }
