"""The Mind: drives an episode through the state-transition loop.

Given a batch of episodes, :class:`Mind` threads the consciousness state across
the context stream, lets the model gate writes into working memory step by step,
and produces a response at the question. It returns everything the loss needs
(answer logits, per-step action logits, the state trajectory) plus the final
memory for inspection.

This is the part that turns a per-step transition function into the actual
"absorb → store → recognize question → respond" behavior.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .long_term_memory import LongTermMemory
from .memory import MemoryState, WorkingMemory
from .model import ACTION_ABSORB, ACTION_NAMES, ConsciousnessTransformer


class Mind(nn.Module):
    """Stateful driver over an :class:`~nsm_ct.dataset.EpisodeBatch`.

    Args:
        model: The state-transition transformer.
        memory: The read/write working memory (the per-episode *local context*).
        answer_mode: ``"mc"`` (score options) or ``"open"`` (classify answer).
        reasoning_hops: Multi-hop passes over memory at the question.
        long_term: Optional persistent :class:`LongTermMemory`. When present, its
            retrieval is added to every memory read, and :meth:`consolidate`
            commits the local context into it after an episode.
    """

    def __init__(
        self,
        model: ConsciousnessTransformer,
        memory: WorkingMemory,
        answer_mode: str = "mc",
        reasoning_hops: int = 1,
        long_term: Optional[LongTermMemory] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.memory = memory
        self.answer_mode = answer_mode
        self.reasoning_hops = max(1, reasoning_hops)
        self.long_term = long_term

    def _read(self, mem: MemoryState, state: torch.Tensor) -> torch.Tensor:
        """Read local context, plus long-term memory when present."""
        read = self.memory.read(mem, state)
        if self.long_term is not None:
            read = read + self.long_term.read(state)
        return read

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Unroll the loop over ``batch`` and return outputs for the loss."""
        device = batch.ctx_ids.device
        b, num_steps, _ = batch.ctx_ids.shape
        state = self.model.initial_state(b, device)
        mem = self.memory.init_state(b, max(num_steps, 1), device)

        states = [state]
        ctx_action_logits = []
        for t in range(num_steps):
            mem_read = self._read(mem, state)
            out = self.model.step(state, batch.ctx_ids[:, t], batch.ctx_mask[:, t], mem_read)
            ctx_action_logits.append(out.action_logits)
            # Gate the write by the ABSORB probability, and only on real steps.
            absorb_gate = F.softmax(out.action_logits, dim=-1)[:, ACTION_ABSORB] * batch.step_mask[:, t]
            mem = self.memory.write(mem, t, out.write_vector, absorb_gate)
            state = out.new_state
            states.append(state)

        # Question step: retrieve, transition, then answer from the updated state.
        mem_read_pre = self._read(mem, state)
        qout = self.model.step(state, batch.q_ids, batch.q_mask, mem_read_pre)
        q_state = qout.new_state
        states.append(q_state)

        # Multi-hop reasoning: re-read memory and re-process the question with the
        # updated state for `reasoning_hops - 1` extra passes ("reason with
        # states"). hops == 1 leaves behavior identical to the single-pass loop.
        for _ in range(self.reasoning_hops - 1):
            mem_read_hop = self._read(mem, q_state)
            hop_out = self.model.step(q_state, batch.q_ids, batch.q_mask, mem_read_hop)
            q_state = hop_out.new_state
            states.append(q_state)

        mem_read_post = self._read(mem, q_state)

        if self.answer_mode == "mc":
            answer_logits = self.model.respond_mc(q_state, mem_read_post, batch.opt_ids, batch.opt_mask)
        else:
            answer_logits = self.model.respond_open(q_state, mem_read_post)

        ctx_stack = (
            torch.stack(ctx_action_logits, dim=1)
            if ctx_action_logits
            else torch.zeros(b, 0, qout.action_logits.shape[-1], device=device)
        )
        return {
            "answer_logits": answer_logits,
            "ctx_action_logits": ctx_stack,         # [B, T, A]
            "q_action_logits": qout.action_logits,  # [B, A]
            "states": torch.stack(states, dim=1),   # [B, T+2, state_dim]
            "memory_occupancy": self.memory.occupancy(mem),  # [B]
            # For long-term consolidation: the written local-context slots, the
            # mask of which are real, the final state, and the retention gate.
            "working_slots": mem.slots,             # [B, S, mem_dim]
            "working_filled": mem.filled,           # [B, S]
            "final_state": q_state,                 # [B, state_dim]
            "consolidate_gate": self.model.consolidate_gate(q_state),  # [B]
        }

    @torch.no_grad()
    def consolidate(self, out: Dict[str, torch.Tensor]) -> int:
        """Commit the episode's local context into long-term memory.

        For each example, the written working-memory slots (the absorbed facts)
        are consolidated into :attr:`long_term`, scaled by the state's retention
        gate, and the facts from one episode are linked as a connection group.

        Returns the number of entries added (0 if no long-term memory).
        """
        if self.long_term is None:
            return 0
        slots = out["working_slots"]      # [B, S, mem_dim]
        filled = out["working_filled"]    # [B, S]
        gate = out["consolidate_gate"]    # [B]
        added = 0
        for b in range(slots.shape[0]):
            mask = filled[b] > 0.1
            vecs = slots[b][mask]
            if vecs.shape[0] == 0:
                continue
            gates = gate[b].expand(vecs.shape[0])
            idxs = self.long_term.consolidate(vecs, gates=gates)
            added += len(idxs)
        return added

    @torch.no_grad()
    def trace(self, batch) -> Dict[str, object]:
        """Human-readable per-step actions and predicted answers (for smoke tests)."""
        out = self.forward(batch)
        ctx_actions = out["ctx_action_logits"].argmax(-1)        # [B, T]
        q_actions = out["q_action_logits"].argmax(-1)            # [B]
        answers = out["answer_logits"].argmax(-1)                # [B]
        actions = []
        for i in range(ctx_actions.shape[0]):
            steps = [ACTION_NAMES[a] for a in ctx_actions[i].tolist()]
            steps.append(ACTION_NAMES[q_actions[i].item()])
            actions.append(steps)
        return {"actions": actions, "answers": answers.tolist()}
