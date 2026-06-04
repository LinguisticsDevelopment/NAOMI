"""The Consciousness Transformer as a state-transition function.

One **step** consumes the current consciousness state, one input object (a
sentence's token ids), and a read from working memory, and produces:

* a **new state** (the transition — this is the spine of the loop),
* an **action** over ``{ABSORB, RESPOND, SKIP}`` (the gate that decides whether
  to write to memory and when to answer),
* a **write vector** (the content to commit to memory if ABSORB fires), and
* (at the question) a **response**, either by scoring multiple-choice options or
  by classifying an open-ended answer.

The transformer attends over a short assembly ``[state | memory | input...]``.
It is intentionally small; the research is in the loop and the state, not the
block (see RESEARCH_NOTES).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig

# Action repertoire for the gating head. The model CHOOSES among these per item;
# nothing supervises the choice positionally (see losses.py / RESEARCH_NOTES).
ACTION_ABSORB = 0   # write the item into local (working) memory
ACTION_APPEND = 1   # commit the item into long-term memory (world facts)
ACTION_RESPOND = 2  # emit a response now
ACTION_SKIP = 3     # ignore the item
NUM_ACTIONS = 4
ACTION_NAMES = ["ABSORB", "APPEND", "RESPOND", "SKIP"]


@dataclass
class StepOutput:
    """Outputs of a single state-transition step.

    Attributes:
        new_state: ``[B, state_dim]`` the next consciousness state.
        action_logits: ``[B, NUM_ACTIONS]`` logits over ABSORB/RESPOND/SKIP.
        write_vector: ``[B, mem_dim]`` content to (gatedly) write to memory.
        pooled: ``[B, d_model]`` the raw state-slot encoding (for reuse).
    """

    new_state: torch.Tensor
    action_logits: torch.Tensor
    write_vector: torch.Tensor
    pooled: torch.Tensor


class ConsciousnessTransformer(nn.Module):
    """A transformer that maps ``(state, input, memory) -> new state + actions``.

    Args:
        vocab_size: Tokenizer vocabulary size.
        answer_vocab_size: Number of distinct open-ended answers (for the
            open-ended response classifier). Use 1 if only MC is needed.
        cfg: Model hyperparameters.
    """

    def __init__(self, vocab_size: int, answer_vocab_size: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        d = cfg.d_model

        self.token_embedding = nn.Embedding(vocab_size, d)
        # +2 positions for the state and memory slots prepended to every input.
        self.position_embedding = nn.Embedding(cfg.max_sentence_len + 2, d)
        self.state_in = nn.Linear(cfg.consciousness_dim, d)
        self.memory_in = nn.Linear(cfg.memory_dim, d)
        self.dropout = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Learned, abstract initial state (its *meaning* is deliberately TBD).
        self.state_init = nn.Parameter(torch.zeros(cfg.consciousness_dim))

        # Heads.
        self.state_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.consciousness_dim)
        )
        self.action_head = nn.Linear(d, NUM_ACTIONS)
        self.write_head = nn.Linear(d, cfg.memory_dim)

        # Response: a query built from (new_state, memory_read).
        self.response_query = nn.Sequential(
            nn.Linear(cfg.consciousness_dim + cfg.memory_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        self.option_proj = nn.Linear(d, d)                       # MC option embeddings
        self.answer_classifier = nn.Linear(d, max(answer_vocab_size, 1))  # open-ended

    # -- one transition step -------------------------------------------------
    def step(
        self,
        state: torch.Tensor,        # [B, state_dim]
        input_ids: torch.Tensor,    # [B, L]
        input_mask: torch.Tensor,   # [B, L] (1 real, 0 pad)
        mem_read: torch.Tensor,     # [B, mem_dim]
    ) -> StepOutput:
        """Run one state transition over a single input sentence."""
        b, length = input_ids.shape
        positions = torch.arange(length + 2, device=input_ids.device).unsqueeze(0).expand(b, -1)

        tok = self.token_embedding(input_ids)                    # [B, L, d]
        state_tok = self.state_in(state).unsqueeze(1)            # [B, 1, d]
        mem_tok = self.memory_in(mem_read).unsqueeze(1)          # [B, 1, d]
        seq = torch.cat([state_tok, mem_tok, tok], dim=1)        # [B, L+2, d]
        seq = seq + self.position_embedding(positions)
        seq = self.dropout(seq)

        # State and memory slots are always attended; input pad positions masked.
        prefix_mask = torch.zeros(b, 2, device=input_ids.device, dtype=torch.bool)
        key_padding_mask = torch.cat([prefix_mask, input_mask == 0], dim=1)
        hidden = self.encoder(seq, src_key_padding_mask=key_padding_mask)

        pooled = hidden[:, 0, :]                                 # state slot output
        return StepOutput(
            new_state=self.state_head(pooled),
            action_logits=self.action_head(pooled),
            write_vector=self.write_head(pooled),
            pooled=pooled,
        )

    # -- responses -----------------------------------------------------------
    def _query(self, state: torch.Tensor, mem_read: torch.Tensor) -> torch.Tensor:
        return self.response_query(torch.cat([state, mem_read], dim=-1))   # [B, d]

    def respond_mc(
        self,
        state: torch.Tensor,        # [B, state_dim]
        mem_read: torch.Tensor,     # [B, mem_dim]
        option_ids: torch.Tensor,   # [B, K, Lo]
        option_mask: torch.Tensor,  # [B, K, Lo]
    ) -> torch.Tensor:
        """Score each multiple-choice option. Returns ``[B, K]`` logits."""
        query = self._query(state, mem_read)                     # [B, d]
        emb = self.token_embedding(option_ids)                   # [B, K, Lo, d]
        denom = option_mask.sum(-1, keepdim=True).clamp(min=1.0)  # [B, K, 1]
        opt = (emb * option_mask.unsqueeze(-1)).sum(2) / denom    # [B, K, d] mean-pool
        opt = self.option_proj(opt)                              # [B, K, d]
        return (opt * query.unsqueeze(1)).sum(-1)                # [B, K]

    def respond_open(self, state: torch.Tensor, mem_read: torch.Tensor) -> torch.Tensor:
        """Open-ended answer logits over the answer vocabulary. Returns ``[B, A]``."""
        return self.answer_classifier(self._query(state, mem_read))

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Broadcast the learned initial state to a batch."""
        return self.state_init.to(device).unsqueeze(0).expand(batch_size, -1).contiguous()
