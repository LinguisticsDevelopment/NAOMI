"""The Consciousness Transformer as a state-transition function.

One **step** consumes the current consciousness state, one input object (a
sentence's token ids), and a read from working memory, and produces:

* a **new state** (the transition — this is the spine of the loop),
* an **action** over ``{ABSORB, APPEND, RESPOND, SKIP}`` (soft gates for local
  write / long-term commit / answer), a **control** distribution over
  ``{READ, THINK, RESPOND}`` (the self-controlled loop), and a **trust** scalar,
* a **write vector** (the content to commit to memory if ABSORB fires), and
* a **response**, either by scoring multiple-choice options or classifying an
  open-ended answer.

The transformer attends over a short assembly ``[state | memory | input...]``.
It is intentionally small; the research is in the loop and the state, not the
block (see RESEARCH_NOTES).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig
from .structure import MAX_DEPTH, NUM_ROLES

# Action repertoire for the gating head. The model CHOOSES among these per item;
# nothing supervises the choice positionally (see losses.py / RESEARCH_NOTES).
ACTION_ABSORB = 0   # write the item into local (working) memory
ACTION_APPEND = 1   # commit the item into long-term memory (world facts)
ACTION_RESPOND = 2  # emit a response now
ACTION_SKIP = 3     # ignore the item
NUM_ACTIONS = 4
ACTION_NAMES = ["ABSORB", "APPEND", "RESPOND", "SKIP"]

# Control repertoire for the self-controlled loop. Each tick the model chooses to
# READ the next input, THINK internally (reason over memory, no new input), or
# RESPOND. Emergent — shaped only by the answer loss (see agent.py).
CONTROL_READ = 0
CONTROL_THINK = 1
CONTROL_RESPOND = 2
NUM_CONTROL = 3
CONTROL_NAMES = ["READ", "THINK", "RESPOND"]


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
        # Self-controlled loop: read / think / respond (from the current state).
        # Start biased toward READ so the loop ingests its input before it learns
        # to think/respond — this makes the controlled loop trainable, while the
        # behavior itself stays fully emergent (the bias is just an init).
        self.control_head = nn.Linear(cfg.consciousness_dim, NUM_CONTROL)
        with torch.no_grad():
            self.control_head.bias.copy_(torch.tensor([2.0, 0.0, 0.0]))
        self.write_head = nn.Linear(d, cfg.memory_dim)
        # Emergent trust: judges an item against what memory already holds. It
        # scales how strongly the item influences memory (and thus the answer),
        # so trustworthy/corroborated info is used and contradicted info is
        # discounted. Learned purely via answer correctness — no trust labels.
        self.trust_head = nn.Sequential(
            nn.Linear(cfg.consciousness_dim + cfg.memory_dim, d), nn.GELU(), nn.Linear(d, 1)
        )

        # Response: a query built from (new_state, memory_read).
        self.response_query = nn.Sequential(
            nn.Linear(cfg.consciousness_dim + cfg.memory_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        self.option_proj = nn.Linear(d, d)                       # MC option embeddings
        self.answer_classifier = nn.Linear(d, max(answer_vocab_size, 1))  # open-ended

        # Tree-aware structure (created LAST so token-mode init is unchanged):
        # per-token parse role + depth, added onto word embeddings. Zero-init, so
        # structure starts as a no-op and is learned only if it helps — and a
        # lossy/noisy parse can never corrupt the word stream.
        self.role_embedding = nn.Embedding(NUM_ROLES, d)
        self.depth_embedding = nn.Embedding(MAX_DEPTH, d)
        nn.init.zeros_(self.role_embedding.weight)
        nn.init.zeros_(self.depth_embedding.weight)

    # -- one transition step -------------------------------------------------
    def _transition(
        self,
        state: torch.Tensor,       # [B, state_dim]
        input_tok: torch.Tensor,   # [B, L, d] (already embedded input tokens)
        input_pad: torch.Tensor,   # [B, L] bool, True = pad (masked out)
        mem_read: torch.Tensor,    # [B, mem_dim]
    ) -> StepOutput:
        """Shared state transition over a prepared input-token assembly."""
        b, length, _ = input_tok.shape
        positions = torch.arange(length + 2, device=state.device).unsqueeze(0).expand(b, -1)

        state_tok = self.state_in(state).unsqueeze(1)            # [B, 1, d]
        mem_tok = self.memory_in(mem_read).unsqueeze(1)          # [B, 1, d]
        seq = torch.cat([state_tok, mem_tok, input_tok], dim=1)  # [B, L+2, d]
        seq = seq + self.position_embedding(positions)
        seq = self.dropout(seq)

        # State and memory slots are always attended; input pad positions masked.
        prefix_mask = torch.zeros(b, 2, device=state.device, dtype=torch.bool)
        key_padding_mask = torch.cat([prefix_mask, input_pad], dim=1)
        hidden = self.encoder(seq, src_key_padding_mask=key_padding_mask)

        pooled = hidden[:, 0, :]                                 # state slot output
        return StepOutput(
            new_state=self.state_head(pooled),
            action_logits=self.action_head(pooled),
            write_vector=self.write_head(pooled),
            pooled=pooled,
        )

    def step(
        self,
        state: torch.Tensor,        # [B, state_dim]
        input_ids: torch.Tensor,    # [B, L]
        input_mask: torch.Tensor,   # [B, L] (1 real, 0 pad)
        mem_read: torch.Tensor,     # [B, mem_dim]
        input_roles: torch.Tensor = None,   # [B, L] per-token parse role id (optional)
        input_depths: torch.Tensor = None,  # [B, L] per-token parse depth (optional)
    ) -> StepOutput:
        """One state transition over a single input sentence (tokens + structure)."""
        tok = self.token_embedding(input_ids)
        if input_roles is not None:
            tok = tok + self.role_embedding(input_roles)            # parse role (zero-init)
        if input_depths is not None:
            tok = tok + self.depth_embedding(input_depths.clamp(max=MAX_DEPTH - 1))
        return self._transition(state, tok, input_mask == 0, mem_read)

    def control_gate(self, state: torch.Tensor) -> torch.Tensor:
        """Loop-control distribution over {READ, THINK, RESPOND} (``[B, 3]``)."""
        return torch.softmax(self.control_head(state), dim=-1)

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

    def trust_gate(self, state: torch.Tensor, mem_read: torch.Tensor) -> torch.Tensor:
        """How much to trust the current item given memory (``[B]`` in [0, 1]).

        Compares the post-item state against what memory already holds; corroborated
        info can be trusted, contradicted info discounted. Emergent — trained only
        through its effect on the answer.
        """
        return torch.sigmoid(self.trust_head(torch.cat([state, mem_read], dim=-1))).squeeze(-1)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Broadcast the learned initial state to a batch."""
        return self.state_init.to(device).unsqueeze(0).expand(batch_size, -1).contiguous()
