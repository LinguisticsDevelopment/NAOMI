"""The Consciousness Transformer.

A small causal Transformer with **two output heads**, per the brief:

1. **Language-modeling head** — predicts response tokens (and, by scoring each
   candidate option's likelihood, drives multiple-choice answer prediction).
2. **Consciousness state-transition head** — reads the ``[CONSC]`` slot and
   predicts the next consciousness state.

Inputs (assembled by :mod:`nsm_ct.features`): tokenized text + serialized parse
trees + a consciousness vector + a retrieved-memory vector. The consciousness
and memory vectors are injected by *adding* their linear projections onto the
embeddings of the reserved ``[CONSC]`` and ``[MEM]`` token positions.

The architecture is intentionally tiny and conventional; the *research* is in
what the consciousness state means and how meaning is composed, not in the
transformer block (see RESEARCH_NOTES).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .features import NUM_SEGMENTS, Batch


@dataclass
class ModelOutput:
    """Bundle of everything a forward pass produces.

    Attributes:
        lm_logits: ``[N, T, vocab]`` next-token logits.
        option_logits: ``[B, num_options]`` length-normalized log-likelihood of
            each option's answer region — used for answer prediction.
        lm_loss_per_row: ``[N]`` mean negative log-likelihood over each row's
            answer region.
        next_consciousness: ``[N, consciousness_dim]`` predicted next state.
        consciousness_readout: ``[N, d_model]`` raw ``[CONSC]``-slot encoding.
    """

    lm_logits: torch.Tensor
    option_logits: torch.Tensor
    lm_loss_per_row: torch.Tensor
    next_consciousness: torch.Tensor
    consciousness_readout: torch.Tensor


class ConsciousnessTransformer(nn.Module):
    """Causal Transformer with LM and consciousness-transition heads.

    Args:
        vocab_size: Size of the tokenizer vocabulary.
        cfg: Model hyperparameters.
    """

    def __init__(self, vocab_size: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.segment_embedding = nn.Embedding(NUM_SEGMENTS, cfg.d_model)

        # Inject the consciousness/memory vectors into the [CONSC]/[MEM] slots.
        self.consciousness_in = nn.Linear(cfg.consciousness_dim, cfg.d_model)
        self.memory_in = nn.Linear(cfg.memory_dim, cfg.d_model)

        self.dropout = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Head 1: language modeling.
        self.lm_head = nn.Linear(cfg.d_model, vocab_size)
        # Head 2: consciousness state transition.
        self.consciousness_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.consciousness_dim),
        )

    # -- internals -----------------------------------------------------------
    def _embed(self, batch: Batch) -> torch.Tensor:
        """Build the input embedding sequence with state vectors injected."""
        n, t = batch.input_ids.shape
        positions = torch.arange(t, device=batch.input_ids.device).unsqueeze(0).expand(n, t)

        emb = (
            self.token_embedding(batch.input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(batch.segment_ids)
        )
        # Inject consciousness at slot 0, memory at slot 1.
        emb[:, 0, :] = emb[:, 0, :] + self.consciousness_in(batch.consciousness)
        emb[:, 1, :] = emb[:, 1, :] + self.memory_in(batch.memory)
        return self.dropout(emb)

    # -- forward -------------------------------------------------------------
    def forward(self, batch: Batch) -> ModelOutput:
        """Run a full forward pass over a :class:`Batch`."""
        n, t = batch.input_ids.shape
        emb = self._embed(batch)

        # Bool masks (True == "not allowed to attend"); same dtype for both so
        # nn.Transformer does not warn about mixed mask types.
        causal_mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=emb.device), diagonal=1
        )
        key_padding_mask = batch.attention_mask == 0  # True where pad

        hidden = self.encoder(
            emb, mask=causal_mask, src_key_padding_mask=key_padding_mask
        )  # [N, T, d]

        lm_logits = self.lm_head(hidden)  # [N, T, V]

        # Per-token log-prob of the *actual* next token (causal shift).
        log_probs = F.log_softmax(lm_logits, dim=-1)
        shifted_lp = log_probs[:, :-1, :]            # predicts tokens at 1..T-1
        targets = batch.input_ids[:, 1:]             # [N, T-1]
        token_lp = shifted_lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [N, T-1]

        # Restrict to the answer region (the response).
        ans_mask = batch.answer_mask[:, 1:]          # [N, T-1]
        denom = ans_mask.sum(dim=1).clamp(min=1.0)
        row_lp = (token_lp * ans_mask).sum(dim=1) / denom        # mean log-prob
        lm_loss_per_row = -row_lp                                # [N]

        # Answer prediction: length-normalized option log-likelihood -> [B, opts].
        option_logits = row_lp.view(-1, batch.num_options)       # [B, num_options]

        # Consciousness transition head reads the [CONSC] slot (position 0).
        consciousness_readout = hidden[:, 0, :]                  # [N, d]
        next_consciousness = self.consciousness_head(consciousness_readout)  # [N, Cdim]

        return ModelOutput(
            lm_logits=lm_logits,
            option_logits=option_logits,
            lm_loss_per_row=lm_loss_per_row,
            next_consciousness=next_consciousness,
            consciousness_readout=consciousness_readout,
        )

    @torch.no_grad()
    def predict(self, batch: Batch) -> torch.Tensor:
        """Return the predicted option index per example: ``[B]`` long."""
        out = self.forward(batch)
        return out.option_logits.argmax(dim=-1)
