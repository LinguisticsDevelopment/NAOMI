"""Turn a :class:`ComprehensionExample` into model-ready tensors.

This module is the integration seam: it composes the (mocked) parser, semantic
mapper, serializer, memory store, and tokenizer to build the four model inputs
named in the brief — tokenized text, serialized parse tree, consciousness
vector, and retrieved memory — for every answer option.

Each example is expanded into **4 rows**, one per multiple-choice option, each
row being a single causal-LM sequence:

    [CONSC] [MEM] <text> [SEP] <serialized parse> [ANS] <option> [EOS]

``[CONSC]`` / ``[MEM]`` are placeholder token positions whose embeddings get the
consciousness / memory vectors added in the model. The answer region (the option
tokens + ``[EOS]``) is what the LM loss and the option-scoring read from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .config import Config
from .data_structures import ComprehensionExample
from .memory import AbstractMemory
from .parser_interface import AbstractParser
from .semantic_mapper import AbstractSemanticMapper
from .serialization import serialize_parse_tree
from .tokenizer import ANS, CONSC, EOS, MEM, SEP, SimpleTokenizer

# Segment ids for the segment embedding.
SEG_PREFIX = 0  # [CONSC]/[MEM]
SEG_TEXT = 1
SEG_PARSE = 2
SEG_ANSWER = 3
NUM_SEGMENTS = 4


@dataclass
class OptionRow:
    """One causal-LM sequence for a single answer option."""

    input_ids: List[int]
    segment_ids: List[int]
    answer_mask: List[int]  # 1 on tokens that belong to the answer/response region


@dataclass
class EncodedExample:
    """All four option rows for an example, plus shared state vectors."""

    rows: List[OptionRow]
    consciousness: np.ndarray  # [consciousness_dim]
    memory: np.ndarray         # [memory_dim]
    answer_idx: int


class FeatureBuilder:
    """Composes the (mock) NLP stack into encoded model inputs.

    Args:
        tokenizer: Vocabulary/encoder.
        parser: Syntactic parser (mock or real).
        semantic_mapper: Maps parse -> NSM meaning + seeds consciousness.
        memory: Retrieved-memory provider.
        config: Holds the consciousness/memory dimensions and max length.
    """

    def __init__(
        self,
        tokenizer: SimpleTokenizer,
        parser: AbstractParser,
        semantic_mapper: AbstractSemanticMapper,
        memory: AbstractMemory,
        config: Config,
    ) -> None:
        self.tok = tokenizer
        self.parser = parser
        self.mapper = semantic_mapper
        self.memory = memory
        self.config = config

    def build(self, example: ComprehensionExample) -> EncodedExample:
        """Build the :class:`EncodedExample` for one comprehension example."""
        context = example.context

        # --- parse + serialize (mock parser) -------------------------------
        tree = self.parser.parse(context)
        parse_tokens = serialize_parse_tree(tree)

        # --- semantic mapping -> consciousness seed (mock) -----------------
        sem = self.mapper.map(tree)
        consciousness = sem.to_consciousness_state(self.config.model.consciousness_dim).vector

        # --- retrieved memory (mock) ---------------------------------------
        memory_vec = self.memory.retrieve(context)

        # --- shared prefix: [CONSC] [MEM] text [SEP] parse -----------------
        text_ids = self.tok.encode(context)
        parse_ids = self.tok.encode_tokens(parse_tokens)

        prefix_ids = [self.tok.id_of(CONSC), self.tok.id_of(MEM)]
        prefix_seg = [SEG_PREFIX, SEG_PREFIX]

        body_ids = text_ids + [self.tok.id_of(SEP)] + parse_ids
        body_seg = [SEG_TEXT] * len(text_ids) + [SEG_TEXT] + [SEG_PARSE] * len(parse_ids)

        rows: List[OptionRow] = []
        for option in example.options:
            opt_ids = self.tok.encode(option)
            ans_ids = [self.tok.id_of(ANS)] + opt_ids + [self.tok.id_of(EOS)]
            ans_seg = [SEG_ANSWER] * len(ans_ids)
            # answer_mask: 1 on the option tokens and EOS (predicted region),
            # 0 on the [ANS] marker (it is context for the first prediction).
            ans_mask = [0] + [1] * (len(opt_ids) + 1)

            input_ids = prefix_ids + body_ids + ans_ids
            segment_ids = prefix_seg + body_seg + ans_seg
            answer_mask = [0] * (len(prefix_ids) + len(body_ids)) + ans_mask

            # Truncate from the left of the body if we blow past max_seq_len,
            # but always keep the prefix and the full answer region intact.
            max_len = self.config.model.max_seq_len
            if len(input_ids) > max_len:
                overflow = len(input_ids) - max_len
                # drop `overflow` tokens from the start of the body region
                start = len(prefix_ids)
                input_ids = input_ids[:start] + input_ids[start + overflow:]
                segment_ids = segment_ids[:start] + segment_ids[start + overflow:]
                answer_mask = answer_mask[:start] + answer_mask[start + overflow:]

            rows.append(OptionRow(input_ids=input_ids, segment_ids=segment_ids, answer_mask=answer_mask))

        return EncodedExample(
            rows=rows,
            consciousness=consciousness,
            memory=memory_vec,
            answer_idx=example.answer_idx,
        )


@dataclass
class Batch:
    """A padded, tensorized batch ready for :class:`ConsciousnessTransformer`.

    ``N = batch_size * 4`` rows (4 options per example), laid out so that rows
    ``[4*b : 4*b+4]`` belong to example ``b``.
    """

    input_ids: torch.Tensor       # [N, T] long
    segment_ids: torch.Tensor     # [N, T] long
    attention_mask: torch.Tensor  # [N, T] float (1 real, 0 pad)
    answer_mask: torch.Tensor     # [N, T] float
    consciousness: torch.Tensor   # [N, Cdim] float
    memory: torch.Tensor          # [N, Mdim] float
    answer_idx: torch.Tensor      # [B] long
    num_options: int = 4

    def to(self, device: torch.device) -> "Batch":
        """Move all tensors to ``device``."""
        return Batch(
            input_ids=self.input_ids.to(device),
            segment_ids=self.segment_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            answer_mask=self.answer_mask.to(device),
            consciousness=self.consciousness.to(device),
            memory=self.memory.to(device),
            answer_idx=self.answer_idx.to(device),
            num_options=self.num_options,
        )


def collate(encoded: List[EncodedExample], pad_id: int) -> Batch:
    """Pad and stack a list of :class:`EncodedExample` into a :class:`Batch`."""
    all_rows: List[OptionRow] = []
    consc: List[np.ndarray] = []
    mem: List[np.ndarray] = []
    answer_idx: List[int] = []
    num_options = len(encoded[0].rows)

    for ex in encoded:
        for row in ex.rows:
            all_rows.append(row)
        # consciousness/memory are shared across the 4 rows; repeat per row.
        for _ in ex.rows:
            consc.append(ex.consciousness)
            mem.append(ex.memory)
        answer_idx.append(ex.answer_idx)

    max_t = max(len(r.input_ids) for r in all_rows)
    n = len(all_rows)

    input_ids = torch.full((n, max_t), pad_id, dtype=torch.long)
    segment_ids = torch.zeros((n, max_t), dtype=torch.long)
    attention_mask = torch.zeros((n, max_t), dtype=torch.float32)
    answer_mask = torch.zeros((n, max_t), dtype=torch.float32)

    for i, row in enumerate(all_rows):
        t = len(row.input_ids)
        input_ids[i, :t] = torch.tensor(row.input_ids, dtype=torch.long)
        segment_ids[i, :t] = torch.tensor(row.segment_ids, dtype=torch.long)
        attention_mask[i, :t] = 1.0
        answer_mask[i, :t] = torch.tensor(row.answer_mask, dtype=torch.float32)

    return Batch(
        input_ids=input_ids,
        segment_ids=segment_ids,
        attention_mask=attention_mask,
        answer_mask=answer_mask,
        consciousness=torch.tensor(np.stack(consc), dtype=torch.float32),
        memory=torch.tensor(np.stack(mem), dtype=torch.float32),
        answer_idx=torch.tensor(answer_idx, dtype=torch.long),
        num_options=num_options,
    )
