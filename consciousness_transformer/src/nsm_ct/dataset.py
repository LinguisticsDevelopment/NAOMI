"""Episode encoding, batching, and dataset plumbing.

Turns :class:`~nsm_ct.episode.Episode` objects into padded tensor
:class:`EpisodeBatch` es the :class:`~nsm_ct.agent.Mind` can unroll. The input
encoder (token or parser) is applied to the context statements and the question;
options/answers are plain-tokenized labels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .config import Config
from .episode import Episode
from .input_encoder import AbstractInputEncoder
from .nsm_primes import PRIME_NAMES
from .tokenizer import SimpleTokenizer

# Parse-tree label vocabulary (NodeType + ConnectionType names) so the optional
# parser encoder's serialized streams are in-vocab. Hard-coded to avoid importing
# quantum_parser; kept loosely in sync with its enums + the mock parser labels.
PARSE_LABELS: List[str] = [
    # NodeTypes
    "NIL", "NOUN", "VERBAL", "PREDICATE", "NOMINAL", "CLAUSE", "SPECIFIER",
    "DESCRIPTOR", "MODIFIER", "COORD", "SUBOORD", "PREP", "PREP_SPEC",
    "PREP_DESC", "PREP_MIX", "ROOT", "S", "CONTENT", "FUNC", "PUNCT", "NUM",
    # ConnectionTypes
    "SUBJECT", "OBJECT", "INDIRECT_OBJECT", "SUBJECT_COMPLEMENT", "DESCRIPTION",
    "SPECIFICATION", "MODIFICATION", "COMPLEMENT", "COORDINATION", "PREPOSITION",
    "PREPOSITION_FROM", "PREPOSITION_TO", "SUBORDINATION", "SUBORDINATION_FROM",
    "SUBORDINATION_TO", "APPOSITION", "REL",
]


# ---------------------------------------------------------------------------
# Tokenizer / answer vocabulary
# ---------------------------------------------------------------------------
def build_tokenizer(episodes: List[Episode]) -> SimpleTokenizer:
    """Build a tokenizer over the episodes, NSM primes, and parse labels."""
    texts: List[str] = []
    for ep in episodes:
        texts.extend(ep.context)
        texts.append(ep.question)
        texts.append(ep.answer_text)
        if ep.options:
            texts.extend(ep.options)
    extra = list(PRIME_NAMES) + PARSE_LABELS
    return SimpleTokenizer.build(texts, extra_tokens=extra)


def build_answer_vocab(episodes: List[Episode]) -> Dict[str, int]:
    """Map distinct open-ended answers to ids (for ``answer_mode == 'open'``)."""
    answers = sorted({ep.answer_text for ep in episodes})
    return {a: i for i, a in enumerate(answers)}


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_episodes(
    episodes: List[Episode], val_fraction: float, seed: int = 0
) -> Tuple[List[Episode], List[Episode]]:
    """Deterministically split into (train, val)."""
    rng = random.Random(seed)
    shuffled = list(episodes)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def _truncate(ids: List[int], max_len: int) -> List[int]:
    return ids[:max_len] if len(ids) > max_len else ids


def encode_episode(
    ep: Episode,
    encoder: AbstractInputEncoder,
    tokenizer: SimpleTokenizer,
    answer_vocab: Dict[str, int],
    cfg: Config,
) -> Dict[str, object]:
    """Encode one episode into id lists (padding happens in :func:`collate`)."""
    max_len = cfg.model.max_sentence_len
    ctx_ids = [_truncate(encoder.encode(s), max_len) or [tokenizer.pad_id] for s in ep.context]
    q_ids = _truncate(encoder.encode(ep.question), max_len) or [tokenizer.pad_id]

    if cfg.data.answer_mode == "mc":
        if not ep.options:
            raise ValueError("answer_mode='mc' but episode has no options")
        opt_ids = [_truncate(tokenizer.encode(o), max_len) or [tokenizer.pad_id] for o in ep.options]
        answer_target = int(ep.answer_idx)
    else:
        opt_ids = []
        answer_target = answer_vocab[ep.answer_text]

    return {
        "ctx_ids": ctx_ids,
        "q_ids": q_ids,
        "opt_ids": opt_ids,
        "answer_target": answer_target,
    }


@dataclass
class EpisodeBatch:
    """A padded, tensorized batch of episodes."""

    ctx_ids: torch.Tensor       # [B, T, L]
    ctx_mask: torch.Tensor      # [B, T, L]
    step_mask: torch.Tensor     # [B, T]  (1 = real statement)
    q_ids: torch.Tensor         # [B, L]
    q_mask: torch.Tensor        # [B, L]
    opt_ids: torch.Tensor       # [B, K, Lo]
    opt_mask: torch.Tensor      # [B, K, Lo]
    answer_target: torch.Tensor  # [B]

    def to(self, device: torch.device) -> "EpisodeBatch":
        return EpisodeBatch(
            ctx_ids=self.ctx_ids.to(device),
            ctx_mask=self.ctx_mask.to(device),
            step_mask=self.step_mask.to(device),
            q_ids=self.q_ids.to(device),
            q_mask=self.q_mask.to(device),
            opt_ids=self.opt_ids.to(device),
            opt_mask=self.opt_mask.to(device),
            answer_target=self.answer_target.to(device),
        )


def _pad_2d(seqs: List[List[int]], rows: int, length: int, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a ragged list of id-lists into ``[rows, length]`` ids + mask."""
    ids = torch.full((rows, length), pad_id, dtype=torch.long)
    mask = torch.zeros((rows, length), dtype=torch.float32)
    for i, seq in enumerate(seqs):
        n = min(len(seq), length)
        if n:
            ids[i, :n] = torch.tensor(seq[:n], dtype=torch.long)
            mask[i, :n] = 1.0
    return ids, mask


def collate(items: List[Dict[str, object]], pad_id: int) -> EpisodeBatch:
    """Pad and stack encoded episodes into an :class:`EpisodeBatch`."""
    b = len(items)
    max_t = max(1, max(len(it["ctx_ids"]) for it in items))
    max_l = max(
        1,
        max((len(s) for it in items for s in it["ctx_ids"]), default=1),
        max(len(it["q_ids"]) for it in items),
    )
    is_mc = len(items[0]["opt_ids"]) > 0
    num_opts = len(items[0]["opt_ids"]) if is_mc else 1
    max_lo = max(
        1,
        max((len(o) for it in items for o in it["opt_ids"]), default=1),
    )

    ctx_ids = torch.full((b, max_t, max_l), pad_id, dtype=torch.long)
    ctx_mask = torch.zeros((b, max_t, max_l), dtype=torch.float32)
    step_mask = torch.zeros((b, max_t), dtype=torch.float32)
    q_ids = torch.full((b, max_l), pad_id, dtype=torch.long)
    q_mask = torch.zeros((b, max_l), dtype=torch.float32)
    opt_ids = torch.full((b, num_opts, max_lo), pad_id, dtype=torch.long)
    opt_mask = torch.zeros((b, num_opts, max_lo), dtype=torch.float32)
    answer_target = torch.zeros((b,), dtype=torch.long)

    for i, it in enumerate(items):
        ctx = it["ctx_ids"]
        ids, msk = _pad_2d(ctx, len(ctx), max_l, pad_id)
        ctx_ids[i, : len(ctx)] = ids
        ctx_mask[i, : len(ctx)] = msk
        step_mask[i, : len(ctx)] = 1.0

        qn = min(len(it["q_ids"]), max_l)
        q_ids[i, :qn] = torch.tensor(it["q_ids"][:qn], dtype=torch.long)
        q_mask[i, :qn] = 1.0

        if is_mc:
            oids, omsk = _pad_2d(it["opt_ids"], num_opts, max_lo, pad_id)
            opt_ids[i] = oids
            opt_mask[i] = omsk
        answer_target[i] = int(it["answer_target"])

    return EpisodeBatch(
        ctx_ids=ctx_ids, ctx_mask=ctx_mask, step_mask=step_mask,
        q_ids=q_ids, q_mask=q_mask, opt_ids=opt_ids, opt_mask=opt_mask,
        answer_target=answer_target,
    )


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------
class EpisodeDataset(Dataset):
    """Encodes episodes lazily via the input encoder."""

    def __init__(
        self,
        episodes: List[Episode],
        encoder: AbstractInputEncoder,
        tokenizer: SimpleTokenizer,
        answer_vocab: Dict[str, int],
        cfg: Config,
    ) -> None:
        self.episodes = episodes
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.answer_vocab = answer_vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return encode_episode(
            self.episodes[idx], self.encoder, self.tokenizer, self.answer_vocab, self.cfg
        )


def make_dataloader(
    dataset: EpisodeDataset, pad_id: int, batch_size: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    """DataLoader whose ``collate_fn`` produces an :class:`EpisodeBatch`."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda items: collate(items, pad_id=pad_id),
    )
