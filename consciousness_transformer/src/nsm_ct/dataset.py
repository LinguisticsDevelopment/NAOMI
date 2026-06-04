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
    """Encode one episode into a uniform item stream (context, question, distractors).

    The question is just another item; ``question_index`` is recorded for metrics
    only and is never fed to the model.
    """
    max_len = cfg.model.max_sentence_len
    item_texts = list(ep.context) + [ep.question] + list(ep.post_context)
    question_index = len(ep.context)
    item_ids = [_truncate(encoder.encode(s), max_len) or [tokenizer.pad_id] for s in item_texts]

    if cfg.data.answer_mode == "mc":
        if not ep.options:
            raise ValueError("answer_mode='mc' but episode has no options")
        opt_ids = [_truncate(tokenizer.encode(o), max_len) or [tokenizer.pad_id] for o in ep.options]
        answer_target = int(ep.answer_idx)
    else:
        opt_ids = []
        answer_target = answer_vocab[ep.answer_text]

    return {
        "item_ids": item_ids,
        "item_texts": item_texts,
        "question_index": question_index,
        "opt_ids": opt_ids,
        "answer_target": answer_target,
    }


@dataclass
class EpisodeBatch:
    """A padded, tensorized batch of episodes (uniform item stream)."""

    item_ids: torch.Tensor       # [B, T, L]  (context + question + distractors)
    item_mask: torch.Tensor      # [B, T, L]  (token mask per item)
    step_mask: torch.Tensor      # [B, T]  (1 = real item)
    question_index: torch.Tensor  # [B]  (metrics only; never fed to the model)
    opt_ids: torch.Tensor        # [B, K, Lo]
    opt_mask: torch.Tensor       # [B, K, Lo]
    answer_target: torch.Tensor  # [B]
    item_texts: object = None    # List[List[str]] provenance (not moved to device)

    def to(self, device: torch.device) -> "EpisodeBatch":
        return EpisodeBatch(
            item_ids=self.item_ids.to(device),
            item_mask=self.item_mask.to(device),
            step_mask=self.step_mask.to(device),
            question_index=self.question_index.to(device),
            opt_ids=self.opt_ids.to(device),
            opt_mask=self.opt_mask.to(device),
            answer_target=self.answer_target.to(device),
            item_texts=self.item_texts,
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
    max_t = max(1, max(len(it["item_ids"]) for it in items))
    max_l = max(1, max((len(s) for it in items for s in it["item_ids"]), default=1))
    is_mc = len(items[0]["opt_ids"]) > 0
    num_opts = len(items[0]["opt_ids"]) if is_mc else 1
    max_lo = max(1, max((len(o) for it in items for o in it["opt_ids"]), default=1))

    item_ids = torch.full((b, max_t, max_l), pad_id, dtype=torch.long)
    item_mask = torch.zeros((b, max_t, max_l), dtype=torch.float32)
    step_mask = torch.zeros((b, max_t), dtype=torch.float32)
    question_index = torch.zeros((b,), dtype=torch.long)
    opt_ids = torch.full((b, num_opts, max_lo), pad_id, dtype=torch.long)
    opt_mask = torch.zeros((b, num_opts, max_lo), dtype=torch.float32)
    answer_target = torch.zeros((b,), dtype=torch.long)
    item_texts: List[List[str]] = []

    for i, it in enumerate(items):
        stream = it["item_ids"]
        ids, msk = _pad_2d(stream, len(stream), max_l, pad_id)
        item_ids[i, : len(stream)] = ids
        item_mask[i, : len(stream)] = msk
        step_mask[i, : len(stream)] = 1.0
        question_index[i] = int(it["question_index"])
        item_texts.append(list(it["item_texts"]))

        if is_mc:
            oids, omsk = _pad_2d(it["opt_ids"], num_opts, max_lo, pad_id)
            opt_ids[i] = oids
            opt_mask[i] = omsk
        answer_target[i] = int(it["answer_target"])

    return EpisodeBatch(
        item_ids=item_ids, item_mask=item_mask, step_mask=step_mask,
        question_index=question_index, opt_ids=opt_ids, opt_mask=opt_mask,
        answer_target=answer_target, item_texts=item_texts,
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
