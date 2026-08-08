"""M32.1 — the trained sense chooser: the missing piece the M32 probe measured
the *absence* of.

``scripts/probe_m32_ambiguity.py`` established the gap: MFS grounding (always
``wn.synsets(word)[0]``) scores 0.247 on the sense-flipped half of
``episode.generate_ambiguity_episodes``; the oracle (gold sense) scores 1.000.
Nothing in the pipeline closed that gap — WSD candidates existed
(:mod:`nsm_ct.wordnet`, :mod:`nsm_ct.wsd`) but nothing *chose* among them from
context. This module is that choice: a tiny (<50k param) scorer over
USVS-space sense vectors.

Design, deliberately minimal — a **policy, not a knowledge store**:

* Candidates for a homograph: ``wordnet.senses(word)`` (MFS-ordered synsets),
  each mapped to its USVS coordinate via
  ``usvs_bridge.usvs_sense_handle(sense_id, d)``. A sense WordNet knows but
  USVS doesn't ground yet gets a zero vector and is masked out (never chosen
  over a real candidate; see :func:`candidate_vectors`).
* Context: the mean USVS handle of the episode's other content words (the
  homograph and a crude stopword list are excluded; see :func:`context_vector`).
* Scorer: ``MLP([candidate; context; candidate*context]) -> scalar``, softmax
  over the (masked) candidates. No token embeddings, no memory of *which*
  word means what — only a match function between a context vector and a
  sense vector. That is what makes the leave-one-family-out test in
  ``scripts/train_sense_chooser.py`` a real generalization test: if the
  model held out "organ" during training and still resolves it, it learned
  *how* to match context to sense geometry, not *that* "organ" means body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .episode import Episode
from .usvs_bridge import usvs_handle, usvs_sense_handle
from .wordnet import senses

D_DEFAULT = 64

# Crude, hand-picked function-word stoplist — this is a policy over content
# words, not a real tokenizer/POS pipeline. Good enough to strip determiners,
# pronouns, and copulas out of the tiny templated sentences in
# ``episode._AMBIGUITY_FAMILIES``.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "was", "were", "be", "been", "being", "am", "are",
    "in", "on", "at", "to", "of", "for", "with", "and", "or", "but", "so",
    "it", "he", "she", "they", "i", "you", "we", "this", "that", "these", "those",
    "his", "her", "its", "their", "my", "your", "our", "him", "them",
    "what", "kind", "do", "does", "did", "not", "no",
})

_PUNCT_RE = re.compile(r"[^\w']+")


def _tokens(text: str) -> List[str]:
    """Lowercase whitespace/punctuation tokenization (no external tokenizer)."""
    return [t for t in _PUNCT_RE.sub(" ", text.lower()).split() if t]


# ---------------------------------------------------------------------------
# Context vector: mean USVS handle of the episode's other content words
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8192)
def _word_vec(word: str, d: int) -> Optional[np.ndarray]:
    return usvs_handle(word, d)


def context_words(ep: Episode, exclude: str) -> List[str]:
    """Content words drawn from ``ep.context`` (question is a fixed template,
    so it's not informative), minus the homograph itself and stopwords."""
    out: List[str] = []
    for sent in ep.context:
        for tok in _tokens(sent):
            if tok == exclude or tok in STOPWORDS:
                continue
            out.append(tok)
    return out


def context_vector(ep: Episode, d: int = D_DEFAULT) -> np.ndarray:
    """Mean unit-norm USVS handle of the episode's other content words.

    Returns a zero vector (never ``None``) if no context word is USVS-known —
    the chooser must always run, even on a context it can't ground.
    """
    words = context_words(ep, ep.meta["homograph"])
    vecs = [v for v in (_word_vec(w, d) for w in words) if v is not None]
    if not vecs:
        return np.zeros(d, dtype=np.float32)
    m = np.mean(np.stack(vecs), axis=0).astype(np.float32)
    n = np.linalg.norm(m)
    return (m / n).astype(np.float32) if n > 1e-9 else m


# ---------------------------------------------------------------------------
# Candidate senses: wordnet.senses(word), MFS-ordered, mapped into USVS space
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def candidate_sense_ids(word: str) -> Tuple[str, ...]:
    """MFS-ordered candidate sense ids for ``word`` (``senses()[0]`` == MFS)."""
    return tuple(s["sense_id"] for s in senses(word.lower()))


@lru_cache(maxsize=256)
def candidate_vectors(word: str, d: int = D_DEFAULT) -> Tuple[Tuple[str, ...], np.ndarray, np.ndarray]:
    """Return ``(sense_ids, vecs [K, d], mask [K])`` for ``word``'s candidates.

    A candidate WordNet lists but USVS can't ground gets a zero row and
    ``mask == 0`` — never scored above a real candidate (see
    :meth:`SenseChooser.forward`). If ``word`` has no WordNet senses at all,
    returns a single masked-out placeholder so downstream code never sees an
    empty ``K`` dimension.
    """
    ids = candidate_sense_ids(word)
    vecs: List[np.ndarray] = []
    mask: List[float] = []
    for sid in ids:
        v = usvs_sense_handle(sid, d)
        if v is None:
            vecs.append(np.zeros(d, dtype=np.float32))
            mask.append(0.0)
        else:
            vecs.append(v.astype(np.float32))
            mask.append(1.0)
    if not vecs:
        ids = ("__unknown__",)
        vecs = [np.zeros(d, dtype=np.float32)]
        mask = [0.0]
    return ids, np.stack(vecs).astype(np.float32), np.array(mask, dtype=np.float32)


# ---------------------------------------------------------------------------
# One decision = one Example
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """One sense-choice decision, ready to batch.

    ``gold_idx`` indexes into ``candidate_ids``; it is ``-100`` (the standard
    ``ignore_index`` for ``F.cross_entropy``) if the episode's gold sense
    isn't among the candidates USVS can ground (shouldn't happen for the M32
    families, but never crash on it).
    """

    episode: Episode
    candidate_ids: Tuple[str, ...]
    candidates: np.ndarray  # [K, d]
    mask: np.ndarray        # [K]
    context: np.ndarray     # [d]
    gold_idx: int


def build_example(ep: Episode, d: int = D_DEFAULT) -> Example:
    word = ep.meta["homograph"]
    ids, vecs, mask = candidate_vectors(word, d)
    ctx = context_vector(ep, d)
    gold = ep.meta.get("gold_sense")
    gold_idx = ids.index(gold) if gold in ids else -100
    return Example(ep, ids, vecs, mask, ctx, gold_idx)


def collate(examples: List[Example]) -> Dict[str, torch.Tensor]:
    """Pad a batch of :class:`Example` to the batch's max candidate count."""
    d = examples[0].context.shape[0]
    k = max(e.candidates.shape[0] for e in examples)
    b = len(examples)
    cand = np.zeros((b, k, d), dtype=np.float32)
    mask = np.zeros((b, k), dtype=np.float32)
    ctx = np.zeros((b, d), dtype=np.float32)
    gold = np.full((b,), -100, dtype=np.int64)
    for i, e in enumerate(examples):
        kk = e.candidates.shape[0]
        cand[i, :kk] = e.candidates
        mask[i, :kk] = e.mask
        ctx[i] = e.context
        gold[i] = e.gold_idx
    return {
        "candidates": torch.from_numpy(cand),
        "mask": torch.from_numpy(mask),
        "context": torch.from_numpy(ctx),
        "gold_idx": torch.from_numpy(gold),
    }


# ---------------------------------------------------------------------------
# The chooser itself — a policy, not a knowledge store
# ---------------------------------------------------------------------------


class SenseChooser(nn.Module):
    """Scores candidate sense vectors against a context vector.

    ``MLP([candidate; context; candidate*context]) -> scalar`` per candidate,
    softmax-able logits over ``K`` (masked). No embedding table keyed on word
    identity: everything the model knows is in the shared MLP weights, which
    is what makes it, in principle, transferable across homograph families
    (see the leave-one-family-out rotation in
    ``scripts/train_sense_chooser.py``).

    Args:
        d: USVS handle dimensionality (candidates and context share it).
        hidden: MLP hidden width. Default keeps the whole module well under
            50k parameters (``3*d -> hidden -> 1``).
    """

    def __init__(self, d: int = D_DEFAULT, hidden: int = 64) -> None:
        super().__init__()
        self.d = d
        self.score = nn.Sequential(
            nn.Linear(3 * d, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        candidates: torch.Tensor,  # [B, K, d]
        mask: torch.Tensor,        # [B, K]  (1 = real, USVS-grounded candidate)
        context: torch.Tensor,     # [B, d]
    ) -> torch.Tensor:
        """Return masked logits ``[B, K]`` (``-inf`` where ``mask == 0``).

        If an example has NO real candidate at all (every sense USVS-ungrounded
        — the unknown-sense fallback case), masking is skipped for that row so
        softmax/argmax stay well-defined; since candidates are MFS-ordered,
        argmax then deterministically falls back to index 0 (the MFS sense).
        """
        b, k, _d = candidates.shape
        ctx = context.unsqueeze(1).expand(-1, k, -1)
        feats = torch.cat([candidates, ctx, candidates * ctx], dim=-1)
        logits = self.score(feats).squeeze(-1)  # [B, K]
        has_real_candidate = (mask.sum(dim=-1, keepdim=True) > 0)
        safe_mask = torch.where(has_real_candidate, mask, torch.ones_like(mask))
        logits = logits.masked_fill(safe_mask < 0.5, float("-inf"))
        return logits


def predicted_sense_ids(examples: List[Example], logits: torch.Tensor) -> List[str]:
    """Argmax-decode a batch of logits back into sense-id strings."""
    idx = logits.argmax(dim=-1).tolist()
    out = []
    for e, j in zip(examples, idx):
        j = min(j, len(e.candidate_ids) - 1)
        out.append(e.candidate_ids[j])
    return out
