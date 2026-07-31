"""Word Sense Disambiguation, grounded in NSM primes, conditioned on memory.

WSD here is a **constrained instance of the semantic-mapping problem**: rather
than composing meaning from scratch, we pick among a *known, finite* set of
candidate senses for a word, using the model's current consciousness state +
working memory as the disambiguating context. Each sense is represented by an
**NSM-prime signature** (a mini-explication), which ties WSD into the same prime
vocabulary as the rest of the system.

STATUS — scaffold (honest about what is real):

* **Real / trainable:** :class:`WSDModule` (the scorer) and :class:`SenseResolver`
  (inventory + scorer glue). The disambiguation signal genuinely comes from a
  context vector you build from ``(state, memory_read)``.
* **Mocked / illustrative:** :class:`MockSenseInventory` and its hand-authored
  sense -> prime signatures. Authoring those signatures *is* the semantic-mapping
  problem; the entries here are loose approximations, flagged as such.
* **Hook:** :class:`WordNetSenseInventory` is a stub — WordNet is the natural real
  inventory (NAOMI already uses it for sense discovery / node splitting).

How it plugs into the loop (not wired yet, by design): a WSD step would, per
content word, build a context from the current state + memory read, choose a
sense, and write the *sense-resolved* representation (not the ambiguous surface
token) into :class:`~nsm_ct.memory.WorkingMemory`. See RESEARCH_NOTES.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data_structures import ParseTree
from .nsm_primes import NUM_PRIMES, prime_index


@dataclass
class Sense:
    """One candidate sense of a word, as an NSM-prime signature.

    Attributes:
        word: The surface word.
        sense_id: Stable identifier, e.g. ``"bank.river"``.
        gloss: Human-readable gloss.
        primes: Map from NSM prime name -> weight (the mini-explication).
    """

    word: str
    sense_id: str
    gloss: str
    primes: Dict[str, float] = field(default_factory=dict)
    meaning: Optional[ParseTree] = field(default=None, compare=False)

    def prime_vector(self) -> np.ndarray:
        """Dense weight vector over the canonical prime inventory."""
        vec = np.zeros(NUM_PRIMES, dtype=np.float32)
        for name, w in self.primes.items():
            vec[prime_index(name)] = w
        return vec


class SenseInventory(abc.ABC):
    """Interface: map a word to its candidate :class:`Sense` objects."""

    @abc.abstractmethod
    def senses(self, word: str) -> List[Sense]:
        raise NotImplementedError

    def is_ambiguous(self, word: str) -> bool:
        return len(self.senses(word)) > 1


# ---------------------------------------------------------------------------
# Mock inventory (illustrative; sense->prime signatures are hand-authored)
# ---------------------------------------------------------------------------
# TODO(wsd-inventory): these prime signatures are loose approximations authored
# by hand to exercise the machinery. Real signatures = the semantic-mapping
# problem; the real inventory should come from WordNet (see WordNetSenseInventory).
_MOCK_SENSES: Dict[str, List[Sense]] = {
    "bank": [
        Sense("bank", "bank.river", "sloping land beside water",
              {"WHERE": 1.0, "NEAR": 0.8, "BIG": 0.4}),
        Sense("bank", "bank.money", "institution that holds money",
              {"SOMETHING": 1.0, "PEOPLE": 0.6, "MINE": 0.7}),
    ],
    "bat": [
        Sense("bat", "bat.animal", "small flying creature",
              {"LIVE": 1.0, "SMALL": 0.7, "MOVE": 0.6, "BODY": 0.5}),
        Sense("bat", "bat.club", "club used to hit a ball",
              {"SOMETHING": 1.0, "DO": 0.7, "TOUCH": 0.6}),
    ],
    "spring": [
        Sense("spring", "spring.season", "the season after winter",
              {"WHEN": 1.0, "LIVE": 0.5}),
        Sense("spring", "spring.coil", "a coiled elastic object",
              {"SOMETHING": 1.0, "MOVE": 0.7, "TOUCH": 0.5}),
    ],
    "light": [
        Sense("light", "light.bright", "brightness one can see by",
              {"SEE": 1.0, "GOOD": 0.3}),
        Sense("light", "light.weight", "not heavy",
              {"SMALL": 0.8, "NOT": 0.6, "BIG": -0.7}),
    ],
}


class MockSenseInventory(SenseInventory):
    """A tiny, hand-authored inventory for a few classic ambiguous words.

    Words not in the table get a single generic sense, so the pipeline always
    returns at least one candidate.
    """

    def senses(self, word: str) -> List[Sense]:
        word = word.lower()
        if word in _MOCK_SENSES:
            return list(_MOCK_SENSES[word])
        # Unknown word: one generic, near-empty sense.
        return [Sense(word, f"{word}.0", "generic sense", {})]


class WordNetSenseInventory(SenseInventory):
    """Real sense inventory backed by WordNet synsets.

    Each synset becomes a :class:`Sense` with:
    - ``sense_id``: the canonical synset name (e.g. ``"bank.n.01"``)
    - ``gloss``: the synset definition string
    - ``primes``: empty dict — grounding into NSM primes is a later stage
    - ``meaning``: None — the explication tree is a later stage

    If WordNet is unavailable (corpus not installed / nltk missing) the
    inventory falls back to a single generic sense so nothing crashes.
    """

    def senses(self, word: str) -> List[Sense]:
        from . import wordnet as _wordnet_mod

        raw = _wordnet_mod.senses(word.lower())
        if raw:
            return [
                Sense(
                    word=word.lower(),
                    sense_id=entry["sense_id"],
                    gloss=entry["gloss"],
                    primes={},
                    meaning=None,
                )
                for entry in raw
            ]
        # Graceful fallback: single generic sense (mirrors MockSenseInventory).
        return [Sense(word.lower(), f"{word.lower()}.0", "generic sense", {})]


class GroundedWordNetSenseInventory(WordNetSenseInventory):
    """WordNet inventory whose ``Sense.primes`` are filled by GROUNDING each synset's
    OWN gloss into NSM primes (M26 — roadmap step A).

    This closes the M22→WSD bridge: `WordNetSenseInventory` returned real senses with
    empty prime signatures; here each sense's signature is the grounded prime activation
    of its gloss (`ground.sense_graph.gloss_prime_weights`), so `Sense.prime_vector()`
    is real and *sense-distinct* (a synset's river gloss and finance gloss land on
    different primes). Same enumeration/fallback as the base class; only `primes` change.
    """

    def __init__(self, *, depth: int = 2, max_children: int = 8):
        self._depth = depth
        self._max_children = max_children

    def senses(self, word: str) -> List[Sense]:
        from .ground.sense_graph import gloss_prime_weights

        out = super().senses(word)
        for s in out:
            if s.gloss and s.gloss != "generic sense":
                s.primes = gloss_prime_weights(s.gloss, depth=self._depth,
                                               max_children=self._max_children)
        return out


# ---------------------------------------------------------------------------
# The (real, trainable) scorer
# ---------------------------------------------------------------------------
class WSDModule(nn.Module):
    """Scores candidate senses against a memory/state-derived context.

    Args:
        context_dim: Width of the context vector (typically
            ``consciousness_dim + memory_dim``).
        hidden: Shared projection width for context and sense signatures.
    """

    def __init__(self, context_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.sense_proj = nn.Linear(NUM_PRIMES, hidden)
        self.context_proj = nn.Linear(context_dim, hidden)

    def forward(
        self,
        context: torch.Tensor,      # [B, context_dim]
        sense_vecs: torch.Tensor,   # [B, S, NUM_PRIMES]
        sense_mask: torch.Tensor,   # [B, S] (1 = real candidate)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(sense_logits [B, S], sense_embedding [B, hidden])``.

        ``sense_embedding`` is the attention-weighted projected signature — the
        sense-resolved vector you would feed into memory.
        """
        q = self.context_proj(context)                 # [B, h]
        k = self.sense_proj(sense_vecs)                 # [B, S, h]
        logits = (k * q.unsqueeze(1)).sum(-1)          # [B, S]
        logits = logits.masked_fill(sense_mask == 0, float("-inf"))
        attn = torch.softmax(logits, dim=-1) * sense_mask  # [B, S]
        sense_emb = (attn.unsqueeze(-1) * k).sum(dim=1)    # [B, h]
        return logits, sense_emb


# ---------------------------------------------------------------------------
# Glue: inventory + scorer
# ---------------------------------------------------------------------------
def candidates_to_tensor(
    sense_lists: List[List[Sense]], device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of per-word candidate lists into ``(sense_vecs, mask)``.

    Args:
        sense_lists: For each example, its list of candidate senses.

    Returns:
        ``sense_vecs`` ``[B, S, NUM_PRIMES]`` and ``mask`` ``[B, S]``.
    """
    b = len(sense_lists)
    s = max(1, max(len(lst) for lst in sense_lists))
    vecs = torch.zeros(b, s, NUM_PRIMES, dtype=torch.float32, device=device)
    mask = torch.zeros(b, s, dtype=torch.float32, device=device)
    for i, lst in enumerate(sense_lists):
        for j, sense in enumerate(lst):
            vecs[i, j] = torch.from_numpy(sense.prime_vector())
            mask[i, j] = 1.0
    return vecs, mask


class SenseResolver:
    """Ties an inventory + a :class:`WSDModule` into a usable disambiguator.

    A future WSD step in :class:`~nsm_ct.agent.Mind` would call :meth:`resolve`
    with a context built from ``(state, memory_read)`` and write the returned
    sense embedding into memory instead of the raw token.
    """

    def __init__(self, inventory: SenseInventory, module: WSDModule) -> None:
        self.inventory = inventory
        self.module = module

    def resolve(self, words: List[str], context: torch.Tensor):
        """Disambiguate a batch of words given a context (single pass).

        Args:
            words: One word per batch element.
            context: ``[B, context_dim]`` memory/state-derived context.

        Returns:
            dict with ``logits`` ``[B, S]``, ``sense_emb`` ``[B, hidden]``,
            ``chosen`` (list of chosen :class:`Sense`), ``candidates``.
        """
        sense_lists = [self.inventory.senses(w) for w in words]
        sense_vecs, mask = candidates_to_tensor(sense_lists, device=context.device)
        logits, sense_emb = self.module(context, sense_vecs, mask)
        chosen = _chosen_senses(sense_lists, logits)
        return {
            "logits": logits,
            "sense_emb": sense_emb,
            "chosen": chosen,
            "candidates": sense_lists,
        }


def _chosen_senses(sense_lists: List[List[Sense]], logits: torch.Tensor) -> List[Sense]:
    """Pick the argmax sense per example, clamped to its candidate list."""
    idx = logits.argmax(dim=-1).tolist()
    return [sense_lists[i][min(j, len(sense_lists[i]) - 1)] for i, j in enumerate(idx)]


class IterativeSenseResolver(nn.Module):
    """Coherence-driven, self-correcting WSD.

    Embodies the user's idea: interpret a word with a sense given the current
    state; ask a learned **coherence** head whether that interpretation "makes
    sense"; if not, **update the state** from the attempted interpretation and
    **re-evaluate all the senses**. The chosen sense can therefore change across
    hops as the state evolves. There are no gold sense labels — the coherence
    signal (not a label) drives the loop.

    This is standalone: it operates on a context vector (which a caller builds
    from ``(state, memory_read)``) plus the candidate senses from an inventory.
    It is NOT wired into :class:`~nsm_ct.agent.Mind` yet (see RESEARCH_NOTES).

    Args:
        inventory: Source of candidate senses.
        context_dim: Width of the incoming context/state vector.
        hidden: Shared projection width (also the sense-embedding width).
        coherence_threshold: Halt when all examples reach this coherence.
    """

    def __init__(
        self,
        inventory: SenseInventory,
        context_dim: int,
        hidden: int = 64,
        coherence_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.inventory = inventory
        self.context_dim = context_dim
        self.threshold = coherence_threshold
        self.module = WSDModule(context_dim, hidden)
        # Is the (state, chosen-sense) interpretation coherent?
        self.coherence_head = nn.Sequential(
            nn.Linear(context_dim + hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        # Re-evaluation: fold the attempted interpretation back into the state.
        self.update = nn.GRUCell(hidden, context_dim)

    def coherence(self, state: torch.Tensor, sense_emb: torch.Tensor) -> torch.Tensor:
        """Coherence score in [0, 1] for ``(state, sense_emb)`` (``[B]``)."""
        return torch.sigmoid(self.coherence_head(torch.cat([state, sense_emb], dim=-1))).squeeze(-1)

    def resolve_iterative(
        self, words: List[str], context: torch.Tensor, max_hops: int = 3
    ) -> dict:
        """Iteratively disambiguate, re-evaluating senses until coherent.

        Args:
            words: One word per batch element.
            context: ``[B, context_dim]`` initial state/memory context.
            max_hops: Maximum re-evaluation passes.

        Returns:
            dict with final ``logits`` ``[B, S]``, ``sense_emb`` ``[B, hidden]``,
            ``coherence`` ``[B]``, ``hops`` (int used), ``history`` (per-hop
            dicts), ``chosen`` (final senses), and ``candidates``.
        """
        sense_lists = [self.inventory.senses(w) for w in words]
        sense_vecs, mask = candidates_to_tensor(sense_lists, device=context.device)

        state = context
        history: List[dict] = []
        logits = sense_emb = coherence = None
        for _hop in range(max_hops):
            logits, sense_emb = self.module(state, sense_vecs, mask)
            coherence = self.coherence(state, sense_emb)
            history.append({
                "logits": logits,
                "coherence": coherence,
                "chosen_idx": logits.argmax(dim=-1),
            })
            if bool((coherence >= self.threshold).all()):
                break
            # Incoherent somewhere: fold the interpretation in and re-evaluate.
            state = self.update(sense_emb, state)

        return {
            "logits": logits,
            "sense_emb": sense_emb,
            "coherence": coherence,
            "hops": len(history),
            "history": history,
            "chosen": _chosen_senses(sense_lists, logits),
            "candidates": sense_lists,
        }
