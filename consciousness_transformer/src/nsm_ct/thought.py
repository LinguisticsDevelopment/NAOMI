"""Thought objects: the model's working unit (a tree of meaning).

A **thought object** is a syntactic parse tree (from the parser) whose word-leaves
each carry a **meaning tree** — the word's reductive explication as a *tree of NSM
primes* (an NSM "molecule"). Meaning here is recursive structure, not a flat
vector: a parse tree over words, and under each word, a tree of primes.

This module defines the thought object, the meaning-tree representation (reusing
:class:`~nsm_ct.data_structures.ParseTree`), and the **resolver** interface that
maps a word → its meaning tree. Stage A ships a deterministic *mock* resolver so
the unit is real before WordNet-grounded explications land (see RESEARCH_NOTES /
the plan: Stage B replaces the mock with WordNet senses → prime trees).

The meaning is fed to the model as a small **bag of prime ids per word** (additive,
zero-init — see ``model.step``); the *full* meaning tree lives on the thought
object and in the lossless serialization (``serialization.serialize_thought``).
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

from .data_structures import ParseNode, ParseTree
from .nsm_primes import NUM_PRIMES, PRIME_NAMES, prime_index

# Per-word meaning is fed to the model as up to MAX_MEANING_PRIMES prime ids.
# Prime ids are 1..NUM_PRIMES; id 0 is the pad / "no prime".
MAX_MEANING_PRIMES = 4
NUM_MEANING_IDS = NUM_PRIMES + 1


def meaning_prime_id(name: str) -> int:
    """Prime name -> meaning id (1-based; 0 reserved for pad / no-prime)."""
    return prime_index(name) + 1


def _stable_hash(text: str) -> int:
    """Deterministic non-negative hash (Python's ``hash`` is salted per run)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


# ---------------------------------------------------------------------------
# Meaning resolver: word -> tree of primes
# ---------------------------------------------------------------------------
class AbstractMeaningResolver(abc.ABC):
    """Maps a word (in context) to its meaning tree (a tree of NSM primes)."""

    @abc.abstractmethod
    def resolve(self, word: str, context: object = None) -> ParseTree:
        raise NotImplementedError


class MockMeaningResolver(AbstractMeaningResolver):
    """Deterministic stand-in: word -> a small tree of primes (mock molecule).

    Meaning-free but *structured* and stable across runs, so the thought-object
    pipeline is real before WordNet-grounded explications exist. The number and
    choice of primes are a hash of the word; one root prime with 0–2 children.
    TODO(meaning): replace with WordNet sense -> NSM explication (Stage B).
    """

    def resolve(self, word: str, context: object = None) -> ParseTree:
        h = _stable_hash(word.lower())
        k = 1 + (h % MAX_MEANING_PRIMES)                      # 1..MAX primes
        names = [PRIME_NAMES[(h // (i + 7)) % NUM_PRIMES] for i in range(k)]
        root = ParseNode(label=names[0])
        for n in names[1:]:
            root.children.append(ParseNode(label=n))
        return ParseTree(root=root)


# ---------------------------------------------------------------------------
# Thought object
# ---------------------------------------------------------------------------
@dataclass
class ThoughtObject:
    """A parse tree whose word-leaves carry meaning trees (the model's unit)."""

    tree: ParseTree
    text: str = ""


def build_thought(
    sentence: str, parse_tree: ParseTree, resolver: AbstractMeaningResolver
) -> ThoughtObject:
    """Attach a meaning tree to every word-leaf of ``parse_tree`` (in place)."""
    for node in parse_tree.iter_preorder():
        if node.token is not None:
            node.meaning = resolver.resolve(node.token)
    return ThoughtObject(tree=parse_tree, text=sentence)


def meaning_prime_ids(meaning: Optional[ParseTree]) -> List[int]:
    """The (capped) list of prime ids in a word's meaning tree (pre-order)."""
    if meaning is None:
        return []
    ids = [meaning_prime_id(n.label) for n in meaning.iter_preorder()]
    return ids[:MAX_MEANING_PRIMES]
