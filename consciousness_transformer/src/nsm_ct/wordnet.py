"""Thin, dependency-graceful WordNet loader for NSM-CT.

Functions here wrap nltk.corpus.wordnet so that:
- If WordNet is unavailable (corpus not downloaded, nltk not installed), all
  functions return empty / None rather than raising.
- The ``wordnet_available()`` probe is cached after the first call.

This module is the relation-graph substrate consumed by
:class:`~nsm_ct.wsd.WordNetSenseInventory` and later pipeline stages.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Availability probe (cached)
# ---------------------------------------------------------------------------

_available: Optional[bool] = None


def wordnet_available() -> bool:
    """Return True if nltk and the WordNet corpus are usable.

    The result is cached after the first call so subsequent calls are free.
    """
    global _available
    if _available is not None:
        return _available
    try:
        from nltk.corpus import wordnet as _wn  # noqa: F401

        # Probe: a known word with multiple senses; must return at least one.
        synsets = _wn.synsets("bank")
        _available = len(synsets) > 0
    except Exception:  # pragma: no cover
        _available = False
    return _available


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _wn():
    """Return the wordnet corpus module (caller must check availability first)."""
    from nltk.corpus import wordnet as _wordnet

    return _wordnet


def senses(word: str) -> List[Dict]:
    """Return a list of sense dicts for *word*, one per WordNet synset.

    Each dict has:
        sense_id  : synset name, e.g. ``"bank.n.01"``
        gloss     : synset definition string
        pos       : part-of-speech tag (``'n'``, ``'v'``, ``'a'``, ``'r'``, ``'s'``)
        lemmas    : list of lemma name strings
        hypernyms : list of hypernym synset names
        hyponyms  : list of hyponym synset names

    Returns an empty list if WordNet is unavailable or the word has no synsets.
    """
    if not wordnet_available():
        return []
    try:
        wn = _wn()
        result = []
        for synset in wn.synsets(word):
            result.append(
                {
                    "sense_id": synset.name(),
                    "gloss": synset.definition(),
                    "pos": synset.pos(),
                    "lemmas": [lemma.name() for lemma in synset.lemmas()],
                    "hypernyms": [h.name() for h in synset.hypernyms()],
                    "hyponyms": [h.name() for h in synset.hyponyms()],
                }
            )
        return result
    except Exception:  # pragma: no cover
        return []


def antonyms(word: str) -> List[str]:
    """Return lemma names that are WordNet antonyms of *word* (deduped, sorted).

    Antonym edges live on lemmas, not synsets, so we walk every synset's lemmas.
    Empty list when WordNet is unavailable or no antonym edge exists. Used by the
    basis search (M17.2): a good basis makes an antonym pair differ on *minimal*
    axes.
    """
    if not wordnet_available():
        return []
    try:
        wn = _wn()
        out = set()
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                for ant in lemma.antonyms():
                    out.add(ant.name())
        return sorted(out)
    except Exception:  # pragma: no cover
        return []


def synonyms(word: str) -> List[str]:
    """Return lemma names sharing a synset with *word* (excluding *word* itself)."""
    if not wordnet_available():
        return []
    try:
        wn = _wn()
        out = set()
        wl = word.lower()
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                name = lemma.name()
                if name.lower() != wl:
                    out.add(name)
        return sorted(out)
    except Exception:  # pragma: no cover
        return []


def hypernyms(word: str) -> List[str]:
    """Return hypernym lemma names for *word* (one hop up the is-a hierarchy)."""
    if not wordnet_available():
        return []
    try:
        wn = _wn()
        out = set()
        for synset in wn.synsets(word):
            for h in synset.hypernyms():
                out.update(lemma.name() for lemma in h.lemmas())
        return sorted(out)
    except Exception:  # pragma: no cover
        return []


def relations(sense_id: str) -> Optional[Dict]:
    """Return a relation dict for the synset identified by *sense_id*.

    Args:
        sense_id: A WordNet synset name such as ``"bank.n.01"``.

    Returns:
        A dict with keys ``hypernyms`` and ``hyponyms`` (each a list of synset
        name strings), or ``None`` if the synset is not found or WordNet is
        unavailable.

    Example::

        >>> relations("bank.n.01")
        {'hypernyms': ['slope.n.01'], 'hyponyms': [...]}
    """
    if not wordnet_available():
        return None
    try:
        wn = _wn()
        synset = wn.synset(sense_id)
        return {
            "hypernyms": [h.name() for h in synset.hypernyms()],
            "hyponyms": [h.name() for h in synset.hyponyms()],
        }
    except Exception:
        return None
