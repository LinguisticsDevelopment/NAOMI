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
        wl = word.lower().replace(" ", "_")
        for synset in wn.synsets(word):
            # per-sense antonyms + frequency come from THIS word's lemma in THIS synset
            ants: List[str] = []
            freq = 0
            for lemma in synset.lemmas():
                if lemma.name().lower() == wl:
                    ants += [a.name() for a in lemma.antonyms()]
                    freq += lemma.count()
            result.append(
                {
                    "sense_id": synset.name(),
                    "gloss": synset.definition(),
                    "pos": synset.pos(),
                    "lemmas": [lemma.name() for lemma in synset.lemmas()],
                    "hypernyms": [h.name() for h in synset.hypernyms()],
                    "hyponyms": [h.name() for h in synset.hyponyms()],
                    "antonyms": sorted(set(ants)),   # per-sense (M22)
                    "frequency": freq,               # lemma.count() for MFS ranking (M22)
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


# ---------------------------------------------------------------------------
# Wider relational sources (M19) — every one named/interpretable, never opaque.
# ---------------------------------------------------------------------------


def _lemmas(synsets) -> List[str]:
    return sorted({l.name() for s in synsets for l in s.lemmas()})


def lexname(word: str) -> Optional[str]:
    """The first sense's lexicographer-file category (e.g. ``noun.animal``,
    ``verb.motion``, ``adj.all``) — a universal coarse semantic sort (44 of them,
    100% synset coverage). Returns ``None`` if WordNet is unavailable / no senses."""
    if not wordnet_available():
        return None
    try:
        ss = _wn().synsets(word)
        return ss[0].lexname() if ss else None
    except Exception:  # pragma: no cover
        return None


def attributes(word: str) -> List[str]:
    """The noun dimension(s) *word* is a value of (adjective -> attribute), e.g.
    ``hot``/``cold`` -> ``temperature``. The shared axis antonyms differ on."""
    if not wordnet_available():
        return []
    try:
        wn = _wn()
        return _lemmas([a for s in wn.synsets(word, wn.ADJ) for a in s.attributes()])
    except Exception:  # pragma: no cover
        return []


def attributes_of(noun: str) -> List[str]:
    """The adjectives that take *noun* as their attribute dimension (reverse of
    :func:`attributes`), e.g. ``temperature`` -> ``[cold, cool, hot, warm]``."""
    if not wordnet_available():
        return []
    try:
        wn = _wn()
        return _lemmas([a for s in wn.synsets(noun, wn.NOUN) for a in s.attributes()])
    except Exception:  # pragma: no cover
        return []


def similar_tos(word: str) -> List[str]:
    """Near-synonym adjective cluster lemmas (``synset.similar_tos``). Antonyms
    have disjoint clusters, so this both pulls synonyms and separates antonyms."""
    if not wordnet_available():
        return []
    try:
        wn = _wn()
        return _lemmas([t for s in wn.synsets(word, wn.ADJ) for t in s.similar_tos()])
    except Exception:  # pragma: no cover
        return []


def derivationally_related(word: str) -> List[str]:
    """Cross-POS derivational family (happy <-> happiness <-> happily)."""
    if not wordnet_available():
        return []
    try:
        wn = _wn()
        out = set()
        wl = word.lower()
        for s in wn.synsets(word):
            for lemma in s.lemmas():
                for d in lemma.derivationally_related_forms():
                    if d.name().lower() != wl:
                        out.add(d.name())
        return sorted(out)
    except Exception:  # pragma: no cover
        return []


def meronyms(word: str) -> List[str]:
    """Part / member / substance meronym lemmas (hand is part of body)."""
    if not wordnet_available():
        return []
    try:
        wn = _wn()
        ms = []
        for s in wn.synsets(word, wn.NOUN):
            ms += s.part_meronyms() + s.member_meronyms() + s.substance_meronyms()
        return _lemmas(ms)
    except Exception:  # pragma: no cover
        return []


def verb_groups(word: str) -> List[str]:
    """Coordinated-verb group lemmas (have <-> own, possess)."""
    if not wordnet_available():
        return []
    try:
        wn = _wn()
        return _lemmas([g for s in wn.synsets(word, wn.VERB) for g in s.verb_groups()])
    except Exception:  # pragma: no cover
        return []
