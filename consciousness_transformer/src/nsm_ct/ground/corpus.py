"""The gloss-vocabulary corpus (M18.0).

The principled population for basis discovery is the **defining vocabulary** —
the words WordNet actually uses to define other words. We count content-word
frequencies across every WordNet synset gloss, rank them, and take the top-n.
This is offline (WordNet is the only offline corpus), deterministic, and spans
both DeepNSM-covered and uncovered words (so the held-out derivation eval AND the
external DeepNSM check both have data).

The full ranking is persisted to ``data/gloss_vocab.json`` so different ``n`` are
cheap; building it from scratch is ~10s over 117k synsets.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List

from ..meaning import _STOPWORDS
from ..tokenizer import basic_tokenize
from ..wordnet import wordnet_available

# corpus.py -> ground -> nsm_ct -> src -> consciousness_transformer/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CACHE = _REPO_ROOT / "data" / "gloss_vocab.json"
_PERSIST_TOP = 50_000  # how many ranked words to persist (covers any reasonable n)


def _build_gloss_counts() -> Counter:
    """Content-word frequencies across all WordNet synset glosses."""
    from nltk.corpus import wordnet as wn  # local import — graceful

    counts: Counter = Counter()
    for synset in wn.all_synsets():
        for tok in basic_tokenize(synset.definition()):
            if tok in _STOPWORDS or len(tok) <= 1 or not tok.isalpha():
                continue
            counts[tok] += 1
    return counts


def _ranked_vocabulary() -> List[str]:
    """The full gloss vocabulary, frequency-desc then alphabetical (deterministic)."""
    counts = _build_gloss_counts()
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def gloss_vocabulary(n: int = 10_000, *, use_cache: bool = True, refresh: bool = False) -> List[str]:
    """Return the top-*n* gloss-vocabulary words (frequency-ranked, deterministic).

    Uses ``data/gloss_vocab.json`` when present unless ``refresh=True``; rebuilds
    and persists the top 50k otherwise. Returns ``[]`` if WordNet is unavailable.
    """
    if use_cache and not refresh and _CACHE.exists():
        try:
            ranked = json.loads(_CACHE.read_text(encoding="utf-8"))
            return ranked[:n]
        except Exception:  # pragma: no cover - corrupt cache -> rebuild
            pass

    if not wordnet_available():
        return []

    ranked = _ranked_vocabulary()
    if use_cache:
        try:
            _CACHE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE.write_text(json.dumps(ranked[:_PERSIST_TOP]), encoding="utf-8")
        except Exception:  # pragma: no cover
            pass
    return ranked[:n]
