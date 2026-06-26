"""Polarity-aware coordinates — the honest syn>ant lever (M18.1).

M17 showed syn>ant discrimination stays *below chance*: antonyms share gloss
structure, so an unsigned coordinate rates them as similar. The fix is to make
the coordinate **signed** so antonyms land on opposite poles — using two
non-circular sources (never the antonym edges themselves):

1. **Signed NSM pole pairs.** The NSM inventory already contains antonymous prime
   *pairs*. Collapse each into one *signed* axis: GOOD/BAD → ±EVAL, BIG/SMALL →
   ±SIZE, MUCH_MANY/LITTLE_FEW → ±QTY. So good/bad and big/small sit at opposite
   signs on one axis instead of in separate dimensions.
2. **Morphological negation.** `un-/in-/im-/il-/ir-/dis-/non-` prefixes and the
   `-less` suffix, when the stripped base is a real word, mean `NOT(base)` — so
   the value is the base's value with its polarity axes flipped (`unhappy` =
   flip(`happy`)). Using the *base's* decomposition (not the negated form's own
   gloss) avoids double-counting.

Residual lexical antonyms with neither an NSM pole nor morphology (hot/cold,
fast/slow) are *not* separated here — reported honestly as the remaining gap.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Sequence

import numpy as np

from ..data_structures import ParseTree
from ..nsm_primes import PRIME_NAMES
from .canonicalization import canon_label, normalize
from .definition_graph import DEFAULT_MAX_DEPTH, naive_decompose

# (signed axis name, +pole prime, -pole prime)
POLARITY_PAIRS: List[tuple] = [
    ("EVAL", "GOOD", "BAD"),
    ("SIZE", "BIG", "SMALL"),
    ("QTY", "MUCH_MANY", "LITTLE_FEW"),
]
_PAIRED = {p for _, pos, neg in POLARITY_PAIRS for p in (pos, neg)}
# +1 signed pole axes plus the gloss-cue MAGNITUDE axis.
_N_POLES = len(POLARITY_PAIRS) + 1

# Gloss polarity cues — a fixed, interpretable lexicon (NOT the antonym edges).
# Most adjective/verb antonyms are *defined* via a magnitude or negation contrast
# (hot="high temperature" vs cold="low temperature"), and decomposition drops the
# negation words as stopwords — so we read them straight off the gloss here.
_POS_CUES = frozenset({
    "high", "higher", "more", "most", "large", "larger", "great", "greater",
    "greatest", "big", "bigger", "strong", "stronger", "increase", "increased",
    "increasing", "above", "major", "full", "much", "many", "abundant", "excess",
    "maximum", "upper", "fast", "faster", "rapid", "quick", "long", "longer",
    "hot", "heavy", "deep", "wide", "rich", "presence", "present",
})
_NEG_CUES = frozenset({
    "not", "no", "without", "lacking", "lack", "absence", "absent", "fails",
    "fail", "cannot", "opposite", "contrary", "devoid", "less", "lesser", "least",
    "low", "lower", "lowest", "small", "smaller", "weak", "weaker", "decrease",
    "decreased", "decreasing", "below", "minor", "deficient", "missing", "loss",
    "minimum", "lower", "slow", "slower", "short", "shorter", "cold", "light",
    "shallow", "narrow", "poor", "empty",
})

# Conservative negation morphology. "a-" is intentionally excluded (apart, around,
# … — far too noisy). Validation requires the stripped base to be a real word.
_NEG_PREFIXES = ("un", "non", "dis", "im", "in", "il", "ir")
_NEG_SUFFIX = "less"
_MIN_BASE_LEN = 3


def signed_axes(axes: Sequence[str] = PRIME_NAMES) -> List[str]:
    """The signed-coordinate layout: non-paired axes, the signed pole axes, MAGNITUDE."""
    return ([a for a in axes if a not in _PAIRED]
            + [name for name, _, _ in POLARITY_PAIRS] + ["MAGNITUDE"])


def gloss_polarity(word: str, *, weight: float = 2.0) -> float:
    """Net magnitude/negation polarity read off *word*'s first WordNet gloss.

    Sums +1 per positive cue and -1 per negative cue, sign-compressed and scaled by
    *weight* so a single decisive cue (hot=high vs cold=low) separates an otherwise
    identical antonym pair. Non-circular: uses gloss text + a fixed cue lexicon,
    never the antonym edges.
    """
    from ..wordnet import senses, wordnet_available
    from ..tokenizer import basic_tokenize
    if not wordnet_available():
        return 0.0
    ss = senses(word)
    if not ss:
        return 0.0
    net = 0
    for tok in basic_tokenize(ss[0]["gloss"]):
        if tok in _POS_CUES:
            net += 1
        elif tok in _NEG_CUES:
            net -= 1
    if net == 0:
        return 0.0
    return weight * (1.0 if net > 0 else -1.0)


def _is_word(word: str) -> bool:
    from ..wordnet import senses, wordnet_available
    if not wordnet_available():
        return False
    return len(word) >= _MIN_BASE_LEN and bool(senses(word))


def negation_base(word: str) -> Optional[str]:
    """If *word* is a morphological negation of a real word, return that base.

    Tries each negating prefix and the ``-less`` suffix; the stripped base must be
    a real WordNet word (plus an ``-e`` restore for ``-less`` cases like
    ``careless`` → ``care``). Returns ``None`` otherwise.
    """
    w = word.lower().strip()
    for p in _NEG_PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= _MIN_BASE_LEN:
            base = w[len(p):]
            if _is_word(base):
                return base
    if w.endswith(_NEG_SUFFIX) and len(w) - len(_NEG_SUFFIX) >= _MIN_BASE_LEN:
        stem = w[: -len(_NEG_SUFFIX)]
        for cand in (stem, stem + "e"):
            if _is_word(cand):
                return cand
    return None


def _counts_vector(tree: ParseTree, axes: Sequence[str], magnitude: float) -> np.ndarray:
    """Signed coordinate: non-paired counts, signed NSM poles, gloss MAGNITUDE."""
    counts = Counter(canon_label(n.label) for n in normalize(tree).iter_preorder())
    base = [float(counts.get(a, 0)) for a in axes if a not in _PAIRED]
    poles = [float(counts.get(pos, 0) - counts.get(neg, 0)) for _, pos, neg in POLARITY_PAIRS]
    return np.array(base + poles + [magnitude], dtype=np.float32)


def _flip_poles(vec: np.ndarray) -> np.ndarray:
    out = vec.copy()
    out[-_N_POLES:] *= -1.0
    return out


def polarity_vector(
    word: str,
    *,
    axes: Sequence[str] = PRIME_NAMES,
    depth: int = DEFAULT_MAX_DEPTH,
    decompose=None,
    _guard: int = 0,
) -> np.ndarray:
    """The signed, polarity-aware coordinate for *word*.

    If *word* is a morphological negation, returns the base word's polarity vector
    with its pole axes flipped; otherwise builds the signed vector from *word*'s
    own decomposition. ``decompose(word)`` overrides the default bounded
    decomposition (pass a cache-backed one for scale).
    """
    if _guard < 2:  # avoid pathological negation chains (e.g. "indisputable")
        base = negation_base(word)
        if base is not None and base != word:
            return _flip_poles(
                polarity_vector(base, axes=axes, depth=depth, decompose=decompose, _guard=_guard + 1)
            )

    tree = decompose(word) if decompose is not None else naive_decompose(word, max_depth=depth)
    return _counts_vector(tree, axes, gloss_polarity(word))
