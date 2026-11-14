"""The honest understanding harness (M17.0).

The legacy system has **no measure of word-meaning understanding at all** — only
task-answer accuracy. This module supplies the missing baseline, computed without
the DeepNSM/gold lookup:

- ``convergence(word)`` — does decomposing one level deeper change the meaning
  coordinate? A word's meaning should stop moving once it bottoms out at the prime
  floor; if it keeps changing (because decomposition truncates at ``UNRESOLVED``),
  the word is not yet understood. This is the baseline form of clause==word
  self-consistency (a fixpoint check).
- ``prime_grounding(word)`` — what fraction of the decomposition's leaves actually
  reached a prime/molecule vs. stayed ``UNRESOLVED``. Quantifies the depth-2
  truncation problem directly.
- ``deepnsm_agreement(word)`` — Jaccard overlap of our generated prime signature
  with the DeepNSM explication's primes. An **independent external check**, never
  a training/runtime source.

``closure(word, value_fn, reduce_fn)`` is the generic hook the M17.1 reduction
operator plugs into (replacing the depth+1 proxy with a real clause reduction).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set

from ..nsm_molecules import MOLECULES_BY_NAME
from ..nsm_primes import PRIME_NAMES
from .canonicalization import canon_label, normalize
from .definition_graph import DEFAULT_MAX_DEPTH, definition_clause, naive_decompose
from .meaning_value import MeaningValue, cosine, jaccard

_PRIME_SET = frozenset(PRIME_NAMES)
_MOLECULE_SET = frozenset(MOLECULES_BY_NAME)

# A small, fixed, kind-spanning vocabulary so the baseline is deterministic and
# does not need a corpus download. Spans concrete nouns, verbs, adjectives,
# emotions, relations, and abstracts — to expose where grounding breaks by kind.
SAMPLE_VOCAB: List[str] = [
    # concrete nouns
    "dog", "kitchen", "water", "stone", "tree", "hand", "house",
    # verbs / events
    "run", "give", "break", "move", "kill", "see",
    # adjectives / qualities
    "big", "hot", "cold", "good", "heavy", "red",
    # emotions
    "sad", "happy", "afraid", "angry", "proud",
    # relations / spatial
    "above", "near", "inside",
    # abstracts
    "justice", "freedom", "number", "truth",
]


def value(word: str, *, depth: int = DEFAULT_MAX_DEPTH, codec=None) -> MeaningValue:
    """The word's generated meaning value via bounded decomposition (our system)."""
    return MeaningValue.from_tree(naive_decompose(word, max_depth=depth), codec=codec)


def prime_grounding(word: str, *, depth: int = DEFAULT_MAX_DEPTH) -> Optional[float]:
    """Fraction of decomposition leaves that reached a prime/molecule (vs UNRESOLVED).

    Returns ``None`` if the tree has no leaves (degenerate).
    """
    tree = normalize(naive_decompose(word, max_depth=depth))
    leaves = tree.leaves()
    if not leaves:
        return None
    grounded = sum(
        1 for n in leaves
        if canon_label(n.label) in _PRIME_SET or n.label in _MOLECULE_SET
    )
    return grounded / len(leaves)


def convergence(word: str, *, depth: int = DEFAULT_MAX_DEPTH) -> float:
    """Cosine stability of the prime coordinate between depth and depth+1.

    1.0 means deeper decomposition does not change the meaning (a stable prime
    fixpoint was reached); low means the word is still dissolving (truncated).
    """
    a = value(word, depth=depth).axis_vec
    b = value(word, depth=depth + 1).axis_vec
    return cosine(a, b)


def deepnsm_primes(word: str, store=None) -> Set[str]:
    """The prime labels in the DeepNSM/gold explication of *word* (external check)."""
    if store is None:
        store = _default_store()
    if store is None or store.is_empty():
        return set()
    entry = store.get(word)
    if entry is None:
        return set()
    tree = store.explication_to_tree(entry["explication"])
    return {canon_label(n.label) for n in tree.iter_preorder() if canon_label(n.label) in _PRIME_SET}


def deepnsm_agreement(word: str, *, depth: int = DEFAULT_MAX_DEPTH, store=None) -> Optional[float]:
    """Jaccard overlap of our prime signature with DeepNSM's (None if not covered)."""
    ext = deepnsm_primes(word, store=store)
    if not ext:
        return None
    ours = value(word, depth=depth).active_axes()
    return jaccard(ours, ext)


def closure(
    word: str,
    *,
    value_fn: Callable[[str], Optional[MeaningValue]],
    reduce_fn: Callable[..., Optional[MeaningValue]],
    sim: Callable = cosine,
) -> Optional[float]:
    """Generic clause==word closure: sim(value(word), reduce(definition(word))).

    M17.0 passes a depth-based proxy; M17.1 passes the real reduction operator
    (``definition_clause(word)`` -> reduced single word/value).
    """
    v = value_fn(word)
    defn = definition_clause(word)
    if v is None or defn is None:
        return None
    r = reduce_fn(defn)
    if r is None:
        return None
    return sim(v.axis_vec, r.axis_vec)


_STORE_CACHE = []


def _default_store():
    if _STORE_CACHE:
        return _STORE_CACHE[0]
    try:
        from ..explications import ExplicationStore
        store = ExplicationStore.load()
    except Exception:  # pragma: no cover
        store = None
    _STORE_CACHE.append(store)
    return store


def report(words: Optional[List[str]] = None, *, depth: int = DEFAULT_MAX_DEPTH) -> Dict:
    """Aggregate the three baseline metrics over *words* (defaults to SAMPLE_VOCAB)."""
    words = words or SAMPLE_VOCAB
    store = _default_store()

    per_word: Dict[str, Dict] = {}
    conv_vals: List[float] = []
    ground_vals: List[float] = []
    agree_vals: List[float] = []
    covered = 0

    for w in words:
        conv = convergence(w, depth=depth)
        ground = prime_grounding(w, depth=depth)
        agree = deepnsm_agreement(w, depth=depth, store=store)
        per_word[w] = {
            "convergence": conv,
            "prime_grounding": ground,
            "deepnsm_agreement": agree,
            "active_axes": sorted(value(w, depth=depth).active_axes()),
        }
        conv_vals.append(conv)
        if ground is not None:
            ground_vals.append(ground)
        if agree is not None:
            agree_vals.append(agree)
            covered += 1

    def _mean(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    return {
        "n_words": len(words),
        "depth": depth,
        "mean_convergence": _mean(conv_vals),
        "mean_prime_grounding": _mean(ground_vals),
        "mean_deepnsm_agreement": _mean(agree_vals),
        "deepnsm_covered": covered,
        "per_word": per_word,
    }
