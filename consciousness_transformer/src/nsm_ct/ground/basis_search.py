"""MDL-driven basis discovery (M17.2).

Finds an interpretable basis by **Minimum Description Length**: minimize the
number of axes *and* the size of the decompositions they produce, jointly. The
search is seeded with NSM-65 and grows by promoting the words the definition
graph keeps bottoming out on (``UNRESOLVED`` leaves — the definitional
cycle-breakers / feedback vertices), accepting a promotion only when it lowers
total description length.

MDL(words, extra) = AXIS_COST * |extra| + sum over words of
    (#nodes in its decomposition) + UNRESOLVED_PENALTY * (#unresolved leaves)

Greedy step: among the most-frequent un-grounded leaf words, promote the one
that most reduces MDL; stop when no promotion helps (or a budget is hit).

The relational signals (antonym minimality, synonym relatedness, hypernym
containment, grounding rate) are reported for seed vs. final so the basis is
*evaluated*, not asserted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from ..nsm_molecules import MOLECULES_BY_NAME
from ..nsm_primes import PRIME_NAMES
from .canonicalization import canon_label, normalize
from .definition_graph import DEFAULT_MAX_DEPTH, DefinitionGraph, naive_decompose
from .meaning_value import axis_vector, cosine
from .semantic_axes import AxisRegistry

UNRESOLVED_PENALTY = 3.0
AXIS_COST = 2.0

_PRIME_SET = frozenset(PRIME_NAMES)
_MOL_SET = frozenset(MOLECULES_BY_NAME)


def _decomp(word: str, extra: FrozenSet[str], depth: int):
    return naive_decompose(word, max_depth=depth, extra_axes=extra)


def _dl_word(word: str, extra: FrozenSet[str], depth: int) -> float:
    tree = _decomp(word, extra, depth)
    n_nodes = tree.num_nodes()
    n_unres = sum(1 for n in tree.leaves() if n.label == "UNRESOLVED")
    return n_nodes + UNRESOLVED_PENALTY * n_unres


def mdl(words, extra: FrozenSet[str], depth: int) -> float:
    """Total description length of *words* given the promoted axis set *extra*."""
    return AXIS_COST * len(extra) + sum(_dl_word(w, extra, depth) for w in words)


def _unresolved_counts(words, extra: FrozenSet[str], depth: int) -> Counter:
    c: Counter = Counter()
    for w in words:
        for leaf in _decomp(w, extra, depth).leaves():
            if leaf.label == "UNRESOLVED" and leaf.token:
                c[leaf.token] += 1
    return c


def _value_vec(word: str, reg: AxisRegistry, depth: int) -> np.ndarray:
    tree = normalize(_decomp(word, reg.extra_axes(), depth))
    return axis_vector(tree, reg.axes)


def _grounding_rate(words, reg: AxisRegistry, depth: int) -> float:
    extra = reg.extra_axes()
    grounded = total = 0
    for w in words:
        for leaf in _decomp(w, extra, depth).leaves():
            total += 1
            if canon_label(leaf.label) in _PRIME_SET or leaf.label in _MOL_SET or leaf.label in extra:
                grounded += 1
    return grounded / total if total else 0.0


def relational_metrics(words, reg: AxisRegistry, depth: int, graph: DefinitionGraph) -> Dict:
    """Evaluate a basis against the relational signals (lower antonym, higher
    synonym/hypernym is better; grounding strictly improves with promotion)."""
    coord = {w: _value_vec(w, reg, depth) for w in words}

    # Antonyms: should differ on *few* axes -> high cosine (similar coordinate,
    # opposite on a minimal subset). We report mean cosine over antonym pairs.
    ant_cos: List[float] = []
    for a, b in graph.antonym_pairs():
        if a in coord and b in coord:
            ant_cos.append(cosine(coord[a], coord[b]))

    # Synonyms: should be near-identical coordinates -> high cosine.
    syn_cos: List[float] = []
    for w in words:
        for s in graph.synonym.get(w, []):
            s = s.lower()
            if s in coord and s != w:
                syn_cos.append(cosine(coord[w], coord[s]))

    # Hypernym containment: the hypernym's axes should be (mostly) a subset of
    # the word's axes (a dog's meaning contains an animal's).
    contain: List[float] = []
    for w in words:
        wa = {a for a, x in zip(reg.axes, coord[w]) if x > 0}
        for h in graph.is_a.get(w, []):
            h = h.lower()
            if h in coord:
                ha = {a for a, x in zip(reg.axes, coord[h]) if x > 0}
                if ha:
                    contain.append(len(ha & wa) / len(ha))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "grounding_rate": _grounding_rate(words, reg, depth),
        "antonym_cos": _mean(ant_cos),
        "n_antonym_pairs": len(ant_cos),
        "synonym_cos": _mean(syn_cos),
        "n_synonym_pairs": len(syn_cos),
        "hypernym_containment": _mean(contain),
        "n_hypernym_pairs": len(contain),
    }


@dataclass
class BasisResult:
    registry: AxisRegistry
    mdl_curve: List[Tuple[int, float]]  # (num_extra_axes, mdl)
    added: List[Tuple[str, int, float]]  # (axis, leverage_freq, mdl_gain)
    seed_metrics: Dict = field(default_factory=dict)
    final_metrics: Dict = field(default_factory=dict)


def search(
    words,
    *,
    depth: int = DEFAULT_MAX_DEPTH,
    max_axes: int = 20,
    min_gain: float = 1e-9,
    graph: Optional[DefinitionGraph] = None,
) -> BasisResult:
    """Greedily grow the basis from NSM-65 by MDL over *words*."""
    words = [w.lower().strip() for w in words]
    if graph is None:
        graph = DefinitionGraph.build(words)

    reg = AxisRegistry.seed()
    extra: set = set()
    cur_mdl = mdl(words, frozenset(extra), depth)
    curve: List[Tuple[int, float]] = [(0, cur_mdl)]
    added: List[Tuple[str, int, float]] = []

    for _ in range(max_axes):
        counts = _unresolved_counts(words, frozenset(extra), depth)
        cand = None
        freq = 0
        for w, f in counts.most_common():
            if w in extra or canon_label(w) in _PRIME_SET or w in _MOL_SET:
                continue
            cand, freq = w, f
            break
        if cand is None:
            break
        new_extra = frozenset(extra | {cand})
        new_mdl = mdl(words, new_extra, depth)
        gain = cur_mdl - new_mdl
        if gain <= min_gain:
            break
        extra.add(cand)
        cur_mdl = new_mdl
        reg.add(cand, f"cycle-break:freq{freq}")
        added.append((cand, freq, gain))
        curve.append((len(extra), cur_mdl))

    seed_metrics = relational_metrics(words, AxisRegistry.seed(), depth, graph)
    final_metrics = relational_metrics(words, reg, depth, graph)
    return BasisResult(reg, curve, added, seed_metrics, final_metrics)
