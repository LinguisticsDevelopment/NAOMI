"""Multi-signal basis selection (M18.2).

M17.2 grew the basis by MDL/unresolved-frequency alone, and the honest negative
was that this didn't improve relatedness (synonym agreement, hypernym
containment). M18.2 keeps the MDL benefit — candidates are still drawn from the
frequent un-grounded words, so every promotion shrinks description length — but
**steers the choice within that shortlist by a relational objective measured on a
TRAIN split** of synonym/hypernym pairs, and reports the result on a **held-out**
split.

Relational objective (higher is better): mean synonym coordinate-cosine + mean
hypernym containment. Antonym separation is left to M18.1 polarity / M18.3 edges.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .basis_search import _PRIME_SET, _MOL_SET, _unresolved_counts, mdl, relational_metrics, search
from .canonicalization import canon_label, normalize
from .definition_graph import DEFAULT_MAX_DEPTH, DefinitionGraph
from .meaning_value import axis_vector, cosine
from .semantic_axes import AxisRegistry

_MAX_PAIR_WORDS = 600  # cap the objective's working set for speed


def relation_pairs(words, graph: DefinitionGraph):
    """Deduped synonym, hypernym, and antonym pairs with both endpoints in *words*."""
    wset = set(words)
    syn, hyp, ant = set(), set(), set()
    for w in words:
        for s in graph.synonym.get(w, []):
            s = s.lower()
            if s in wset and s != w:
                syn.add(tuple(sorted((w, s))))
        for h in graph.is_a.get(w, []):
            h = h.lower()
            if h in wset and h != w:
                hyp.add((w, h))  # directional: w IS_A h
    ant = set(graph.antonym_pairs())
    return sorted(syn), sorted(hyp), sorted(ant)


def split_pairs(pairs, train_frac: float = 0.5):
    """Deterministic train/test split (crc32 of the pair, stable across runs)."""
    train, test = [], []
    for p in pairs:
        key = "|".join(p)
        (train if (zlib.crc32(key.encode()) % 100) < train_frac * 100 else test).append(p)
    return train, test


def _coords_for(words_set, axes, extra, depth, cache) -> Dict[str, np.ndarray]:
    ex = frozenset(extra)
    return {w: axis_vector(normalize(cache.decompose(w, depth, ex)), axes) for w in words_set}


def _rel_score(coord, syn_pairs, hyp_pairs, axes) -> float:
    syn = [cosine(coord[a], coord[b]) for a, b in syn_pairs if a in coord and b in coord]
    hyp = []
    for w, h in hyp_pairs:
        if w in coord and h in coord:
            wa = {ax for ax, x in zip(axes, coord[w]) if x > 0}
            ha = {ax for ax, x in zip(axes, coord[h]) if x > 0}
            if ha:
                hyp.append(len(ha & wa) / len(ha))
    return (float(np.mean(syn)) if syn else 0.0) + (float(np.mean(hyp)) if hyp else 0.0)


@dataclass
class MultiSignalResult:
    registry: AxisRegistry
    added: List[Tuple[str, float]]  # (axis, rel_delta)
    train_syn: List = field(default_factory=list)
    test_pairs: Dict = field(default_factory=dict)
    mdl_only_metrics: Dict = field(default_factory=dict)
    multisignal_metrics: Dict = field(default_factory=dict)


def multisignal_search(
    words,
    *,
    depth: int = DEFAULT_MAX_DEPTH,
    max_axes: int = 15,
    graph: Optional[DefinitionGraph] = None,
    cache=None,
    train_frac: float = 0.5,
    candidate_shortlist: int = 25,
) -> MultiSignalResult:
    """Grow the basis steering by relatedness on a train split; eval held-out."""
    words = [w.lower().strip() for w in words]
    if graph is None:
        graph = DefinitionGraph.build(words)
    if cache is None:
        from .cache import DecompCache
        cache = DecompCache(depth=depth).warm(words)

    syn, hyp, ant = relation_pairs(words, graph)
    train_syn, test_syn = split_pairs(syn, train_frac)
    train_hyp, test_hyp = split_pairs(hyp, train_frac)

    pair_words = sorted({w for p in (train_syn + train_hyp) for w in p})[:_MAX_PAIR_WORDS]

    reg = AxisRegistry.seed()
    extra: set = set()
    added: List[Tuple[str, float]] = []

    for _ in range(max_axes):
        counts = _unresolved_counts(words, frozenset(extra), depth, cache)
        shortlist = [
            w for w, _ in counts.most_common()
            if w not in extra and canon_label(w) not in _PRIME_SET and w not in _MOL_SET
        ][:candidate_shortlist]
        if not shortlist:
            break
        base_coord = _coords_for(pair_words, reg.axes, extra, depth, cache)
        base_score = _rel_score(base_coord, train_syn, train_hyp, reg.axes)
        best = None  # (rel_delta, word)
        for c in shortlist:
            axes_c = reg.axes + [c]
            coord_c = _coords_for(pair_words, axes_c, extra | {c}, depth, cache)
            rd = _rel_score(coord_c, train_syn, train_hyp, axes_c) - base_score
            if best is None or rd > best[0]:
                best = (rd, c)
        rd, cand = best
        extra.add(cand)
        reg.add(cand, f"multisignal:reldelta{rd:+.3f}")
        added.append((cand, rd))

    # Compare against an MDL-only basis of the same size on HELD-OUT pairs.
    mdl_reg = search(words, depth=depth, max_axes=max_axes, graph=graph, cache=cache).registry

    def _held_out_metrics(r: AxisRegistry) -> Dict:
        coord = _coords_for(sorted({w for p in (test_syn + test_hyp) for w in p}),
                            r.axes, r.extra_axes(), depth, cache)
        return {
            "synonym_cos": _rel_score(coord, test_syn, [], r.axes),
            "hypernym_containment": _rel_score(coord, [], test_hyp, r.axes),
            "n_test_syn": len(test_syn),
            "n_test_hyp": len(test_hyp),
        }

    return MultiSignalResult(
        registry=reg,
        added=added,
        train_syn=train_syn,
        test_pairs={"syn": test_syn, "hyp": test_hyp},
        mdl_only_metrics=_held_out_metrics(mdl_reg),
        multisignal_metrics=_held_out_metrics(reg),
    )
