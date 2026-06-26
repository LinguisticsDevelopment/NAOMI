"""Graph-aware relational closeness (M18.3).

The honest end of the antonym story. M18.1 showed coordinates alone can't push
synonym/antonym discrimination above chance. Here the **WordNet relational web
does the work the coordinates can't**: closeness = polarity-coordinate cosine,
minus an antonym penalty derived from the graph — but using **only train-split
edges**, and propagated one hop through synonyms so a *held-out* antonym pair is
penalised via *other* antonym edges, never its own.

Evaluation is a clean held-out discrimination: do held-out **synonym** pairs get
higher closeness than held-out **antonym** pairs? Compared three ways — pure
(unsigned) coordinate, polarity coordinate, and graph-aware closeness — so the
edges' contribution is isolated and circularity-free (the test pair's own edge is
in the test split, never the train edges).
"""

from __future__ import annotations

import zlib
from typing import Callable, Dict, List, Tuple

import numpy as np

from .definition_graph import DefinitionGraph
from .meaning_value import cosine


def split_pairs(pairs, train_frac: float = 0.5):
    """Deterministic train/test split (crc32 of the pair)."""
    train, test = [], []
    for p in pairs:
        key = "|".join(p)
        (train if (zlib.crc32(key.encode()) % 100) < train_frac * 100 else test).append(p)
    return train, test


def _adj(pairs) -> Dict[str, set]:
    adj: Dict[str, set] = {}
    for a, b in pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def antonym_linked(a: str, b: str, train_ant: Dict[str, set], syn: Dict[str, set]) -> bool:
    """True if *a* and *b* are antonym-linked through TRAIN antonym edges, allowing
    one synonym hop (a is the antonym of a synonym of b, or vice versa)."""
    if b in train_ant.get(a, ()):  # direct train edge (absent for held-out pairs)
        return True
    for x in train_ant.get(a, ()):
        if b in syn.get(x, ()) or x in syn.get(b, ()):
            return True
    for y in train_ant.get(b, ()):
        if a in syn.get(y, ()) or y in syn.get(a, ()):
            return True
    return False


def make_graph_closeness(coord, train_ant, syn, *, lam: float = 1.0):
    """A closeness(a, b) using polarity coordinate minus a train-graph antonym penalty."""
    def close(a: str, b: str) -> float:
        base = cosine(coord[a], coord[b])
        if antonym_linked(a, b, train_ant, syn):
            base -= lam
        return base
    return close


def discrimination(syn_pairs, ant_pairs, close: Callable[[str, str], float], coord) -> Tuple[float, int, int]:
    """P(closeness of a synonym pair > closeness of an antonym pair) — an AUC-like
    held-out separation score (0.5 = chance). Pairs missing a coordinate are skipped."""
    syn_scores = np.array([close(a, b) for a, b in syn_pairs if a in coord and b in coord], dtype=np.float32)
    ant_scores = np.array([close(a, b) for a, b in ant_pairs if a in coord and b in coord], dtype=np.float32)
    if len(syn_scores) == 0 or len(ant_scores) == 0:
        return 0.5, len(syn_scores), len(ant_scores)
    wins = float((syn_scores[:, None] > ant_scores[None, :]).mean())
    return wins, len(syn_scores), len(ant_scores)


def evaluate_closeness(
    words,
    *,
    graph: DefinitionGraph,
    coord_unsigned: Dict[str, np.ndarray],
    coord_polarity: Dict[str, np.ndarray],
    train_frac: float = 0.5,
    lam: float = 1.0,
) -> Dict:
    """Compare pure-coordinate, polarity-coordinate, and graph-aware closeness on a
    held-out synonym-vs-antonym discrimination."""
    wset = set(words)
    syn = []
    for w in words:
        for s in graph.synonym.get(w, []):
            s = s.lower()
            if s in wset and s != w:
                syn.append(tuple(sorted((w, s))))
    syn = sorted(set(syn))
    ant = sorted(set(graph.antonym_pairs()))

    train_ant, test_ant = split_pairs(ant, train_frac)
    _, test_syn = split_pairs(syn, train_frac)

    train_ant_adj = _adj(train_ant)
    syn_adj = _adj(syn)  # synonym edges are not the held-out signal -> use all

    pure = discrimination(test_syn, test_ant, lambda a, b: cosine(coord_unsigned[a], coord_unsigned[b]), coord_unsigned)
    polar = discrimination(test_syn, test_ant, lambda a, b: cosine(coord_polarity[a], coord_polarity[b]), coord_polarity)
    graph_close = make_graph_closeness(coord_polarity, train_ant_adj, syn_adj, lam=lam)
    graphed = discrimination(test_syn, test_ant, graph_close, coord_polarity)

    return {
        "pure_coordinate": pure[0],
        "polarity_coordinate": polar[0],
        "graph_closeness": graphed[0],
        "n_test_syn": pure[1],
        "n_test_ant": pure[2],
        "n_train_ant": len(train_ant),
    }
