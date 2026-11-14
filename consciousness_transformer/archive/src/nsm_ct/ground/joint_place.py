"""Joint all-signals placement (M25) — train the position, don't propagate it.

The placement so far folds relations into position by DETERMINISTIC label propagation
over synonym+similar edges only (`placement.relax`). M23 showed propagation's wall: it
only helps pairs *connected* by training edges and does not generalize. Here we instead
FIT the per-word values on the SAME named axis set using ALL relational signals jointly
(synonym / similar / antonym / hypernym / meronym+derivational / random), so the position
is learned as a function of the whole relational context.

Thesis guardrails (hard): values live on the existing NAMED axes; the Null mask (the
axes that actually apply to a word) is re-imposed every step, so words never gain content
on inapplicable axes (non-overlap, minimum values) and axes never rotate or mix
(interpretable). Only the values move, initialized from the anchored coordinate.

Held-out by construction: the caller passes TRAIN-split pairs only; the scored test pairs
are never in the optimization (the M24 rule).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .axes import MeaningAxes
from .placement import anchored_coordinate
from .relations import RelationGraph


def _pidx(pairs, idx) -> np.ndarray:
    return np.array([(idx[a], idx[b]) for a, b in pairs if a in idx and b in idx and a != b], dtype=np.int64)


def joint_place(
    words,
    graph: RelationGraph,
    axes: MeaningAxes,
    *,
    cache=None,
    depth: int = 3,
    train_syn: List = None,
    train_sim: List = None,
    train_ant: List = None,
    train_hyp: List = None,
    train_rel: List = None,          # meronym + derivational (mild relatedness)
    n_neg: int = 4000,
    iters: int = 300,
    lr: float = 0.05,
    w_syn: float = 1.0,
    w_sim: float = 0.5,
    w_ant: float = 1.0,
    w_hyp: float = 0.5,
    w_rel: float = 0.3,
    w_neg: float = 1.0,
    w_anchor: float = 0.5,
    t_hyp: float = 0.5,
    t_rel: float = 0.4,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Fit per-word values on the named axes from ALL relation signals (train split),
    masked to applicable axes. Returns {word: vector} over `axes` (drop-in for `place`)."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    words = list(words)
    idx = {w: i for i, w in enumerate(words)}
    N = len(words)

    A = anchored_coordinate(words, graph, axes, cache=cache, depth=depth).astype(np.float32)
    mask_np = (np.abs(A) > 1e-9).astype(np.float32)   # the Null mask: applicable axes only

    mask = torch.tensor(mask_np)
    A0 = torch.tensor(A)
    V = A0.clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)

    def t(pairs):
        arr = _pidx(pairs or [], idx)
        return torch.tensor(arr, dtype=torch.long) if len(arr) else None

    syn, sim, ant, hyp, rel = t(train_syn), t(train_sim), t(train_ant), t(train_hyp), t(train_rel)

    def cos(pairs):
        a = V[pairs[:, 0]]
        b = V[pairs[:, 1]]
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)

    for _ in range(iters):
        neg = torch.tensor(rng.randint(0, N, (n_neg, 2)), dtype=torch.long)
        opt.zero_grad()
        loss = w_neg * (cos(neg) ** 2).mean()                 # random -> 0 (separation)
        loss = loss + w_anchor * ((V - A0) ** 2).mean()       # stay near anchored meaning
        if syn is not None:
            loss = loss + w_syn * (1.0 - cos(syn)).mean()     # synonym -> +1
        if sim is not None:
            loss = loss + w_sim * (1.0 - cos(sim)).mean()     # similar -> close
        if ant is not None:
            loss = loss + w_ant * (1.0 + cos(ant)).clamp(min=0).mean()   # antonym -> -1
        if hyp is not None:
            loss = loss + w_hyp * (t_hyp - cos(hyp)).clamp(min=0).mean() ** 1  # hypernym -> related
        if rel is not None:
            loss = loss + w_rel * (t_rel - cos(rel)).clamp(min=0).mean()       # mero/deriv -> mild
        loss.backward()
        opt.step()
        with torch.no_grad():
            V.mul_(mask)   # re-impose the Null mask: sparsity + interpretability preserved

    out = V.detach().numpy().astype(np.float32)
    return {w: out[i] for i, w in enumerate(words)}


def split_all_relations(words, graph: RelationGraph, *, train_frac: float = 0.5) -> Tuple[Dict, Dict]:
    """Deterministic held-out split of EVERY relation used by the joint objective
    (hypernym = the ``is_a`` store). Returns (train, test) dicts keyed by short name."""
    from .closeness import split_pairs

    def tp(rel):
        try:
            return graph.typed_pairs(rel)
        except Exception:
            return []

    groups = {
        "syn": tp("synonym"),
        "sim": tp("similar"),
        "ant": tp("antonym"),
        "hyp": tp("is_a"),                                   # hypernym
        "rel": sorted(set(tp("meronym")) | set(tp("derivational"))),
    }
    train, test = {}, {}
    for name, pairs in groups.items():
        train[name], test[name] = split_pairs(pairs, train_frac)
    return train, test
