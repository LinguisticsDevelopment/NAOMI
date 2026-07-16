"""Contrastive placement objective (M21, checkpoint b).

The missing lever: placement so far only PULLS related words together. Here we
optimize the per-axis VALUES so that, jointly, synonyms→close, antonyms→opposite,
and random/unrelated→far (negative sampling). The optimization is transparent and
keeps every axis interpretable: it only adjusts a word's values on the axes that
already apply to it (the Null mask is re-applied each step, so words never gain
content on inapplicable axes and axes are never rotated/mixed) — a word's value on
``EVAL`` is still its goodness, just fit to the relations.

Honest contract (per the plan): if this can't push held-out syn>ant above the M20
tanh baseline (0.756 held-out; originally cited as 0.94, which M24 found leaked) at
the per-word level, that is recorded as a negative — the relational propagation may
remain necessary — not hidden.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .sparse_value import SparseSpace


def contrastive_optimize(
    space: SparseSpace,
    *,
    train_syn: np.ndarray,
    train_ant: np.ndarray,
    n_neg: int = 4000,
    iters: int = 300,
    lr: float = 0.05,
    w_syn: float = 1.0,
    w_ant: float = 1.0,
    w_neg: float = 1.0,
    w_anchor: float = 0.5,
    seed: int = 0,
) -> SparseSpace:
    """Return a new SparseSpace whose values are contrastively optimized (mask/axes
    unchanged, so the representation stays Null-aware and interpretable)."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    N = space.value.shape[0]

    mask = torch.tensor(space.mask, dtype=torch.float32)
    sw = torch.tensor(np.sqrt(space.idf), dtype=torch.float32)
    V0 = torch.tensor(space.value, dtype=torch.float32)  # the principled anchored signs
    V = V0.clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)

    syn = torch.tensor(train_syn, dtype=torch.long) if len(train_syn) else None
    ant = torch.tensor(train_ant, dtype=torch.long) if len(train_ant) else None

    def sim(pairs):
        a = V[pairs[:, 0]] * sw
        b = V[pairs[:, 1]] * sw
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)

    for _ in range(iters):
        neg = torch.tensor(rng.randint(0, N, (n_neg, 2)), dtype=torch.long)
        opt.zero_grad()
        loss = w_neg * (sim(neg) ** 2).mean()            # random -> 0
        loss = loss + w_anchor * ((V - V0) ** 2).mean()  # stay near principled signs
        if syn is not None:
            loss = loss + w_syn * (1.0 - sim(syn)).mean()  # synonym -> +1
        if ant is not None:
            loss = loss + w_ant * (1.0 + sim(ant)).clamp(min=0).mean()  # antonym -> -1
        loss.backward()
        opt.step()
        with torch.no_grad():
            V.mul_(mask)  # re-impose the Null mask: keep sparsity + interpretability

    return SparseSpace(words=space.words, axes=space.axes,
                       value=V.detach().numpy().astype(np.float32),
                       mask=space.mask, idf=space.idf)
