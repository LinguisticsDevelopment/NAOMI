"""M31 bridge: USVS coordinates as deterministic filler/handle vectors.

The mind's graph nodes carry a lossy vector *address* (handle) next to the
lossless structure. This bridge makes a content word's handle a **fixed
deterministic projection of its USVS coordinate** — no learning, rebuild-stable
(the projection is keyed on axis NAMES, not indices, so it survives axis-set
changes for unchanged axes). The USVS artifact stays the readable source of
truth; the handle is just its address in a d-dim slot.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "usvs"


@lru_cache(maxsize=1)
def default_usvs(path: str = str(_DEFAULT_DIR)):
    from .ground.usvs import load_usvs
    return load_usvs(path)


def _axis_row(name: str, d: int) -> np.ndarray:
    """Deterministic unit vector for one named axis (seeded by the name)."""
    seed = int.from_bytes(hashlib.sha256(f"{name}|{d}".encode()).digest()[:8], "big")
    rng = np.random.RandomState(seed % (2**32))
    v = rng.standard_normal(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


@lru_cache(maxsize=8)
def projection(axes_key: tuple, d: int) -> np.ndarray:
    """[n_axes, d] projection matrix, one deterministic row per axis name."""
    return np.stack([_axis_row(name, d) for name in axes_key])


def _project(coord: np.ndarray, axes: list, d: int) -> np.ndarray:
    P = projection(tuple(axes), d)
    v = coord.astype(np.float32) @ P
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def usvs_handle(word: str, d: int, *, usvs=None) -> Optional[np.ndarray]:
    """The word's d-dim handle: placed core coordinate if the word is in the
    core, else its MFS sense signature; None if USVS doesn't know the word."""
    u = usvs or default_usvs()
    coord = u.word_coord(word)
    if coord is None:
        sids = u.senses_of(word)
        if not sids:
            return None
        coord = u.sense_dense(sids[0])
        if coord is None or not coord.any():
            return None
    return _project(coord, u.axes, d)


def usvs_sense_handle(sense_id: str, d: int, *, usvs=None) -> Optional[np.ndarray]:
    u = usvs or default_usvs()
    coord = u.sense_dense(sense_id)
    if coord is None or not coord.any():
        return None
    return _project(coord, u.axes, d)
