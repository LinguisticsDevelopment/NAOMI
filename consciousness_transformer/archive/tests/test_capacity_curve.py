"""Tests for M57d's memory capacity-curve probe (scripts/probe_capacity_curve.py).

Synthetic-tensor tests, no parser dependency -- mirrors tests/test_instances.py's
own isolation discipline. Grids here are deliberately tiny (the probe script
itself runs the real, larger grid; these just check the measurement machinery
is correct, bounded, monotone-trending, and deterministic).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from nsm_ct import entity_memory as em  # noqa: E402
from probe_capacity_curve import (  # noqa: E402
    _batched_reads,
    _draw_seed_material,
    _fill_memory,
    measure_cell,
)

SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# The fast batched-read path reproduces the real entity_memory ops exactly.
# ---------------------------------------------------------------------------
def test_fill_and_reads_match_real_entity_memory_ops():
    dim, ni, nr, V = 8, 3, 2, 8
    e, r, cb, vi = _draw_seed_material(dim, ni, nr, V, "random", 0, None)
    entities = torch.from_numpy(e).unsqueeze(0)
    relations = torch.from_numpy(r).unsqueeze(0)
    codebook = torch.from_numpy(cb).unsqueeze(0)
    val_idx = torch.from_numpy(vi).unsqueeze(0)
    values = codebook[0, val_idx[0]].unsqueeze(0)

    memory = _fill_memory(entities, relations, values, ni, nr)

    # Reference: real nsm_ct.entity_memory.write, one call per fact, same order.
    ref = torch.zeros(1, dim, dim, dim)
    gate = torch.ones(1)
    for n in range(ni):
        for rel in range(nr):
            ref = em.write(ref, entities[:, n, :], relations[:, rel, :], values[:, n, rel, :], gate)
    assert torch.allclose(memory, ref)

    pred, entity_pred = _batched_reads(memory, entities, relations, values)
    for n in range(ni):
        for rel in range(nr):
            ref_q = em.query(memory, entities[:, n, :], relations[:, rel, :])
            assert torch.allclose(pred[:, n, rel, :], ref_q, atol=1e-5)
            ref_iq = em.query_entity(memory, relations[:, rel, :], values[:, n, rel, :])
            assert torch.allclose(entity_pred[:, n, rel, :], ref_iq, atol=1e-5)


# ---------------------------------------------------------------------------
# measure_cell: metric ranges, monotone trends, determinism.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("source", ["codec", "random"])
def test_metrics_are_in_unit_interval(source):
    from nsm_ct.tpr import TPRCodec

    dim = 16
    codec = TPRCodec(dim=dim) if source == "codec" else None
    m = measure_cell(dim, 4, 2, 8, source, SEEDS, codec=codec)
    for key in ("forward_acc", "overwrite_new_acc"):
        assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of [0,1]"
    assert 0.0 <= m["inverse_acc"] <= 1.0 or m["inverse_acc"] != m["inverse_acc"]  # allow NaN if no unique cases
    assert 0.0 <= m["inverse_coverage"] <= 1.0
    assert -1.0 <= m["overwrite_stale_cosine"] <= 1.0
    assert -1.0 <= m["forward_margin"] <= 1.0


def test_more_facts_does_not_increase_forward_accuracy():
    """More facts crammed into the same memory should never IMPROVE forward
    recall (it may hold steady while everything's still easy, but the trend
    must not go up) -- a real interference effect, on fixed seeds."""
    dim = 24
    small = measure_cell(dim, 4, 2, 8, "random", SEEDS)   # 8 facts
    large = measure_cell(dim, 16, 4, 8, "random", SEEDS)  # 64 facts
    assert large["forward_acc"] <= small["forward_acc"] + 1e-9


def test_larger_dim_does_not_decrease_forward_accuracy():
    """Holding the fact count fixed, a bigger memory should never recall
    WORSE than a smaller one (more room to be near-orthogonal), on fixed
    seeds."""
    small_dim = measure_cell(16, 8, 4, 8, "random", SEEDS)   # 32 facts, dim=16
    large_dim = measure_cell(64, 8, 4, 8, "random", SEEDS)   # 32 facts, dim=64
    assert large_dim["forward_acc"] >= small_dim["forward_acc"] - 1e-9


def test_determinism_same_seeds_same_numbers():
    dim = 20
    m1 = measure_cell(dim, 6, 3, 8, "random", SEEDS)
    m2 = measure_cell(dim, 6, 3, 8, "random", SEEDS)
    assert m1 == m2


def test_determinism_codec_source():
    from nsm_ct.tpr import TPRCodec

    dim = 20
    codec_a = TPRCodec(dim=dim)
    codec_b = TPRCodec(dim=dim)
    m1 = measure_cell(dim, 6, 3, 8, "codec", SEEDS, codec=codec_a)
    m2 = measure_cell(dim, 6, 3, 8, "codec", SEEDS, codec=codec_b)
    assert m1 == m2


# ---------------------------------------------------------------------------
# A tiny memory with one fact must recall perfectly (sanity floor).
# ---------------------------------------------------------------------------
def test_single_fact_recalls_perfectly():
    m = measure_cell(32, 1, 1, 8, "random", SEEDS)
    assert m["forward_acc"] == 1.0
