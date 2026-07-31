"""Tests for bootstrap_memory: seeding the NSM semantic web into LongTermMemory.

Spec (from the task brief):
* After seed_bootstrap_memory:
  - len(long_term) >= NUM_PRIMES + len(MOLECULES)
  - at least some molecule->prime edges exist (long_term.num_connections > 0)
  - long_term.read(torch.randn(1, consciousness_dim)) is non-zero
  - metas carry kind in {"prime", "molecule"}
  - molecule metas carry a non-empty "source"
* A subsequent consolidate of a NEW vector grows the repo (APPEND-style growth).
"""

import torch
import numpy as np
import pytest

from nsm_ct.long_term_memory import LongTermMemory
from nsm_ct.nsm_primes import NUM_PRIMES, PRIME_NAMES
from nsm_ct.nsm_molecules import MOLECULES
from nsm_ct.bootstrap_memory import (
    seed_bootstrap_memory,
    prime_signature_to_vector,
    _PROJECTION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MEMORY_DIM = 32
STATE_DIM = 32  # consciousness_dim


def _make_ltm(overwrite: bool = True) -> LongTermMemory:
    torch.manual_seed(0)
    return LongTermMemory(
        mem_dim=MEMORY_DIM,
        state_dim=STATE_DIM,
        max_size=10000,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Unit tests: projection helpers
# ---------------------------------------------------------------------------

def test_projection_shape_and_determinism():
    """_PROJECTION returns [NUM_PRIMES, memory_dim] and is cached/stable."""
    p1 = _PROJECTION(MEMORY_DIM)
    p2 = _PROJECTION(MEMORY_DIM)
    assert p1.shape == (NUM_PRIMES, MEMORY_DIM)
    # Exactly the same object (cached)
    assert p1 is p2


def test_projection_different_dims_are_independent():
    p16 = _PROJECTION(16)
    p32 = _PROJECTION(32)
    assert p16.shape == (NUM_PRIMES, 16)
    assert p32.shape == (NUM_PRIMES, 32)
    # Different dimensions must not share memory
    assert p16 is not p32


def test_prime_signature_to_vector_shape_and_tanh_range():
    sig = np.zeros(NUM_PRIMES, dtype=np.float32)
    sig[0] = 1.0
    vec = prime_signature_to_vector(sig, MEMORY_DIM)
    assert vec.shape == (MEMORY_DIM,)
    assert vec.dtype == np.float32
    # tanh output is strictly in (-1, 1)
    assert np.all(vec > -1.0) and np.all(vec < 1.0)


def test_prime_signature_zero_gives_zero_vector():
    """Zero signature -> zero projection -> tanh(0) = 0 everywhere."""
    sig = np.zeros(NUM_PRIMES, dtype=np.float32)
    vec = prime_signature_to_vector(sig, MEMORY_DIM)
    np.testing.assert_allclose(vec, np.zeros(MEMORY_DIM, dtype=np.float32))


def test_different_primes_give_different_vectors():
    sig_a = np.zeros(NUM_PRIMES, dtype=np.float32); sig_a[0] = 1.0
    sig_b = np.zeros(NUM_PRIMES, dtype=np.float32); sig_b[-1] = 1.0
    va = prime_signature_to_vector(sig_a, MEMORY_DIM)
    vb = prime_signature_to_vector(sig_b, MEMORY_DIM)
    assert not np.allclose(va, vb), "Different prime one-hots must produce different vectors"


# ---------------------------------------------------------------------------
# Integration: seed_bootstrap_memory
# ---------------------------------------------------------------------------

def test_seed_returns_correct_total():
    ltm = _make_ltm()
    total = seed_bootstrap_memory(ltm, MEMORY_DIM)
    assert total == NUM_PRIMES + len(MOLECULES), (
        f"Expected {NUM_PRIMES + len(MOLECULES)} entries, got {total}"
    )


def test_ltm_length_ge_primes_plus_molecules():
    ltm = _make_ltm()
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    assert len(ltm) >= NUM_PRIMES + len(MOLECULES)


def test_edges_exist_after_bootstrap():
    """Molecule->prime (and category-group) edges must be created."""
    ltm = _make_ltm()
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    assert ltm.num_connections > 0, "Expected molecule->prime edges after bootstrap"


def test_read_is_nonzero_after_bootstrap():
    ltm = _make_ltm()
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    state = torch.randn(1, STATE_DIM)
    out = ltm.read(state)
    assert out.shape == (1, MEMORY_DIM)
    assert out.abs().sum().item() > 0, "read() should be non-zero after seeding"


def test_meta_kinds_are_correct():
    """Every meta must have kind in {"prime", "molecule"}."""
    ltm = _make_ltm()
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    kinds = {m.get("kind") for m in ltm.metas}
    assert kinds == {"prime", "molecule"}, f"Unexpected kinds: {kinds}"


def test_molecule_metas_have_nonempty_source():
    """Anti-hallucination: every molecule entry must carry a non-empty source."""
    ltm = _make_ltm()
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    for meta in ltm.metas:
        if meta.get("kind") == "molecule":
            src = meta.get("source", "")
            assert src, f"Molecule {meta.get('name')!r} has empty source"


def test_prime_metas_are_complete():
    """All NUM_PRIMES are present and have correct meta fields."""
    ltm = _make_ltm()
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    prime_metas = [m for m in ltm.metas if m.get("kind") == "prime"]
    assert len(prime_metas) == NUM_PRIMES
    names_seeded = {m["name"] for m in prime_metas}
    assert names_seeded == set(PRIME_NAMES)


def test_molecule_metas_are_complete():
    """All MOLECULES are present and have correct meta fields."""
    ltm = _make_ltm()
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    mol_metas = [m for m in ltm.metas if m.get("kind") == "molecule"]
    assert len(mol_metas) == len(MOLECULES)
    names_seeded = {m["name"] for m in mol_metas}
    expected = {mol.name for mol in MOLECULES}
    assert names_seeded == expected


def test_molecule_metas_carry_exponents():
    """Molecule metas must carry a non-empty exponents list."""
    ltm = _make_ltm()
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    for meta in ltm.metas:
        if meta.get("kind") == "molecule":
            exps = meta.get("exponents", [])
            assert exps, f"Molecule {meta.get('name')!r} has empty exponents"


# ---------------------------------------------------------------------------
# APPEND growth: a new consolidate after bootstrap grows the repo
# ---------------------------------------------------------------------------

def test_subsequent_consolidate_grows_repo():
    """APPEND-style growth: a genuinely new vector expands the repo."""
    ltm = _make_ltm()
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    size_before = len(ltm)

    # Generate a vector that is orthogonal enough to avoid overwriting any existing entry.
    # Use a very sparse random vector — it will not match any seeded vector.
    torch.manual_seed(999)
    new_vec = torch.zeros(1, MEMORY_DIM)
    new_vec[0, MEMORY_DIM // 2] = 5.0  # dominated by a single dimension, unusual pattern
    new_meta = [{"text": "test:new_fact_xyz", "kind": "learned"}]

    idxs = ltm.consolidate(new_vec, metas=new_meta)
    assert len(ltm) > size_before, (
        f"After seeding {size_before} entries, a novel consolidate should grow the repo"
    )
    assert len(idxs) == 1


def test_re_seeding_does_not_duplicate_entries():
    """Re-running seed_bootstrap_memory with overwrite=True must not duplicate entries."""
    ltm = _make_ltm(overwrite=True)
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    size_after_first = len(ltm)

    # Re-seed: the 'text' key dedup in LongTermMemory._overwrite_match should prevent duplication.
    seed_bootstrap_memory(ltm, MEMORY_DIM)
    assert len(ltm) == size_after_first, (
        f"Re-seeding should not duplicate entries "
        f"(was {size_after_first}, now {len(ltm)})"
    )


# ---------------------------------------------------------------------------
# Integration with build_default_stack
# ---------------------------------------------------------------------------

def test_build_default_stack_seeds_bootstrap():
    """When use_long_term=True, build_default_stack pre-populates LTM."""
    from nsm_ct import build_default_stack, load_config
    from nsm_ct.episode import CurriculumGenerator

    cfg = load_config()
    cfg.model.use_long_term = True
    cfg.model.d_model = 32
    cfg.model.num_layers = 1
    cfg.model.nhead = 2
    cfg.model.dim_feedforward = 32
    cfg.input_encoder = "token"
    cfg.data.num_episodes = 8

    episodes = CurriculumGenerator(max_level=1, seed=0).generate(8)
    stack = build_default_stack(cfg, episodes)

    assert stack.long_term is not None
    assert len(stack.long_term) >= NUM_PRIMES + len(MOLECULES), (
        f"Stack LTM should contain at least {NUM_PRIMES + len(MOLECULES)} bootstrap entries"
    )
    # Facts list should be non-empty
    assert any(stack.long_term.facts())


def test_build_default_stack_no_ltm_unchanged():
    """When use_long_term=False, long_term stays None (no behaviour change)."""
    from nsm_ct import build_default_stack, load_config
    from nsm_ct.episode import CurriculumGenerator

    cfg = load_config()
    cfg.model.use_long_term = False
    cfg.data.num_episodes = 4

    episodes = CurriculumGenerator(max_level=1, seed=0).generate(4)
    stack = build_default_stack(cfg, episodes)

    assert stack.long_term is None
