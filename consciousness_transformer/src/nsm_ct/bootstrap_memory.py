"""Bootstrap long-term memory with the NSM semantic web.

The NSM semantic web — primes and molecules, with their prime groundings — is
seeded into a :class:`~nsm_ct.long_term_memory.LongTermMemory` as baked-in
"fact" knowledge the model STARTS with.  Subsequent APPEND actions during the
reasoning loop grow the repository on top of this foundation.

Design notes
------------
* **Projection**: a fixed (seeded) random projection maps a NUM_PRIMES-wide
  signature (one-hot for primes, multi-hot for molecules) to memory_dim via
  ``tanh(signature @ P)``, mirroring :class:`~nsm_ct.semantic_mapper.SemanticRepresentation`.
* **Primes first, then molecules** — prime indices are stable before molecules
  are consolidated, so the molecule→prime edge wiring is unambiguous.
* **Anti-hallucination**: every molecule entry carries the ``source`` citation
  from :mod:`nsm_ct.nsm_molecules` verbatim.
* **Deterministic**: ``np.random.default_rng(seed=42)`` is used for the
  projection matrix and cached per ``memory_dim``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .long_term_memory import LongTermMemory
from .nsm_molecules import MOLECULES, MoleculeCategory
from .nsm_primes import NUM_PRIMES, PRIME_NAMES, prime_index
from .thought import meaning_prime_ids

# ---------------------------------------------------------------------------
# Fixed projection: NUM_PRIMES -> memory_dim
# ---------------------------------------------------------------------------

_PROJ_CACHE: Dict[int, np.ndarray] = {}
_PROJ_SEED = 42  # deterministic; never change (would shift all bootstrap vectors)


def _PROJECTION(memory_dim: int) -> np.ndarray:
    """Return (and cache) the fixed [NUM_PRIMES, memory_dim] projection matrix.

    Args:
        memory_dim: Width of the target memory vectors.

    Returns:
        float32 array of shape ``[NUM_PRIMES, memory_dim]``, stable across calls
        with the same ``memory_dim``.
    """
    if memory_dim not in _PROJ_CACHE:
        rng = np.random.default_rng(seed=_PROJ_SEED)
        proj = rng.standard_normal((NUM_PRIMES, memory_dim)).astype(np.float32)
        _PROJ_CACHE[memory_dim] = proj
    return _PROJ_CACHE[memory_dim]


def prime_signature_to_vector(
    signature: np.ndarray,
    memory_dim: int,
) -> np.ndarray:
    """Project a NUM_PRIMES-wide activation signature to a memory vector.

    Args:
        signature: float32 array of shape ``[NUM_PRIMES]``.
        memory_dim: Target dimensionality.

    Returns:
        float32 array of shape ``[memory_dim]``:
        ``tanh(signature @ _PROJECTION(memory_dim))``.
    """
    proj = _PROJECTION(memory_dim)
    return np.tanh(signature.astype(np.float32) @ proj).astype(np.float32)


# ---------------------------------------------------------------------------
# Bootstrap seed
# ---------------------------------------------------------------------------

def seed_bootstrap_memory(
    long_term: LongTermMemory,
    memory_dim: int,
    resolver=None,
) -> int:
    """Seed *long_term* with the NSM prime/molecule semantic web.

    Primes are consolidated first (one entry each, one-hot signature) so their
    indices are stable before molecule→prime edges are created.  Then molecules
    are consolidated (multi-hot signature over constituent primes) and connected
    to their constituent primes.  Finally, molecules within the same category
    are group-connected.

    Args:
        long_term: A :class:`LongTermMemory` instance (may already be empty or
            have prior content; bootstrap entries are appended/overwritten using
            the normal ``consolidate`` dedup logic).
        memory_dim: Width of stored vectors.  Must equal ``long_term.mem_dim``.
        resolver: Optional :class:`~nsm_ct.meaning.NSMMeaningResolver` used to
            compute a molecule's prime bag when the molecule has no
            ``explication``.  Defaults to a freshly constructed
            ``NSMMeaningResolver()``.

    Returns:
        Total number of entries that were seeded (primes + molecules).
    """
    if resolver is None:
        from .meaning import NSMMeaningResolver
        resolver = NSMMeaningResolver()

    # ------------------------------------------------------------------ #
    # 1.  Seed PRIMES                                                      #
    # ------------------------------------------------------------------ #
    prime_vecs: List[np.ndarray] = []
    prime_metas: List[dict] = []

    for name in PRIME_NAMES:
        sig = np.zeros(NUM_PRIMES, dtype=np.float32)
        sig[prime_index(name)] = 1.0
        vec = prime_signature_to_vector(sig, memory_dim)
        prime_vecs.append(vec)
        prime_metas.append({
            "text": f"prime:{name}",
            "name": name,
            "kind": "prime",
        })

    prime_tensor = torch.from_numpy(
        np.stack(prime_vecs, axis=0).astype(np.float32)
    )
    prime_indices: List[int] = long_term.consolidate(prime_tensor, metas=prime_metas)
    # Map name -> consolidated index
    prime_name_to_idx: Dict[str, int] = {
        name: idx for name, idx in zip(PRIME_NAMES, prime_indices)
    }

    # ------------------------------------------------------------------ #
    # 2.  Seed MOLECULES                                                   #
    # ------------------------------------------------------------------ #
    mol_vecs: List[np.ndarray] = []
    mol_metas: List[dict] = []
    # Per-molecule: which prime names were activated (for later edge wiring)
    mol_prime_names_list: List[List[str]] = []

    for mol in MOLECULES:
        # Compute the prime bag for this molecule.
        if mol.explication is not None:
            raw_ids = meaning_prime_ids(mol.explication)
        else:
            # Fall back: resolve the first exponent via the resolver, then
            # extract prime ids from the resulting meaning tree.
            try:
                tree = resolver.resolve(mol.exponents[0])
                raw_ids = meaning_prime_ids(tree)
            except Exception:
                raw_ids = []

        # Build multi-hot signature (id is 1-based; index = id - 1).
        sig = np.zeros(NUM_PRIMES, dtype=np.float32)
        activated_prime_names: List[str] = []
        for pid in raw_ids:
            if 1 <= pid <= NUM_PRIMES:
                idx = pid - 1
                sig[idx] = 1.0
                activated_prime_names.append(PRIME_NAMES[idx])

        # Fallback if still empty: use the molecule's own name hash to pick a prime
        # (preserves non-zero vectors so the memory is always meaningful).
        if sig.sum() == 0.0:
            import hashlib
            h = int(hashlib.sha256(mol.name.encode()).hexdigest(), 16)
            fallback_idx = h % NUM_PRIMES
            sig[fallback_idx] = 0.5
            activated_prime_names.append(PRIME_NAMES[fallback_idx])

        vec = prime_signature_to_vector(sig, memory_dim)
        mol_vecs.append(vec)
        mol_prime_names_list.append(activated_prime_names)
        mol_metas.append({
            "text": f"molecule:{mol.name}",
            "name": mol.name,
            "kind": "molecule",
            "source": mol.source,
            "exponents": list(mol.exponents),
            "category": mol.category.value,
        })

    mol_tensor = torch.from_numpy(
        np.stack(mol_vecs, axis=0).astype(np.float32)
    )
    mol_indices: List[int] = long_term.consolidate(mol_tensor, metas=mol_metas)
    # Map molecule name -> consolidated index
    mol_name_to_idx: Dict[str, int] = {
        mol.name: idx for mol, idx in zip(MOLECULES, mol_indices)
    }

    # ------------------------------------------------------------------ #
    # 3.  Edges: molecule -> constituent primes                            #
    # ------------------------------------------------------------------ #
    for mol, mol_idx, activated_names in zip(MOLECULES, mol_indices, mol_prime_names_list):
        for pname in activated_names:
            p_idx = prime_name_to_idx.get(pname)
            if p_idx is not None:
                long_term.connect(mol_idx, p_idx, weight=1.0)

    # ------------------------------------------------------------------ #
    # 4.  Edges: group-connect molecules by category                       #
    # ------------------------------------------------------------------ #
    category_groups: Dict[MoleculeCategory, List[int]] = {}
    for mol in MOLECULES:
        cat = mol.category
        if cat not in category_groups:
            category_groups[cat] = []
        mol_idx_for_cat = mol_name_to_idx.get(mol.name)
        if mol_idx_for_cat is not None:
            category_groups[cat].append(mol_idx_for_cat)

    for cat, idxs in category_groups.items():
        if len(idxs) > 1:
            long_term.connect_group(idxs, weight=0.5)

    total = len(prime_indices) + len(mol_indices)
    return total
