"""M57a: the entity-instance subsystem (registry + minting + attribute-fact
writes + provenance log). See dev/MIND_INTERFACE.md's "v2 addendum -- the
entity-instance subsystem" and CLAUDE.md's "M57 memory-schema decision".

Standing defect this fixes: today an entity atom is minted deterministically
FROM THE NAME STRING (``codec.filler_vec("var:" + name)``,
:func:`nsm_ct.clause_reactor._ent_vec`) -- so "mary" is the SAME vector
everywhere and a second person who happens to share that name is
unrepresentable (they'd collapse onto the same memory entity). An INSTANCE is
a discourse referent: a fresh, arbitrary atom (a variable, not a meaning --
mary#1 != mary#2) that the word "mary" gets bound to as an ATTRIBUTE FACT
(``name(e, "mary")``) rather than as an identity.

Two pieces, cleanly separated per the locked design (knowledge lives in the
tensor, not the index):

- :class:`InstanceRegistry` -- an ENUMERATION INDEX ONLY. Maps instance ids
  (``"inst:mary#1"``) to their minted atom vectors so candidate generation
  can enumerate which instances exist. Carries ZERO knowledge about any
  instance's properties -- those are ordinary writes into the existing
  entity(x)relation(x)value memory (:mod:`nsm_ct.entity_memory`), exactly
  like every other relation fact: entity = the instance atom, relation = an
  ``"attr:<name>"`` filler vector minted via :class:`nsm_ct.tpr.TPRCodec`
  the SAME way ``"rel:"`` relations already are (``codec.filler_vec("rel:" +
  name)``, e.g. :func:`nsm_ct.clause_reactor._reasoning_steps`), value = the
  attribute's own vector (name/kind/gender ...). See
  :func:`write_attribute` / :func:`query_attribute`.
- :class:`ProvenanceLog` -- a membrane-side, append-only audit trail (plain
  Python, one record per write). The tensor CANNOT carry this (superposition
  destroys who-said-what/when/how-trusted); MIND_INTERFACE.md invariant #4
  ("every memory write is gated, local, and auditable") is satisfied by this
  log, not by the tensor itself.

Unbatched by design: instances are DISCOURSE-level bookkeeping (one registry
per episode/document), not BATCH-level like :mod:`nsm_ct.entity_memory`'s
``[B, ...]`` tensors -- mirroring how :mod:`nsm_ct.membrane` itself is
per-episode candidate-set bookkeeping around the batched reactor. Every op
here takes a single ``[d]`` atom / ``[d, d, d]`` memory (internally these
wrap :mod:`nsm_ct.entity_memory`'s batched ops with a batch size of 1 --
that module has no unbatched variant, so this is the seam). A caller that
needs this batched over an episode dimension (M57b, resolver-driven
write-back -- explicitly OUT OF SCOPE here) wraps these ops in the same
per-episode Python loop :mod:`nsm_ct.clause_reactor` already uses for
candidate-set construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import entity_memory as em
from .membrane import Candidate, CandidateSet
from .tpr import TPRCodec

__all__ = [
    "InstanceRegistry",
    "ProvenanceRecord",
    "ProvenanceLog",
    "write_attribute",
    "query_attribute",
    "candidates_for",
    "inverse_query",
    "to_candidate_set",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_tensor(x, dtype: torch.dtype) -> torch.Tensor:
    """Accept either a torch tensor or a numpy array (codec vectors are
    numpy) and return a detached ``dtype`` torch tensor -- the same
    ``torch.from_numpy`` seam :mod:`nsm_ct.clause_reactor` uses everywhere
    it consumes a :class:`~nsm_ct.tpr.TPRCodec` vector."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dtype)


def _attr_vec(attr_name: str, codec: TPRCodec, dtype: torch.dtype) -> torch.Tensor:
    """The relation vector for an attribute fact: ``codec.filler_vec("attr:"
    + attr_name)``, minted the SAME way ``"rel:"`` relations already are
    (see module docstring) -- an attribute fact is an ordinary relation, not
    a new kind of thing."""
    return _as_tensor(codec.filler_vec("attr:" + attr_name), dtype)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity, ``[..., d] x [..., d] -> [...]``. Same
    norm+eps convention as :mod:`nsm_ct.clause_reactor`'s contrastive answer
    (``rn = r / (r.norm(...) + 1e-8)``)."""
    an = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    bn = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (an * bn).sum(dim=-1)


# ---------------------------------------------------------------------------
# InstanceRegistry -- enumeration index only, zero knowledge.
# ---------------------------------------------------------------------------
class InstanceRegistry:
    """Mints fresh instance atoms and remembers (id -> atom) so candidate
    generation can enumerate which instances exist. NOT a knowledge store --
    see module docstring.

    Minting is deterministic given ``seed``: a registry built with the same
    ``seed`` and minted in the same order reproduces byte-identical atoms
    (:func:`test_instances.test_determinism_same_seed`). Atoms are fresh
    i.i.d. unit Gaussian directions (not hashed from the name -- that would
    reintroduce the exact "identity IS the string" defect this subsystem
    fixes), so two mints of the SAME ``name_hint`` are two DIFFERENT atoms
    (the two-Marys premise): ``mary#1 != mary#2``.

    Args:
        dim: vector dimension ``d``, matching the :class:`TPRCodec` /
            entity-memory dimension in use.
        seed: RNG seed; a fresh :class:`numpy.random.Generator` per registry
            (never the codec's global seed -- instance atoms are discourse-
            local, not part of the deterministic label codebook).
    """

    def __init__(self, dim: int, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._ids: List[str] = []
        self._atoms: Dict[str, torch.Tensor] = {}
        self._name_counts: Dict[str, int] = {}

    def mint(self, name_hint: str) -> Tuple[str, torch.Tensor]:
        """Mint one fresh instance atom, id'd ``"inst:<name_hint>#<n>"``
        where ``n`` is the 1-based count of mints for that ``name_hint``
        within THIS registry (so ``mint("mary")`` twice yields
        ``"inst:mary#1"`` then ``"inst:mary#2"``, per the addendum's own
        example). Returns ``(instance_id, atom)``; ``atom`` is a unit-norm
        ``[dim]`` tensor drawn fresh from the registry's RNG stream (fresh
        variable, not a meaning)."""
        key = (name_hint or "").lower()
        n = self._name_counts.get(key, 0) + 1
        self._name_counts[key] = n
        instance_id = f"inst:{key}#{n}"
        v = self._rng.standard_normal(self.dim).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        atom = torch.from_numpy(v)
        self._ids.append(instance_id)
        self._atoms[instance_id] = atom
        return instance_id, atom

    def ids(self) -> List[str]:
        """Every minted instance id, in mint order."""
        return list(self._ids)

    def atoms(self) -> torch.Tensor:
        """Every minted atom stacked in mint order: ``[N, dim]`` (``N == 0``
        gives an empty ``[0, dim]`` tensor, never an error)."""
        if not self._ids:
            return torch.zeros(0, self.dim)
        return torch.stack([self._atoms[i] for i in self._ids], dim=0)

    def lookup(self, instance_id: str) -> torch.Tensor:
        """The minted atom for ``instance_id``; raises ``KeyError`` if it
        was never minted by this registry."""
        return self._atoms[instance_id]

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, instance_id: str) -> bool:
        return instance_id in self._atoms


# ---------------------------------------------------------------------------
# ProvenanceLog -- membrane-side, append-only, auditable.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProvenanceRecord:
    """One audit-trail entry for one gated attribute write. Immutable
    (``frozen=True``) -- once written, a record cannot be edited in place;
    only new records get appended. ``timestamp`` is a caller-supplied value
    (e.g. a step counter or a fixed clock reading passed in by the caller),
    NEVER a wall-clock read, so callers -- and tests -- stay deterministic."""

    instance_id: str
    relation: str
    value_label: str
    source: str
    language: str
    timestamp: float
    trust: float
    step: Optional[int] = None
    # M57d (PROVENANCE wiring into the live reactor, CLAUDE.md's M57
    # memory-schema decision): two OPTIONAL fields, both defaulting to
    # ``None`` -- every pre-M57d caller of :func:`write_attribute` (which
    # never sets these) is unaffected. ``surface`` is the human-readable
    # context-sentence text the write came from (when the caller has one);
    # ``candidate_ids`` is the ordered tuple of candidate instance ids the
    # write's address was resolved AMONG (only meaningful for a redirected
    # write -- ``None``/empty for a directly-addressed one). Together they
    # are what :func:`nsm_ct.provenance.explain` needs to render a readable
    # audit line ("... resolved from [inst:mary#1, inst:mary#2]") that the
    # bare (instance_id, relation, value_label) fields above cannot supply
    # on their own.
    surface: Optional[str] = None
    candidate_ids: Optional[Tuple[str, ...]] = None


class ProvenanceLog:
    """Append-only log of :class:`ProvenanceRecord`\\ s -- the auditable
    record MIND_INTERFACE.md invariant #4 requires. The entity(x)relation(x)
    value tensor is a superposed, non-auditable store by construction (that
    is the whole point of a distributed memory); this log is the structure
    that answers "what was written, when, from what evidence, at what trust
    setting" for a given instance."""

    def __init__(self) -> None:
        self._records: List[ProvenanceRecord] = []

    def append(self, record: ProvenanceRecord) -> None:
        self._records.append(record)

    def records_for(self, instance_id: str) -> List[ProvenanceRecord]:
        """Every record touching ``instance_id``, in write order."""
        return [r for r in self._records if r.instance_id == instance_id]

    @property
    def records(self) -> Tuple[ProvenanceRecord, ...]:
        """All records, in write order, as an immutable tuple (a copy --
        callers cannot mutate the log through this view)."""
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)


# ---------------------------------------------------------------------------
# Attribute facts -- ordinary gated writes/reads into entity_memory.
# ---------------------------------------------------------------------------
def write_attribute(
    memory: torch.Tensor,
    registry: InstanceRegistry,
    instance_id: str,
    attr_name: str,
    value_vec,
    codec: TPRCodec,
    gate: float = 1.0,
    *,
    log: ProvenanceLog,
    source: str,
    language: str,
    timestamp: float,
    trust: float,
    value_label: str,
    step: Optional[int] = None,
    overwrite: Optional[float] = None,
) -> torch.Tensor:
    """Write one attribute fact -- ``instance_id``'s ``attr_name`` :=
    ``value_vec`` -- as an ordinary gated write into ``memory`` (entity =
    the instance's minted atom, relation = ``codec.filler_vec("attr:" +
    attr_name)``), and append one :class:`ProvenanceRecord` to ``log``.

    Out-of-place (like :func:`nsm_ct.entity_memory.write`): returns a NEW
    ``[dim, dim, dim]`` memory tensor, ``memory`` itself is untouched.
    ``gate=1.0`` with ``overwrite=None`` (-> ``overwrite=gate``) is the
    default -- a plain overwrite/update, per the locked design ("gate=1
    overwrite semantics by default"); pass a smaller ``gate`` /
    ``overwrite=0`` for the accumulate ("vote") reaction
    :func:`nsm_ct.entity_memory.write` already supports.

    ``value_label`` is a caller-supplied human-readable string for the audit
    trail (e.g. ``"mary"``, ``"doctor"``, ``"F"``) -- the tensor only ever
    holds the vector, never a label, so the log is the only place this is
    recoverable at all.
    """
    entity = registry.lookup(instance_id)
    relation = _attr_vec(attr_name, codec, memory.dtype)
    value = _as_tensor(value_vec, memory.dtype)
    g = torch.tensor([float(gate)], dtype=memory.dtype)
    ow = None if overwrite is None else torch.tensor([float(overwrite)], dtype=memory.dtype)
    new_memory = em.write(
        memory.unsqueeze(0), entity.unsqueeze(0), relation.unsqueeze(0),
        value.unsqueeze(0), g, overwrite=ow,
    ).squeeze(0)
    log.append(ProvenanceRecord(
        instance_id=instance_id, relation="attr:" + attr_name, value_label=value_label,
        source=source, language=language, timestamp=timestamp, trust=trust, step=step,
    ))
    return new_memory


def query_attribute(
    memory: torch.Tensor,
    registry: InstanceRegistry,
    instance_id: str,
    attr_name: str,
    codec: TPRCodec,
) -> torch.Tensor:
    """Read ``instance_id``'s ``attr_name`` value back out of ``memory``:
    ``[dim]``. Exact when the (instance atom, attr relation) keys of every
    stored cell are orthonormal; otherwise noisy (interference from other
    bindings sharing the memory), same caveat as
    :func:`nsm_ct.entity_memory.query` -- clean up against a value codebook
    (e.g. :meth:`nsm_ct.tpr.TPRCodec.cleanup`, or an argmax-cosine over a
    small candidate-value codebook, as the tests do) to recover a label."""
    entity = registry.lookup(instance_id)
    relation = _attr_vec(attr_name, codec, memory.dtype)
    return em.query(memory.unsqueeze(0), entity.unsqueeze(0), relation.unsqueeze(0)).squeeze(0)


# ---------------------------------------------------------------------------
# Candidate generation by attribute match -- every referring expression
# ("mary" -> name=mary instances, "the doctor" -> kind=doctor instances)
# becomes an enumerate-and-score pass over the registry.
# ---------------------------------------------------------------------------
def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise dot product, ``[..., d] x [..., d] -> [...]`` -- magnitude-
    AWARE, unlike :func:`_cosine`. See :func:`candidates_for`'s docstring
    (``score`` kwarg) for why this is now the default: a candidate whose
    readout is a small SCALED COPY of the target direction (the
    single-writer interference case) ties :func:`_cosine` at ~1.0 with the
    true match but is correctly discounted here."""
    return (a * b).sum(dim=-1)


def candidates_for(
    memory: torch.Tensor,
    registry: InstanceRegistry,
    codec: TPRCodec,
    *,
    attr_name: str,
    target_vec,
    threshold: float = 0.5,
    score: str = "dot",
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Enumerate every minted instance, read its ``attr_name`` slot back out
    of ``memory``, score against ``target_vec``, and keep the ones at or
    above ``threshold`` -- the general mechanism behind EVERY referring
    expression in the v2 addendum's design: "mary" is
    ``candidates_for(attr_name="name", target_vec=vec("mary"))`` (two Marys
    = a real 2-candidate set), "the doctor" is
    ``candidates_for(attr_name="kind", target_vec=vec("doctor"))``.

    ``score`` -- ``"dot"`` (default) or ``"cosine"``. Recorded gap this
    fixes (dev/OP_INVENTORY.md Sec.5, "``candidates_for`` has no tie-
    break"): when a relation has exactly ONE writer in the whole memory,
    every OTHER instance's readout at that relation is a small SCALED COPY
    of the writer's value (interference term ``(other_atom . writer_atom) *
    value``, same direction, tiny magnitude) -- ``cosine`` normalizes the
    scale away and reports ~1.0 for BOTH the true writer and the scaled
    copy, a false-positive tie; ``dot`` is magnitude-aware and correctly
    scores the scaled copy near zero. ``"cosine"`` is kept for callers that
    want the old direction-only behavior (e.g. comparing candidates whose
    readouts legitimately differ in norm for reasons other than
    interference). See :func:`nsm_ct.ops.similarity` for the shared
    dot+cosine primitive this mirrors.

    Returns ``(ids, atoms, scores)`` -- the matching instance ids, their
    ``[K, dim]`` stacked atoms, and their ``[K]`` scores, in registry
    (mint) order restricted to matches. Empty (``K=0``) tensors, never an
    error, when nothing clears ``threshold`` or the registry is empty. See
    :func:`to_candidate_set` for the thin adapter onto
    :mod:`nsm_ct.membrane`'s :class:`~nsm_ct.membrane.CandidateSet` shape
    that the existing resolver contract consumes.
    """
    ids = registry.ids()
    d = memory.shape[-1]
    if not ids:
        return [], torch.zeros(0, d, dtype=memory.dtype), torch.zeros(0, dtype=memory.dtype)
    n = len(ids)
    atoms = registry.atoms().to(memory.dtype)                          # [N, d]
    relation = _attr_vec(attr_name, codec, memory.dtype).unsqueeze(0).expand(n, -1)  # [N, d]
    mem_batch = memory.unsqueeze(0).expand(n, -1, -1, -1)               # [N, d, d, d] (view)
    values = em.query(mem_batch, atoms, relation)                       # [N, d]
    target = _as_tensor(target_vec, memory.dtype).unsqueeze(0).expand(n, -1)
    if score == "dot":
        scores = _dot(values, target)                                   # [N]
    elif score == "cosine":
        scores = _cosine(values, target)                                # [N]
    else:
        raise ValueError(f"candidates_for: unknown score {score!r}, expected 'dot' or 'cosine'")
    keep = scores >= threshold
    kept_ids = [i for i, k in zip(ids, keep.tolist()) if k]
    return kept_ids, atoms[keep], scores[keep]


def inverse_query(
    memory: torch.Tensor,
    registry: InstanceRegistry,
    codec: TPRCodec,
    attr_name: str,
    value_vec,
    *,
    score: str = "dot",
) -> Tuple[List[str], torch.Tensor]:
    """"Who is a doctor?" -- the same enumerate-and-score machinery as
    :func:`candidates_for`, but returns scores over EVERY minted instance,
    unthresholded (the caller decides how many to keep / where the margin
    is, e.g. by taking the top-K). ``score`` -- see :func:`candidates_for`
    (``"dot"`` default, magnitude-aware). Returns ``(ids, scores)`` in
    registry (mint) order; ``scores`` is ``[N]``, empty if the registry is
    empty."""
    ids, atoms, scores = candidates_for(
        memory, registry, codec, attr_name=attr_name, target_vec=value_vec, threshold=-1.0, score=score,
    )
    return ids, scores


def to_candidate_set(
    ids: List[str],
    scores: torch.Tensor,
    *,
    provenance: Optional[Dict[str, object]] = None,
) -> CandidateSet:
    """Thin adapter: wrap :func:`candidates_for`'s ``(ids, scores)`` into
    the EXISTING :mod:`nsm_ct.membrane` candidate-set shape
    (:class:`~nsm_ct.membrane.CandidateSet` / Candidate.key = instance id)
    so instance-based candidate generation plugs into the SAME resolver
    contract (:mod:`nsm_ct.resolver`) that already consumes
    :class:`~nsm_ct.membrane.EntityCandidateSet` /
    :class:`~nsm_ct.membrane.SenseCandidateSet` -- unchanged.
    :mod:`nsm_ct.membrane` itself stays torch-free by its own docstring's
    hard constraint ("no torch: the membrane types are perception-side"),
    which is why this adapter lives here (where torch/codec are already in
    scope) rather than there.

    Priors are the match scores normalized to sum to 1 (clamped at 0 first
    -- a negative cosine is not "more implausible than impossible"), mirror
    ing :class:`~nsm_ct.membrane.SenseCandidateSet`'s own "one honest
    structural signal, not a learned one" prior philosophy: falls back to a
    uniform prior only if every score is non-positive (the same "perception
    never guesses beyond what it can support" contract). Empty candidate
    list -> an empty :class:`~nsm_ct.membrane.CandidateSet`, matching every
    other builder in :mod:`nsm_ct.membrane`.
    """
    if not ids:
        return CandidateSet(provenance=dict(provenance or {}))
    s = np.clip(scores.detach().cpu().numpy().astype(np.float64), 0.0, None)
    total = float(s.sum())
    priors = (s / total) if total > 0 else np.full(len(ids), 1.0 / len(ids))
    candidates = [Candidate(key=k, prior=float(p)) for k, p in zip(ids, priors)]
    return CandidateSet(candidates=candidates, provenance=dict(provenance or {}))
