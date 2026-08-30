"""The deterministic op library: consciousness_transformer's explicit
instruction set, in one place.

dev/OP_INVENTORY.md names 22 ops, fixed vs. learned, and where each lives
today (scattered across `entity_memory.py`, `instances.py`, `ltm.py`,
`resolver.py`, and inline in `clause_reactor.py`). This module is the
robustification pass for that inventory: **one pure function per op**,
covering every FIXED op the table lists plus the ones OP_INVENTORY names
as gaps (recency, cleanup-with-abstain, permute, temporal links, register
file). It does not change any existing call path -- see
`dev/OP_LIBRARY_MAP.md` for exactly where each op below will plug into
`clause_reactor.py` at a LATER integration milestone.

Literature grounding (cited per-op below, not repeated here): VSA
bind/unbind/superpose/permute/cleanup/similarity (Plate 1995 Holographic
Reduced Representations; Kanerva 2009 hyperdimensional computing; Gayler
1998/2003 Multiply-Add-Permute); DNC temporal links/allocation/erase
(Graves et al. 2016, "Hybrid computing using a neural network with dynamic
external memory"); CLS consolidation tiers (McClelland, McNaughton &
O'Reilly 1995, complementary learning systems); centering theory salience
(Grosz, Joshi & Weinstein 1995; Walker, Joshi & Prince 1998).

Conventions
-----------
- Batched-first: every op that touches the entity-memory tensor takes
  ``[B, ...]`` inputs, matching `nsm_ct.entity_memory`. Where
  `nsm_ct.instances` already established an UNBATCHED single-instance
  convenience (`memory` `[d, d, d]` instead of `[B, d, d, d]`), the
  corresponding op here accepts either shape and auto-detects which one
  it got (mirroring `instances.write_attribute`'s own
  ``unsqueeze(0)/squeeze(0)`` seam), so callers never have to fake a
  batch dimension of 1 for single-episode bookkeeping.
- Every dial is a **module-level named constant** (MIND_INTERFACE.md
  invariant #6: dials are explicit named scalars, never magic numbers).
- No gradients are blocked anywhere a real training signal could need
  one (`bind_write`/`unbind_query`/`erase`/`superpose_vote`/`select`
  (soft)/`branch` (soft) are fully differentiable); a few ops are
  legitimately eval-only reference implementations (`cleanup`'s argmax,
  `select`/`branch` hard mode, `halt`, `abstain`) -- called out inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from . import entity_memory as em
from .instances import InstanceRegistry
from .ltm import link_decision as link          # TIERS: re-export, unchanged
from .ltm import mem_total as recall             # TIERS: re-export, unchanged
from .ltm import promote                         # TIERS: re-export, unchanged

__all__ = [
    # dials
    "CLEANUP_MARGIN", "FORGET_DECAY", "PATIENCE", "CAUTION", "BRANCH_THRESHOLD",
    "RECENCY_NEVER", "PERMUTE_SEED",
    # memory
    "bind_write", "unbind_query", "inverse_query_entity", "superpose_vote", "erase",
    "cleanup", "similarity", "Similarity", "permute", "unpermute", "allocate",
    "recency", "RecencyFeatures", "temporal_link", "forget_decay",
    # tiers (re-exports)
    "recall", "promote", "link",
    # control
    "select", "abstain", "compare", "branch", "halt", "emit",
    # registers
    "RegisterFile",
]

EPS = 1e-8  # same norm+eps convention as instances.py's _cosine / clause_reactor's contrastive answer

# ---------------------------------------------------------------------------
# Dials -- explicit, named, never buried in code (MIND_INTERFACE.md #6).
# ---------------------------------------------------------------------------
CLEANUP_MARGIN: float = 0.1
"""`cleanup`'s abstain threshold: below this top1-top2 score gap, cleanup
reports `abstain=True` instead of trusting its argmax -- the `caution`
dial (dev/MIND_INTERFACE.md) made concrete for the cleanup op."""

FORGET_DECAY: float = 1.0
"""`forget_decay`'s multiplicative rate; `1.0` = off (identity, byte-
exact no-op) -- MIND_INTERFACE.md's "byte-identical by default" rule."""

PATIENCE: int = 12  # raised 6->12 (director, 2026-08-30): K_max must fit the longest gold program (writeback_addr_redirect = 12 op-steps; EXECUTOR_DESIGN Sec 1.3)
"""`halt`'s thinking-budget cutoff -- matches dev/TRACK_C_DESIGN.md
Sec.1.6's `K_max = 6` ("one slack step over the longest known [4-5 step]
gold program"), the same `patience` dial MIND_INTERFACE.md names."""

CAUTION: float = 0.1
"""`abstain`'s margin threshold -- MIND_INTERFACE.md's `caution` dial
("minimum collapse margin to hard-bind; below it the ambiguity is HELD,
not guessed"), realized here as the deterministic v1 gate dev/
OP_INVENTORY.md Sec.5 flags as never wired anywhere."""

BRANCH_THRESHOLD: float = 0.5
"""`branch`'s hard-mode decision threshold on `cond_scalar` (a cosine/
probability-shaped `[-1, 1]` or `[0, 1]` value)."""

RECENCY_NEVER: float = 1e6
"""`recency`'s sentinel `steps_since` value for a candidate that has
never been mentioned (large but finite, so it stays a valid float
feature rather than `inf`, which would poison a downstream MLP)."""

PERMUTE_SEED: int = 0x5EED
"""Base seed for `permute`/`unpermute`'s fixed base permutation family
(deterministic, non-trainable -- one seeded `torch.randperm` per
dimension `d`, exactly like `tpr.TPRCodec`'s fillers are one seeded draw
per label)."""


# ---------------------------------------------------------------------------
# MEMORY -- VSA bind/unbind/superpose/cleanup/similarity/permute,
# DNC-flavored allocate/temporal_link/forget_decay.
# ---------------------------------------------------------------------------
def _is_unbatched_memory(memory: torch.Tensor) -> bool:
    """`True` for a single `[d, d, d]` memory, `False` for `[B, d, d, d]`."""
    return memory.dim() == 3


def bind_write(
    memory: torch.Tensor,
    entity: torch.Tensor,
    relation: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    overwrite: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """VSA **bind** (Plate 1995 HRR; Smolensky 1990 TPR) plus a DNC-style
    gated write: direct wrap of `nsm_ct.entity_memory.write`
    (dev/OP_INVENTORY.md's "write (bind)" row).

    Parameters
    ----------
    memory : Tensor
        `[B, d, d, d]`, or `[d, d, d]` (unbatched convenience -- every
        other argument is then treated unbatched too, and the result is
        squeezed back, mirroring `nsm_ct.instances.write_attribute`'s own
        `unsqueeze(0)/squeeze(0)` seam).
    entity, relation, value : Tensor
        `[B, d]` (or `[d]` to match an unbatched `memory`).
    gate : Tensor
        `[B]` write strength in `[0, 1]` (or a python float / 0-d tensor
        for the unbatched path).
    overwrite : Tensor, optional
        `[B]`; defaults to `gate` (see `entity_memory.write`).

    Returns
    -------
    Tensor
        The new memory, same shape as `memory`. Out-of-place.
    """
    if not _is_unbatched_memory(memory):
        return em.write(memory, entity, relation, value, gate, overwrite=overwrite)
    g = gate.reshape(1) if torch.is_tensor(gate) else torch.tensor([float(gate)])
    ow = (
        None if overwrite is None
        else (overwrite.reshape(1) if torch.is_tensor(overwrite) else torch.tensor([float(overwrite)]))
    )
    out = em.write(
        memory.unsqueeze(0), entity.unsqueeze(0), relation.unsqueeze(0), value.unsqueeze(0), g, overwrite=ow,
    )
    return out.squeeze(0)


def unbind_query(memory: torch.Tensor, entity: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
    """VSA **unbind** (matched-filter decoding, Plate/Kanerva/Gayler):
    direct wrap of `nsm_ct.entity_memory.query` (dev/OP_INVENTORY.md's
    "query" row). Same batched/unbatched convenience as `bind_write`.

    Returns
    -------
    Tensor
        `[B, d]` (or `[d]` unbatched): the value bound to `(entity,
        relation)`, exact only when the stored keys are orthonormal.
    """
    if not _is_unbatched_memory(memory):
        return em.query(memory, entity, relation)
    return em.query(memory.unsqueeze(0), entity.unsqueeze(0), relation.unsqueeze(0)).squeeze(0)


def inverse_query_entity(memory: torch.Tensor, relation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """VSA **unbind** along the ENTITY axis instead of the value axis:
    direct wrap of `nsm_ct.entity_memory.query_entity`
    (dev/OP_INVENTORY.md's "query_entity (inverse)" row) -- "who holds
    (relation, value)?" instead of `unbind_query`'s "what does (entity,
    relation) hold?". Same batched/unbatched convenience as `bind_write`.
    """
    if not _is_unbatched_memory(memory):
        return em.query_entity(memory, relation, value)
    return em.query_entity(memory.unsqueeze(0), relation.unsqueeze(0), value.unsqueeze(0)).squeeze(0)


def superpose_vote(
    memory: torch.Tensor,
    entity: torch.Tensor,
    relation: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """VSA **superpose** (bundling; Plate/Kanerva -- repeated assertions
    accumulate additively so the majority direction wins) realized as
    `bind_write` with `overwrite=0`: `nsm_ct.entity_memory.write`'s own
    "a vote -- repeated assertions accumulate" branch
    (dev/OP_INVENTORY.md's "write (bind)" row, "overwrite vs vote" row).
    Same batched/unbatched convenience as `bind_write`.
    """
    gate_t = gate if torch.is_tensor(gate) else torch.as_tensor(gate, dtype=torch.float32)
    zero = torch.zeros_like(gate_t)
    return bind_write(memory, entity, relation, value, gate, overwrite=zero)


def erase(memory: torch.Tensor, entity: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
    """DNC-style **erase** (Graves et al. 2016's erase vector, which
    clears a written slot's content before new content lands) realized as
    an explicit slot clear: ``memory - outer(entity, relation, readout)``
    where ``readout = unbind_query(memory, entity, relation)``.

    Equivalent to (and cross-checked in tests/test_ops.py against)
    calling `bind_write` at the same slot with ``gate=0, overwrite=1``
    (the "write with negative gate" framing) -- both zero that
    `(entity, relation)` binding's contribution to any future read at
    the same address, leaving every other slot untouched.
    """
    unbatched = _is_unbatched_memory(memory)
    if unbatched:
        memory, entity, relation = memory.unsqueeze(0), entity.unsqueeze(0), relation.unsqueeze(0)
    readout = em.query(memory, entity, relation)
    out = memory - torch.einsum("bi,bj,bk->bijk", entity, relation, readout)
    return out.squeeze(0) if unbatched else out


def cleanup(
    vec: torch.Tensor,
    codebook: torch.Tensor,
    *,
    margin_dial: float = CLEANUP_MARGIN,
    mode: str = "cosine",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VSA **cleanup** (Plate 1995 HRR; Kanerva 2009 hyperdimensional
    computing): snap a noisy, possibly-superposed vector to its nearest
    codebook entry, and report whether the match is decisive enough to
    trust (`abstain_flag`) -- the `caution` dial (MIND_INTERFACE.md) made
    concrete: this realizes sanctioned uncertainty forms 3 and 4 (a low
    margin, an abstention) as one gate, `dev/OP_INVENTORY.md`'s "caution
    never gates anything" gap. `nsm_ct.tpr.TPRCodec.cleanup` is the
    existing single-vector, cosine-only, dict-keyed analogue; this is its
    batched, dot-or-cosine, tensor-codebook generalization.

    Parameters
    ----------
    vec : Tensor
        `[B, d]` (or `[d]`, squeezed back on return).
    codebook : Tensor
        `[N, d]`, `N >= 1`.
    margin_dial : float, default `CLEANUP_MARGIN`
        Abstain when `top1 - top2 < margin_dial`.
    mode : {"cosine", "dot"}
        `"cosine"` (default, matches `TPRCodec.cleanup`'s own
        direction-only convention) or `"dot"` (magnitude-aware --
        the fix item 5 applies to `instances.candidates_for`, see
        `similarity`'s docstring for why magnitude carries signal a
        pure-direction score throws away).

    Returns
    -------
    index : LongTensor
        `[B]`, the argmax codebook row (returned even when abstaining --
        the caller decides whether to trust it).
    cleaned_vec : Tensor
        `[B, d]`, `codebook[index]`.
    margin : Tensor
        `[B]`, `top1 - top2` score gap (`+inf` when `codebook` has a
        single row -- no ambiguity possible).
    abstain : BoolTensor
        `[B]`, `margin < margin_dial`.
    """
    unbatched = vec.dim() == 1
    if unbatched:
        vec = vec.unsqueeze(0)
    if codebook.shape[0] == 0:
        raise ValueError("cleanup: codebook is empty")
    if mode == "cosine":
        v = F.normalize(vec, dim=-1, eps=EPS)
        cb = F.normalize(codebook, dim=-1, eps=EPS)
    elif mode == "dot":
        v, cb = vec, codebook
    else:
        raise ValueError(f"cleanup: unknown mode {mode!r}, expected 'cosine' or 'dot'")
    scores = v @ cb.t()                                   # [B, N]
    n = scores.shape[-1]
    k = min(2, n)
    top, idx = torch.topk(scores, k=k, dim=-1)
    index = idx[:, 0]
    cleaned = codebook.index_select(0, index)
    if k == 2:
        margin = top[:, 0] - top[:, 1]
    else:
        margin = torch.full((scores.shape[0],), float("inf"), device=vec.device, dtype=scores.dtype)
    abstain_flag = margin < margin_dial
    if unbatched:
        return index.squeeze(0), cleaned.squeeze(0), margin.squeeze(0), abstain_flag.squeeze(0)
    return index, cleaned, margin, abstain_flag


@dataclass
class Similarity:
    """`similarity`'s return value: both comparators side by side."""

    cosine: torch.Tensor
    dot: torch.Tensor


def similarity(a: torch.Tensor, b: torch.Tensor) -> Similarity:
    """VSA **similarity** (Plate/Kanerva/Gayler), both variants at once:
    direction-only (`cosine`) and magnitude-aware (`dot`), batched
    `[..., d] x [..., d] -> [...]` (broadcastable).

    Why both: a vector that is a small SCALED COPY of the right direction
    (e.g. `nsm_ct.entity_memory`'s cross-term interference when a
    relation has exactly one writer -- every OTHER entity's readout at
    that relation is `(their_atom . writer_atom) * value`, same
    direction, tiny magnitude) ties `cosine` at ~1.0 with the true match
    but is correctly discounted by `dot`. This is exactly item 5's fix to
    `nsm_ct.instances.candidates_for`/`inverse_query` (see that module
    and `tests/test_instances.py::test_dot_breaks_cosine_tie_single_writer`).

    Returns
    -------
    Similarity
        `.cosine`, `.dot`, each shaped like the broadcast of `a`, `b`
        with the last dim reduced.
    """
    dot = (a * b).sum(dim=-1)
    an = a.norm(dim=-1)
    bn = b.norm(dim=-1)
    cos = dot / (an * bn + EPS)
    return Similarity(cosine=cos, dot=dot)


_perm_cache: Dict[Tuple[int, int], torch.Tensor] = {}
_base_perm_cache: Dict[int, torch.Tensor] = {}


def _base_permutation(dim: int) -> torch.Tensor:
    if dim not in _base_perm_cache:
        g = torch.Generator().manual_seed(PERMUTE_SEED + dim)
        _base_perm_cache[dim] = torch.randperm(dim, generator=g)
    return _base_perm_cache[dim]


def _permutation_power(dim: int, k: int) -> torch.Tensor:
    """The fixed base permutation `Pi` (one per `dim`, seeded from
    `PERMUTE_SEED`) raised to integer power `k` (negative = power of its
    inverse), cached per `(dim, k)`. Powers of ONE fixed permutation
    (rather than `k` independent random permutations) is what makes
    `permute` compose: `Pi^a . Pi^b == Pi^(a+b)`."""
    key = (dim, k)
    if key in _perm_cache:
        return _perm_cache[key]
    base = _base_permutation(dim)
    if k == 0:
        perm = torch.arange(dim)
    elif k > 0:
        perm = torch.arange(dim)
        for _ in range(k):
            perm = base[perm]
    else:
        inv = torch.empty_like(base)
        inv[base] = torch.arange(dim)
        perm = torch.arange(dim)
        for _ in range(-k):
            perm = inv[perm]
    _perm_cache[key] = perm
    return perm


def permute(vec: torch.Tensor, k: int) -> torch.Tensor:
    """VSA **permute** (Plate 1995's protect-and-permute; Gayler's MAP;
    Kanerva 2009's permutation-for-sequence-encoding): apply `Pi^k`, the
    `k`-th power of one FIXED base permutation, to the last dimension --
    the standard VSA device for binding a filler to a POSITION/ORDER slot
    without needing a second filler vector (`bind` binds two *different*
    vectors; `permute` binds one vector to a slot index). Composes
    exactly: ``permute(permute(x, 1), 1) == permute(x, 2)`` (see
    `_permutation_power`).

    Parameters
    ----------
    vec : Tensor
        `[..., d]`.
    k : int
        Permutation power; `k=0` is the identity.
    """
    d = vec.shape[-1]
    perm = _permutation_power(d, k).to(vec.device)
    return vec.index_select(-1, perm)


def unpermute(vec: torch.Tensor, k: int) -> torch.Tensor:
    """Exact inverse of `permute`: ``unpermute(permute(x, k), k) == x``
    (permutation matrices are orthogonal; `Pi^k`'s inverse is `Pi^{-k}`,
    which `_permutation_power` computes directly)."""
    return permute(vec, -k)


def allocate(registry: InstanceRegistry, name_hint: str) -> Tuple[str, torch.Tensor]:
    """DNC-style **allocation** (Graves et al. 2016: a fresh address for
    new content, gated by a learned usage vector in the DNC; here, the
    deterministic v1 policy dev/OP_INVENTORY.md Sec.4 requires before any
    learned gate is added) realized as
    `nsm_ct.instances.InstanceRegistry.mint` (dev/OP_INVENTORY.md's
    "mint (instance)" row) -- direct re-export under the op-algebra's
    DNC-derived name, no new logic.
    """
    return registry.mint(name_hint)


@dataclass
class RecencyFeatures:
    """`recency`'s return value: three per-candidate salience features,
    each `[B, C]`."""

    steps_since: torch.Tensor
    log_count: torch.Tensor
    is_most_recent: torch.Tensor


def recency(
    mention_steps: torch.Tensor,
    current_step: Union[torch.Tensor, int, float],
    *,
    mention_counts: Optional[torch.Tensor] = None,
) -> RecencyFeatures:
    """Centering-theory-style salience features (Grosz, Joshi & Weinstein
    1995; Walker, Joshi & Prince 1998: the most recently/frequently
    mentioned entity is the likeliest pronoun referent), as per-candidate
    scorer features -- dev/OP_INVENTORY.md's named gap ("no recency
    feature in the resolver... same-gender pronoun / ambiguous name cases
    stay low; resolver has no recency register at all"). Feeds the same
    per-candidate feature slot `evidence_interaction` already occupies
    (dev/TRACK_C_DESIGN.md Sec.1.4's `P.feat[i]`).

    Parameters
    ----------
    mention_steps : Tensor
        `[B, C]`, the step index each candidate was last mentioned at, or
        any negative value (e.g. `-1`) for "never mentioned this
        episode".
    current_step : Tensor or scalar
        `[B]`, or a python scalar broadcast to every row.
    mention_counts : Tensor, optional
        `[B, C]`, how many times each candidate has been mentioned so
        far; defaults to zeros (`log_count` is then `0` everywhere,
        `is_most_recent` is unaffected).

    Returns
    -------
    RecencyFeatures
        `steps_since` : `[B, C]` float, `current_step - mention_steps`,
            clamped to `RECENCY_NEVER` wherever `mention_steps < 0`
            (monotone: a candidate mentioned longer ago always scores
            higher `steps_since` than one mentioned more recently).
        `log_count` : `[B, C]` float, `log1p(mention_counts)` (monotone
            in count, compresses the tail, `0` counts give exactly `0`).
        `is_most_recent` : `[B, C]` bool, one-hot at each row's minimum
            `steps_since` among EVER-mentioned candidates (ties broken by
            lower candidate index); all-`False` for a row with no
            mentions at all.
    """
    if not torch.is_tensor(current_step):
        current_step = torch.full((mention_steps.shape[0],), float(current_step))
    current_step = current_step.to(torch.float32).unsqueeze(-1)          # [B, 1]
    mention_steps = mention_steps.to(torch.float32)
    never = mention_steps < 0                                             # [B, C]
    steps_since = current_step - mention_steps
    steps_since = torch.where(never, torch.full_like(steps_since, RECENCY_NEVER), steps_since)

    if mention_counts is None:
        mention_counts = torch.zeros_like(mention_steps)
    log_count = torch.log1p(mention_counts.to(torch.float32).clamp(min=0))

    masked = torch.where(never, torch.full_like(steps_since, float("inf")), steps_since)
    min_val, min_idx = masked.min(dim=-1)
    has_any = ~never.all(dim=-1)
    is_most_recent = torch.zeros_like(mention_steps, dtype=torch.bool)
    rows = torch.nonzero(has_any, as_tuple=True)[0]
    if rows.numel() > 0:
        is_most_recent[rows, min_idx[rows]] = True

    return RecencyFeatures(steps_since=steps_since, log_count=log_count, is_most_recent=is_most_recent)


def temporal_link(write_order: Sequence) -> Tuple[Dict, Dict]:
    """DNC-style **temporal link** (Graves et al. 2016's temporal link
    matrix `L`, which after enough steps converges to "most recent
    transition wins"): a deterministic reference giving each written slot
    its immediate predecessor/successor in WRITE ORDER -- the closed form
    of what `L` asymptotically represents, not an approximation of it.

    Parameters
    ----------
    write_order : sequence of hashable
        One entry per write EVENT, e.g. `(entity_id, relation_id)` pairs
        (a slot may repeat if written more than once; the LATEST
        occurrence is what `predecessor`/`successor` describe).

    Returns
    -------
    predecessor, successor : dict
        Slot id -> the id of the slot written immediately before/after
        its most recent occurrence in `write_order`. A slot missing from
        one of these dicts has no such neighbor (first/last write, or a
        slot that only ever appears at one end of the sequence).
    """
    predecessor: Dict = {}
    successor: Dict = {}
    for prev, curr in zip(write_order, write_order[1:]):
        successor[prev] = curr
        predecessor[curr] = prev
    return predecessor, successor


def forget_decay(memory: torch.Tensor, decay: float = FORGET_DECAY) -> torch.Tensor:
    """A fixed-rate forgetting law over the WHOLE memory tensor --
    complementary to `promote`'s selective, trust-gated tier
    consolidation (McClelland, McNaughton & O'Reilly 1995's complementary
    learning systems: a fast-decaying store alongside a slow, curated
    one). `decay=1.0` (the `FORGET_DECAY` default) is the identity, byte-
    exact no-op -- dev/OP_INVENTORY.md Sec.4's "deterministic v1 policy
    first" rule; this has no live call path yet.
    """
    return memory * decay


# ---------------------------------------------------------------------------
# CONTROL -- deterministic reference implementations of the Track C ops
# the executor will later route between (dev/TRACK_C_DESIGN.md Sec.1.3/1.5).
# ---------------------------------------------------------------------------
def select(logits: torch.Tensor, mask: Optional[torch.Tensor] = None, *, hard: bool) -> torch.Tensor:
    """Track C's `select` op: `({Scalar}_C) -> Dist` -- identical to
    `nsm_ct.clause_reactor.ClauseReactor._collapse_weights`
    (clause_reactor.py:2624), generalized with an explicit candidate
    `mask` instead of that method's caller-side `has_cand` masking.

    Parameters
    ----------
    logits : Tensor
        `[..., C]`.
    mask : BoolTensor, optional
        `[..., C]`, `True`/nonzero = eligible candidate; ineligible
        candidates are masked to `-1e9` before softmax/argmax (matches
        `_collapse`'s own masking convention).
    hard : bool
        `False` (training) -> softmax (differentiable); `True` (eval) ->
        one-hot argmax.

    Returns
    -------
    Tensor
        `[..., C]`, a (possibly soft) distribution.
    """
    if mask is not None:
        logits = logits.masked_fill(~mask.bool(), -1e9)
    if not hard:
        return torch.softmax(logits, dim=-1)
    idx = logits.argmax(dim=-1)
    return F.one_hot(idx, num_classes=logits.shape[-1]).to(logits.dtype)


def abstain(
    margin: Union[torch.Tensor, float],
    dial: float = CAUTION,
    *,
    has_candidates: Union[torch.Tensor, bool] = True,
) -> Union[Optional[str], List[Optional[str]]]:
    """MIND_INTERFACE.md's sanctioned uncertainty forms 3 (low margin) and
    4 (an abstention answer), made into one deterministic gate --
    dev/OP_INVENTORY.md's "abstain (idk/MAYBE)" row, whose gap note reads
    "`caution`/margin never gates anything... OPEN-binding and margin-
    gated abstention are unimplemented". This is that gate.

    Parameters
    ----------
    margin : Tensor or float
        A collapse top1-top2 margin (`[B]` or scalar).
    dial : float, default `CAUTION`
        Below this margin, hold open with `"MAYBE"` rather than hard-bind.
    has_candidates : Tensor or bool
        Whether there was anything to decide among at all; `False` means
        no signal whatsoever -> `"idk"` regardless of `margin`.

    Returns
    -------
    str, None, or list of these
        Per element: `"idk"` (`has_candidates` False), `"MAYBE"`
        (`margin < dial`), or `None` (confident enough to hard-bind --
        the caller proceeds with the ordinary collapse result). Returns a
        single value for scalar inputs, a list of `B` values for batched
        `[B]` inputs (this feeds a discrete answer-atom choice, not
        further tensor math, so python objects -- not a tensor -- are the
        natural output type).
    """
    def _one(m: float, hc: bool) -> Optional[str]:
        if not hc:
            return "idk"
        if m < dial:
            return "MAYBE"
        return None

    if torch.is_tensor(margin) and margin.dim() > 0:
        b = margin.shape[0]
        hc = has_candidates
        if not torch.is_tensor(hc):
            hc = torch.full((b,), bool(hc))
        return [_one(float(margin[i]), bool(hc[i])) for i in range(b)]
    m = float(margin.item()) if torch.is_tensor(margin) else float(margin)
    hc = bool(has_candidates.item()) if torch.is_tensor(has_candidates) else bool(has_candidates)
    return _one(m, hc)


def compare(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Track C's `compare` op (dev/TRACK_C_DESIGN.md Sec.1.3): `(Vec,
    Vec) -> Scalar` (cosine), feeding `branch`'s condition -- exists
    today only at the FINAL output stage (`cosine(response_vec,
    option_vecs)`), "not currently used inside any resolver"; this is the
    reusable form. Equivalent to `similarity(a, b).cosine`.
    """
    return similarity(a, b).cosine


def branch(
    cond: torch.Tensor,
    then_vec: torch.Tensor,
    else_vec: torch.Tensor,
    *,
    hard: bool,
    threshold: float = BRANCH_THRESHOLD,
) -> torch.Tensor:
    """Track C's `branch` op: `(Scalar, Vec, Vec) -> Vec`, the reference
    conditional the executor will later route `compare`'s output through.

    Parameters
    ----------
    cond : Tensor
        `[B]`, typically `compare`'s cosine output.
    then_vec, else_vec : Tensor
        `[B, d]`.
    hard : bool
        `True` -> discrete select (`cond >= threshold` picks `then_vec`,
        else `else_vec`) -- an eval-time collapse, mirrors `select`'s
        hard mode. `False` -> differentiable interpolation: `cond`
        clamped to `[0, 1]` as the mixing weight, `w*then_vec +
        (1-w)*else_vec` -- gradients flow through `cond` itself, not just
        the two branches.
    threshold : float, default `BRANCH_THRESHOLD`
        Only used when `hard=True`.
    """
    if hard:
        take_then = (cond >= threshold).unsqueeze(-1)
        return torch.where(take_then, then_vec, else_vec)
    w = cond.clamp(0.0, 1.0).unsqueeze(-1)
    return w * then_vec + (1.0 - w) * else_vec


def halt(step: Union[torch.Tensor, int], budget: int = PATIENCE) -> torch.Tensor:
    """Track C's chain-length budget (dev/TRACK_C_DESIGN.md Sec.1.6): the
    `patience` dial (MIND_INTERFACE.md) made concrete as a hard cutoff --
    "at `K_max` an implicit SELECT+EMIT fires regardless of whether the
    router asked for it".

    Parameters
    ----------
    step : Tensor or int
        Current step count; `[B]` per-row counters or a single int.
    budget : int, default `PATIENCE`

    Returns
    -------
    BoolTensor
        `step >= budget`, same shape as `step`.
    """
    if torch.is_tensor(step):
        return step >= budget
    return torch.tensor(bool(step >= budget))


def emit(state_vec: torch.Tensor, mem_read: torch.Tensor) -> torch.Tensor:
    """Track C's `emit` op: `(Dist, {Vec}_C) -> Vec` in the algebra. The
    LEARNED weighted-sum aggregation and response-content head both live
    in the reactor (`nsm_ct.clause_reactor.ClauseReactor.response`,
    clause_reactor.py:2602, and the `(w.unsqueeze(-1)*RV).sum(dim=1)`
    aggregation, clause_reactor.py:3057) -- this is an identity/reference
    stand-in only, so the op algebra has a total function to chain
    through before that head is wired to it.

    Returns `mem_read` unchanged; `state_vec` is accepted, unused, purely
    for signature parity with the reactor's own
    `response(cat([state, mem_read]))` call.
    """
    del state_vec  # signature parity only -- see docstring
    return mem_read


# ---------------------------------------------------------------------------
# REGISTERS -- the executor's program trace format
# (dev/TRACK_C_DESIGN.md Sec.1.4's G./P. register model, generalized to
# one named [B, R, d] bank instead of separately-typed python attributes).
# ---------------------------------------------------------------------------
@dataclass
class RegisterFile:
    """A tiny register bank: `[B, R, d]` values plus a typed, named slot
    map, and a `trace` list recording every write as `(op_name, args,
    step)` -- **the trace supervision format for Track C**: a gold
    program is exactly a sequence of these triples, and this is the
    concrete container an executor's routing policy will learn to
    reproduce (dev/TRACK_C_DESIGN.md Sec.1.5: "the program is the
    sequence of (op, which registers) choices a router makes").

    Generalizes TRACK_C_DESIGN.md Sec.1.4's `G.*`/`P.*[i]` register model
    (global vs. per-candidate slots) into one uniform name -> row mapping;
    a caller wanting per-candidate registers names them `"P.mem_0"`,
    `"P.mem_1"`, ... (or keeps a python list of `RegisterFile`s, one per
    candidate slot -- either composes with the flat name/type map here).

    Attributes
    ----------
    values : Tensor
        `[B, R, d]`, one row per named register.
    slot_index : dict[str, int]
        Register name -> row index into `values`.
    slot_type : dict[str, str]
        Register name -> Track C type tag (`"Addr"`, `"Vec"`, `"Feat"`,
        `"Scalar"`, `"Dist"` -- dev/TRACK_C_DESIGN.md Sec.1.2). `Scalar`
        registers are stored broadcast across every one of the `d`
        columns (so the underlying tensor stays uniformly `[B, R, d]`);
        `read` collapses a `Scalar` register back to `[B]` via `.mean`
        (exact, since every column holds the same value by construction).
    trace : list of (str, tuple, int)
        Append-only; one entry per `write` call (`op_name`, the args it
        was called with, the step index), plus any manual `record` calls.
    """

    values: torch.Tensor
    slot_index: Dict[str, int]
    slot_type: Dict[str, str]
    trace: List[Tuple[str, tuple, int]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        batch: int,
        dim: int,
        slots: Dict[str, str],
        *,
        device: Optional[torch.device] = None,
    ) -> "RegisterFile":
        """Allocate a zero-initialized `[batch, len(slots), dim]` bank.
        `slots`: register name -> type tag, in the order rows are laid
        out (`dict` insertion order, Python 3.7+)."""
        names = list(slots.keys())
        values = torch.zeros(batch, len(names), dim, device=device)
        return cls(values=values, slot_index={n: i for i, n in enumerate(names)}, slot_type=dict(slots))

    def read(self, name: str) -> torch.Tensor:
        """`[B, d]` (or `[B]` for a `"Scalar"`-typed register)."""
        r = self.slot_index[name]
        v = self.values[:, r, :]
        if self.slot_type[name] == "Scalar":
            return v.mean(dim=-1)
        return v

    def write(
        self,
        name: str,
        value: torch.Tensor,
        *,
        step: int,
        op_name: str = "reg_write",
        args: tuple = (),
    ) -> None:
        """Write `value` into register `name` and append a trace entry
        `(op_name, (name,) + args, step)`. A `"Scalar"`-typed register
        accepts a `[B]` value (broadcast across all `d` columns); every
        other type expects `[B, d]`. Out-of-place on `self.values`
        (reassigns a cloned tensor, so an aliased earlier `read()` result
        is never mutated under the reader)."""
        r = self.slot_index[name]
        d = self.values.shape[-1]
        if self.slot_type[name] == "Scalar":
            value = value.reshape(value.shape[0], 1).expand(-1, d)
        new_values = self.values.clone()
        new_values[:, r, :] = value
        self.values = new_values
        self.trace.append((op_name, (name,) + tuple(args), step))

    def record(self, op_name: str, args: tuple, step: int) -> None:
        """Append a trace entry for an op that doesn't itself write a
        register (e.g. a `branch` decision consumed immediately, or a
        `halt` check) -- keeps the trace a complete program record
        without forcing every op through `write`."""
        self.trace.append((op_name, args, step))
