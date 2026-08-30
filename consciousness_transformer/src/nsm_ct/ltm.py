"""M59a: episodic LONG-TERM MEMORY (LTM) -- see CLAUDE.md's "LTM decisions"
and dev/LTM_DESIGN_BRIEF.md Sec.5 (locked design) for the decisions this
module implements. dev/OP_INVENTORY.md's op table names the four ops this
milestone builds: **recall** (additive LTM read), **link** (LTM identity,
via the existing resolver), **wind-down/consolidate** (a substate machine,
``scripts/_train_common.DocumentRunner``), and **promote** (tier-generic
gated copy, this module).

Locked design, restated as code contracts:

1. **Separate LTM tensor; additive reads; STM-only writes.** LTM is a
   second ``[B, d, d, d]`` order-3 memory, the SAME shape/algebra as STM's
   (:mod:`nsm_ct.entity_memory`), that persists across the PASSAGES of one
   document/session while STM keeps resetting per
   :meth:`nsm_ct.clause_reactor.ClauseReactor.forward` call (unchanged).
   Every READ (the step ``mem_read``, the post-collapse re-read,
   candidate-evidence reads via ``query_candidates``, the entity-axis
   inverse read) queries :func:`mem_total`'s ``memory + ltm`` view instead
   of ``memory`` alone; every WRITE still lands only in STM's ``memory``
   (:func:`nsm_ct.entity_memory.write`, unchanged). ``ltm=None`` (the
   default, and every batch/call before this milestone) makes
   :func:`mem_total` return ``memory`` unchanged -- byte-identical to
   pre-M59a :meth:`~nsm_ct.clause_reactor.ClauseReactor.forward`
   (regression-tested in ``tests/test_ltm.py``).
2. **recall is not a new op.** It is simply :func:`mem_total`'s additive
   combination, read by the SAME ``entity_memory.query``/``query_entity``
   einsums STM already uses -- see :func:`mem_total`'s own docstring for
   why this is exact (not an approximation) given both memories share one
   vector space by construction.
3. **promote is tier-generic.** :func:`promote` is the SAME function for
   STM->LTM (``dial_name="trust_ltm"``, the default gate
   ``record.trust >= dial``) and, later, LTM->Truth
   (``dial_name="trust_truth"``, a corroboration-count ``criterion``) --
   see its own docstring and ``tests/test_ltm.py``'s genericity test.
4. **wind-down / consolidate is a named substate machine**, not a
   per-clause gate: READING -> WIND_DOWN -> CONSOLIDATE, fired at the end
   of each passage. Built in ``scripts/_train_common.DocumentRunner``
   (kept out of this module -- the substate machine needs
   :class:`~nsm_ct.clause_reactor.ClauseBatch`/``ClauseReactor.forward``
   plumbing that belongs with the other training-loop machinery in
   ``_train_common.py``, not in the op-definitions module).
5. **link reuses the existing resolver contract.** LTM candidates join the
   SAME entity candidate set STM candidates already use, with one new
   per-candidate feature (``from_ltm``, 0/1) and a named dial
   (:data:`LINK_THRESHOLD`). :func:`link_decision` is the deterministic v1
   policy that turns a (post-collapse) probability distribution over that
   candidate set into "linked to candidate i" or "NEW" -- see its own
   docstring for exactly how "mint a new instance" is realized
   curriculum-side, not as a fourth answer option the resolver has to
   learn.

Dials (dev/MIND_INTERFACE.md invariant #6: dials are explicit named
scalars, never magic numbers): :data:`TRUST_LTM` (STM->LTM promotion gate),
:data:`LINK_THRESHOLD` (identity-linking collapse-vs-mint-new threshold),
:data:`LTM_DETACH` (whether ``DocumentRunner`` detaches the LTM tensor
between passages -- see its own docstring for why this bounds autograd).

-----------------------------------------------------------------------
Interface contract for the curriculum agent (M59b)
-----------------------------------------------------------------------
This module (:func:`promote`, :func:`link_decision`, :func:`mem_total`)
and ``scripts._train_common.DocumentRunner`` are built against the
following contract; M59b's curriculum generator must produce data that
satisfies it, not the other way around -- nothing here builds a
curriculum, this is the shape the curriculum has to hand back.

- A **document** is an ORDERED list of episodes (whatever unit M59b's
  generator emits -- today's ``episode.Episode``, extended the same way
  every prior milestone extended it) that share one ``meta["doc_id"]``
  value, each carrying its position via ``meta["passage_index"]``
  (0-based, contiguous, no gaps). A document's episodes are its
  **passages**, read in ``passage_index`` order.
- ``DocumentRunner.run_document`` does not itself discover documents from
  a flat episode list -- the CALLER groups episodes by ``doc_id`` (e.g.
  sort by ``(doc_id, passage_index)`` then group) and hands
  ``run_document`` one already-ordered list of per-passage
  :class:`~nsm_ct.clause_reactor.ClauseBatch`\\ es for a SINGLE document.
- Every instance/entity atom mentioned ANYWHERE in a document -- passage 1
  or passage 7 -- must be minted from the SAME
  :class:`~nsm_ct.instances.InstanceRegistry`, constructed ONCE per
  document and threaded through every passage's batch-build call. This is
  what lets a passage-2 mention's candidate set legally include a
  passage-1 instance id: both ids came from one mint stream, so
  ``registry.lookup(id)`` resolves identically no matter which passage
  looks it up.
- A candidate set for a CROSS-passage mention (one that could refer to an
  instance introduced in an earlier, already-consolidated passage) carries
  one ``from_ltm`` flag PER CANDIDATE (0/1) alongside the existing entity-
  candidate fields -- see
  :attr:`~nsm_ct.membrane.EntityCandidateSet.from_ltm` and
  :attr:`~nsm_ct.clause_reactor.ClauseBatch.cand_from_ltm`. 1 marks a
  candidate whose only source is a prior passage's consolidated LTM
  (unseen this passage); 0 marks an ordinary same-passage STM candidate.
  Both kinds sit in the SAME candidate set/tensor (mirrors M57b's "one
  field, both kinds share it" discipline) -- ``from_ltm`` is a SCORER
  FEATURE, not a filter; the resolver still chooses among all of them
  together, through the SAME collapse machinery every other candidate kind
  uses.
- The candidate set MAY ALSO include one freshly-minted "NEW" candidate
  atom (``registry.mint(...)`` called at CURRICULUM-BUILD time, before the
  resolver ever runs) meaning "this mention introduces a referent never
  seen before, in this passage or any prior one." There is no separate
  NEW/OPEN sentinel value collapse itself can emit -- "start a new
  instance" is realized ENTIRELY as the resolver picking that ordinary
  candidate out of the set. :func:`link_decision`'s ``NEW`` (``-1``)
  return value is a CONVENIENCE for the runner/curriculum -- either "no
  candidate cleared ``link_threshold``" (an abstention) or, when the
  caller has arranged for the minted "NEW" candidate to occupy a known
  index, a shorthand for detecting that outcome -- never a fourth answer
  option the resolver has to learn to emit on its own.
- Every cross-passage candidate set carries a gold link index through the
  SAME ``EntityCandidateSet.gold_index`` / ``ClauseBatch.cand_gold``
  contract every other candidate kind already uses, so the aux resolver
  loss (``cand_gold``-supervised cross-entropy -- the SAME loss every
  M53b/M57 battery already trains with) covers cross-episode identity
  linking with zero new loss-function code.
- ``DocumentRunner.run_document`` returns ``(reports, total_loss)``:
  ``reports`` is one ``dict`` PER PASSAGE, in passage order, with keys --
  ``"passage_index"`` (int), ``"out"`` (the passage's raw
  ``ClauseReactor.forward`` output dict, built with
  ``return_write_trace=True, return_memory=True``), ``"ltm"`` (the
  ``[B, d, d, d]`` LTM tensor AFTER this passage's CONSOLIDATE step -- what
  the NEXT passage's forward call reads), ``"n_records"`` (provenance
  records appended from this passage's STM writes), ``"n_promoted"`` (of
  those, how many cleared ``trust_ltm`` and were copied into LTM),
  ``"events"`` (the substate trace, e.g.
  ``["reading", "wind_down", "consolidate"]``), and ``"loss"`` (this
  passage's own loss tensor when a ``loss_fn`` was supplied, else
  ``None``). ``total_loss`` is the SUM of every passage's own loss (a
  single tensor a training script calls ``.backward()`` on once), or
  ``None`` when no ``loss_fn`` was given (the eval-only path). A script
  computing per-passage answer accuracy reads
  ``report["out"]["answer_logits"]`` against that passage's own
  ``batch.answer``, exactly like every non-document training script
  already does with a plain ``model(batch)`` call.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch

from . import entity_memory as em
from .instances import InstanceRegistry, ProvenanceLog, ProvenanceRecord

__all__ = [
    "TRUST_LTM",
    "LINK_THRESHOLD",
    "LTM_DETACH",
    "NEW",
    "mem_total",
    "promote",
    "link_decision",
]


# ---------------------------------------------------------------------------
# Dials (MIND_INTERFACE.md invariant #6: named scalars, never magic numbers).
# ---------------------------------------------------------------------------
TRUST_LTM: float = 0.5     # STM->LTM promotion gate (promote's default criterion)
LINK_THRESHOLD: float = 0.5  # identity-linking collapse-vs-mint-new threshold (link_decision)
LTM_DETACH: bool = True    # DocumentRunner: detach ltm between passages (see DocumentRunner docstring)

NEW: int = -1  # link_decision sentinel: no candidate cleared LINK_THRESHOLD -- mint a fresh instance
               # (curriculum-side; see this module's interface-contract docstring above)


# ---------------------------------------------------------------------------
# recall -- not a new op, just the additive read (design decision #1).
# ---------------------------------------------------------------------------
def mem_total(memory: torch.Tensor, ltm: Optional[torch.Tensor]) -> torch.Tensor:
    """The **recall** op (dev/OP_INVENTORY.md's "recall (additive LTM
    read)" row): every read this milestone touches -- the step
    ``mem_read``, the post-collapse re-read, candidate-evidence reads via
    :func:`nsm_ct.resolver.query_candidates`, the entity-axis inverse read
    -- queries this ADDITIVE combination of STM (``memory``) and LTM
    (``ltm``), never LTM alone and never STM alone once an ``ltm`` tensor
    is threaded into a call. There is no separate "recall" function beyond
    this: recall IS :func:`nsm_ct.entity_memory.query`/``query_entity`` run
    against ``mem_total``'s result, exactly the same einsum STM-only code
    already used -- summing the two memories before the read is EXACT (not
    an approximation), because both are written with the SAME bilinear
    :func:`nsm_ct.entity_memory.write` form over the SAME instance atoms /
    attribute-relation fillers, so ``query(A + B, e, r) == query(A, e, r) +
    query(B, e, r)`` by the einsum's own bilinearity.

    ``ltm=None`` (every pre-M59a call, and every LTM-free forward call
    after it) returns ``memory`` unchanged -- byte-identical.

    Callers (:meth:`nsm_ct.clause_reactor.ClauseReactor.forward`) compute
    this ONCE per step and reuse the result for every read that step
    (``mem_read``, the post-collapse re-read, ``_collapse``'s candidate/
    inverse reads) rather than re-adding ``memory + ltm`` at each read
    site -- the "do not double-allocate per step" requirement
    (LTM_DESIGN_BRIEF Sec.5.1). ``memory`` itself changes every step
    (writes), so the combined view can't be cached ACROSS steps, but it is
    allocated exactly once PER step, not once per read within a step.
    """
    return memory if ltm is None else memory + ltm


# ---------------------------------------------------------------------------
# promote -- tier-generic gated copy (design decision #3/#4 in the brief).
# ---------------------------------------------------------------------------
def _as_tensor(x, dtype: torch.dtype) -> torch.Tensor:
    """Accept a torch tensor or a numpy array (codec vectors are numpy) and
    return a detached ``dtype`` torch tensor -- the same
    ``torch.from_numpy`` seam :mod:`nsm_ct.instances` uses for the same
    purpose (duplicated locally, not imported, so this module doesn't
    reach into ``instances.py``'s private helpers)."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dtype)


def promote(
    source: torch.Tensor,
    target: torch.Tensor,
    registry: InstanceRegistry,
    log: ProvenanceLog,
    *,
    records: Sequence[ProvenanceRecord],
    dial: float,
    dial_name: str,
    criterion: Optional[Callable[[ProvenanceRecord], bool]] = None,
    codec,
    timestamp: float,
) -> Tuple[torch.Tensor, int]:
    """Tier-generic gated copy: ``(Mem_N, ProvenanceLog, Scalar) -> Mem_{N+1}``
    (dev/OP_INVENTORY.md's **promote** row). The SAME function serves
    STM->LTM (``dial_name="trust_ltm"``, ``dial=`` :data:`TRUST_LTM`, the
    default ``criterion``) and, later, LTM->Truth (``dial_name=
    "trust_truth"``, a corroboration-count ``criterion``) -- see
    ``tests/test_ltm.py``'s genericity test for a worked example of the
    latter with a counting criterion.

    ``source``/``target`` are UNBATCHED ``[d, d, d]`` memories (one
    document's own STM / LTM slice -- callers with a batched ``[B, d, d,
    d]`` LTM tensor, e.g. ``scripts._train_common.DocumentRunner``, loop
    this over the batch dimension, matching
    :mod:`nsm_ct.instances`'s own "unbatched by design" stance for
    discourse-level bookkeeping).

    For each record in ``records`` (this passage's OWN new provenance --
    never the whole historical log), DEDUPLICATED by ``(instance_id,
    relation)`` with LAST WRITE WINS (a passage that overwrites its own
    earlier statement, e.g. an M57b write-back, only promotes the final
    belief), whose ``criterion(record)`` is true (default:
    ``record.trust >= dial``, MIND_INTERFACE.md invariant #6's named-dial
    gate): read the value back out of ``source`` at
    (``registry.lookup(record.instance_id)``, ``codec.filler_vec(record.
    relation)`` -- ``record.relation`` already carries its own ``"attr:"``
    prefix, the SAME convention :func:`nsm_ct.instances.write_attribute`
    writes it with, so no second prefix is added here), and gated-overwrite
    (``gate=1`` -- "the newest passage's belief wins", per the locked
    design) that value into ``target`` at the SAME (entity, relation) slot.
    Appends one :class:`~nsm_ct.instances.ProvenanceRecord` to ``log`` per
    promoted fact, ``source=f"promote:{dial_name}"`` (the tier tag
    MIND_INTERFACE.md invariant #4's audit trail needs), everything else
    (instance/relation/value_label/trust/step/surface/candidate_ids)
    copied from the original record -- promotion doesn't change WHAT was
    believed or how strongly, only WHERE it now also lives.

    Out-of-place (mirrors :func:`nsm_ct.entity_memory.write`/
    :func:`nsm_ct.instances.write_attribute`): returns a NEW ``[d, d, d]``
    tensor, ``target`` itself is untouched. Returns ``(new_target,
    n_promoted)`` -- the number of records actually copied (after dedup
    and the criterion gate), for a caller/report to log.
    """
    if criterion is None:
        gate_dial = dial

        def criterion(record: ProvenanceRecord, _dial: float = gate_dial) -> bool:
            return record.trust >= _dial

    # Deduplicate by (instance_id, relation): iterate in order, last write
    # wins. A plain dict overwrite achieves this -- Python dicts keep a
    # key's FIRST insertion position but its LATEST assigned value, which
    # is exactly "last write wins" (write ORDER among the deduplicated
    # keys doesn't matter below: every promoted write lands at a distinct
    # (entity, relation) slot, so the writes commute).
    deduped: dict = {}
    for record in records:
        deduped[(record.instance_id, record.relation)] = record

    new_target = target
    n_promoted = 0
    for (instance_id, relation), record in deduped.items():
        if not criterion(record):
            continue
        entity = registry.lookup(instance_id).to(source.dtype)
        relation_vec = _as_tensor(codec.filler_vec(relation), source.dtype)
        value = em.query(
            source.unsqueeze(0), entity.unsqueeze(0), relation_vec.unsqueeze(0)
        ).squeeze(0)
        gate = torch.ones(1, dtype=new_target.dtype)
        new_target = em.write(
            new_target.unsqueeze(0), entity.unsqueeze(0), relation_vec.unsqueeze(0),
            value.unsqueeze(0), gate,
        ).squeeze(0)
        log.append(ProvenanceRecord(
            instance_id=instance_id,
            relation=relation,
            value_label=record.value_label,
            source=f"promote:{dial_name}",
            language=record.language,
            timestamp=timestamp,
            trust=record.trust,
            step=record.step,
            surface=record.surface,
            candidate_ids=record.candidate_ids,
        ))
        n_promoted += 1
    return new_target, n_promoted


# ---------------------------------------------------------------------------
# link -- deterministic v1 collapse-vs-mint-new decision (design decision #5).
# ---------------------------------------------------------------------------
def link_decision(probs: torch.Tensor, threshold: float = LINK_THRESHOLD) -> torch.Tensor:
    """Deterministic v1 policy for **link** (dev/OP_INVENTORY.md's "link
    (LTM identity)" row): given a probability distribution ``probs`` over
    an entity candidate set (the SAME ``[..., C]`` shape/space
    :meth:`nsm_ct.clause_reactor.ClauseReactor._collapse`'s entity branch
    already produces -- pass it ``torch.softmax(logits.masked_fill(mask<=0,
    -1e9), dim=-1)``, or the eval-mode hard-collapse weights ``w`` directly,
    since argmax of either agrees), return the argmax candidate index where
    the max probability clears ``threshold``, else :data:`NEW` (``-1``) --
    "mint a new instance" (see this module's interface-contract docstring
    for how the curriculum realizes that outcome as an ordinary candidate,
    not a fourth answer option).

    Works over any leading batch shape: ``probs`` ``[C]`` -> a 0-d
    ``LongTensor``; ``probs`` ``[B, C]`` -> ``[B]``. A caller may call this
    directly on the resolver's own collapse output to decide, per (row,
    step), whether the identity-linking mention just bound to an existing
    LTM candidate or should be treated as introducing a new one -- this is
    a POST-HOC decision (:func:`nsm_ct.clause_reactor.ClauseReactor.
    _collapse` itself is unaware of ``link_threshold``; it always collapses
    softly/hard-argmax like every other candidate kind), used by the
    runner/curriculum to report/gate on, not by the collapse arithmetic
    itself.
    """
    max_val, max_idx = probs.max(dim=-1)
    new_sentinel = torch.full_like(max_idx, NEW)
    return torch.where(max_val >= threshold, max_idx, new_sentinel)
