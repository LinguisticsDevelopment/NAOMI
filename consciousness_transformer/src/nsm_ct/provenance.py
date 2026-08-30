"""M57d: wiring PROVENANCE into the live reactor's writes.

CLAUDE.md's M57 memory-schema decision: "Provenance (source, language,
timestamp, trust) is a membrane-side append-only log, one record per gated
write -- it cannot live in the tensor (superposition destroys the audit
trail)." :mod:`nsm_ct.instances` already builds that log
(:class:`~nsm_ct.instances.ProvenanceRecord` / :class:`~nsm_ct.instances.
ProvenanceLog`) for the *offline* attribute-fact API
(:func:`nsm_ct.instances.write_attribute`); this module is the SAME log
wired onto the reactor's OWN live writes (:meth:`nsm_ct.clause_reactor.
ClauseReactor.forward`'s ``em.write`` call) instead, making
dev/MIND_INTERFACE.md invariant #4 ("every memory write is gated, local, and
auditable") true of a trained model's actual training-time/eval-time
behavior, not just of a hand-driven demonstration.

Two pieces the reactor now exposes for exactly this purpose (both inert by
default -- see their own docstrings for the byte-identity guarantee):

- :meth:`nsm_ct.clause_reactor.ClauseReactor.forward`'s ``return_write_trace``
  flag -- ``out["_write_trace"]``, a ``{gate, overwrite, neg, redirected,
  resolved_index}`` dict of ``[B, T]`` tensors describing every step's
  ACTUAL write (the same ``gate``/``overwrite``/``neg`` values fed to
  ``em.write`` itself, plus what the entity-branch resolver collapsed to).
- :class:`nsm_ct.clause_reactor.ClauseBatch`'s ``step_meta`` field -- a
  Python-side (never a tensor) ``[B][T]`` list of per-statement-step label
  dicts (``sentence_index``, ``surface``, ``relation_label``,
  ``value_label``, ``entity_label``, ``candidate_ids``, ``referring_device``,
  ``episode_index``), populated by :func:`nsm_ct.clause_reactor.
  build_clause_batch` for writeback/instance/rich episodes.

:func:`record_writes` zips the two together into
:class:`~nsm_ct.instances.ProvenanceRecord`\\ s; :func:`explain` and
:func:`overwrites_for` are the read side of the same audit trail.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .clause_reactor import ClauseBatch
from .instances import ProvenanceLog, ProvenanceRecord

__all__ = ["record_writes", "explain", "overwrites_for"]


def record_writes(batch: ClauseBatch, out: Dict[str, torch.Tensor], log: ProvenanceLog, *,
                   source: str, language: str = "en",
                   trust_threshold: float = 0.0, timestamp_base: float = 0.0) -> int:
    """Append one :class:`~nsm_ct.instances.ProvenanceRecord` to ``log`` for
    every (row, step) that both (a) has ``ClauseBatch.step_meta`` -- a
    labeled statement/write step (see that field's docstring: ``None`` at
    every question step and at every episode kind this milestone doesn't
    label) -- and (b) actually wrote with ``gate > trust_threshold`` (the
    trust gate MIND_INTERFACE.md invariant #4 names: a write nobody trusted
    enough to commit leaves no audit trail either).

    ``out`` must be a :meth:`~nsm_ct.clause_reactor.ClauseReactor.forward`
    call's return value built with ``return_write_trace=True`` (i.e. must
    carry ``out["_write_trace"]`` -- see that flag's own docstring); a
    ``batch`` with no ``step_meta`` at all, or an ``out`` with no write
    trace, records nothing and returns 0 rather than raising -- the same
    "absent optional data is a no-op" discipline :mod:`nsm_ct.clause_reactor`
    itself uses throughout.

    The instance a redirected write is recorded AGAINST is the RESOLVED
    candidate (``step_meta["candidate_ids"][resolved_index]``) -- the entity
    the write actually landed on after collapse, per
    :meth:`~nsm_ct.clause_reactor.ClauseReactor._collapse`'s address-redirect
    arithmetic -- never the step's own (garbage) placeholder address. A
    directly-addressed step (no candidate set at all) is recorded against
    its statically-known ``entity_label`` instead. Returns the number of
    records appended.
    """
    if batch.step_meta is None or "_write_trace" not in out:
        return 0
    trace = out["_write_trace"]
    gate, redirected, resolved_index = trace["gate"], trace["redirected"], trace["resolved_index"]
    b, T = gate.shape
    count = 0
    for i in range(b):
        row_meta = batch.step_meta[i]
        for t in range(min(T, len(row_meta))):
            meta = row_meta[t]
            if meta is None:
                continue
            g = float(gate[i, t])
            if g <= trust_threshold:
                continue
            if bool(redirected[i, t]):
                idx = int(resolved_index[i, t])
                candidate_ids = meta.get("candidate_ids") or []
                if idx < 0 or idx >= len(candidate_ids):
                    continue   # nothing resolvable this step -- no honest address to record
                instance_id = candidate_ids[idx]
            else:
                instance_id = meta.get("entity_label")
                if instance_id is None:
                    continue   # neither a resolved nor a statically-known address
            log.append(ProvenanceRecord(
                instance_id=instance_id,
                relation=meta["relation_label"],
                value_label=meta["value_label"],
                source=f"{source}:ep{meta['episode_index']}:s{meta['sentence_index']}",
                language=language,
                timestamp=timestamp_base + t,
                trust=g,
                step=t,
                surface=meta.get("surface"),
                candidate_ids=tuple(meta["candidate_ids"]) if meta.get("candidate_ids") else None,
            ))
            count += 1
    return count


def explain(log: ProvenanceLog, instance_id: str) -> str:
    """A human-readable audit trail for ``instance_id``, one line per
    :class:`~nsm_ct.instances.ProvenanceRecord` touching it, in write order
    -- e.g. ``"[t=5 trust=0.98] attr:trait := tall  (from 'she is tall .',
    resolved from [inst:mary#1, inst:mary#2])"``. The parenthetical is
    omitted piecewise when a record carries no ``surface``/``candidate_ids``
    (a directly-addressed write, or a record predating M57d's two optional
    :class:`~nsm_ct.instances.ProvenanceRecord` fields). Returns a one-line
    placeholder, never an empty string or an exception, when ``instance_id``
    has no records at all.
    """
    records = log.records_for(instance_id)
    if not records:
        return f"(no provenance records for {instance_id})"
    lines = []
    for r in records:
        line = f"[t={r.step} trust={r.trust:.2f}] {r.relation} := {r.value_label}"
        extra = []
        if r.surface:
            extra.append(f"from '{r.surface}'")
        if r.candidate_ids:
            extra.append(f"resolved from [{', '.join(r.candidate_ids)}]")
        if extra:
            line += "  (" + ", ".join(extra) + ")"
        lines.append(line)
    return "\n".join(lines)


def overwrites_for(log: ProvenanceLog, instance_id: str, relation: str) -> List[str]:
    """The sequence of values written to ``instance_id``'s ``relation`` slot,
    in time order -- the audit answer to "why does the system believe X is
    Y" (the LAST entry is the currently-believed value; every earlier entry
    is what it overwrote). Empty when nothing was ever written to that
    exact (``instance_id``, ``relation``) pair."""
    return [r.value_label for r in log.records_for(instance_id) if r.relation == relation]
