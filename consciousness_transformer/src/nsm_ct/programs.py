"""EXECUTOR PHASE 1: the six gold programs as DATA (dev/EXECUTOR_DESIGN.md
Sec.1.1's frozen v0 register table), plus the mapping from a real
:class:`~nsm_ct.clause_reactor.ClauseBatch` (row, step) to which of these six
families the EXISTING pipeline implicitly runs there.

**"The pipeline as one fixed program" claim, made concrete.** Today's
:meth:`nsm_ct.clause_reactor.ClauseReactor.forward` never branches on an
explicit "which program" variable -- it runs ONE Python code path every
clause step, and gets different behavior per (row, step) purely because its
own tensors are masked/``torch.where``-selected by structural flags already
sitting on the batch: ``batch.cand_mask`` (is there a candidate set at all),
``batch.cand_addr_mask`` (does collapse redirect the WRITE ADDRESS instead of
the value), ``batch.inverse_mask`` (is this an entity-axis inverse read),
``batch.cand_from_ltm`` (does the candidate set include cross-passage LTM
candidates), and ``batch.is_q`` (statement vs question -- EPILOGUE-W vs
EPILOGUE-Q). :func:`program_for_step` reads exactly those flags, in exactly
the priority order :meth:`~nsm_ct.clause_reactor.ClauseReactor._collapse`
itself applies them (mutually exclusive per row: an inverse-query row is
never also an addr-redirect row), and returns which of the six families that
(row, step) is -- this function IS the "pipeline as one fixed program" claim:
if it ever needs a seventh branch to stay exhaustive over a real batch, the
claim is false and dev/EXECUTOR_DESIGN.md Sec.1.1's register table is
incomplete.

Two caveats already known from reading ``clause_reactor.py`` (see
``executor.py``'s module docstring "Phase 1 findings" section for the full
discussion, not repeated here):

- ``CAND_FOR`` (programs 4/6, step 1) is a NO-OP/pass-through in this
  Executor: ``dev/OP_INVENTORY.md`` marks ``candidates_for``/``inverse_query``
  as "offline-tested -- not yet the live batch-build path" -- every
  candidate set a real batch carries today was pre-enumerated by the
  CURRICULUM generator, not searched for live off ``V_desc`` at run time.
  ``program_for_step`` distinguishes "definite-desc read" (family 4) from
  "write-back addr-redirect" (family 3) using ``is_q`` alone (both are
  ``addr_redirect=True`` candidate sets in every existing generator; the
  design table's CAND_FOR-vs-direct-QUERY distinction at step 1 does not
  exist as two different code paths in ``build_clause_batch`` today).
- ``batch.cand_from_ltm`` -- EXECUTOR PHASE 2 update: M59b has since landed
  (``clause_reactor._document_steps``'s final-passage mention step sets
  ``EntityCandidateSet.from_ltm=[1, 0]``, and ``build_clause_batch`` copies
  it into ``ClauseBatch.cand_from_ltm`` whenever any candidate set in the
  batch carries one -- ``clause_reactor.py:2441-2448``) -- so
  ``program_for_step``'s existing ``from_ltm_row`` branch (below) already
  classifies a real :class:`~nsm_ct.curriculum2.DocumentGenerator` FINAL
  passage's mention step as ``"recall_link"`` with NO code change needed
  here, PROVIDED the batch is built correctly: each document's passages
  must share ONE :class:`~nsm_ct.instances.InstanceRegistry` across
  ``build_clause_batch`` calls (passage 0 first, mints the referent; the
  final passage's own ``build_clause_batch`` call, same registry, mints the
  "NEW" candidate second) -- exactly what a real multi-passage run (M59a's
  :class:`~nsm_ct.checkpoint`-adjacent ``scripts._train_common.
  DocumentRunner`) already does passage-by-passage. Passage 0 alone (no
  registry continuity, no mention step) never classifies as
  ``"recall_link"`` -- it has no candidate set at all, hence `Phase 1`'s
  own single-passage-0 test battery never saw this family (its own note,
  ``tests/test_executor_phase1.py``'s ``_document_single_passage_batch``).
- Similarly, :func:`~nsm_ct.curriculum2.PronounCurriculumGenerator`
  episodes classify as ``"pronoun_value_redirect"`` out of the box IFF
  ``build_clause_batch`` is given a REAL parser (``_pronoun_context_step``
  gates on ``hasattr(parser, "_parse_graph")`` -- with ``parser=None`` the
  pronoun context sentence silently falls through to the ordinary
  ``_context_steps`` path with NO candidate set at all, and every step of
  that episode classifies as ``"plain_fact"`` instead, a SILENT trap for a
  corpus builder that reuses Phase 1's ``parser=None`` convenience). See
  ``scripts/train_executor.py``'s corpus builder for the
  ``ParserInputEncoder(SimpleTokenizer.build(...))`` construction this
  requires (mirrors ``scripts/train_resolver.py``'s own pattern).

Both of the above were verified empirically (not just read off the source)
before this note was written: given a correctly-built batch, EVERY ONE of
:data:`FAMILY_NAMES`' six families is produced by :func:`program_for_step`
UNCHANGED -- Executor Phase 2 needed no new structural-flag logic here, only
this documentation and :func:`family_of_step` below (the name the LOFO
splitter keys off).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

__all__ = [
    "Step",
    "PLAIN_FACT",
    "PRONOUN_VALUE_REDIRECT",
    "WRITEBACK_ADDR_REDIRECT",
    "DEFINITE_DESC_READ",
    "INVERSE_QUERY",
    "RECALL_LINK",
    "GOLD_PROGRAMS",
    "FAMILY_NAMES",
    "program_for_step",
    "program_for_step_batch",
    "family_of_step",
]


@dataclass(frozen=True)
class Step:
    """One row of dev/EXECUTOR_DESIGN.md Sec.1.1's register table: an op
    name, the register names it READS (``"state"`` is the clause-level GRU
    state -- not a Sec.1.1 register, called out there as such, kept as a
    literal string here rather than a fake register so a reader can tell
    the two apart at a glance), the ONE register it WRITES (``None`` for an
    op that only has a side effect -- ``HALT``), and the Track C type
    signature string, purely documentary (matches the table's own "type"
    column verbatim, e.g. ``"Addr,Addr->Vec"``)."""

    op: str
    reads: Tuple[str, ...]
    writes: Optional[str]
    type_sig: str


def _epilogue(x: str, *, write: bool) -> List[Step]:
    """dev/EXECUTOR_DESIGN.md Sec.1.1's ``EPILOGUE-W[X]``/``EPILOGUE-Q[X]``
    macro, expanded to its primitive ops. ``write=True`` -> EPILOGUE-W (ends
    in a write, ``TICK -> GATE/OVERWRITE/NEGATE -> WRITE -> RESPOND/RESPONSE
    -> HALT``); ``write=False`` -> EPILOGUE-Q (the same minus
    GATE/OVERWRITE/NEGATE/WRITE, Sec.1.1's own words: "same minus
    GATE/OVERWRITE/NEGATE and WRITE"). Sec.1.3 notes RESPOND/RESPONSE are
    "outside the op loop, scored once per episode" -- kept here as part of
    the flat per-clause program list anyway (this table is a per-CLAUSE
    trace, not the op-loop's own K_max accounting; see ``executor.py``'s
    ``Executor.program_length`` for the op-loop-only count K_max is
    actually meant to bound).
    """
    steps = [Step("TICK", (x, "A_r", "V_v", "V_read", "state"), "state", "GRU")]
    if write:
        steps += [
            Step("GATE", ("state",), "S_gate", "Scalar"),
            Step("OVERWRITE", ("state",), "S_owr", "Scalar"),
            Step("NEGATE", ("state", "V_v"), "S_neg", "Scalar"),
            Step("WRITE", ("M_mem", x, "A_r", "V_v", "S_gate", "S_owr", "S_neg"), "M_mem", "Mem"),
        ]
    steps += [
        Step("RESPOND", ("state",), "S_respond", "Scalar"),
        Step("RESPONSE", ("state", "V_read"), "V_answer", "Vec"),
        Step("HALT", (), None, "control"),
    ]
    return steps


# ---------------------------------------------------------------------------
# The six gold programs, transcribed verbatim off dev/EXECUTOR_DESIGN.md
# Sec.1.1's table (T1-T3, the Phase-3 transfer tasks, are explicitly OUT of
# Phase 1 scope -- "curricula not authored until Phase 3, after M59b").
# ---------------------------------------------------------------------------
PLAIN_FACT: List[Step] = [
    Step("QUERY", ("A_e", "A_r"), "V_read", "Addr,Addr->Vec"),
    *_epilogue("A_e", write=True),
]
"""Program 1 -- "plain fact": a direct-addressed statement/question, no
collapse at all (no resolver, or no candidate set this step)."""

PRONOUN_VALUE_REDIRECT: List[Step] = [
    Step("QUERY", ("A_e", "A_r"), "V_read", "Addr,Addr->Vec"),
    Step("QUERY_CAND", ("P.addr", "A_r"), "P.mem", "per-i->Vec"),
    Step("SCORE", ("P.addr", "P.feat", "P.prior", "P.mem", "V_ev"), "P.score", "per-i->Scalar"),
    Step("SELECT", ("P.score",), "D_w", "->Dist"),
    Step("EMIT", ("D_w", "P.mem"), "V_v", "->Vec"),
    *_epilogue("A_e", write=True),
]
"""Program 2 -- pronoun VALUE-redirect ("she found the ball ."): collapse
resolves WHICH candidate's memory readout becomes the stated value; the
WRITE address is untouched (``A_e``, the pronoun's own placeholder slot)."""

WRITEBACK_ADDR_REDIRECT: List[Step] = [
    Step("QUERY", ("A_e", "A_r"), "V_read", "Addr,Addr->Vec"),
    Step("QUERY_CAND", ("P.addr", "A_r"), "P.mem", "per-i->Vec"),
    Step("SCORE", ("P.addr", "P.prior", "P.mem", "V_ev"), "P.score", "per-i->Scalar"),
    Step("SELECT", ("P.score",), "D_w", "->Dist"),
    Step("EMIT", ("D_w", "P.addr"), "A_w", "->Addr"),
    Step("REREAD", ("A_w", "A_r"), "V_read", "Addr->Vec"),
    *_epilogue("A_w", write=True),
]
"""Program 3 -- write-back ADDRESS-redirect ("she is tall ."): collapse
resolves WHICH candidate's own ATOM becomes the write address; the write
lands on the resolved node, not the pronoun's placeholder."""

DEFINITE_DESC_READ: List[Step] = [
    Step("CAND_FOR", ("M_mem", "V_desc"), "P.addr", "per-i (no-op/pass-through in Phase 1, see module docstring)"),
    Step("QUERY_CAND", ("P.addr", "A_r"), "P.mem", "per-i->Vec"),
    Step("INTERACT", ("P.mem", "V_ev"), "P.score_extra", "per-i->Scalar"),
    Step("SCORE", ("P.addr", "P.prior", "P.mem", "P.score_extra", "state"), "P.score", "per-i->Scalar"),
    Step("SELECT", ("P.score",), "D_w", "->Dist"),
    Step("EMIT", ("D_w", "P.addr"), "A_w", "->Addr"),
    Step("REREAD", ("A_w", "A_r"), "V_read", "Addr->Vec"),
    *_epilogue("A_w", write=False),
]
"""Program 4 -- definite-description READ ("what is the doctor's place ?"),
a QUESTION: same address-redirect collapse as program 3, but ends in
EPILOGUE-Q (no write -- a question asserts nothing new)."""

INVERSE_QUERY: List[Step] = [
    Step("QUERY_ENTITY", ("A_r", "V_v"), "V_read", "Addr,Vec->Vec"),
    *_epilogue("A_e", write=False),
]
"""Program 5 -- inverse query ("who is tall ?"), a QUESTION: no candidate
set at all -- the entity axis is unbound directly from (relation, value)."""

RECALL_LINK: List[Step] = [
    Step("CAND_FOR", ("mem_total(M_mem,M_ltm)", "V_desc"), "P.addr",
         "per-i (no-op/pass-through in Phase 1, see module docstring)"),
    Step("QUERY_CAND", ("P.addr", "A_r"), "P.mem", "per-i->Vec"),
    Step("INTERACT", ("P.mem", "V_ev"), "P.score_extra", "per-i->Scalar"),
    Step("SCORE", ("P.addr", "P.prior", "P.mem", "P.score_extra", "P.from_ltm", "state"), "P.score",
         "link, per-i->Scalar"),
    Step("SELECT", ("P.score",), "D_w", "below LINK_THRESHOLD -> minted NEW, never a 4th atom"),
    Step("EMIT", ("D_w", "P.addr"), "A_w", "->Addr"),
    Step("REREAD", ("A_w", "A_r"), "V_read", "Addr->Vec"),
    *_epilogue("A_w", write=False),
]
"""Program 6 -- cross-passage recall+link (LTM, M59a), a QUESTION: the SAME
shape as program 4, plus the ``from_ltm`` scorer column and
``LINK_THRESHOLD``'s collapse-vs-mint-new semantics at SELECT. Not yet
exercised by any existing generator (M59b's job) -- see module docstring."""

GOLD_PROGRAMS = {
    "plain_fact": PLAIN_FACT,
    "pronoun_value_redirect": PRONOUN_VALUE_REDIRECT,
    "writeback_addr_redirect": WRITEBACK_ADDR_REDIRECT,
    "definite_desc_read": DEFINITE_DESC_READ,
    "inverse_query": INVERSE_QUERY,
    "recall_link": RECALL_LINK,
}
FAMILY_NAMES: Tuple[str, ...] = tuple(GOLD_PROGRAMS.keys())


# ---------------------------------------------------------------------------
# program_for_step -- the (row, step) -> family classifier, mechanically
# off the SAME structural flags ClauseReactor._collapse itself branches on,
# in the SAME priority order (mutually exclusive per row/step by
# construction -- clause_reactor.py's own candidate-kind sets never overlap
# on one row/step, verified by dev/EXECUTOR_DESIGN.md Sec.1.4's "candidate
# families are mutually exclusive per row/step" claim).
# ---------------------------------------------------------------------------
def program_for_step(batch, t: int) -> List[str]:
    """The gold family name per ROW at clause step ``t``, length ``B``.
    Exhaustive by construction: every row falls through to ``"plain_fact"``
    when none of the special-case flags fire (this is Phase 1's `_1_ op
    program`, not an error case) -- see ``tests/test_executor_phase1.py``'s
    exhaustiveness test, which asserts this never raises/returns ``None``
    over every real (row, step) of every generator this milestone tests.
    """
    b = batch.entity.shape[0]

    has_cand = None
    if batch.cand_mask is not None:
        has_cand = batch.cand_mask[:, t].sum(-1) > 0
    addr_row = None
    if batch.cand_addr_mask is not None and has_cand is not None:
        addr_row = has_cand & (batch.cand_addr_mask[:, t] > 0)
    from_ltm_row = None
    if batch.cand_from_ltm is not None and has_cand is not None:
        from_ltm_row = has_cand & (batch.cand_from_ltm[:, t].sum(-1) > 0)
    inv_row = batch.inverse_mask[:, t] > 0 if batch.inverse_mask is not None else None
    is_q = batch.is_q[:, t] > 0

    fam = []
    for i in range(b):
        if inv_row is not None and bool(inv_row[i]):
            fam.append("inverse_query")
        elif from_ltm_row is not None and bool(from_ltm_row[i]):
            fam.append("recall_link")
        elif addr_row is not None and bool(addr_row[i]):
            fam.append("definite_desc_read" if bool(is_q[i]) else "writeback_addr_redirect")
        elif has_cand is not None and bool(has_cand[i]):
            fam.append("pronoun_value_redirect")
        else:
            fam.append("plain_fact")
    return fam


def program_for_step_batch(batch) -> List[List[str]]:
    """:func:`program_for_step` for every step ``t`` -- ``[T][B]``."""
    T = batch.entity.shape[1]
    return [program_for_step(batch, t) for t in range(T)]


def family_of_step(batch, t: int) -> List[str]:
    """EXECUTOR PHASE 2 (dev/EXECUTOR_DESIGN.md Sec.3's LOFO gate): the
    per-(row) gold family at clause step ``t``, length ``B`` -- the name
    ``scripts/train_executor.py``'s LOFO splitter and
    :meth:`nsm_ct.executor.Executor.run_learned` key trace-loss masking
    off. Identical computation to :func:`program_for_step` (see this
    module's docstring's verified-coverage note above) -- exposed under
    this SEPARATE public name rather than aliased silently so a caller
    grepping for "LOFO" finds the entry point the design doc names, without
    this module maintaining two classifiers that could drift apart."""
    return program_for_step(batch, t)
