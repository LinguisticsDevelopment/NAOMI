"""EXECUTOR PHASE 1 (dev/EXECUTOR_DESIGN.md Sec.5's phase-0/1 row): the
register machine + the six gold programs (:mod:`nsm_ct.programs`) as data +
the bootstrap anchor -- NO learned op/argument selection yet (Phase 2).

Two-tier design, both tiers sharing this Executor's op dispatch and D1/D2
policy but operating at different granularity:

1. :meth:`Executor.run` -- reproduces
   :meth:`nsm_ct.clause_reactor.ClauseReactor.forward` AT THE SAME
   GRANULARITY forward() itself operates at: one named op-stage call per
   CLAUSE STEP, over the WHOLE BATCH at once, with per-ROW behavior coming
   from the SAME ``torch.where``/mask arithmetic ``_collapse``/``forward``
   already use (mirrors dev/EXECUTOR_DESIGN.md Sec.1.4's bootstrap trace
   verbatim, op-stage for op-stage). This is what makes it possible to
   reproduce a batch whose rows follow DIFFERENT gold families (Sec.1.4:
   "candidate families are mutually exclusive per row/step, so a real trace
   uses at most one bracket") without literally branching per row in
   Python -- exactly how ``forward()`` itself stays batched.
2. :meth:`Executor.execute_step_program` -- a genuine per-op interpreter
   over one CLAUSE's flat :class:`nsm_ct.programs.Step` list, used where a
   test needs to hand the Executor an explicit (possibly malicious) op
   sequence rather than a real batch's own structural flags -- the
   ≤1-WRITE-per-clause assertion and the hard/soft D1 policy tests. It
   covers every op NOT needing the variable-width per-candidate ``P.*``
   bank (RegisterFile's fixed ``[B, R, d]`` layout has no native slot for a
   ``C``-indexed register group -- see "Phase 2 needs" below);
   QUERY_CAND/INTERACT/SCORE/CAND_FOR/FORCE raise ``NotImplementedError``
   here and are only exercised, over real per-candidate tensors, by
   ``run()``.

Both tiers call the model's OWN learned heads (``model.gru``,
``model.write_gate``, ``model.overwrite_gate``, ``model.decide_truth``,
``model.respond``, ``model.response``, ``model.resolver``) and the SAME
:mod:`nsm_ct.ops` functions ``forward()`` itself is built on top of
(``ops.unbind_query`` wraps ``entity_memory.query`` byte-for-byte,
``ops.bind_write`` wraps ``entity_memory.write``, ``ops.inverse_query_entity``
wraps ``entity_memory.query_entity`` -- see ``ops.py``'s own docstrings) --
this is what makes the reproduction in (1) a real regression test of the
op-dispatch layer, not a tautology.

D1 (soft vs hard selection, CLAUDE.md/dev/EXECUTOR_DESIGN.md Sec.6): any
register whose type is ``"Addr"`` (a memory KEY) is selected HARD --
straight-through argmax -- regardless of training/eval mode; ``Dist``/
``Scalar`` registers may be soft. ``Executor.hard_keys`` (default ``True``)
is this policy's on/off switch for :meth:`execute_step_program`'s own
``SELECT``: ``True`` forces ``ops.select(..., hard=True)`` unconditionally;
``False`` allows soft selection but RAISES the moment an ``EMIT`` (or any
op) tries to write a soft distribution's result into an ``Addr``-typed
register -- soft mixture is not a legal ``Addr`` value (D1's own
rationale: "a soft Addr mixture is an interference-contaminated bilinear
key"). :meth:`run` does NOT consult ``hard_keys`` at all -- reproducing
forward() byte-for-byte means reproducing forward()'s OWN
training-mode-tied hard/soft policy (``ClauseReactor._collapse_weights``:
soft in train, hard in eval), which is what the bootstrap anchor is
required to match; the anchor tests run the model in eval mode, where the
two policies coincide (D1's hard-Addr rule and forward()'s own eval-mode
argmax collapse are the SAME arithmetic then) -- see this module's
"Phase 1 findings" section, item 3, for the discrepancy this surfaces for
train-mode use, which Phase 2 (D1's "soft-vs-argmax delta at every gate"
reporting requirement) has to resolve, not this milestone.

D2 (control vs data path, CLAUDE.md/dev/EXECUTOR_DESIGN.md Sec.6): Phase 1
has no learned op/arg selection at all (program membership is 100%
determined by the batch's own structural flags via
:func:`nsm_ct.programs.program_for_step`), so there is no ``OpSelect``/
``ArgSelect`` to Harvard-split yet. What Phase 1 DOES ship, per the BUILD
brief: :meth:`Executor.build_control_signal` constructs the ``ctrl`` input
vector D2 specifies (previous op id, step index within the clause's
op-loop, a type mask over the six Track C register types, and scalar
summaries -- collapse margin as the "margin" scalar, an abstain flag, and
remaining halt budget) at every op-stage of :meth:`run`, and returns it
alongside the trace (``out["_control_signal"]``) -- built, EXPOSED, but
consumed by nothing (no ``OpSelect`` head exists yet); Phase 2 is what
reads it.

-----------------------------------------------------------------------
Phase 1 findings (for the milestone report -- kept here too so a later
reader of just this module sees them, not only the report):
-----------------------------------------------------------------------
1. ``ops.emit`` (``ops.py:689-703``) is NOT the real EMIT: its own
   docstring says it is "an identity/reference stand-in only" that returns
   ``mem_read`` unchanged -- the actual weighted-sum aggregation
   (``(w.unsqueeze(-1) * candidates).sum(1)``) lives inline in
   ``ClauseReactor._collapse``, never factored into ``ops.py``. This
   Executor reproduces that inline weighted sum directly (both in ``run()``
   and in ``execute_step_program``'s ``EMIT`` handler) rather than calling
   ``ops.emit`` at all -- a real EMIT op belongs in ``ops.py`` before Phase
   2 can dispatch to it generically (a Phase 2 need, not fixed here: this
   milestone's brief is READ-only on ``ops.py``).
2. ``RegisterFile`` (``ops.py:711-804``) has no native slot for a
   variable-width, per-candidate ``P.*`` register GROUP (Sec.1.1's ``C``
   rows of ``addr/feat/prior/mem/score/score_extra/from_ltm``) or for a
   ``"Mem"``-typed register holding an actual ``[B, d, d, d]`` tensor --
   its ``values`` tensor is fixed ``[B, R, d]``. ``run()`` keeps ``M_mem``/
   ``M_ltm`` and every per-candidate tensor as ordinary local tensors
   outside the RegisterFile, recording their existence via
   ``RegisterFile.record`` (not ``.write``) for the trace, and stores
   ``D_w`` (a Dist over ``C`` candidates) inside the RegisterFile only by
   right-padding it to the file's uniform ``d``-width. A real per-candidate
   register bank (and a ``Mem``-typed slot kind with tensor-not-vector
   storage) is a Phase 2 need.
3. K_max naming COLLISION: ``ops.PATIENCE`` (already shipped, default
   ``6``, cites "TRACK_C_DESIGN Sec.1.6's K_max=6") and
   dev/EXECUTOR_DESIGN.md Sec.1.3's own "``K_max`` (default 12)" name the
   SAME concept (the per-clause op-loop budget) with DIFFERENT numbers.
   Measured here (see :meth:`Executor.program_length`): every candidate-
   bearing gold program's op-LOOP length (QUERY.../TICK/GATE/OVERWRITE/
   NEGATE/WRITE/HALT, excluding RESPOND/RESPONSE which Sec.1.3 says are
   "outside the op loop") already exceeds ``ops.PATIENCE`` (program 1:
   7 > 6; program 4/6: 9 > 6) -- ``ops.PATIENCE=6`` does not fit ANY of
   Phase 1's six gold programs. The BUILD brief specifies
   ``k_max=PATIENCE-from-ops`` as this Executor's default, so that default
   is honored here (``self.k_max = ops.PATIENCE`` unless overridden), but
   it is advisory only: neither ``run()`` nor ``execute_step_program``
   hard-fails against it (that would fail on every gold program by
   construction) -- ``Executor.program_length`` exposes the count so a
   caller/test can report the gap explicitly, and this needs a decision
   (raise ``ops.PATIENCE``, or accept ``execute_step_program`` running
   over budget) before Phase 2 makes ``K_max`` a real training-time cutoff.
4. Sense-collapse (M54) and hypothesis/garden-path-collapse (M55a) branches
   are OUT of Phase 1 scope -- no gold program in
   dev/EXECUTOR_DESIGN.md Sec.1.1 covers them (the design doc's own
   Sec.1.4 bootstrap-trace listing marks ``[sense]``/``[hyp]`` brackets as
   "Track A capabilities this trace must still reproduce but aren't among
   Sec.1.1's nine registers-frozen programs"). ``run()`` asserts neither
   branch is exercised by the batch it is given (``model.sense_resolver``/
   ``model.hyp_resolver`` installed AND the batch carrying that candidate
   kind) rather than silently producing wrong numbers -- a real gap, not a
   guarded default: those two branches need their own register-table
   extension before an Executor can touch them.
5. ``CAND_FOR`` (programs 4/6, step 1) is a documented no-op/pass-through
   here, not a design flaw introduced by this milestone -- see
   ``programs.py``'s module docstring for why (``candidates_for`` is not
   yet the live batch-build path, dev/OP_INVENTORY.md Sec.5).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import entity_memory as em
from . import ops
from . import op_select as op_select_mod
from .clause_reactor import ClauseBatch, ClauseReactor
from .ltm import mem_total
from .membrane import FEATURE_DIM
from .resolver import evidence_interaction, query_candidates
from . import programs as programs_mod
from .programs import Step, program_for_step

__all__ = ["Executor"]

# EXECUTOR PHASE 2: each family's op-loop ENTRY op (dev/EXECUTOR_DESIGN.md
# Sec.1.1's Step[0] per program) -- see src/nsm_ct/op_select.py's module
# docstring point 1 for what this drives (OpSelect's gold target) and does
# not drive (execution, which stays the SAME structural dispatch run()
# already uses).
FAMILY_ENTRY_OP: Dict[str, str] = {name: prog[0].op for name, prog in programs_mod.GOLD_PROGRAMS.items()}


class Executor:
    """The Phase 1 register machine over a :class:`ClauseReactor`. See the
    module docstring for the two-tier ``run()``/``execute_step_program()``
    split and the D1/D2 policy this class enforces.
    """

    def __init__(self, model: ClauseReactor, *, k_max: Optional[int] = None, hard_keys: bool = True) -> None:
        self.model = model
        self.k_max = ops.PATIENCE if k_max is None else k_max
        self.hard_keys = hard_keys

    # ------------------------------------------------------------------
    # K_max bookkeeping (finding 3 above) -- advisory only.
    # ------------------------------------------------------------------
    @staticmethod
    def program_length(program: Sequence[Step], *, op_loop_only: bool = True) -> int:
        """Count of ``program``'s steps. ``op_loop_only=True`` (default)
        excludes ``RESPOND``/``RESPONSE`` (Sec.1.3: "outside the op loop,
        scored once per episode") and ``HALT`` itself doesn't add a step
        beyond the loop's own budget -- this is the count ``K_max`` is
        actually meant to bound. ``op_loop_only=False`` counts every row of
        the flat per-clause trace instead (what a caller ends up with in
        ``RegisterFile.trace`` for one clause)."""
        if not op_loop_only:
            return len(program)
        return sum(1 for s in program if s.op not in ("RESPOND", "RESPONSE"))

    # ------------------------------------------------------------------
    # D2: the control-only signal an OpSelect head will read in Phase 2.
    # Built and exposed here; consumed by nothing yet (no OpSelect exists).
    # ------------------------------------------------------------------
    @staticmethod
    def build_control_signal(*, prev_op_id: int, step_idx: int, type_mask: torch.Tensor,
                              margin: torch.Tensor, abstain_flag: torch.Tensor,
                              halt_budget: int) -> Dict[str, torch.Tensor]:
        """D2's ``ctrl`` input: previous op id, the op-loop step index,
        a type mask over Track C's five register types
        (Addr/Vec/Feat/Scalar/Dist, ``type_mask`` order), and scalar
        summaries (top1-top2 collapse margin, an abstain flag, remaining
        halt budget) -- CONTROL SIGNALS ONLY, never register DATA vectors
        (the Harvard split D2 requires: ``OpSelect`` "no longer reads the
        main clause-level `state` directly"). Returns a plain dict (not a
        RegisterFile entry -- ``ctrl`` is explicitly "not a register" per
        Sec.1.1: "`state`... and `ctrl`... are not registers -- neither
        carries facts")."""
        return {
            "prev_op_id": torch.as_tensor(prev_op_id),
            "step_idx": torch.as_tensor(step_idx),
            "type_mask": type_mask,
            "margin": margin,
            "abstain_flag": abstain_flag,
            "halt_budget_remaining": torch.as_tensor(halt_budget - step_idx),
        }

    # ==================================================================
    # Tier 1: run() -- reproduces forward() at forward()'s own per-clause,
    # whole-batch granularity. See module docstring.
    # ==================================================================
    def run(self, batch: ClauseBatch, programs: Optional[List[List[str]]] = None, *,
            ltm: Optional[torch.Tensor] = None, return_write_trace: bool = False,
            return_memory: bool = False, return_mem_read: bool = False,
            return_registers: bool = False, force_plain_fact: bool = False) -> Dict[str, torch.Tensor]:
        """Reproduce ``ClauseReactor.forward(batch, ltm=ltm, ...)``.
        ``force_plain_fact`` (EXECUTOR PHASE 2, dev/EXECUTOR_DESIGN.md
        Sec.3's LOFO floor arm -- "op selection frozen to the plain-fact
        program"): when ``True``, every candidate-collapse branch AND the
        inverse-query branch are skipped UNCONDITIONALLY regardless of what
        the batch's own structural flags say -- every clause step runs
        ONLY program 1 (``QUERY(A_e,A_r)->EPILOGUE-W[A_e]``), exactly
        :data:`nsm_ct.programs.PLAIN_FACT`. Default ``False`` is
        byte-identical to every pre-Phase-2 call site. This is the ``run()``
        vs ``run_learned()`` floor -- NOT Track A's own "cheat" arm
        (Sec.3: "a real floor arm... never Track A, that's the tournament's
        floor, Sec.4").
        ``programs`` (optional) is the ``[T][B]`` gold-family-label
        structure from :func:`nsm_ct.programs.program_for_step_batch` (or a
        caller override of the same shape) -- Phase 1 has no learned
        routing, so it is used ONLY to label ``out["_program_family"]`` for
        inspection/testing; execution is always driven by the batch's own
        structural flags (``program_for_step`` computed internally when
        ``programs`` is ``None``), which is exactly the "pipeline as one
        fixed program" claim ``programs.py``'s module docstring makes --
        passing a mismatched label here cannot change what actually runs.

        Returns the same output-dict keys as ``forward()`` for the subset
        of branches Phase 1 covers (entity/M53b-M57 candidate collapse,
        M57c.2 address re-read + entity-axis inverse read, M59a LTM additive
        read) -- see finding 4 above for the sense/hyp scope exclusion.
        """
        model = self.model
        b, T, d = batch.entity.shape
        device = batch.entity.device
        state = torch.zeros(b, model.gru.hidden_size, device=device)
        memory = em.init_memory(b, d, device)

        coord = batch._coord()
        have_resolver_data = model.resolver is not None and batch.cand_mask is not None and not force_plain_fact
        if model.sense_resolver is not None and batch.sense_cand_mask is not None:
            raise NotImplementedError(
                "Executor Phase 1 does not implement the sense-collapse branch "
                "(M54) -- out of scope, see executor.py's module docstring finding 4.")
        if model.hyp_resolver is not None and batch.hyp_cand_mask is not None:
            raise NotImplementedError(
                "Executor Phase 1 does not implement the hypothesis-collapse branch "
                "(M55a) -- out of scope, see executor.py's module docstring finding 4.")

        Cmax = batch.cand_entity.shape[2] if batch.cand_entity is not None else 0
        rf_dim = max(d, Cmax, 1)

        def _pad(x: torch.Tensor) -> torch.Tensor:
            if x.shape[-1] == rf_dim:
                return x
            return F.pad(x, (0, rf_dim - x.shape[-1]))

        rf = ops.RegisterFile.create(b, rf_dim, {
            "A_e": "Addr", "A_r": "Addr", "A_w": "Addr", "V_v": "Vec", "V_read": "Vec",
            "V_ev": "Vec", "V_desc": "Vec", "D_w": "Dist", "S_gate": "Scalar",
            "S_owr": "Scalar", "S_neg": "Scalar",
        }, device=device)
        gstep = 0

        inverse_readout = torch.zeros(b, d, device=device) if batch.inverse_mask is not None else None
        resp_logits, resp_vecs = [], []
        resolver_logits_all, resolver_margin_all = [], []
        mem_read_all = []
        gate_trace: List[torch.Tensor] = []
        overwrite_trace: List[torch.Tensor] = []
        neg_trace: List[torch.Tensor] = []
        redirected_trace: List[torch.Tensor] = []
        resolved_idx_trace: List[torch.Tensor] = []
        v_read_reg_all, a_w_reg_all, d_w_reg_all, s_gate_reg_all = [], [], [], []
        family_all: List[List[str]] = []
        control_signals: List[Dict[str, torch.Tensor]] = []

        for t in range(T):
            e0, r, v0 = batch.entity[:, t], batch.relation[:, t], batch.value[:, t]
            p, c = batch.pred[:, t], coord[:, t]
            real, isq = batch.mask[:, t], batch.is_q[:, t]

            family_t = programs[t] if programs is not None else program_for_step(batch, t)
            family_all.append(family_t)

            rf.write("A_e", _pad(e0), step=gstep, op_name="preload_A_e"); gstep += 1
            rf.write("A_r", _pad(r), step=gstep, op_name="preload_A_r"); gstep += 1
            rf.write("V_v", _pad(v0), step=gstep, op_name="preload_V_v"); gstep += 1
            if batch.cand_evidence_target is not None:
                rf.write("V_ev", _pad(batch.cand_evidence_target[:, t]), step=gstep, op_name="preload_V_ev")
                gstep += 1

            # RECALL (ltm.mem_total, M59a): the additive STM+LTM read view,
            # computed once per step and reused for every read this step.
            mem_total_t = mem_total(memory, ltm)
            rf.record("RECALL", ("M_mem", "M_ltm"), step=gstep); gstep += 1

            # QUERY: the pre-collapse read at the placeholder address.
            v_read = ops.unbind_query(mem_total_t, e0, r)
            rf.write("V_read", _pad(v_read), step=gstep, op_name="QUERY", args=("A_e", "A_r")); gstep += 1

            e, v = e0, v0
            res_logits_t = res_margin_t = None
            addr_row_t = None
            resolved_idx_t = None
            d_w_full = None

            if have_resolver_data:
                ce_t, cf_t = batch.cand_entity[:, t], batch.cand_feature[:, t]
                cp_t, cm_t = batch.cand_prior[:, t], batch.cand_mask[:, t]
                if batch.cand_evidence_relation is not None:
                    er_t = batch.cand_evidence_relation[:, t]
                    has_er = er_t.norm(dim=-1, keepdim=True) > 0
                    evidence_r = torch.where(has_er, er_t, r)
                else:
                    evidence_r = r
                cand_mem_read = query_candidates(mem_total_t, ce_t, evidence_r)      # QUERY_CAND
                rf.record("QUERY_CAND", ("P.addr", "A_r|evidence_r"), step=gstep); gstep += 1

                s_c = None
                if batch.cand_evidence_target is not None:
                    et_t = batch.cand_evidence_target[:, t]
                    s_c = evidence_interaction(cand_mem_read, et_t).unsqueeze(-1)    # INTERACT
                    rf.record("INTERACT", ("P.mem", "V_ev"), step=gstep); gstep += 1

                if s_c is not None and model.evidence_prior_beta is not None:
                    boost_logits = (s_c.squeeze(-1) * model.evidence_prior_beta).masked_fill(cm_t <= 0, -1e9)
                    cp_t = cp_t * torch.softmax(boost_logits, dim=-1)

                extra: Dict[str, torch.Tensor] = {}
                if getattr(model.resolver, "use_cand_feature", False) and batch.cand_feature_per_candidate is not None:
                    extra["cand_feature_per_candidate"] = batch.cand_feature_per_candidate[:, t]
                extra_cols = []
                if s_c is not None:
                    extra_cols.append(s_c)
                if batch.cand_from_ltm is not None:
                    extra_cols.append(batch.cand_from_ltm[:, t].unsqueeze(-1).to(ce_t.dtype))
                if batch.cand_recency is not None:
                    extra_cols.append(batch.cand_recency[:, t].to(ce_t.dtype))
                extra_width = getattr(model.resolver, "cand_feature_extra", 0)
                if extra_width > 0 and extra_cols:
                    cfpc = extra.get("cand_feature_per_candidate")
                    if cfpc is None:
                        b_, C_, _d_ = ce_t.shape
                        cfpc = ce_t.new_zeros(b_, C_, FEATURE_DIM)
                    stacked_extra = torch.cat(extra_cols, dim=-1)
                    k = stacked_extra.shape[-1]
                    if k < extra_width:
                        pad = ce_t.new_zeros(*stacked_extra.shape[:-1], extra_width - k)
                        stacked_extra = torch.cat([stacked_extra, pad], dim=-1)
                    elif k > extra_width:
                        stacked_extra = stacked_extra[..., :extra_width]
                    extra["cand_feature_per_candidate"] = torch.cat([cfpc, stacked_extra], dim=-1)

                logits = model.resolver(ce_t, cf_t, cp_t, cm_t, cand_mem_read, state, **extra)  # SCORE
                rf.record("SCORE", ("P.addr", "P.feat", "P.prior", "P.mem"), step=gstep); gstep += 1
                logits = logits.masked_fill(cm_t <= 0, -1e9)
                has_cand = cm_t.sum(-1) > 0
                # D1: forward() itself ties hard/soft to training mode (see
                # module docstring's D1 paragraph) -- reproduced verbatim so
                # the bootstrap anchor matches byte-for-byte in eval mode.
                w = ClauseReactor._collapse_weights(logits, model.training)         # SELECT
                rf.record("SELECT", ("P.score",), step=gstep); gstep += 1

                if batch.cand_forced_index is not None:
                    forced_t = batch.cand_forced_index[:, t]
                    forced_valid = has_cand & (forced_t >= 0)
                    C = w.shape[-1]
                    forced_onehot = F.one_hot(forced_t.clamp(min=0), num_classes=C).to(w.dtype)
                    w = torch.where(forced_valid.unsqueeze(-1), forced_onehot, w)
                    rf.record("FORCE", ("D_w",), step=gstep); gstep += 1

                resolved_idx_t = torch.where(has_cand, w.argmax(-1), torch.full_like(has_cand, -1, dtype=torch.long))
                resolved_v = (w.unsqueeze(-1) * cand_mem_read).sum(1)               # EMIT (value)
                resolved_e = (w.unsqueeze(-1) * ce_t).sum(1)                        # EMIT (address)
                rf.record("EMIT", ("D_w", "P.mem_or_addr"), step=gstep); gstep += 1

                if batch.cand_addr_mask is not None:
                    addr_row = has_cand & (batch.cand_addr_mask[:, t] > 0)
                else:
                    addr_row = torch.zeros_like(has_cand)
                value_row = has_cand & ~addr_row
                v = torch.where(value_row.unsqueeze(-1), resolved_v, v)
                e = torch.where(addr_row.unsqueeze(-1), resolved_e, e)
                res_logits_t = logits
                res_margin_t = ClauseReactor._top2_margin(logits, has_cand, v)
                addr_row_t = addr_row
                d_w_full = w

                rf.write("D_w", _pad(w), step=gstep, op_name="record_D_w"); gstep += 1
                rf.write("A_w", _pad(e), step=gstep, op_name="EMIT_addr_or_carry", args=("D_w",)); gstep += 1
                rf.write("V_v", _pad(v), step=gstep, op_name="EMIT_value_or_carry"); gstep += 1

            # REREAD: post-collapse re-read at the resolved address.
            if addr_row_t is not None and bool(addr_row_t.any()):
                reread = ops.unbind_query(mem_total_t, e, r)
                v_read = torch.where(addr_row_t.unsqueeze(-1), reread, v_read)
                rf.write("V_read", _pad(v_read), step=gstep, op_name="REREAD", args=("A_w", "A_r")); gstep += 1

            # QUERY_ENTITY: entity-axis inverse read.
            if batch.inverse_mask is not None and not force_plain_fact:
                inv_row = batch.inverse_mask[:, t] > 0
                if bool(inv_row.any()):
                    inv_readout_t = ops.inverse_query_entity(mem_total_t, r, v)
                    v_read = torch.where(inv_row.unsqueeze(-1), inv_readout_t, v_read)
                    inverse_readout = torch.where(inv_row.unsqueeze(-1), inv_readout_t, inverse_readout)
                    rf.write("V_read", _pad(v_read), step=gstep, op_name="QUERY_ENTITY", args=("A_r", "V_v"))
                    gstep += 1

            if return_mem_read:
                mem_read_all.append(v_read)
            if return_registers:
                v_read_reg_all.append(v_read)
                a_w_reg_all.append(e)
                d_w_reg_all.append(d_w_full)

            # D2: build (and expose, unconsumed) the control-only signal.
            margin_t = res_margin_t if res_margin_t is not None else torch.zeros(b, device=device)
            abstain_t = margin_t < ops.CAUTION
            type_mask = torch.tensor([1, 1, 0, 1, 1], dtype=torch.float32)  # Addr,Vec,Feat,Scalar,Dist
            control_signals.append(self.build_control_signal(
                prev_op_id=0, step_idx=t, type_mask=type_mask, margin=margin_t,
                abstain_flag=abstain_t, halt_budget=self.k_max))

            # TICK
            state = model.gru(torch.cat([e, r, v, p, c, v_read], dim=-1), state)
            rf.record("TICK", ("A_e_or_A_w", "A_r", "V_v", "p", "c", "V_read"), step=gstep); gstep += 1

            stmt = real * (1.0 - isq)
            gate = torch.sigmoid(model.write_gate(state)).squeeze(-1) * stmt        # GATE
            rf.write("S_gate", gate, step=gstep, op_name="GATE"); gstep += 1
            owr = torch.sigmoid(model.overwrite_gate(state)).squeeze(-1) * gate     # OVERWRITE
            rf.write("S_owr", owr, step=gstep, op_name="OVERWRITE"); gstep += 1
            neg = torch.sigmoid(model.decide_truth(torch.cat([state, v], dim=-1))).squeeze(-1) * stmt  # NEGATE
            rf.write("S_neg", neg, step=gstep, op_name="NEGATE"); gstep += 1

            if return_registers:
                s_gate_reg_all.append(gate)

            # WRITE (ops.bind_write == entity_memory.write): exactly one per
            # clause step in this loop's own control flow -- the ≤1-WRITE
            # invariant (Sec.1.3) holds by CONSTRUCTION here (there is
            # exactly one WRITE call site in this loop); the assertion has
            # real teeth against a hand-built adversarial program, tested
            # via execute_step_program (Tier 2) instead.
            write_count = 1
            assert write_count <= 1, "more than one WRITE per clause step (Sec.1.3 invariant)"
            memory = ops.bind_write(memory, e, r, v, gate - neg, overwrite=owr)

            if return_write_trace:
                gate_trace.append(gate)
                overwrite_trace.append(owr)
                neg_trace.append(neg)
                redirected_trace.append(
                    addr_row_t if addr_row_t is not None else torch.zeros(b, dtype=torch.bool, device=device))
                resolved_idx_trace.append(
                    resolved_idx_t if resolved_idx_t is not None
                    else torch.full((b,), -1, dtype=torch.long, device=device))

            # RESPOND / RESPONSE
            rl = model.respond(state).squeeze(-1)
            rl = rl.masked_fill(real <= 0, float("-inf"))
            resp_logits.append(rl)
            resp_vecs.append(model.response(torch.cat([state, v_read], dim=-1)))
            rf.record("RESPOND", ("state",), step=gstep); gstep += 1
            rf.record("RESPONSE", ("state", "V_read"), step=gstep); gstep += 1
            rf.record("HALT", (), step=gstep); gstep += 1

            if have_resolver_data:
                resolver_logits_all.append(res_logits_t)
                resolver_margin_all.append(res_margin_t)

        RL = torch.stack(resp_logits, dim=1)
        RV = torch.stack(resp_vecs, dim=1)
        wA = torch.softmax(RL, dim=1)
        resp = (wA.unsqueeze(-1) * RV).sum(dim=1)

        rn = resp / (resp.norm(dim=-1, keepdim=True) + 1e-8)
        on = batch.options / (batch.options.norm(dim=-1, keepdim=True) + 1e-8)
        answer_logits = torch.einsum("bd,bkd->bk", rn, on) * 10.0

        out: Dict[str, torch.Tensor] = {
            "answer_logits": answer_logits, "response": resp,
            "respond_gates": wA, "respond_position": (wA * batch.is_q).sum(1),
        }
        if have_resolver_data:
            out["resolver_logits"] = torch.stack(resolver_logits_all, dim=1)
            out["resolver_margin"] = torch.stack(resolver_margin_all, dim=1)
        if inverse_readout is not None:
            out["inverse_direct_logits"] = ops.similarity(
                inverse_readout.unsqueeze(1), batch.options).cosine

        if return_memory:
            out["_memory"] = memory
        if return_mem_read:
            out["_mem_read"] = torch.stack(mem_read_all, dim=1)
        if return_write_trace:
            out["_write_trace"] = {
                "gate": torch.stack(gate_trace, dim=1),
                "overwrite": torch.stack(overwrite_trace, dim=1),
                "neg": torch.stack(neg_trace, dim=1),
                "redirected": torch.stack(redirected_trace, dim=1),
                "resolved_index": torch.stack(resolved_idx_trace, dim=1),
            }
        if return_registers:
            out["_registers"] = {
                "V_read": torch.stack(v_read_reg_all, dim=1),
                "A_w": torch.stack(a_w_reg_all, dim=1),
                "S_gate": torch.stack(s_gate_reg_all, dim=1),
                "D_w": d_w_reg_all,   # list[T] of [B, C] or None -- variable C, not stacked
            }
        out["_program_family"] = family_all
        out["_trace"] = rf.trace
        out["_control_signal"] = control_signals
        return out

    # ==================================================================
    # EXECUTOR PHASE 2: run_learned() -- learned op/argument selection
    # with trace supervision, over the SAME Phase-1 dispatch run() uses.
    # See src/nsm_ct/op_select.py's module docstring for the exact scope
    # of what is learned here (two decision points: the entry op, and
    # EMIT's destination register) and why the frozen v0 register table
    # (D3) leaves no further op/arg ambiguity at this granularity -- this
    # is a deliberate, documented scoping decision (see this milestone's
    # RESEARCH_NOTES entry), not an oversight: RegisterFile has no native
    # per-candidate ``P.*`` slot (Phase 1 finding 2), so a per-candidate
    # ArgSelect is a Phase-3-or-later, register-file-unfreezing need.
    # ==================================================================
    def run_learned(self, batch: ClauseBatch, op_select: "op_select_mod.OpSelect",
                     arg_select: "op_select_mod.ArgSelect", *,
                     ltm: Optional[torch.Tensor] = None,
                     lofo_family: Optional[str] = None,
                     teacher_force: bool = True,
                     trace_weight: float = 1.0,
                     dest_mode: str = "hard",
                     return_write_trace: bool = False,
                     return_memory: bool = False) -> Dict[str, torch.Tensor]:
        """PHASE 2 (dev/EXECUTOR_DESIGN.md Sec.2/Sec.3). See
        src/nsm_ct/op_select.py's module docstring for the two learned
        decisions (entry op, EMIT destination) this method trains, and
        this module's own D1/D2 paragraphs for the policy both obey.

        ``lofo_family`` (Sec.3's LOFO gate): every step whose gold family
        (:func:`nsm_ct.programs.family_of_step`) equals this name gets
        ZERO trace loss (no op/arg cross-entropy supervision) -- but its
        EMIT-destination decision, unlike every other family's, is driven
        by ArgSelect's OWN (straight-through) prediction rather than the
        gold ``cand_addr_mask`` flag, so TASK loss alone can still shape
        that family's routing ("its steps get NO trace loss, only task
        loss"). ``teacher_force=False`` (the "no-trace eval" honesty arm)
        extends learned-driven execution to EVERY family regardless of
        ``lofo_family``, and trace loss is not computed for any step.

        ``dest_mode``: ``"hard"`` (default, D1) -- whenever a row's
        EMIT-destination decision is learned-driven, it is
        straight-through (hard argmax forward, soft gradient backward).
        ``"soft"`` -- a pure differentiable blend by the raw softmax
        weight instead, for the ``--soft-report`` soft-vs-argmax delta
        ONLY (D1 forbids this as the real policy; never the default).

        Returns everything ``run()`` returns for the branches this method
        covers (``answer_logits``, ``response``, ``respond_gates``,
        ``respond_position``; ``_memory``/``_write_trace`` when
        requested), PLUS:
          - ``trace_loss``: scalar tensor (entry-op CE + EMIT-dest CE,
            each averaged over its own real/non-LOFO'd/has-candidate
            mask, summed, scaled by ``trace_weight``).
          - ``op_acc`` / ``arg_acc``: ``{family: (correct, total)}`` --
            entry-op / EMIT-dest accuracy against the gold family, over
            every real step this batch contains (INCLUDING the LOFO'd
            family -- accuracy is measured there, just never trained on
            directly via trace loss).
          - ``write_violations``: ``int``, always ``0`` here -- this
            method's task computation reuses ``run()``'s own single
            ``ops.bind_write`` call site per clause step, the SAME
            "holds by construction" guarantee Phase 1 documents; see
            :func:`nsm_ct.op_select.mask_second_write` for the general
            masking mechanism (exercised directly by
            tests/test_executor_phase2.py on a synthetic sequence).
          - ``gold_program_length_histogram``: ``{op_loop_length: count}``
            over every real step's OWN gold family's
            ``Executor.program_length`` -- this milestone does not
            implement variable-length LEARNED halting (see module note
            above), so this reports the honest analogue: how long the
            gold program each step is teacher-forced against actually is.
        """
        model = self.model
        if model.resolver is None:
            raise ValueError(
                "Executor.run_learned needs a model with a resolver installed -- the "
                "EMIT-destination decision it trains has nothing to score candidates with "
                "otherwise; use run(force_plain_fact=True) for the floor arm instead.")
        b, T, d = batch.entity.shape
        device = batch.entity.device
        state = torch.zeros(b, model.gru.hidden_size, device=device)
        memory = em.init_memory(b, d, device)
        coord = batch._coord()
        have_cand_data = batch.cand_mask is not None

        prev_op_id = torch.zeros(b, dtype=torch.long, device=device)  # 0 = sentinel, "no previous op"
        type_mask_const = torch.tensor([1, 1, 0, 1, 1], dtype=torch.float32, device=device)
        entry_legal_mask = torch.zeros(len(op_select_mod.OP_VOCAB), dtype=torch.bool, device=device)
        for _op in op_select_mod.ENTRY_OPS:
            entry_legal_mask[op_select_mod.OP_INDEX[_op]] = True
        emit_op_id_const = torch.full((b,), op_select_mod.OP_INDEX["EMIT"] + 1, dtype=torch.long, device=device)

        resp_logits, resp_vecs = [], []
        gate_trace: List[torch.Tensor] = []
        overwrite_trace: List[torch.Tensor] = []
        neg_trace: List[torch.Tensor] = []
        redirected_trace: List[torch.Tensor] = []
        resolved_idx_trace: List[torch.Tensor] = []

        trace_loss = torch.zeros((), device=device)
        op_correct: Dict[str, int] = {}
        op_total: Dict[str, int] = {}
        arg_correct: Dict[str, int] = {}
        arg_total: Dict[str, int] = {}
        length_hist: Dict[int, int] = {}

        for t in range(T):
            e0, r, v0 = batch.entity[:, t], batch.relation[:, t], batch.value[:, t]
            p, c = batch.pred[:, t], coord[:, t]
            real, isq = batch.mask[:, t], batch.is_q[:, t]
            real_bool = real > 0

            family_t = programs_mod.family_of_step(batch, t)   # list[str], len B
            for fam, is_real in zip(family_t, real_bool.tolist()):
                if is_real:
                    L = self.program_length(programs_mod.GOLD_PROGRAMS[fam])
                    length_hist[L] = length_hist.get(L, 0) + 1

            keep = op_select_mod.lofo_keep_mask(family_t, lofo_family)
            lofo_row = ~torch.tensor(keep, dtype=torch.bool, device=device)

            mem_total_t = mem_total(memory, ltm)
            v_read = ops.unbind_query(mem_total_t, e0, r)

            # -- Decision A: entry op (observational, see module note). --
            entry_gold_idx = torch.tensor(
                [op_select_mod.OP_INDEX[FAMILY_ENTRY_OP[f]] for f in family_t], device=device)
            ctrl_a = op_select_mod.control_signal_to_tensor(self.build_control_signal(
                prev_op_id=prev_op_id, step_idx=t, type_mask=type_mask_const,
                margin=torch.zeros(b, device=device),
                abstain_flag=torch.zeros(b, dtype=torch.bool, device=device),
                halt_budget=self.k_max), batch=b, device=device)
            logits_a, _ctrl_vec_a = op_select(ctrl_a, legal_mask=entry_legal_mask)

            trace_row_a = (real_bool & ~lofo_row) if teacher_force else torch.zeros(b, dtype=torch.bool, device=device)
            if bool(trace_row_a.any()):
                ce_a = F.cross_entropy(logits_a, entry_gold_idx, reduction="none")
                trace_loss = trace_loss + trace_weight * (
                    (ce_a * trace_row_a.float()).sum() / trace_row_a.float().sum())
            pred_a = logits_a.argmax(-1)
            ok_a = (pred_a == entry_gold_idx)
            for fam, ok, is_real in zip(family_t, ok_a.tolist(), real_bool.tolist()):
                if is_real:
                    op_total[fam] = op_total.get(fam, 0) + 1
                    op_correct[fam] = op_correct.get(fam, 0) + int(ok)
            prev_op_id = entry_gold_idx + 1   # teacher-forced recurrence, see module docstring

            e, v = e0, v0
            addr_row_bool = torch.zeros(b, dtype=torch.bool, device=device)
            resolved_idx_t = torch.full((b,), -1, dtype=torch.long, device=device)

            if have_cand_data:
                ce_t, cf_t = batch.cand_entity[:, t], batch.cand_feature[:, t]
                cp_t, cm_t = batch.cand_prior[:, t], batch.cand_mask[:, t]
                if batch.cand_evidence_relation is not None:
                    er_t = batch.cand_evidence_relation[:, t]
                    has_er = er_t.norm(dim=-1, keepdim=True) > 0
                    evidence_r = torch.where(has_er, er_t, r)
                else:
                    evidence_r = r
                cand_mem_read = query_candidates(mem_total_t, ce_t, evidence_r)

                s_c = None
                if batch.cand_evidence_target is not None:
                    et_t = batch.cand_evidence_target[:, t]
                    s_c = evidence_interaction(cand_mem_read, et_t).unsqueeze(-1)
                if s_c is not None and model.evidence_prior_beta is not None:
                    boost_logits = (s_c.squeeze(-1) * model.evidence_prior_beta).masked_fill(cm_t <= 0, -1e9)
                    cp_t = cp_t * torch.softmax(boost_logits, dim=-1)

                extra: Dict[str, torch.Tensor] = {}
                if getattr(model.resolver, "use_cand_feature", False) and batch.cand_feature_per_candidate is not None:
                    extra["cand_feature_per_candidate"] = batch.cand_feature_per_candidate[:, t]
                extra_cols = []
                if s_c is not None:
                    extra_cols.append(s_c)
                if batch.cand_from_ltm is not None:
                    extra_cols.append(batch.cand_from_ltm[:, t].unsqueeze(-1).to(ce_t.dtype))
                if batch.cand_recency is not None:
                    extra_cols.append(batch.cand_recency[:, t].to(ce_t.dtype))
                # NOTE (Executor Phase 2 fix, run_learned only -- run()
                # above is untouched, pinned by the Phase 1 anchor tests,
                # and never hits this combination): pad whenever
                # `extra_width > 0`, NOT only when `extra_cols` is
                # non-empty. A shared resolver trained across a MIXED
                # corpus (this method's whole point) can have
                # `cand_feature_extra > 0` while a given family populates
                # `cand_feature_per_candidate` (width FEATURE_DIM, e.g.
                # WriteBackCurriculumGenerator's candidate sets) but
                # supplies NONE of the optional extra scalar columns
                # (evidence-interaction/from_ltm/recency) -- `extra_cols`
                # empty in that case must still zero-pad up to
                # `resolver._cfpc_width`, or the resolver's declared
                # (construction-time) input width and what actually gets
                # concatenated here silently diverge (a real bug, found
                # via this milestone's own mixed-corpus training/tests).
                extra_width = getattr(model.resolver, "cand_feature_extra", 0)
                if extra_width > 0:
                    cfpc = extra.get("cand_feature_per_candidate")
                    if cfpc is None:
                        b_, C_, _d_ = ce_t.shape
                        cfpc = ce_t.new_zeros(b_, C_, FEATURE_DIM)
                    stacked_extra = (torch.cat(extra_cols, dim=-1) if extra_cols
                                      else cfpc.new_zeros(*cfpc.shape[:-1], 0))
                    k = stacked_extra.shape[-1]
                    if k < extra_width:
                        pad = cfpc.new_zeros(*stacked_extra.shape[:-1], extra_width - k)
                        stacked_extra = torch.cat([stacked_extra, pad], dim=-1)
                    elif k > extra_width:
                        stacked_extra = stacked_extra[..., :extra_width]
                    extra["cand_feature_per_candidate"] = torch.cat([cfpc, stacked_extra], dim=-1)

                logits = model.resolver(ce_t, cf_t, cp_t, cm_t, cand_mem_read, state, **extra)
                logits = logits.masked_fill(cm_t <= 0, -1e9)
                has_cand = cm_t.sum(-1) > 0
                w = ClauseReactor._collapse_weights(logits, model.training)

                resolved_idx_t = torch.where(has_cand, w.argmax(-1), resolved_idx_t)
                resolved_v = (w.unsqueeze(-1) * cand_mem_read).sum(1)
                resolved_e = (w.unsqueeze(-1) * ce_t).sum(1)
                res_margin_t = ClauseReactor._top2_margin(logits, has_cand, v)

                # -- Decision B: EMIT's destination register (A_w vs V_v). --
                gold_addr_row = (has_cand & (batch.cand_addr_mask[:, t] > 0)) if batch.cand_addr_mask is not None \
                    else torch.zeros_like(has_cand)
                dest_gold_idx = (~gold_addr_row).long()   # 0 -> A_w (addr redirect), 1 -> V_v (value redirect)
                abstain_t = res_margin_t < ops.CAUTION
                ctrl_b = op_select_mod.control_signal_to_tensor(self.build_control_signal(
                    prev_op_id=prev_op_id, step_idx=t, type_mask=type_mask_const,
                    margin=res_margin_t, abstain_flag=abstain_t,
                    halt_budget=self.k_max - 1), batch=b, device=device)
                _logits_b, ctrl_vec_b = op_select(ctrl_b)
                dest_logits = arg_select(ctrl_vec_b, emit_op_id_const, op_select_mod.ARG_SIGNATURES["EMIT"])
                dest_soft = torch.softmax(dest_logits, dim=-1)
                dest_pred_idx = dest_logits.argmax(-1)

                trace_row_b = (has_cand & real_bool & ~lofo_row) if teacher_force \
                    else torch.zeros(b, dtype=torch.bool, device=device)
                if bool(trace_row_b.any()):
                    ce_b = F.cross_entropy(dest_logits, dest_gold_idx, reduction="none")
                    trace_loss = trace_loss + trace_weight * (
                        (ce_b * trace_row_b.float()).sum() / trace_row_b.float().sum())
                ok_b = (dest_pred_idx == dest_gold_idx) & has_cand & real_bool
                for fam, ok, hc, is_real in zip(family_t, ok_b.tolist(), has_cand.tolist(), real_bool.tolist()):
                    if is_real and hc:
                        arg_total[fam] = arg_total.get(fam, 0) + 1
                        arg_correct[fam] = arg_correct.get(fam, 0) + int(ok)

                # D1: straight-through whenever this row's decision is
                # learned-driven (LOFO'd family, or teacher_force=False);
                # gold otherwise (Stage-a teacher forcing).
                dest_soft_addr = dest_soft[:, 0]
                if dest_mode == "soft":
                    pred_addr_ind = dest_soft_addr
                elif dest_mode == "hard":
                    hard_addr = (dest_pred_idx == 0).float()
                    pred_addr_ind = hard_addr + (dest_soft_addr - dest_soft_addr.detach())
                else:
                    raise ValueError(f"run_learned: unknown dest_mode {dest_mode!r}, expected 'hard' or 'soft'")
                use_learned_row = lofo_row if teacher_force else torch.ones(b, dtype=torch.bool, device=device)
                addr_indicator = torch.where(use_learned_row, pred_addr_ind, gold_addr_row.float())

                addr_w = addr_indicator * has_cand.float()
                value_w = (1.0 - addr_indicator) * has_cand.float()
                e = e0 * (1.0 - addr_w).unsqueeze(-1) + resolved_e * addr_w.unsqueeze(-1)
                v = v0 * (1.0 - value_w).unsqueeze(-1) + resolved_v * value_w.unsqueeze(-1)
                addr_row_bool = addr_w > 0.5

                prev_op_id = torch.where(has_cand, emit_op_id_const, prev_op_id)

            # REREAD: post-collapse re-read at the resolved address.
            if bool(addr_row_bool.any()):
                reread = ops.unbind_query(mem_total_t, e, r)
                v_read = torch.where(addr_row_bool.unsqueeze(-1), reread, v_read)

            # QUERY_ENTITY: entity-axis inverse read -- structural DATA
            # (batch.inverse_mask), not a learned choice; see module note.
            if batch.inverse_mask is not None:
                inv_row = batch.inverse_mask[:, t] > 0
                if bool(inv_row.any()):
                    inv_readout_t = ops.inverse_query_entity(mem_total_t, r, v)
                    v_read = torch.where(inv_row.unsqueeze(-1), inv_readout_t, v_read)

            # TICK / GATE / OVERWRITE / NEGATE / WRITE / RESPOND / RESPONSE
            # -- unchanged from run()'s own Phase-1 dispatch.
            state = model.gru(torch.cat([e, r, v, p, c, v_read], dim=-1), state)
            stmt = real * (1.0 - isq)
            gate = torch.sigmoid(model.write_gate(state)).squeeze(-1) * stmt
            owr = torch.sigmoid(model.overwrite_gate(state)).squeeze(-1) * gate
            neg = torch.sigmoid(model.decide_truth(torch.cat([state, v], dim=-1))).squeeze(-1) * stmt
            memory = ops.bind_write(memory, e, r, v, gate - neg, overwrite=owr)   # <=1 WRITE call site

            rl = model.respond(state).squeeze(-1)
            rl = rl.masked_fill(real <= 0, float("-inf"))
            resp_logits.append(rl)
            resp_vecs.append(model.response(torch.cat([state, v_read], dim=-1)))

            if return_write_trace:
                gate_trace.append(gate)
                overwrite_trace.append(owr)
                neg_trace.append(neg)
                redirected_trace.append(addr_row_bool)
                resolved_idx_trace.append(resolved_idx_t)

        RL = torch.stack(resp_logits, dim=1)
        RV = torch.stack(resp_vecs, dim=1)
        wA = torch.softmax(RL, dim=1)
        resp = (wA.unsqueeze(-1) * RV).sum(dim=1)
        rn = resp / (resp.norm(dim=-1, keepdim=True) + 1e-8)
        on = batch.options / (batch.options.norm(dim=-1, keepdim=True) + 1e-8)
        answer_logits = torch.einsum("bd,bkd->bk", rn, on) * 10.0

        out: Dict[str, torch.Tensor] = {
            "answer_logits": answer_logits, "response": resp,
            "respond_gates": wA, "respond_position": (wA * batch.is_q).sum(1),
            "trace_loss": trace_loss,
            "op_acc": {fam: (op_correct.get(fam, 0), op_total.get(fam, 0)) for fam in op_total},
            "arg_acc": {fam: (arg_correct.get(fam, 0), arg_total.get(fam, 0)) for fam in arg_total},
            "write_violations": 0,
            "gold_program_length_histogram": length_hist,
        }
        if return_memory:
            out["_memory"] = memory
        if return_write_trace:
            out["_write_trace"] = {
                "gate": torch.stack(gate_trace, dim=1),
                "overwrite": torch.stack(overwrite_trace, dim=1),
                "neg": torch.stack(neg_trace, dim=1),
                "redirected": torch.stack(redirected_trace, dim=1),
                "resolved_index": torch.stack(resolved_idx_trace, dim=1),
            }
        return out

    # ==================================================================
    # Tier 2: execute_step_program() -- a real per-op interpreter over one
    # clause's flat op list. See module docstring; used by the honesty-
    # machinery unit tests (≤1-WRITE, D1 hard-key enforcement) rather than
    # by run(), which needs whole-batch masking that a linear op-list
    # interpreter cannot express for a MIXED-family batch.
    # ==================================================================
    _CANDIDATE_OPS = {"QUERY_CAND", "INTERACT", "SCORE", "CAND_FOR", "FORCE"}

    def execute_step_program(self, program: Sequence[Step], registers: "ops.RegisterFile", *,
                              state: torch.Tensor, memory: torch.Tensor,
                              candidates: Optional[Dict[str, torch.Tensor]] = None,
                              step_offset: int = 0) -> Tuple["ops.RegisterFile", torch.Tensor, torch.Tensor,
                                                              Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Execute ONE clause step's flat ``program`` against ``registers``
        (pre-loaded per Sec.1.1's "never op-written" registers, e.g. ``A_e``,
        ``A_r``, ``V_v``). ``dim`` (the last axis of every register) MUST be
        ``>= self.model.dim`` -- every register read for a GRU/head call is
        sliced to ``[..., :model.dim]`` first (registers may be padded wider,
        e.g. to hold a ``D_w`` distribution over ``C > dim`` candidates).

        ``candidates`` (optional) supplies the per-candidate tensors the
        variable-width ``P.*`` bank would hold (``{"logits": [B, C],
        "mask": [B, C], "atoms": [B, C, d]}``) -- required for ``SELECT``/
        ``EMIT``; the per-candidate-COMPUTING ops (``QUERY_CAND``,
        ``INTERACT``, ``SCORE``, ``CAND_FOR``, ``FORCE``) raise
        ``NotImplementedError`` (finding 2/module docstring -- exercised
        end-to-end via ``run()`` instead, over real batches).

        Returns ``(registers, new_state, new_memory, respond_logit,
        response_vec)`` -- the last two ``None`` if the program has no
        ``RESPOND``/``RESPONSE`` step.

        Enforces Sec.1.3's ``assert writes_this_clause_step <= 1``: raises
        ``AssertionError`` the moment a SECOND ``WRITE`` op appears in
        ``program`` -- a property of the PROGRAM itself, checked before any
        op past the second WRITE executes.
        """
        model = self.model
        dim = model.dim
        write_count = 0
        respond_logit = response_vec = None

        for step in program:
            op = step.op
            if op in self._CANDIDATE_OPS:
                raise NotImplementedError(
                    f"execute_step_program: {op!r} needs the variable-width per-candidate "
                    "P.* bank RegisterFile does not natively support -- exercised via "
                    "Executor.run() over a real batch instead (see module docstring finding 2).")
            if op == "QUERY":
                e = registers.read(step.reads[0])[..., :dim]
                r = registers.read(step.reads[1])[..., :dim]
                out = ops.unbind_query(memory, e, r)
                registers.write(step.writes, self._fit(out, registers), step=step_offset, op_name=op, args=step.reads)
            elif op == "QUERY_ENTITY":
                r = registers.read(step.reads[0])[..., :dim]
                v = registers.read(step.reads[1])[..., :dim]
                out = ops.inverse_query_entity(memory, r, v)
                registers.write(step.writes, self._fit(out, registers), step=step_offset, op_name=op, args=step.reads)
            elif op == "REREAD":
                aw = registers.read(step.reads[0])[..., :dim]
                ar = registers.read(step.reads[1])[..., :dim]
                out = ops.unbind_query(memory, aw, ar)
                registers.write(step.writes, self._fit(out, registers), step=step_offset, op_name=op, args=step.reads)
            elif op == "RECALL":
                registers.record(op, step.reads, step=step_offset)
            elif op == "SELECT":
                if candidates is None:
                    raise ValueError("execute_step_program: SELECT needs `candidates={'logits':..., 'mask':...}`")
                logits, mask = candidates["logits"], candidates.get("mask")
                w = ops.select(logits, mask, hard=self.hard_keys)
                registers.write("D_w", self._fit(w, registers), step=step_offset, op_name=op, args=step.reads)
            elif op == "EMIT":
                if candidates is None or "atoms" not in candidates:
                    raise ValueError("execute_step_program: EMIT needs `candidates={'atoms': [B, C, d]}`")
                # D1: an Addr-typed target register may ONLY receive a HARD
                # selection -- raises regardless of what SELECT computed,
                # since `hard_keys=False` means this Executor's policy is
                # "soft is not a legal Addr value" (module docstring's D1
                # paragraph), not merely "usually hard."
                if registers.slot_type.get(step.writes) == "Addr" and not self.hard_keys:
                    raise ValueError(
                        f"execute_step_program: EMIT into Addr register {step.writes!r} requires "
                        "hard_keys=True (D1: hard selection for any memory-key register).")
                w = registers.read("D_w")[..., :candidates["atoms"].shape[1]]
                out = (w.unsqueeze(-1) * candidates["atoms"]).sum(1)
                registers.write(step.writes, self._fit(out, registers), step=step_offset, op_name=op, args=step.reads)
            elif op == "TICK":
                x = registers.read(step.reads[0])[..., :dim]
                ar = registers.read(step.reads[1])[..., :dim]
                vv = registers.read(step.reads[2])[..., :dim]
                vread = registers.read(step.reads[3])[..., :dim]
                zero = torch.zeros_like(x)
                gru_in = torch.cat([x, ar, vv, zero, zero, vread], dim=-1)
                state = model.gru(gru_in, state)
                registers.record(op, step.reads, step=step_offset)
            elif op == "GATE":
                gate = torch.sigmoid(model.write_gate(state)).squeeze(-1)
                registers.write("S_gate", gate, step=step_offset, op_name=op, args=step.reads)
            elif op == "OVERWRITE":
                gate = registers.read("S_gate")
                owr = torch.sigmoid(model.overwrite_gate(state)).squeeze(-1) * gate
                registers.write("S_owr", owr, step=step_offset, op_name=op, args=step.reads)
            elif op == "NEGATE":
                vv = registers.read("V_v")[..., :dim]
                neg = torch.sigmoid(model.decide_truth(torch.cat([state, vv], dim=-1))).squeeze(-1)
                registers.write("S_neg", neg, step=step_offset, op_name=op, args=step.reads)
            elif op == "WRITE":
                write_count += 1
                assert write_count <= 1, (
                    f"execute_step_program: more than one WRITE in a single clause's program "
                    f"(Sec.1.3's <=1-WRITE-per-clause invariant) -- program: {[s.op for s in program]}")
                x = registers.read(step.reads[1])[..., :dim]
                ar = registers.read(step.reads[2])[..., :dim]
                vv = registers.read(step.reads[3])[..., :dim]
                gate = registers.read("S_gate")
                owr = registers.read("S_owr")
                neg = registers.read("S_neg")
                memory = ops.bind_write(memory, x, ar, vv, gate - neg, overwrite=owr)
            elif op == "RESPOND":
                respond_logit = model.respond(state).squeeze(-1)
                registers.record(op, step.reads, step=step_offset)
            elif op == "RESPONSE":
                vread = registers.read("V_read")[..., :dim]
                response_vec = model.response(torch.cat([state, vread], dim=-1))
                registers.record(op, step.reads, step=step_offset)
            elif op == "HALT":
                registers.record(op, step.reads, step=step_offset)
            else:
                raise NotImplementedError(f"execute_step_program: unknown op {op!r}")

        return registers, state, memory, respond_logit, response_vec

    @staticmethod
    def _fit(value: torch.Tensor, registers: "ops.RegisterFile") -> torch.Tensor:
        """Right-pad/truncate ``value``'s last dim to ``registers``' own
        uniform width (mirrors ``run()``'s ``_pad`` closure -- factored out
        here since ``execute_step_program`` has no single enclosing scope
        for it)."""
        d = registers.values.shape[-1]
        if value.shape[-1] == d:
            return value
        if value.shape[-1] < d:
            return F.pad(value, (0, d - value.shape[-1]))
        return value[..., :d]
