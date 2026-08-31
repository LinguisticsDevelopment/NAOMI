"""EXECUTOR PHASE 2 (dev/EXECUTOR_DESIGN.md Sec.2, CLAUDE.md's D1/D2
executor decisions): the LEARNED op/argument-selection heads --
``OpSelect`` (categorical, over the op vocabulary) and ``ArgSelect``
(pointer, over the FIXED register file) -- consumed by
:meth:`nsm_ct.executor.Executor.run_learned`.

Scope, read before the classes below (also documented at
``executor.py``'s ``run_learned`` module note, kept in sync): the v0
register table (dev/EXECUTOR_DESIGN.md Sec.1.1, transcribed verbatim as
data in :mod:`nsm_ct.programs`) is, per FAMILY, a fully deterministic flat
``Step`` list with NO within-family branching -- every op, every register
read/written, is pinned the moment a clause step's FAMILY is known. The
only two genuinely AMBIGUOUS, learnable decisions this register table
leaves open at the granularity Phase 1's ``RegisterFile`` actually
supports (no native per-candidate ``P.*`` slot -- Phase 1 finding 2,
``executor.py``'s module docstring) are:

1. **Entry op** -- which of ``{QUERY, CAND_FOR, QUERY_ENTITY}`` a clause
   step's op-loop begins with (families 1/2/3 -> QUERY; 4/6 -> CAND_FOR;
   5 -> QUERY_ENTITY). ``OpSelect`` predicts this from the control signal
   alone. NOTE: this prediction is purely OBSERVATIONAL in
   ``run_learned``'s teacher-forced (Stage-a) training path -- it does not
   gate execution, because ``ClauseReactor.forward``'s (and Phase 1
   ``Executor.run``'s) own arithmetic ALREADY computes the same
   ``ops.unbind_query`` read unconditionally every step regardless of
   family (``programs.py``'s own "the pipeline as one fixed program"
   claim) -- CAND_FOR is a documented pass-through (Phase 1 finding 5),
   and the QUERY_ENTITY/inverse branch is driven by ``batch.inverse_mask``
   (structural DATA, not a "choice" any more than ``is_q`` is). Predicting
   it is still a real, non-trivial classification task (it must recover
   family membership from control-signal-only features -- see
   ``run_learned``'s ``op_acc`` reporting) and IS trace-loss-supervised.
2. **EMIT's destination register** -- ``A_w`` (Addr, address-redirect,
   families 3/4/6) vs ``V_v`` (Vec, value-redirect, family 2) -- Sec.1.1's
   own "EMIT(D_w,P.mem)->V_v OR EMIT(D_w,P.addr)->A_w (never both)".
   Unlike (1), this DOES gate real computation (which resolved candidate
   value lands where the write/re-read machinery reads from next) -- see
   ``run_learned``'s D1 straight-through handling.

Every other op in a family's flat program (``TICK``, ``GATE``,
``OVERWRITE``, ``NEGATE``, ``WRITE``, ``RESPOND``, ``RESPONSE``, ``HALT``,
the per-candidate ``QUERY_CAND``/``INTERACT``/``SCORE``/``SELECT``) reads
and writes registers with NO remaining ambiguity once (1) and (2) above
are fixed -- ``ArgSelect``'s general ``ARG_SIGNATURES`` machinery below
would, for any of them, return their single legal register with
probability 1 (no learned parameter exercised) -- EMIT is the only entry
with more than one candidate slot today. This is not a special case
hand-cut to fit; it is what the FROZEN v0 register table actually
contains (D3, dev/EXECUTOR_DESIGN.md Sec.6) -- extending it is a
Phase-3-or-later, register-file-unfreezing decision, not this milestone's.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import programs as programs_mod

__all__ = [
    "OP_VOCAB", "OP_INDEX", "ENTRY_OPS", "REGISTER_SLOTS", "REGISTER_TYPES",
    "ADDR_TYPED", "ARG_SIGNATURES", "CONTROL_TYPE_ORDER",
    "control_signal_to_tensor", "OpSelect", "ArgSelect", "count_params",
    "mask_second_write", "lofo_keep_mask", "family_balance_weights",
]

# ---------------------------------------------------------------------------
# The op vocabulary -- every op name appearing in programs.GOLD_PROGRAMS's
# Step lists. PROVENANCE/FORCE never appear there by construction (Sec.1.2:
# "not in the learned vocabulary"), so excluding them needs no special-case
# filter -- they were simply never gold-program data to begin with.
# ---------------------------------------------------------------------------
OP_VOCAB: Tuple[str, ...] = tuple(sorted({
    step.op for program in programs_mod.GOLD_PROGRAMS.values() for step in program
}))
OP_INDEX: Dict[str, int] = {op: i for i, op in enumerate(OP_VOCAB)}

# The op-loop's first op, one per family (see module docstring point 1).
ENTRY_OPS: Tuple[str, ...] = ("QUERY", "CAND_FOR", "QUERY_ENTITY")
for _op in ENTRY_OPS:
    assert _op in OP_INDEX, f"op_select.ENTRY_OPS: {_op!r} missing from OP_VOCAB"

# ---------------------------------------------------------------------------
# The FIXED (non-per-candidate) v0 register file -- exactly the slot map
# Executor.run() itself allocates (executor.py's `ops.RegisterFile.create`
# call). ArgSelect points at these by NAME/IDENTITY (a learned embedding
# per slot) -- never their live DATA (dev/EXECUTOR_DESIGN.md Sec.2: "never
# P.addr[i]'s raw identity vectors as key" -- the SAME discipline, applied
# here to the fixed bank since Phase 2 does not implement a per-candidate
# ArgSelect at all, see module docstring).
# ---------------------------------------------------------------------------
REGISTER_SLOTS: Tuple[str, ...] = (
    "A_e", "A_r", "A_w", "V_v", "V_read", "V_ev", "V_desc", "D_w", "S_gate", "S_owr", "S_neg",
)
REGISTER_TYPES: Dict[str, str] = {
    "A_e": "Addr", "A_r": "Addr", "A_w": "Addr",
    "V_v": "Vec", "V_read": "Vec", "V_ev": "Vec", "V_desc": "Vec",
    "D_w": "Dist", "S_gate": "Scalar", "S_owr": "Scalar", "S_neg": "Scalar",
}
ADDR_TYPED = frozenset(name for name, t in REGISTER_TYPES.items() if t == "Addr")

# ARG_SIGNATURES: op -> the (>=1) FIXED register slots ArgSelect chooses
# among for that op's one variable destination -- see module docstring.
ARG_SIGNATURES: Dict[str, Tuple[str, ...]] = {
    "EMIT": ("A_w", "V_v"),
}

CONTROL_TYPE_ORDER: Tuple[str, ...] = ("Addr", "Vec", "Feat", "Scalar", "Dist")


def _broadcast(x, batch: int, device) -> torch.Tensor:
    t = torch.as_tensor(x, device=device)
    if t.dim() == 0:
        t = t.expand(batch).clone()
    return t


def control_signal_to_tensor(ctrl: Dict[str, torch.Tensor], *, batch: int,
                              device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Normalizes :meth:`nsm_ct.executor.Executor.build_control_signal`'s
    dict (whose scalar fields may be plain python ints/0-d tensors OR
    already-batched ``[B]`` tensors -- ``run_learned``'s per-row
    ``prev_op_id``/margin/abstain are ``[B]``, ``step_idx``/
    ``halt_budget_remaining`` are shared clause-step scalars) into
    UNIFORMLY batched tensors -- ``[B]`` for every scalar field, ``[B, 5]``
    for ``type_mask``. CONTROL SIGNALS ONLY (D2) -- every value here is
    either an op id, a step index, a type flag, or a scalar summary; never
    a register data vector.
    """
    type_mask = ctrl["type_mask"]
    if type_mask.dim() == 1:
        type_mask = type_mask.unsqueeze(0).expand(batch, -1)
    return {
        "prev_op_id": _broadcast(ctrl["prev_op_id"], batch, device).long(),
        "step_idx": _broadcast(ctrl["step_idx"], batch, device).long(),
        "type_mask": type_mask.to(device=device).float(),
        "margin": _broadcast(ctrl["margin"], batch, device).float(),
        "abstain_flag": _broadcast(ctrl["abstain_flag"], batch, device).float(),
        "halt_budget_remaining": _broadcast(ctrl["halt_budget_remaining"], batch, device).float(),
    }


class OpSelect(nn.Module):
    """Categorical op-selection head: a small MLP over the D2 control
    signal ONLY (prev-op embedding + step-index embedding + register-type
    mask + scalar summaries) -- NEVER register data. Emits logits over
    :data:`OP_VOCAB` (PROVENANCE/FORCE excluded by construction, see
    module docstring), masked to type/structurally-legal next ops by the
    caller (``run_learned`` passes an entry-op mask; see that method).

    Parameter budget: with the defaults below (``op_embed_dim=8``,
    ``step_embed_dim=4``, ``hidden=16``) and ``k_max=12``,
    ``len(OP_VOCAB)=17``: op-embedding ``18*8=144``, step-embedding
    ``13*4=52``, MLP ``(28*16+16)+(16*17+17)=464+289=753``; total 949
    params -- see :func:`count_params` for the exact count at whatever
    ``k_max``/vocab size a caller actually builds with.
    """

    def __init__(self, *, k_max: int, op_embed_dim: int = 8, step_embed_dim: int = 4,
                 hidden: int = 16) -> None:
        super().__init__()
        self.k_max = k_max
        # +1: embedding index 0 reserved for "no previous op" (clause-loop
        # start) -- OP_INDEX itself is 0-based, so a real op's embedding
        # index is OP_INDEX[op] + 1 everywhere in this module.
        self.op_embed = nn.Embedding(len(OP_VOCAB) + 1, op_embed_dim)
        self.step_embed = nn.Embedding(k_max + 1, step_embed_dim)
        self.ctrl_dim = op_embed_dim + step_embed_dim + len(CONTROL_TYPE_ORDER) + 3
        self.net = nn.Sequential(
            nn.Linear(self.ctrl_dim, hidden), nn.Tanh(), nn.Linear(hidden, len(OP_VOCAB)),
        )

    def encode(self, ctrl: Dict[str, torch.Tensor]) -> torch.Tensor:
        """``ctrl`` -> ``[B, ctrl_dim]``. Asserts the assembled vector's
        width matches ``self.ctrl_dim`` exactly (the "input dimensionality
        matches the control signal exactly" invariant the LOCKED DESIGN
        brief requires) -- a caller that accidentally concatenates a
        register-data vector in here would trip this the moment its width
        changed, not silently."""
        prev = ctrl["prev_op_id"].clamp(min=0, max=len(OP_VOCAB))
        step = ctrl["step_idx"].clamp(min=0, max=self.k_max)
        op_e = self.op_embed(prev)
        step_e = self.step_embed(step)
        vec = torch.cat([
            op_e, step_e, ctrl["type_mask"],
            ctrl["margin"].unsqueeze(-1),
            ctrl["abstain_flag"].unsqueeze(-1),
            (ctrl["halt_budget_remaining"] / max(self.k_max, 1)).unsqueeze(-1),
        ], dim=-1)
        assert vec.shape[-1] == self.ctrl_dim, (
            f"OpSelect.encode: control-signal vector width {vec.shape[-1]} != "
            f"self.ctrl_dim {self.ctrl_dim} -- an op/arg-selection head's input "
            "must be exactly the control signal, never register data.")
        return vec

    def forward(self, ctrl: Dict[str, torch.Tensor], *,
                legal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(logits [B, len(OP_VOCAB)], ctrl_vec [B, ctrl_dim])``
        -- ``ctrl_vec`` is handed back so :class:`ArgSelect` can reuse the
        SAME control encoding for its own query (D2: one control signal
        per op-loop step, not a second one silently rebuilt each head)."""
        ctrl_vec = self.encode(ctrl)
        logits = self.net(ctrl_vec)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask.bool(), -1e9)
        return logits, ctrl_vec


class ArgSelect(nn.Module):
    """Pointer argument-selection head: scores the (>=1) FIXED register
    slots legal for the chosen op's one variable argument
    (:data:`ARG_SIGNATURES`), by register IDENTITY (a learned per-slot
    embedding, never the register's live value) -- ``query =
    MLP([ctrl_vec ; op_embedding])``, ``logits = query . slot_embeddings``.
    D1 (hard Addr, soft-eligible Dist/Scalar): the CALLER
    (``run_learned``) is responsible for straight-through-hardening any
    selection landing in an ``Addr``-typed slot -- this module returns raw
    logits/softmax only (mirrors ``ops.select``'s own ``hard=`` split
    living at the call site, not inside the scorer).

    Shares its op-identity embedding table with an :class:`OpSelect`
    instance (passed in, not reconstructed) -- one op-identity concept,
    not two independently-learned ones; keeps the combined parameter
    budget down too. Parameter budget: with ``slot_embed_dim=8``,
    ``query_hidden=16``, ``ctrl_dim=28`` (this module's own defaults):
    slot-embedding ``11*8=88``, query MLP ``(36*16+16)+(16*8+8)=592+136=728``;
    total 816 (own params only -- the shared ``op_embed`` table's params
    are already counted once, in ``OpSelect``).
    """

    def __init__(self, op_embed: nn.Embedding, *, ctrl_dim: int,
                 slot_embed_dim: int = 8, query_hidden: int = 16) -> None:
        super().__init__()
        self.op_embed = op_embed  # SHARED with OpSelect -- see class docstring
        self.slot_embed = nn.Embedding(len(REGISTER_SLOTS), slot_embed_dim)
        self.slot_index: Dict[str, int] = {n: i for i, n in enumerate(REGISTER_SLOTS)}
        in_dim = ctrl_dim + op_embed.embedding_dim
        self.query = nn.Sequential(
            nn.Linear(in_dim, query_hidden), nn.Tanh(), nn.Linear(query_hidden, slot_embed_dim),
        )

    def slot_ids(self, names: Sequence[str], device=None) -> torch.Tensor:
        return torch.tensor([self.slot_index[n] for n in names], device=device, dtype=torch.long)

    def forward(self, ctrl_vec: torch.Tensor, op_id: torch.Tensor,
                candidate_slots: Sequence[str]) -> torch.Tensor:
        """``ctrl_vec`` ``[B, ctrl_dim]`` (from :meth:`OpSelect.encode`),
        ``op_id`` ``[B]`` long (index into :data:`OP_VOCAB`, the OP being
        argument-selected FOR -- typically ``EMIT``'s own id, broadcast),
        ``candidate_slots`` the legal register names for this op's
        ambiguous argument (:data:`ARG_SIGNATURES`, length >= 1). Returns
        ``logits [B, len(candidate_slots)]``."""
        op_e = self.op_embed(op_id)
        q = self.query(torch.cat([ctrl_vec, op_e], dim=-1))
        ids = self.slot_ids(candidate_slots, device=ctrl_vec.device)
        keys = self.slot_embed(ids)          # [K, E]
        return q @ keys.t()                   # [B, K]


def count_params(*modules: nn.Module) -> int:
    """Total trainable parameter count across ``modules`` -- de-duplicates
    shared submodules (e.g. :class:`ArgSelect`'s shared ``op_embed``
    table) by ``id()`` so passing both an :class:`OpSelect` and the
    :class:`ArgSelect` built from it does not double-count the table."""
    seen: set = set()
    total = 0
    for m in modules:
        for p in m.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
    return total


def lofo_keep_mask(families: Sequence[str], lofo_family: Optional[str]) -> List[bool]:
    """dev/EXECUTOR_DESIGN.md Sec.3's LOFO gate, as a pure, directly
    testable function: ``True`` at row ``i`` means step ``i``'s gold
    family (:func:`nsm_ct.programs.family_of_step`) gets trace
    supervision this call; ``False`` means it is DROPPED entirely (Sec.3:
    "drop that family's trace supervision entirely"). ``lofo_family=None``
    keeps every row (the non-LOFO training arm) -- byte-identical to a
    caller that never passes ``lofo_family`` at all.
    :meth:`nsm_ct.executor.Executor.run_learned` calls this directly
    (rather than re-deriving the same comparison inline) so the splitter's
    behavior is unit-testable in ONE place, independent of the rest of
    that method's tensor plumbing.
    """
    if lofo_family is None:
        return [True] * len(families)
    return [f != lofo_family for f in families]


def family_balance_weights(counts: Dict[str, int], *, power: float = 0.5) -> Dict[str, float]:
    """EXECUTOR LOFO GATE REPAIR #1 (RESEARCH_NOTES "Executor LOFO gate #1"
    instrument defect (1)): inverse-family-frequency per-step weights for
    :meth:`nsm_ct.executor.Executor.run_learned`'s trace cross-entropy.

    Gate #1's diagnosis: ``plain_fact`` is ~87% of every real clause step
    (``family_of_step`` coverage), so the entry-op selector's UNWEIGHTED
    mean cross-entropy is dominated by ``plain_fact``'s own loss and the
    minority families (``definite_desc_read`` n=316, ``inverse_query``
    n=254 out of ~30k in gate #1's own corpus) get gradient too small to
    move their op_acc off 0.000, even with traces present. This is the
    standard CLASS-BALANCED reweighting fix (not a resampled step mix --
    resampling would need restructuring ``train_epoch``'s per-corpus-source
    minibatch loop, which processes one curriculum's own batch per
    optimizer step; a loss weight is a one-line change at the SAME
    granularity trace loss already averages over, and is exactly as
    documented "family/class balancing" in gate #1's instrument-defect
    note).

    ``weight[f] = total * counts[f]**(-power) / sum_g(counts[g]**(1-power))``
    -- chosen so that (a) every family's TOTAL weighted mass across one
    full pass over ``counts`` is IDENTICAL (``weight[f] * counts[f]**power``
    is the SAME constant for every family present, directly unit-testable
    regardless of ``power``), and (b) the GRAND total weighted mass across
    all families equals ``total`` unchanged (``sum(weight[f]*counts[f]) ==
    total`` -- also regardless of ``power``), so this reweighting shifts
    the trace loss's per-family BALANCE without silently rescaling its
    overall magnitude (no separate ``trace_weight`` retuning needed).
    Families with zero count are dropped (never divide by zero); a
    caller's ``.get(fam, 1.0)`` lookup handles a family absent from
    ``counts`` (e.g. an empty minibatch) falling back to unweighted.

    ``power`` (default ``0.5``, inverse-SQUARE-ROOT frequency -- a
    standard NLP class-imbalance dampening, e.g. subsampled-frequent-word
    ratios): ``power=1.0`` is the textbook FULL inverse-frequency scheme
    (every family's total mass exactly equalized) -- measured, at this
    milestone's own smoke scale (``scripts/train_executor.py
    --episodes-per-family 60 --dim 24 --epochs 15 --batch-size 16``), to
    be TOO aggressive: a ~110x max/min weight ratio (``plain_fact``
    n=3105 vs ``inverse_query`` n=28) destabilized the MAJORITY family's
    own convergence inside the smoke battery's short epoch budget
    (``plain_fact`` op_acc fell from 0.998 unweighted to 0.52 at
    ``power=1.0``, even as the minority families rose off 0.000).
    ``power=0.5`` keeps every family within roughly a ``sqrt`` of that
    ratio (~10x here) and converges every family without depressing the
    majority one (this milestone's report has the before/after table).
    """
    counts = {f: c for f, c in counts.items() if c > 0}
    if not counts:
        return {}
    total = sum(counts.values())
    denom = sum(c ** (1.0 - power) for c in counts.values())
    return {f: total * (c ** -power) / denom for f, c in counts.items()}


def mask_second_write(op_sequence: Sequence[str]) -> Tuple[List[str], int]:
    """dev/EXECUTOR_DESIGN.md Sec.1.3's <=1-WRITE-per-clause invariant,
    applied to a LEARNED (possibly wrong) flat op sequence rather than
    raising: every ``"WRITE"`` in ``op_sequence`` AFTER the first is
    replaced with the sentinel ``"_MASKED_WRITE"``. Returns ``(masked,
    violations)``. Contrast with
    :meth:`nsm_ct.executor.Executor.execute_step_program`'s own
    ``assert``-and-raise version of this invariant (Phase 1's honesty
    machinery, tests/test_executor_phase1.py) -- appropriate for a
    hand-built ADVERSARIAL test program; raising on a model's own noisy
    prediction mid-training would just crash the run, so this masks and
    COUNTS instead, the count reported as ``run_learned``'s
    ``write_violations``.
    """
    out = list(op_sequence)
    seen = False
    violations = 0
    for i, op in enumerate(out):
        if op == "WRITE":
            if seen:
                out[i] = "_MASKED_WRITE"
                violations += 1
            else:
                seen = True
    return out, violations
