"""The cognitive instruction set (M2) — the VM's operations + faithful trace types.

The learned controller (M3) will *emit* a sequence of these ops; a deterministic
executor (:mod:`nsm_ct.mind.executor`) *applies* them to the substrate. Designed
set, learned policy — the ops are fixed and primitive, the choice of which op (and
on what operands) is what the controller learns.

An op is one instruction; a :class:`TraceStep` is one *executed* instruction with
its result and provenance. Because each ``INFER`` step records the ``DerivStep``
support that produced its conclusion, the executed trace is **faithful by
construction** — "I know A, rule A⇒B, therefore B" is the actual derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# -- the instruction set ------------------------------------------------------
PERCEIVE = "PERCEIVE"        # write a stimulus clause into STM (optionally negated)
RECALL = "RECALL"            # read (subject, relation) from STM (+ durable LTM)
INFER = "INFER"              # derive new beliefs (focus-chaining / rule firing) into STM
CONSOLIDATE = "CONSOLIDATE"  # promote settled STM facts into durable LTM
SUPERSEDE = "SUPERSEDE"      # resolve recency/negation for (subject, relation)
RESPOND = "RESPOND"          # emit an answer for (subject, relation), or abstain
RESPOND_VERIFY = "RESPOND_VERIFY"  # verify a polarized query literal → TRUE/FALSE/Unknown (ProofWriter)
HALT = "HALT"                # stop the loop

OPS = (PERCEIVE, RECALL, INFER, CONSOLIDATE, SUPERSEDE, RESPOND, RESPOND_VERIFY, HALT)

# Sentinel answer for derive-or-abstain ("I cannot derive it").
ABSTAIN = "idk"


@dataclass
class Op:
    """One instruction in a (gold or emitted) op-trace.

    Args:
        op: One of :data:`OPS`.
        operands: Op-specific arguments, e.g. ``{"subject": "mary", "relation":
            "PLACE", "value": "kitchen", "negate": False}`` for ``PERCEIVE`` or
            ``{"subject": "mary", "relation": "CAN_SEE"}`` for ``RESPOND``.
    """

    op: str
    operands: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStep:
    """One executed instruction, recorded for inspection / faithfulness.

    Args:
        op: The op name.
        operands: The operands it ran with.
        result: What it produced (a value label, a status, the answer, ...).
        support: Provenance — for ``INFER``/``RESPOND``, the chain of
            ``reasoning_oracle.DerivStep`` that derived the conclusion (``[]`` if
            none / directly stated).
    """

    op: str
    operands: Dict[str, Any]
    result: Any = None
    support: List[Any] = field(default_factory=list)


__all__ = [
    "PERCEIVE", "RECALL", "INFER", "CONSOLIDATE", "SUPERSEDE", "RESPOND",
    "RESPOND_VERIFY", "HALT", "OPS", "ABSTAIN", "Op", "TraceStep",
]
