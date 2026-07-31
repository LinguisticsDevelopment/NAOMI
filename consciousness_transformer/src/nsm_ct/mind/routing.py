"""Learned act-routing (M12) — the controller decides, per clause, *what to do*.

The single door (:meth:`ConsciousLoop.consume`) no longer switches on a clause's
string tag; instead the controller reads each clause's encoding and **emits its
governing act** via the existing ``op_head`` (``out["op_logits"]``): a declarative
clause → ``PERCEIVE`` (absorb), an interrogative clause → ``RESPOND`` / ``RESPOND_VERIFY``
(answer). The mood lives in the predicate slot (``p:?`` vs ``p:is``) — the necessary
"?" marker — and ``is_q`` is never a model input, so the decision is honestly learned
from the clause, not read off a routing flag.

This module encodes a list of clauses into a one-step-per-clause ``ClauseBatch``,
provides the gold act per clause (teacher = the clause's meaning-type), and the CE
routing loss over ``op_logits``. Reuses the ``build_proofsearch_batch`` atom scheme.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..clause_psyche import OPS
from ..clause_reactor import ClauseBatch
from ..reasoning_oracle import Rule

# act vocabulary used for routing (indices into the controller's op_head OPS).
# The learned decision is binary in spirit: WRITE (absorb a declarative) vs RESPOND
# (answer an interrogative). yes/no-vs-wh is a content-shape choice made in consume.
ABSORB = OPS.index("WRITE")                   # declarative → learn it
ANSWER = OPS.index("RESPOND")                 # interrogative → reason & answer
ANSWER_ACTS = (ANSWER,)


def _is_query(clause) -> bool:
    return clause[0] == "query"


def gold_act(clause) -> int:
    """The teacher act for a feed clause (its meaning-type): the label the controller
    learns to predict *from the clause encoding alone*."""
    if clause[0] in ("fact", "disj", "rule"):
        return ABSORB
    if clause[0] == "query":
        return ANSWER
    raise ValueError(f"unknown clause: {clause!r}")


def act_targets(clauses) -> torch.Tensor:
    return torch.tensor([gold_act(c) for c in clauses], dtype=torch.long)


def _repr_literal(clause) -> Tuple[str, str, str, str, bool]:
    """A clause's representative ``(s, r, o, pol, interrogative)`` for encoding.

    A rule is encoded by its consequent (a declarative statement); a wh-query has an
    unknown value; a yes/no query carries its literal. Mood (interrogative) → ``p:?``.
    """
    tag = clause[0]
    if tag == "rule":
        rule = clause[1] if isinstance(clause[1], Rule) else None
        if rule is not None:
            s, r, o, pol = rule.consequent[0], rule.consequent[1], rule.consequent[2], \
                (rule.consequent[3] if len(rule.consequent) > 3 else "+")
        else:                                        # ("rule", ants, cons)
            cons = clause[2]
            s, r, o = cons[0], cons[1], cons[2]
            pol = cons[3] if len(cons) > 3 else "+"
        return (s, r, o, _pol(pol), False)
    if tag == "query" and len(clause) == 3:          # wh: ("query", s, r)
        return (clause[1], clause[2], "?", "+", True)
    if tag == "query":                               # yes/no: ("query", s, r, v, pol)
        return (clause[1], clause[2], clause[3], _pol(clause[4]), True)
    # fact / disj: ("fact", s, r, v, pol?)
    pol = clause[4] if len(clause) > 4 else "+"
    return (clause[1], clause[2], clause[3], _pol(pol), False)


def _pol(x) -> str:
    if isinstance(x, bool):
        return "-" if x else "+"
    return "-" if x == "-" else "+"


def build_routing_batch(clauses, codec) -> ClauseBatch:
    """Encode each clause as a **one-step** ``ClauseBatch`` row (mood in the predicate;
    ``is_q`` left 0 so it is never a model input). ``op_logits[:, 0, :]`` is the act."""
    d = codec.dim
    E = lambda x: codec.filler_vec("e:" + x)
    R = lambda x: codec.filler_vec("r:" + x)
    SV = lambda o, pol: codec.filler_vec(f"v:{pol}:{o}")
    pred_is, pred_q = codec.filler_vec("p:is"), codec.filler_vec("p:?")

    b = len(clauses)
    ent = torch.zeros(b, 1, d); rel = torch.zeros(b, 1, d); val = torch.zeros(b, 1, d)
    prd = torch.zeros(b, 1, d); crd = torch.zeros(b, 1, d)
    is_q = torch.zeros(b, 1); mask = torch.ones(b, 1)
    for i, clause in enumerate(clauses):
        s, r, o, pol, interrog = _repr_literal(clause)
        ent[i, 0] = torch.from_numpy(E(s))
        rel[i, 0] = torch.from_numpy(R(r))
        val[i, 0] = torch.from_numpy(SV(o, pol) if o != "?" else np.zeros(d, np.float32))
        prd[i, 0] = torch.from_numpy(pred_q if interrog else pred_is)
    options = torch.zeros(b, 1, d); answer = torch.zeros(b, dtype=torch.long)
    return ClauseBatch(ent, rel, val, prd, is_q, mask, options, answer, crd, torch.ones(b))


def predict_acts(controller, clauses, codec) -> List[int]:
    """The controller's predicted act per clause (argmax ``op_logits``)."""
    controller.eval()
    with torch.no_grad():
        out = controller(build_routing_batch(clauses, codec))
    return out["op_logits"][:, 0, :].argmax(-1).tolist()


def act_routing_loss(out: dict, targets: torch.Tensor) -> torch.Tensor:
    """CE over the per-clause act logits at the (single) clause step."""
    return F.cross_entropy(out["op_logits"][:, 0, :], targets)


__all__ = [
    "ABSORB", "ANSWER", "ANSWER_ACTS",
    "gold_act", "act_targets", "build_routing_batch", "predict_acts", "act_routing_loss",
]
