"""Neural-guided proof search (M10 step 2) — the controller's *learned navigation*.

The controller does not compute derivations (the M8/M9 mistake). It **navigates**:
from the query goal + current facts it **selects which rule to apply next** (the
proven contrastive head over encoded candidate rules), and the deterministic
:class:`~nsm_ct.mind.executor.Executor` applies that one rule **symbolically**
(unification + materialize). Bounded and goal-directed: it stops the moment the
goal (or its negation) enters the closure, far short of full saturation; an
exhausted step budget is the OWA abstain (Unknown).

This is "learn how to think through and navigate meaning using the tools" made
literal: the logic stays in the symbolic engine; only *which move next* is learned.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from ..tpr import TPRCodec
from .controller import MindController
from .datasets import proofwriter as pw
from .executor import Executor


class ProofSearch:
    """Roll out the controller's rule-selection policy over the symbolic executor."""

    def __init__(self, controller: MindController, codec: TPRCodec) -> None:
        self.controller = controller
        self.codec = codec

    def run(self, facts, rules, query, *, max_steps: int = 8) -> Tuple[str, int]:
        """Goal-directed bounded search → ``(verdict, steps_taken)``.

        ``verdict`` ∈ {TRUE, FALSE, UNKNOWN}; UNKNOWN = budget exhausted (no rule the
        policy fired ever closed the goal — the derive-or-abstain case)."""
        ex = Executor(codec=self.codec)
        ex.load_theory(facts, rules)
        s, p, o, qpol = query
        opp = "-" if qpol == "+" else "+"
        if not rules:
            return self._verdict(ex, s, p, o, qpol, opp), 0
        self.controller.eval()
        for step in range(max_steps):
            if (s, p, o, qpol) in ex.pw_closure:
                return pw.TRUE, step
            if (s, p, o, opp) in ex.pw_closure:
                return pw.FALSE, step
            batch = pw.build_proofsearch_batch(
                [(sorted(ex.pw_closure), query, rules, 0)], self.codec)
            with torch.no_grad():
                out = self.controller(batch)
            idx = int(out["answer_logits"].argmax(-1)[0])
            ex.apply_rule(rules[idx])                     # symbolic single-rule move
        return self._verdict(ex, s, p, o, qpol, opp), max_steps

    @staticmethod
    def _verdict(ex, s, p, o, qpol, opp) -> str:
        if (s, p, o, qpol) in ex.pw_closure:
            return pw.TRUE
        if (s, p, o, opp) in ex.pw_closure:
            return pw.FALSE
        return pw.UNKNOWN


__all__ = ["ProofSearch"]
