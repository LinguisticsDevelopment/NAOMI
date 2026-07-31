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

from ..reasoning_oracle import _is_var, ground, unify
from ..tpr import TPRCodec
from .controller import MindController
from .datasets import proofwriter as pw
from .executor import Executor


def backward_step(subgoal, rule):
    """The symbolic BACKWARD move: can ``rule`` prove ``subgoal``? Unify the rule's
    consequent with the (ground) subgoal; if it matches, return the rule's antecedents
    as the new subgoals (grounded by the binding) — else ``None``. This is the engine
    (unification), not the weights; the controller only *chose* the rule."""
    theta = unify(rule.consequent, subgoal)
    if theta is None:
        return None
    subs = [ground(ant, theta) for ant in rule.antecedents]
    if any(_is_var(x) for s in subs for x in s):     # v1: ground subgoals only
        return None
    return subs


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

    def collect_dagger(self, items, *, max_steps: int = 6) -> List[Tuple]:
        """DAgger: roll out the *current* policy and, at every state it actually
        visits, emit a training example labeled with the EXPERT's recovery move
        (:func:`pw.expert_action`). This is what cures exposure bias — the policy
        learns the right move from the off-gold-path states its own mistakes create.

        Returns examples shaped exactly like :func:`pw.navigation_examples`
        (``(current_facts, query, rules, expert_idx)``). Unknown items have no proof
        plan and are skipped (the budget handles their abstention)."""
        self.controller.eval()
        out: List[Tuple] = []
        for (facts, rules, query, _a, _d) in items:
            needed, rule_of, label = pw.gold_plan(facts, rules, query)
            if not needed:                                # Unknown — nothing to navigate
                continue
            ex = Executor(codec=self.codec)
            ex.load_theory(facts, rules)
            s, p, o, qpol = query
            opp = "-" if qpol == "+" else "+"
            for _step in range(max_steps):
                if (s, p, o, qpol) in ex.pw_closure or (s, p, o, opp) in ex.pw_closure:
                    break
                expert = pw.expert_action(ex.pw_closure, needed, rule_of)
                if expert is None:                        # goal's literals all present
                    break
                out.append((sorted(ex.pw_closure), query, rules, expert))
                batch = pw.build_proofsearch_batch(
                    [(sorted(ex.pw_closure), query, rules, 0)], self.codec)
                with torch.no_grad():
                    idx = int(self.controller(batch)["answer_logits"].argmax(-1)[0])
                ex.apply_rule(rules[idx])                 # advance on the POLICY's move
        return out


class BackwardSearch:
    """Goal-directed BACKWARD navigation with the SAME controller + contrastive head.

    Forward search makes the depth-2 *first* move (a non-goal-matching intermediate)
    near-impossible for the learned 1-hop heuristic. Backward search decomposes the
    proof into a chain of 1-hop decisions the policy already does perfectly: "to prove
    this subgoal, pick the rule whose consequent matches it." The fixed STM facts are
    the context; the changing **subgoal** sits in the ``is_q`` slot of the *unchanged*
    selection batch. The engine (unification) expands each chosen rule; only *which
    subgoal to expand next* is learned."""

    def __init__(self, controller: MindController, codec: TPRCodec) -> None:
        self.controller = controller
        self.codec = codec

    def _prove(self, facts_set, rules, goal, max_steps) -> Tuple[bool, int]:
        """Bounded backward proof of ``goal`` → ``(proved, steps)``. A subgoal in STM
        is discharged; otherwise the controller picks a rule and the engine expands it
        backward. A non-matching pick (or budget exhaustion) fails the branch."""
        stack = [goal]
        steps = 0
        while stack:
            if steps >= max_steps:
                return False, steps
            g = stack.pop()
            if g in facts_set:                            # known fact — discharged
                continue
            batch = pw.build_proofsearch_batch([(sorted(facts_set), g, rules, 0)], self.codec)
            with torch.no_grad():
                idx = int(self.controller(batch)["answer_logits"].argmax(-1)[0])
            subs = backward_step(g, rules[idx])           # engine expands the chosen rule
            steps += 1
            if subs is None:                              # picked a rule that can't prove g
                return False, steps
            stack.extend(subs)
        return True, steps                                # every subgoal reduced to facts

    def run(self, facts, rules, query, *, max_steps: int = 8) -> Tuple[str, int]:
        """Verdict by backward proof: prove the query → TRUE; else prove its negation →
        FALSE; else UNKNOWN (budget/branch exhausted — the OWA abstain)."""
        self.controller.eval()
        facts_set = set(facts)
        s, p, o, qpol = query
        opp = "-" if qpol == "+" else "+"
        if not rules:
            if query in facts_set:
                return pw.TRUE, 0
            if (s, p, o, opp) in facts_set:
                return pw.FALSE, 0
            return pw.UNKNOWN, 0
        proved, st1 = self._prove(facts_set, rules, query, max_steps)
        if proved:
            return pw.TRUE, st1
        neg, st2 = self._prove(facts_set, rules, (s, p, o, opp), max_steps)
        if neg:
            return pw.FALSE, st1 + st2
        return pw.UNKNOWN, st1 + st2


__all__ = ["ProofSearch", "BackwardSearch", "backward_step"]
