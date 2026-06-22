"""The conscious loop — stimulus-driven reactive reasoning (M4 + M11).

The single runtime door is :meth:`ConsciousLoop.consume`: a **feed of tagged
meaning clauses** flows in, and the loop decides *per clause, from the clause's own
form* what to do — absorb a `fact`/`rule` into working memory (learning by graph
write), or, when a clause is a `query`, **reason over everything accumulated so
far** (the M10 backward search) and *derive* the answer. Nothing is told "this is
the question," and no answer options are supplied; the question is interrogative in
its own meaning-type and the answer is produced, not selected.

`recall`/`respond` remain as internal/legacy helpers. Knowledge enters only via the
feed (working theory) and the graph; the controller's weights are never touched here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..clause_reactor import build_clause_batch
from ..reasoning_oracle import Rule, forward_chain
from ..tpr import TPRCodec
from . import ops, teacher
from .controller import MindController, emit_op_trace
from .datasets import proofwriter as pw
from .executor import Executor, _TrivialResolver
from .knowledge import KnowledgeGraph
from .proof_search import BackwardSearch


def _norm_pol(x) -> str:
    """Normalize a polarity field (bool negate flag or '+'/'-' string) to '+'/'-'."""
    if isinstance(x, bool):
        return "-" if x else "+"
    return "-" if x == "-" else "+"


def _norm_fact(clause) -> Tuple[str, str, str, str]:
    """A feed `fact` object → a 4-tuple `(s, r, v, pol)`. Accepts the membrane's
    `("fact", s, r, v, negate)` and the adapter's `("fact", s, r, v, pol)`."""
    _, s, r, v = clause[:4]
    pol = _norm_pol(clause[4]) if len(clause) > 4 else "+"
    return (s, r, v, pol)


def _norm_lit(t) -> Tuple[str, str, str, str]:
    return (t[0], t[1], t[2], _norm_pol(t[3]) if len(t) > 3 else "+")


def _norm_rule(clause) -> Rule:
    """A feed `rule` object → a :class:`Rule` of 4-tuples. Accepts `("rule", Rule)`
    and `("rule", antecedents, consequent)` (3- or 4-tuple patterns)."""
    if isinstance(clause[1], Rule):
        return clause[1]
    _, ants, cons = clause
    return Rule(tuple(_norm_lit(a) for a in ants), _norm_lit(cons), name="fed")


class ConsciousLoop:
    """Reactive loop over one STM episode + a durable LTM knowledge graph.

    Args:
        ltm: The durable :class:`KnowledgeGraph` (recall source / write-back target).
        controller: The learned :class:`MindController` (runtime reasoner). Optional —
            the deterministic ``recall`` path works without it.
        codec: TPR codec (defaults to the LTM's).
        resolver: Word→meaning resolver for encoding (trivial, dependency-free default).
    """

    def __init__(self, ltm: KnowledgeGraph, *, controller: Optional[MindController] = None,
                 codec: Optional[TPRCodec] = None, resolver=None) -> None:
        self.ltm = ltm
        self.codec = codec or ltm.codec
        self.controller = controller
        self.resolver = resolver or _TrivialResolver()

    # -- THE single door: a clause feed in, self-routed responses out --------
    def consume(self, feed) -> List[Dict[str, object]]:
        """Process a feed of tagged meaning clauses through one door.

        Each clause is routed by its **own meaning-type** (not an external label):
          * ``("fact", s, r, v, pol?)`` / ``("disj", …)`` → learn it into working memory
          * ``("rule", Rule)`` / ``("rule", ants, cons)`` → learn the rule
          * ``("query", s, r, v, pol)`` → yes/no: reason (``BackwardSearch``) → TRUE/FALSE/Unknown
          * ``("query", s, r)``         → wh: derive the value over the accumulated theory

        Returns one response dict per query, in order; learned facts/rules persist
        across the feed (a later query uses what earlier clauses taught). No weights
        are updated. ``self.facts`` / ``self.rules`` hold the accumulated theory.
        """
        self.facts: List[Tuple[str, str, str, str]] = []
        self.rules: List[Rule] = []
        searcher = BackwardSearch(self.controller, self.codec) if self.controller else None
        responses: List[Dict[str, object]] = []
        for clause in feed:
            tag = clause[0]
            if tag in ("fact", "disj"):
                self.facts.append(_norm_fact(clause))
            elif tag == "rule":
                self.rules.append(_norm_rule(clause))
            elif tag == "query":
                responses.append(self._answer(clause, searcher))
            else:
                raise ValueError(f"unknown feed clause: {clause!r}")
        return responses

    def _answer(self, clause, searcher) -> Dict[str, object]:
        """Answer one query clause by reasoning over the accumulated theory."""
        if len(clause) >= 5:                              # ("query", s, r, v, pol) — yes/no
            s, r, v, pol = clause[1], clause[2], clause[3], _norm_pol(clause[4])
            query = (s, r, v, pol)
            if searcher is None:
                raise ValueError("yes/no queries need a learned controller")
            verdict, nsteps = searcher.run(self.facts, self.rules, query,
                                           max_steps=8)
            return {"query": query, "answer": verdict, "steps": nsteps}
        _, s, r = clause                                  # ("query", s, r) — wh (derive a value)
        known, chain = forward_chain(list(self.facts), self.rules)
        for (fs, fr, fo, fp) in sorted(known):
            if fs == s and fr == r and fp == "+":
                return {"query": (s, r), "answer": fo, "chain": chain}
        return {"query": (s, r), "answer": ops.ABSTAIN, "chain": []}

    # -- durable recall (graph read; no weights) -----------------------------
    def recall(self, subject: str, relation: str):
        """Answer ``(subject, relation)`` from durable LTM — a graph read, no weights.

        Returns ``(value | None, chain)``; ``None`` ⇒ not derivable (abstain).
        """
        return self.ltm.derive(subject, relation)

    # -- learned reactive answer ---------------------------------------------
    def respond(self, episode) -> Dict[str, object]:
        """Run the learned controller on an episode → answer + faithful op-trace."""
        if self.controller is None:
            raise ValueError("respond() needs a learned controller; use recall() otherwise")
        batch = build_clause_batch([episode], None, self.resolver, self.codec)
        self.controller.eval()
        import torch
        with torch.no_grad():
            out = self.controller(batch)
        abstain = bool(out["abstain_prob"][0] >= 0.5)
        idx = int(out["answer_logits"].argmax(-1)[0])
        answer = ops.ABSTAIN if abstain else episode.options[idx]
        query = tuple(episode.meta.get("query", (None, None)))
        trace = emit_op_trace(out, self.controller.relation_codebook, 0, query)
        return {"answer": answer, "abstain": abstain, "trace": trace}

    # -- symbolic validator (cross-check) ------------------------------------
    def validate(self, episode) -> Dict[str, object]:
        """Symbolic ground-truth answer for an episode (the validator cross-check)."""
        return teacher.replay(episode, dim=self.codec.dim)


__all__ = ["ConsciousLoop"]
