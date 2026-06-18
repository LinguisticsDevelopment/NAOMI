"""The deterministic executor — the VM the learned controller will govern (M2).

The cognitive instruction set (:mod:`nsm_ct.mind.ops`) realized as **deterministic
operations** over the substrate: a live :class:`~nsm_ct.clause_psyche_graph.STM`
(working memory, with built-in recency/negation/disjunction resolution) plus a
durable :class:`~nsm_ct.mind.knowledge.KnowledgeGraph` (LTM rules + facts, from M1).
Nothing here is learned; an op-trace is supplied (hand-written gold traces at M2,
the learned controller's emitted trace at M3).

``INFER`` is **focus-chaining traversal realized exactly**: it forward-chains over
(STM facts ∪ LTM facts ∪ LTM rules) and **materializes** each derived triple back
into STM. ``reasoning_oracle.forward_chain`` returns the per-step ``DerivStep``
chain — the unrolled traversal itself (each step = one edge-follow / rule firing
with ``support`` provenance). The §0n *vector* focus-chaining
(``entity_memory.query``, parameter-free) is the *learned realization* of this same
mechanism in M3: identical traversal, exact symbolic substrate now vs. learned
vector substrate later.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..clause_psyche_graph import MAYBE, RESOLVED, STM
from ..data_structures import ParseNode, ParseTree
from ..reasoning_oracle import Triple, forward_chain
from ..tpr import TPRCodec
from . import ops
from .knowledge import KnowledgeGraph


class _TrivialResolver:
    """Minimal meaning resolver: word → a single-leaf tree labelled by the word.

    The symbolic substrate only needs node *labels* (which come from the token);
    the lossy handle is unused for symbolic derivation. This keeps the executor
    runnable with no nltk/WordNet dependency.
    """

    def resolve(self, word: str) -> ParseTree:  # noqa: D401 - simple
        return ParseTree(root=ParseNode(label=word, token=word))


class Executor:
    """Run cognitive ops over an episode's STM and a durable LTM knowledge graph.

    Args:
        ltm: The durable :class:`KnowledgeGraph` (rules + facts). A fresh one is
            built if omitted.
        codec: TPR codec; reused for both STM and LTM. Defaults to the LTM's codec.
        resolver: Word→meaning resolver for STM concept nodes (a trivial one by
            default — labels are all the symbolic path needs).
    """

    def __init__(
        self,
        ltm: Optional[KnowledgeGraph] = None,
        *,
        codec: Optional[TPRCodec] = None,
        resolver=None,
    ) -> None:
        self.ltm = ltm or KnowledgeGraph(codec=codec)
        self.codec = codec or self.ltm.codec
        self.resolver = resolver or _TrivialResolver()
        self.stm = STM(self.codec, self.resolver)

    # -- locating clauses (for negation / disjunction) -----------------------
    def _matching_clause(self, subject: str, relation: str, value: str) -> Optional[int]:
        subj = self.stm.graph.referent_index.get(subject.lower())
        if subj is None:
            return None
        for c in self.stm._facts_about(subj, relation):
            fnid = self.stm._value_of(c, relation)
            if fnid is not None and self.stm.graph.node(fnid).label == value.lower():
                return c
        return None

    def _has_evidence(self, subject: str, relation: str) -> bool:
        """True if STM holds any clause for (subject, relation) — distinguishes a
        real (uncertain) disjunction from a never-asserted query (→ abstain)."""
        subj = self.stm.graph.referent_index.get(subject.lower())
        return subj is not None and bool(self.stm._facts_about(subj, relation))

    # -- the ops -------------------------------------------------------------
    def perceive(
        self, subject: str, relation: str, value: str,
        *, negate: bool = False, predicate: str = "is",
    ) -> ops.TraceStep:
        """Write a stimulus clause into STM. ``negate=True`` marks the matching
        clause FALSE (find-or-add it first), realizing "X is not in Y"."""
        if negate:
            nid = self._matching_clause(subject, relation, value)
            if nid is None:
                nid = self.stm.add_clause(subject, relation, value, predicate)
            self.stm.negate(nid)
            result = "FALSE"
        else:
            nid = self.stm.add_clause(subject, relation, value, predicate)
            result = "TRUE"
        return ops.TraceStep(
            ops.PERCEIVE,
            {"subject": subject, "relation": relation, "value": value, "negate": negate},
            result=result,
        )

    def perceive_disjunction(
        self, subject: str, relation: str, values: List[str], predicate: str = "is",
    ) -> ops.TraceStep:
        """Write ``subject relation A or B`` (an uncertain disjunction) into STM."""
        self.stm.add_disjunction(subject, relation, values, predicate)
        return ops.TraceStep(
            ops.PERCEIVE,
            {"subject": subject, "relation": relation, "values": list(values),
             "disjunction": True},
            result="MAYBE",
        )

    def recall(self, subject: str, relation: str) -> ops.TraceStep:
        """Read (subject, relation) from STM → (status, value label)."""
        status, val_nid = self.stm.read(subject, relation)
        return ops.TraceStep(
            ops.RECALL, {"subject": subject, "relation": relation},
            result=(status, self.stm.value_label(val_nid)),
        )

    def _stm_facts(self) -> List[Triple]:
        """Every currently-resolved STM fact as a ``(subject, relation, value)``."""
        out: List[Triple] = []
        for subj_name, subj in self.stm.graph.referent_index.items():
            relations = {rel for c in self.stm.graph.clauses_about(subj)
                         for rel, _ in self.stm.graph.payloads[c].slots
                         if rel != "SUBJECT"}
            for relation in relations:
                status, val_nid = self.stm.read(subj_name, relation)
                if status == RESOLVED:
                    out.append((subj_name, relation, self.stm.value_label(val_nid)))
        return out

    def infer(self, max_iters: int = 16) -> ops.TraceStep:
        """Derive new beliefs (focus-chaining / rule firing) and materialize them.

        Forward-chains over (STM facts ∪ LTM facts ∪ LTM rules); each newly derived
        triple is written back into STM as a clause, so the conclusion is readable
        and the ``DerivStep`` chain is the faithful traversal record.
        """
        facts = self._stm_facts() + self.ltm.facts()
        known, chain = forward_chain(facts, self.ltm.rules(), max_iters=max_iters)
        known_now = set(self._stm_facts())
        for (e, r, v) in known:
            if (e, r, v) not in known_now and r != "SUBJECT":
                self.stm.add_clause(e, r, v, supersede=False)
        return ops.TraceStep(ops.INFER, {}, result=[s.derived for s in chain], support=list(chain))

    def supersede(self, subject: str, relation: str) -> ops.TraceStep:
        """Resolve recency/negation for (subject, relation) (records the resolution)."""
        status, val_nid = self.stm.read(subject, relation)
        return ops.TraceStep(
            ops.SUPERSEDE, {"subject": subject, "relation": relation},
            result=(status, self.stm.value_label(val_nid)),
        )

    def consolidate(self) -> ops.TraceStep:
        """Promote settled (resolved) STM facts into durable LTM."""
        promoted = self._stm_facts()
        for (e, r, v) in promoted:
            self.ltm.add_fact(e, r, v)
        return ops.TraceStep(ops.CONSOLIDATE, {}, result=len(promoted))

    def respond(self, subject: str, relation: str) -> ops.TraceStep:
        """Emit the answer for (subject, relation), ``"maybe"``, or **abstain**.

        ``RESOLVED`` → the value; an uncertain disjunction with evidence →
        ``"maybe"``; a durable LTM derivation → its value; otherwise → ABSTAIN.
        """
        status, val_nid = self.stm.read(subject, relation)
        operands = {"subject": subject, "relation": relation}
        if status == RESOLVED:
            return ops.TraceStep(ops.RESPOND, operands, result=self.stm.value_label(val_nid))
        if status == MAYBE and self._has_evidence(subject, relation):
            return ops.TraceStep(ops.RESPOND, operands, result="maybe")
        # durable fallback: can LTM derive it on its own?
        value, chain = self.ltm.derive(subject, relation)
        if value is not None:
            return ops.TraceStep(ops.RESPOND, operands, result=value, support=list(chain))
        return ops.TraceStep(ops.RESPOND, operands, result=ops.ABSTAIN)

    def halt(self) -> ops.TraceStep:
        return ops.TraceStep(ops.HALT, {})

    # -- driver --------------------------------------------------------------
    _DISPATCH = {
        ops.PERCEIVE: "_run_perceive",
        ops.RECALL: "recall",
        ops.INFER: "infer",
        ops.CONSOLIDATE: "consolidate",
        ops.SUPERSEDE: "supersede",
        ops.RESPOND: "respond",
        ops.HALT: "halt",
    }

    def _run_perceive(self, **kw) -> ops.TraceStep:
        if kw.get("disjunction") or "values" in kw:
            return self.perceive_disjunction(
                kw["subject"], kw["relation"], kw["values"], kw.get("predicate", "is"))
        return self.perceive(
            kw["subject"], kw["relation"], kw["value"],
            negate=kw.get("negate", False), predicate=kw.get("predicate", "is"))

    def run_trace(self, trace: List[ops.Op]) -> Dict[str, object]:
        """Execute a gold/emitted op-trace; return the answer + recorded steps.

        Returns ``{"answer", "abstained", "trace"}`` where ``answer`` is the last
        ``RESPOND`` result and ``trace`` is the list of executed :class:`TraceStep`.
        """
        steps: List[ops.TraceStep] = []
        answer: Optional[object] = None
        for op in trace:
            method = getattr(self, self._DISPATCH[op.op])
            step = method(**op.operands)
            steps.append(step)
            if op.op == ops.RESPOND:
                answer = step.result
            if op.op == ops.HALT:
                break
        return {"answer": answer, "abstained": answer == ops.ABSTAIN, "trace": steps}


__all__ = ["Executor"]
