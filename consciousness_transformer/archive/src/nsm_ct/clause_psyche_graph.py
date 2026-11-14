"""STM / LTM discourse engine over the meaning graph (numpy, deterministic).

The **deterministic substrate** the learned controller (Stage 5) will sit on:

* :class:`STM` — the live discourse subgraph. Clauses are distinct nodes that
  **share referent nodes** (co-reference, not duplication); facts are resolved at
  **read time** by recency / negation / disjunction over the graph, never by
  collapsing contradictions into one slot.
* :class:`LTM` — a persistent, collapsed concept/fact store; ``consolidate_stm``
  folds the settled STM facts into it.

What is deterministic here: the graph ops, co-reference, FALSE-filtering, and the
recency/disjunction resolution. What Stage 5 learns: *when* to write/supersede/
negate/respond and *which* referent is in focus.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .clause import is_entity
from .data_structures import ParseNode, ParseTree
from .meaning_graph import (
    OPERATES_ON,
    SUPERSEDES,
    ClausePayload,
    MeaningGraph,
    apply_operator,
)
from .serialization import serialize_thought
from .tpr import TPRCodec

# read() status codes
RESOLVED = "RESOLVED"
MAYBE = "MAYBE"
UNKNOWN = "UNKNOWN"


class STM:
    """Short-term memory: the live discourse subgraph for one episode."""

    def __init__(self, codec: TPRCodec, resolver) -> None:
        self.codec = codec
        self.resolver = resolver
        self.graph = MeaningGraph(codec)
        self._clock = 0

    # -- writing --------------------------------------------------------------
    def _concept(self, word: str) -> int:
        return self.graph.add_concept(word, self.resolver.resolve(word))

    def _value_node(self, value: str) -> int:
        return self.graph.add_referent(value) if is_entity(value) else self._concept(value)

    def add_clause(
        self, subject: str, relation: str, value: str,
        predicate: str = "is", *, supersede: bool = True,
    ) -> int:
        """Add one fact ``predicate(subject, relation=value)`` as a CLAUSE node."""
        g = self.graph
        subj = g.add_referent(subject)
        pred = self._concept(predicate)
        val = self._value_node(value)

        root = ParseNode(label=predicate, token=predicate)
        root.children.append(ParseNode(label="SOMEONE", token=subject, relation="SUBJECT"))
        root.children.append(ParseNode(
            label="SOMEONE" if is_entity(value) else "SOMEWHERE", token=value, relation=relation))
        tree = ParseTree(root=root, text=f"{subject} {predicate} {value}")
        handle = self.codec.contract(self.codec.encode_matrix(tree.root))

        self._clock += 1
        payload = ClausePayload(predicate_nid=pred, slots=[("SUBJECT", subj), (relation, val)])
        nid = g.add_clause(
            payload, handle, structure=serialize_thought(tree),
            meta={"recency": self._clock, "truth": "TRUE"},
        )
        if supersede:
            for prior in self._facts_about(subj, relation):
                if prior != nid:
                    g.add_edge(SUPERSEDES, nid, prior)
        return nid

    def add_disjunction(
        self, subject: str, relation: str, values: List[str], predicate: str = "is",
    ) -> Tuple[int, List[int]]:
        """Add ``subject relation A or B ...`` — disjuncts wrapped in a MAYBE node."""
        nids = [self.add_clause(subject, relation, v, predicate, supersede=False)
                for v in values]
        op = apply_operator(self.graph, "MAYBE", nids, self.codec, set_false=False)
        return op, nids

    def negate(self, clause_nid: int) -> int:
        """Negate a clause (NOT operator-node + FALSE truth tag)."""
        return apply_operator(self.graph, "NOT", clause_nid, self.codec)

    # -- reading (resolution) -------------------------------------------------
    def _value_of(self, clause_nid: int, relation: str) -> Optional[int]:
        for rel, filler_nid in self.graph.payloads[clause_nid].slots:
            if rel == relation:
                return filler_nid
        return None

    def _facts_about(self, subj_nid: int, relation: str) -> List[int]:
        return [c for c in self.graph.clauses_about(subj_nid)
                if self._value_of(c, relation) is not None]

    def _uncertain(self, clause_nid: int) -> bool:
        return any(self.graph.node(e.src).label == "MAYBE"
                   for e in self.graph.in_(clause_nid, OPERATES_ON))

    def read(self, subject: str, relation: str, ltm: Optional["LTM"] = None):
        """Resolve ``(subject, relation)`` → ``(status, value_nid)``.

        Order: drop FALSE (negated) clauses; an affirmed clause wins by recency;
        a pure disjunction with >1 surviving value is MAYBE; a disjunction narrowed
        to one surviving value resolves to it. STM-first, then optional LTM.
        """
        subj = self.graph.referent_index.get(subject.lower())
        if subj is None:
            return ltm.recall(subject, relation) if ltm is not None else (UNKNOWN, None)
        survivors = [c for c in self._facts_about(subj, relation)
                     if self.graph.node(c).meta.get("truth") != "FALSE"]
        if not survivors:
            return ltm.recall(subject, relation) if ltm is not None else (MAYBE, None)
        affirmed = [c for c in survivors if not self._uncertain(c)]
        if affirmed:
            best = max(affirmed, key=lambda c: self.graph.node(c).meta.get("recency", 0))
            return RESOLVED, self._value_of(best, relation)
        values = {self.graph.node(self._value_of(c, relation)).label for c in survivors}
        if len(values) == 1:
            return RESOLVED, self._value_of(survivors[0], relation)
        return MAYBE, None

    def value_label(self, value_nid: Optional[int]) -> Optional[str]:
        return None if value_nid is None else self.graph.node(value_nid).label


class LTM:
    """A persistent, collapsed fact/concept store (minimal — see plan scope)."""

    def __init__(self, codec: TPRCodec) -> None:
        self.codec = codec
        self.graph = MeaningGraph(codec)          # persistent concept graph
        self.facts: dict = {}                     # (subject, relation) -> value label

    def consolidate_stm(self, stm: STM) -> int:
        """Fold STM's settled (resolved, affirmed) facts into the store."""
        added = 0
        for subj_name, subj in stm.graph.referent_index.items():
            relations = {rel for c in stm.graph.clauses_about(subj)
                         for rel, _ in stm.graph.payloads[c].slots if rel != "SUBJECT"}
            for relation in relations:
                status, val = stm.read(subj_name, relation)
                if status == RESOLVED:
                    self.facts[(subj_name, relation)] = stm.value_label(val)
                    added += 1
        return added

    def recall(self, subject: str, relation: str):
        v = self.facts.get((subject.lower(), relation))
        return (RESOLVED, v) if v is not None else (MAYBE, None)
