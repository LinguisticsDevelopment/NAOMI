"""The knowledge layer — LTM as one durable meaning graph (M1).

This is where "what is known" lives **entirely in the graph** (the core
invariant: no information in weights). A :class:`KnowledgeGraph` wraps the
existing :class:`~nsm_ct.meaning_graph.MeaningGraph` and stores three kinds of
content as graph data:

* **Facts** — ``(subject, relation, value)`` asserted as CLAUSE nodes
  (truth-tagged), e.g. ``(robin, IS_A, bird)`` / ``(bird, CAN, fly)``.
* **Taxonomy** — is-a / KIND facts (same machinery; the conceptual scaffolding).
* **Variable-bearing rules** — Horn rules stored as graph data with **variable
  slots** (referent nodes whose label begins with ``?``), e.g.
  ``(?x IS_A ?y) ∧ (?y CAN ?z) ⇒ (?x CAN ?z)``.

Reasoning lifts the existing forward-chaining unifier out of the grading oracle
(:mod:`nsm_ct.reasoning_oracle`) and runs it over **graph-resident** facts +
rules: :meth:`derive` reproduces the oracle's answers, but the rules now live in
the (persistable) graph rather than as Python constants. *Learned later:* which
rule to retrieve when. *Never learned:* the rule content or the binding mechanism.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..meaning_graph import DEFINES, ClausePayload, MeaningGraph, NodeKind
from ..reasoning_oracle import Rule, Triple, derive as _oracle_derive
from ..tpr import TPRCodec
from . import persistence
from . import schema

_FACT = "fact"
_RULE = "rule"
_ANTECEDENT = "antecedent"
_CONSEQUENT = "consequent"


class KnowledgeGraph:
    """A durable, persistable graph of facts + taxonomy + variable-bearing rules.

    Args:
        codec: The TPR codec (its ``dim`` fixes the handle space). A fresh one is
            built if omitted.
        dim: Handle dimension when constructing a fresh codec.
    """

    def __init__(self, codec: Optional[TPRCodec] = None, *, dim: int = 256) -> None:
        self.codec = codec or TPRCodec(dim=dim)
        self.graph = MeaningGraph(self.codec)
        self._rule_seq = 0  # unique-id counter so rules sharing a name stay distinct
        self._predicate_nid = self._concept("is")  # shared predicate node

    # -- term nodes (co-reference-deduped) -----------------------------------
    def _referent(self, name: str) -> int:
        """A REFERENT node for an entity or a ``?var`` (one per name)."""
        return self.graph.add_referent(name)

    def _concept(self, label: str) -> int:
        """A CONCEPT node for a content word (one per label, deduped + persisted)."""
        existing = self.graph.concept_index.get(label)
        if existing is not None:
            return existing
        nid = self.graph.add_node(
            NodeKind.CONCEPT, self.codec.filler_vec("concept:" + label), label=label,
        )
        self.graph.concept_index[label] = nid
        return nid

    def _term(self, label: str) -> int:
        """A node for a triple term: referent for entities/variables, else concept."""
        if schema.is_variable(label):
            return self._referent(label)
        # Heuristic: a single lowercased token that is not a known content concept
        # is treated as a referent (entity); everything else as a concept. For the
        # M1 reasoning gate only label recovery matters, so this stays simple.
        return self._referent(label)

    # -- writing facts -------------------------------------------------------
    def add_fact(
        self, subject: str, relation: str, value: str, *, truth: str = "TRUE",
        kind: str = _FACT, meta_extra: Optional[Dict[str, object]] = None,
    ) -> int:
        """Assert ``(subject, relation, value)`` as a truth-tagged CLAUSE node."""
        subj = self._term(subject)
        val = self._term(value)
        handle = self.codec.filler_vec(f"{kind}:{subject}:{relation}:{value}")
        payload = ClausePayload(
            predicate_nid=self._predicate_nid,
            slots=[(schema.SUBJECT_ROLE, subj), (relation, val)],
        )
        meta: Dict[str, object] = {"truth": truth, "kind": kind}
        if meta_extra:
            meta.update(meta_extra)
        return self.graph.add_clause(payload, handle, meta=meta)

    def assert_facts(self, triples: List[Triple]) -> List[int]:
        """Assert many ``(subject, relation, value)`` triples; return their nids."""
        return [self.add_fact(s, r, v) for (s, r, v) in triples]

    # -- writing rules (as graph data) ---------------------------------------
    def add_rule(self, rule: Rule) -> List[int]:
        """Store a Horn rule as graph data — one CLAUSE node per pattern triple.

        Each antecedent / the consequent is a CLAUSE node tagged with the rule
        name + role + order, so the rule is reconstructable from the graph alone.
        Variables (``?x``) are ordinary REFERENT nodes whose label marks them.
        """
        uid = f"{rule.name}#{self._rule_seq}"  # distinct even when names collide (e.g. "mp")
        self._rule_seq += 1
        nids: List[int] = []
        for i, (s, r, v) in enumerate(rule.antecedents):
            nids.append(self.add_fact(
                s, r, v, kind=_RULE,
                meta_extra={"rule": uid, "rule_name": rule.name, "role": _ANTECEDENT, "order": i},
            ))
        cs, cr, cv = rule.consequent
        nids.append(self.add_fact(
            cs, cr, cv, kind=_RULE,
            meta_extra={"rule": uid, "rule_name": rule.name, "role": _CONSEQUENT, "order": 0},
        ))
        return nids

    def add_rules(self, rules: List[Rule]) -> None:
        for rule in rules:
            self.add_rule(rule)

    # -- definition (DEFINES edge) -------------------------------------------
    def define(self, concept_label: str, prime_names: List[str]) -> int:
        """Wire ``DEFINES``: a concept → its explication (a bag of NSM primes).

        Closes the declared-but-unused ``DEFINES`` edge: the explication is stored
        as its own node carrying the prime structure, linked from the concept.
        """
        concept = self._concept(concept_label)
        exp = self.graph.add_node(
            NodeKind.CONCEPT,
            self.codec.filler_vec("explication:" + concept_label),
            structure=list(prime_names),
            label=concept_label + "/def",
            meta={"kind": "explication", "primes": list(prime_names)},
        )
        self.graph.add_edge(DEFINES, concept, exp)
        return exp

    # -- reading: extract triples / rules back out of the graph --------------
    def _triple_of(self, clause_nid: int) -> Optional[Triple]:
        payload = self.graph.payloads.get(clause_nid)
        if payload is None:
            return None
        subject = value = relation = None
        for rel, fnid in payload.slots:
            if rel == schema.SUBJECT_ROLE:
                subject = self.graph.node(fnid).label
            else:
                relation, value = rel, self.graph.node(fnid).label
        if subject is None or relation is None or value is None:
            return None
        return (subject, relation, value)

    def facts(self) -> List[Triple]:
        """Every asserted (non-rule, non-FALSE) fact as a ``(s, r, v)`` triple."""
        out: List[Triple] = []
        for nid, node in self.graph.nodes.items():
            if node.kind is not NodeKind.CLAUSE:
                continue
            if node.meta.get("kind") != _FACT:
                continue
            if node.meta.get("truth") == "FALSE":
                continue
            t = self._triple_of(nid)
            if t is not None:
                out.append(t)
        return out

    def rules(self) -> List[Rule]:
        """Reconstruct every stored Horn rule from its graph-resident pattern nodes."""
        grouped: Dict[str, Dict[str, object]] = {}
        for nid, node in self.graph.nodes.items():
            if node.kind is not NodeKind.CLAUSE or node.meta.get("kind") != _RULE:
                continue
            uid = str(node.meta.get("rule", ""))  # the per-rule unique id (names may collide)
            t = self._triple_of(nid)
            if t is None:
                continue
            entry = grouped.setdefault(uid, {"ants": [], "cons": None, "name": node.meta.get("rule_name", uid)})
            if node.meta.get("role") == _CONSEQUENT:
                entry["cons"] = t
            else:
                entry["ants"].append((int(node.meta.get("order", 0)), t))  # type: ignore[union-attr]
        out: List[Rule] = []
        for entry in grouped.values():
            if entry["cons"] is None:
                continue
            ants = tuple(t for _, t in sorted(entry["ants"]))  # type: ignore[index]
            out.append(Rule(antecedents=ants, consequent=entry["cons"], name=str(entry["name"])))  # type: ignore[arg-type]
        return out

    # -- reasoning (the live executor) ---------------------------------------
    def derive(self, subject: str, relation: str) -> Tuple[Optional[str], list]:
        """Answer ``(subject, relation)`` by forward-chaining over the graph.

        Reuses the oracle's unifier (:func:`reasoning_oracle.derive`) but over
        **graph-resident** facts + rules. ``value is None`` ⇒ unanswerable (abstain).
        """
        return _oracle_derive(self.facts(), self.rules(), (subject, relation))

    # -- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the whole knowledge graph to disk (M1 persistence gate)."""
        persistence.save_graph(self.graph, path)

    @classmethod
    def load(cls, path: str, codec: Optional[TPRCodec] = None) -> "KnowledgeGraph":
        """Reload a knowledge graph saved by :meth:`save`, identically."""
        graph = persistence.load_graph(path, codec=codec)
        obj = cls.__new__(cls)
        obj.codec = graph.codec
        obj.graph = graph
        # Restore the rule-id counter past any existing uid so new rules stay distinct.
        seq = 0
        for node in graph.nodes.values():
            uid = node.meta.get("rule")
            if isinstance(uid, str) and "#" in uid:
                seq = max(seq, int(uid.rsplit("#", 1)[1]) + 1)
        obj._rule_seq = seq
        pred = graph.concept_index.get("is")
        obj._predicate_nid = pred if pred is not None else obj._concept("is")
        return obj


__all__ = ["KnowledgeGraph"]
