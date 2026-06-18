"""M0/M1 gates for ``nsm_ct.mind``.

* **M0:** the frozen schema imports and a meaning object round-trips via
  collapse/expand.
* **M1:** ``KnowledgeGraph.derive`` over **graph-resident** rules reproduces the
  symbolic oracle's answers (parity), and the graph persists + reloads identically.
"""

from __future__ import annotations

import numpy as np
import pytest

from nsm_ct.mind import schema
from nsm_ct.mind.knowledge import KnowledgeGraph
from nsm_ct.reasoning_oracle import (
    INHERITANCE,
    IS_A_TRANS,
    conditional_rule,
    derive as oracle_derive,
)


# --------------------------------------------------------------------------- M0
def test_schema_contract_present():
    assert schema.MEANING_OPERATORS == ("NOT", "MAYBE", "AND", "OR", "IF")
    assert "IS_A" in schema.REASONING_RELATIONS and "CAN" in schema.REASONING_RELATIONS
    assert schema.is_variable("?x") and not schema.is_variable("robin")
    assert schema.NUM_PRIMES >= 60


def test_meaning_object_round_trip():
    """A clause node files into the graph and expands back to its exact structure."""
    from nsm_ct.collapse import collapse, expand
    from nsm_ct.data_structures import ParseNode, ParseTree
    from nsm_ct.meaning_graph import MeaningGraph, NodeKind
    from nsm_ct.tpr import TPRCodec

    codec = TPRCodec(dim=128)
    g = MeaningGraph(codec)
    root = ParseNode(label="is", token="is")
    root.children.append(ParseNode(label="SOMEONE", token="mary", relation="SUBJECT"))
    root.children.append(ParseNode(label="SOMEWHERE", token="kitchen", relation="PLACE"))
    tree = ParseTree(root=root, text="mary is kitchen")
    nid = collapse(g, tree, codec, kind=NodeKind.CLAUSE)
    back = expand(g, nid)
    assert [n.label for n in back.root.iter_preorder()] == [n.label for n in tree.root.iter_preorder()]


# --------------------------------------------------------------------------- M1
def _inheritance_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph(dim=128)
    kg.assert_facts([("robin", "IS_A", "bird"), ("bird", "CAN", "fly")])
    kg.add_rule(INHERITANCE)
    return kg


def test_derive_inheritance_matches_oracle():
    kg = _inheritance_kg()
    value, chain = kg.derive("robin", "CAN")
    # Parity: the oracle, given the SAME facts+rules, derives the same answer.
    facts = [("robin", "IS_A", "bird"), ("bird", "CAN", "fly")]
    expected, _ = oracle_derive(facts, [INHERITANCE], ("robin", "CAN"))
    assert value == expected == "fly"
    assert chain  # a non-empty derivation chain


def test_derive_is_a_transitivity():
    from nsm_ct.reasoning_oracle import forward_chain

    kg = KnowledgeGraph(dim=128)
    kg.assert_facts([("robin", "IS_A", "bird"), ("bird", "IS_A", "animal")])
    kg.add_rule(IS_A_TRANS)
    # robin IS_A {bird, animal} are BOTH valid; assert the transitive fact is in the
    # derived closure (derive() returns an arbitrary one of the two, set-order dependent).
    known, _ = forward_chain(kg.facts(), kg.rules())
    assert ("robin", "IS_A", "animal") in known


def test_derive_modus_ponens():
    kg = KnowledgeGraph(dim=128)
    kg.add_fact("mary", "PLACE", "kitchen")
    kg.add_rule(conditional_rule(place="kitchen", obj="window"))
    value, _ = kg.derive("mary", "CAN_SEE")
    assert value == "window"


def test_derive_abstains_when_underivable():
    kg = _inheritance_kg()
    value, _ = kg.derive("robin", "PLACE")  # nothing derives a place for robin
    assert value is None


def test_facts_and_rules_recovered_from_graph():
    kg = _inheritance_kg()
    assert set(kg.facts()) == {("robin", "IS_A", "bird"), ("bird", "CAN", "fly")}
    rules = kg.rules()
    assert len(rules) == 1
    assert rules[0].name == "inheritance"
    assert rules[0].consequent == ("?x", "CAN", "?z")
    assert set(rules[0].antecedents) == {("?x", "IS_A", "?y"), ("?y", "CAN", "?z")}


def test_persistence_round_trip(tmp_path):
    kg = _inheritance_kg()
    kg.define("bird", ["SOMETHING", "CAN", "MOVE"])  # exercise the DEFINES edge too
    path = str(tmp_path / "kg.json")
    kg.save(path)

    reloaded = KnowledgeGraph.load(path)
    # Facts, rules, and derivation all survive a round trip identically.
    assert set(reloaded.facts()) == set(kg.facts())
    assert {r.name for r in reloaded.rules()} == {r.name for r in kg.rules()}
    assert reloaded.derive("robin", "CAN") == kg.derive("robin", "CAN")
    # Node/edge counts and handles are byte-identical.
    assert len(reloaded.graph.nodes) == len(kg.graph.nodes)
    assert len(reloaded.graph.edges) == len(kg.graph.edges)
    for nid, node in kg.graph.nodes.items():
        assert np.allclose(reloaded.graph.node(nid).handle, node.handle)


def test_defines_edge_created():
    from nsm_ct.meaning_graph import DEFINES

    kg = KnowledgeGraph(dim=64)
    kg.define("bird", ["SOMETHING", "CAN", "MOVE"])
    assert len(kg.graph.edges_of(DEFINES)) == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
