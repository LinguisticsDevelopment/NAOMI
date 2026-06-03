"""Tests for the core data structures: parse trees, causal tables, state."""

import numpy as np
import pytest

from nsm_ct.data_structures import (
    CausalTable,
    ComprehensionExample,
    ConsciousnessState,
    ParseNode,
    ParseTree,
)


# -- parse trees -------------------------------------------------------------
def _toy_tree() -> ParseTree:
    leaves = [ParseNode("CONTENT", "cat"), ParseNode("FUNC", "is"), ParseNode("CONTENT", "red")]
    return ParseTree(root=ParseNode("S", None, leaves), text="cat is red")


def test_parse_tree_traversal_and_leaves():
    tree = _toy_tree()
    # root + 3 leaves
    assert tree.num_nodes() == 4
    leaf_tokens = [n.token for n in tree.leaves()]
    assert leaf_tokens == ["cat", "is", "red"]


def test_parse_node_is_leaf():
    leaf = ParseNode("CONTENT", "cat")
    internal = ParseNode("S", None, [leaf])
    assert leaf.is_leaf
    assert not internal.is_leaf


def test_preorder_order():
    tree = _toy_tree()
    labels = [n.label for n in tree.iter_preorder()]
    assert labels[0] == "S"  # root first in pre-order
    assert labels[1:] == ["CONTENT", "FUNC", "CONTENT"]


# -- causal table ------------------------------------------------------------
def test_causal_table_add_and_query():
    table = CausalTable()
    table.add("rain", "wet", weight=0.9)
    table.add("rain", "cold")
    table.add("fire", "wet")  # contrived

    assert len(table) == 3
    effects = table.get_effects("rain")
    assert {r.effect for r in effects} == {"wet", "cold"}
    causes = table.get_causes("wet")
    assert {r.cause for r in causes} == {"rain", "fire"}
    # default relation is the NSM prime BECAUSE
    assert effects[0].relation == "BECAUSE"


def test_causal_relation_defaults():
    table = CausalTable()
    rel = table.add("a", "b")
    assert rel.weight == 1.0
    assert rel.cause == "a" and rel.effect == "b"


# -- consciousness state -----------------------------------------------------
def test_consciousness_state_dim_and_zeros():
    state = ConsciousnessState.zeros(8)
    assert state.dim == 8
    assert np.allclose(state.vector, 0.0)


def test_consciousness_state_distance():
    a = ConsciousnessState(np.zeros(4, dtype=np.float32))
    b = ConsciousnessState(np.array([0, 0, 3, 4], dtype=np.float32))
    assert a.distance(b) == pytest.approx(5.0)
    assert a.distance(a) == pytest.approx(0.0)


def test_consciousness_state_dim_mismatch_raises():
    a = ConsciousnessState.zeros(4)
    b = ConsciousnessState.zeros(5)
    with pytest.raises(ValueError):
        a.distance(b)


# -- comprehension example ---------------------------------------------------
def test_comprehension_example_validation():
    ex = ComprehensionExample("p.", "q?", ["a", "b", "c", "d"], answer_idx=2)
    assert ex.answer_text == "c"
    assert "p." in ex.context and "q?" in ex.context

    with pytest.raises(ValueError):
        ComprehensionExample("p", "q", ["a", "b", "c"], answer_idx=0)  # wrong option count
    with pytest.raises(ValueError):
        ComprehensionExample("p", "q", ["a", "b", "c", "d"], answer_idx=9)  # bad index
