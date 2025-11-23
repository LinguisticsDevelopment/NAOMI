"""Test data structures."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parser.enums import Tag, NodeType, SubType, SubCat, ConnectionType
from src.parser.data_structures import Word, Node, Edge, Hypothesis, ParseChart, create_initial_chart


def test_word_creation():
    """Test creating a Word."""
    word = Word("dog", Tag.NOUN, [SubType.SINGULAR])
    assert word.text == "dog"
    assert word.pos == Tag.NOUN
    assert SubType.SINGULAR in word.subtypes
    print("[OK] Word creation works")


def test_node_creation():
    """Test creating a Node."""
    word = Word("runs", Tag.VERB)
    node = Node(
        type=NodeType.VERBAL,
        original_type=NodeType.VERBAL,
        value=word,
        pos=Tag.VERB,
        index=0
    )
    assert node.type == NodeType.VERBAL
    assert node.value.text == "runs"
    print("[OK] Node creation works")


def test_edge_creation():
    """Test creating an Edge."""
    edge = Edge(ConnectionType.SUBJECT, parent=1, child=0)
    assert edge.type == ConnectionType.SUBJECT
    assert edge.parent == 1
    assert edge.child == 0
    print("[OK] Edge creation works")


def test_hypothesis():
    """Test Hypothesis operations."""
    word1 = Word("the", Tag.DET)
    word2 = Word("dog", Tag.NOUN)

    node1 = Node(NodeType.DESCRIPTOR, NodeType.DESCRIPTOR, word1, Tag.DET, index=0)
    node2 = Node(NodeType.NOUN, NodeType.NOUN, word2, Tag.NOUN, index=1)

    hyp = Hypothesis(nodes=[node1, node2])

    # Initially nothing consumed
    assert len(hyp.get_unconsumed()) == 2

    # Consume node 0
    hyp.consume(0)
    assert len(hyp.get_unconsumed()) == 1
    assert 0 in hyp.consumed

    # Add edge
    edge = Edge(ConnectionType.DESCRIPTION, parent=1, child=0)
    hyp.add_edge(edge)
    assert len(hyp.edges) == 1

    print("[OK] Hypothesis operations work")


def test_initial_chart():
    """Test creating initial parse chart."""
    words = [
        Word("the", Tag.DET),
        Word("dog", Tag.NOUN),
        Word("runs", Tag.VERB)
    ]

    chart = create_initial_chart(words)

    assert len(chart.words) == 3
    assert len(chart.nodes) == 3
    assert len(chart.hypotheses) == 1

    # Initial hypothesis should have all nodes unconsumed
    initial_hyp = chart.hypotheses[0]
    assert len(initial_hyp.get_unconsumed()) == 3

    print("[OK] Initial chart creation works")


def test_subcategory_matching():
    """Test subcategory value retrieval."""
    node = Node(
        type=NodeType.NOUN,
        original_type=NodeType.NOUN,
        value=Word("casa", Tag.NOUN),
        pos=Tag.NOUN,
        flags=[SubType.FEMININE, SubType.SINGULAR],
        index=0
    )

    # Get gender
    gender = node.get_subcategory_value(SubCat.GENDER)
    assert gender == SubType.FEMININE

    # Get number
    number = node.get_subcategory_value(SubCat.NUMBER)
    assert number == SubType.SINGULAR

    # Check for non-existent category
    verb_form = node.get_subcategory_value(SubCat.VERB)
    assert verb_form is None

    print("[OK] Subcategory matching works")


if __name__ == "__main__":
    print("Testing data structures...")
    print()

    test_word_creation()
    test_node_creation()
    test_edge_creation()
    test_hypothesis()
    test_initial_chart()
    test_subcategory_matching()

    print()
    print("All tests passed! [OK]")
