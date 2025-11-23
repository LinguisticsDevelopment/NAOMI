"""Test quantum parser with simple examples."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parser import QuantumParser, Word, Tag, SubType


def test_simple_parse():
    """Test parsing a simple sentence."""
    # Create parser with test grammar
    parser = QuantumParser("grammars/test_grammar.json")

    # Simple sentence: "very big dog"
    words = [
        Word("very", Tag.ADV),   # → SPECIFIER
        Word("big", Tag.ADJ),    # → DESCRIPTOR
        Word("dog", Tag.NOUN)    # → NOUN
    ]

    # Parse
    chart = parser.parse(words)

    # Should have at least one hypothesis
    assert len(chart.hypotheses) > 0

    best = chart.best_hypothesis()
    assert best is not None

    print(f"[OK] Parsed 'very big dog' into {len(chart.hypotheses)} hypothesis(es)")
    print(f"    Best score: {best.score:.3f}")
    print(f"    Edges: {len(best.edges)}")
    print(f"    Unconsumed: {len(best.get_unconsumed())}")


def test_parser_initialization():
    """Test that parser initializes correctly."""
    parser = QuantumParser("grammars/test_grammar.json")

    assert parser.grammar is not None
    assert parser.grammar.language == "test"
    assert len(parser.grammar.order) == 2

    print("[OK] Parser initialized")


def test_hypothesis_creation():
    """Test that parser creates hypotheses."""
    parser = QuantumParser("grammars/test_grammar.json")

    words = [
        Word("big", Tag.ADJ),
        Word("dog", Tag.NOUN)
    ]

    chart = parser.parse(words)

    # Should create at least initial hypothesis
    assert len(chart.hypotheses) >= 1

    print(f"[OK] Created {len(chart.hypotheses)} hypothesis(es)")


if __name__ == "__main__":
    print("Testing quantum parser...")
    print()

    test_parser_initialization()
    test_hypothesis_creation()
    test_simple_parse()

    print()
    print("All parser tests passed! [OK]")
