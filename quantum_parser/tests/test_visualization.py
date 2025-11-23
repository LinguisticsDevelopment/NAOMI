"""Test visualization and POS tagging."""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.parser import QuantumParser, hypothesis_to_dot, chart_to_dot_multi, print_hypothesis_tree, save_dot
from src.parser.pos_tagger import tag_sentence


def test_pos_tagger():
    """Test automatic POS tagging."""
    print("=== Testing POS Tagger ===\n")

    sentence = "the big dog runs quickly"
    words = tag_sentence(sentence)

    print(f"Input: '{sentence}'")
    print("Tagged:")
    for word in words:
        print(f"  {word.text} -> {word.pos.name}")
    print()


def test_text_visualization():
    """Test text-based tree visualization."""
    print("=== Testing Text Visualization ===\n")

    parser = QuantumParser("grammars/english.json")
    words = tag_sentence("the dog chases the cat")

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print_hypothesis_tree(best)


def test_dot_generation():
    """Test DOT file generation."""
    print("=== Testing DOT Generation ===\n")

    parser = QuantumParser("grammars/english.json")
    words = tag_sentence("the man saw the boy with the telescope")

    chart = parser.parse(words)

    # Single hypothesis
    best = chart.best_hypothesis()
    dot_single = hypothesis_to_dot(best, "ambiguous_parse")
    print("Single hypothesis DOT:")
    print(dot_single[:200] + "...")
    print()

    # Save single
    save_dot(dot_single, "output/ambiguous_single.dot")

    # Multiple hypotheses
    dot_multi = chart_to_dot_multi(chart, top_k=3, name="parse_forest")
    print("Multi-hypothesis DOT:")
    print(dot_multi[:200] + "...")
    print()

    # Save multi
    save_dot(dot_multi, "output/ambiguous_multi.dot")


def test_visualization_complex():
    """Test visualization with complex sentence."""
    print("=== Testing Complex Sentence Visualization ===\n")

    parser = QuantumParser("grammars/english.json")
    words = tag_sentence("cats and dogs")

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"Sentence: 'cats and dogs'")
    print(f"Hypotheses generated: {len(chart.hypotheses)}\n")

    print_hypothesis_tree(best)

    # Save DOT
    dot = hypothesis_to_dot(best, "coordination")
    save_dot(dot, "output/coordination.dot")


if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("VISUALIZATION AND POS TAGGING TESTS")
    print("=" * 60)
    print()

    try:
        test_pos_tagger()
        test_text_visualization()
        test_dot_generation()
        test_visualization_complex()

        print()
        print("=" * 60)
        print("ALL VISUALIZATION TESTS PASSED! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
