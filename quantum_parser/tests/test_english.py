"""Test quantum parser with English grammar."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parser import QuantumParser, Word, Tag, SubType


def test_simple_sentence():
    """Test: The dog runs"""
    parser = QuantumParser("grammars/english.json")

    words = [
        Word("the", Tag.DET),      # → DESCRIPTOR
        Word("dog", Tag.NOUN),     # → NOUN → NOMINAL
        Word("runs", Tag.VERB)     # → VERBAL → PREDICATE
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"[OK] 'The dog runs': {len(chart.hypotheses)} hypothesis(es), score={best.score:.3f}")
    print(f"    Edges: {len(best.edges)}, Unconsumed: {len(best.get_unconsumed())}")


def test_descriptors():
    """Test: The very big red dog"""
    parser = QuantumParser("grammars/english.json")

    words = [
        Word("the", Tag.DET),      # → DESCRIPTOR
        Word("very", Tag.ADV),     # → SPECIFIER
        Word("big", Tag.ADJ),      # → DESCRIPTOR
        Word("red", Tag.ADJ),      # → DESCRIPTOR
        Word("dog", Tag.NOUN)      # → NOUN
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"[OK] 'The very big red dog': {len(chart.hypotheses)} hypothesis(es), score={best.score:.3f}")


def test_prepositional_phrase():
    """Test: The dog on the table"""
    parser = QuantumParser("grammars/english.json")

    words = [
        Word("the", Tag.DET),
        Word("dog", Tag.NOUN),
        Word("on", Tag.ADP),       # → PREP
        Word("the", Tag.DET),
        Word("table", Tag.NOUN)
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"[OK] 'The dog on the table': {len(chart.hypotheses)} hypothesis(es), score={best.score:.3f}")


def test_transitive_verb():
    """Test: The dog chases the cat"""
    parser = QuantumParser("grammars/english.json")

    words = [
        Word("the", Tag.DET),
        Word("dog", Tag.NOUN),
        Word("chases", Tag.VERB),
        Word("the", Tag.DET),
        Word("cat", Tag.NOUN)
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"[OK] 'The dog chases the cat': {len(chart.hypotheses)} hypothesis(es), score={best.score:.3f}")


def test_adverb():
    """Test: The dog runs quickly"""
    parser = QuantumParser("grammars/english.json")

    words = [
        Word("the", Tag.DET),
        Word("dog", Tag.NOUN),
        Word("runs", Tag.VERB),
        Word("quickly", Tag.ADV)   # → SPECIFIER
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"[OK] 'The dog runs quickly': {len(chart.hypotheses)} hypothesis(es), score={best.score:.3f}")


def test_coordination():
    """Test: The dog and the cat"""
    parser = QuantumParser("grammars/english.json")

    words = [
        Word("the", Tag.DET),
        Word("dog", Tag.NOUN),
        Word("and", Tag.CCONJ),    # → COORD
        Word("the", Tag.DET),
        Word("cat", Tag.NOUN)
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"[OK] 'The dog and the cat': {len(chart.hypotheses)} hypothesis(es), score={best.score:.3f}")


def test_complex_sentence():
    """Test: The big dog runs quickly on the table"""
    parser = QuantumParser("grammars/english.json")

    words = [
        Word("the", Tag.DET),
        Word("big", Tag.ADJ),
        Word("dog", Tag.NOUN),
        Word("runs", Tag.VERB),
        Word("quickly", Tag.ADV),
        Word("on", Tag.ADP),
        Word("the", Tag.DET),
        Word("table", Tag.NOUN)
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"[OK] 'The big dog runs quickly on the table': {len(chart.hypotheses)} hypothesis(es), score={best.score:.3f}")


def test_copula():
    """Test: The dog is happy"""
    parser = QuantumParser("grammars/english.json")

    words = [
        Word("the", Tag.DET),
        Word("dog", Tag.NOUN),
        Word("is", Tag.VERB),
        Word("happy", Tag.ADJ)
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"[OK] 'The dog is happy': {len(chart.hypotheses)} hypothesis(es), score={best.score:.3f}")


if __name__ == "__main__":
    print("Testing English grammar...")
    print()

    try:
        test_simple_sentence()
        test_descriptors()
        test_prepositional_phrase()
        test_transitive_verb()
        test_adverb()
        test_coordination()
        test_complex_sentence()
        test_copula()

        print()
        print("All English tests passed! [OK]")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
