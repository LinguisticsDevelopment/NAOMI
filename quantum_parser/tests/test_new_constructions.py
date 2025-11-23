"""Test the 5 new grammar constructions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.parser import ChartParser
from src.parser.pos_tagger import tag_sentence


def test_infinitives():
    """Test infinitive phrases."""
    print("=" * 60)
    print("TEST: Infinitives")
    print("=" * 60)

    parser = ChartParser("grammars/english.json")

    test_cases = [
        "to run",
        "to see things",
        "I want to eat",
    ]

    for sentence in test_cases:
        print(f"\n'{sentence}':")
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        print(f"  Hypotheses: {len(chart.hypotheses)}")
        if chart.hypotheses:
            unconsumed = chart.hypotheses[0].get_unconsumed()
            print(f"  Best score: {chart.hypotheses[0].score:.3f}, Unconsumed: {len(unconsumed)}")


def test_gerunds():
    """Test gerunds as nominals."""
    print("\n" + "=" * 60)
    print("TEST: Gerunds")
    print("=" * 60)

    parser = ChartParser("grammars/english.json")

    test_cases = [
        "running is fun",
        "I like swimming",
    ]

    for sentence in test_cases:
        print(f"\n'{sentence}':")
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        print(f"  Hypotheses: {len(chart.hypotheses)}")
        if chart.hypotheses:
            unconsumed = chart.hypotheses[0].get_unconsumed()
            print(f"  Best score: {chart.hypotheses[0].score:.3f}, Unconsumed: {len(unconsumed)}")


def test_participles():
    """Test participles as descriptors."""
    print("\n" + "=" * 60)
    print("TEST: Participles")
    print("=" * 60)

    parser = ChartParser("grammars/english.json")

    test_cases = [
        "running water",
        "the flying bird",
    ]

    for sentence in test_cases:
        print(f"\n'{sentence}':")
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        print(f"  Hypotheses: {len(chart.hypotheses)}")
        if chart.hypotheses:
            unconsumed = chart.hypotheses[0].get_unconsumed()
            print(f"  Best score: {chart.hypotheses[0].score:.3f}, Unconsumed: {len(unconsumed)}")


def test_possessives():
    """Test possessive constructions."""
    print("\n" + "=" * 60)
    print("TEST: Possessives")
    print("=" * 60)

    parser = ChartParser("grammars/english.json")

    test_cases = [
        "John's book",
        "the dog's tail",
    ]

    for sentence in test_cases:
        print(f"\n'{sentence}':")
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        print(f"  Hypotheses: {len(chart.hypotheses)}")
        if chart.hypotheses:
            unconsumed = chart.hypotheses[0].get_unconsumed()
            print(f"  Best score: {chart.hypotheses[0].score:.3f}, Unconsumed: {len(unconsumed)}")


def test_negation():
    """Test negation."""
    print("\n" + "=" * 60)
    print("TEST: Negation")
    print("=" * 60)

    parser = ChartParser("grammars/english.json")

    test_cases = [
        "not run",
        "no dogs",
        "I do not run",
    ]

    for sentence in test_cases:
        print(f"\n'{sentence}':")
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        print(f"  Hypotheses: {len(chart.hypotheses)}")
        if chart.hypotheses:
            unconsumed = chart.hypotheses[0].get_unconsumed()
            print(f"  Best score: {chart.hypotheses[0].score:.3f}, Unconsumed: {len(unconsumed)}")


if __name__ == "__main__":
    try:
        test_infinitives()
        test_gerunds()
        test_participles()
        test_possessives()
        test_negation()

        print("\n" + "=" * 60)
        print("ALL NEW CONSTRUCTION TESTS COMPLETED")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
