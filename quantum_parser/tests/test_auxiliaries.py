"""Test auxiliary verbs and modals."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.parser import ChartParser
from src.parser.pos_tagger import tag_sentence


def test_modals():
    """Test modal auxiliary verbs."""
    print("=" * 60)
    print("TEST: Modal Auxiliaries")
    print("=" * 60)

    parser = ChartParser("grammars/english.json")

    test_cases = [
        "can run",
        "should go",
        "must leave",
        "will see",
        "could walk",
    ]

    for sentence in test_cases:
        print(f"\n'{sentence}':")
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        print(f"  Hypotheses: {len(chart.hypotheses)}")
        if chart.hypotheses:
            unconsumed = chart.hypotheses[0].get_unconsumed()
            print(f"  Best score: {chart.hypotheses[0].score:.3f}, Unconsumed: {len(unconsumed)}")


def test_perfect_aspect():
    """Test perfect aspect (have + past participle)."""
    print("\n" + "=" * 60)
    print("TEST: Perfect Aspect")
    print("=" * 60)

    parser = ChartParser("grammars/english.json")

    test_cases = [
        "has run",
        "had seen",
        "have eaten",
    ]

    for sentence in test_cases:
        print(f"\n'{sentence}':")
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        print(f"  Hypotheses: {len(chart.hypotheses)}")
        if chart.hypotheses:
            unconsumed = chart.hypotheses[0].get_unconsumed()
            print(f"  Best score: {chart.hypotheses[0].score:.3f}, Unconsumed: {len(unconsumed)}")


def test_progressive_aspect():
    """Test progressive aspect (be + present participle)."""
    print("\n" + "=" * 60)
    print("TEST: Progressive Aspect")
    print("=" * 60)

    parser = ChartParser("grammars/english.json")

    test_cases = [
        "is running",
        "was walking",
        "are jumping",
    ]

    for sentence in test_cases:
        print(f"\n'{sentence}':")
        words = tag_sentence(sentence)
        chart = parser.parse(words)
        print(f"  Hypotheses: {len(chart.hypotheses)}")
        if chart.hypotheses:
            unconsumed = chart.hypotheses[0].get_unconsumed()
            print(f"  Best score: {chart.hypotheses[0].score:.3f}, Unconsumed: {len(unconsumed)}")


def test_auxiliary_stacking():
    """Test stacking of auxiliary verbs."""
    print("\n" + "=" * 60)
    print("TEST: Auxiliary Stacking")
    print("=" * 60)

    parser = ChartParser("grammars/english.json")

    test_cases = [
        "has been running",
        "will have seen",
        "could have been walking",
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
        test_modals()
        test_perfect_aspect()
        test_progressive_aspect()
        test_auxiliary_stacking()

        print("\n" + "=" * 60)
        print("AUXILIARY TESTS COMPLETED")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
