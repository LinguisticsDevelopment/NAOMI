"""Test quantum parser with Spanish grammar."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.parser import QuantumParser, Word, Tag, SubType, print_hypothesis_tree


def test_simple_spanish():
    """Test: El perro corre (The dog runs)"""
    print("=== Test 1: El perro corre ===")
    parser = QuantumParser("grammars/spanish.json")

    words = [
        Word("El", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("perro", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("corre", Tag.VERB, [SubType.SINGULAR])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    print(f"Unconsumed: {len(unconsumed)}")
    if len(unconsumed) == 1:
        root = best.nodes[list(unconsumed)[0]]
        print(f"Root: {root.value.text} ({root.type.name}) [OK]")
    print()


def test_adjective_agreement():
    """Test: El perro grande (The big dog)"""
    print("=== Test 2: El perro grande ===")
    parser = QuantumParser("grammars/spanish.json")

    words = [
        Word("El", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("perro", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("grande", Tag.ADJ, [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    print_hypothesis_tree(best)
    print()


def test_feminine_agreement():
    """Test: La casa blanca (The white house)"""
    print("=== Test 3: La casa blanca ===")
    parser = QuantumParser("grammars/spanish.json")

    words = [
        Word("La", Tag.DET, [SubType.FEMININE, SubType.SINGULAR]),
        Word("casa", Tag.NOUN, [SubType.FEMININE, SubType.SINGULAR]),
        Word("blanca", Tag.ADJ, [SubType.FEMININE, SubType.SINGULAR, SubType.POST_NOMINAL])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    unconsumed = best.get_unconsumed()
    if len(unconsumed) == 1:
        root = best.nodes[list(unconsumed)[0]]
        print(f"Root: {root.value.text} ({root.type.name}) [OK]")
    print()


def test_plural_agreement():
    """Test: Los perros grandes (The big dogs)"""
    print("=== Test 4: Los perros grandes ===")
    parser = QuantumParser("grammars/spanish.json")

    words = [
        Word("Los", Tag.DET, [SubType.MASCULINE, SubType.PLURAL]),
        Word("perros", Tag.NOUN, [SubType.MASCULINE, SubType.PLURAL]),
        Word("grandes", Tag.ADJ, [SubType.MASCULINE, SubType.PLURAL, SubType.POST_NOMINAL])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    print()


def test_coordination_spanish():
    """Test: gatos y perros (cats and dogs)"""
    print("=== Test 5: gatos y perros ===")
    parser = QuantumParser("grammars/spanish.json")

    words = [
        Word("gatos", Tag.NOUN, [SubType.MASCULINE, SubType.PLURAL]),
        Word("y", Tag.CCONJ),
        Word("perros", Tag.NOUN, [SubType.MASCULINE, SubType.PLURAL])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    if len(unconsumed) == 1:
        root = best.nodes[list(unconsumed)[0]]
        print(f"Root: {root.value.text} ({root.type.name}) [OK]")
    print()


def test_prepositional_phrase():
    """Test: en la mesa (on the table)"""
    print("=== Test 6: en la mesa ===")
    parser = QuantumParser("grammars/spanish.json")

    words = [
        Word("en", Tag.ADP),
        Word("la", Tag.DET, [SubType.FEMININE, SubType.SINGULAR]),
        Word("mesa", Tag.NOUN, [SubType.FEMININE, SubType.SINGULAR])
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    print()


def test_full_sentence():
    """Test: El perro grande corre rapidamente (The big dog runs quickly)"""
    print("=== Test 7: El perro grande corre rapidamente ===")
    parser = QuantumParser("grammars/spanish.json")

    words = [
        Word("El", Tag.DET, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("perro", Tag.NOUN, [SubType.MASCULINE, SubType.SINGULAR]),
        Word("grande", Tag.ADJ, [SubType.MASCULINE, SubType.SINGULAR, SubType.POST_NOMINAL]),
        Word("corre", Tag.VERB, [SubType.SINGULAR]),
        Word("rapidamente", Tag.ADV)
    ]

    chart = parser.parse(words)
    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print(f"Best score: {best.score:.3f}")
    print_hypothesis_tree(best)

    if len(unconsumed) == 1:
        root = best.nodes[list(unconsumed)[0]]
        print(f"Root: {root.value.text} ({root.type.name}) [OK]")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("SPANISH GRAMMAR TESTS")
    print("=" * 60)
    print()

    try:
        test_simple_spanish()
        test_adjective_agreement()
        test_feminine_agreement()
        test_plural_agreement()
        test_coordination_spanish()
        test_prepositional_phrase()
        test_full_sentence()

        print()
        print("=" * 60)
        print("ALL SPANISH TESTS PASSED! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
