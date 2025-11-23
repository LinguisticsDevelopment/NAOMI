"""Test POS ambiguity handling in quantum parser."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.parser import QuantumParser, Word, Tag, print_hypothesis_tree
from src.parser.pos_tagger import tag_sentence


def test_duck_ambiguity():
    """Test: I saw her duck (duck = VERB or NOUN)"""
    print("=== Test 1: I saw her duck ===")
    parser = QuantumParser("grammars/english.json")

    # Tag sentence - "duck" will be ambiguous
    words = tag_sentence("I saw her duck")

    print(f"Words: {[f'{w.text}({w.pos.name})' for w in words]}")

    # Parse with POS ambiguity enabled
    chart = parser.parse(words)

    print(f"Initial hypotheses generated: {len(chart.hypotheses)}")
    print(f"Final hypotheses: {len(chart.hypotheses)}")

    # Show top 3 parses
    chart.sort_hypotheses()
    for i, hyp in enumerate(chart.hypotheses[:3]):
        unconsumed = hyp.get_unconsumed()
        print(f"\nHypothesis {i+1} (Score: {hyp.score:.3f}, Unconsumed: {len(unconsumed)}):")

        # Show POS assignment
        pos_tags = [node.pos.name for node in hyp.nodes]
        print(f"  POS: {pos_tags}")

        if len(unconsumed) <= 2:
            print_hypothesis_tree(hyp)

    print()


def test_book_ambiguity():
    """Test: Book the flight vs Read the book"""
    print("=== Test 2: Book the flight ===")
    parser = QuantumParser("grammars/english.json")

    words = tag_sentence("book the flight")
    print(f"Words: {[f'{w.text}({w.pos.name})' for w in words]}")

    chart = parser.parse(words)
    print(f"Hypotheses: {len(chart.hypotheses)}")

    best = chart.best_hypothesis()
    print(f"Best score: {best.score:.3f}")
    print(f"POS assignment: {[node.pos.name for node in best.nodes]}")

    # "book" should ideally be VERB here
    book_node = best.nodes[0]
    if book_node.pos == Tag.VERB:
        print("Correct: 'book' tagged as VERB [OK]")
    else:
        print(f"Note: 'book' tagged as {book_node.pos.name}")

    print()


def test_time_flies():
    """Test: Time flies like an arrow (multiple ambiguities)"""
    print("=== Test 3: Time flies like an arrow ===")
    parser = QuantumParser("grammars/english.json")

    words = tag_sentence("time flies like an arrow")
    print(f"Words: {[f'{w.text}({w.pos.name})' for w in words]}")

    chart = parser.parse(words)
    print(f"Hypotheses: {len(chart.hypotheses)}")

    # Show top 2 interpretations
    chart.sort_hypotheses()
    for i, hyp in enumerate(chart.hypotheses[:2]):
        unconsumed = hyp.get_unconsumed()
        print(f"\nHypothesis {i+1} (Score: {hyp.score:.3f}, Unconsumed: {len(unconsumed)}):")
        pos_tags = [node.pos.name for node in hyp.nodes]
        print(f"  POS: {pos_tags}")

    print()


def test_light_ambiguity():
    """Test: Light the candle vs The light is bright"""
    print("=== Test 4: Light ambiguity ===")
    parser = QuantumParser("grammars/english.json")

    # Test 1: "light" as verb
    print("Part A: light the candle")
    words1 = tag_sentence("light the candle")
    chart1 = parser.parse(words1)
    best1 = chart1.best_hypothesis()
    print(f"  POS: {[node.pos.name for node in best1.nodes]}")
    print(f"  Score: {best1.score:.3f}")

    # Test 2: "light" as noun
    print("\nPart B: the light is bright")
    words2 = tag_sentence("the light is bright")
    chart2 = parser.parse(words2)
    best2 = chart2.best_hypothesis()
    print(f"  POS: {[node.pos.name for node in best2.nodes]}")
    print(f"  Score: {best2.score:.3f}")

    print()


def test_can_ambiguity():
    """Test: I can run vs Open the can"""
    print("=== Test 5: Can ambiguity ===")
    parser = QuantumParser("grammars/english.json")

    # Test 1: "can" as auxiliary
    print("Part A: I can run")
    words1 = tag_sentence("I can run")
    chart1 = parser.parse(words1)
    best1 = chart1.best_hypothesis()
    pos1 = [node.pos.name for node in best1.nodes]
    print(f"  POS: {pos1}")
    print(f"  Score: {best1.score:.3f}")

    # "can" should ideally be AUX
    can_node1 = best1.nodes[1]
    if can_node1.pos == Tag.AUX:
        print("  Correct: 'can' tagged as AUX [OK]")

    # Test 2: "can" as noun
    print("\nPart B: open the can")
    words2 = tag_sentence("open the can")
    chart2 = parser.parse(words2)
    best2 = chart2.best_hypothesis()
    pos2 = [node.pos.name for node in best2.nodes]
    print(f"  POS: {pos2}")
    print(f"  Score: {best2.score:.3f}")

    print()


def test_no_ambiguity():
    """Test: Simple sentence with no ambiguous words"""
    print("=== Test 6: No ambiguity (the dog runs) ===")
    parser = QuantumParser("grammars/english.json")

    words = tag_sentence("the dog runs")
    chart = parser.parse(words)

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print("Should be 1 hypothesis (no ambiguous words)")

    best = chart.best_hypothesis()
    unconsumed = best.get_unconsumed()
    print(f"Best score: {best.score:.3f}")
    print(f"Unconsumed: {len(unconsumed)}")

    if len(unconsumed) == 1:
        root = best.nodes[list(unconsumed)[0]]
        print(f"Root: {root.value.text} ({root.type.name}) [OK]")

    print()


def test_disabled_ambiguity():
    """Test with POS ambiguity disabled"""
    print("=== Test 7: POS ambiguity disabled ===")

    from src.parser.data_structures import ParserConfig

    config = ParserConfig()
    config.enable_pos_ambiguity = False

    parser = QuantumParser("grammars/english.json", config=config)

    words = tag_sentence("I saw her duck")
    chart = parser.parse(words)

    print(f"Hypotheses: {len(chart.hypotheses)}")
    print("Should be fewer hypotheses with POS ambiguity disabled")

    best = chart.best_hypothesis()
    print(f"Best score: {best.score:.3f}")
    print(f"POS: {[node.pos.name for node in best.nodes]}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("POS AMBIGUITY TESTS")
    print("=" * 60)
    print()

    try:
        test_duck_ambiguity()
        test_book_ambiguity()
        test_time_flies()
        test_light_ambiguity()
        test_can_ambiguity()
        test_no_ambiguity()
        test_disabled_ambiguity()

        print()
        print("=" * 60)
        print("ALL POS AMBIGUITY TESTS COMPLETED! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
