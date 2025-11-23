"""Comprehensive tests for English grammar with complex structures."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parser import QuantumParser, Word, Tag, SubType


def print_parse_result(sentence_str, chart, show_unconsumed=True):
    """Helper to print parse results."""
    best = chart.best_hypothesis()
    print(f"[OK] '{sentence_str}'")
    print(f"     Hypotheses: {len(chart.hypotheses)}, Best score: {best.score:.3f}")
    print(f"     Edges: {len(best.edges)}", end="")

    if show_unconsumed:
        unconsumed = best.get_unconsumed()
        print(f", Unconsumed: {len(unconsumed)}", end="")
        if unconsumed:
            unconsumed_words = [best.nodes[i].value.text for i in unconsumed]
            print(f" ({', '.join(unconsumed_words)})", end="")
    print()


def test_basic_structures():
    """Test basic sentence structures."""
    print("\n=== Basic Structures ===")
    parser = QuantumParser("grammars/english.json")

    # Intransitive
    words = [Word("dogs", Tag.NOUN), Word("run", Tag.VERB)]
    chart = parser.parse(words)
    print_parse_result("dogs run", chart)

    # Transitive
    words = [
        Word("the", Tag.DET),
        Word("cat", Tag.NOUN),
        Word("chases", Tag.VERB),
        Word("mice", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the cat chases mice", chart)

    # Copula
    words = [
        Word("the", Tag.DET),
        Word("sky", Tag.NOUN),
        Word("is", Tag.VERB),
        Word("blue", Tag.ADJ)
    ]
    chart = parser.parse(words)
    print_parse_result("the sky is blue", chart)


def test_modification():
    """Test various types of modification."""
    print("\n=== Modification ===")
    parser = QuantumParser("grammars/english.json")

    # Multiple adjectives
    words = [
        Word("the", Tag.DET),
        Word("big", Tag.ADJ),
        Word("red", Tag.ADJ),
        Word("shiny", Tag.ADJ),
        Word("ball", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the big red shiny ball", chart)

    # Adverb chains
    words = [
        Word("very", Tag.ADV),
        Word("quickly", Tag.ADV),
        Word("running", Tag.VERB)
    ]
    chart = parser.parse(words)
    print_parse_result("very quickly running", chart)

    # Adverb modifying adjective
    words = [
        Word("extremely", Tag.ADV),
        Word("happy", Tag.ADJ),
        Word("child", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("extremely happy child", chart)


def test_prepositional_phrases():
    """Test prepositional phrase attachment."""
    print("\n=== Prepositional Phrases ===")
    parser = QuantumParser("grammars/english.json")

    # PP modifying noun
    words = [
        Word("the", Tag.DET),
        Word("book", Tag.NOUN),
        Word("on", Tag.ADP),
        Word("the", Tag.DET),
        Word("table", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the book on the table", chart)

    # PP modifying verb
    words = [
        Word("runs", Tag.VERB),
        Word("in", Tag.ADP),
        Word("the", Tag.DET),
        Word("park", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("runs in the park", chart)

    # Multiple PPs
    words = [
        Word("the", Tag.DET),
        Word("cat", Tag.NOUN),
        Word("on", Tag.ADP),
        Word("the", Tag.DET),
        Word("mat", Tag.NOUN),
        Word("in", Tag.ADP),
        Word("the", Tag.DET),
        Word("house", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the cat on the mat in the house", chart)


def test_coordination():
    """Test coordination of various constituents."""
    print("\n=== Coordination ===")
    parser = QuantumParser("grammars/english.json")

    # Noun coordination
    words = [
        Word("cats", Tag.NOUN),
        Word("and", Tag.CCONJ),
        Word("dogs", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("cats and dogs", chart)

    # Adjective coordination
    words = [
        Word("big", Tag.ADJ),
        Word("and", Tag.CCONJ),
        Word("red", Tag.ADJ),
        Word("ball", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("big and red ball", chart)

    # Verb coordination
    words = [
        Word("runs", Tag.VERB),
        Word("and", Tag.CCONJ),
        Word("jumps", Tag.VERB)
    ]
    chart = parser.parse(words)
    print_parse_result("runs and jumps", chart)

    # Three-way coordination
    words = [
        Word("cats", Tag.NOUN),
        Word("and", Tag.CCONJ),
        Word("dogs", Tag.NOUN),
        Word("and", Tag.CCONJ),
        Word("birds", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("cats and dogs and birds", chart)


def test_complex_sentences():
    """Test complex sentence structures."""
    print("\n=== Complex Sentences ===")
    parser = QuantumParser("grammars/english.json")

    # Modified subject and object
    words = [
        Word("the", Tag.DET),
        Word("big", Tag.ADJ),
        Word("dog", Tag.NOUN),
        Word("chases", Tag.VERB),
        Word("the", Tag.DET),
        Word("small", Tag.ADJ),
        Word("cat", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the big dog chases the small cat", chart)

    # With adverb
    words = [
        Word("the", Tag.DET),
        Word("dog", Tag.NOUN),
        Word("quickly", Tag.ADV),
        Word("chases", Tag.VERB),
        Word("the", Tag.DET),
        Word("cat", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the dog quickly chases the cat", chart)

    # With PP modification
    words = [
        Word("the", Tag.DET),
        Word("dog", Tag.NOUN),
        Word("runs", Tag.VERB),
        Word("in", Tag.ADP),
        Word("the", Tag.DET),
        Word("park", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the dog runs in the park", chart)

    # Multiple modifiers on different constituents
    words = [
        Word("the", Tag.DET),
        Word("very", Tag.ADV),
        Word("big", Tag.ADJ),
        Word("dog", Tag.NOUN),
        Word("runs", Tag.VERB),
        Word("extremely", Tag.ADV),
        Word("quickly", Tag.ADV),
        Word("in", Tag.ADP),
        Word("the", Tag.DET),
        Word("green", Tag.ADJ),
        Word("park", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the very big dog runs extremely quickly in the green park", chart)


def test_coordination_complex():
    """Test coordination in complex contexts."""
    print("\n=== Complex Coordination ===")
    parser = QuantumParser("grammars/english.json")

    # Coordinated NPs with modifiers
    words = [
        Word("the", Tag.DET),
        Word("big", Tag.ADJ),
        Word("dog", Tag.NOUN),
        Word("and", Tag.CCONJ),
        Word("the", Tag.DET),
        Word("small", Tag.ADJ),
        Word("cat", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the big dog and the small cat", chart)

    # Coordinated verbs with object
    words = [
        Word("chases", Tag.VERB),
        Word("and", Tag.CCONJ),
        Word("catches", Tag.VERB),
        Word("the", Tag.DET),
        Word("mouse", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("chases and catches the mouse", chart)


def test_ambiguous_structures():
    """Test sentences with structural ambiguity."""
    print("\n=== Ambiguous Structures (Multiple Hypotheses) ===")
    parser = QuantumParser("grammars/english.json")

    # PP attachment ambiguity
    words = [
        Word("the", Tag.DET),
        Word("man", Tag.NOUN),
        Word("saw", Tag.VERB),
        Word("the", Tag.DET),
        Word("boy", Tag.NOUN),
        Word("with", Tag.ADP),
        Word("the", Tag.DET),
        Word("telescope", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the man saw the boy with the telescope", chart)
    print(f"     (PP can attach to verb or object noun)")

    # Coordination scope ambiguity
    words = [
        Word("old", Tag.ADJ),
        Word("men", Tag.NOUN),
        Word("and", Tag.CCONJ),
        Word("women", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("old men and women", chart)
    print(f"     (Does 'old' modify both or just 'men'?)")


def test_edge_cases():
    """Test edge cases and special structures."""
    print("\n=== Edge Cases ===")
    parser = QuantumParser("grammars/english.json")

    # Single word
    words = [Word("runs", Tag.VERB)]
    chart = parser.parse(words)
    print_parse_result("runs", chart)

    # Just a noun phrase (no verb)
    words = [
        Word("the", Tag.DET),
        Word("big", Tag.ADJ),
        Word("red", Tag.ADJ),
        Word("ball", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("the big red ball", chart)

    # Apposition
    words = [
        Word("John", Tag.NOUN),
        Word("the", Tag.DET),
        Word("teacher", Tag.NOUN)
    ]
    chart = parser.parse(words)
    print_parse_result("John the teacher", chart)


if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE ENGLISH GRAMMAR TESTS")
    print("=" * 60)

    try:
        test_basic_structures()
        test_modification()
        test_prepositional_phrases()
        test_coordination()
        test_complex_sentences()
        test_coordination_complex()
        test_ambiguous_structures()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("ALL COMPREHENSIVE TESTS PASSED! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
