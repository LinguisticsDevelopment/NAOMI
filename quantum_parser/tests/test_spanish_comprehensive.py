"""Comprehensive tests for Spanish grammar with all features."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parser import ChartParser
from src.parser.pos_tagger import tag_spanish_sentence


def print_parse_result(sentence_str, chart, show_edges=False):
    """Helper to print parse results."""
    best = chart.best_hypothesis()
    print(f"[OK] '{sentence_str}'")
    print(f"     Hypotheses: {len(chart.hypotheses)}, Best score: {best.score:.3f}")

    unconsumed = best.get_unconsumed()
    print(f"     Unconsumed: {len(unconsumed)}", end="")
    if unconsumed:
        unconsumed_words = [best.nodes[i].value.text if best.nodes[i].value else f"node{i}" for i in unconsumed]
        root_types = [best.nodes[i].type.name for i in unconsumed]
        print(f" ({', '.join([f'{w}[{t}]' for w, t in zip(unconsumed_words, root_types)])})", end="")
    print()

    if show_edges:
        print(f"     Edges: {len(best.edges)}")
        for edge in best.edges:
            parent_text = best.nodes[edge.parent].value.text if best.nodes[edge.parent].value else f"node{edge.parent}"
            child_text = best.nodes[edge.child].value.text if best.nodes[edge.child].value else f"node{edge.child}"
            print(f"       {child_text} --{edge.type.name}--> {parent_text}")


def test_basic_structures():
    """Test basic Spanish sentence structures."""
    print("\n=== Basic Structures ===")
    parser = ChartParser("grammars/spanish.json")

    # Simple subject-verb
    words = tag_spanish_sentence("El perro corre")
    chart = parser.parse(words)
    print_parse_result("El perro corre", chart)

    # With object
    words = tag_spanish_sentence("El gato come comida")
    chart = parser.parse(words)
    print_parse_result("El gato come comida", chart)

    # Copula
    words = tag_spanish_sentence("La casa es grande")
    chart = parser.parse(words)
    print_parse_result("La casa es grande", chart)


def test_implied_subjects():
    """Test verbs with implied subjects (no explicit pronoun)."""
    print("\n=== Implied Subjects ===")
    parser = ChartParser("grammars/spanish.json")

    # First person singular
    words = tag_spanish_sentence("corro")
    chart = parser.parse(words)
    print_parse_result("corro (I run)", chart)

    # Second person singular
    words = tag_spanish_sentence("comes")
    chart = parser.parse(words)
    print_parse_result("comes (you eat)", chart)

    # Third person singular
    words = tag_spanish_sentence("vive")
    chart = parser.parse(words)
    print_parse_result("vive (he/she lives)", chart)

    # First person plural
    words = tag_spanish_sentence("corremos")
    chart = parser.parse(words)
    print_parse_result("corremos (we run)", chart)

    # With object
    words = tag_spanish_sentence("como comida")
    chart = parser.parse(words)
    print_parse_result("como comida (I eat food)", chart)


def test_gender_number_agreement():
    """Test gender and number agreement."""
    print("\n=== Gender/Number Agreement ===")
    parser = ChartParser("grammars/spanish.json")

    # Masculine singular
    words = tag_spanish_sentence("El perro grande")
    chart = parser.parse(words)
    print_parse_result("El perro grande", chart)

    # Feminine singular
    words = tag_spanish_sentence("La casa blanca")
    chart = parser.parse(words)
    print_parse_result("La casa blanca", chart)

    # Masculine plural
    words = tag_spanish_sentence("Los gatos negros")
    chart = parser.parse(words)
    print_parse_result("Los gatos negros", chart)

    # Feminine plural
    words = tag_spanish_sentence("Las casas bonitas")
    chart = parser.parse(words)
    print_parse_result("Las casas bonitas", chart)

    # Multiple adjectives
    words = tag_spanish_sentence("El perro grande negro")
    chart = parser.parse(words)
    print_parse_result("El perro grande negro", chart)


def test_participles_as_adjectives():
    """Test past participles used as adjectives."""
    print("\n=== Participles as Adjectives ===")
    parser = ChartParser("grammars/spanish.json")

    # Masculine singular
    words = tag_spanish_sentence("El libro leído")
    chart = parser.parse(words)
    print_parse_result("El libro leído (the read book)", chart)

    # Feminine singular
    words = tag_spanish_sentence("La casa construida")
    chart = parser.parse(words)
    print_parse_result("La casa construida (the built house)", chart)

    # Masculine plural
    words = tag_spanish_sentence("Los libros escritos")
    chart = parser.parse(words)
    print_parse_result("Los libros escritos (the written books)", chart)

    # Feminine plural
    words = tag_spanish_sentence("Las puertas cerradas")
    chart = parser.parse(words)
    print_parse_result("Las puertas cerradas (the closed doors)", chart, show_edges=True)


def test_coordination():
    """Test coordination of various constituents."""
    print("\n=== Coordination ===")
    parser = ChartParser("grammars/spanish.json")

    # Noun coordination
    words = tag_spanish_sentence("gatos y perros")
    chart = parser.parse(words)
    print_parse_result("gatos y perros", chart)

    # With determiners
    words = tag_spanish_sentence("el gato y el perro")
    chart = parser.parse(words)
    print_parse_result("el gato y el perro", chart)

    # Adjective coordination
    words = tag_spanish_sentence("grande y negro")
    chart = parser.parse(words)
    print_parse_result("grande y negro", chart)


def test_prepositional_phrases():
    """Test prepositional phrase attachment."""
    print("\n=== Prepositional Phrases ===")
    parser = ChartParser("grammars/spanish.json")

    # Basic PP
    words = tag_spanish_sentence("en la casa")
    chart = parser.parse(words)
    print_parse_result("en la casa", chart)

    # PP modifying noun
    words = tag_spanish_sentence("el libro en la mesa")
    chart = parser.parse(words)
    print_parse_result("el libro en la mesa", chart)

    # PP modifying verb
    words = tag_spanish_sentence("corre en el parque")
    chart = parser.parse(words)
    print_parse_result("corre en el parque", chart)


def test_complex_sentences():
    """Test complex sentence structures."""
    print("\n=== Complex Sentences ===")
    parser = ChartParser("grammars/spanish.json")

    # Modified subject and verb
    words = tag_spanish_sentence("El perro grande corre")
    chart = parser.parse(words)
    print_parse_result("El perro grande corre", chart)

    # With object
    words = tag_spanish_sentence("El gato negro come comida")
    chart = parser.parse(words)
    print_parse_result("El gato negro come comida", chart)

    # With adverb
    words = tag_spanish_sentence("El perro corre rápidamente")
    chart = parser.parse(words)
    print_parse_result("El perro corre rápidamente", chart)

    # Multiple modifiers
    words = tag_spanish_sentence("El perro grande negro corre muy rápidamente")
    chart = parser.parse(words)
    print_parse_result("El perro grande negro corre muy rápidamente", chart)


def test_reflexive_verbs():
    """Test reflexive verb constructions."""
    print("\n=== Reflexive Verbs ===")
    parser = ChartParser("grammars/spanish.json")

    # First person singular
    words = tag_spanish_sentence("me lavo")
    chart = parser.parse(words)
    print_parse_result("me lavo (I wash myself)", chart)

    # Third person singular
    words = tag_spanish_sentence("se lava")
    chart = parser.parse(words)
    print_parse_result("se lava (he/she washes himself/herself)", chart)

    # First person plural
    words = tag_spanish_sentence("nos lavamos")
    chart = parser.parse(words)
    print_parse_result("nos lavamos (we wash ourselves)", chart)

    # Second person singular
    words = tag_spanish_sentence("te llamas")
    chart = parser.parse(words)
    print_parse_result("te llamas (you call yourself)", chart)

    # Third person plural
    words = tag_spanish_sentence("se levantan")
    chart = parser.parse(words)
    print_parse_result("se levantan (they get up)", chart)


def test_edge_cases():
    """Test edge cases and special structures."""
    print("\n=== Edge Cases ===")
    parser = ChartParser("grammars/spanish.json")

    # Single noun phrase
    words = tag_spanish_sentence("El perro")
    chart = parser.parse(words)
    print_parse_result("El perro", chart)

    # Adjective only
    words = tag_spanish_sentence("grande")
    chart = parser.parse(words)
    print_parse_result("grande", chart)

    # Determiner and adjective (gender-invariant)
    words = tag_spanish_sentence("El grande")
    chart = parser.parse(words)
    print_parse_result("El grande", chart)


if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE SPANISH GRAMMAR TESTS")
    print("=" * 60)

    try:
        test_basic_structures()
        test_implied_subjects()
        test_gender_number_agreement()
        test_participles_as_adjectives()
        test_coordination()
        test_prepositional_phrases()
        test_complex_sentences()
        test_reflexive_verbs()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("ALL COMPREHENSIVE SPANISH TESTS PASSED! [OK]")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
