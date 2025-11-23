"""Test structural ambiguity - scope of modification."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.parser import QuantumParser, print_hypothesis_tree
from src.parser.pos_tagger import tag_sentence


def test_old_men_and_women():
    """Test: old men and women - does 'old' modify both or just 'men'?"""
    print("=== Test: old men and women ===")
    print("Expected 2+ interpretations:")
    print("  1. (old men) and women  - 'old' modifies only 'men'")
    print("  2. old (men and women)  - 'old' modifies both")
    print()

    parser = QuantumParser("grammars/english.json")
    words = tag_sentence("old men and women")

    chart = parser.parse(words)
    print(f"Total hypotheses: {len(chart.hypotheses)}")
    print()

    # Sort and examine top hypotheses
    chart.sort_hypotheses()

    # Look for the two different structures
    interpretations = {}

    for i, hyp in enumerate(chart.hypotheses[:10]):
        unconsumed = hyp.get_unconsumed()

        # Check structure by examining edges
        # Find if 'old' modifies 'men' directly or modifies coordination
        old_node_idx = 0  # 'old' is first word
        men_node_idx = 1  # 'men' is second word
        and_node_idx = 2  # 'and' is third word

        # Find what 'old' is connected to
        old_modifies = None
        for edge in hyp.edges:
            if edge.child == old_node_idx:
                parent_node = hyp.nodes[edge.parent]
                old_modifies = (edge.parent, parent_node.value.text if parent_node.value else parent_node.type.name)

        if old_modifies:
            key = f"old->{old_modifies[1]}"
            if key not in interpretations:
                interpretations[key] = hyp

    print(f"Distinct structural interpretations found: {len(interpretations)}")
    print()

    # Show each interpretation
    for idx, (key, hyp) in enumerate(interpretations.items(), 1):
        print(f"Interpretation {idx}: {key}")
        print(f"Score: {hyp.score:.3f}")
        print_hypothesis_tree(hyp)
        print()

    # Check for specific structures
    has_old_men_structure = False  # old modifies men directly
    has_old_coord_structure = False  # old modifies coordination result

    for hyp in chart.hypotheses:
        old_node_idx = 0
        men_node_idx = 1
        and_node_idx = 2

        # Check edges to see structure
        old_parent = None
        and_type = None

        for edge in hyp.edges:
            if edge.child == old_node_idx:
                old_parent = edge.parent
                old_parent_orig = hyp.nodes[edge.parent].original_type.name

        for node in hyp.nodes:
            if node.index == and_node_idx:
                and_type = node.type.name

        # If old modifies node at index 1 (men), that's interpretation 1
        if old_parent == men_node_idx:
            has_old_men_structure = True

        # If old modifies a node that resulted from coordination, that's interpretation 2
        if old_parent is not None and old_parent == and_node_idx and and_type == "NOMINAL":
            has_old_coord_structure = True

    print("=" * 60)
    if has_old_men_structure:
        print("[OK] Found interpretation: old modifies 'men' only")
    else:
        print("[MISSING] Interpretation where old modifies 'men' only")

    if has_old_coord_structure:
        print("[OK] Found interpretation: old modifies coordination")
    else:
        print("[MISSING] Interpretation where old modifies coordination")
    print()


def test_saw_boy_with_telescope():
    """Test: the man saw the boy with the telescope - PP attachment ambiguity"""
    print("=== Test: the man saw the boy with the telescope ===")
    print("Expected 2+ interpretations:")
    print("  1. saw (the boy with the telescope)  - PP modifies 'boy'")
    print("  2. saw (the boy) (with the telescope) - PP modifies 'saw'")
    print()

    parser = QuantumParser("grammars/english.json")
    words = tag_sentence("the man saw the boy with the telescope")

    chart = parser.parse(words)
    print(f"Total hypotheses: {len(chart.hypotheses)}")
    print()

    chart.sort_hypotheses()

    # Show top 3 parses
    for i, hyp in enumerate(chart.hypotheses[:3]):
        unconsumed = hyp.get_unconsumed()
        print(f"Hypothesis {i+1} (Score: {hyp.score:.3f}, Unconsumed: {len(unconsumed)}):")

        # Find where "with the telescope" attaches
        with_idx = 4  # "with" is 5th word (index 4)

        # Find what 'with' PP is connected to
        with_parent = None
        for edge in hyp.edges:
            # Find PP node that contains 'with'
            if edge.child == with_idx or hyp.nodes[edge.child].index == with_idx:
                with_parent = edge.parent
                parent_node = hyp.nodes[edge.parent]
                print(f"  PP attachment: {parent_node.value.text if parent_node.value else parent_node.type.name}")

        if len(unconsumed) <= 2:
            print_hypothesis_tree(hyp)
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("STRUCTURAL AMBIGUITY TESTS")
    print("=" * 60)
    print()

    try:
        test_old_men_and_women()
        test_saw_boy_with_telescope()

        print("=" * 60)
        print("STRUCTURAL AMBIGUITY TESTS COMPLETED")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
