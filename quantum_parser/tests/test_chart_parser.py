"""Test chart parser on structural ambiguity cases."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.parser import ChartParser, print_hypothesis_tree
from src.parser.pos_tagger import tag_sentence


def test_old_men_and_women():
    """Test: old men and women - does 'old' modify both or just 'men'?"""
    print("=" * 60)
    print("TEST: old men and women")
    print("=" * 60)
    print()
    print("Expected 2+ interpretations:")
    print("  1. (old men) and women  - 'old' modifies only 'men'")
    print("  2. old (men and women)  - 'old' modifies both")
    print()

    parser = ChartParser("grammars/english.json")
    words = tag_sentence("old men and women")

    print("Input words:")
    for i, w in enumerate(words):
        print(f"  {i}: {w.text} ({w.pos.name})")
    print()

    chart = parser.parse(words)
    print(f"Total hypotheses: {len(chart.hypotheses)}")
    print()

    # First, show ALL hypotheses with full detail
    print("All hypotheses:")
    for idx, hyp in enumerate(chart.hypotheses):
        unconsumed = hyp.get_unconsumed()
        print(f"\n  Hypothesis {idx+1} (Score: {hyp.score:.3f}, Unconsumed: {len(unconsumed)}):")

        # Show edges
        print("    Edges:")
        for edge in hyp.edges:
            parent_text = hyp.nodes[edge.parent].value.text if hyp.nodes[edge.parent].value else f"node{edge.parent}"
            child_text = hyp.nodes[edge.child].value.text if hyp.nodes[edge.child].value else f"node{edge.child}"
            print(f"      {child_text}[{edge.child}] --{edge.type.name}--> {parent_text}[{edge.parent}]")
    print()

    # Analyze structures
    interpretations = {}

    for i, hyp in enumerate(chart.hypotheses):
        unconsumed = hyp.get_unconsumed()

        # Check structure by examining what 'old' connects to
        old_node_idx = 0  # 'old' is first word
        men_node_idx = 1  # 'men' is second word
        and_node_idx = 2  # 'and' is third word

        # Find what 'old' is connected to (old is the parent in DESCRIPTION edges)
        old_child = None
        old_child_type = None

        for edge in hyp.edges:
            if edge.parent == old_node_idx:  # old is the parent
                old_child = edge.child
                child_node = hyp.nodes[edge.child]
                old_child_type = child_node.type.name

        # Find what the 'and' node became
        and_node_type = hyp.nodes[and_node_idx].type.name

        # Create structural key
        if old_child is not None:
            key = f"old modifies {old_child_type}[{old_child}], and={and_node_type}"
        else:
            key = f"old modifies NONE, and={and_node_type}"

        if key not in interpretations:
            interpretations[key] = hyp

    print(f"Distinct structural interpretations: {len(interpretations)}")
    print()

    # Show each interpretation
    for idx, (key, hyp) in enumerate(interpretations.items(), 1):
        unconsumed = hyp.get_unconsumed()
        print(f"Interpretation {idx}: {key}")
        print(f"Score: {hyp.score:.3f}")
        print(f"Unconsumed nodes: {len(unconsumed)}")

        # Show node transformations
        print("Nodes:")
        for i, node in enumerate(hyp.nodes):
            text = node.value.text if node.value else "constructed"
            orig = node.original_type.name
            curr = node.type.name
            consumed = "consumed" if i not in unconsumed else "UNCONSUMED"
            if orig != curr:
                print(f"  [{i}] {text}: {orig} -> {curr} ({consumed})")
            else:
                print(f"  [{i}] {text}: {curr} ({consumed})")

        # Show edges
        print("Edges:")
        if hyp.edges:
            for edge in hyp.edges:
                parent_text = hyp.nodes[edge.parent].value.text if hyp.nodes[edge.parent].value else f"node{edge.parent}"
                child_text = hyp.nodes[edge.child].value.text if hyp.nodes[edge.child].value else f"node{edge.child}"
                print(f"  {child_text}[{edge.child}] --{edge.type.name}--> {parent_text}[{edge.parent}]")
        else:
            print("  (no edges)")

        print()

    # Check for specific structures
    has_old_men_structure = False  # old modifies men directly
    has_old_coord_structure = False  # old modifies coordination result

    for hyp in chart.hypotheses:
        old_node_idx = 0
        men_node_idx = 1
        and_node_idx = 2

        # Check edges to see structure (old is parent in DESCRIPTION edges)
        old_child = None

        for edge in hyp.edges:
            if edge.parent == old_node_idx:  # old is the parent
                old_child = edge.child

        # Check what 'and' became
        and_type = hyp.nodes[and_node_idx].type.name

        # If old modifies node at index 1 (men), that's interpretation 1
        if old_child == men_node_idx:
            has_old_men_structure = True

        # If old modifies a node that resulted from coordination, that's interpretation 2
        if old_child is not None and old_child == and_node_idx and and_type == "NOMINAL":
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

    return len(interpretations) >= 2


if __name__ == "__main__":
    try:
        success = test_old_men_and_women()
        print("=" * 60)
        if success:
            print("TEST PASSED - Multiple structural interpretations found")
        else:
            print("TEST FAILED - Expected at least 2 interpretations")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
