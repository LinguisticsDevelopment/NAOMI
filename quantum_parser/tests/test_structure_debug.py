"""Debug structural ambiguity."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.parser import QuantumParser, print_hypothesis_tree
from src.parser.pos_tagger import tag_sentence


def debug_old_men_and_women():
    """Debug: old men and women"""
    parser = QuantumParser("grammars/english.json")
    words = tag_sentence("old men and women")

    print("Input words:")
    for i, w in enumerate(words):
        print(f"  {i}: {w.text} ({w.pos.name})")
    print()

    chart = parser.parse(words)
    chart.sort_hypotheses()

    print(f"Total hypotheses: {len(chart.hypotheses)}")
    print()

    # Show ALL hypotheses with full detail
    for idx, hyp in enumerate(chart.hypotheses):
        unconsumed = hyp.get_unconsumed()
        print(f"=== Hypothesis {idx+1} (Score: {hyp.score:.3f}, Unconsumed: {len(unconsumed)}) ===")

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


if __name__ == "__main__":
    debug_old_men_and_women()
