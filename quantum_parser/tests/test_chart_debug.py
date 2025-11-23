"""Debug chart parser schedule generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.parser import ChartParser
from src.parser.pos_tagger import tag_sentence


def debug_schedules():
    """Debug what schedules are being generated."""
    parser = ChartParser("grammars/english.json")

    print("Grammar order:")
    print(parser.grammar.order)
    print()

    print("Branching rulesets:")
    print(parser.branching_rulesets)
    print()

    schedules = parser._generate_schedules()
    print(f"Generated {len(schedules)} schedules:")
    for i, schedule in enumerate(schedules):
        print(f"\nSchedule {i+1}:")
        print(schedule)
    print()


def debug_parse_with_schedules():
    """Debug what happens with each schedule."""
    parser = ChartParser("grammars/english.json")
    words = tag_sentence("old men and women")

    from src.parser.data_structures import create_initial_chart
    chart = create_initial_chart(words, parser.config)

    schedules = parser._generate_schedules()

    for i, schedule in enumerate(schedules):
        print(f"\n{'='*60}")
        print(f"SCHEDULE {i+1}: {schedule}")
        print(f"{'='*60}")

        hypotheses = parser._parse_with_schedule(chart, schedule)

        print(f"Generated {len(hypotheses)} hypotheses:")

        for j, hyp in enumerate(hypotheses):
            unconsumed = hyp.get_unconsumed()
            print(f"\nHypothesis {j+1} (Score: {hyp.score:.3f}, Unconsumed: {len(unconsumed)}):")

            # Show node transformations
            print("  Nodes:")
            for idx, node in enumerate(hyp.nodes):
                text = node.value.text if node.value else "constructed"
                orig = node.original_type.name
                curr = node.type.name
                consumed = "C" if idx not in unconsumed else "U"
                if orig != curr:
                    print(f"    [{idx}] {text}: {orig} -> {curr} ({consumed})")
                else:
                    print(f"    [{idx}] {text}: {curr} ({consumed})")

            # Show edges
            print("  Edges:")
            if hyp.edges:
                for edge in hyp.edges:
                    parent_text = hyp.nodes[edge.parent].value.text if hyp.nodes[edge.parent].value else f"node{edge.parent}"
                    child_text = hyp.nodes[edge.child].value.text if hyp.nodes[edge.child].value else f"node{edge.child}"
                    print(f"    {child_text}[{edge.child}] --{edge.type.name}--> {parent_text}[{edge.parent}]")
            else:
                print("    (no edges)")


if __name__ == "__main__":
    debug_schedules()
    debug_parse_with_schedules()
