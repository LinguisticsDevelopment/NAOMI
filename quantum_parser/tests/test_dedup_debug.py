"""Debug deduplication issue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.parser import ChartParser
from src.parser.pos_tagger import tag_sentence


def debug_deduplication():
    """Debug why two different structures are being deduplicated."""
    parser = ChartParser("grammars/english.json")
    words = tag_sentence("old men and women")

    from src.parser.data_structures import create_initial_chart
    chart = create_initial_chart(words, parser.config)

    schedules = parser._generate_schedules()

    # Get hypotheses from each schedule
    hyp1_list = parser._parse_with_schedule(chart, schedules[0])
    hyp2_list = parser._parse_with_schedule(chart, schedules[1])

    print(f"Schedule 1 produced {len(hyp1_list)} hypotheses")
    print(f"Schedule 2 produced {len(hyp2_list)} hypotheses")
    print()

    if len(hyp1_list) > 0 and len(hyp2_list) > 0:
        hyp1 = hyp1_list[0]
        hyp2 = hyp2_list[0]

        print("Hypothesis 1 edges:")
        edge_set1 = set((e.type, e.parent, e.child) for e in hyp1.edges)
        for edge_tuple in edge_set1:
            print(f"  {edge_tuple}")
        print()

        print("Hypothesis 2 edges:")
        edge_set2 = set((e.type, e.parent, e.child) for e in hyp2.edges)
        for edge_tuple in edge_set2:
            print(f"  {edge_tuple}")
        print()

        print(f"Edge sets equal: {edge_set1 == edge_set2}")
        print(f"is_equivalent: {hyp1.is_equivalent(hyp2)}")
        print()

        # Check each component
        print("Node types match:")
        for i, (n1, n2) in enumerate(zip(hyp1.nodes, hyp2.nodes)):
            match = n1.type == n2.type
            print(f"  [{i}] {n1.type.name} vs {n2.type.name}: {match}")
        print()

        print(f"Consumed sets: {hyp1.consumed} vs {hyp2.consumed}")
        print(f"Consumed match: {hyp1.consumed == hyp2.consumed}")


if __name__ == "__main__":
    debug_deduplication()
