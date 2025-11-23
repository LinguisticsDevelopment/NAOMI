"""
Visualization tools for parse trees using Graphviz DOT format.
"""

from typing import List, Optional, TextIO
from .data_structures import Hypothesis, ParseChart


def hypothesis_to_dot(hypothesis: Hypothesis, name: str = "parse_tree") -> str:
    """
    Convert a hypothesis to DOT format for Graphviz.

    Args:
        hypothesis: Parse hypothesis to visualize
        name: Name for the graph

    Returns:
        DOT format string
    """
    lines = []
    lines.append(f'digraph {name} {{')
    lines.append('    rankdir=BT;')  # Bottom to top (leaves to root)
    lines.append('    node [shape=box, style=filled];')
    lines.append('')

    # Add nodes
    unconsumed = hypothesis.get_unconsumed()
    for i, node in enumerate(hypothesis.nodes):
        word = node.value.text
        node_type = node.type.name
        orig_type = node.original_type.name

        # Color coding
        if i in unconsumed:
            color = 'lightgreen'  # Unconsumed = root candidates
        else:
            color = 'lightgray'   # Consumed

        # Node type changed?
        if node.type != node.original_type:
            label = f'{word}\\n{orig_type} -> {node_type}'
        else:
            label = f'{word}\\n{node_type}'

        lines.append(f'    n{i} [label="{label}", fillcolor={color}];')

    lines.append('')

    # Add edges
    for edge in hypothesis.edges:
        edge_type = edge.type.name
        parent = edge.parent
        child = edge.child

        lines.append(f'    n{child} -> n{parent} [label="{edge_type}"];')

    lines.append('')
    lines.append(f'    labelloc="t";')
    lines.append(f'    label="Score: {hypothesis.score:.3f} | Edges: {len(hypothesis.edges)} | Unconsumed: {len(unconsumed)}";')
    lines.append('}')

    return '\\n'.join(lines)


def chart_to_dot_multi(chart: ParseChart, top_k: int = 3, name: str = "parse_forest") -> str:
    """
    Create a DOT file showing multiple top hypotheses side-by-side.

    Args:
        chart: ParseChart with multiple hypotheses
        top_k: Number of top hypotheses to show
        name: Name for the graph

    Returns:
        DOT format string with subgraphs
    """
    lines = []
    lines.append(f'digraph {name} {{')
    lines.append('    rankdir=BT;')
    lines.append('    node [shape=box, style=filled];')
    lines.append('    compound=true;')
    lines.append('')

    # Sort and take top K
    chart.sort_hypotheses()
    hypotheses = chart.hypotheses[:top_k]

    for hyp_idx, hyp in enumerate(hypotheses):
        lines.append(f'    subgraph cluster_{hyp_idx} {{')
        lines.append(f'        label="Hypothesis {hyp_idx + 1} (Score: {hyp.score:.3f})";')
        lines.append('        style=dashed;')
        lines.append('')

        # Add nodes for this hypothesis
        unconsumed = hyp.get_unconsumed()
        for i, node in enumerate(hyp.nodes):
            word = node.value.text
            node_type = node.type.name
            orig_type = node.original_type.name

            if i in unconsumed:
                color = 'lightgreen'
            else:
                color = 'lightgray'

            if node.type != node.original_type:
                label = f'{word}\\n{orig_type} -> {node_type}'
            else:
                label = f'{word}\\n{node_type}'

            lines.append(f'        h{hyp_idx}_n{i} [label="{label}", fillcolor={color}];')

        lines.append('')

        # Add edges for this hypothesis
        for edge in hyp.edges:
            edge_type = edge.type.name
            parent = edge.parent
            child = edge.child
            lines.append(f'        h{hyp_idx}_n{child} -> h{hyp_idx}_n{parent} [label="{edge_type}"];')

        lines.append('    }')
        lines.append('')

    lines.append('}')

    return '\\n'.join(lines)


def save_dot(dot_string: str, filepath: str) -> None:
    """Save DOT string to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(dot_string)
    print(f"Saved visualization to {filepath}")
    print(f"To render: dot -Tpng {filepath} -o {filepath.replace('.dot', '.png')}")


def print_hypothesis_tree(hypothesis: Hypothesis, file: Optional[TextIO] = None) -> None:
    """
    Print a text-based tree representation of the hypothesis.

    Args:
        hypothesis: Hypothesis to print
        file: Output file (default: stdout)
    """
    import sys
    if file is None:
        file = sys.stdout

    unconsumed = hypothesis.get_unconsumed()

    print(f"Parse Tree (Score: {hypothesis.score:.3f})", file=file)
    print("=" * 60, file=file)

    # Build parent -> children mapping
    children_map = {i: [] for i in range(len(hypothesis.nodes))}
    for edge in hypothesis.edges:
        children_map[edge.parent].append((edge.child, edge.type.name))

    # Print from roots (unconsumed nodes)
    for root_idx in unconsumed:
        _print_tree_recursive(hypothesis, root_idx, children_map, file=file)

    print(file=file)


def _print_tree_recursive(hyp: Hypothesis, node_idx: int, children_map: dict,
                          prefix: str = "", file: TextIO = None, is_last: bool = True):
    """Recursively print tree structure."""
    import sys
    if file is None:
        file = sys.stdout

    node = hyp.nodes[node_idx]

    # Print current node
    connector = "+-- " if is_last else "|-- "
    node_str = f"{node.value.text} ({node.type.name})"

    if node.type != node.original_type:
        node_str += f" <- {node.original_type.name}"

    print(f"{prefix}{connector}{node_str}", file=file)

    # Print children
    children = children_map.get(node_idx, [])
    for i, (child_idx, edge_type) in enumerate(children):
        is_last_child = (i == len(children) - 1)

        # Print edge label
        edge_prefix = prefix + ("    " if is_last else "|   ")
        edge_connector = "+--" if is_last_child else "|--"
        print(f"{edge_prefix}{edge_connector}[{edge_type}]", file=file)

        # Recurse to child
        child_prefix = prefix + ("    " if is_last else "|   ") + ("    " if is_last_child else "|   ")
        _print_tree_recursive(hyp, child_idx, children_map, child_prefix, file, is_last_child)
