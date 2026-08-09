"""
Pattern matching for quantum parser.

Implements find_matches and related functions for matching grammar rules to nodes.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from .data_structures import Node, Hypothesis
from .dsl import Rule, PatternElement
from .enums import SubCat, get_subcat


@dataclass
class Match:
    """
    A successful rule match.

    Attributes:
        anchor_idx: Index of anchor node
        before_indices: Indices of matched before elements
        after_indices: Indices of matched after elements
        rule: The rule that matched
    """
    anchor_idx: int
    before_indices: List[int]
    after_indices: List[int]
    rule: Rule

    def __repr__(self) -> str:
        return f"Match(anchor={self.anchor_idx}, before={self.before_indices}, after={self.after_indices})"


def find_matches(hypothesis: Hypothesis, anchor_idx: int, rule: Rule) -> List[Match]:
    """
    Find all ways this rule can match at anchor node.

    Args:
        hypothesis: Current parse hypothesis
        anchor_idx: Index of potential anchor node
        rule: Rule to try matching

    Returns:
        List of Match objects (may be empty if no matches)
    """
    matches = []

    anchor = hypothesis.nodes[anchor_idx]

    # Check if anchor matches rule's anchor pattern
    if not matches_pattern(anchor, rule.anchor, anchor):
        return []

    # Try to find "before" elements — each element searches left from the
    # previously matched element (chained, innermost first).
    before_match_sets = []
    before_search_start = anchor_idx
    for before_pattern in rule.before:
        candidates = search_direction(
            hypothesis,
            before_search_start,
            direction="left",
            pattern=before_pattern,
            anchor_node=anchor
        )
        if not candidates:
            return []  # Required element not found
        before_match_sets.append(candidates)
        # Next before element searches further left from the leftmost match found
        before_search_start = min(candidates)

    # Try to find "after" elements — each element searches right from the
    # previously matched element (chained), so after[1] starts past after[0].
    after_match_sets = []
    after_search_start = anchor_idx
    for after_pattern in rule.after:
        candidates = search_direction(
            hypothesis,
            after_search_start,
            direction="right",
            pattern=after_pattern,
            anchor_node=anchor
        )
        if not candidates:
            return []  # Required element not found
        after_match_sets.append(candidates)
        # Next after element searches further right from the rightmost match found
        after_search_start = max(candidates)

    # Generate all combinations of matched elements
    # For now, just use first match from each set (simplification)
    # TODO: Implement full combinatorial matching for multiple hypotheses
    before_indices = [candidates[0] for candidates in before_match_sets] if before_match_sets else []
    after_indices = [candidates[0] for candidates in after_match_sets] if after_match_sets else []

    # Optional absolute-position requirements (e.g. "anchor must be the
    # sentence's first word" for subject-aux inversion). Empty for every
    # pre-existing rule, so this is a no-op unless a rule opts in.
    if rule.position_constraints:
        for ref, required_idx in rule.position_constraints.items():
            actual_idx = _resolve_position_ref(ref, anchor_idx, before_indices, after_indices)
            if actual_idx != required_idx:
                return []

    match = Match(
        anchor_idx=anchor_idx,
        before_indices=before_indices,
        after_indices=after_indices,
        rule=rule
    )
    matches.append(match)

    return matches


def _resolve_position_ref(ref: str, anchor_idx: int, before_indices: List[int],
                           after_indices: List[int]) -> Optional[int]:
    """Resolve a rule reference ("anchor"/"before[N]"/"after[N]") to a node index."""
    if ref == "anchor":
        return anchor_idx
    if ref.startswith("before["):
        i = int(ref[7:-1])
        return before_indices[i] if 0 <= i < len(before_indices) else None
    if ref.startswith("after["):
        i = int(ref[6:-1])
        return after_indices[i] if 0 <= i < len(after_indices) else None
    return None


def matches_pattern(node: Node, pattern: PatternElement, anchor_node: Node) -> bool:
    """
    Check if node matches pattern requirements.

    Args:
        node: Node to check
        pattern: Pattern to match against
        anchor_node: The anchor node (for subcategory matching)

    Returns:
        True if node matches pattern
    """
    # Type check
    if pattern.original_type is not None:
        # Check original type instead of current type
        if node.original_type != pattern.type:
            return False
    else:
        # Check current type
        if node.type != pattern.type:
            return False

    # SubType check (AND logic: node must have ALL specified subtypes)
    for required_subtype in pattern.subtypes:
        if not node.has_subtype(required_subtype):
            return False

    # SubCategory check (must match anchor node's values)
    for required_subcat in pattern.subcategories:
        # Get anchor's value for this subcategory
        anchor_value = anchor_node.get_subcategory_value(required_subcat)

        # Get this node's value for this subcategory
        node_value = node.get_subcategory_value(required_subcat)

        # Must match (or both be None)
        if anchor_value != node_value:
            return False

    return True


def search_direction(
    hypothesis: Hypothesis,
    anchor_idx: int,
    direction: str,
    pattern: PatternElement,
    anchor_node: Node
) -> List[int]:
    """
    Search left or right from anchor for nodes matching pattern.

    Args:
        hypothesis: Current parse hypothesis
        anchor_idx: Index of anchor node
        direction: "left" or "right"
        pattern: Pattern to match
        anchor_node: The anchor node (for subcategory matching)

    Returns:
        List of node indices matching pattern (respects quantifier)
    """
    step = -1 if direction == "left" else 1
    matches = []

    current_idx = anchor_idx + step

    while 0 <= current_idx < len(hypothesis.nodes):
        node = hypothesis.nodes[current_idx]

        # Skip consumed nodes
        if current_idx in hypothesis.consumed:
            current_idx += step
            continue

        # Check if matches pattern
        if matches_pattern(node, pattern, anchor_node):
            matches.append(current_idx)

            # Respect quantifier
            if pattern.quantifier == "one":
                return matches  # Found one, stop

            elif pattern.quantifier == "all":
                # Continue searching for all consecutive matches
                current_idx += step
                continue

            elif pattern.quantifier == "one_or_more":
                # Found at least one, return it
                # (Could continue searching, but for simplicity return first)
                return matches

        else:
            # Non-matching node encountered
            if pattern.quantifier == "all":
                # Already collected all consecutive matches
                return matches
            else:
                # Required match not found
                return []

        current_idx += step

    # Reached end of sentence
    if pattern.quantifier in ["all", "one_or_more"] and matches:
        return matches
    else:
        return []
