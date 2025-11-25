"""
Quantum Parser - Main parsing engine with parallel hypothesis exploration.

Implements the quantum parsing algorithm that maintains multiple parse
interpretations simultaneously.
"""

from typing import List, Optional
from copy import deepcopy

from .data_structures import (
    Word, Node, Edge, Hypothesis, ParseChart,
    create_initial_chart, ParserConfig
)
from .dsl import Grammar, Rule, load_grammar
from .matcher import find_matches, Match
from .scorer import score_hypothesis
from .enums import ConnectionType


class QuantumParser:
    """
    Quantum parser with parallel hypothesis exploration.

    Attributes:
        grammar: Loaded grammar rules
        config: Parser configuration
    """

    def __init__(self, grammar_path: str, config: Optional[ParserConfig] = None):
        """
        Initialize parser with grammar file.

        Args:
            grammar_path: Path to grammar JSON file
            config: Parser configuration (uses default if None)
        """
        self.grammar = load_grammar(grammar_path)
        self.config = config if config is not None else ParserConfig()

    def parse(self, words: List[Word]) -> ParseChart:
        """
        Parse a list of words into a ParseChart with multiple hypotheses.

        Args:
            words: Input sentence as list of Word objects

        Returns:
            ParseChart containing all viable parse hypotheses, ranked by score
        """
        # Validate input
        if not words:
            raise ValueError("Cannot parse empty sentence")

        if len(words) > self.config.max_sentence_length:
            raise ValueError(f"Sentence too long ({len(words)} > {self.config.max_sentence_length})")

        # Create initial chart
        chart = create_initial_chart(words, self.config)

        # Apply rulesets in order
        for ruleset_name in self.grammar.order:
            ruleset = self.grammar.rulesets[ruleset_name]

            # Generate new hypotheses by applying rules
            new_hypotheses = []

            for current_hyp in chart.hypotheses:
                # Collect ALL possible rule matches for this hypothesis
                all_matches = []

                # Try each unconsumed node as potential anchor
                for unconsumed_idx in current_hyp.get_unconsumed():
                    # Try each rule in ruleset
                    for rule in ruleset.rules:
                        # Find all ways this rule can match
                        matches = find_matches(current_hyp, unconsumed_idx, rule)
                        all_matches.extend(matches)

                # SMARTER QUANTUM BRANCHING: Only branch on actual ambiguity
                if len(all_matches) > 1:
                    # Check if matches are independent (different anchors) or conflicting
                    anchor_indices = [m.anchor_idx for m in all_matches]

                    if len(set(anchor_indices)) == len(anchor_indices):
                        # ALL DIFFERENT ANCHORS: Independent transformations
                        # Apply all matches to create a single hypothesis
                        new_hyp = current_hyp
                        for match in all_matches:
                            new_hyp = apply_rule(new_hyp, match)

                            # If recursive rule, keep applying until no more matches
                            if match.rule.recursive:
                                new_hyp = apply_ruleset_recursively(new_hyp, ruleset)

                        new_hypotheses.append(new_hyp)
                    else:
                        # CONFLICTING ANCHORS: True ambiguity (multiple ways to parse same anchor)
                        # Create a hypothesis for each match (parallel exploration)
                        for match in all_matches:
                            new_hyp = apply_rule(current_hyp, match)

                            # If recursive rule, keep applying until no more matches
                            if match.rule.recursive:
                                new_hyp = apply_ruleset_recursively(new_hyp, ruleset)

                            new_hypotheses.append(new_hyp)

                elif len(all_matches) == 1:
                    # SINGLE MATCH: No ambiguity, just transform in-place
                    match = all_matches[0]
                    new_hyp = apply_rule(current_hyp, match)

                    # If recursive rule, keep applying until no more matches
                    if match.rule.recursive:
                        new_hyp = apply_ruleset_recursively(new_hyp, ruleset)

                    new_hypotheses.append(new_hyp)

                else:
                    # NO MATCHES: Keep hypothesis unchanged
                    new_hypotheses.append(current_hyp)

            # Score all hypotheses
            for hyp in new_hypotheses:
                hyp.score = score_hypothesis(hyp, chart.embeddings)

            # DEDUPLICATION: Remove structurally equivalent hypotheses
            deduplicated = []
            for hyp in new_hypotheses:
                # Check if this hypothesis is equivalent to any already added
                is_duplicate = False
                for existing in deduplicated:
                    if hyp.is_equivalent(existing):
                        # Keep the one with better score
                        if hyp.score > existing.score:
                            deduplicated.remove(existing)
                            deduplicated.append(hyp)
                        is_duplicate = True
                        break

                if not is_duplicate:
                    deduplicated.append(hyp)

            # Update chart hypotheses
            chart.hypotheses = deduplicated

            # Prune if configured to score continuously
            if chart.config.score_continuously:
                chart.prune_hypotheses()

        # Final sort by score
        chart.sort_hypotheses()

        # Filter for complete parses only (exactly 1 unconsumed root node)
        complete_hypotheses = [h for h in chart.hypotheses if len(h.get_unconsumed()) == 1]
        if complete_hypotheses:
            chart.hypotheses = complete_hypotheses

        return chart


def apply_rule(hypothesis: Hypothesis, match: Match) -> Hypothesis:
    """
    Apply a rule match to create a new hypothesis.

    Args:
        hypothesis: Original hypothesis
        match: Successful rule match

    Returns:
        New hypothesis with rule applied
    """
    # Create a copy of the hypothesis
    new_hyp = hypothesis.copy()

    # Transform anchor node type
    anchor = new_hyp.nodes[match.anchor_idx]
    anchor.type = match.rule.result

    # Pull categories if specified
    if match.rule.pull_categories:
        for subcat in match.rule.pull_categories:
            # Find a child with this subcategory value
            # (Simplified: just pull from first child)
            for child_idx in match.before_indices + match.after_indices:
                child = new_hyp.nodes[child_idx]
                value = child.get_subcategory_value(subcat)
                if value and value not in anchor.flags:
                    anchor.flags.append(value)
                    break

    # Pop categories if specified
    if match.rule.pop_categories:
        for subcat in match.rule.pop_categories:
            # Remove all flags of this subcategory
            from .enums import SUBTYPE_TO_SUBCAT
            anchor.flags = [
                flag for flag in anchor.flags
                if SUBTYPE_TO_SUBCAT.get(flag) != subcat
            ]

    # Create connections
    for conn_spec in match.rule.connections:
        # Resolve node references
        from_indices = resolve_reference(conn_spec.from_ref, match)
        to_indices = resolve_reference(conn_spec.to_ref, match)

        # Create edges for all combinations
        for from_idx in from_indices:
            for to_idx in to_indices:
                edge = Edge(
                    type=conn_spec.type,
                    parent=from_idx,
                    child=to_idx,
                    source_rule=match.rule.note
                )
                new_hyp.add_edge(edge)

    # Mark nodes as consumed
    if "before" in match.rule.consume:
        for idx in match.before_indices:
            new_hyp.consume(idx)

    if "after" in match.rule.consume:
        for idx in match.after_indices:
            new_hyp.consume(idx)

    if "anchor" in match.rule.consume:
        new_hyp.consume(match.anchor_idx)

    return new_hyp


def resolve_reference(ref: str, match: Match) -> List[int]:
    """
    Resolve a node reference to list of indices.

    Args:
        ref: Reference string ("anchor", "before[0]", "after[*]", etc.)
        match: Rule match containing indices

    Returns:
        List of node indices
    """
    if ref == "anchor":
        return [match.anchor_idx]

    elif ref.startswith("before["):
        # Extract index
        idx_str = ref[7:-1]  # Remove "before[" and "]"

        if idx_str == "*":
            # All before elements
            return match.before_indices
        else:
            # Specific index
            idx = int(idx_str)
            if 0 <= idx < len(match.before_indices):
                return [match.before_indices[idx]]
            else:
                return []

    elif ref.startswith("after["):
        # Extract index
        idx_str = ref[6:-1]  # Remove "after[" and "]"

        if idx_str == "*":
            # All after elements
            return match.after_indices
        else:
            # Specific index
            idx = int(idx_str)
            if 0 <= idx < len(match.after_indices):
                return [match.after_indices[idx]]
            else:
                return []

    else:
        raise ValueError(f"Invalid node reference: {ref}")


def apply_ruleset_recursively(hypothesis: Hypothesis, ruleset) -> Hypothesis:
    """
    Keep applying a ruleset until no more matches are found.

    Args:
        hypothesis: Starting hypothesis
        ruleset: Ruleset to apply recursively

    Returns:
        Hypothesis after exhaustive rule application
    """
    max_iterations = 100  # Prevent infinite loops
    iterations = 0

    current_hyp = hypothesis

    while iterations < max_iterations:
        iterations += 1
        matched = False

        # Try to find a match
        for unconsumed_idx in current_hyp.get_unconsumed():
            for rule in ruleset.rules:
                matches = find_matches(current_hyp, unconsumed_idx, rule)

                if matches:
                    # Apply first match
                    current_hyp = apply_rule(current_hyp, matches[0])
                    matched = True
                    break

            if matched:
                break

        if not matched:
            # No more matches, done
            break

    return current_hyp
