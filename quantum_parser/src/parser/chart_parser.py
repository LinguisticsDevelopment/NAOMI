"""
Simplified Chart Parser - Reuses quantum parser's apply_rule() logic.

This is a hybrid approach: chart-based exploration with quantum parser's rule application.
"""

from typing import List
from .data_structures import Word, Hypothesis, ParseChart, ParserConfig
from .quantum_parser import apply_rule, apply_ruleset_recursively
from .matcher import Match, find_matches
from .scorer import score_hypothesis
from .dsl import load_grammar
import itertools


class ChartParser:
    """
    Chart-inspired parser that explores multiple rule application orders.

    Key insight: Instead of full chart parsing, we generate hypotheses for
    all orderings of branching rulesets, then apply rules as in quantum parser.
    """

    def __init__(self, grammar_path: str, config: ParserConfig = None):
        """Initialize parser with grammar."""
        self.grammar = load_grammar(grammar_path)
        self.config = config if config else ParserConfig()

        # Identify which rulesets involve structural ambiguity
        # These are applied in multiple orders
        self.branching_rulesets = self._identify_branching_rulesets()

    def _identify_branching_rulesets(self) -> List[str]:
        """
        Identify rulesets that create structural ambiguity.

        For now, hardcode the known ambiguous constructions:
        - Coordination (*2 rulesets)
        - PP attachment (noun3, verb1)
        - Modification (noun1 with descriptors)
        """
        branching = []

        for name in self.grammar.order:
            # Coordination rulesets
            if name.endswith('2'):  # adv2, adj2, noun2, etc.
                branching.append(name)

            # PP attachment
            if name in ['noun3', 'verb1', 'adj1']:
                branching.append(name)

            # Descriptor attachment
            if name == 'noun1':
                branching.append(name)

        return branching

    def parse(self, words: List[Word]) -> ParseChart:
        """
        Parse with strategic branching on ambiguous constructions.

        Algorithm:
        1. Identify branching points in grammar order
        2. Generate permutations of branching rulesets
        3. For each permutation, run modified quantum parse
        4. Deduplicate and return all unique hypotheses
        """
        from .data_structures import create_initial_chart

        # Create initial chart with POS ambiguity
        chart = create_initial_chart(words, self.config)

        # Generate rule application schedules
        schedules = self._generate_schedules()

        # Collect hypotheses from all schedules
        all_hypotheses = []

        for schedule in schedules:
            # Run quantum-style parsing with this schedule
            schedule_hypotheses = self._parse_with_schedule(chart, schedule)
            all_hypotheses.extend(schedule_hypotheses)

        # Deduplicate
        unique_hypotheses = self._deduplicate_hypotheses(all_hypotheses)

        # Score and sort
        for hyp in unique_hypotheses:
            hyp.score = score_hypothesis(hyp, chart.embeddings)

        chart.hypotheses = unique_hypotheses
        chart.sort_hypotheses()

        # Filter for complete parses only (exactly 1 unconsumed root node)
        complete_hypotheses = [h for h in chart.hypotheses if len(h.get_unconsumed()) == 1]
        if complete_hypotheses:
            chart.hypotheses = complete_hypotheses

        # Prune
        if len(chart.hypotheses) > self.config.max_hypotheses:
            chart.prune_hypotheses()

        return chart

    def _generate_schedules(self) -> List[List[str]]:
        """
        Generate different rule application orders.

        Strategy: Create a few key permutations of branching rulesets
        while keeping non-branching rulesets in fixed positions.
        """
        # Get base order
        base_order = self.grammar.order.copy()

        # For now, generate 2-3 strategic permutations
        # Full permutation would be factorial(branching) - too many!

        schedules = [base_order]  # Original order

        # If we have coordination and modification, try swapping them
        if 'noun1' in base_order and 'noun2' in base_order:
            # Create schedule where coordination happens before modification
            alt_order = base_order.copy()
            idx1 = alt_order.index('noun1')
            idx2 = alt_order.index('noun2')

            if idx1 < idx2:
                # Swap them
                alt_order[idx1], alt_order[idx2] = alt_order[idx2], alt_order[idx1]
                schedules.append(alt_order)

        return schedules

    def _parse_with_schedule(self, initial_chart: ParseChart, schedule: List[str]) -> List[Hypothesis]:
        """
        Run parsing with a specific ruleset schedule.

        Reuses quantum parser's apply_rule logic with custom schedule.
        """
        from .data_structures import ParseChart

        chart = ParseChart(
            words=initial_chart.words,
            nodes=initial_chart.nodes,
            hypotheses=[hyp.copy() for hyp in initial_chart.hypotheses],
            config=self.config,
            embeddings=initial_chart.embeddings
        )

        # Apply rulesets in given order
        for ruleset_name in schedule:
            if ruleset_name not in self.grammar.rulesets:
                continue

            ruleset = self.grammar.rulesets[ruleset_name]
            new_hypotheses = []

            for current_hyp in chart.hypotheses:
                # Collect all matches for this hypothesis
                all_matches = []

                for unconsumed_idx in current_hyp.get_unconsumed():
                    for rule in ruleset.rules:
                        matches = find_matches(current_hyp, unconsumed_idx, rule)
                        all_matches.extend(matches)

                # Apply smart branching logic (same as quantum parser)
                if len(all_matches) > 1:
                    # Check if matches are independent (different anchors) or conflicting
                    anchor_indices = [m.anchor_idx for m in all_matches]

                    if len(set(anchor_indices)) == len(anchor_indices):
                        # ALL DIFFERENT ANCHORS: Independent transformations
                        # Apply all matches to create a single hypothesis
                        new_hyp = current_hyp
                        for match in all_matches:
                            new_hyp = apply_rule(new_hyp, match)
                            if match.rule.recursive:
                                new_hyp = apply_ruleset_recursively(new_hyp, ruleset)
                        new_hypotheses.append(new_hyp)
                    else:
                        # CONFLICTING ANCHORS: True ambiguity
                        # Create hypothesis for each match
                        for match in all_matches:
                            new_hyp = apply_rule(current_hyp, match)
                            if match.rule.recursive:
                                new_hyp = apply_ruleset_recursively(new_hyp, ruleset)
                            new_hypotheses.append(new_hyp)
                elif len(all_matches) == 1:
                    # Single match: transform in-place
                    match = all_matches[0]
                    new_hyp = apply_rule(current_hyp, match)
                    if match.rule.recursive:
                        new_hyp = apply_ruleset_recursively(new_hyp, ruleset)
                    new_hypotheses.append(new_hyp)
                else:
                    # No matches: keep unchanged
                    new_hypotheses.append(current_hyp)

            # Score all hypotheses
            for hyp in new_hypotheses:
                hyp.score = score_hypothesis(hyp, chart.embeddings)

            # Deduplicate
            deduplicated = []
            for hyp in new_hypotheses:
                is_duplicate = False
                for existing in deduplicated:
                    if hyp.is_equivalent(existing):
                        if hyp.score > existing.score:
                            deduplicated.remove(existing)
                            deduplicated.append(hyp)
                        is_duplicate = True
                        break
                if not is_duplicate:
                    deduplicated.append(hyp)

            chart.hypotheses = deduplicated

            # Prune if needed
            if self.config.score_continuously and len(chart.hypotheses) > self.config.max_hypotheses * 2:
                chart.sort_hypotheses()
                chart.hypotheses = chart.hypotheses[:self.config.max_hypotheses * 2]

        return chart.hypotheses

    def _deduplicate_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Remove structurally equivalent hypotheses."""
        unique = []

        for hyp in hypotheses:
            is_duplicate = False
            for existing in unique:
                if hyp.is_equivalent(existing):
                    # Keep the one with better score (or first one if tied)
                    if hyp.score > existing.score:
                        unique.remove(existing)
                        unique.append(hyp)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(hyp)

        return unique
