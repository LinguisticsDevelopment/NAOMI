"""
Chart Parser Implementation - Order-independent parsing with active chart algorithm.

Uses predict/scan/complete operations to explore all rule application orders.
"""

from typing import List
from .chart_structures import Chart, ChartEdge, ActiveEdge
from .data_structures import Word, Node, Hypothesis, ParseChart, ParserConfig, create_initial_chart
from .enums import TAG_TO_NODE_TYPE, NodeType
from .dsl import Grammar, load_grammar
from .matcher import matches_pattern
from .scorer import score_hypothesis
import copy


class ChartParser:
    """
    Active Chart Parser adapted for directional grammar rules.

    Maintains passive edges (complete constituents) and active edges (partial rules).
    Processes agenda until all possible parses are found.
    """

    def __init__(self, grammar_path: str, config: ParserConfig = None):
        """Initialize parser with grammar."""
        self.grammar = load_grammar(grammar_path)
        self.config = config if config else ParserConfig()

    def parse(self, words: List[Word]) -> ParseChart:
        """
        Parse a sentence using chart parsing.

        Args:
            words: Input sentence words

        Returns:
            ParseChart with all valid parse hypotheses
        """
        # Initialize chart with word nodes
        chart = self._initialize_chart(words)

        # Process agenda until empty
        while chart.agenda:
            item_type, item = chart.agenda.pop(0)

            if item_type == 'passive':
                self._complete(chart, item)
            elif item_type == 'active':
                if item.is_complete():
                    self._create_passive(chart, item)
                else:
                    self._scan(chart, item)

        # Extract final hypotheses
        hypotheses = self._extract_hypotheses(chart)

        # Create ParseChart
        parse_chart = ParseChart(
            words=words,
            nodes=[],  # Nodes are in hypotheses
            hypotheses=hypotheses,
            config=self.config
        )

        # Score and prune
        for hyp in parse_chart.hypotheses:
            hyp.score = score_hypothesis(hyp, None)

        parse_chart.sort_hypotheses()

        # Apply beam search pruning
        if len(parse_chart.hypotheses) > self.config.max_hypotheses:
            parse_chart.prune_hypotheses()

        return parse_chart

    def _initialize_chart(self, words: List[Word]) -> Chart:
        """
        Initialize chart with passive edges for each word.

        For POS ambiguity, create separate passive edges for each POS possibility.
        """
        chart = Chart(words=words, config=self.config)

        # Handle POS ambiguity if enabled
        if self.config.enable_pos_ambiguity:
            from .pos_tagger import get_possible_tags

            # For each word, create passive edges for all possible POS tags
            for i, word in enumerate(words):
                possible_tags = get_possible_tags(word)

                for tag in possible_tags:
                    node_type = TAG_TO_NODE_TYPE.get(tag, NodeType.NIL)

                    # Create node with this POS tag
                    node = Node(
                        type=node_type,
                        original_type=node_type,
                        value=word,
                        pos=tag,
                        flags=word.subtypes.copy(),
                        index=i
                    )

                    # Create hypothesis for this single node
                    hyp = Hypothesis(
                        nodes=[node],
                        edges=[],
                        consumed=set(),
                        score=0.0
                    )

                    # Create passive edge
                    edge = ChartEdge(
                        hypothesis=hyp,
                        start=i,
                        end=i + 1,
                        node_type=node_type
                    )

                    chart.add_passive_edge(edge)
        else:
            # Original behavior - single POS per word
            for i, word in enumerate(words):
                node_type = TAG_TO_NODE_TYPE.get(word.pos, NodeType.NIL)

                node = Node(
                    type=node_type,
                    original_type=node_type,
                    value=word,
                    pos=word.pos,
                    flags=word.subtypes.copy(),
                    index=i
                )

                hyp = Hypothesis(
                    nodes=[node],
                    edges=[],
                    consumed=set(),
                    score=0.0
                )

                edge = ChartEdge(
                    hypothesis=hyp,
                    start=i,
                    end=i + 1,
                    node_type=node_type
                )

                chart.add_passive_edge(edge)

        return chart

    def _predict(self, chart: Chart, passive_edge: ChartEdge):
        """
        PREDICT: Create active edges for rules that could use this constituent.

        For each rule in the grammar, check if passive_edge matches any pattern element.
        If so, create an active edge starting from that match.
        """
        for ruleset in self.grammar.rulesets.values():
            for rule in ruleset.rules:
                # Check if passive_edge could match anchor
                if self._matches_pattern_element(passive_edge, rule.pattern.anchor):
                    self._create_active_at_anchor(chart, rule, passive_edge)

                # Check if passive_edge could match first before element
                if rule.pattern.before:
                    first_before = rule.pattern.before[0]
                    if self._matches_pattern_element(passive_edge, first_before):
                        self._create_active_at_before(chart, rule, passive_edge)

                # Check if passive_edge could match first after element
                if rule.pattern.after:
                    first_after = rule.pattern.after[0]
                    if self._matches_pattern_element(passive_edge, first_after):
                        self._create_active_at_after(chart, rule, passive_edge)

    def _matches_pattern_element(self, edge: ChartEdge, pattern) -> bool:
        """Check if a passive edge could match a pattern element."""
        if edge.node_type != pattern.type:
            return False

        # Get the node from the hypothesis
        unconsumed = edge.hypothesis.get_unconsumed()
        if not unconsumed:
            return False

        node = edge.hypothesis.nodes[list(unconsumed)[0]]

        # Check subtypes
        for required_subtype in pattern.subtypes:
            if not node.has_subtype(required_subtype):
                return False

        # Note: Subcategory checking is deferred until we have an anchor to compare against
        return True

    def _create_active_at_anchor(self, chart: Chart, rule, passive_edge: ChartEdge):
        """Create active edge with passive_edge as anchor."""
        active = ActiveEdge(
            rule=rule,
            start=passive_edge.start,
            end=passive_edge.end,
            anchor_edge=passive_edge
        )
        chart.add_active_edge(active)

    def _create_active_at_before(self, chart: Chart, rule, passive_edge: ChartEdge):
        """Create active edge with passive_edge as first before element."""
        active = ActiveEdge(
            rule=rule,
            start=passive_edge.start,
            end=passive_edge.end,
            before_edges=[passive_edge]
        )
        chart.add_active_edge(active)

    def _create_active_at_after(self, chart: Chart, rule, passive_edge: ChartEdge):
        """Create active edge with passive_edge as first after element (needs anchor first)."""
        # Can't start with after - must have anchor first
        # This case is handled by _scan when anchor is matched
        pass

    def _scan(self, chart: Chart, active_edge: ActiveEdge):
        """
        SCAN: Try to extend active edge by matching next pattern element.

        Looks for passive edges that could match the next needed constituent.
        """
        location, pattern = active_edge.get_next_pattern_element()
        if not pattern:
            return

        # Determine where to look for matching constituent
        if location == 'before':
            # Before elements go right-to-left, so look before current start
            # For now, simplified: look at position just before start
            search_end = active_edge.start
            search_start = max(0, search_end - 1)

            matches = chart.get_passive_edges_at(search_start, search_end, pattern.type)

            for match_edge in matches:
                if self._check_pattern_match(match_edge, pattern, active_edge.anchor_edge):
                    new_active = ActiveEdge(
                        rule=active_edge.rule,
                        start=match_edge.start,  # Extend start backwards
                        end=active_edge.end,
                        before_edges=[match_edge] + active_edge.before_edges,
                        anchor_edge=active_edge.anchor_edge,
                        after_edges=active_edge.after_edges.copy()
                    )
                    chart.add_active_edge(new_active)

        elif location == 'anchor':
            # Anchor is at current position
            search_start = active_edge.start
            # For anchor, we need to look at all possible spans starting here
            # Simplified: look for single-word constituents for now
            search_end = active_edge.start + 1

            matches = chart.get_passive_edges_at(search_start, search_end, pattern.type)

            for match_edge in matches:
                if self._check_pattern_match(match_edge, pattern, None):  # No anchor to check against yet
                    new_active = ActiveEdge(
                        rule=active_edge.rule,
                        start=active_edge.start,
                        end=match_edge.end,
                        before_edges=active_edge.before_edges.copy(),
                        anchor_edge=match_edge,
                        after_edges=active_edge.after_edges.copy()
                    )
                    chart.add_active_edge(new_active)

        elif location == 'after':
            # After elements go left-to-right, so look after current end
            search_start = active_edge.end
            # Look for constituents starting at active_edge.end
            # Try all possible end positions
            for possible_end in range(search_start + 1, len(chart.words) + 1):
                matches = chart.get_passive_edges_at(search_start, possible_end, pattern.type)

                for match_edge in matches:
                    if self._check_pattern_match(match_edge, pattern, active_edge.anchor_edge):
                        new_active = ActiveEdge(
                            rule=active_edge.rule,
                            start=active_edge.start,
                            end=match_edge.end,  # Extend end forward
                            before_edges=active_edge.before_edges.copy(),
                            anchor_edge=active_edge.anchor_edge,
                            after_edges=active_edge.after_edges + [match_edge]
                        )
                        chart.add_active_edge(new_active)

    def _check_pattern_match(self, edge: ChartEdge, pattern, anchor_edge: ChartEdge) -> bool:
        """Check if edge matches pattern requirements (subtypes and subcategories)."""
        unconsumed = edge.hypothesis.get_unconsumed()
        if not unconsumed:
            return False

        node = edge.hypothesis.nodes[list(unconsumed)[0]]

        # Check subtypes
        for required_subtype in pattern.subtypes:
            if not node.has_subtype(required_subtype):
                return False

        # Check subcategories (agreement with anchor)
        if anchor_edge and pattern.subcategories:
            anchor_unconsumed = anchor_edge.hypothesis.get_unconsumed()
            if anchor_unconsumed:
                anchor_node = anchor_edge.hypothesis.nodes[list(anchor_unconsumed)[0]]

                for required_subcat in pattern.subcategories:
                    anchor_value = anchor_node.get_subcategory_value(required_subcat)
                    node_value = node.get_subcategory_value(required_subcat)

                    if anchor_value != node_value:
                        return False

        return True

    def _complete(self, chart: Chart, passive_edge: ChartEdge):
        """
        COMPLETE: Use completed passive edge to advance active edges.

        Also triggers PREDICT to create new active edges.
        """
        # PREDICT new active edges
        self._predict(chart, passive_edge)

        # Advance existing active edges that need this constituent
        # (This is handled by SCAN when active edges are processed from agenda)

    def _create_passive(self, chart: Chart, active_edge: ActiveEdge):
        """
        CREATE_PASSIVE: Convert completed active edge into passive edge.

        Applies the rule transformation to create the new constituent.
        """
        # Merge hypotheses from all matched edges
        hypothesis = self._apply_rule_to_edges(
            active_edge.before_edges,
            active_edge.anchor_edge,
            active_edge.after_edges,
            active_edge.rule
        )

        # Create passive edge
        passive = ChartEdge(
            hypothesis=hypothesis,
            start=active_edge.start,
            end=active_edge.end,
            node_type=active_edge.rule.result
        )

        chart.add_passive_edge(passive)

        # If rule is recursive, immediately predict with result
        if active_edge.rule.recursive:
            self._predict(chart, passive)

    def _apply_rule_to_edges(self, before_edges, anchor_edge, after_edges, rule) -> Hypothesis:
        """
        Apply rule transformation to matched edges.

        Merges constituent hypotheses and applies rule logic.
        """
        # This is simplified - full implementation would use apply_rule() from quantum_parser
        # For now, create a basic merged hypothesis

        # Combine all nodes
        all_nodes = []
        all_edges = []
        node_offset = 0
        node_mapping = {}

        # Add before nodes
        for edge in before_edges:
            for node in edge.hypothesis.nodes:
                old_idx = node.index
                new_node = copy.deepcopy(node)
                new_node.index = len(all_nodes)
                node_mapping[(id(edge), old_idx)] = len(all_nodes)
                all_nodes.append(new_node)

            # Add edges with remapped indices
            for e in edge.hypothesis.edges:
                new_edge = copy.deepcopy(e)
                new_edge.parent = node_mapping[(id(edge), e.parent)]
                new_edge.child = node_mapping[(id(edge), e.child)]
                all_edges.append(new_edge)

        # Add anchor node
        anchor_idx_in_result = None
        if anchor_edge:
            for node in anchor_edge.hypothesis.nodes:
                old_idx = node.index
                new_node = copy.deepcopy(node)
                new_node.index = len(all_nodes)
                node_mapping[(id(anchor_edge), old_idx)] = len(all_nodes)

                # Transform anchor node type
                if new_node.original_type == new_node.type:  # Not yet transformed
                    anchor_idx_in_result = len(all_nodes)
                    new_node.type = rule.result
                all_nodes.append(new_node)

            for e in anchor_edge.hypothesis.edges:
                new_edge = copy.deepcopy(e)
                new_edge.parent = node_mapping[(id(anchor_edge), e.parent)]
                new_edge.child = node_mapping[(id(anchor_edge), e.child)]
                all_edges.append(new_edge)

        # Add after nodes
        for edge in after_edges:
            for node in edge.hypothesis.nodes:
                old_idx = node.index
                new_node = copy.deepcopy(node)
                new_node.index = len(all_nodes)
                node_mapping[(id(edge), old_idx)] = len(all_nodes)
                all_nodes.append(new_node)

            for e in edge.hypothesis.edges:
                new_edge = copy.deepcopy(e)
                new_edge.parent = node_mapping[(id(edge), e.parent)]
                new_edge.child = node_mapping[(id(edge), e.child)]
                all_edges.append(new_edge)

        # Add connections from rule
        if anchor_idx_in_result is not None:
            for conn in rule.connections:
                # Simplified connection creation
                # TODO: Full implementation of connection logic
                pass

        # Mark consumed nodes
        consumed = set()
        if rule.consume:
            # Mark before/after nodes as consumed
            for i, node in enumerate(all_nodes):
                if i != anchor_idx_in_result:
                    consumed.add(i)

        hypothesis = Hypothesis(
            nodes=all_nodes,
            edges=all_edges,
            consumed=consumed,
            score=0.0
        )

        return hypothesis

    def _extract_hypotheses(self, chart: Chart) -> List[Hypothesis]:
        """Extract final parse hypotheses from chart."""
        hypotheses = []

        # Look for passive edges spanning entire sentence
        full_span = (0, len(chart.words))

        if full_span in chart.passive_edges:
            for edge in chart.passive_edges[full_span]:
                # Check if it has only one unconsumed node (the root)
                unconsumed = edge.hypothesis.get_unconsumed()
                if len(unconsumed) == 1:
                    hypotheses.append(edge.hypothesis)

        # If no complete parses, return best partial parses
        if not hypotheses:
            # Find longest spans
            max_span = 0
            for (start, end) in chart.passive_edges:
                span_len = end - start
                if span_len > max_span:
                    max_span = span_len

            # Return edges with longest spans
            for (start, end), edges in chart.passive_edges.items():
                if end - start == max_span:
                    for edge in edges:
                        if edge.hypothesis:
                            hypotheses.append(edge.hypothesis)

        return hypotheses
