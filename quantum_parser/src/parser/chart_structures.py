"""
Data structures for chart parsing.

Defines passive edges (complete constituents) and active edges (partial rules).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from .data_structures import Hypothesis, Word, ParserConfig
from .enums import NodeType
from .dsl import Rule


@dataclass
class ChartEdge:
    """
    Passive edge representing a complete constituent.

    Spans [start, end) in the input sentence.
    Contains a complete hypothesis with the parse for this span.
    """
    hypothesis: Hypothesis
    start: int
    end: int
    node_type: NodeType
    score: float = 0.0

    def __repr__(self) -> str:
        return f"Passive[{self.start},{self.end}]:{self.node_type.name}(score={self.score:.3f})"

    def __hash__(self):
        # For deduplication - edges with same span and structure are equivalent
        return hash((self.start, self.end, self.node_type, id(self.hypothesis)))


@dataclass
class ActiveEdge:
    """
    Active edge representing a partially applied rule.

    Tracks which pattern elements have been matched and what's needed next.
    """
    rule: Rule
    start: int  # Where this rule application started
    end: int    # Current position (where we're looking for next constituent)

    # Matched constituents
    before_edges: List[ChartEdge] = field(default_factory=list)
    anchor_edge: Optional[ChartEdge] = None
    after_edges: List[ChartEdge] = field(default_factory=list)

    # Hypothesis being built
    hypothesis: Optional[Hypothesis] = None

    def __repr__(self) -> str:
        before_len = len(self.before_edges)
        after_len = len(self.after_edges)
        has_anchor = "A" if self.anchor_edge else "_"
        return f"Active[{self.start},{self.end}]:{self.rule.result.name}(B{before_len},{has_anchor},A{after_len})"

    def is_complete(self) -> bool:
        """Check if all pattern elements have been matched."""
        before_needed = len(self.rule.pattern.before)
        after_needed = len(self.rule.pattern.after)

        before_matched = len(self.before_edges)
        anchor_matched = 1 if self.anchor_edge else 0
        after_matched = len(self.after_edges)

        return (before_matched == before_needed and
                anchor_matched == 1 and
                after_matched == after_needed)

    def get_next_pattern_element(self):
        """Return the next pattern element to match, or None if complete."""
        if self.is_complete():
            return None, None

        before_needed = len(self.rule.pattern.before)
        after_needed = len(self.rule.pattern.after)

        # Match in order: before elements (right-to-left), anchor, after elements
        if len(self.before_edges) < before_needed:
            idx = len(self.before_edges)
            return 'before', self.rule.pattern.before[idx]
        elif not self.anchor_edge:
            return 'anchor', self.rule.pattern.anchor
        elif len(self.after_edges) < after_needed:
            idx = len(self.after_edges)
            return 'after', self.rule.pattern.after[idx]

        return None, None


@dataclass
class Chart:
    """
    Parse chart containing all edges and agenda.

    Passive edges are organized by span for efficient lookup.
    Active edges are processed from the agenda.
    """
    words: List[Word]
    config: ParserConfig

    # Passive edges indexed by (start, end) span
    passive_edges: Dict[Tuple[int, int], List[ChartEdge]] = field(default_factory=dict)

    # Active edges waiting to be completed
    active_edges: List[ActiveEdge] = field(default_factory=list)

    # Agenda for processing (FIFO queue)
    agenda: List = field(default_factory=list)

    # Deduplication sets
    seen_passive: Set[Tuple[int, int, NodeType]] = field(default_factory=set)
    seen_active: Set = field(default_factory=set)

    def add_passive_edge(self, edge: ChartEdge) -> bool:
        """Add a passive edge to the chart."""
        key = (edge.start, edge.end, edge.node_type)

        # Check for duplicates
        if key in self.seen_passive:
            return False

        self.seen_passive.add(key)

        # Add to chart
        span = (edge.start, edge.end)
        if span not in self.passive_edges:
            self.passive_edges[span] = []

        self.passive_edges[span].append(edge)
        self.agenda.append(('passive', edge))

        return True

    def add_active_edge(self, edge: ActiveEdge) -> bool:
        """Add an active edge to the chart."""
        # Create signature for deduplication
        sig = (
            id(edge.rule),
            edge.start,
            edge.end,
            len(edge.before_edges),
            edge.anchor_edge is not None,
            len(edge.after_edges)
        )

        if sig in self.seen_active:
            return False

        self.seen_active.add(sig)
        self.active_edges.append(edge)
        self.agenda.append(('active', edge))

        return True

    def get_passive_edges_at(self, start: int, end: int,
                             node_type: Optional[NodeType] = None) -> List[ChartEdge]:
        """Get passive edges spanning [start, end) of given type."""
        span = (start, end)
        if span not in self.passive_edges:
            return []

        edges = self.passive_edges[span]

        if node_type is None:
            return edges

        return [e for e in edges if e.node_type == node_type]

    def get_active_edges_needing_at(self, position: int,
                                    node_type: Optional[NodeType] = None) -> List[ActiveEdge]:
        """Get active edges that need a constituent at this position."""
        result = []

        for active in self.active_edges:
            if active.end != position:
                continue

            location, pattern = active.get_next_pattern_element()
            if pattern is None:
                continue

            if node_type is None or pattern.type == node_type:
                result.append(active)

        return result

    def __repr__(self) -> str:
        num_passive = sum(len(edges) for edges in self.passive_edges.values())
        return f"Chart(passive={num_passive}, active={len(self.active_edges)}, agenda={len(self.agenda)})"
