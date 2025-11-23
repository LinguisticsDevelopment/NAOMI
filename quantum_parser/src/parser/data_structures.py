"""
Core data structures for quantum parser.

Defines Word, Node, Edge, Hypothesis, and ParseChart.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any
from .enums import Tag, NodeType, ConnectionType, SubType, SubCat


@dataclass
class Word:
    """
    A single word with POS tag and morphological features.

    Attributes:
        text: The word string
        pos: Part-of-speech tag
        subtypes: Morphological features (gender, number, tense, etc.)
    """
    text: str
    pos: Tag
    subtypes: List[SubType] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Word('{self.text}', {self.pos.name})"


@dataclass
class Node:
    """
    A node in the parse tree.

    Can represent:
    - A single word (leaf node)
    - A constituent built from multiple words (internal node)

    Attributes:
        type: Current syntactic type
        original_type: Type before any transformations
        value: Original word (None for constructed nodes)
        pos: Part-of-speech tag
        flags: Accumulated morphological features
        index: Position in original sentence (-1 for constructed nodes)
    """
    type: NodeType
    original_type: NodeType
    value: Optional[Word]
    pos: Tag
    flags: List[SubType] = field(default_factory=list)
    index: int = -1  # Position in sentence

    def __repr__(self) -> str:
        if self.value:
            return f"Node[{self.index}]({self.type.name}, '{self.value.text}')"
        else:
            return f"Node({self.type.name}, constructed)"

    def has_subtype(self, subtype: SubType) -> bool:
        """Check if node has a specific subtype."""
        return subtype in self.flags

    def get_subcategory_value(self, subcat: SubCat) -> Optional[SubType]:
        """
        Get the subtype value for a given subcategory.

        Example: node.get_subcategory_value(SubCat.GENDER) → SubType.FEMININE
        """
        for flag in self.flags:
            from .enums import SUBTYPE_TO_SUBCAT
            if SUBTYPE_TO_SUBCAT.get(flag) == subcat:
                return flag
        return None


@dataclass
class Edge:
    """
    A directed connection between nodes.

    Represents a grammatical relationship (subject, object, modification, etc.)

    Attributes:
        type: Connection type (SUBJECT, OBJECT, etc.)
        parent: Index of parent node in hypothesis.nodes
        child: Index of child node in hypothesis.nodes
        source_rule: Name of rule that created this edge
    """
    type: ConnectionType
    parent: int
    child: int
    source_rule: str = ""

    def __repr__(self) -> str:
        return f"Edge({self.parent} --[{self.type.name}]--> {self.child})"

    def __eq__(self, other) -> bool:
        """Two edges are equal if they connect same nodes with same type."""
        if not isinstance(other, Edge):
            return False
        return (self.type == other.type and
                self.parent == other.parent and
                self.child == other.child)

    def __hash__(self) -> int:
        return hash((self.type, self.parent, self.child))


@dataclass
class Hypothesis:
    """
    A single parse interpretation.

    Represents one way to analyze the sentence structure.

    Attributes:
        nodes: List of nodes (shared reference to ParseChart.nodes)
        edges: List of edges in this interpretation
        consumed: Set of node indices marked as consumed
        score: Semantic coherence score (0.0-1.0, higher is better)
    """
    nodes: List[Node]
    edges: List[Edge] = field(default_factory=list)
    consumed: Set[int] = field(default_factory=set)
    score: float = 0.0

    def get_unconsumed(self) -> List[int]:
        """Return indices of unconsumed nodes."""
        return [i for i in range(len(self.nodes)) if i not in self.consumed]

    def get_root(self) -> Optional[Node]:
        """
        Return root node (should be exactly one unconsumed node at end).

        Returns None if parse is incomplete or ambiguous.
        """
        unconsumed = self.get_unconsumed()
        if len(unconsumed) == 1:
            return self.nodes[unconsumed[0]]
        return None

    def copy(self) -> 'Hypothesis':
        """Create a deep copy of this hypothesis."""
        import copy
        return Hypothesis(
            nodes=[copy.deepcopy(node) for node in self.nodes],  # Deep copy nodes!
            edges=self.edges.copy(),
            consumed=self.consumed.copy(),
            score=self.score
        )

    def add_edge(self, edge: Edge) -> None:
        """Add an edge if it doesn't already exist."""
        if edge not in self.edges:
            self.edges.append(edge)

    def consume(self, index: int) -> None:
        """Mark a node as consumed."""
        self.consumed.add(index)

    def is_equivalent(self, other: 'Hypothesis') -> bool:
        """
        Check if two hypotheses are structurally equivalent.

        Two hypotheses are equivalent if they have:
        - Same node types at same indices
        - Same set of edges
        - Same consumed set
        """
        if not isinstance(other, Hypothesis):
            return False

        # Check node types match
        if len(self.nodes) != len(other.nodes):
            return False

        for i, (n1, n2) in enumerate(zip(self.nodes, other.nodes)):
            if n1.type != n2.type or n1.index != n2.index:
                return False

        # Check edges match (order-independent)
        if len(self.edges) != len(other.edges):
            return False

        # Convert to sets for comparison (edges have __eq__ defined)
        self_edge_set = set((e.type, e.parent, e.child) for e in self.edges)
        other_edge_set = set((e.type, e.parent, e.child) for e in other.edges)

        if self_edge_set != other_edge_set:
            return False

        # Check consumed set matches
        if self.consumed != other.consumed:
            return False

        return True

    def __repr__(self) -> str:
        unconsumed = len(self.get_unconsumed())
        return f"Hypothesis(edges={len(self.edges)}, unconsumed={unconsumed}, score={self.score:.3f})"


@dataclass
class ParserConfig:
    """Configuration for quantum parser."""

    max_hypotheses: int = 20          # Keep top-K hypotheses
    prune_threshold: float = 0.4      # Drop hypotheses below this fraction of best score
    score_continuously: bool = True   # Prune during parse (vs. only at end)
    max_sentence_length: int = 100    # Reject sentences longer than this
    max_initial_hypotheses: int = 50  # Cap on POS combination hypotheses
    enable_pos_ambiguity: bool = True # Enable POS ambiguity handling


@dataclass
class ParseChart:
    """
    Container for all parse hypotheses.

    Represents the "superposition" of all valid parse trees.

    Attributes:
        words: Original input words
        nodes: Shared nodes (referenced by all hypotheses)
        hypotheses: Alternative parse interpretations
        embeddings: Word embeddings (optional, for semantic scoring)
        config: Parser configuration
    """
    words: List[Word]
    nodes: List[Node]
    hypotheses: List[Hypothesis] = field(default_factory=list)
    embeddings: Optional[Dict[str, Any]] = None
    config: ParserConfig = field(default_factory=ParserConfig)

    def best_hypothesis(self) -> Optional[Hypothesis]:
        """Return highest-scoring hypothesis."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.score)

    def sort_hypotheses(self) -> None:
        """Sort hypotheses by score (descending)."""
        self.hypotheses.sort(key=lambda h: h.score, reverse=True)

    def prune_hypotheses(self) -> None:
        """
        Reduce hypothesis set to manageable size.

        Keeps top-K and drops those below threshold * best_score.
        """
        if not self.hypotheses:
            return

        # Sort by score
        self.sort_hypotheses()

        best_score = self.hypotheses[0].score
        min_score = self.config.prune_threshold * best_score

        # Keep top-K and above threshold
        kept = []
        for hyp in self.hypotheses[:self.config.max_hypotheses]:
            if hyp.score >= min_score:
                kept.append(hyp)
            else:
                break  # Sorted, so rest are below threshold

        self.hypotheses = kept

    def add_hypothesis(self, hyp: Hypothesis) -> None:
        """Add a hypothesis to the chart."""
        self.hypotheses.append(hyp)

    def __repr__(self) -> str:
        return f"ParseChart(words={len(self.words)}, hypotheses={len(self.hypotheses)})"


def create_initial_chart(words: List[Word], config: ParserConfig = None) -> ParseChart:
    """
    Create initial parse chart from words.

    If POS ambiguity is enabled, creates multiple initial hypotheses
    for different POS tag assignments.

    Args:
        words: Input sentence words
        config: Parser configuration (uses default if None)

    Returns:
        ParseChart with one or more initial hypotheses (all nodes unconsumed)
    """
    if config is None:
        config = ParserConfig()

    from .enums import TAG_TO_NODE_TYPE

    # Check if POS ambiguity handling is enabled
    if not config.enable_pos_ambiguity:
        # Original single-hypothesis behavior
        nodes = []
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
            nodes.append(node)

        initial_hyp = Hypothesis(nodes=nodes, edges=[], consumed=set(), score=0.0)

        chart = ParseChart(
            words=words,
            nodes=nodes,
            hypotheses=[initial_hyp],
            config=config
        )

        return chart

    # POS ambiguity handling enabled
    from .pos_tagger import get_possible_tags
    import itertools

    # Get all possible tags for each word
    possible_tags_per_word = []
    for word in words:
        tags = get_possible_tags(word)
        possible_tags_per_word.append(tags)

    # Generate all combinations of POS assignments
    tag_combinations = list(itertools.product(*possible_tags_per_word))

    # Limit number of combinations
    if len(tag_combinations) > config.max_initial_hypotheses:
        # Use simple heuristic: prefer combinations with more common tags
        # For now, just take first N combinations
        # TODO: Could use tag frequency or bigram probability for better selection
        tag_combinations = tag_combinations[:config.max_initial_hypotheses]

    # Create one hypothesis for each POS assignment
    hypotheses = []
    nodes_list = []  # Keep track of all node sets

    for tag_assignment in tag_combinations:
        nodes = []
        for i, (word, tag) in enumerate(zip(words, tag_assignment)):
            node_type = TAG_TO_NODE_TYPE.get(tag, NodeType.NIL)
            node = Node(
                type=node_type,
                original_type=node_type,
                value=word,
                pos=tag,  # Use assigned tag from this combination
                flags=word.subtypes.copy(),
                index=i
            )
            nodes.append(node)

        hyp = Hypothesis(nodes=nodes, edges=[], consumed=set(), score=0.0)
        hypotheses.append(hyp)
        nodes_list.append(nodes)

    # Use first nodes list as reference (all have same structure)
    reference_nodes = nodes_list[0] if nodes_list else []

    chart = ParseChart(
        words=words,
        nodes=reference_nodes,
        hypotheses=hypotheses,
        config=config
    )

    return chart
