# Quantum Parser Architecture

**Complete Technical Design Document**

Version 1.0 - Initial Design
Last Updated: 2025-11-22

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Quantum Parser](#quantum-parser)
3. [DSL Specification](#dsl-specification)
4. [Grammar System](#grammar-system)
5. [Knowledge Graph](#knowledge-graph)
6. [Neural Network Architecture](#neural-network-architecture)
7. [Training Pipeline](#training-pipeline)
8. [Sense Splitting Algorithm](#sense-splitting-algorithm)
9. [Evaluation Methodology](#evaluation-methodology)
10. [Future Extensions](#future-extensions)

---

## System Overview

### Vision

Create Artificial General Intelligence (AGI) through universal semantic representation where:
- All languages map to the same continuous vector space
- Pure meaning is represented geometrically
- Logical operators serve as fixed anchor points
- AI learns to reason directly over meaning

### MVP Goal

**Prove**: Sentences can be mapped to and from continuous vector space while preserving semantic meaning.

**Success Metrics:**
- Parse 10,000 dictionary definitions accurately
- Train embeddings achieving 80%+ synonym detection
- Discover word senses matching WordNet (60%+ F1 score)

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Input: Sentence                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│               Quantum Parser (Hypothesis Exploration)        │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Grammar Rules    │  │  Semantic Scorer │                │
│  │ (Language-Agnostic│ │ (Vector-Based)   │                │
│  │  DSL)            │  │                  │                │
│  └──────────────────┘  └──────────────────┘                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Parse Chart (Multiple Hypotheses)               │
│  Hypothesis 1: [Parse Tree A]  Score: 0.95                  │
│  Hypothesis 2: [Parse Tree B]  Score: 0.87                  │
│  ...                                                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           Semantic Triple Extractor                          │
│  Parse Tree → (subject, relation, object) triples           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Dual Knowledge Graph                        │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ WordNet Graph    │  │  Parsed Graph    │                │
│  │ (Expert Labels)  │  │ (Parser Output)  │                │
│  └──────────────────┘  └──────────────────┘                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│            Embedding Model (Custom Neural Network)           │
│  ┌──────────────────────────────────────────────┐           │
│  │ Fixed Anchors (51 dims):                     │           │
│  │  - 27 Semantic (nominals, scopes, roles)     │           │
│  │  - 15 Grammatical (tense, aspect, mood, ...) │           │
│  │  - 9 Logical (AND, OR, NOT, ...)             │           │
│  └──────────────────────────────────────────────┘           │
│  ┌──────────────────────────────────────────────┐           │
│  │ Learned Embeddings (461 dims):               │           │
│  │  - Trained on dual graph structure           │           │
│  │  - Fuzzy logical constraints                 │           │
│  └──────────────────────────────────────────────┘           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Semantic Vector Space (512-dim)              │
│  - Words as points                                           │
│  - Relationships as vectors                                  │
│  - Continuous meaning representation                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Sense Splitting (Contradiction Detection)       │
│  Detect nodes with conflicting edges → Split into senses    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  Output: Disambiguated Embeddings            │
└─────────────────────────────────────────────────────────────┘
```

---

## Quantum Parser

### Conceptual Model

**Traditional Parsers:** Commit to single interpretation, apply rules sequentially.

**Quantum Parser:** Maintains superposition of all valid parse trees, scores by semantic coherence, "collapses" to most probable interpretation.

**Analogy:**
- **Quantum Mechanics**: Particle exists in superposition until measured
- **Slime Mold**: Explores all paths simultaneously, reinforces successful routes
- **Quantum Parser**: Sentence exists in multiple parse states until semantically scored

### Core Algorithm

```python
def parse(words: List[Word], grammar: Grammar) -> ParseChart:
    """
    Quantum parsing algorithm with parallel hypothesis exploration.

    Returns:
        ParseChart containing all viable parse hypotheses, ranked by score
    """
    # Initialize: One hypothesis per word (all unconsumed)
    chart = ParseChart(words)
    initial_hypothesis = Hypothesis(nodes=to_nodes(words), edges=[], consumed=set())
    chart.hypotheses = [initial_hypothesis]

    # Apply rulesets in order of importance
    for ruleset_name in grammar.order:
        ruleset = grammar.rulesets[ruleset_name]

        # Generate new hypotheses by applying rules
        new_hypotheses = []

        for current_hyp in chart.hypotheses:
            for unconsumed_node_idx in get_unconsumed(current_hyp):
                for rule in ruleset.rules:
                    # Try to match rule at this node
                    matches = find_matches(current_hyp, unconsumed_node_idx, rule)

                    for match in matches:
                        # Create new hypothesis with this interpretation
                        new_hyp = apply_rule(current_hyp, rule, match)

                        # If recursive rule and matched, restart this ruleset
                        if rule.recursive:
                            new_hyp = apply_ruleset_recursively(new_hyp, ruleset)

                        new_hypotheses.append(new_hyp)

        # Score all hypotheses
        for hyp in new_hypotheses:
            hyp.score = score_hypothesis(hyp, chart.embeddings)

        # Prune: Keep top-K hypotheses
        chart.hypotheses = prune_hypotheses(
            new_hypotheses,
            max_k=chart.config.max_hypotheses,  # Default: 20
            threshold=chart.config.prune_threshold  # Default: 0.4
        )

        # Early termination if only one high-confidence hypothesis remains
        if len(chart.hypotheses) == 1 and chart.hypotheses[0].score > 0.95:
            break

    # Sort by score (best first)
    chart.hypotheses.sort(key=lambda h: h.score, reverse=True)

    return chart
```

### Data Structures

#### ParseChart

```python
@dataclass
class ParseChart:
    """Container for all parse hypotheses."""

    words: List[Word]              # Original input words
    nodes: List[Node]              # Shared nodes (referenced by all hypotheses)
    hypotheses: List[Hypothesis]   # Alternative parse interpretations
    embeddings: Optional[Dict]     # Word embeddings (if loaded)
    config: ParserConfig           # Configuration (max_hypotheses, etc.)

    def best_hypothesis(self) -> Hypothesis:
        """Return highest-scoring hypothesis."""
        return self.hypotheses[0] if self.hypotheses else None

    def visualize(self, output_path: str, show_all: bool = False):
        """
        Generate graphviz visualization.

        Args:
            output_path: Where to save PNG
            show_all: If True, show all hypotheses; else just best
        """
        if show_all:
            for i, hyp in enumerate(self.hypotheses[:5]):  # Top 5
                hyp.to_dot(f"{output_path}_hyp{i}.png")
        else:
            self.best_hypothesis().to_dot(output_path)
```

#### Hypothesis

```python
@dataclass
class Hypothesis:
    """A single parse interpretation."""

    nodes: List[Node]          # References to ParseChart.nodes
    edges: List[Edge]          # Connections in this interpretation
    consumed: Set[int]         # Node indices marked as consumed
    score: float = 0.0         # Semantic coherence score

    def get_unconsumed(self) -> List[int]:
        """Return indices of unconsumed nodes."""
        return [i for i in range(len(self.nodes)) if i not in self.consumed]

    def get_root(self) -> Optional[Node]:
        """Return root node (final unconsumed node)."""
        unconsumed = self.get_unconsumed()
        return self.nodes[unconsumed[0]] if len(unconsumed) == 1 else None

    def to_dot(self, output_path: str):
        """Generate graphviz DOT visualization."""
        # Implementation: Create directed graph with labeled edges
        ...
```

#### Node

```python
@dataclass
class Node:
    """
    A node in the parse tree.

    Represents either:
    - A single word (leaf node)
    - A constituent built from multiple words (internal node)
    """

    type: NodeType                  # Current type (NOUN, VERBAL, NOMINAL, etc.)
    original_type: NodeType         # Type before transformations
    value: Optional[Word]           # Original word (None for constructed nodes)
    pos: Tag                        # Part of speech
    flags: List[SubType]            # Accumulated properties (PLURAL, PAST, etc.)

    # Connections are stored at Hypothesis level, not Node level
    # This allows different hypotheses to have different edge sets
```

#### Edge

```python
@dataclass
class Edge:
    """A directed connection between nodes."""

    type: ConnectionType        # SUBJECT, OBJECT, DESCRIPTION, etc.
    parent: int                 # Node index in hypothesis
    child: int                  # Node index in hypothesis
    source_rule: str            # Name of rule that created this edge

    def __eq__(self, other):
        """Two edges are equal if they connect the same nodes with same type."""
        return (self.type == other.type and
                self.parent == other.parent and
                self.child == other.child)
```

### Rule Matching

```python
def find_matches(hypothesis: Hypothesis, anchor_idx: int, rule: Rule) -> List[Match]:
    """
    Find all ways this rule can match at anchor node.

    Returns:
        List of Match objects, each representing a valid application of the rule
    """
    matches = []

    anchor = hypothesis.nodes[anchor_idx]

    # Check if anchor matches rule's anchor pattern
    if not matches_pattern(anchor, rule.anchor):
        return []

    # Try to find "before" elements
    before_matches = []
    for before_pattern in rule.before:
        candidates = search_direction(
            hypothesis,
            anchor_idx,
            direction="left",
            pattern=before_pattern
        )
        if not candidates:
            return []  # Required element not found
        before_matches.append(candidates)

    # Try to find "after" elements
    after_matches = []
    for after_pattern in rule.after:
        candidates = search_direction(
            hypothesis,
            anchor_idx,
            direction="right",
            pattern=after_pattern
        )
        if not candidates:
            return []  # Required element not found
        after_matches.append(candidates)

    # Generate all combinations of matched elements
    for before_combo in itertools.product(*before_matches):
        for after_combo in itertools.product(*after_matches):
            matches.append(Match(
                anchor=anchor_idx,
                before=before_combo,
                after=after_combo,
                rule=rule
            ))

    return matches

def matches_pattern(node: Node, pattern: PatternElement) -> bool:
    """Check if node matches pattern requirements."""

    # Type check
    if pattern.original_type:
        if node.original_type != pattern.type:
            return False
    else:
        if node.type != pattern.type:
            return False

    # SubType check
    if pattern.subtypes:
        if not all(st in node.flags for st in pattern.subtypes):
            return False

    # SubCategory check (e.g., GENDER, NUMBER must match)
    if pattern.subcategories:
        for subcat in pattern.subcategories:
            if not has_matching_subcategory(node, subcat):
                return False

    return True

def search_direction(hypothesis: Hypothesis, anchor_idx: int,
                     direction: str, pattern: PatternElement) -> List[int]:
    """
    Search left or right from anchor for nodes matching pattern.

    Args:
        direction: "left" or "right"
        pattern: PatternElement to match

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
        if matches_pattern(node, pattern):
            matches.append(current_idx)

            # Respect quantifier
            if pattern.quantifier == "one":
                return matches  # Found one, stop
            elif pattern.quantifier == "all":
                # Continue searching
                pass
            elif pattern.quantifier == "one_or_more":
                # Found at least one, could continue or stop
                # For now, take first match
                return matches
        else:
            # Non-matching node blocks further search
            if pattern.quantifier == "all":
                # Already collected all matches up to here
                return matches
            else:
                # Required match not found
                return []

        current_idx += step

    return matches if pattern.quantifier in ["all", "one_or_more"] else []
```

### Hypothesis Scoring

```python
def score_hypothesis(hypothesis: Hypothesis, embeddings: Optional[Dict] = None) -> float:
    """
    Score hypothesis by structural coherence and semantic plausibility.

    Returns:
        Float in [0, 1], higher is better
    """
    # Structural score (always available)
    struct_score = compute_structural_score(hypothesis)

    # Semantic score (requires embeddings)
    if embeddings:
        sem_score = compute_semantic_score(hypothesis, embeddings)
        return 0.5 * struct_score + 0.5 * sem_score
    else:
        return struct_score

def compute_structural_score(hypothesis: Hypothesis) -> float:
    """
    Score based on parse tree structure.

    Criteria:
    - Prefer balanced trees (not too deep, not too flat)
    - Penalize unconsumed nodes (incomplete parse)
    - Reward full connectivity
    - Penalize crossing edges (non-projective)
    """
    num_nodes = len(hypothesis.nodes)
    num_consumed = len(hypothesis.consumed)
    num_edges = len(hypothesis.edges)

    # Coverage: What fraction of nodes are consumed?
    coverage = num_consumed / num_nodes if num_nodes > 0 else 0

    # Connectivity: Should have roughly num_nodes - 1 edges (tree property)
    expected_edges = num_nodes - 1
    connectivity = 1.0 - abs(num_edges - expected_edges) / num_nodes

    # Projectivity: Penalize crossing edges
    crossing_penalty = count_crossing_edges(hypothesis) * 0.1
    projectivity = max(0, 1.0 - crossing_penalty)

    # Balance: Prefer trees with moderate depth
    depth = compute_tree_depth(hypothesis)
    ideal_depth = math.log2(num_nodes) if num_nodes > 0 else 1
    balance = 1.0 - abs(depth - ideal_depth) / num_nodes

    # Weighted combination
    score = (
        0.4 * coverage +
        0.3 * connectivity +
        0.2 * projectivity +
        0.1 * balance
    )

    return max(0.0, min(1.0, score))

def compute_semantic_score(hypothesis: Hypothesis, embeddings: Dict) -> float:
    """
    Score based on semantic coherence in vector space.

    Criteria:
    - Connected words should be semantically related
    - Subject-verb-object should form coherent event
    - Modifiers should be relevant to modified words
    """
    scores = []

    for edge in hypothesis.edges:
        parent = hypothesis.nodes[edge.parent]
        child = hypothesis.nodes[edge.child]

        # Get word embeddings
        parent_vec = embeddings.get(parent.value.text.lower())
        child_vec = embeddings.get(child.value.text.lower())

        if parent_vec is None or child_vec is None:
            continue  # Unknown word, skip

        # Compute similarity
        similarity = cosine_similarity(parent_vec, child_vec)

        # Weight by connection type
        # SUBJECT/OBJECT = strong semantic constraint
        # DESCRIPTION/MODIFICATION = weaker constraint
        if edge.type in [ConnectionType.SUBJECT, ConnectionType.OBJECT]:
            weight = 1.0
        elif edge.type in [ConnectionType.DESCRIPTION, ConnectionType.MODIFICATION]:
            weight = 0.5
        else:
            weight = 0.3

        scores.append(weight * similarity)

    return np.mean(scores) if scores else 0.5

def count_crossing_edges(hypothesis: Hypothesis) -> int:
    """
    Count number of crossing edge pairs (non-projective structures).

    Two edges (i1, j1) and (i2, j2) cross if:
    i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1
    """
    crossings = 0
    edges = [(e.parent, e.child) for e in hypothesis.edges]

    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            e1_min, e1_max = min(edges[i]), max(edges[i])
            e2_min, e2_max = min(edges[j]), max(edges[j])

            # Check for crossing
            if (e1_min < e2_min < e1_max < e2_max) or \
               (e2_min < e1_min < e2_max < e1_max):
                crossings += 1

    return crossings
```

### Pruning Strategy

```python
def prune_hypotheses(hypotheses: List[Hypothesis],
                     max_k: int,
                     threshold: float) -> List[Hypothesis]:
    """
    Reduce hypothesis set to manageable size.

    Strategy:
    1. Sort by score
    2. Keep top-K
    3. Drop hypotheses scoring below threshold * best_score

    Args:
        max_k: Maximum hypotheses to keep (default: 20)
        threshold: Drop hypotheses below this fraction of best score (default: 0.4)
    """
    # Sort by score (descending)
    sorted_hyps = sorted(hypotheses, key=lambda h: h.score, reverse=True)

    if not sorted_hyps:
        return []

    best_score = sorted_hyps[0].score
    min_score = threshold * best_score

    # Keep top-K and above threshold
    kept = []
    for hyp in sorted_hyps[:max_k]:
        if hyp.score >= min_score:
            kept.append(hyp)
        else:
            break  # Sorted, so all remaining are below threshold

    return kept
```

---

## DSL Specification

### Design Philosophy

**Goals:**
- **Explicit**: Every field has clear meaning
- **Validated**: Errors caught at load time, not runtime
- **Extensible**: Easy to add new features
- **Language-Agnostic**: Same DSL for all languages

**Trade-off**: Verbose JSON vs. terse syntax

**Decision**: Verbose JSON (better for parser that reads it naturally)

### Grammar File Structure

```json
{
    "metadata": {
        "language": "english",
        "version": "1.0",
        "author": "William",
        "description": "English syntactic grammar for quantum parser"
    },

    "order": [
        "clause1",
        "clause2",
        "verbal1",
        "verbal2",
        ...
    ],

    "rulesets": {
        "ruleset_name": {
            "description": "Human-readable explanation of what this ruleset does",
            "result": "DESCRIPTOR",
            "rules": [
                { ... rule 1 ... },
                { ... rule 2 ... }
            ]
        },
        ...
    }
}
```

### Rule Schema

```json
{
    "result": "DESCRIPTOR",           // Type to transform anchor node into
    "recursive": false,               // Restart ruleset after match?

    "pattern": {
        "anchor": {                   // Root of the pattern (marked with ?)
            "type": "DESCRIPTOR",
            "subtypes": [],           // Optional: must have these subtypes
            "subcategories": [],      // Optional: must match these categories
            "original_type": null     // Optional: check OG field instead
        },

        "before": [                   // Elements to find left of anchor
            {
                "type": "SPECIFIER",
                "quantifier": "one",  // "one", "all", "one_or_more"
                "subtypes": [],
                "subcategories": []
            }
        ],

        "after": [                    // Elements to find right of anchor
            {
                "type": "CLAUSE",
                "quantifier": "one",
                "subtypes": ["SUBOORDINATE"],
                "subcategories": []
            }
        ]
    },

    "connections": [                  // Edges to create
        {
            "type": "SPECIFICATION",  // ConnectionType enum value
            "from": "before[0]",      // Source node reference
            "to": "anchor"            // Target node reference
        },
        {
            "type": "SUBOORDINATION_TO",
            "from": "anchor",
            "to": "after[0]"
        }
    ],

    "consume": ["before", "after"],   // Which matched elements to mark consumed

    "flags": {
        "pull_categories": [],        // SubCategories to propagate to anchor
        "pop_categories": []          // SubCategories to remove from anchor
    }
}
```

### Field Reference

#### `result`
- **Type**: String (NodeType enum value)
- **Purpose**: What type to transform the anchor node into
- **Example**: `"DESCRIPTOR"`, `"NOMINAL"`, `"PREDICATE"`

#### `recursive`
- **Type**: Boolean
- **Default**: `false`
- **Purpose**: If true, restart this ruleset after successful match
- **Use case**: Collecting all adjectives after a noun

#### `pattern.anchor`
- **Required**: Always
- **Purpose**: The root node where matching starts (the `?` in old DSL)

#### `pattern.before` / `pattern.after`
- **Type**: List of PatternElement
- **Purpose**: Elements to find left/right of anchor
- **Empty list**: No elements required in that direction

#### `quantifier`
- **Values**:
  - `"one"`: Match exactly one element
  - `"all"`: Match all consecutive matching elements
  - `"one_or_more"`: Match at least one (greedy)
- **Default**: `"one"`

#### `subtypes`
- **Type**: List of String (SubType enum values)
- **Purpose**: Node must have ALL these subtypes
- **Example**: `["SUBOORDINATE"]`, `["PLURAL", "FEMININE"]`

#### `subcategories`
- **Type**: List of String (SubCategory enum values)
- **Purpose**: Node must match anchor node in these categories
- **Example**: `["GENDER", "NUMBER"]` (for Spanish adjective agreement)
- **How it works**: Both nodes must have same value for each subcategory

#### `original_type`
- **Type**: String (NodeType enum value) or null
- **Purpose**: If specified, check node's OG field instead of current type
- **Use case**: Match nodes that were originally X but transformed to Y

#### `connections`
- **Type**: List of ConnectionSpec
- **Purpose**: Edges to create between matched nodes

#### `from` / `to`
- **Syntax**:
  - `"anchor"`: The anchor node
  - `"before[0]"`: First element from before list
  - `"before[*]"`: All elements from before list (creates multiple edges)
  - `"after[2]"`: Third element from after list
- **Purpose**: Reference matched nodes by position

#### `consume`
- **Type**: List of String
- **Values**: `["before", "after", "anchor"]`
- **Purpose**: Which matched elements to mark as consumed
- **Note**: Typically DON'T consume anchor (it becomes parent of new subtree)

#### `flags.pull_categories`
- **Type**: List of String (SubCategory values)
- **Purpose**: Copy these categories from children to anchor
- **Example**: Pull GENDER and NUMBER from noun to nominal phrase

#### `flags.pop_categories`
- **Type**: List of String (SubCategory values)
- **Purpose**: Remove these categories from anchor
- **Use case**: Clean up temporary flags after processing

### Example: Complete Rule

```json
{
    "result": "NOMINAL",
    "recursive": true,
    "pattern": {
        "anchor": {
            "type": "NOUN"
        },
        "before": [],
        "after": [
            {
                "type": "DESCRIPTOR",
                "quantifier": "all",
                "subcategories": ["GENDER", "NUMBER"]
            }
        ]
    },
    "connections": [
        {
            "type": "DESCRIPTION",
            "from": "after[*]",
            "to": "anchor"
        }
    ],
    "consume": ["after"],
    "flags": {
        "pull_categories": ["GENDER", "NUMBER"],
        "pop_categories": []
    },
    "note": "Spanish: Collect all post-nominal adjectives matching gender/number"
}
```

**Interpretation:**
1. Find a NOUN node (anchor)
2. Look right for ALL DESCRIPTORs that match anchor's GENDER and NUMBER
3. Create DESCRIPTION edges from each descriptor to the noun
4. Transform noun to NOMINAL
5. Mark descriptors as consumed
6. Pull GENDER and NUMBER up to the NOMINAL
7. If matched, restart this ruleset (recursive) to catch more adjectives

---

## Grammar System

### Rule Application Order

**Philosophy**: Strip decorations until you find the core Subject-Verb-Object structure.

**Order of Importance (Universal across languages):**

1. **Clauses** (highest level): Subordinate, relative, independent
2. **Verbal Predicates**: Main verb + complements, auxiliary constructions
3. **Subjects**: Who/what performs the action
4. **Objects**: Direct, indirect, prepositional
5. **Prepositional Phrases**: Modify nouns or verbs
6. **Adjectives**: Describe nouns
7. **Adverbs**: Modify verbs/adjectives
8. **Determiners**: Articles, demonstratives, quantifiers
9. **Coordination**: And, or, but
10. **Specifiers**: Very, more, most

**Rationale**: Process larger constituents first, build from clauses → predicates → arguments → modifiers → specifiers.

### Subcategory System

**Purpose**: Handle agreement phenomena (gender, number, case, etc.)

**How it works:**
1. Words tagged with subcategory values during POS tagging
2. Rules specify which subcategories must match
3. Parser checks: Do both nodes have the same value for this subcategory?

**Example: Spanish Gender/Number Agreement**

```json
// Tag noun with gender and number
Word("casa", POS=NOUN, subtypes=[FEMININE, SINGULAR])

// Tag adjective with matching features
Word("roja", POS=ADJECTIVE, subtypes=[FEMININE, SINGULAR])

// Rule requires matching
{
    "pattern": {
        "anchor": {"type": "NOUN"},
        "after": [
            {
                "type": "DESCRIPTOR",
                "subcategories": ["GENDER", "NUMBER"]
            }
        ]
    }
}

// Parser checks:
// anchor.GENDER == after[0].GENDER? (FEMININE == FEMININE ✓)
// anchor.NUMBER == after[0].NUMBER? (SINGULAR == SINGULAR ✓)
// Match succeeds!
```

**Subcategories Defined:**
- **SC_GENDER**: Masculine, Feminine, Neuter, Common
- **SC_NUMBER**: Singular, Plural, Dual
- **SC_CASE**: Nominative, Accusative, Genitive, Dative, etc.
- **SC_PERSON**: First, Second, Third
- **SC_TENSE**: Past, Present, Future, etc.
- **SC_VERB**: Modal, Nominal, etc.
- **SC_DESCRIPTOR**: Comparative, Superlative
- **SC_CLAUSE**: Subordinate, Independent
- **SC_LOGIC**: AND, OR, NOT, etc.

### Language-Specific Considerations

#### English
- **Word Order**: Relatively rigid (SVO)
- **Agreement**: Minimal (subject-verb number, pronoun case)
- **Strategy**: Position-based rules with light agreement checking

#### Spanish
- **Word Order**: Flexible (SVO default, but VSO, OVS possible)
- **Agreement**: Extensive (gender, number across nouns, adjectives, articles)
- **Pro-drop**: Subjects often omitted (verb conjugation carries person/number)
- **Clitics**: Pronouns attach to verbs (dámelo = give-me-it)
- **Strategy**: Heavy subcategory matching, semantic role assignment over position

#### Future Languages
- **German**: Case system (nominative, accusative, dative, genitive)
- **Russian**: Free word order, rich case morphology
- **Japanese**: SOV order, topic-comment structure, honorifics
- **Arabic**: VSO order, root-pattern morphology

**Design Principle**: Grammar DSL should support all these through subcategory system + flexible pattern matching.

---

## Knowledge Graph

### Dual-Graph Architecture

**Why Dual?**

We have TWO sources of semantic knowledge:
1. **WordNet**: Expert-labeled sense distinctions and relationships
2. **Parser Output**: Relationships extracted from dictionary definitions

**Approach**: Train on BOTH simultaneously to learn the pattern of how syntax encodes WordNet-style semantics.

### Graph Schema

```python
class KnowledgeGraph:
    """
    Dual-source knowledge graph.

    Nodes: Word senses (may be merged initially, split later)
    Edges: Typed semantic relationships, marked by source
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    @dataclass
    class Node:
        id: str                    # Unique identifier (word_POS or synset_id)
        words: List[str]           # Lemmas for this sense
        pos: str                   # Part of speech (n, v, a, r, s)
        definition: str            # Definition text
        examples: List[str]        # Example sentences
        source: str                # "wordnet" or "wiktionary"
        parent: Optional[str]      # If split from another node

    @dataclass
    class Edge:
        source: str                # Node ID
        target: str                # Node ID
        relation: str              # "synonym", "antonym", "hypernym", etc.
        edge_source: str           # "wordnet" or "parsed"
        confidence: float = 1.0    # Edge weight
```

### Relationship Types

#### From WordNet:
- **hypernym** (is-a): dog → canine → mammal → animal
- **hyponym** (inverse): animal → mammal → canine → dog
- **meronym** (part-of): dog → tail, dog → paw
- **holonym** (has-part): tail → dog
- **synonym**: dog ≈ domestic dog ≈ Canis familiaris
- **antonym**: hot ↔ cold
- **similar-to**: sad ~ unhappy
- **entailment**: snore → sleep (can't snore without sleeping)
- **cause**: kill → die

#### From Parser:
- **is-a**: Extracted from copula constructions ("X is a Y")
- **has-property**: Extracted from adjective modification
- **used-for**: Extracted from purpose clauses ("tool for X")
- **located-in**: Extracted from location prepositional phrases
- **causes**: Extracted from causal verbs/conjunctions
- **part-of**: Extracted from possessive/compositional constructions

### Graph Construction Pipeline

```python
def build_dual_graph(wordnet_path: str, definitions: List[Dict]) -> KnowledgeGraph:
    """
    Build knowledge graph from WordNet + parsed definitions.

    Args:
        wordnet_path: Path to WordNet data
        definitions: List of {word, pos, text, triples}

    Returns:
        KnowledgeGraph with dual-source edges
    """
    graph = KnowledgeGraph()

    # Step 1: Import WordNet
    import_wordnet(graph, wordnet_path)

    # Step 2: Add nodes for words in definitions (may overlap with WordNet)
    for defn in definitions:
        node_id = f"{defn['word']}_{defn['pos']}"

        if node_id not in graph.nodes:
            graph.add_node(
                id=node_id,
                words=[defn['word']],
                pos=defn['pos'],
                definition=defn['text'],
                examples=[],
                source="wiktionary"
            )

    # Step 3: Add edges from parsed triples
    for defn in definitions:
        for triple in defn['triples']:
            graph.add_edge(
                source=triple.subject,
                target=triple.object,
                relation=triple.relation,
                edge_source="parsed"
            )

    # Step 4: Link WordNet and Wiktionary nodes (same word → same node)
    merge_equivalent_nodes(graph)

    return graph

def import_wordnet(graph: KnowledgeGraph, path: str):
    """Import WordNet synsets and relationships."""
    from nltk.corpus import wordnet as wn

    for synset in wn.all_synsets():
        # Add node for this synset
        graph.add_node(
            id=synset.name(),
            words=synset.lemma_names(),
            pos=synset.pos(),
            definition=synset.definition(),
            examples=synset.examples(),
            source="wordnet"
        )

        # Add edges for relationships
        for hypernym in synset.hypernyms():
            graph.add_edge(synset.name(), hypernym.name(),
                          "hypernym", "wordnet")

        for hyponym in synset.hyponyms():
            graph.add_edge(synset.name(), hyponym.name(),
                          "hyponym", "wordnet")

        # ... other relationships
```

### Semantic Triple Extraction

**Goal**: Convert parse tree → (subject, relation, object) triples

**Strategy**: Map Connection types to semantic relationships based on context

```python
def extract_triples(hypothesis: Hypothesis) -> List[Triple]:
    """
    Extract semantic triples from parse tree.

    Returns:
        List of (subject, relation, object) triples
    """
    triples = []

    for edge in hypothesis.edges:
        parent = hypothesis.nodes[edge.parent]
        child = hypothesis.nodes[edge.child]

        # Rule 1: SUBJECT + copula "is/are" → is-a
        if edge.type == ConnectionType.SUBJECT:
            verb = find_verb(hypothesis, edge.parent)
            if verb and verb.value.text.lower() in ["is", "are", "was", "were"]:
                # "Dog is a mammal" → (dog, is-a, mammal)
                object_node = find_object(hypothesis, edge.parent)
                if object_node:
                    triples.append(Triple(
                        subject=get_head_word(child),
                        relation="is-a",
                        object=get_head_word(object_node)
                    ))

        # Rule 2: DESCRIPTION → has-property
        elif edge.type == ConnectionType.DESCRIPTION:
            # "Red car" → (car, has-property, red)
            triples.append(Triple(
                subject=get_head_word(parent),
                relation="has-property",
                object=get_head_word(child)
            ))

        # Rule 3: Prepositional phrases → context-dependent
        elif edge.type == ConnectionType.PREPOSITION_TO:
            prep = get_preposition_word(hypothesis, edge)

            if prep in ["in", "at", "on"]:
                # "Dog in house" → (dog, located-in, house)
                triples.append(Triple(
                    subject=get_head_word(parent),
                    relation="located-in",
                    object=get_head_word(child)
                ))

            elif prep in ["for", "to"]:
                # "Tool for cutting" → (tool, used-for, cutting)
                triples.append(Triple(
                    subject=get_head_word(parent),
                    relation="used-for",
                    object=get_head_word(child)
                ))

            elif prep in ["from", "of"]:
                # "Part of car" → (part, part-of, car)
                # "Wheel of car" → (wheel, part-of, car)
                triples.append(Triple(
                    subject=get_head_word(parent),
                    relation="part-of",
                    object=get_head_word(child)
                ))

        # Rule 4: Causal verbs → causes
        elif edge.type == ConnectionType.OBJECT:
            verb = get_head_word(parent)
            if verb in ["cause", "make", "create", "produce", "lead to"]:
                # "X causes Y" → (X, causes, Y)
                subject = find_subject(hypothesis, edge.parent)
                if subject:
                    triples.append(Triple(
                        subject=get_head_word(subject),
                        relation="causes",
                        object=get_head_word(child)
                    ))

        # ... more extraction rules

    return triples

def get_head_word(node: Node) -> str:
    """Get the lexical head of a constituent."""
    if node.value:
        return node.value.text.lower()
    else:
        # Constructed node, find head child
        # (Implementation: traverse children to find lexical head)
        ...
```

---

## Neural Network Architecture

### Design Philosophy

**Requirements:**
1. **Dynamic structure**: Support adding/removing nodes (sense splitting/merging)
2. **Fixed anchors**: 51 predefined dimensions must stay constant
3. **Dual-source learning**: Train on WordNet + parsed graph simultaneously
4. **Interpretable**: Understand why embeddings are where they are
5. **CPU-friendly**: No GPU required, overnight training OK

**Decision**: Custom implementation (no PyTorch/TensorFlow)

**Why**: Standard frameworks can't handle dynamic graph modification during training.

### Model Architecture

```python
class GraphEmbeddingModel:
    """
    Custom graph embedding with fixed anchor dimensions.

    Architecture:
    - Input: Dual knowledge graph (WordNet + parsed)
    - Output: 512-dim embeddings per node
    - Constraint: First 51 dims fixed as semantic/grammatical/logical anchors
    """

    def __init__(self, num_nodes: int, embedding_dim: int = 512, num_anchors: int = 51):
        self.embedding_dim = embedding_dim
        self.num_anchors = num_anchors
        self.num_learned = embedding_dim - num_anchors

        # Node embeddings: [num_nodes, embedding_dim]
        # First num_anchors dims are fixed
        # Remaining dims are learned
        self.embeddings = np.random.randn(num_nodes, embedding_dim) * 0.01

        # Freeze anchor dimensions
        self.anchor_mask = np.zeros(embedding_dim, dtype=bool)
        self.anchor_mask[:num_anchors] = True

        # Relation offsets: learnable translation vectors
        # One per relation type (hypernym, meronym, etc.)
        self.relation_offsets = {
            "hypernym": np.random.randn(embedding_dim) * 0.01,
            "meronym": np.random.randn(embedding_dim) * 0.01,
            "synonym": np.random.randn(embedding_dim) * 0.01,
            # ... etc
        }

        # Node ID → embedding index mapping
        self.node_to_idx = {}
        self.idx_to_node = {}

    def forward(self, triple: Triple) -> float:
        """
        Compute loss for a single triple.

        Uses TransE: h + r ≈ t
        Where h = head embedding, r = relation offset, t = tail embedding
        """
        head_idx = self.node_to_idx[triple.head]
        tail_idx = self.node_to_idx[triple.tail]

        h = self.embeddings[head_idx]
        t = self.embeddings[tail_idx]
        r = self.relation_offsets[triple.relation]

        # Predict tail from head + relation
        predicted_t = h + r

        # L2 loss
        loss = np.linalg.norm(predicted_t - t) ** 2

        return loss

    def compute_gradient(self, triple: Triple) -> Dict:
        """
        Compute gradients for a triple.

        ∂L/∂h = 2(h + r - t)
        ∂L/∂t = -2(h + r - t)
        ∂L/∂r = 2(h + r - t)
        """
        head_idx = self.node_to_idx[triple.head]
        tail_idx = self.node_to_idx[triple.tail]

        h = self.embeddings[head_idx]
        t = self.embeddings[tail_idx]
        r = self.relation_offsets[triple.relation]

        diff = h + r - t

        grad_h = 2 * diff
        grad_t = -2 * diff
        grad_r = 2 * diff

        return {
            "head": (head_idx, grad_h),
            "tail": (tail_idx, grad_t),
            "relation": (triple.relation, grad_r)
        }

    def update(self, gradients: List[Dict], learning_rate: float):
        """
        Apply gradients, respecting anchor mask.
        """
        # Accumulate gradients
        embedding_grads = defaultdict(list)
        relation_grads = defaultdict(list)

        for grad in gradients:
            head_idx, grad_h = grad["head"]
            tail_idx, grad_t = grad["tail"]
            rel, grad_r = grad["relation"]

            embedding_grads[head_idx].append(grad_h)
            embedding_grads[tail_idx].append(grad_t)
            relation_grads[rel].append(grad_r)

        # Update embeddings (only learned dimensions)
        for idx, grads in embedding_grads.items():
            avg_grad = np.mean(grads, axis=0)
            # Zero out gradients for anchor dims
            avg_grad[self.anchor_mask] = 0
            # Update
            self.embeddings[idx] -= learning_rate * avg_grad

        # Update relation offsets
        for rel, grads in relation_grads.items():
            avg_grad = np.mean(grads, axis=0)
            self.relation_offsets[rel] -= learning_rate * avg_grad
```

### Anchor System

**Fixed Dimensions (51 total):**

```python
def initialize_anchors():
    """
    Create fixed basis vectors for semantic/grammatical/logical dimensions.

    Returns:
        Dict mapping dimension name → index
    """
    anchors = {}
    idx = 0

    # Semantic dimensions (27)
    # From LangAI repo: nominals, scopes, roles
    semantic = [
        # Nominals (6)
        "determinatory", "personal", "living",
        "permanence", "embodiment", "magnitude",

        # Scopes (11)
        "temporal", "frequency", "location", "manner",
        "extent", "reason", "attitude", "relative",
        "direction", "spacialExtent", "beneficiary",

        # Roles (10)
        "fundemental", "subject", "subjectComp", "objects",
        "results", "instruments", "sources", "goals",
        "experiencer", "nominal"
    ]

    for name in semantic:
        anchors[name] = idx
        idx += 1

    # Grammatical dimensions (15)
    grammatical = [
        "tense", "aspect", "mood", "voice", "person",
        "number", "gender", "case", "definiteness", "polarity",
        "animacy", "countability", "degree", "transitivity", "evidentiality"
    ]

    for name in grammatical:
        anchors[name] = idx
        idx += 1

    # Logical operators (9)
    logical = [
        "AND", "OR", "XOR", "NAND", "IF",
        "XIF", "NOT", "NOR", "XNOR"
    ]

    for name in logical:
        anchors[name] = idx
        idx += 1

    assert idx == 51, f"Expected 51 anchors, got {idx}"

    return anchors
```

**How Anchors Work:**

1. **Initialization**: Embeddings start with random values in all 512 dimensions
2. **Training**: Gradients are zeroed for first 51 dimensions (anchor mask)
3. **Result**: Anchor dimensions stay at initial values, act as fixed coordinate system
4. **Interpretation**: Words' positions in learned dims (52-512) are relative to this fixed basis

**Example**:
- Dimension 0 = "determinatory" (nominal property)
- Words like "the", "a", "this" have high values in dim 0
- Words like "dog", "car" have low values in dim 0
- This dimension's meaning is FIXED across all training

### Fuzzy Logical Constraints

**Concept**: Semantic relationships aren't binary; they're continuous.

**Implementation**: Each relationship type has a constraint function

```python
def compute_constraint_loss(embedding_model, edge):
    """
    Compute loss for semantic constraint.

    Each relationship type has ideal geometric configuration in vector space.
    """
    v1 = embedding_model.get_embedding(edge.source)
    v2 = embedding_model.get_embedding(edge.target)

    if edge.relation == "synonym":
        # Synonyms should point in same direction (cosine ≈ 1)
        similarity = cosine_similarity(v1, v2)
        loss = (1.0 - similarity) ** 2

    elif edge.relation == "antonym":
        # Antonyms should point in opposite directions (cosine ≈ -1)
        similarity = cosine_similarity(v1, v2)
        loss = (1.0 + similarity) ** 2

    elif edge.relation == "hypernym":
        # Hypernyms: v1 + offset ≈ v2
        # (TransE-style: "dog" + IS-A → "animal")
        offset = embedding_model.relation_offsets["hypernym"]
        predicted = v1 + offset
        loss = np.linalg.norm(predicted - v2) ** 2

    elif edge.relation == "hyponym":
        # Hyponyms: reverse of hypernym
        offset = embedding_model.relation_offsets["hypernym"]
        predicted = v2 + offset
        loss = np.linalg.norm(predicted - v1) ** 2

    elif edge.relation == "meronym":
        # Meronyms (part-of): subspace relationship
        # v1 should lie in subspace spanned by v2
        projection = project_onto(v1, v2)
        loss = np.linalg.norm(v1 - projection) ** 2

    elif edge.relation == "similar-to":
        # Similar but not identical (cosine ≈ 0.7-0.9)
        similarity = cosine_similarity(v1, v2)
        ideal = 0.8
        loss = (similarity - ideal) ** 2

    else:
        # Generic relationship: just use TransE
        offset = embedding_model.relation_offsets.get(
            edge.relation,
            np.zeros(embedding_model.embedding_dim)
        )
        predicted = v1 + offset
        loss = np.linalg.norm(predicted - v2) ** 2

    return loss

def cosine_similarity(v1, v2):
    """Cosine similarity in [-1, 1]."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2 + 1e-8)

def project_onto(v, basis):
    """Project v onto direction of basis."""
    dot = np.dot(v, basis)
    norm_sq = np.dot(basis, basis)
    return (dot / norm_sq) * basis
```

---

## Training Pipeline

### Data Preparation

```python
def prepare_training_data(wordnet_graph, parsed_graph, selected_words):
    """
    Prepare dual-source training data.

    Args:
        wordnet_graph: Graph from WordNet import
        parsed_graph: Graph from definition parsing
        selected_words: Top 10K words to focus on

    Returns:
        dual_graph, wordnet_triples, parsed_triples
    """
    # Merge graphs (same word+POS → same node)
    dual_graph = merge_graphs(wordnet_graph, parsed_graph)

    # Filter to selected words + their 1-hop neighbors
    subgraph = extract_subgraph(dual_graph, selected_words, hops=1)

    # Extract triples by source
    wordnet_triples = [
        (e.source, e.relation, e.target)
        for e in subgraph.edges
        if e.edge_source == "wordnet"
    ]

    parsed_triples = [
        (e.source, e.relation, e.target)
        for e in subgraph.edges
        if e.edge_source == "parsed"
    ]

    print(f"WordNet triples: {len(wordnet_triples)}")
    print(f"Parsed triples: {len(parsed_triples)}")

    return subgraph, wordnet_triples, parsed_triples
```

### Training Loop

```python
def train_embeddings(dual_graph, wordnet_triples, parsed_triples, config):
    """
    Train embeddings with dual-source loss.

    Args:
        config: TrainingConfig (epochs, batch_size, learning_rate, etc.)

    Returns:
        Trained GraphEmbeddingModel
    """
    # Initialize model
    model = GraphEmbeddingModel(
        num_nodes=len(dual_graph.nodes),
        embedding_dim=config.embedding_dim,
        num_anchors=config.num_anchors
    )

    # Map node IDs to indices
    model.node_to_idx = {node_id: i for i, node_id in enumerate(dual_graph.nodes.keys())}
    model.idx_to_node = {i: node_id for node_id, i in model.node_to_idx.items()}

    # Initialize anchors
    initialize_anchor_dimensions(model)

    # Training loop
    best_loss = float('inf')
    patience = 0

    for epoch in range(config.num_epochs):
        # Sample batches
        wordnet_batch = sample_batch(wordnet_triples, config.batch_size)
        parsed_batch = sample_batch(parsed_triples, config.batch_size)

        # Compute losses
        wordnet_loss = 0
        wordnet_grads = []

        for triple in wordnet_batch:
            loss = model.forward(triple)
            wordnet_loss += loss
            wordnet_grads.append(model.compute_gradient(triple))

        parsed_loss = 0
        parsed_grads = []

        for triple in parsed_batch:
            loss = model.forward(triple)
            parsed_loss += loss
            parsed_grads.append(model.compute_gradient(triple))

        # Constraint loss
        constraint_loss = 0
        constraint_grads = []

        for edge in dual_graph.edges:
            if edge.edge_source == "wordnet":  # Only apply constraints to expert labels
                loss = compute_constraint_loss(model, edge)
                constraint_loss += loss
                # Compute gradient (numerical or analytical)
                grad = compute_constraint_gradient(model, edge)
                constraint_grads.append(grad)

        # Combined loss
        total_loss = (
            config.wordnet_weight * wordnet_loss +
            config.parsed_weight * parsed_loss +
            config.constraint_weight * constraint_loss +
            config.anchor_weight * compute_anchor_preservation_loss(model)
        )

        # Combine gradients
        all_grads = (
            [g * config.wordnet_weight for g in wordnet_grads] +
            [g * config.parsed_weight for g in parsed_grads] +
            [g * config.constraint_weight for g in constraint_grads]
        )

        # Update model
        model.update(all_grads, learning_rate=config.learning_rate)

        # Logging
        if epoch % 10 == 0:
            avg_loss = total_loss / (len(wordnet_batch) + len(parsed_batch))
            print(f"Epoch {epoch}/{config.num_epochs}")
            print(f"  Total loss: {avg_loss:.4f}")
            print(f"  WordNet: {wordnet_loss:.4f}")
            print(f"  Parsed: {parsed_loss:.4f}")
            print(f"  Constraint: {constraint_loss:.4f}")

            # Save checkpoint
            save_checkpoint(model, f"checkpoints/epoch_{epoch}.pkl")

            # Validation
            val_loss = evaluate_on_validation_set(model, validation_data)
            print(f"  Validation loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                save_checkpoint(model, "checkpoints/best.pkl")
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    print("Early stopping triggered")
                    break

        # Learning rate decay
        if epoch % 50 == 0 and epoch > 0:
            config.learning_rate *= 0.9

    return model

def compute_anchor_preservation_loss(model):
    """
    Penalize changes to anchor dimensions.

    Anchor dims should stay near their initialization (usually 0).
    """
    loss = 0
    for i in range(model.num_nodes):
        anchor_values = model.embeddings[i, :model.num_anchors]
        loss += np.sum(anchor_values ** 2)
    return loss / model.num_nodes
```

### Training Configuration

```python
@dataclass
class TrainingConfig:
    # Model architecture
    embedding_dim: int = 512
    num_anchors: int = 51

    # Training
    num_epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 0.001

    # Loss weights
    wordnet_weight: float = 0.6      # Expert knowledge
    parsed_weight: float = 0.4       # Parser output
    constraint_weight: float = 0.1   # Fuzzy logical constraints
    anchor_weight: float = 0.05      # Keep anchors fixed

    # Regularization
    l2_reg: float = 1e-5
    early_stopping_patience: int = 20

    # Optimization
    optimizer: str = "adam"          # "adam" or "sgd"
    gradient_clip: float = 5.0       # Clip gradients

    # Checkpointing
    checkpoint_every: int = 10
    checkpoint_dir: str = "checkpoints"
```

### Dimension Analysis

```python
def analyze_embedding_dimensions(model, graph):
    """
    Analyze learned embeddings to find effective dimensionality.

    Techniques:
    1. Variance analysis: Which dims capture most variance?
    2. PCA: How many components needed for 95% variance?
    3. Correlation analysis: Are some dims redundant?
    """
    embeddings = model.embeddings[:, model.num_anchors:]  # Only learned dims

    # Variance per dimension
    variances = np.var(embeddings, axis=0)
    sorted_indices = np.argsort(variances)[::-1]

    print("Top 10 highest-variance dimensions:")
    for i in range(10):
        idx = sorted_indices[i]
        print(f"  Dim {idx + model.num_anchors}: variance = {variances[idx]:.4f}")

    # Cumulative variance
    cumsum = np.cumsum(sorted(variances, reverse=True))
    total_var = np.sum(variances)
    cumsum_pct = cumsum / total_var

    effective_dim = np.argmax(cumsum_pct > 0.95) + 1
    print(f"\nEffective dimensionality (95% variance): {effective_dim}")

    # Low-variance dimensions (candidates for pruning)
    threshold = 0.01 * np.max(variances)
    low_var_dims = np.where(variances < threshold)[0]
    print(f"Low-variance dimensions (<1% of max): {len(low_var_dims)}")

    # Correlation analysis
    corr_matrix = np.corrcoef(embeddings.T)
    high_corr_pairs = np.argwhere(np.abs(corr_matrix) > 0.9)
    high_corr_pairs = high_corr_pairs[high_corr_pairs[:, 0] < high_corr_pairs[:, 1]]
    print(f"Highly correlated dimension pairs (|r| > 0.9): {len(high_corr_pairs)}")

    # Visualization
    plot_variance_distribution(variances, sorted_indices)
    plot_cumulative_variance(cumsum_pct)
    plot_correlation_heatmap(corr_matrix)

    return {
        "effective_dim": effective_dim,
        "low_variance_dims": low_var_dims,
        "high_corr_pairs": high_corr_pairs
    }
```

---

## Sense Splitting Algorithm

### Contradiction Detection

**Goal**: Find nodes with edges pulling in opposite semantic directions.

**Method**: Cluster outgoing edges by target embedding similarity.

```python
def detect_contradictions(graph, embeddings, threshold=0.5):
    """
    Find nodes with contradictory outgoing edges.

    Args:
        graph: KnowledgeGraph
        embeddings: Trained node embeddings
        threshold: Minimum silhouette score to flag as contradictory

    Returns:
        List of {node_id, separation_score, clusters}
    """
    contradictory_nodes = []

    for node_id in graph.nodes.keys():
        # Get outgoing edges
        edges = [e for e in graph.edges if e.source == node_id]

        if len(edges) < 2:
            continue  # Can't have contradiction with <2 edges

        # Get target embeddings
        target_ids = [e.target for e in edges]
        target_vecs = np.array([embeddings[tid] for tid in target_ids])

        # Cluster targets by embedding similarity
        # Use k-means with varying k (2 to min(5, num_targets))
        max_k = min(5, len(target_ids))
        best_score = -1
        best_labels = None
        best_k = 2

        for k in range(2, max_k + 1):
            labels = kmeans(target_vecs, k)
            score = silhouette_score(target_vecs, labels)

            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k

        # High silhouette score → well-separated clusters → contradiction
        if best_score > threshold:
            # Group edges by cluster
            clusters = [[] for _ in range(best_k)]
            for i, edge in enumerate(edges):
                cluster_id = best_labels[i]
                clusters[cluster_id].append(edge)

            contradictory_nodes.append({
                'node_id': node_id,
                'separation_score': best_score,
                'num_senses': best_k,
                'clusters': clusters
            })

    # Sort by separation score (highest first)
    contradictory_nodes.sort(key=lambda x: x['separation_score'], reverse=True)

    return contradictory_nodes

def silhouette_score(vectors, labels):
    """
    Compute silhouette score (measure of cluster quality).

    Score in [-1, 1]:
    - Near +1: Well-clustered (tight clusters, far apart)
    - Near 0: Overlapping clusters
    - Near -1: Misclustered
    """
    n = len(vectors)
    scores = []

    for i in range(n):
        # a(i): Average distance to points in same cluster
        same_cluster = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if same_cluster:
            a_i = np.mean([np.linalg.norm(vectors[i] - vectors[j]) for j in same_cluster])
        else:
            a_i = 0

        # b(i): Average distance to points in nearest different cluster
        other_clusters = set(labels) - {labels[i]}
        b_i = float('inf')

        for cluster in other_clusters:
            cluster_points = [j for j in range(n) if labels[j] == cluster]
            avg_dist = np.mean([np.linalg.norm(vectors[i] - vectors[j]) for j in cluster_points])
            b_i = min(b_i, avg_dist)

        # Silhouette score for this point
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        scores.append(s_i)

    return np.mean(scores)
```

### Node Splitting

**Process:**
1. Create new sense nodes (one per cluster)
2. Initialize embeddings near original
3. Reassign edges to appropriate sense
4. Retrain local neighborhood

```python
def split_node(graph, embeddings, node_id, clusters):
    """
    Split a node into multiple senses based on edge clusters.

    Args:
        graph: KnowledgeGraph
        embeddings: Current embeddings (will be modified)
        node_id: Node to split
        clusters: List of edge lists (one per sense)

    Returns:
        List of new sense node IDs
    """
    original_node = graph.nodes[node_id]
    original_embedding = embeddings[node_id]

    new_sense_ids = []

    for i, cluster in enumerate(clusters):
        # Create new sense node
        sense_id = f"{node_id}_sense{i}"

        graph.add_node(
            id=sense_id,
            words=original_node.words,  # Same lemmas
            pos=original_node.pos,
            definition=f"Sense {i} of {original_node.words[0]}",
            examples=[],
            source="split",
            parent=node_id
        )

        # Initialize embedding near original + small random noise
        noise = np.random.randn(len(original_embedding)) * 0.01
        embeddings[sense_id] = original_embedding + noise

        # Reassign edges in this cluster
        for edge in cluster:
            edge.source = sense_id

        new_sense_ids.append(sense_id)

    # Mark original node as deprecated (or remove it)
    graph.nodes[node_id].deprecated = True

    # Retrain local neighborhood (2-hop radius from new senses)
    affected_nodes = get_neighborhood(graph, new_sense_ids, radius=2)
    retrain_local(graph, embeddings, affected_nodes, num_epochs=50)

    return new_sense_ids

def get_neighborhood(graph, seed_nodes, radius):
    """Get all nodes within radius hops of seed nodes."""
    visited = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(radius):
        next_frontier = set()
        for node in frontier:
            # Add neighbors
            for edge in graph.edges:
                if edge.source == node:
                    next_frontier.add(edge.target)
                if edge.target == node:
                    next_frontier.add(edge.source)
        visited.update(next_frontier)
        frontier = next_frontier

    return list(visited)

def retrain_local(graph, embeddings, affected_nodes, num_epochs):
    """
    Retrain embeddings for local neighborhood only.

    Freeze unaffected nodes, update only affected ones.
    """
    # Extract subgraph
    subgraph_edges = [
        e for e in graph.edges
        if e.source in affected_nodes or e.target in affected_nodes
    ]

    # Training loop (same as main training, but only update affected_nodes)
    for epoch in range(num_epochs):
        for edge in subgraph_edges:
            # Compute loss and gradient
            # ... (same as main training)

            # Update only if node is affected
            if edge.source in affected_nodes:
                # Update source embedding
                ...
            if edge.target in affected_nodes:
                # Update target embedding
                ...
```

### Validation Against WordNet

**Goal**: Measure how well discovered splits match expert-labeled senses.

```python
def validate_splitting(discovered_splits, wordnet_ground_truth):
    """
    Compare discovered sense splits to WordNet ground truth.

    Args:
        discovered_splits: Dict[word, List[sense_ids]]
        wordnet_ground_truth: Dict[word, List[synset_ids]]

    Returns:
        Dict with precision, recall, F1 scores
    """
    precision_scores = []
    recall_scores = []

    for word in discovered_splits.keys():
        if word not in wordnet_ground_truth:
            continue  # Can't evaluate if no ground truth

        discovered = set(discovered_splits[word])
        true_senses = set(wordnet_ground_truth[word])

        # How many discovered senses are real?
        # (This is tricky: discovered senses don't map 1:1 to WordNet synsets)
        # Approach: Use edge overlap

        # For each discovered sense, find closest matching WordNet synset
        matches = []
        for disc_sense in discovered:
            disc_edges = get_edges(graph, disc_sense)
            best_match = None
            best_overlap = 0

            for true_sense in true_senses:
                true_edges = get_edges(wordnet_graph, true_sense)
                overlap = compute_edge_overlap(disc_edges, true_edges)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = true_sense

            if best_overlap > 0.5:  # Threshold: >50% overlap = match
                matches.append(best_match)

        # Precision: What fraction of discovered senses are valid?
        precision = len(set(matches)) / len(discovered) if discovered else 0

        # Recall: What fraction of true senses were discovered?
        recall = len(set(matches)) / len(true_senses) if true_senses else 0

        precision_scores.append(precision)
        recall_scores.append(recall)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': f1,
        'num_words_evaluated': len(precision_scores)
    }

def compute_edge_overlap(edges1, edges2):
    """
    Compute overlap between two edge sets.

    Returns fraction of shared edges (by target word, not exact node ID).
    """
    targets1 = set(get_word(e.target) for e in edges1)
    targets2 = set(get_word(e.target) for e in edges2)

    intersection = targets1 & targets2
    union = targets1 | targets2

    return len(intersection) / len(union) if union else 0
```

---

## Evaluation Methodology

### Metrics

#### 1. Synonym Detection (Target: 80%+)

**Test**: For each word, are its synonyms in top-5 nearest neighbors?

```python
def test_synonym_detection(embeddings, wordnet):
    """
    Test synonym detection accuracy.

    For each synset, check if other lemmas in same synset are close in vector space.
    """
    correct = 0
    total = 0

    for synset in wordnet.all_synsets():
        lemmas = synset.lemma_names()

        if len(lemmas) < 2:
            continue  # No synonyms to test

        for i, lemma1 in enumerate(lemmas):
            # Get other lemmas as synonyms
            synonyms = [l for j, l in enumerate(lemmas) if j != i]

            # Find nearest neighbors to lemma1
            vec1 = embeddings.get(lemma1.lower())
            if vec1 is None:
                continue

            neighbors = find_nearest_neighbors(vec1, embeddings, k=5)

            # Are any synonyms in top-5?
            if any(syn.lower() in neighbors for syn in synonyms):
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Synonym detection accuracy: {accuracy:.2%} ({correct}/{total})")

    return accuracy

def find_nearest_neighbors(vec, embeddings, k=5):
    """Find k nearest neighbors in embedding space."""
    similarities = {
        word: cosine_similarity(vec, emb)
        for word, emb in embeddings.items()
    }

    sorted_words = sorted(similarities.keys(), key=lambda w: similarities[w], reverse=True)

    # Skip first (itself)
    return sorted_words[1:k+1]
```

#### 2. Antonym Separation (Target: 90%+)

**Test**: Are antonyms in bottom 10% of similarities?

```python
def test_antonym_separation(embeddings, wordnet):
    """
    Test antonym separation.

    Antonyms should be far apart (low similarity).
    """
    correct = 0
    total = 0

    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            antonyms = [ant.name() for ant in lemma.antonyms()]

            if not antonyms:
                continue

            vec1 = embeddings.get(lemma.name().lower())
            if vec1 is None:
                continue

            # Compute similarities to all words
            all_sims = [
                cosine_similarity(vec1, emb)
                for word, emb in embeddings.items()
            ]

            # Check each antonym
            for ant in antonyms:
                vec_ant = embeddings.get(ant.lower())
                if vec_ant is None:
                    continue

                sim = cosine_similarity(vec1, vec_ant)

                # What percentile is this similarity?
                percentile = percentileofscore(all_sims, sim)

                # Should be in bottom 10%
                if percentile < 10:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Antonym separation accuracy: {accuracy:.2%} ({correct}/{total})")

    return accuracy
```

#### 3. Hypernym Alignment (Target: 70%+)

**Test**: Does word + IS-A offset ≈ hypernym?

```python
def test_hypernym_alignment(embeddings, model, wordnet):
    """
    Test hypernym relationship encoding.

    Check if v(dog) + IS-A offset ≈ v(animal).
    """
    correct = 0
    total = 0

    hypernym_offset = model.relation_offsets["hypernym"]

    for synset in wordnet.all_synsets():
        hypernyms = synset.hypernyms()

        if not hypernyms:
            continue

        # Get synset embedding (average of lemmas)
        synset_vec = get_synset_embedding(synset, embeddings)
        if synset_vec is None:
            continue

        # Predict hypernym
        predicted = synset_vec + hypernym_offset

        # Find nearest neighbors to predicted vector
        neighbors = find_nearest_synsets(predicted, wordnet, embeddings, k=10)

        # Are true hypernyms in top-10?
        true_hypernym_ids = [h.name() for h in hypernyms]
        if any(hid in neighbors for hid in true_hypernym_ids):
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Hypernym alignment accuracy: {accuracy:.2%} ({correct}/{total})")

    return accuracy

def get_synset_embedding(synset, embeddings):
    """Average embeddings of all lemmas in synset."""
    lemma_vecs = []
    for lemma in synset.lemma_names():
        vec = embeddings.get(lemma.lower())
        if vec is not None:
            lemma_vecs.append(vec)

    return np.mean(lemma_vecs, axis=0) if lemma_vecs else None
```

#### 4. Sense Splitting Accuracy (Target: 60%+ F1)

**Test**: Do discovered splits match WordNet senses?

```python
def test_sense_splitting(discovered_splits, wordnet):
    """
    Test sense splitting against WordNet ground truth.

    Already implemented in validate_splitting() above.
    """
    results = validate_splitting(discovered_splits, wordnet)

    print(f"Sense splitting results:")
    print(f"  Precision: {results['precision']:.2%}")
    print(f"  Recall: {results['recall']:.2%}")
    print(f"  F1: {results['f1']:.2%}")

    return results['f1']
```

### Comprehensive Evaluation Report

```python
def run_full_evaluation(model, graph, wordnet, discovered_splits):
    """
    Run all evaluation tests and generate report.

    Returns:
        Dict with all metrics
    """
    results = {}

    print("="*60)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*60)

    # Test 1: Synonym Detection
    print("\n1. SYNONYM DETECTION")
    results['synonym_accuracy'] = test_synonym_detection(model.embeddings, wordnet)
    results['synonym_pass'] = results['synonym_accuracy'] >= 0.80

    # Test 2: Antonym Separation
    print("\n2. ANTONYM SEPARATION")
    results['antonym_accuracy'] = test_antonym_separation(model.embeddings, wordnet)
    results['antonym_pass'] = results['antonym_accuracy'] >= 0.90

    # Test 3: Hypernym Alignment
    print("\n3. HYPERNYM ALIGNMENT")
    results['hypernym_accuracy'] = test_hypernym_alignment(model.embeddings, model, wordnet)
    results['hypernym_pass'] = results['hypernym_accuracy'] >= 0.70

    # Test 4: Sense Splitting
    print("\n4. SENSE SPLITTING")
    results['splitting_f1'] = test_sense_splitting(discovered_splits, wordnet)
    results['splitting_pass'] = results['splitting_f1'] >= 0.60

    # Overall MVP Success
    print("\n" + "="*60)
    print("MVP SUCCESS CRITERIA")
    print("="*60)

    all_passed = all([
        results['synonym_pass'],
        results['antonym_pass'],
        results['hypernym_pass'],
        results['splitting_pass']
    ])

    print(f"Synonym Detection:   {'✓' if results['synonym_pass'] else '✗'} ({results['synonym_accuracy']:.1%})")
    print(f"Antonym Separation:  {'✓' if results['antonym_pass'] else '✗'} ({results['antonym_accuracy']:.1%})")
    print(f"Hypernym Alignment:  {'✓' if results['hypernym_pass'] else '✗'} ({results['hypernym_accuracy']:.1%})")
    print(f"Sense Splitting:     {'✓' if results['splitting_pass'] else '✗'} ({results['splitting_f1']:.1%})")

    print("\n" + "="*60)
    if all_passed:
        print("MVP COMPLETE: All success criteria met! 🎉")
    else:
        print("MVP IN PROGRESS: Some criteria not yet met.")
    print("="*60)

    return results
```

---

## Future Extensions

### 1. Morpheme Composition

**Goal**: Represent words as compositions of stems + morphemes.

**Approach**:
- Stems get learned embeddings
- Morphemes get transformation operators
- "unhappiness" = NOT(HAPPY + NOUN_IFICATION)

**Implementation**:
- Add morphological analyzer (morfessor or rule-based)
- Learn morpheme operators (matrices or functions)
- Train compositional model

### 2. Continuous Learning

**Goal**: Accept new truths without retraining entire model.

**Approach**:
- Add new triple → update graph
- Detect affected region (k-hop neighborhood)
- Freeze distant nodes, retrain local region only
- Validate consistency (check for contradictions)

### 3. Multi-Lingual Extension

**Goal**: Extend to Spanish, then more languages.

**Approach**:
- Parse Spanish Wiktionary with Spanish grammar
- Extract Spanish triples
- Add to same semantic space (shared nodes for concepts)
- Train: "perro" and "dog" → same embedding

**Validation**: Cross-lingual synonym detection

### 4. Perception Libraries

**Goal**: Map vision/audio to same semantic space.

**Approach**:
- Image embeddings (from ResNet, CLIP, etc.)
- Map to semantic space via learned projection
- "dog" (word) close to <image of dog>

### 5. Logical Reasoning

**Goal**: Use logical operators (AND, OR, NOT) for inference.

**Approach**:
- Define vector operations for logical ops
- AND(v1, v2) = intersection in space
- NOT(v) = reflection across origin
- Train to satisfy logical constraints

**Application**:
- Query: "What is NOT a dog but IS an animal?"
- Answer: NOT(dog) ∩ IS-A(animal) → finds "cat", "bird", etc.

### 6. Dialogue Systems

**Goal**: Enable conversational AI through semantic space navigation.

**Approach**:
- Encode user query → vector
- Navigate space to find relevant concepts
- Decode back to natural language response

### 7. Translation Beyond Language

**Goal**: Translate between modalities (text ↔ image ↔ code ↔ math).

**Approach**:
- Same semantic space for all modalities
- Each modality has encoder/decoder
- Translation: Modality A → semantic vector → Modality B

---

## Appendix: Implementation Details

### File Organization

See `README.md` for directory structure.

### Dependencies

See `requirements.txt`:
```
numpy>=1.24.0
scipy>=1.10.0
networkx>=3.0
nltk>=3.8
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.2.0
pandas>=2.0.0
tqdm>=4.65.0
```

### Performance Optimization

**CPU Training Strategies:**
1. **Vectorization**: Use NumPy operations (avoid Python loops)
2. **Sparse matrices**: For large graphs, use scipy.sparse
3. **Batching**: Process multiple triples in parallel
4. **Multiprocessing**: Parallelize definition parsing
5. **Caching**: Store frequently-accessed embeddings
6. **Early stopping**: Don't overtrain
7. **Dimension reduction**: Prune low-variance dims after initial training

**Expected Performance:**
- Parsing: ~10 sentences/second
- Training: ~1000 triples/second on CPU
- Full training (10K words, 200 epochs): 8-12 hours overnight

### Testing Strategy

**Unit Tests:**
- DSL parser (grammar file validation)
- Rule matching logic
- Hypothesis tracking
- Triple extraction

**Integration Tests:**
- End-to-end parsing (sentence → parse tree)
- Graph construction (triples → graph)
- Training pipeline (graph → embeddings)
- Splitting algorithm (contradictions → senses)

**Regression Tests:**
- Parse output consistency (same sentence → same tree)
- Embedding stability (same data → same vectors ± noise)

---

## Glossary

**AGI**: Artificial General Intelligence
**Chart**: Parse data structure storing multiple hypotheses
**Connection**: Directed edge between nodes (SUBJECT, OBJECT, etc.)
**Contradiction**: Node with semantically conflicting edges
**DSL**: Domain-Specific Language (grammar rule syntax)
**Dual-Graph**: Graph combining WordNet + parsed relationships
**Embedding**: Vector representation of a word/concept
**Fuzzy Logical Constraint**: Continuous approximation of logical relationship (synonym ~, antonym !~)
**Hypothesis**: A single parse interpretation
**Hypernym**: Is-a relationship (dog → animal)
**Hyponym**: Inverse of hypernym (animal → dog)
**Meronym**: Part-of relationship (wheel → car)
**Node**: Element in parse tree (word or constituent)
**Quantum Parser**: Parser maintaining multiple parse hypotheses simultaneously
**Semantic Triple**: (subject, relation, object) extracted from text
**Sense**: Distinct meaning of a polysemous word
**SubCategory**: Grammatical feature category (GENDER, NUMBER, etc.)
**SubType**: Specific value of subcategory (MASCULINE, PLURAL, etc.)
**Synset**: Synonym set in WordNet
**TransE**: Translation-based embedding (h + r ≈ t)
**WordNet**: Lexical database of English

---

**End of Architecture Document**

Version 1.0 - 2025-11-22
Author: William (with Claude assistance)
Status: Initial design, ready for implementation
