"""
Semantic scorer for ranking parse hypotheses.

Scores hypotheses by structural coherence and optional semantic similarity.
"""

import math
from typing import Optional, Dict, Any


from .data_structures import Hypothesis


def score_hypothesis(hypothesis: Hypothesis, embeddings: Optional[Dict[str, Any]] = None) -> float:
    """
    Score hypothesis by structural coherence and semantic plausibility.

    Args:
        hypothesis: Hypothesis to score
        embeddings: Optional word embeddings for semantic scoring

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

    Returns:
        Float in [0, 1]
    """
    num_nodes = len(hypothesis.nodes)
    if num_nodes == 0:
        return 0.0

    num_consumed = len(hypothesis.consumed)
    num_edges = len(hypothesis.edges)

    # Coverage: What fraction of nodes are consumed?
    coverage = num_consumed / num_nodes

    # Connectivity: Should have roughly num_nodes - 1 edges (tree property)
    expected_edges = num_nodes - 1
    if expected_edges > 0:
        connectivity = 1.0 - abs(num_edges - expected_edges) / num_nodes
    else:
        connectivity = 1.0 if num_edges == 0 else 0.0

    # Projectivity: Penalize crossing edges
    crossing_penalty = count_crossing_edges(hypothesis) * 0.1
    projectivity = max(0, 1.0 - crossing_penalty)

    # Balance: Prefer trees with moderate depth
    depth = compute_tree_depth(hypothesis)
    ideal_depth = math.log2(num_nodes) if num_nodes > 1 else 1
    if num_nodes > 0:
        balance = 1.0 - min(1.0, abs(depth - ideal_depth) / num_nodes)
    else:
        balance = 1.0

    # Weighted combination
    score = (
        0.4 * coverage +
        0.3 * connectivity +
        0.2 * projectivity +
        0.1 * balance
    )

    return max(0.0, min(1.0, score))


def compute_semantic_score(hypothesis: Hypothesis, embeddings: Dict[str, Any]) -> float:
    """
    Score based on semantic coherence in vector space.

    Args:
        hypothesis: Hypothesis to score
        embeddings: Word embeddings dictionary

    Returns:
        Float in [0, 1]
    """
    # TODO: Implement when embeddings are available
    # For now, return neutral score
    return 0.5


def count_crossing_edges(hypothesis: Hypothesis) -> int:
    """
    Count number of crossing edge pairs (non-projective structures).

    Two edges (i1, j1) and (i2, j2) cross if:
    i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1

    Returns:
        Number of crossing pairs
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


def compute_tree_depth(hypothesis: Hypothesis) -> int:
    """
    Compute maximum depth of parse tree.

    Returns:
        Maximum depth (0 if no edges)
    """
    if not hypothesis.edges:
        return 0

    # Build adjacency list
    children = {i: [] for i in range(len(hypothesis.nodes))}
    for edge in hypothesis.edges:
        children[edge.parent].append(edge.child)

    # Find roots (unconsumed nodes)
    roots = hypothesis.get_unconsumed()

    if not roots:
        # No clear root, use node 0
        roots = [0]

    # DFS to find maximum depth
    def dfs(node_idx: int) -> int:
        if node_idx not in children or not children[node_idx]:
            return 0
        return 1 + max(dfs(child) for child in children[node_idx])

    return max(dfs(root) for root in roots)
