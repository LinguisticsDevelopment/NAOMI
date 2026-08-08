"""Domain signal (topic/region/usage domains)."""

from typing import Any, Dict, List

from .relations import RelationGraph


def extras(vocab: List[str], graph: RelationGraph) -> Dict[str, Any]:
    """Extract domain features in-vocabulary."""
    feature_extra = {w: graph.domain[w] for w in graph.domain if graph.domain[w]}
    return {"feature_extra": feature_extra}
