"""Entailment signal (synset.entailments())."""

from typing import Any, Dict, List, Tuple

from .relations import RelationGraph


def extras(vocab: List[str], graph: RelationGraph) -> Dict[str, Any]:
    """Extract entailment pairs in-vocabulary."""
    pairs = [
        (w, x)
        for w in graph.gloss
        for x in graph.entailment.get(w, [])
        if x in graph.gloss and x != w
    ]
    return {"close_extra": pairs}
