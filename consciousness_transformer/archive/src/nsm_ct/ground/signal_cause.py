"""Cause signal (synset.causes())."""

from typing import Any, Dict, List, Tuple

from .relations import RelationGraph


def extras(vocab: List[str], graph: RelationGraph) -> Dict[str, Any]:
    """Extract cause pairs in-vocabulary."""
    pairs = [
        (w, x)
        for w in graph.gloss
        for x in graph.cause.get(w, [])
        if x in graph.gloss and x != w
    ]
    return {"close_extra": pairs}
