"""Also-see signal (synset.also_sees())."""

from typing import Any, Dict, List, Tuple

from .relations import RelationGraph


def extras(vocab: List[str], graph: RelationGraph) -> Dict[str, Any]:
    """Extract also-see pairs in-vocabulary."""
    pairs = [
        (w, x)
        for w in graph.gloss
        for x in graph.also_see.get(w, [])
        if x in graph.gloss and x != w
    ]
    return {"close_extra": pairs}
