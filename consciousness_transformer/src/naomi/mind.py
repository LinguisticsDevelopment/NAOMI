"""Mind: a tiny teach facade over Reader + Memory. Answering is not built yet."""

from __future__ import annotations

from typing import List, Optional

from .memory import Memory
from .reader import Reader
from .tree import compact
from .types import Answer, Tree


def split_sentences(text: str) -> List[str]:
    """Sentence split of the corpus pipeline (also lowercases/tokenises)."""
    from nsm_ct.corpus import iter_sentences
    return iter_sentences(text)


def render_tree(t: Optional[Tree]) -> str:
    return "" if t is None else compact(t)


class Mind:
    def __init__(self, reader: Optional[Reader] = None, memory: Optional[Memory] = None,
                 cache_path: Optional[str] = None):
        self.reader = reader or Reader(cache_path=cache_path)
        self.memory = memory or Memory()

    def teach(self, text_or_sentences, layer: str = "current", source=None) -> dict:
        sents = (split_sentences(text_or_sentences) if isinstance(text_or_sentences, str)
                 else list(text_or_sentences))
        counts = {"sentences": 0, "trees": 0, "parser": 0, "encoder_v6_legacy": 0, "none": 0}
        for i, s in enumerate(sents):
            rr = self.reader.read(s, i)
            counts["sentences"] += 1
            counts[rr.source if rr.source in counts else "none"] += 1
            if rr.tree is not None:
                self.memory.store(rr.tree, layer, {"source": source, "sentence_index": i, "reader": rr.source})
                counts["trees"] += 1
        return counts

    def ask(self, question: str, options=None, layers=None) -> Answer:
        raise NotImplementedError("answering is learned by the consciousness model; not built yet")

    def sleep(self) -> int:
        return self.memory.consolidate("current", "long_term")

    def forget(self, layer: Optional[str] = None):
        self.memory.clear(layer)

    def dump(self) -> str:
        return self.memory.dump()
