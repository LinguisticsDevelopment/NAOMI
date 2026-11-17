"""Mind: teach facade over Reader + Memory, and ``ask`` through the learned one-step policy (naomi.mind_v1)."""

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
    DEFAULT_POLICY = "runs/mind_v1/policy_all_best.pt"

    def __init__(self, reader: Optional[Reader] = None, memory: Optional[Memory] = None,
                 cache_path: Optional[str] = None, policy=None, rel_vocab=None):
        self.reader = reader or Reader(cache_path=cache_path)
        self.memory = memory or Memory()
        self.policy, self.rel_vocab = policy, rel_vocab
        self.last_trace: Optional[dict] = None

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

    def _policy(self):
        if self.policy is None:
            from . import mind_v1
            self.policy, self.rel_vocab = mind_v1.load_policy(Reader._resolve(self.DEFAULT_POLICY))
        return self.policy, self.rel_vocab

    def ask(self, question: str, options=None, layers=None) -> Answer:
        """Answer = a memory node and its subtree (rendered as text), or "not stated" (``stated`` False, ``text`` None).
        ``options`` is unused: the policy never sees them. The full trace (candidates, scores, sentences used) is
        ``last_trace`` and, JSON-encoded, ``Answer.trace[0]``."""
        import json
        from . import mind_v1
        policy, rel_vocab = self._policy()
        rr = self.reader.read_question(question)
        if rr.tree is None:
            self.last_trace = {"question": question, "stated": False, "text": None, "candidates_considered": 0,
                               "top": [], "sentence_ids": [], "facts": [], "abstain_prob": 1.0}
            return Answer(text=None, stated=False, trace=[json.dumps(self.last_trace)])
        tr = mind_v1.answer_from_memory(policy, rel_vocab, self.memory, rr.tree, layers)
        self.last_trace = tr
        support = None
        if tr["stated"]:
            refs, _ = mind_v1.node_table(self.memory, layers)
            support = mind_v1.subtree_tree(refs[tr["node_gid"]].node, tr["facts"][0]["sentence"])
        return Answer(text=tr["text"], stated=tr["stated"], support=support, score=float(tr.get("prob", 0.0)),
                      trace=[json.dumps(tr)])

    def sleep(self) -> int:
        return self.memory.consolidate("current", "long_term")

    def forget(self, layer: Optional[str] = None):
        self.memory.clear(layer)

    def dump(self) -> str:
        return self.memory.dump()
