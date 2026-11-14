"""Mind: a tiny teach-then-ask facade over Reader + Memory + recall."""

from __future__ import annotations

import re
from typing import List, Optional

from .memory import Memory
from .reader import Reader
from .recall import _isa, recall
from .types import Answer, ClauseTree, Filler

_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SPLIT.split(text.strip()) if s.strip()]


def render_clause(c: Optional[ClauseTree]) -> str:
    if c is None:
        return ""
    args = " ".join(f"{a.role}={a.lemma}" + ("?" if a.is_hole else "") for a in c.args)
    p = c.predicate.lemma + ("?" if c.predicate.is_hole else "")
    return f"{'NOT ' if c.negated else ''}{p}({args})"


_HE_SHE = frozenset({"he", "she"})


class _Salience:
    """Light coreference: remembers recent non-pronoun subjects so a clause whose SUBJECT is
    he/she/it can record the most recent compatible entity as its ``antecedent``.
    he/she -> a person-like subject (a name WordNet does not know, or a person noun);
    it -> a non-person noun. Deliberately shallow: no gender, no number, recency only."""

    def __init__(self, wn=None, keep: int = 6):
        self.wn, self.keep, self.recent = wn, keep, []

    def _kind(self, f: Filler) -> str:
        if self.wn is not None and not self.wn.synsets(f.lemma):
            return "person"        # unknown word == a name
        return "person" if _isa(f.lemma, "person") else "thing"

    def resolve(self, pron: Filler) -> Optional[dict]:
        want = "person" if pron.lemma in _HE_SHE else ("thing" if pron.lemma == "it" else None)
        if want is None:
            return None
        for f, k, idx in reversed(self.recent):
            if k == want:
                return {"lemma": f.lemma, "token": f.token, "sentence_index": idx}
        return None

    def note(self, c: ClauseTree, idx: int) -> None:
        sj = c.arg("SUBJECT")
        if sj is None or sj.lemma in {"i", "you", "he", "she", "it", "we", "they", "who", "what", "there"}:
            return
        self.recent = [t for t in self.recent if t[0].lemma != sj.lemma][-(self.keep - 1):]
        self.recent.append((sj, self._kind(sj), idx))


class Mind:
    def __init__(self, reader: Optional[Reader] = None, memory: Optional[Memory] = None,
                 cache_path: Optional[str] = None):
        self.reader = reader or Reader(cache_path=cache_path)
        self.memory = memory or Memory()

    def teach(self, text_or_sentences, layer: str = "current", source=None) -> dict:
        sents = (split_sentences(text_or_sentences) if isinstance(text_or_sentences, str)
                 else list(text_or_sentences))
        counts = {"sentences": 0, "clauses": 0, "parser": 0, "encoder": 0, "none": 0, "negated": 0}
        sal = _Salience(self.memory._wn)
        for i, s in enumerate(sents):
            rr = self.reader.read(s, i)
            counts["sentences"] += 1
            counts[rr.source if rr.source in counts else "none"] += 1
            for c in rr.clauses:
                prov = {"source": source, "sentence_index": i, "reader": c.source}
                sj = c.arg("SUBJECT")
                if sj is not None and sj.lemma in {"he", "she", "it"}:
                    ant = sal.resolve(sj)
                    if ant:
                        prov["antecedent"] = ant
                self.memory.store(c, layer, prov)
                sal.note(c, i)
                counts["clauses"] += 1
                counts["negated"] += bool(c.negated)
        return counts

    def read_question(self, question: str):
        return self.reader.read_question(question)

    def ask(self, question: str, options=None, layers=None) -> Answer:
        rr = self.reader.read_question(question)
        if not rr.clauses:
            return Answer(None, trace=["question produced no clause -> abstain"])
        q = rr.clauses[0]
        if q.hole() is None:
            return Answer(None, trace=["question has no hole -> abstain"])
        return recall(self.memory, q, layers=layers, options=options)

    def say(self, answer: Answer) -> str:
        if answer.text is None or not answer.stated:
            return "not stated"
        if answer.negated:
            return f"no - {render_clause(answer.support)} is negated"
        return answer.text

    def sleep(self) -> int:
        return self.memory.consolidate("current", "long_term")

    def forget(self, layer: Optional[str] = None):
        self.memory.clear(layer)
