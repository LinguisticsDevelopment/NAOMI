"""A stateful conversation session (M14, L2 + L4) — natural back-and-forth.

``ConsciousLoop.converse`` is stateless: every call resets, one query → one terse
answer or a dead-end "I don't know." A real dialogue needs *continuity* (L4) and
*cooperative limits* (L2). :class:`Conversation` adds both, deterministically, with
no training:

* **L4 — cross-turn memory.** Facts/rules taught in one turn persist and are used in
  later turns; a persistent :class:`~nsm_ct.mind.coref.Coref` resolves "she/it" across
  turns; a ``topic`` tracks the last subject.
* **L2 — ask when blocked.** When a query can't be derived, instead of dead-ending the
  session computes the **exact missing premise** (one-hop backward,
  :func:`~nsm_ct.reasoning_oracle.find_missing_premise`) and **asks for it** as an
  English question, remembering the blocked query. When a later turn supplies the
  premise, the pending query **resolves** ("Then yes — …"). Deeper chains surface one
  premise per turn — itself natural back-and-forth.

It wraps a :class:`~nsm_ct.mind.conscious_loop.ConsciousLoop` and re-feeds the
*accumulated* statements to the existing ``consume`` each turn, so the door's per-call
reset is a non-issue — no surgery on the runtime path. Grounded by construction: it
only ever asks for a premise a rule actually needs, and only answers what it derives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from ..reasoning_oracle import Rule, find_missing_premise
from . import grammar, ops, verbalize
from .conscious_loop import ConsciousLoop, _norm_fact, _norm_rule

try:
    from .datasets.proofwriter import UNKNOWN as _UNKNOWN
except Exception:                                        # pragma: no cover - defensive
    _UNKNOWN = "Unknown"


@dataclass
class TurnOutcome:
    """The structured result of one processed sentence — the per-turn reward/telemetry
    signal a future learned drive (L6) would optimize. Recorded; not yet consumed by any
    policy. ``kind`` ∈ {answered, asked, resolved, abstained, learned}; ``knowledge_gained``
    is the count of newly-derivable facts a statement added."""

    kind: str
    detail: str = ""
    knowledge_gained: int = 0


def _split_sentences(text: str) -> List[str]:
    """Split an utterance into sentences, keeping the terminal '.'/'?'."""
    parts = re.findall(r"[^.?]*[.?]", text)
    return [p.strip() for p in parts if p.strip()] or ([text.strip()] if text.strip() else [])


class Conversation:
    """One stateful dialogue session over a :class:`ConsciousLoop`."""

    def __init__(self, loop: ConsciousLoop) -> None:
        self.loop = loop
        from . import coref
        self.statements: List[tuple] = []                # accumulated fact/disj/rule objects
        self.tracker = coref.Coref()                     # persistent cross-turn coreference
        self.topic = None                                # last subject mentioned
        self.pending: List[Tuple[tuple, tuple]] = []     # (blocked query obj, missing premise)
        self.log: List[TurnOutcome] = []                 # per-turn telemetry (L6 signal)

    # -- one turn ------------------------------------------------------------
    def say(self, text: str) -> List[str]:
        """Process one user utterance (possibly several sentences) → English replies.

        Statements are absorbed (and may resolve a pending question); questions are
        answered, or — if blocked — turned into a request for the missing premise.
        """
        replies: List[str] = []
        for sentence in _split_sentences(text):
            obj = grammar.parse(sentence)
            if obj is None:
                continue                                 # unparsable → abstain silently
            obj = self.loop._resolve_refs(obj, self.tracker)
            if obj[0] == "query":
                replies.append(self._handle_query(obj))
            else:                                        # fact / disj / rule
                self.statements.append(obj)
                if obj[0] in ("fact", "disj"):
                    self.topic = obj[1]
                resolutions = self._retry_pending()      # what this statement unblocked
                # knowledge-gained signal (cheap): the count of pending questions this
                # statement made answerable — the immediately-useful new knowledge.
                self.log.append(TurnOutcome("learned", sentence, len(resolutions)))
                replies.extend(resolutions)
        return replies

    # -- answering / asking --------------------------------------------------
    def _handle_query(self, obj) -> str:
        resp = self._reason(obj)
        answer, query = resp["answer"], resp["query"]
        if not self._blocked(answer):
            self.topic = query[0]
            self.log.append(TurnOutcome("answered", str(query)))
            return self._render_answer(query, answer, resp)
        premise = find_missing_premise(self._facts(), self._rules(), query)
        if premise is not None:                          # cooperative ASK (L2)
            self.pending.append((obj, premise))
            self.log.append(TurnOutcome("asked", str(premise[:3])))
            return verbalize.verbalize_ask(premise)
        self.log.append(TurnOutcome("abstained", str(query)))
        return self._render_answer(query, answer, resp)  # honest "I don't know."

    def _retry_pending(self) -> List[str]:
        """Re-attempt blocked queries now that new knowledge has arrived (L2+L4)."""
        replies, still = [], []
        for (obj, _premise) in self.pending:
            resp = self._reason(obj)
            if self._blocked(resp["answer"]):
                still.append((obj, _premise))
            else:
                self.topic = resp["query"][0]
                self.log.append(TurnOutcome("resolved", str(resp["query"])))
                replies.append(verbalize.verbalize_resolution(resp["query"], resp["answer"]))
        self.pending = still
        return replies

    # -- helpers -------------------------------------------------------------
    def _reason(self, query_obj):
        """Run the accumulated theory + this query through the existing door."""
        return self.loop.consume(self.statements + [query_obj])[-1]

    @staticmethod
    def _blocked(answer) -> bool:
        return answer in (ops.ABSTAIN, _UNKNOWN, None)

    def _facts(self):
        return [_norm_fact(o) for o in self.statements if o[0] == "fact"]

    def _rules(self) -> List[Rule]:
        return [_norm_rule(o) for o in self.statements
                if o[0] == "rule" or (len(o) > 1 and isinstance(o[1], Rule))]

    def _render_answer(self, query, answer, resp) -> str:
        if len(query) >= 4:                              # yes/no verdict
            return verbalize.verbalize_verdict(answer)
        return verbalize.verbalize_answer(query, answer)


__all__ = ["Conversation", "TurnOutcome"]
