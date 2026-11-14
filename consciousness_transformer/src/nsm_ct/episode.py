"""Slim Episode record (the rest of the curriculum machinery is archived)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

_NAMES = ["mary", "john", "sandra", "daniel", "bill", "fred"]


@dataclass
class Episode:
    """A single reasoning episode.

    Attributes:
        context: Ordered statements (the stream to absorb into memory).
        question: The question that should trigger a response.
        answer_text: The correct answer as a string (open-ended supervision).
        options: For multiple choice, the candidate answers (correct one included).
        answer_idx: Index of the correct option in ``options`` (MC supervision).
        level: Curriculum difficulty (0 = easiest). Provenance/debug only.
        meta: Free-form provenance.
    """

    context: List[str]
    question: str
    answer_text: str
    options: Optional[List[str]] = None
    answer_idx: Optional[int] = None
    level: int = 0
    post_context: List[str] = field(default_factory=list)  # distractors AFTER the question
    trust_labels: Optional[List[float]] = None  # per-context-item: 1 trustworthy, 0 contradicted (metrics only)
    disjuncts: Optional[List[str]] = None  # the OR alternatives (e.g. ["kitchen", "office"])
    truth_labels_per_disjunct: Optional[List[float]] = None  # per-disjunct truth (metrics only)
    # Multi-question episodes: item indices of ALL questions in the stream and the
    # correct option index per question (the shared `options` list scores them all).
    question_positions: Optional[List[int]] = None
    question_targets: Optional[List[int]] = None
    # Reasoning levels (oracle-provided): the gold derivation chain and whether the
    # query is derivable at all (False -> the gold answer is "idk"/abstain).
    gold_chain: Optional[List] = None
    answerable: bool = True
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.options is not None:
            if self.answer_idx is None or not 0 <= self.answer_idx < len(self.options):
                raise ValueError("answer_idx must index into options")
            if self.options[self.answer_idx] != self.answer_text:
                # Keep them consistent; the text is the source of truth.
                self.answer_text = self.options[self.answer_idx]

    @property
    def is_multiple_choice(self) -> bool:
        return self.options is not None

