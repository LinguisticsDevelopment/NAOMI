"""Shared data contract for the minimal NAOMI loop (read -> remember -> recall).

This file is the interface between the reader, the memory and recall. Keep it
small; change it only with the director's sign-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Memory layers (working names, not law): one graph, every clause tagged with one.
LAYERS = ("current", "short_term", "long_term", "fact")


@dataclass
class Filler:
    """One word in a meaning tree: a predicate or a role-bound argument."""

    role: str                       # "PREDICATE", "SUBJECT", "OBJECT", "PLACE", ...
    token: str                      # surface token, lowercased
    lemma: str                      # lemma (== token if unknown)
    token_index: Optional[int] = None
    # Candidate word senses (WordNet synset ids), best-first. The reader does NOT
    # commit to one: memory picks `sense` when the clause is stored.
    senses: List[str] = field(default_factory=list)
    sense: Optional[str] = None
    is_hole: bool = False           # True for the wh-slot of a question


@dataclass
class ClauseTree:
    """A predicate plus its role-bound fillers -- the unit that gets stored."""

    predicate: Filler
    args: List[Filler] = field(default_factory=list)
    negated: bool = False           # "not"/"n't"/"never"/"no ..." scoped to this clause
    source: str = "parser"          # "parser" (deterministic teacher) | "encoder" (learned)
    sentence: str = ""
    sentence_index: Optional[int] = None

    def arg(self, role: str) -> Optional[Filler]:
        return next((a for a in self.args if a.role == role), None)

    def hole(self) -> Optional[Filler]:
        if self.predicate.is_hole:
            return self.predicate
        return next((a for a in self.args if a.is_hole), None)


@dataclass
class ReadResult:
    sentence: str
    clauses: List[ClauseTree] = field(default_factory=list)
    source: str = "none"            # "parser" | "encoder" | "none"
    seconds: float = 0.0


@dataclass
class Answer:
    text: Optional[str]             # None == "not stated" (abstain)
    stated: bool = False
    negated: bool = False           # the supporting clause was negated
    support: Optional[ClauseTree] = None
    score: float = 0.0
    trace: List[str] = field(default_factory=list)
