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
    """LEGACY (old clause contract, kept for naomi.score and the v6 encoder output). One word in a meaning tree: a predicate or a role-bound argument."""

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
    """LEGACY (old clause contract). A predicate plus its role-bound fillers -- the unit that gets stored."""

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


# ----------------------------------------------------------------------------- new tree contract
# Relation labels are the parser grammar's own structural edge types (SUBJECT, OBJECT, DESCRIPTION,
# SPECIFICATION, MODIFICATION, PREPOSITION, SUBORDINATION, COORDINATION, ...), never semantic roles.
# The only labels this package adds are the structural placeholders below.
REL_ROOT = "ROOT"                   # the tree's top node (the main predicate as the parser identifies it)
REL_UNATTACHED = "UNATTACHED"       # the grammar left this token (or component) unattached
KIND_WORD, KIND_NAME = "word", "name"   # a name (proper noun / numeral) is a variable with no meaning point
SOURCE_PARSER, SOURCE_LEGACY, SOURCE_NONE = "parser", "encoder_v6_legacy", "none"


@dataclass
class Node:
    """One token of the sentence. Every non-punctuation token is exactly one Node, attached to
    exactly one head (its parent in the nested ``children`` lists) by the relation ``rel``."""

    rel: str                        # structural relation to the head (grammar edge type), or ROOT/UNATTACHED
    token: str                      # surface token, lowercased
    token_index: Optional[int]      # position in the tagged sentence; None only for the synthetic ROOT
    lemma: str
    pos: str                        # coarse part-of-speech tag as the pipeline's tagger gives it
    kind: str = KIND_WORD           # "word" | "name"
    senses: List[str] = field(default_factory=list)   # candidate senses, best-first, uncommitted
    sense: Optional[str] = None     # memory picks this when the tree is stored
    label: Optional[str] = None     # the parser's own node category (diagnostic only)
    subtypes: List[str] = field(default_factory=list)  # the tagger's own subtype names (diagnostic only)
    children: List["Node"] = field(default_factory=list)


@dataclass
class Tree:
    """A sentence as a tree of Nodes plus the adapter's diagnostics."""

    sentence: str
    root: Node
    source: str = SOURCE_PARSER     # "parser" | "encoder_v6_legacy"
    diagnostics: List[dict] = field(default_factory=list)


@dataclass
class ReadResult:
    sentence: str
    tree: Optional[Tree] = None
    source: str = SOURCE_NONE       # "parser" | "encoder_v6_legacy" | "none"
    seconds: float = 0.0


@dataclass
class Answer:
    text: Optional[str]             # None == "not stated" (abstain)
    stated: bool = False
    support: Optional[Tree] = None
    score: float = 0.0
    trace: List[str] = field(default_factory=list)
