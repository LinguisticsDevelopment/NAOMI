"""
Tests for referential structure: relative clauses, subordinate clauses, and the
infinitive-vs-preposition reading of "to". Verifies the relevant words are
reachable from the root (BFS over parent->child edges), under AUTO tagging
(tag_sentence), matching real end-to-end usage.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import deque
import pytest

from src.parser import QuantumParser
from src.parser.pos_tagger import tag_sentence


def reachable_words(hyp):
    """Set of word texts reachable from the single unconsumed root (parent->child)."""
    unconsumed = hyp.get_unconsumed()
    if not unconsumed:
        return set()
    children = {}
    for edge in hyp.edges:
        children.setdefault(edge.parent, []).append(edge.child)
    visited, queue = set(), deque([unconsumed[0]])
    while queue:
        idx = queue.popleft()
        if idx in visited:
            continue
        visited.add(idx)
        queue.extend(children.get(idx, []))
    return {hyp.nodes[i].value.text for i in visited if hyp.nodes[i].value is not None}


@pytest.fixture(scope="module")
def parser():
    return QuantumParser("grammars/english.json")


def _best(parser, sentence):
    best = parser.parse(tag_sentence(sentence)).best_hypothesis()
    assert best is not None, f"no parse for {sentence!r}"
    return best


def test_relative_clause_pronoun_reachable(parser):
    """'the man who saw mary left' — the relative pronoun and its clause survive."""
    best = _best(parser, "the man who saw mary left")
    reached = reachable_words(best)
    for w in ("man", "who", "saw", "mary"):
        assert w in reached, f"{w!r} not reachable; reachable={reached}"


def test_subordinate_clause_reachable(parser):
    """'john said that he left' — the subordinate clause attaches and is reachable."""
    best = _best(parser, "john said that he left")
    assert len(best.get_unconsumed()) == 1, "subordinate clause should yield one root"
    reached = reachable_words(best)
    for w in ("said", "that", "he", "left"):
        assert w in reached, f"{w!r} not reachable; reachable={reached}"


def test_infinitive_to_reachable(parser):
    """'she wants to run' — 'to' reads as an infinitive marker, all reachable."""
    best = _best(parser, "she wants to run")
    reached = reachable_words(best)
    for w in ("wants", "to", "run"):
        assert w in reached, f"{w!r} not reachable; reachable={reached}"


def test_prepositional_to_reachable(parser):
    """'john moved to the office' — 'to' reads as a preposition, PP reachable."""
    best = _best(parser, "john moved to the office")
    reached = reachable_words(best)
    for w in ("moved", "to", "office"):
        assert w in reached, f"{w!r} not reachable; reachable={reached}"
