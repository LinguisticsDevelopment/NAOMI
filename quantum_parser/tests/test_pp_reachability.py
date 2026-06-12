"""
Tests for PP-attachment reachability.

Verifies that words in prepositional phrases are reachable
from the root of the best hypothesis by following edges
parent→child (BFS from the single unconsumed root node).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from collections import deque
from src.parser import QuantumParser, Word, Tag, SubType


def reachable_words(hyp):
    """
    BFS from the single unconsumed root, following edges parent→child.

    Returns a set of word texts reachable from the root.
    """
    unconsumed = hyp.get_unconsumed()
    if not unconsumed:
        return set()

    # Build children map: parent_idx -> [child_idx, ...]
    children = {}
    for edge in hyp.edges:
        children.setdefault(edge.parent, []).append(edge.child)

    # BFS starting from root (first unconsumed node)
    root_idx = unconsumed[0]
    visited = set()
    queue = deque([root_idx])
    while queue:
        idx = queue.popleft()
        if idx in visited:
            continue
        visited.add(idx)
        for child_idx in children.get(idx, []):
            queue.append(child_idx)

    # Collect word texts for all visited nodes that have a word value
    words = set()
    for idx in visited:
        node = hyp.nodes[idx]
        if node.value is not None:
            words.add(node.value.text)
    return words


@pytest.fixture(scope="module")
def parser():
    return QuantumParser("grammars/english.json")


def test_mary_is_in_the_kitchen_single_root(parser):
    """'mary is in the kitchen' must parse to a single root."""
    words = [
        Word("mary", Tag.PROPN),
        Word("is", Tag.AUX, [SubType.PROGRESSIVE]),
        Word("in", Tag.ADP),
        Word("the", Tag.DET),
        Word("kitchen", Tag.NOUN),
    ]
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    assert len(best.get_unconsumed()) == 1, (
        f"Expected single root, got {len(best.get_unconsumed())} unconsumed nodes"
    )


def test_mary_is_in_the_kitchen_pp_reachable(parser):
    """'in' and 'kitchen' must be reachable from root in 'mary is in the kitchen'."""
    words = [
        Word("mary", Tag.PROPN),
        Word("is", Tag.AUX, [SubType.PROGRESSIVE]),
        Word("in", Tag.ADP),
        Word("the", Tag.DET),
        Word("kitchen", Tag.NOUN),
    ]
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    reached = reachable_words(best)
    assert "in" in reached, f"'in' not reachable; reachable={reached}"
    assert "kitchen" in reached, f"'kitchen' not reachable; reachable={reached}"


def test_dog_on_table_runs_single_root(parser):
    """'the dog on the table runs' must parse to a single root."""
    words = [
        Word("the", Tag.DET),
        Word("dog", Tag.NOUN),
        Word("on", Tag.ADP),
        Word("the", Tag.DET),
        Word("table", Tag.NOUN),
        Word("runs", Tag.VERB),
    ]
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    assert len(best.get_unconsumed()) == 1, (
        f"Expected single root, got {len(best.get_unconsumed())} unconsumed nodes"
    )


def test_dog_on_table_runs_pp_reachable(parser):
    """'on' and 'table' must be reachable from root in 'the dog on the table runs'."""
    words = [
        Word("the", Tag.DET),
        Word("dog", Tag.NOUN),
        Word("on", Tag.ADP),
        Word("the", Tag.DET),
        Word("table", Tag.NOUN),
        Word("runs", Tag.VERB),
    ]
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    reached = reachable_words(best)
    assert "on" in reached, f"'on' not reachable; reachable={reached}"
    assert "table" in reached, f"'table' not reachable; reachable={reached}"


def test_she_is_in_the_kitchen_reachable(parser):
    """With 'she' tagged as PRON, 'she' and 'in' must be reachable."""
    words = [
        Word("she", Tag.PRON),
        Word("is", Tag.AUX, [SubType.PROGRESSIVE]),
        Word("in", Tag.ADP),
        Word("the", Tag.DET),
        Word("kitchen", Tag.NOUN),
    ]
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    assert len(best.get_unconsumed()) == 1, (
        f"Expected single root, got {len(best.get_unconsumed())} unconsumed nodes"
    )
    reached = reachable_words(best)
    assert "she" in reached, f"'she' not reachable; reachable={reached}"
    assert "in" in reached, f"'in' not reachable; reachable={reached}"


@pytest.mark.xfail(
    reason=(
        "'to' is tagged as Tag.PART (infinitive marker) by default, not Tag.ADP, "
        "so no PP is formed and 'to'/'office' may not be reachable. "
        "Pass Word('to', Tag.ADP) explicitly to force PP attachment."
    ),
    strict=False,
)
def test_john_moved_to_the_office_pp_reachable_auto(parser):
    """
    'john moved to the office' with auto-tagging: 'to' gets Tag.PART, not Tag.ADP.

    This test documents the known tagging conflict and is expected to fail
    under auto-tagging.  The explicit-ADP variant below always passes.
    """
    from src.parser.pos_tagger import simple_tag
    words_auto = [
        Word("john", simple_tag("john")),
        Word("moved", simple_tag("moved")),
        Word("to", simple_tag("to")),
        Word("the", simple_tag("the")),
        Word("office", simple_tag("office")),
    ]
    chart = parser.parse(words_auto)
    best = chart.best_hypothesis()
    reached = reachable_words(best)
    assert "to" in reached, f"'to' not reachable; reachable={reached}"
    assert "office" in reached, f"'office' not reachable; reachable={reached}"


def test_john_moved_to_the_office_explicit_adp(parser):
    """
    'john moved to the office' with 'to' explicitly tagged as ADP.

    When 'to' is forced to ADP a PP forms correctly and 'to'/'office'
    are reachable from the root.
    """
    words = [
        Word("john", Tag.PROPN),
        Word("moved", Tag.VERB),
        Word("to", Tag.ADP),
        Word("the", Tag.DET),
        Word("office", Tag.NOUN),
    ]
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    reached = reachable_words(best)
    assert "to" in reached, f"'to' not reachable; reachable={reached}"
    assert "office" in reached, f"'office' not reachable; reachable={reached}"
