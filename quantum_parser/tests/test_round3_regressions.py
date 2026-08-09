"""Regression tests for the round-3 parser fixes (M45 + the bare-passive /
object-relative-clause / clause-coordination follow-ups).

Each test pins one mechanism at the engine level (the end-to-end behavior is
also covered by consciousness_transformer's probe_parser_stress battery and
tests/test_discourse.py, but those exercise the extraction layer too --
these isolate the parser/grammar layer alone):

1. Quantum-branching fix (quantum_parser.py): an ambiguous anchor no longer
   discards another, independent anchor's unambiguous transformation.
2. PUNCT tagging (pos_tagger.py): bare punctuation is tagged PUNCT/NIL, never
   falls through to a phantom NOMINAL.
3. completeness_key subject_count tie-break (scorer.py).
4. The object-relative-clause grammar rule (rel1, round-3 addition): an
   object gap ("the ball that mary found") merges into a single NOMINAL the
   same way a subject gap ("the man who came") always has.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parser import ConnectionType, Edge, Hypothesis, Node, NodeType, QuantumParser, Tag
from parser.pos_tagger import tag_sentence, tag_words
from parser.scorer import completeness_key

_GRAMMAR = str(Path(__file__).resolve().parent.parent / "grammars" / "english.json")


def _find(hyp: Hypothesis, text: str) -> int:
    return next(i for i, n in enumerate(hyp.nodes) if n.value and n.value.text == text)


def _has_edge(hyp: Hypothesis, etype: ConnectionType, parent_idx: int, child_idx: int) -> bool:
    return any(e.type == etype and e.parent == parent_idx and e.child == child_idx for e in hyp.edges)


# -- 1. quantum-branching: ambiguous anchor must not eat independent matches --

def test_ambiguous_anchor_does_not_discard_independent_transformation():
    """'thinks' is ambiguous (intransitive vs. transitive-object reading);
    the embedded 'is' -> PREDICATE(PP) transformation is independent of that
    ambiguity and must still land on the winning branch (M45's fix)."""
    parser = QuantumParser(_GRAMMAR)
    words = tag_sentence("mary thinks the ball is in the shed .")
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    assert best is not None
    is_idx = _find(best, "is")
    ball_idx = _find(best, "ball")
    assert _has_edge(best, ConnectionType.SUBJECT, is_idx, ball_idx), (
        "the independent 'is'->SUBJECT(ball) edge must survive even though "
        "'thinks' is an ambiguous anchor in the same pass"
    )


# -- 2. PUNCT tagging -----------------------------------------------------

def test_bare_punctuation_tagged_punct():
    (w,) = tag_words(["."])
    assert w.pos == Tag.PUNCT


def test_bare_punctuation_stays_nil_type_in_a_real_parse():
    """Before the fix, a sentence-final '.' fell through to the default-NOUN
    heuristic and could satisfy a rule's 'NOMINAL after' pattern (the
    PLACE='.' artifact). It must end up NodeType.NIL, never NOMINAL/NOUN."""
    parser = QuantumParser(_GRAMMAR)
    words = tag_sentence("the window was broken .")
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    assert best is not None
    period_idx = _find(best, ".")
    assert best.nodes[period_idx].type == NodeType.NIL


# -- 3. completeness_key subject_count tie-break ---------------------------

def _synthetic_hyp(n_subjects: int, n_other_core: int = 0) -> Hypothesis:
    """A minimal hypothesis exposing only what completeness_key reads (edge
    types/counts) -- node content is irrelevant to the function under test."""
    nodes = [Node(NodeType.NIL, NodeType.NIL, None, Tag.NOUN) for _ in range(4)]
    edges = [Edge(ConnectionType.SUBJECT, 0, 1) for _ in range(n_subjects)]
    edges += [Edge(ConnectionType.OBJECT, 2, 3) for _ in range(n_other_core)]
    return Hypothesis(nodes=nodes, edges=edges)


def test_completeness_key_any_subject_beats_none():
    has_subject = _synthetic_hyp(1, n_other_core=0)
    no_subject = _synthetic_hyp(0, n_other_core=10)
    assert completeness_key(has_subject) > completeness_key(no_subject)


def test_completeness_key_more_subjects_beats_more_other_roles():
    """Two complete clauses (two SUBJECT edges) must outrank a reading that
    folded one clause's subject into an OBJECT elsewhere, even though the
    latter has more OBJECT/INDIRECT_OBJECT/SUBJECT_COMPLEMENT edges."""
    two_subjects = _synthetic_hyp(2, n_other_core=0)
    one_subject_many_objects = _synthetic_hyp(1, n_other_core=5)
    assert completeness_key(two_subjects) > completeness_key(one_subject_many_objects)


# -- 4. object-relative clause (round-3 rel1 addition) ---------------------

def test_object_relative_clause_merges_into_one_nominal():
    """'the ball that mary found' must collapse to a single unconsumed
    NOMINAL ('ball') carrying the embedded clause via SUBORDINATION, exactly
    like the subject-gap case ('the man who came'), so a later matrix
    predicate can still take it as its own subject."""
    parser = QuantumParser(_GRAMMAR)
    words = tag_sentence("the ball that mary found is in the garden .")
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    assert best is not None
    ball_idx = _find(best, "ball")
    mary_idx = _find(best, "mary")
    found_idx = _find(best, "found")
    is_idx = _find(best, "is")
    # the embedded clause: found's own subject is mary, subordinated to ball
    assert _has_edge(best, ConnectionType.SUBJECT, found_idx, mary_idx)
    assert _has_edge(best, ConnectionType.SUBORDINATION, ball_idx, found_idx)
    # "ball" is still available, unconsumed, as the matrix subject of "is"
    assert _has_edge(best, ConnectionType.SUBJECT, is_idx, ball_idx)


def test_subject_relative_clause_unaffected_by_the_new_rule():
    """Regression guard: 'the man who came' (subject gap, the pre-existing
    rel1 rule) must still produce no SUBJECT edge for the relative clause
    itself -- 'who' stands for the head noun, there is no separate subject."""
    parser = QuantumParser(_GRAMMAR)
    words = tag_sentence("the man who came is in the kitchen .")
    chart = parser.parse(words)
    best = chart.best_hypothesis()
    assert best is not None
    man_idx = _find(best, "man")
    came_idx = _find(best, "came")
    is_idx = _find(best, "is")
    assert _has_edge(best, ConnectionType.SUBORDINATION, man_idx, came_idx)
    assert not any(e.type == ConnectionType.SUBJECT and e.child == man_idx for e in best.edges
                   if e.parent == came_idx)
    assert _has_edge(best, ConnectionType.SUBJECT, is_idx, man_idx)
