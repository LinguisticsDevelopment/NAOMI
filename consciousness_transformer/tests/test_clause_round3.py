"""Regression tests for the round-3 parser/extraction fixes.

Covers the three probe_parser_stress fixes landed alongside this batch
(bare passives, object-relative-clause headedness, clausal-coordination
recovery) plus the M45 TODO items that shipped without dedicated tests:
the sentence splitter / graph merge at the encoder seam, coordinated-subject
extraction, and secondary-sentence fact clauses.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause import extract_discourse  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder, _merge_graphs, _split_sentences  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.quantum_adapter import HypGraph  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402

_SENTENCES = [
    "the window was broken .",
    "the ball that mary found is in the garden .",
    "the man who came is in the kitchen .",
    "the ball is in the garden and the bat is in the shed .",
    "mary and john are in the garden .",
    "mary went to the garden . she found the ball .",
    "mary is in the garden .",
]


@pytest.fixture(scope="module")
def parser():
    tok = SimpleTokenizer.build(_SENTENCES, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    p = ParserInputEncoder(tok)
    if getattr(p, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return p


def _roles(cl) -> dict:
    out = {"PRED": (cl.predicate or "").lower()}
    for rel, arg in cl.args:
        out.setdefault(rel, (arg.token or "").lower())
    return out


def _any_match(clauses, expect: dict) -> bool:
    return any(all(_roles(c).get(k) == v for k, v in expect.items()) for c in clauses)


# -- 1. bare passive: SUBJECT-only fact clause -----------------------------

def test_bare_passive_yields_a_subject_only_clause(parser):
    clauses, _links = extract_discourse(parser._parse_graph("the window was broken ."))
    assert _any_match(clauses, {"SUBJECT": "window", "PRED": "broken"})


# -- 2. object-relative clause headedness ----------------------------------

def test_object_relative_clause_recovers_the_matrix_clause(parser):
    graph = parser._parse_graph("the ball that mary found is in the garden .")
    clauses, _links = extract_discourse(graph)
    assert _any_match(clauses, {"SUBJECT": "ball", "PLACE": "garden"})


def test_subject_relative_clause_still_works(parser):
    """Regression guard: the pre-existing subject-gap shape ('who') must not
    be disturbed by the new object-gap rule."""
    graph = parser._parse_graph("the man who came is in the kitchen .")
    clauses, _links = extract_discourse(graph)
    assert _any_match(clauses, {"SUBJECT": "man", "PLACE": "kitchen"})


# -- 3. clausal coordination recovery --------------------------------------

def test_clausal_coordination_recovers_the_second_clause(parser):
    graph = parser._parse_graph("the ball is in the garden and the bat is in the shed .")
    clauses, _links = extract_discourse(graph)
    assert _any_match(clauses, {"SUBJECT": "ball", "PLACE": "garden"})
    assert _any_match(clauses, {"SUBJECT": "bat", "PLACE": "shed"})


def test_coordination_recovery_does_not_fire_without_an_orphan_predicate(parser):
    """The recovery heuristic must stay inert on the legitimate coordination
    shapes it must never touch: coordinated-subject clauses already give
    their predicate a real SUBJECT edge, so no extra/duplicate clause should
    appear."""
    graph = parser._parse_graph("mary and john are in the garden .")
    clauses, _links = extract_discourse(graph)
    assert len(clauses) == 2


# -- 4. coordinated-subject extraction (M45, untested until now) ----------

def test_coordinated_subject_extraction(parser):
    graph = parser._parse_graph("mary and john are in the garden .")
    clauses, _links = extract_discourse(graph)
    got = [_roles(c) for c in clauses]
    assert any(r.get("SUBJECT") == "mary" and r.get("PLACE") == "garden" for r in got)
    assert any(r.get("SUBJECT") == "john" and r.get("PLACE") == "garden" for r in got)


# -- 5. secondary-sentence fact clauses (M45, untested until now) ---------

def test_secondary_sentence_fact_clause(parser):
    graph = parser._parse_graph("mary went to the garden . she found the ball .")
    clauses, _links = extract_discourse(graph)
    assert _any_match(clauses, {"SUBJECT": "she", "OBJECT": "ball"})


# -- 6. sentence splitter -----------------------------------------------

def test_split_sentences_single_sentence_is_one_segment():
    sent = "mary is in the garden ."
    assert _split_sentences(sent) == [sent]


def test_split_sentences_splits_on_terminal_punctuation():
    assert _split_sentences("mary went to the garden . she found the ball .") == [
        "mary went to the garden .",
        "she found the ball .",
    ]


def test_single_sentence_parse_graph_path_is_byte_identical(parser):
    """`_parse_graph` on a single sentence must take the exact same path as
    `_parse_graph_one` (no merge machinery involved) -- the M45 contract."""
    sent = "mary is in the garden ."
    direct = parser._parse_graph_one(sent)
    via_split = parser._parse_graph(sent)
    assert via_split.nodes == direct.nodes
    assert via_split.edges == direct.edges
    assert via_split.roots == direct.roots


# -- 7. graph merge: index offsetting --------------------------------------

def test_merge_graphs_offsets_node_edge_and_root_indices():
    g1 = HypGraph(nodes=[(0, "NOMINAL", "mary"), (1, "CLAUSE", "went")],
                  edges=[("SUBJECT", 1, 0)], roots=[1])
    g2 = HypGraph(nodes=[(0, "NOMINAL", "she"), (1, "CLAUSE", "found"), (2, "NOMINAL", "ball")],
                  edges=[("SUBJECT", 1, 0), ("OBJECT", 1, 2)], roots=[1])
    merged = _merge_graphs([g1, g2])
    assert merged.nodes == [
        (0, "NOMINAL", "mary"), (1, "CLAUSE", "went"),
        (2, "NOMINAL", "she"), (3, "CLAUSE", "found"), (4, "NOMINAL", "ball"),
    ]
    assert merged.edges == [
        ("SUBJECT", 1, 0),
        ("SUBJECT", 3, 2), ("OBJECT", 3, 4),
    ]
    assert merged.roots == [1, 3]


def test_merge_graphs_single_graph_is_unchanged():
    g1 = HypGraph(nodes=[(0, "NOMINAL", "mary")], edges=[], roots=[0])
    merged = _merge_graphs([g1])
    assert merged.nodes == g1.nodes
    assert merged.edges == g1.edges
    assert merged.roots == g1.roots
