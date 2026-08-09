"""Regression tests for the M50 adapter changes: node-flags passthrough,
the agentive-"by" guard, QUESTION reachability, and the 2-arg extraction
ceiling lift.

Three tickets, one underlying change: quantum_parser nodes carry ``flags``
(a ``List[SubType]`` -- PASSIVE stamped by ``aux1`` since M38, QUESTION by
``question1`` since M49), but ``hypothesis_to_graph``/``hypothesis_to_tree``
used to drop them (``HypGraph`` nodes were bare ``(idx, label, token)``,
``ParseNode`` had no flags field). This file pins:

(a) the flags passthrough itself, on both adapter views, with backward-
    compatible constructors (no ``flags=`` kwarg still works);
(b) the agentive-"by" guard in ``nsm_ct.clause`` that the passthrough
    unblocks (a "by" PP on a PASSIVE predicate resolves to AGENT, not PLACE
    -- the module-level landmine comment in clause.py, TODO since M38);
(c) ``Clause.is_question`` becoming reachable off the QUESTION flag;
(d) the M48/M50 fix that lifts the 2-arg extraction ceiling (every PP/
    OBJECT/INDIRECT_OBJECT is now emitted, not just the first one), including
    that the primary/first role's identity is unchanged.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause import extract_discourse  # noqa: E402
from nsm_ct.data_structures import ParseNode  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.quantum_adapter import HypGraph  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402

_SENTENCES = [
    "the ball was found by mary .",
    "mary is by the garden .",
    "mary gave the ball to john in the garden .",
    "mary found the ball .",
    "where is the ball ?",
    "is the ball in the garden ?",
    "mary is in the garden .",
]


@pytest.fixture(scope="module")
def parser():
    tok = SimpleTokenizer.build(_SENTENCES, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    p = ParserInputEncoder(tok)
    if getattr(p, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return p


def _args(cl):
    return [(rel, arg.token) for rel, arg in cl.args]


def _any_clause(clauses, pred=None):
    for c in clauses:
        if pred is None or (c.predicate or "").lower() == pred:
            return c
    return None


# -- (a) flags passthrough + backward-compatible constructors --------------

def test_hypgraph_constructor_without_flags_kwarg_still_works():
    """M50 added HypGraph.flags; every pre-M50 call site (tests included)
    constructs HypGraph without it -- must keep defaulting to {}."""
    g = HypGraph(nodes=[(0, "NOMINAL", "mary")], edges=[], roots=[0])
    assert g.flags == {}
    assert g.flags_of(0) == []


def test_parsenode_constructor_without_flags_kwarg_still_works():
    n = ParseNode(label="CONTENT", token="cat")
    assert n.flags == []


def test_passive_flag_reaches_the_hypgraph(parser):
    graph = parser._parse_graph("the ball was found by mary .")
    clause_idx = next(idx for idx, label, tok in graph.nodes if tok == "found")
    assert "PASSIVE" in graph.flags_of(clause_idx)


def test_passive_flag_reaches_the_parse_tree(parser):
    tree = parser._parse_tree("the ball was found by mary .")
    found = next(n for n in tree.iter_preorder() if n.token == "found")
    assert "PASSIVE" in found.flags


def test_question_flag_reaches_the_hypgraph(parser):
    graph = parser._parse_graph("where is the ball ?")
    clause_idx = next(idx for idx, label, tok in graph.nodes if tok == "is")
    assert "QUESTION" in graph.flags_of(clause_idx)


def test_non_passive_predicate_has_no_passive_flag(parser):
    """Regression guard for the agentive-'by' guard's specificity: an
    ordinary copula must NOT be mistaken for a passive predicate."""
    graph = parser._parse_graph("mary is by the garden .")
    clause_idx = next(idx for idx, label, tok in graph.nodes if tok == "is")
    assert "PASSIVE" not in graph.flags_of(clause_idx)


# -- (b) the agentive-"by" guard --------------------------------------------

def test_agentive_by_on_passive_clause_yields_agent(parser):
    graph = parser._parse_graph("the ball was found by mary .")
    clauses, _links = extract_discourse(graph)
    cl = _any_clause(clauses, "found")
    assert cl is not None
    args = dict(_args(cl))
    assert args.get("SUBJECT") == "ball"
    assert args.get("AGENT") == "mary"
    assert "PLACE" not in args  # must not ALSO carry the old mislabel


def test_by_on_non_passive_clause_still_means_place(parser):
    """'mary is by the garden .' -- a pure locative 'by', no passive voice
    anywhere: must resolve to PLACE exactly as before M50."""
    graph = parser._parse_graph("mary is by the garden .")
    clauses, _links = extract_discourse(graph)
    cl = _any_clause(clauses, "is")
    assert cl is not None
    args = dict(_args(cl))
    assert args.get("SUBJECT") == "mary"
    assert args.get("PLACE") == "garden"
    assert "AGENT" not in args


def test_battery_passive_by_case_still_has_subject_ball(parser):
    """The hand battery's expectation (probe_parser_stress.py, 'passive'
    category) is a subset check {"SUBJECT": "ball"} -- adding AGENT must not
    disturb it."""
    graph = parser._parse_graph("the ball was found by mary .")
    clauses, _links = extract_discourse(graph)
    assert any(dict(_args(c)).get("SUBJECT") == "ball" for c in clauses)


# -- (c) QUESTION reachability -----------------------------------------------

def test_question_marker_reachable_on_wh_question(parser):
    graph = parser._parse_graph("where is the ball ?")
    clauses, _links = extract_discourse(graph)
    cl = _any_clause(clauses, "is")
    assert cl is not None and cl.is_question is True


def test_question_marker_reachable_on_yes_no_question(parser):
    graph = parser._parse_graph("is the ball in the garden ?")
    clauses, _links = extract_discourse(graph)
    cl = _any_clause(clauses, "is")
    assert cl is not None and cl.is_question is True


def test_question_marker_false_on_declarative(parser):
    graph = parser._parse_graph("mary is in the garden .")
    clauses, _links = extract_discourse(graph)
    cl = _any_clause(clauses, "is")
    assert cl is not None and cl.is_question is False


# -- (d) multi-arg extraction (2-arg ceiling lift) --------------------------

def test_two_pps_both_survive_extraction(parser):
    """The headline fix: 'mary gave the ball to john in the garden .' has
    TWO PPs ('to john', 'in the garden') plus a direct OBJECT ('ball') --
    all three must now be in the clause, not just the first PP."""
    graph = parser._parse_graph("mary gave the ball to john in the garden .")
    clauses, _links = extract_discourse(graph)
    cl = _any_clause(clauses, "gave")
    assert cl is not None
    tokens = {(rel, tok) for rel, tok in _args(cl)}
    assert ("SUBJECT", "mary") in tokens
    assert ("OBJECT", "ball") in tokens
    assert ("PLACE", "john") in tokens
    assert ("PLACE", "garden") in tokens
    assert len(cl.args) == 4


def test_primary_role_ordering_preserved(parser):
    """The primary (first non-SUBJECT) role must stay exactly what it was
    before the multi-arg fix -- args[0] is SUBJECT, args[1] is the first PP
    ('to john'), and the second PP is appended AFTER it, never reordered."""
    graph = parser._parse_graph("mary gave the ball to john in the garden .")
    clauses, _links = extract_discourse(graph)
    cl = _any_clause(clauses, "gave")
    assert cl is not None
    assert cl.args[0][0] == "SUBJECT" and cl.args[0][1].token == "mary"
    assert cl.args[1][0] == "PLACE" and cl.args[1][1].token == "john"


def test_bare_transitive_object_no_longer_dropped(parser):
    """'mary found the ball .' has no PP at all -- previously the primary
    single-clause path only recognized PP/locative-verb objects, so this
    degraded to a SUBJECT-only stump. M50 recovers the OBJECT."""
    graph = parser._parse_graph("mary found the ball .")
    clauses, _links = extract_discourse(graph)
    cl = _any_clause(clauses, "found")
    assert cl is not None
    args = dict(_args(cl))
    assert args.get("SUBJECT") == "mary"
    assert args.get("OBJECT") == "ball"
