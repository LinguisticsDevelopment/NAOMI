"""Regression tests for M43's question-extraction fix.

quantum_parser's ``question1`` ruleset (grammars/english.json) gives a
subject-initial copula -- subject-aux inversion ("is the ball in the garden
?") or wh-fronting ("where is the ball ?") -- a SUBJECT edge before
``predicate1``'s generic transitive-object rule or ``noun3``'s NP-internal
PP attachment can misparse the postposed subject. These tests pin the
extraction-layer contract (probe_parser_stress.py's "question" category,
2/2) at the unit level.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause import extract_discourse  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402

_SENTENCES = [
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


def _roles(cl) -> dict:
    out = {"PRED": (cl.predicate or "").lower()}
    for rel, arg in cl.args:
        out.setdefault(rel, (arg.token or "").lower())
    return out


def _any_match(clauses, expect: dict) -> bool:
    return any(all(_roles(c).get(k) == v for k, v in expect.items()) for c in clauses)


def test_wh_question_extracts_subject(parser):
    graph = parser._parse_graph("where is the ball ?")
    clauses, _links = extract_discourse(graph)
    assert _any_match(clauses, {"SUBJECT": "ball"})


def test_yes_no_question_extracts_subject_and_place(parser):
    graph = parser._parse_graph("is the ball in the garden ?")
    clauses, _links = extract_discourse(graph)
    assert _any_match(clauses, {"SUBJECT": "ball", "PLACE": "garden"})


def test_declarative_copula_still_extracts_normally(parser):
    """Regression guard: the question fix must not touch an ordinary
    mid-sentence copula's SUBJECT/PLACE extraction."""
    graph = parser._parse_graph("mary is in the garden .")
    clauses, _links = extract_discourse(graph)
    assert _any_match(clauses, {"SUBJECT": "mary", "PLACE": "garden"})
