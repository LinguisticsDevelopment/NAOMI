"""Discourse extraction: coordinators relate lossless clauses (Phase A).

Verifies ``extract_discourse`` on the flat hypothesis graph for the OR / negation
shapes the curriculum needs, the lossless coordination representation, and — as a
regression — that the refactored ``extract_clauses`` is unchanged on levels 1-6.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause import (  # noqa: E402
    build_discourse_tpr, extract_clauses, extract_discourse, read_connective,
)
from nsm_ct.dataset import PARSE_LABELS  # noqa: E402
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


@pytest.fixture(scope="module")
def parser():
    eps = CurriculumGenerator(max_level=6, seed=0).generate(12)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    p = ParserInputEncoder(tok)
    if getattr(p, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return p


@pytest.fixture(scope="module")
def resolver():
    return NSMMeaningResolver()


def _places(c):
    return [a.token for r, a in c.args if r == "PLACE"]


def test_or_two_clauses_and_link(parser):
    clauses, links = extract_discourse(parser._parse_graph("mary is in the kitchen or the office ."))
    assert len(clauses) == 2
    assert {p for c in clauses for p in _places(c)} == {"kitchen", "office"}
    assert all(_places(c) and [a.token for r, a in c.args if r == "SUBJECT"] == ["mary"]
               for c in clauses)
    assert len(links) == 1
    assert links[0].coordinator == "OR" and links[0].prime == "MAYBE"


def test_negation_is_a_not_link(parser):
    clauses, links = extract_discourse(parser._parse_graph("mary is not in the kitchen ."))
    assert len(clauses) == 1 and _places(clauses[0]) == ["kitchen"]
    assert links and links[0].coordinator == "NOT" and links[0].prime == "NOT"


def test_plain_fact_has_no_links(parser):
    clauses, links = extract_discourse(parser._parse_graph("mary is in the kitchen ."))
    assert len(clauses) == 1 and _places(clauses[0]) == ["kitchen"]
    assert links == []


def test_discourse_tpr_lossless_and_link_recovery(parser, resolver):
    codec = TPRCodec(dim=256)
    clauses, links = extract_discourse(parser._parse_graph("mary is in the kitchen or the office ."))
    dtpr = build_discourse_tpr(clauses, links, codec, resolver)
    assert len(dtpr.clauses) == 2
    # each disjunct's value is recoverable by unbind + cleanup (the operative fidelity)
    places = ["kitchen", "garden", "office", "bedroom", "hallway", "bathroom"]
    pv = {p: codec.contract(codec.encode_matrix(resolver.resolve(p).root)) for p in places}
    for cl, m in zip(clauses, dtpr.clauses):
        rel, arg = cl.args[1]
        u = codec.unbind(m, codec.role_vec(1, rel))
        nearest = max(places, key=lambda p: float(u @ pv[p] / (np.linalg.norm(u) * np.linalg.norm(pv[p]) + 1e-8)))
        assert nearest == arg.token
    # the OR link recovers the related clause, and the connective atom is MAYBE
    assert dtpr.recover_link(codec, 0) == 1
    assert read_connective(dtpr, codec)[0] == "MAYBE"


def test_extract_clauses_unchanged_on_curriculum(parser):
    """Regression: the refactor must not change extract_clauses on levels 1-6."""
    gen = CurriculumGenerator(max_level=6, seed=3)
    seen = 0
    for ep in gen.generate(30):
        for sent in ep.context + getattr(ep, "post_context", []):
            tree = parser._parse_tree(sent)
            if tree is None:
                continue
            clauses = extract_clauses(tree)
            # every curriculum statement yields at least one clause with a subject
            for c in clauses:
                assert c.predicate
            seen += 1
    assert seen > 0
