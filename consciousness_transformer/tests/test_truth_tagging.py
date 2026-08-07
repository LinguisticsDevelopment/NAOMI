"""Truth-tagging + store-as-OR-then-decide (Phase A).

A clause matrix can carry a TRUE/FALSE/MAYBE adjective (non-destructively); a
disjunction is stored with both disjuncts MAYBE, and evidence (a negation) re-tags
the refuted disjunct FALSE and the survivor TRUE while keeping BOTH recoverable.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause import (  # noqa: E402
    DisjunctionBuffer, build_discourse_tpr, extract_discourse, read_truth,
    tag_truth, truth_book,
)
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


@pytest.fixture(scope="module")
def env():
    eps = CurriculumGenerator(max_level=6, seed=0).generate(12)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    p = ParserInputEncoder(tok)
    if getattr(p, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return p, NSMMeaningResolver(), TPRCodec(dim=256)


def test_truth_tag_roundtrip(env):
    _, resolver, codec = env
    m = codec.encode_matrix(resolver.resolve("kitchen").root)
    for value in truth_book(codec):
        tagged = tag_truth(m, value, codec)
        assert read_truth(tagged, codec)[0] == value


def test_tag_does_not_destroy_the_clause(env):
    parser, resolver, codec = env
    clauses, links = extract_discourse(parser._parse_graph("mary is in the kitchen ."))
    base = build_discourse_tpr(clauses, links, codec, resolver).clauses[0]
    tagged = tag_truth(base, "FALSE", codec)
    rel, arg = clauses[0].args[1]
    # the place is still recoverable from the FALSE-tagged matrix (overwrite, don't forget)
    pv = codec.contract(codec.encode_matrix(resolver.resolve(arg.token).root))
    u = codec.unbind(tagged, codec.role_vec(1, rel))
    assert float(u @ pv) > float(u @ codec.filler_vec("var:mary"))


def test_store_or_then_decide_by_negation(env):
    parser, resolver, codec = env
    clauses, links = extract_discourse(parser._parse_graph("mary is in the kitchen or the office ."))
    dtpr = build_discourse_tpr(clauses, links, codec, resolver)

    buf = DisjunctionBuffer(codec)
    buf.store_disjunction(dtpr)
    assert [e["truth"] for e in buf.group] == ["MAYBE", "MAYBE"]
    assert buf.query()[0] == "MAYBE"          # unresolved -> first-class MAYBE

    # evidence: "mary is not in the kitchen ." refutes the kitchen disjunct
    nclauses, _ = extract_discourse(parser._parse_graph("mary is not in the kitchen ."))
    refuted = codec.contract(codec.encode_matrix(resolver.resolve(nclauses[0].args[1][1].token).root))
    buf.decide_truth(refuted)

    truths = {e["truth"] for e in buf.group}
    assert truths == {"TRUE", "FALSE"}
    state, value = buf.query()
    assert state == "RESOLVED"
    # the resolved value is the office (the surviving disjunct)
    places = ["kitchen", "garden", "office", "bedroom", "hallway", "bathroom"]
    pv = {p: codec.contract(codec.encode_matrix(resolver.resolve(p).root)) for p in places}
    nearest = max(places, key=lambda p: float(value @ pv[p] / (np.linalg.norm(value) * np.linalg.norm(pv[p]) + 1e-8)))
    assert nearest == "office"
