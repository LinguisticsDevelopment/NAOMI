"""M8 gates: ProofWriter ingestion + the substrate reproduces its gold logic.

Data-free unit gates (parser + verify on hand-built theories) always run; the
real-data parity gate skips if the (git-ignored, downloadable) data is absent.
"""

from __future__ import annotations

import os

import pytest

from nsm_ct.mind.datasets import proofwriter as pw


# ------------------------------------------------------------- parser (data-free)
def test_parse_literal_and_variable_mapping():
    assert pw.parse_literal('("Gary" "is" "kind" "+")') == ("gary", "is", "kind", "+")
    assert pw.parse_literal('("cow" "needs" "bear" "-")') == ("cow", "needs", "bear", "-")
    # ProofWriter's universal variable ("someone"/"they") maps to the oracle's ?x.
    assert pw.parse_literal('("someone" "is" "furry" "+")') == ("?x", "is", "furry", "+")


def test_parse_rule():
    r = pw.parse_rule('((("someone" "is" "furry" "+")) -> ("someone" "is" "kind" "+"))')
    assert r.antecedents == (("?x", "is", "furry", "+"),)
    assert r.consequent == ("?x", "is", "kind", "+")
    r2 = pw.parse_rule('((("Anne" "is" "cold" "+") ("Anne" "is" "big" "+")) -> ("Anne" "is" "red" "+"))')
    assert len(r2.antecedents) == 2 and r2.consequent == ("anne", "is", "red", "+")


# ------------------------------------------------------------- verify (data-free)
def test_verify_true_false_unknown():
    facts = [("alice", "is", "furry", "+")]
    rules = [pw.parse_rule('((("someone" "is" "furry" "+")) -> ("someone" "is" "kind" "+"))')]
    assert pw.verify(facts, rules, ("alice", "is", "kind", "+")) == pw.TRUE       # derivable
    assert pw.verify(facts, rules, ("alice", "is", "kind", "-")) == pw.FALSE      # opposite derivable
    assert pw.verify(facts, rules, ("alice", "is", "green", "+")) == pw.UNKNOWN   # abstain


def test_verify_negative_query_against_positive_fact():
    facts = [("cow", "needs", "bear", "+")]
    # "cow does NOT need bear" is FALSE because the positive fact is present.
    assert pw.verify(facts, [], ("cow", "needs", "bear", "-")) == pw.FALSE


def test_multihop_chain():
    facts = [("bob", "is", "furry", "+")]
    rules = [
        pw.parse_rule('((("someone" "is" "furry" "+")) -> ("someone" "is" "kind" "+"))'),
        pw.parse_rule('((("someone" "is" "kind" "+")) -> ("someone" "is" "smart" "+"))'),
    ]
    assert pw.verify(facts, rules, ("bob", "is", "smart", "+")) == pw.TRUE        # 2-hop


# ------------------------------------------------------------- real-data parity
def _data_present() -> bool:
    return os.path.exists(os.path.join(pw.default_data_dir(), "owa-depth1-test.jsonl"))


@pytest.mark.skipif(not _data_present(), reason="ProofWriter data absent (run scripts/fetch_proofwriter.py)")
def test_proofwriter_gold_parity():
    """Forward-chain reproduces ProofWriter's gold True/False/Unknown labels."""
    ok = total = 0
    for depth in ("0", "1", "2", "3", "5"):
        path = os.path.join(pw.default_data_dir(), f"owa-depth{depth}-test.jsonl")
        for rec in pw.load_records(path, limit=40):
            ex = pw.parse_record(rec)
            for (lit, gold, _qd) in ex.questions:
                ok += pw.verify(ex.facts, ex.rules, lit) == gold
                total += 1
    assert total > 500
    assert ok / total >= 0.95, ok / total   # measured ~0.99 across depths 0-5


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
