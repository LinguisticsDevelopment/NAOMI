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


# ----------------------------------------- verification-mode batch (data-free)
def test_build_pw_batch_and_controller_forward():
    """Step-2 perception: facts+rules+query -> a 3-way {true,false,idk} MC batch
    the controller can score (verification = MC over the three answer atoms)."""
    from nsm_ct.tpr import TPRCodec
    from nsm_ct.mind.controller import MindController
    from nsm_ct.reasoning_oracle import Rule

    codec = TPRCodec(dim=32)
    facts = [("alice", "is", "furry", "+")]
    rules = [Rule(antecedents=(("?x", "is", "furry", "+"),),
                  consequent=("?x", "is", "kind", "+"), name="pw")]
    ex = pw.PWExample(facts=facts, rules=rules, questions=[
        (("alice", "is", "kind", "+"), pw.TRUE, 1),
        (("alice", "is", "green", "+"), pw.UNKNOWN, 0),
    ])
    items = pw.flatten([ex])
    batch = pw.build_pw_batch(items, codec)
    assert batch.options.shape == (2, 3, 32)
    assert batch.answer.tolist() == [0, 2]            # true -> 0, Unknown -> 2
    out = MindController(codec, hidden=32, hops=4, halting=False)(batch)
    assert out["answer_logits"].shape == (2, 3)       # 3-way verification readout


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
