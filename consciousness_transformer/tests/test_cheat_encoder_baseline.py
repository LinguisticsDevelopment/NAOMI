"""Tests for scripts/cheat_encoder_baseline.py (dev/K12_BENCHMARK_FIX.md
item 2) -- 3 tiny hand-built gold records, scored through the SAME
``nsm_ct.encoder_model.score_record``/``aggregate_recall`` the learned
encoder itself is scored with (no parallel metric). Expected precision/
recall/F1 are hand-derived (see the comment on each record) and verified
once by running the real code before being hardcoded here.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from cheat_encoder_baseline import evaluate_heuristic, heuristic_forest  # noqa: E402
from nsm_ct.encoder_model import score_record  # noqa: E402


def _clause(predicate, pred_gtype, roles):
    return {
        "predicate": predicate,
        "predicate_grounding": {"type": pred_gtype},
        "predicate_token_index": None,
        "roles": roles,
    }


def _role(relation, word, token_index, gtype, is_entity=False):
    return {"relation": relation, "word": word, "token_index": token_index,
            "is_entity": is_entity, "grounding": {"type": gtype}}


# record A: "the cat sat on the mat ." -- gold SUBJECT is "entity"-typed but
# the heuristic (no sense candidates for "cat", not PROPN) guesses
# "reference" -- a gtype mismatch, so SUBJECT does NOT count as overlap
# even though relation+token_index agree. gold PLACE ("mat") DOES match
# (both "sense", since idx5 has sense candidates). The heuristic's own
# OBJECT/DESCRIPTION edges are pure over-generation (not in gold at all).
# -> overlap=2 (PREDICATE, PLACE) / heuristic-tree-size=5 edges (+1 predicate
# counted separately in _tree_edge_set, so 6 total) -> precision=2/6=1/3;
# gold-size=3 -> recall=2/3.
RECORD_A = {
    "tokens": ["the", "cat", "sat", "on", "the", "mat", "."],
    "pos": ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"],
    "token_sense_candidates": [
        {"index": 2, "sense_candidates": ["sit.v.01"]},
        {"index": 5, "sense_candidates": ["mat.n.01"]},
    ],
    "lattice": {"trees": [{"clauses": [_clause(
        "sat", "sense",
        [_role("SUBJECT", "cat", 1, "entity", is_entity=True),
         _role("PLACE", "mat", 5, "sense")],
    )]}]},
}

# record B: "dogs run ." -- gold SUBJECT is "entity"-typed, heuristic again
# guesses "reference" -> mismatch. PREDICATE matches. overlap=1/2 edges ->
# precision=0.5, recall=0.5.
RECORD_B = {
    "tokens": ["dogs", "run", "."],
    "pos": ["NOUN", "VERB", "PUNCT"],
    "token_sense_candidates": [{"index": 1, "sense_candidates": ["run.v.01"]}],
    "lattice": {"trees": [{"clauses": [_clause(
        "run", "sense", [_role("SUBJECT", "dogs", 0, "entity", is_entity=True)],
    )]}]},
}

# record C: "birds fly ." -- gold SUBJECT is "reference"-typed, which is
# exactly what the heuristic guesses (no sense candidates, not PROPN) ->
# a PERFECT match: precision=recall=1.0, and this record's tree_hits=1
# (the whole-clause skeleton matches, not just individual edges).
RECORD_C = {
    "tokens": ["birds", "fly", "."],
    "pos": ["NOUN", "VERB", "PUNCT"],
    "token_sense_candidates": [{"index": 1, "sense_candidates": ["fly.v.01"]}],
    "lattice": {"trees": [{"clauses": [_clause(
        "fly", "sense", [_role("SUBJECT", "birds", 0, "reference")],
    )]}]},
}


def test_heuristic_forest_shape():
    forest = heuristic_forest(RECORD_A)
    assert len(forest) == 1
    clause = forest[0]["clauses"][0]
    assert clause["predicate"]["token_index"] == 2  # first VERB
    relations = {r["relation"] for r in clause["roles"]}
    assert {"SUBJECT", "OBJECT", "PLACE", "DESCRIPTION"} <= relations


def test_record_a_partial_overlap():
    forest = heuristic_forest(RECORD_A)
    rr = score_record(RECORD_A, forest)
    assert rr.edge_precision == pytest.approx(1 / 3)
    assert rr.edge_recall == pytest.approx(2 / 3)


def test_record_b_subject_gtype_mismatch():
    forest = heuristic_forest(RECORD_B)
    rr = score_record(RECORD_B, forest)
    assert rr.edge_precision == pytest.approx(0.5)
    assert rr.edge_recall == pytest.approx(0.5)


def test_record_c_perfect_match():
    forest = heuristic_forest(RECORD_C)
    rr = score_record(RECORD_C, forest)
    assert rr.edge_precision == pytest.approx(1.0)
    assert rr.edge_recall == pytest.approx(1.0)
    assert rr.tree_hits == 1


def test_evaluate_heuristic_aggregate():
    agg = evaluate_heuristic([RECORD_A, RECORD_B, RECORD_C])
    expected_p = (1 / 3 + 0.5 + 1.0) / 3
    expected_r = (2 / 3 + 0.5 + 1.0) / 3
    assert agg["edge_precision"] == pytest.approx(expected_p)
    assert agg["edge_recall"] == pytest.approx(expected_r)
    # only record C's whole-clause skeleton matches gold exactly.
    assert agg["structure_recall"] == pytest.approx(1 / 3)
    assert agg["n_records"] == 3
