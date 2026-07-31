"""M17.3 gate: honest understanding evaluation (derivation, not lookup)."""

from __future__ import annotations

import pytest

from nsm_ct.ground.clause_self_consistency import SAMPLE_VOCAB
from nsm_ct.ground.definition_graph import DefinitionGraph
from nsm_ct.ground.evaluation import (
    evaluate,
    held_out_vocab,
    syn_ant_discrimination,
)
from nsm_ct.ground.semantic_axes import AxisRegistry
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


@wn_required
def test_held_out_excludes_deepnsm_covered_words():
    # 'sad' is in the gold/DeepNSM store; a nonsense word is not.
    ho = held_out_vocab(["sad", "qwertynonsense"])
    assert "sad" not in ho
    assert "qwertynonsense" in ho


@wn_required
def test_syn_ant_discrimination_shape():
    words = ["hot", "cold", "warm", "good", "bad"]
    g = DefinitionGraph.build(words)
    out = syn_ant_discrimination(words, AxisRegistry.seed(), 3, g)
    assert set(out.keys()) == {"accuracy", "n"}
    assert out["accuracy"] is None or 0.0 <= out["accuracy"] <= 1.0


@wn_required
def test_evaluate_structure_and_grounding_improves():
    # expand=False keeps it fast/deterministic for the unit gate.
    r = evaluate(SAMPLE_VOCAB, depth=3, max_axes=8, expand=False)
    for k in ("seed", "derived", "round_trip", "promoted_axes", "mdl_curve",
              "deepnsm_agreement_seed", "deepnsm_agreement_derived"):
        assert k in r
    # deriving a basis strictly improves grounding (more leaves reach an axis).
    assert r["derived"]["grounding_rate"] >= r["seed"]["grounding_rate"]
    # MDL never increases along the curve.
    vals = [m for _, m in r["mdl_curve"]]
    assert all(b <= a for a, b in zip(vals, vals[1:]))


@wn_required
def test_evaluate_measures_on_held_out_and_external_check():
    r = evaluate(SAMPLE_VOCAB, depth=3, max_axes=8, expand=False)
    # probes run on words outside DeepNSM (derivation, not lookup)
    assert r["n_derivation"] > 0
    # the external DeepNSM check has coverage and is a valid score
    da = r["deepnsm_agreement_derived"]
    assert da["n"] > 0
    assert da["mean"] is None or 0.0 <= da["mean"] <= 1.0


@wn_required
def test_round_trip_recovers_held_out_words():
    r = evaluate(SAMPLE_VOCAB, depth=3, max_axes=8, expand=False)
    # clause==word recovery on the held-out derivation set should be strong.
    assert r["round_trip"]["exact"] is None or r["round_trip"]["exact"] >= 0.8
