"""M17.0 gate: the baseline understanding harness + relational accessors."""

from __future__ import annotations

import pytest

from nsm_ct.nsm_primes import NUM_PRIMES
from nsm_ct.ground import clause_self_consistency as csc
from nsm_ct.ground.canonicalization import tree_key
from nsm_ct.ground.definition_graph import (
    DefinitionGraph,
    definition_clause,
    naive_decompose,
)
from nsm_ct.ground.meaning_value import MeaningValue
from nsm_ct.wordnet import antonyms, hypernyms, synonyms, wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


@wn_required
def test_antonyms_are_reachable():
    assert "cold" in antonyms("hot")
    assert synonyms("hot")  # non-empty list of lemmas
    assert hypernyms("dog")  # has hypernyms


@wn_required
def test_naive_decompose_is_deterministic():
    a = naive_decompose("dog", max_depth=4)
    b = naive_decompose("dog", max_depth=4)
    assert tree_key(a) == tree_key(b)


@wn_required
def test_prime_words_decompose_to_their_axis():
    # "good" is the exponent of the GOOD prime -> grounds directly.
    mv = MeaningValue.from_tree(naive_decompose("good", max_depth=4))
    assert "GOOD" in mv.active_axes()


def test_meaning_value_coordinate_shape():
    mv = MeaningValue.from_tree(naive_decompose("good", max_depth=2))
    assert mv.axis_vec.shape == (NUM_PRIMES,)
    # coupled: it keeps the structured object too
    assert mv.tree is not None


@wn_required
def test_metrics_are_in_range():
    for w in ["dog", "sad", "good", "hot", "justice"]:
        conv = csc.convergence(w, depth=3)
        assert 0.0 <= conv <= 1.0
        ground = csc.prime_grounding(w, depth=3)
        assert ground is None or 0.0 <= ground <= 1.0


@wn_required
def test_deepnsm_agreement_is_an_external_check_in_range():
    # 'sad' is covered by the gold/DeepNSM store; agreement must be a valid score.
    agree = csc.deepnsm_agreement("sad", depth=3)
    assert agree is None or 0.0 <= agree <= 1.0


@wn_required
def test_report_runs_and_aggregates():
    r = csc.report(words=["dog", "sad", "good", "hot"], depth=3)
    assert r["n_words"] == 4
    assert 0.0 <= r["mean_convergence"] <= 1.0
    assert 0.0 <= r["mean_prime_grounding"] <= 1.0
    assert set(["convergence", "prime_grounding", "deepnsm_agreement"]).issubset(
        r["per_word"]["dog"].keys()
    )


@wn_required
def test_closure_hook_accepts_a_reduce_fn():
    # The generic clause==word hook M17.1 will plug into: here a trivial reducer
    # that decomposes each definition word and unions the result.
    def reduce_fn(defn_tree):
        from nsm_ct.data_structures import ParseNode, ParseTree
        head = ParseNode(label="EXPLICATION")
        for leaf in defn_tree.leaves():
            if leaf.token:
                head.children.append(naive_decompose(leaf.token, max_depth=2).root)
        return MeaningValue.from_tree(ParseTree(root=head))

    score = csc.closure(
        "dog",
        value_fn=lambda w: csc.value(w, depth=3),
        reduce_fn=reduce_fn,
    )
    assert score is None or 0.0 <= score <= 1.0


@wn_required
def test_definition_graph_builds_with_relations():
    g = DefinitionGraph.build(["hot", "cold", "dog"])
    assert "hot" in g.words()
    assert g.content["hot"]  # definition content words
    # antonym pair surfaces when both endpoints are in the graph
    pairs = g.antonym_pairs()
    assert ("cold", "hot") in pairs


@wn_required
def test_definition_clause_is_words_not_primes():
    defn = definition_clause("dog")
    assert defn is not None
    # leaves are surface WORD tokens (the clause to be reduced), not primes
    assert all(leaf.label == "WORD" and leaf.token for leaf in defn.leaves())
