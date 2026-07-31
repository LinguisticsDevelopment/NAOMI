"""M17.1 gate: the deterministic clause==word reduction operator."""

from __future__ import annotations

import pytest

from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.ground.canonicalization import tree_key
from nsm_ct.ground.clause_self_consistency import SAMPLE_VOCAB
from nsm_ct.ground.definition_graph import definition_clause
from nsm_ct.ground.reduction import (
    ReducedDefinitionIndex,
    lexicalize,
    perturbed_clause,
    reduce,
    round_trip,
)
from nsm_ct.wordnet import wordnet_available

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")


def _clause(*words):
    head = ParseNode(label="EXPLICATION")
    head.children = [ParseNode(label="WORD", token=w) for w in words]
    return ParseTree(root=head)


@pytest.fixture(scope="module")
def index():
    return ReducedDefinitionIndex.build(SAMPLE_VOCAB, depth=3)


@wn_required
def test_reduce_is_deterministic():
    c = _clause("dog", "animal", "feeling")
    assert tree_key(reduce(c, depth=3).tree) == tree_key(reduce(c, depth=3).tree)


@wn_required
def test_reduce_is_idempotent_and_terminates():
    c = definition_clause("sad")
    r = reduce(c, depth=3)
    # Re-reducing the fixpoint yields the same canonical key (terminated).
    assert tree_key(reduce(r.tree, depth=3).tree) == tree_key(r.tree)


@wn_required
def test_reduce_is_confluent_under_reordering_and_duplication():
    a = _clause("good", "bad", "feel")
    b = _clause("feel", "bad", "good", "good")  # reordered + duplicate
    assert tree_key(reduce(a, depth=3).tree) == tree_key(reduce(b, depth=3).tree)


@wn_required
def test_reduce_grounds_toward_primes():
    # A clause of prime-exponent words reduces to exactly those axes.
    mv = reduce(_clause("good", "bad"), depth=3)
    assert mv.active_axes() == {"GOOD", "BAD"}


@wn_required
def test_exact_round_trip_recovers_most_words(index):
    hits = sum(1 for w in SAMPLE_VOCAB if round_trip(w, index)[0] == w)
    # Misses are reduced-definition key collisions (first-writer-wins) — honest.
    assert hits / len(SAMPLE_VOCAB) >= 0.8


@wn_required
def test_perturbed_clause_recovers_via_closeness(index):
    total = hits = 0
    for w in SAMPLE_VOCAB:
        pc = perturbed_clause(w)
        if pc is None:
            continue
        total += 1
        if lexicalize(pc, index)[0] == w:
            hits += 1
    # Dropping a content word should still recover the word via coordinate
    # closeness for the majority — the generalization signal.
    assert total > 0 and hits / total >= 0.6


@wn_required
def test_lexicalize_return_shape(index):
    word, how, score = lexicalize(definition_clause("dog"), index)
    assert how in {"exact", "closest", "none"}
    assert 0.0 <= score <= 1.0


@wn_required
def test_unknown_clause_does_not_crash(index):
    # A clause of nonsense words grounds to nothing; lexicalize abstains cleanly.
    word, how, score = lexicalize(_clause("zzqq", "wxyz"), index)
    assert word is None or how in {"closest", "none"}
