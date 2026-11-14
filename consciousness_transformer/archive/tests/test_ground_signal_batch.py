"""Tests for new wordnet wrappers and signal modules (M26 batch)."""

from __future__ import annotations

import pytest

from nsm_ct.ground.relations import RelationGraph
from nsm_ct.ground.signal_also_see import extras as also_see_extras
from nsm_ct.ground.signal_cause import extras as cause_extras
from nsm_ct.ground.signal_domain import extras as domain_extras
from nsm_ct.ground.signal_entailment import extras as entailment_extras
from nsm_ct.ground.signal_pertainym import extras as pertainym_extras
from nsm_ct.ground.signal_verbgroup import extras as verbgroup_extras
from nsm_ct.wordnet import (
    also_sees,
    causes,
    domains,
    entailments,
    pertainyms,
    wordnet_available,
)

wn_required = pytest.mark.skipif(not wordnet_available(), reason="WordNet unavailable")

_SMALL_VOCAB = ["dental", "snore", "show", "hot", "baseball", "move"]


class TestWordNetWrappers:
    """Test the new wordnet wrapper functions."""

    @wn_required
    def test_pertainyms_returns_list(self):
        """pertainyms returns a list."""
        result = pertainyms("daily")
        assert isinstance(result, list)

    @wn_required
    def test_pertainyms_never_contains_word(self):
        """pertainyms result never contains the query word."""
        result = pertainyms("daily")
        assert "daily" not in result

    @wn_required
    def test_pertainyms_deterministic(self):
        """Two calls to pertainyms are equal."""
        a = pertainyms("daily")
        b = pertainyms("daily")
        assert a == b

    @wn_required
    def test_entailments_returns_list(self):
        """entailments returns a list."""
        result = entailments("move")
        assert isinstance(result, list)

    @wn_required
    def test_entailments_never_contains_word(self):
        """entailments result never contains the query word."""
        result = entailments("move")
        assert "move" not in result

    @wn_required
    def test_entailments_deterministic(self):
        """Two calls to entailments are equal."""
        a = entailments("move")
        b = entailments("move")
        assert a == b

    @wn_required
    def test_causes_returns_list(self):
        """causes returns a list."""
        result = causes("move")
        assert isinstance(result, list)

    @wn_required
    def test_causes_never_contains_word(self):
        """causes result never contains the query word."""
        result = causes("move")
        assert "move" not in result

    @wn_required
    def test_causes_deterministic(self):
        """Two calls to causes are equal."""
        a = causes("move")
        b = causes("move")
        assert a == b

    @wn_required
    def test_also_sees_returns_list(self):
        """also_sees returns a list."""
        result = also_sees("hot")
        assert isinstance(result, list)

    @wn_required
    def test_also_sees_never_contains_word(self):
        """also_sees result never contains the query word."""
        result = also_sees("hot")
        assert "hot" not in result

    @wn_required
    def test_also_sees_deterministic(self):
        """Two calls to also_sees are equal."""
        a = also_sees("hot")
        b = also_sees("hot")
        assert a == b

    @wn_required
    def test_domains_returns_list(self):
        """domains returns a list."""
        result = domains("baseball")
        assert isinstance(result, list)

    @wn_required
    def test_domains_never_contains_word(self):
        """domains result never contains the query word."""
        result = domains("baseball")
        assert "baseball" not in result

    @wn_required
    def test_domains_deterministic(self):
        """Two calls to domains are equal."""
        a = domains("baseball")
        b = domains("baseball")
        assert a == b


class TestRelationGraphNewFields:
    """Test that RelationGraph.build populates new fields."""

    @wn_required
    def test_relation_graph_builds_new_fields(self):
        """RelationGraph.build populates all new relation fields."""
        g = RelationGraph.build(_SMALL_VOCAB)
        # All new fields should exist
        assert hasattr(g, "pertainym")
        assert hasattr(g, "entailment")
        assert hasattr(g, "cause")
        assert hasattr(g, "also_see")
        assert hasattr(g, "domain")
        # They should be dicts
        assert isinstance(g.pertainym, dict)
        assert isinstance(g.entailment, dict)
        assert isinstance(g.cause, dict)
        assert isinstance(g.also_see, dict)
        assert isinstance(g.domain, dict)

    @wn_required
    def test_relation_graph_new_fields_have_vocab(self):
        """All words are keys in the new relation fields."""
        g = RelationGraph.build(_SMALL_VOCAB)
        for w in _SMALL_VOCAB:
            assert w in g.pertainym
            assert w in g.entailment
            assert w in g.cause
            assert w in g.also_see
            assert w in g.domain

    @wn_required
    def test_relation_graph_new_fields_values_are_lists(self):
        """All values in new relation fields are lists."""
        g = RelationGraph.build(_SMALL_VOCAB)
        for w in _SMALL_VOCAB:
            assert isinstance(g.pertainym[w], list)
            assert isinstance(g.entailment[w], list)
            assert isinstance(g.cause[w], list)
            assert isinstance(g.also_see[w], list)
            assert isinstance(g.domain[w], list)


class TestSignalModules:
    """Test that signal modules return only in-vocab pairs."""

    @wn_required
    def test_pertainym_signal_in_vocab(self):
        """Pertainym signal returns only in-vocab pairs."""
        g = RelationGraph.build(_SMALL_VOCAB)
        result = pertainym_extras(_SMALL_VOCAB, g)
        assert "close_extra" in result
        vocab_set = set(g.gloss)
        for w, x in result["close_extra"]:
            assert w in vocab_set
            assert x in vocab_set
            assert w != x

    @wn_required
    def test_entailment_signal_in_vocab(self):
        """Entailment signal returns only in-vocab pairs."""
        g = RelationGraph.build(_SMALL_VOCAB)
        result = entailment_extras(_SMALL_VOCAB, g)
        assert "close_extra" in result
        vocab_set = set(g.gloss)
        for w, x in result["close_extra"]:
            assert w in vocab_set
            assert x in vocab_set
            assert w != x

    @wn_required
    def test_cause_signal_in_vocab(self):
        """Cause signal returns only in-vocab pairs."""
        g = RelationGraph.build(_SMALL_VOCAB)
        result = cause_extras(_SMALL_VOCAB, g)
        assert "close_extra" in result
        vocab_set = set(g.gloss)
        for w, x in result["close_extra"]:
            assert w in vocab_set
            assert x in vocab_set
            assert w != x

    @wn_required
    def test_also_see_signal_in_vocab(self):
        """Also-see signal returns only in-vocab pairs."""
        g = RelationGraph.build(_SMALL_VOCAB)
        result = also_see_extras(_SMALL_VOCAB, g)
        assert "close_extra" in result
        vocab_set = set(g.gloss)
        for w, x in result["close_extra"]:
            assert w in vocab_set
            assert x in vocab_set
            assert w != x

    @wn_required
    def test_domain_signal_in_vocab(self):
        """Domain signal returns only in-vocab words."""
        g = RelationGraph.build(_SMALL_VOCAB)
        result = domain_extras(_SMALL_VOCAB, g)
        assert "feature_extra" in result
        vocab_set = set(g.gloss)
        for w in result["feature_extra"]:
            assert w in vocab_set

    @wn_required
    def test_verbgroup_signal_in_vocab(self):
        """Verb-group signal returns only in-vocab pairs."""
        g = RelationGraph.build(_SMALL_VOCAB)
        result = verbgroup_extras(_SMALL_VOCAB, g)
        assert "close_extra" in result
        vocab_set = set(g.gloss)
        for w, x in result["close_extra"]:
            assert w in vocab_set
            assert x in vocab_set
            assert w != x
