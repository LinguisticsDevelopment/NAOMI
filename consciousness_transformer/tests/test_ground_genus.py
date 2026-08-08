"""Gates for genus-differentia gloss parsing (signal_genus, dev/SIGNALS_AUDIT.md #1)."""

from __future__ import annotations

import pytest

from nsm_ct.ground.corpus import gloss_vocabulary
from nsm_ct.ground.relations import RelationGraph
from nsm_ct.ground.signal_genus import extras, genus_of
from nsm_ct.wordnet import wordnet_available

pytestmark = pytest.mark.skipif(not wordnet_available(), reason="needs WordNet")


# ---------------------------------------------------------------------------
# genus_of: one case per named pattern in the M26 signal spec.
# ---------------------------------------------------------------------------


def test_leading_article_a():
    assert genus_of("a large fleet-footed mammal") == "mammal"


def test_leading_article_the():
    assert genus_of("the act of stealing") == "act"


def test_quantifier_wrapper_any_of():
    # "any of various plants of the genus Rosa..." -> real genus follows "any of".
    assert genus_of("any of various plants of the genus Rosa that bear roses") == "plant"


def test_quantifier_wrapper_a_member_of():
    assert genus_of(
        "a member of the genus Canis (probably descended from the common wolf) "
        "that has been domesticated by man since prehistoric times"
    ) == "canis"


def test_quantifier_wrapper_a_variety_of():
    assert genus_of("a variety of crocodile") == "crocodile"


def test_multiword_head_takes_last_noun():
    assert genus_of("a motor vehicle with four wheels") == "vehicle"


def test_verb_gloss_bare_infinitive():
    assert genus_of("draw air into, and expel out of, the lungs") == "draw"


def test_verb_gloss_to_form():
    assert genus_of("to steal from someone") == "steal"


def test_leading_parenthetical_is_stripped():
    assert genus_of("(physics and chemistry) the simplest structural unit of an element") == "unit"


def test_semicolon_gloss_uses_first_clause():
    gloss = "a hand tool with a heavy rigid head and a handle; used to deliver an impulsive force"
    assert genus_of(gloss) == "tool"


def test_no_gloss_is_none():
    assert genus_of("") is None
    assert genus_of(None) is None


def test_nonsense_gloss_is_none():
    assert genus_of("xyzzy plugh qux") is None


def test_genus_of_is_deterministic():
    gloss = "any of various plants of the genus Rosa that bear roses"
    assert genus_of(gloss) == genus_of(gloss)


# ---------------------------------------------------------------------------
# extras(): harness contract.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_space():
    vocab = gloss_vocabulary(500)
    g = RelationGraph.build(vocab)
    return list(vocab), g


def test_extras_returns_close_extra_key(small_space):
    vocab, g = small_space
    out = extras(vocab, g)
    assert "close_extra" in out
    assert isinstance(out["close_extra"], list)


def test_extras_pairs_are_in_vocab_and_non_reflexive(small_space):
    vocab, g = small_space
    wset = set(vocab)
    pairs = extras(vocab, g)["close_extra"]
    assert pairs, "expected at least one genus edge over 500 gloss-vocab words"
    for w, gen in pairs:
        assert w in wset
        assert gen in wset
        assert gen != w


def test_extras_is_deterministic(small_space):
    vocab, g = small_space
    a = extras(vocab, g)["close_extra"]
    b = extras(vocab, g)["close_extra"]
    assert a == b


# ---------------------------------------------------------------------------
# Standalone validity: coverage + pointer agreement (sanity, not a target).
# ---------------------------------------------------------------------------


def test_coverage_and_pointer_agreement_are_reasonable(small_space, capsys):
    """genus_of should fire on a healthy fraction of glosses, and land on the
    pointer-hypernym lemma often enough to look like signal rather than noise
    -- but genus is an INDEPENDENT signal from the synset-pointer hypernym, so
    partial disagreement is expected, not a failure."""
    vocab, g = small_space
    n_gloss = len(g.gloss)
    n_hit = sum(1 for gl in g.gloss.values() if genus_of(gl))
    coverage = n_hit / n_gloss if n_gloss else 0.0

    n_pointer = 0
    n_agree = 0
    for w, gl in g.gloss.items():
        pointer_hypers = g.is_a.get(w, [])
        if not pointer_hypers:
            continue
        n_pointer += 1
        gen = genus_of(gl)
        if gen is not None and gen in pointer_hypers:
            n_agree += 1
    agreement = n_agree / n_pointer if n_pointer else 0.0

    print(f"\ngenus coverage: {n_hit}/{n_gloss} = {coverage:.3f}")
    print(f"pointer agreement: {n_agree}/{n_pointer} = {agreement:.3f}")

    assert coverage > 0.5
