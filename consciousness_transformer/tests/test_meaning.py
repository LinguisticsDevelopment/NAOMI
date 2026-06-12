"""Tests for NSMMeaningResolver (Stage B meaning grounding).

Every assertion here is grounded in real data:
- Prime / molecule names come from nsm_primes / nsm_molecules registries.
- Person detection uses WordNet lexnames / hypernyms (real data; not mocked).
- meaning_prime_ids confirms the produced trees yield at least one prime id.
"""

import pytest

from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.meaning import NSMMeaningResolver
from nsm_ct.thought import meaning_prime_ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def resolver():
    """A single shared resolver so the cache is exercised across tests."""
    return NSMMeaningResolver()


# ---------------------------------------------------------------------------
# 1. Prime exponent
# ---------------------------------------------------------------------------

def test_resolve_i_returns_prime_I(resolver):
    tree = resolver.resolve("i")
    assert tree.root.label == "I"
    assert tree.root.children == []  # leaf
    ids = meaning_prime_ids(tree)
    assert len(ids) >= 1


def test_resolve_me_returns_prime_I(resolver):
    """'me' is an allolex of the I prime."""
    tree = resolver.resolve("me")
    assert tree.root.label == "I"


def test_resolve_you_returns_prime_YOU(resolver):
    tree = resolver.resolve("you")
    assert tree.root.label == "YOU"


def test_resolve_thing_returns_prime_SOMETHING(resolver):
    """'thing' is an allolex of SOMETHING."""
    tree = resolver.resolve("thing")
    assert tree.root.label == "SOMETHING"


# ---------------------------------------------------------------------------
# 2. Molecule exponent
# ---------------------------------------------------------------------------

def test_resolve_water_returns_molecule_WATER(resolver):
    tree = resolver.resolve("water")
    assert tree.root.label == "WATER"
    # WATER has no explication in the registry, so it's a bare molecule node
    assert tree.root.children == []


def test_resolve_fire_returns_molecule_FIRE(resolver):
    tree = resolver.resolve("fire")
    assert tree.root.label == "FIRE"


def test_resolve_hand_returns_molecule_HANDS(resolver):
    """'hand' is an exponent for the HANDS molecule."""
    tree = resolver.resolve("hand")
    assert tree.root.label == "HANDS"


# ---------------------------------------------------------------------------
# 3. WordNet — person sense -> SOMEONE
# ---------------------------------------------------------------------------

def test_resolve_teacher_returns_SOMEONE(resolver):
    """'teacher' is noun.person in WordNet -> SOMEONE."""
    from nsm_ct.wordnet import wordnet_available
    if not wordnet_available():
        pytest.skip("WordNet not available")
    tree = resolver.resolve("teacher")
    assert tree.root.label == "SOMEONE"
    ids = meaning_prime_ids(tree)
    assert len(ids) >= 1


def test_resolve_doctor_returns_SOMEONE(resolver):
    """'doctor' is noun.person in WordNet -> SOMEONE."""
    from nsm_ct.wordnet import wordnet_available
    if not wordnet_available():
        pytest.skip("WordNet not available")
    tree = resolver.resolve("doctor")
    assert tree.root.label == "SOMEONE"


# ---------------------------------------------------------------------------
# 4. WordNet — gloss decomposition
# ---------------------------------------------------------------------------

def test_resolve_bank_has_grounded_prime_ids(resolver):
    """'bank' has WordNet senses -> gloss decomposition -> >= 1 prime id."""
    from nsm_ct.wordnet import wordnet_available
    if not wordnet_available():
        pytest.skip("WordNet not available")
    tree = resolver.resolve("bank")
    assert tree is not None
    assert tree.root is not None
    ids = meaning_prime_ids(tree)
    assert len(ids) >= 1, f"Expected >=1 prime id, got {ids!r}; tree root={tree.root.label!r}"


def test_resolve_bank_tree_not_empty(resolver):
    """'bank' should produce a non-empty tree (root label set)."""
    from nsm_ct.wordnet import wordnet_available
    if not wordnet_available():
        pytest.skip("WordNet not available")
    tree = resolver.resolve("bank")
    assert tree.root.label  # non-empty label


# ---------------------------------------------------------------------------
# 5. Fallback — nonsense / unknown words
# ---------------------------------------------------------------------------

def test_resolve_nonsense_returns_SOMETHING(resolver):
    tree = resolver.resolve("zzzqqx")
    assert tree.root.label == "SOMETHING"


def test_resolve_nonsense_propn_returns_SOMEONE(resolver):
    # The cache key includes the context POS tag, so "zzzqqx" with PROPN context
    # is cached independently from "zzzqqx" without context.
    tree = resolver.resolve("zzzqqx", context={"pos": "PROPN"})
    assert tree.root.label == "SOMEONE"


def test_resolve_nonsense_pron_returns_SOMEONE(resolver):
    tree = resolver.resolve("zzzabcpron", context={"pos": "PRON"})
    assert tree.root.label == "SOMEONE"


# ---------------------------------------------------------------------------
# 6. Caching
# ---------------------------------------------------------------------------

def test_caching_returns_same_object(resolver):
    """Second call (same word, same context) returns the exact same ParseTree object."""
    t1 = resolver.resolve("water")
    t2 = resolver.resolve("water")
    assert t1 is t2


def test_caching_returns_same_structure_for_bank(resolver):
    """Repeated resolution of 'bank' returns the same object from cache."""
    t1 = resolver.resolve("bank")
    t2 = resolver.resolve("bank")
    # Same object from cache
    assert t1 is t2


def test_caching_context_independent_for_known_words(resolver):
    """'water' (a molecule) resolves the same regardless of context."""
    t1 = resolver.resolve("water")
    t2 = resolver.resolve("water", context={"pos": "PROPN"})
    # Different cache keys but same underlying resolution
    assert t1.root.label == t2.root.label == "WATER"


# ---------------------------------------------------------------------------
# 7. ParserInputEncoder uses NSMMeaningResolver by default
# ---------------------------------------------------------------------------

def test_parser_input_encoder_uses_nsm_resolver():
    """ParserInputEncoder should default to NSMMeaningResolver (not Mock)."""
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.tokenizer import SimpleTokenizer

    tok = SimpleTokenizer.build(["hello world"])
    enc = ParserInputEncoder(tok)
    assert isinstance(enc._resolver, NSMMeaningResolver)
