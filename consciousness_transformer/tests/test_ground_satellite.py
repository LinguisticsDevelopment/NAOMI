"""Gates for the satellite-cluster indirect antonymy signal (signal_satellite)."""

import pytest

from nsm_ct.ground.relations import RelationGraph
from nsm_ct.ground.signal_satellite import expanded_antonyms, extras
from nsm_ct.wordnet import wordnet_available

pytestmark = pytest.mark.skipif(not wordnet_available(), reason="needs WordNet")

# damp is a satellite of wet; arid is a satellite of dry; wet/dry are head
# antonyms. Skip (rather than hardcode a chain WordNet may not have) if the
# expected structure isn't actually present in the installed corpus.
_CHAIN_WORDS = ("damp", "wet", "dry", "arid")


def _chain_available() -> bool:
    from nltk.corpus import wordnet as wn

    if not (wn.synsets("damp", wn.ADJ_SAT) and wn.synsets("arid", wn.ADJ_SAT)):
        return False
    wet = wn.synset("wet.a.01")
    dry = wn.synset("dry.a.01")
    ant_names = {a.name() for l in wet.lemmas() for a in l.antonyms()}
    if "dry" not in ant_names:
        return False
    damp_heads = {h.name() for h in wn.synset("damp.s.01").similar_tos()}
    if "wet.a.01" not in damp_heads:
        return False
    dry_sat_names = {l.name() for s in dry.similar_tos() for l in s.lemmas()}
    return "arid" in dry_sat_names


pytestmark_chain = pytest.mark.skipif(
    not wordnet_available() or not _chain_available(),
    reason="needs WordNet with the damp/wet/dry/arid satellite chain",
)


@pytest.fixture(scope="module")
def small_graph():
    words = ["damp", "wet", "dry", "arid", "hot", "cold", "happy", "sad", "table"]
    return words, RelationGraph.build(words)


@pytestmark_chain
def test_known_chain_damp_to_dry():
    """damp ~ wet, wet <-> dry => damp <-> dry (indirect antonym via satellite)."""
    out = expanded_antonyms("damp")
    assert "dry" in out


@pytestmark_chain
def test_known_chain_damp_to_arid():
    """damp <-> dry's satellite arid too (satellite-to-satellite via shared heads)."""
    out = expanded_antonyms("damp")
    assert "arid" in out


@pytestmark_chain
def test_known_chain_symmetric():
    """The expansion is symmetric: arid <-> damp iff damp <-> arid."""
    assert "arid" in expanded_antonyms("damp")
    assert "damp" in expanded_antonyms("arid")
    assert "dry" in expanded_antonyms("damp")
    assert "damp" in expanded_antonyms("dry")


def test_excludes_word_itself():
    for w in ("damp", "wet", "dry", "hot", "cold"):
        out = expanded_antonyms(w)
        assert w not in out


def test_deterministic():
    for w in ("damp", "wet", "dry", "hot"):
        a = expanded_antonyms(w)
        b = expanded_antonyms(w)
        assert a == b
        assert a == sorted(set(a))  # deduped and sorted


def test_no_adjective_senses_returns_empty():
    # "table" has no adjective senses in WordNet -> no chain to walk.
    from nltk.corpus import wordnet as wn

    assert not wn.synsets("table", wn.ADJ) and not wn.synsets("table", wn.ADJ_SAT)
    assert expanded_antonyms("table") == []


def test_extras_contract_shape(small_graph):
    words, g = small_graph
    out = extras(words, g)
    assert set(out.keys()) == {"antonym_extra"}
    pairs = out["antonym_extra"]
    assert isinstance(pairs, list)
    for p in pairs:
        assert isinstance(p, tuple) and len(p) == 2
        a, b = p
        assert a in words and b in words
        assert a != b
        assert tuple(sorted(p)) == p  # canonical (sorted) pair form


def test_extras_excludes_existing_antonym_pairs(small_graph):
    words, g = small_graph
    existing = {tuple(sorted(p)) for p in g.typed_pairs("antonym")}
    assert existing, "fixture needs at least one existing antonym pair (hot/cold)"
    out = extras(words, g)
    for p in out["antonym_extra"]:
        assert p not in existing


@pytestmark_chain
def test_extras_finds_damp_dry(small_graph):
    words, g = small_graph
    out = extras(words, g)
    assert ("damp", "dry") in out["antonym_extra"]


def test_extras_only_in_vocab_pairs(small_graph):
    words, g = small_graph
    out = extras(words, g)
    wset = set(words)
    for a, b in out["antonym_extra"]:
        assert a in wset and b in wset


def test_extras_deterministic(small_graph):
    words, g = small_graph
    a = extras(words, g)
    b = extras(words, g)
    assert a == b


def test_extras_no_duplicates(small_graph):
    words, g = small_graph
    pairs = extras(words, g)["antonym_extra"]
    assert len(pairs) == len(set(pairs))
