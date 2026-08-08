"""M33 gate — the meaning graph's optional USVS handle path (opt-in, zero
default-behavior-change).

``MeaningGraph.add_concept`` / ``collapse.collapse`` grow an optional
``handle_fn`` hook: absent (``None``, the default) the CONCEPT handle is built
exactly as before (``contract(encode_matrix(tree.root))``); when a hook is
given and returns a vector for the word label, that (unit-normalized) vector
becomes the handle instead. This mirrors the M31 measurement
(``scripts/probe_m31_handles.py``) but drives it through the graph's own API.
"""

import hashlib

import numpy as np
import pytest

from nsm_ct.collapse import collapse, dereference_by_vector, expand
from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.meaning_graph import MeaningGraph, NodeKind
from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.tpr import TPRCodec
from nsm_ct.usvs_bridge import _DEFAULT_DIR, usvs_handle
from nsm_ct.wordnet import wordnet_available

pytestmark = pytest.mark.skipif(
    not wordnet_available() or not _DEFAULT_DIR.exists(),
    reason="needs WordNet + a built data/usvs artifact",
)


class StubResolver:
    """Deterministic word -> distinct small prime tree (no WordNet; fast)."""

    def resolve(self, word, context=None):
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        root = ParseNode(label=PRIME_NAMES[h % len(PRIME_NAMES)], token=word)
        root.children.append(
            ParseNode(label=PRIME_NAMES[(h // 7) % len(PRIME_NAMES)], relation="DESCRIPTION")
        )
        root.children.append(
            ParseNode(label=PRIME_NAMES[(h // 13) % len(PRIME_NAMES)], relation="SPECIFICATION")
        )
        return ParseTree(root=root, text=word)


WORDS = ["dog", "cat", "kitchen", "office", "garden", "justice", "bedroom"]
DIM = 256


def _default_handle(word: str, res: StubResolver, codec: TPRCodec) -> np.ndarray:
    """The exact pre-M33 handle-building path, replicated for comparison."""
    tree = res.resolve(word)
    return codec.contract(codec.encode_matrix(tree.root))


def _build_graph(words, *, handle_fn=None, dim=DIM):
    codec = TPRCodec(dim=dim)
    res = StubResolver()
    g = MeaningGraph(codec)
    nids = {}
    for w in words:
        nids[w] = collapse(g, res.resolve(w), codec, label=w, handle_fn=handle_fn)
    return g, res, nids


# -- default path is byte-identical to pre-M33 behavior ----------------------

def test_default_path_matches_precomputed_handles_exactly():
    """No handle_fn (the default) -> same handle bytes as building it by hand."""
    codec = TPRCodec(dim=DIM)
    res = StubResolver()
    g = MeaningGraph(codec)
    for w in WORDS:
        nid = collapse(g, res.resolve(w), codec, label=w)
        expected = _default_handle(w, res, codec)
        assert np.array_equal(g.node(nid).handle, expected)


def test_default_path_deterministic_across_two_builds():
    g1, _, nids1 = _build_graph(WORDS)
    g2, _, nids2 = _build_graph(WORDS)
    for w in WORDS:
        assert np.array_equal(g1.node(nids1[w]).handle, g2.node(nids2[w]).handle)


def test_handle_fn_absent_is_indistinguishable_from_old_add_concept_signature():
    """add_concept(label, tree) with no handle_fn kwarg at all still works."""
    codec = TPRCodec(dim=DIM)
    res = StubResolver()
    g = MeaningGraph(codec)
    nid = g.add_concept("kitchen", res.resolve("kitchen"))
    expected = _default_handle("kitchen", res, codec)
    assert np.array_equal(g.node(nid).handle, expected)


# -- opt-in USVS provider ------------------------------------------------------

def _usvs_provider(dim=DIM):
    return lambda word: usvs_handle(word, dim)


def test_usvs_provider_changes_concept_handles():
    plain, _, plain_nids = _build_graph(WORDS)
    hooked, _, hooked_nids = _build_graph(WORDS, handle_fn=_usvs_provider())
    changed = 0
    for w in WORDS:
        if usvs_handle(w, DIM) is None:
            continue  # word unknown to USVS -> fallback, handled separately
        h_plain = plain.node(plain_nids[w]).handle
        h_hooked = hooked.node(hooked_nids[w]).handle
        assert not np.array_equal(h_plain, h_hooked)
        changed += 1
    assert changed > 0  # the pool must contain at least one USVS-known word


def test_usvs_provider_handles_are_unit_norm():
    g, _, nids = _build_graph(WORDS, handle_fn=_usvs_provider())
    for w in WORDS:
        if usvs_handle(w, DIM) is None:
            continue
        h = g.node(nids[w]).handle
        assert np.linalg.norm(h) == pytest.approx(1.0, abs=1e-5)


def test_usvs_provider_concept_dereferences_correctly():
    g, _, nids = _build_graph(WORDS, handle_fn=_usvs_provider())
    for w in WORDS:
        nid = nids[w]
        found, margin = dereference_by_vector(g, g.node(nid).handle, kind_filter=NodeKind.CONCEPT)
        assert found == nid
        assert margin >= 0.0


def test_usvs_provider_concept_still_expands_losslessly():
    """The hook only touches the vector handle; the stored structure (and thus
    expand/lossless round-trip) is untouched."""
    res = StubResolver()
    g, _, nids = _build_graph(WORDS, handle_fn=_usvs_provider())
    for w in WORDS:
        recovered = expand(g, nids[w])
        assert recovered.root.label == res.resolve(w).root.label


def test_unknown_word_falls_back_to_default_handle():
    word = "zzz-not-a-real-word-xyz"
    codec = TPRCodec(dim=DIM)
    res = StubResolver()
    g = MeaningGraph(codec)
    nid = collapse(g, res.resolve(word), codec, label=word, handle_fn=_usvs_provider())
    expected = _default_handle(word, res, codec)
    assert np.array_equal(g.node(nid).handle, expected)


def test_provider_returning_wrong_dim_falls_back_to_default():
    """Defensive: a hook returning a vector of the wrong width is ignored, not
    crashed on."""
    codec = TPRCodec(dim=DIM)
    res = StubResolver()
    g = MeaningGraph(codec)
    bad_fn = lambda word: np.ones(7, dtype=np.float32)
    nid = collapse(g, res.resolve("kitchen"), codec, label="kitchen", handle_fn=bad_fn)
    expected = _default_handle("kitchen", res, codec)
    assert np.array_equal(g.node(nid).handle, expected)


# -- non-CONCEPT nodes are untouched by the hook ------------------------------

def test_referent_nodes_identical_with_and_without_hook():
    codec = TPRCodec(dim=64)
    g_plain = MeaningGraph(codec)
    g_hooked = MeaningGraph(codec)
    a = g_plain.add_referent("mary")
    b = g_hooked.add_referent("mary")
    assert np.array_equal(g_plain.node(a).handle, g_hooked.node(b).handle)


def test_clause_and_operator_nodes_unaffected_by_handle_fn():
    """collapse() with kind=CLAUSE never consults handle_fn (no label routing
    to add_concept applies), so results are identical regardless of the hook."""
    codec = TPRCodec(dim=64)
    root = ParseNode(label="is", token="is")
    root.children.append(ParseNode(label="SOMEONE", token="mary", relation="SUBJECT"))
    tree = ParseTree(root=root, text="mary is")

    g1 = MeaningGraph(codec)
    nid1 = collapse(g1, tree, codec, kind=NodeKind.CLAUSE)
    g2 = MeaningGraph(codec)
    nid2 = collapse(g2, tree, codec, kind=NodeKind.CLAUSE, handle_fn=_usvs_provider())

    assert np.array_equal(g1.node(nid1).handle, g2.node(nid2).handle)
