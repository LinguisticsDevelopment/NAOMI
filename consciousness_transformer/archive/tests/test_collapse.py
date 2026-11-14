"""Stage 2 gate — collapse/expand is lossless; vector deref reports a margin."""

import hashlib

import numpy as np

from nsm_ct.collapse import collapse, dereference_by_vector, expand, flatten_concept
from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.meaning_graph import MeaningGraph, NodeKind
from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.tpr import TPRCodec


class StubResolver:
    """Deterministic word → distinct small prime tree (no WordNet; fast)."""

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


def _same(a: ParseNode, b: ParseNode) -> bool:
    return (
        a.label == b.label and a.relation == b.relation and a.token == b.token
        and len(a.children) == len(b.children)
        and all(_same(x, y) for x, y in zip(a.children, b.children))
    )


WORDS = ["kitchen", "office", "garden", "bedroom", "hallway", "is", "moved"]


def _collapsed_graph(dim=128):
    g = MeaningGraph(TPRCodec(dim=dim))
    res = StubResolver()
    nids = {w: collapse(g, res.resolve(w), g.codec, label=w) for w in WORDS}
    return g, res, nids


def test_collapse_expand_is_lossless_round_trip():
    g, res, nids = _collapsed_graph()
    exact = 0
    for w, nid in nids.items():
        if _same(expand(g, nid).root, res.resolve(w).root):
            exact += 1
    assert exact == len(WORDS)  # 100% — the hard Stage-2 gate


def test_dereference_by_exact_handle_finds_its_node():
    g, _, nids = _collapsed_graph()
    for w, nid in nids.items():
        found, margin = dereference_by_vector(
            g, g.node(nid).handle, kind_filter=NodeKind.CONCEPT,
        )
        assert found == nid
        assert margin > 0.0  # separable from the runner-up


def test_dereference_survives_small_noise_on_distinct_concepts():
    g, _, nids = _collapsed_graph(dim=256)
    rng = np.random.default_rng(0)
    for w, nid in nids.items():
        h = g.node(nid).handle
        noisy = h + 0.05 * np.linalg.norm(h) * rng.standard_normal(h.shape).astype(np.float32)
        found, _ = dereference_by_vector(g, noisy, kind_filter=NodeKind.CONCEPT)
        assert found == nid


def test_collapse_a_clause_round_trips():
    g = MeaningGraph(TPRCodec(dim=64))
    root = ParseNode(label="is", token="is")
    root.children.append(ParseNode(label="SOMEONE", token="mary", relation="SUBJECT"))
    root.children.append(ParseNode(label="SOMEWHERE", token="kitchen", relation="PLACE"))
    tree = ParseTree(root=root, text="mary is in the kitchen")
    nid = collapse(g, tree, g.codec, kind=NodeKind.CLAUSE)
    assert g.node(nid).kind is NodeKind.CLAUSE
    assert _same(expand(g, nid).root, root)


def test_flatten_concept_terminates_and_returns_atoms():
    g, _, nids = _collapsed_graph()
    atoms = flatten_concept(g, nids["kitchen"])
    assert atoms and all(isinstance(a, str) for a in atoms)
