"""Stage 1 gate — the meaning-graph substrate (no parser, no training)."""

import hashlib

import numpy as np

from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.meaning_graph import (
    ABOUT,
    ClausePayload,
    MeaningGraph,
    NodeKind,
    SLOT,
)
from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.serialization import deserialize_thought
from nsm_ct.tpr import TPRCodec


class StubResolver:
    """Deterministic word → small prime tree (no WordNet; keeps tests fast)."""

    def resolve(self, word, context=None):
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        root = ParseNode(label=PRIME_NAMES[h % len(PRIME_NAMES)], token=word)
        root.children.append(
            ParseNode(label=PRIME_NAMES[(h // 7) % len(PRIME_NAMES)], relation="DESCRIPTION")
        )
        return ParseTree(root=root, text=word)


def _same_tree(a: ParseNode, b: ParseNode) -> bool:
    return (
        a.label == b.label
        and a.relation == b.relation
        and a.token == b.token
        and a.index == b.index
        and len(a.children) == len(b.children)
        and all(_same_tree(x, y) for x, y in zip(a.children, b.children))
    )


def _graph(dim=64):
    return MeaningGraph(TPRCodec(dim=dim)), StubResolver()


def test_referent_is_shared_not_duplicated():
    g, _ = _graph()
    a = g.add_referent("mary")
    b = g.add_referent("Mary")  # case-insensitive
    assert a == b
    assert g.node(a).kind is NodeKind.REFERENT
    assert len(g.referent_index) == 1


def test_concept_node_has_handle_and_lossless_structure():
    g, res = _graph(dim=64)
    nid = g.add_concept("kitchen", res.resolve("kitchen"))
    node = g.node(nid)
    assert node.kind is NodeKind.CONCEPT
    assert node.handle.shape == (64,)
    assert node.structure and isinstance(node.structure, list)
    # the same word reuses the one concept node
    assert g.add_concept("kitchen", res.resolve("kitchen")) == nid


def test_structure_round_trips_to_resolver_tree_identity():
    g, res = _graph()
    tree = res.resolve("office")
    nid = g.add_concept("office", tree)
    recovered = deserialize_thought(g.node(nid).structure)
    assert _same_tree(recovered.root, tree.root)


def test_adjacency_helpers_on_a_clause():
    g, res = _graph()
    mary = g.add_referent("mary")
    pred = g.add_concept("is", res.resolve("is"))
    kitchen = g.add_concept("kitchen", res.resolve("kitchen"))
    payload = ClausePayload(predicate_nid=pred, slots=[("SUBJECT", mary), ("PLACE", kitchen)])
    clause = g.add_clause(payload, np.zeros(64, dtype=np.float32))

    # SLOT edges: predicate + 2 args
    assert len(g.out(clause, SLOT)) == 3
    # the clause is ABOUT mary (a referent), not about the kitchen concept
    assert g.clauses_about(mary) == [clause]
    assert g.clauses_about(kitchen) == []
    assert set(g.neighbors(clause)) == {pred, mary, kitchen}
    assert {e.rel for e in g.out(clause, SLOT)} == {"PREDICATE", "SUBJECT", "PLACE"}
