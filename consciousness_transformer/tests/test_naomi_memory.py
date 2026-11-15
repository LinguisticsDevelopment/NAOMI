"""Memory-only tests (storage, layers, sense reuse, handles). No answering."""
import numpy as np
import pytest

from naomi.memory import Memory
from naomi.tree import compact
from naomi.types import Node, Tree


def N(rel, lemma, token=None, senses=None, kind="word", children=None, idx=0):
    return Node(rel=rel, token=token or lemma, token_index=idx, lemma=lemma, pos="X", kind=kind,
                senses=list(senses or []), children=list(children or []))


def C(pred, args, ptoken=None, extra_children=None):
    """args: (rel, lemma[, senses]) tuples -> a one-level Tree."""
    kids = [N(a[0], a[1], senses=a[2] if len(a) > 2 else None, idx=i + 1) for i, a in enumerate(args)]
    return Tree("s", N("ROOT", pred, ptoken, children=kids + list(extra_children or [])))


def test_store_and_stats():
    m = Memory()
    m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], ptoken="ate"))
    m.store(C("sleep", [("SUBJECT", "cat"), ("PLACE", "kitchen")]))
    assert len(m.trees()) == 2 and m.stats()["trees"] == 2
    assert [c.rel for c in m.trees()[0].tree.root.children] == ["SUBJECT", "OBJECT"]
    assert "ate" in m.dump()


def test_negator_is_just_a_node_and_changes_the_handle():
    m = Memory()
    plain = C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")])
    neg = C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone"), ("SPECIFICATION", "not")])
    assert not np.allclose(m.handle_of(plain), m.handle_of(neg))
    m.store(neg)
    assert "not" in m.dump()


def test_nesting_changes_the_handle():
    m = Memory()
    flat = C("saw", [("SUBJECT", "king"), ("DESCRIPTION", "the")])
    nested = C("saw", [("SUBJECT", "king")])
    nested.root.children[0].children.append(N("DESCRIPTION", "the", idx=9))
    assert not np.allclose(m.handle_of(flat), m.handle_of(nested))


def test_referent_vs_concept_from_kind_only():
    m = Memory()
    m.store(C("see", [("SUBJECT", "he"), ("OBJECT", "jim")]))
    kinds = m.stats()["node_kinds"]
    assert kinds.get("CONCEPT", 0) >= 3 and kinds.get("REFERENT", 0) == 0
    t = C("see", [("SUBJECT", "jim")])
    t.root.children[0].kind = "name"
    m2 = Memory()
    m2.store(t)
    assert m2.stats()["node_kinds"].get("REFERENT", 0) == 1


def test_layers():
    m = Memory()
    m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")]), layer="long_term")
    assert len(m.trees("long_term")) == 1 and len(m.trees("current")) == 0
    m2 = Memory()
    m2.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")]))
    assert len(m2.trees("long_term")) == 0
    assert m2.consolidate() == 1
    assert len(m2.trees("long_term")) == 1 and len(m2.trees("current")) == 0
    with pytest.raises(ValueError):
        m2.store(C("eat", [("SUBJECT", "dog")]), layer="bogus")
    m2.clear("long_term")
    assert m2.stats()["trees"] == 0
    assert "long_term" not in m2.dump()


def test_sense_reuse():
    m = Memory()
    m.store(C("see", [("SUBJECT", "man", ["man.n.01"]), ("OBJECT", "bank", ["bank.n.02", "bank.n.01"])]))
    m.store(C("see", [("SUBJECT", "woman", ["woman.n.01"]), ("OBJECT", "bank", ["bank.n.01", "bank.n.02"])]))
    senses = [[n for n in c.tree.root.children if n.rel == "OBJECT"][0].sense for c in m.trees()]
    assert senses == ["bank.n.02", "bank.n.02"]
    m.store(C("go", [("SUBJECT", "zorp")]))
    assert m.trees()[-1].tree.root.children[0].sense is None
    assert "reuse_lemma_sense" in str(m.trees()[1].provenance)


def test_handles():
    m = Memory()
    t1 = C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], ptoken="eat")
    t2 = C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], ptoken="ate")
    h1, h1b, h2 = m.handle_of(t1), m.handle_of(t1), m.handle_of(t2)
    assert np.array_equal(h1, h1b) and np.array_equal(h1, h2)
    q = C("eat", [("SUBJECT", "dog")])
    assert np.linalg.norm(m.handle_of(q)) > 0
    m.store(t1)
    assert np.allclose(m.trees()[0].handle, h1)
    top = m.similar(m.handle_of(t2), k=1)
    assert top[0][1] > 0.99
