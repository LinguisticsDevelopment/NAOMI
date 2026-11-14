"""Memory-only tests (storage, layers, sense reuse, handles). No answering."""
import numpy as np
import pytest

from naomi.memory import Memory
from naomi.types import ClauseTree, Filler


def F(role, lemma, token=None, senses=None, hole=False):
    return Filler(role=role, token=token or lemma, lemma=lemma,
                  senses=list(senses or []), is_hole=hole)


def C(pred, args, neg=False, ptoken=None):
    """args: (role, lemma[, senses]) tuples."""
    return ClauseTree(predicate=F("PREDICATE", pred, ptoken),
                      args=[F(a[0], a[1], senses=a[2] if len(a) > 2 else None) for a in args],
                      negated=neg)


def test_store_and_stats():
    m = Memory()
    m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], ptoken="ate"))
    m.store(C("sleep", [("SUBJECT", "cat"), ("PLACE", "kitchen")]))
    assert len(m.clauses()) == 2 and m.stats()["clauses"] == 2
    assert m.clauses()[0].tree.arg("OBJECT").lemma == "bone"
    assert "eat" in m.dump()


def test_negated_stored_under_operator():
    m = Memory()
    nid = m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], neg=True))
    assert m.clauses()[0].negated
    assert m.graph.in_(nid, "OPERATES_ON")


def test_layers():
    m = Memory()
    m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")]), layer="long_term")
    assert len(m.clauses("long_term")) == 1 and len(m.clauses("current")) == 0
    m2 = Memory()
    m2.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")]))
    assert len(m2.clauses("long_term")) == 0
    assert m2.consolidate() == 1
    assert len(m2.clauses("long_term")) == 1 and len(m2.clauses("current")) == 0
    with pytest.raises(ValueError):
        m2.store(C("eat", [("SUBJECT", "dog")]), layer="bogus")
    m2.clear("long_term")
    assert m2.stats()["clauses"] == 0
    assert "long_term" not in m2.dump()


def test_sense_reuse():
    m = Memory()
    m.store(C("see", [("SUBJECT", "man", ["man.n.01"]), ("OBJECT", "bank", ["bank.n.02", "bank.n.01"])]))
    m.store(C("see", [("SUBJECT", "woman", ["woman.n.01"]), ("OBJECT", "bank", ["bank.n.01", "bank.n.02"])]))
    senses = [c.tree.arg("OBJECT").sense for c in m.clauses()]
    assert senses == ["bank.n.02", "bank.n.02"]
    m.store(C("go", [("SUBJECT", "zorp")]))
    assert m.clauses()[-1].tree.arg("SUBJECT").sense is None
    assert "reuse_lemma_sense" in str(m.clauses()[1].provenance)


def test_handles():
    m = Memory()
    t1 = C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], ptoken="eat")
    t2 = C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], ptoken="ate")
    h1, h1b, h2 = m.handle_of(t1), m.handle_of(t1), m.handle_of(t2)
    assert np.array_equal(h1, h1b) and np.array_equal(h1, h2)
    q = ClauseTree(F("PREDICATE", "eat"), [F("SUBJECT", "dog"), F("OBJECT", "what", hole=True)])
    assert np.linalg.norm(m.handle_of(q)) > 0
    m.store(t1)
    assert np.allclose(m.clauses()[0].handle, h1)
    top = m.similar(m.handle_of(t2), k=1)
    assert top[0][1] > 0.99
