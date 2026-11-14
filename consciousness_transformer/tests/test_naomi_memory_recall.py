import numpy as np
import pytest

from naomi.memory import Memory
from naomi.recall import option_index, recall
from naomi.types import ClauseTree, Filler


def F(role, lemma, token=None, senses=None, hole=False):
    return Filler(role=role, token=token or lemma, lemma=lemma,
                  senses=list(senses or []), is_hole=hole)


def C(pred, args, neg=False, ptoken=None, psenses=None):
    return ClauseTree(predicate=F("PREDICATE", pred, ptoken, psenses),
                      args=[F(r, l, senses=s) if not isinstance(l, Filler) else l
                            for r, l, *rest in [(a + (None,))[:2] + ((a[2],) if len(a) > 2 else (None,))
                                                for a in args]
                            for s in [rest[0]]],
                      negated=neg)


def Q(pred, args, hole_role, hole_pred=False):
    pf = F("PREDICATE", pred, hole=hole_pred)
    a = [F(r, l) for r, l in args]
    if not hole_pred:
        a.append(F(hole_role, "what", hole=True))
    return ClauseTree(predicate=pf, args=a)


def test_what_who_where():
    m = Memory()
    m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], ptoken="ate"))
    m.store(C("sleep", [("SUBJECT", "cat"), ("PLACE", "kitchen")]))
    a = recall(m, Q("eat", [("SUBJECT", "dog")], "OBJECT"))
    assert a.stated and a.text == "bone" and not a.negated
    a = recall(m, Q("eat", [("OBJECT", "bone")], "SUBJECT"))
    assert a.text == "dog"
    a = recall(m, Q("sleep", [("SUBJECT", "cat")], "PLACE"))
    assert a.text == "kitchen"
    assert a.trace


def test_not_stated():
    m = Memory()
    m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")]))
    a = recall(m, Q("fly", [("SUBJECT", "wizard")], "OBJECT"))
    assert a.text is None and not a.stated
    # predicate matches but subject conflicts
    a = recall(m, Q("eat", [("SUBJECT", "wizard")], "OBJECT"))
    assert a.text is None


def test_negated():
    m = Memory()
    nid = m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], neg=True))
    assert m.clauses()[0].negated
    assert m.graph.in_(nid, "OPERATES_ON")
    a = recall(m, Q("eat", [("SUBJECT", "dog")], "OBJECT"))
    assert a.stated and a.negated and a.text == "bone"


def test_positive_beats_negated():
    m = Memory()
    m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], neg=True))
    m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "cake")]))
    a = recall(m, Q("eat", [("SUBJECT", "dog")], "OBJECT"))
    assert a.text == "cake" and not a.negated


def test_distractors():
    m = Memory()
    animals = ["dog", "cat", "cow", "pig", "hen", "fox", "owl", "bat"]
    foods = ["bone", "fish", "hay", "corn", "egg", "nut", "pie", "ham"]
    verbs = ["eat", "see", "chase", "find"]
    n = 0
    for i, an in enumerate(animals):
        for j, fd in enumerate(foods):
            if n >= 60:
                break
            v = verbs[(i + j) % len(verbs)]
            if not (v == "eat" and (an == "fox" or fd == "nut")):
                m.store(C(v, [("SUBJECT", an), ("OBJECT", fd)]))
                n += 1
    m.store(C("eat", [("SUBJECT", "fox"), ("OBJECT", "nut")]))
    assert len(m.clauses()) >= 60
    a = recall(m, Q("eat", [("SUBJECT", "fox")], "OBJECT"))
    assert a.text == "nut"
    a = recall(m, Q("eat", [("OBJECT", "nut")], "SUBJECT"))
    assert a.text == "fox"


def test_multiple_choice():
    m = Memory()
    m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")]))
    a = recall(m, Q("eat", [("SUBJECT", "dog")], "OBJECT"),
               options=["a red ball", "the big bone", "some water"])
    assert a.text == "the big bone" and option_index(a) == 1
    a = recall(m, Q("eat", [("SUBJECT", "dog")], "OBJECT"),
               options=["a red ball", "some water"])
    assert a.text is None and not a.stated


def test_layers():
    m = Memory()
    m.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")]), layer="long_term")
    q = Q("eat", [("SUBJECT", "dog")], "OBJECT")
    assert recall(m, q, layers=["current"]).text is None
    assert recall(m, q).text == "bone"
    assert recall(m, q, layers=["long_term"]).text == "bone"
    m2 = Memory()
    m2.store(C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")]))
    assert recall(m2, q, layers=["long_term"]).text is None
    assert m2.consolidate() == 1
    assert recall(m2, q, layers=["long_term"]).text == "bone"
    assert recall(m2, q, layers=["current"]).text is None
    with pytest.raises(ValueError):
        m2.store(C("eat", [("SUBJECT", "dog")]), layer="bogus")
    m2.clear("long_term")
    assert m2.stats()["clauses"] == 0
    assert "long_term" not in m2.dump()


def test_sense_reuse():
    m = Memory()
    m.store(C("see", [("SUBJECT", "man"), ("OBJECT", ("bank"))]) if False else
            C("see", [("SUBJECT", "man", ["man.n.01"]), ("OBJECT", "bank", ["bank.n.02", "bank.n.01"])]))
    m.store(C("see", [("SUBJECT", "woman", ["woman.n.01"]),
                      ("OBJECT", "bank", ["bank.n.01", "bank.n.02"])]))
    senses = [c.tree.arg("OBJECT").sense for c in m.clauses()]
    assert senses == ["bank.n.02", "bank.n.02"]
    # no senses: stays None
    m.store(C("go", [("SUBJECT", "zorp")]))
    assert m.clauses()[-1].tree.arg("SUBJECT").sense is None
    assert "reuse_lemma_sense" in str(m.clauses()[1].provenance)


def test_handles():
    m = Memory()
    t1 = C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], ptoken="eat")
    t2 = C("eat", [("SUBJECT", "dog"), ("OBJECT", "bone")], ptoken="ate")
    h1, h1b, h2 = m.handle_of(t1), m.handle_of(t1), m.handle_of(t2)
    assert np.array_equal(h1, h1b) and np.array_equal(h1, h2)
    q = Q("eat", [("SUBJECT", "dog")], "OBJECT")
    assert np.linalg.norm(m.handle_of(q)) > 0
    m.store(t1)
    assert np.allclose(m.clauses()[0].handle, h1)
    top = m.similar(m.handle_of(t2), k=1)
    assert top[0][1] > 0.99
