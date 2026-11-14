"""Fast tests for naomi.reader (hybrid parser/encoder reader)."""
import pytest

from naomi.reader import Reader


@pytest.fixture(scope="module")
def reader():
    return Reader()


def _roles(c):
    return {a.role: a.lemma for a in c.args}


def test_simple_svo(reader):
    rr = reader.read("the dog ate the bone .")
    assert rr.source == "parser" and rr.clauses
    c = rr.clauses[0]
    assert c.predicate.role == "PREDICATE" and c.predicate.lemma == "eat"
    assert _roles(c)["SUBJECT"] == "dog" and _roles(c)["OBJECT"] == "bone"
    assert not c.negated


def test_negation_did_not(reader):
    c = reader.read("the dog did not eat the bone .").clauses[0]
    assert c.negated and c.predicate.lemma == "eat"
    assert _roles(c)["SUBJECT"] == "dog" and _roles(c)["OBJECT"] == "bone"


def test_negation_nt(reader):
    c = reader.read("she could n't find the key .").clauses[0]
    assert c.negated and c.predicate.lemma == "find"


def test_negation_no_one(reader):
    c = reader.read("no one came to the house .").clauses[0]
    assert c.negated and c.predicate.lemma == "come"


def test_encoder_fallback(reader):
    rr = reader.read("james did n't know what to do and was scared .")
    assert rr.source == "encoder"
    assert len(rr.clauses) >= 1
    assert all(c.source == "encoder" for c in rr.clauses)


def test_question_what_did(reader):
    c = reader.read_question("what did the dog eat ?").clauses[0]
    assert c.predicate.lemma == "eat" and c.arg("SUBJECT").lemma == "dog"
    h = c.hole()
    assert h is not None and h.role == "OBJECT"
    assert sum(a.is_hole for a in c.args) == 1


def test_question_who(reader):
    c = reader.read_question("who ate the bone ?").clauses[0]
    assert c.predicate.lemma == "eat" and c.hole().role == "SUBJECT"


def test_question_where(reader):
    c = reader.read_question("where is mary ?").clauses[0]
    assert c.hole().role == "PLACE" and c.arg("SUBJECT").lemma == "mary"


def test_question_what_do_predicate_hole(reader):
    c = reader.read_question("what did the dog do ?").clauses[0]
    assert c.predicate.is_hole and c.arg("SUBJECT").lemma == "dog"


def test_senses_uncommitted(reader):
    c = reader.read("the dog ate the bone .").clauses[0]
    dog = c.arg("SUBJECT")
    assert dog.senses and dog.senses[0].startswith("dog.n.") and len(dog.senses) <= 5
    assert dog.sense is None and c.predicate.sense is None


def test_cache_round_trip(tmp_path):
    p = tmp_path / "cache.jsonl"
    r1 = Reader(cache_path=str(p), use_encoder=False)
    a = r1.read("the dog ate the bone .")
    q = r1.read_question("who ate the bone ?")
    r1.save_cache()
    r2 = Reader(cache_path=str(p), use_encoder=False)
    b = r2.read("the dog ate the bone .")
    q2 = r2.read_question("who ate the bone ?")
    assert b.clauses == a.clauses and q2.clauses == q.clauses
    assert r2._parser is None  # served from cache, parser never loaded for the result
