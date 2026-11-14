"""Fast tests for the pattern-free naomi.reader (parser, encoder fallback, WordNet fill, cache)."""
import pytest

from naomi.mind import Mind
from naomi.reader import Reader, normalize


@pytest.fixture(scope="module")
def reader():
    return Reader()


def _roles(c):
    return {a.role: a.lemma for a in c.args}


def test_normalize_is_corpus_tokenisation():
    assert normalize("The dog ate.") == "the dog ate ."
    assert normalize("She couldn't go") == "she could n't go"


def test_simple_svo_from_parser(reader):
    rr = reader.read("the dog ate the bone .")
    assert rr.source == "parser" and rr.clauses
    c = rr.clauses[0]
    assert c.predicate.role == "PREDICATE" and c.predicate.lemma == "eat" and c.source == "parser"
    assert _roles(c)["SUBJECT"] == "dog" and _roles(c)["OBJECT"] == "bone"
    assert c.predicate.token_index == 2 and c.arg("SUBJECT").token_index == 1


def test_question_is_just_read(reader):
    a = reader.read_question("who ate the bone ?")
    b = reader.read("who ate the bone ?")
    assert a.clauses == b.clauses
    assert not any(f.is_hole for c in a.clauses for f in [c.predicate] + c.args)


def test_parser_failure_falls_back_to_encoder(reader):
    s = "james did n't know what to do and was scared ."
    assert reader.read_parser(s) is None
    rr = reader.read(s)
    assert rr.source == "encoder" and rr.clauses
    assert all(c.source == "encoder" for c in rr.clauses)


def test_read_both(reader):
    p, e = reader.read_both("the dog ate the bone .")
    assert p is not None and p.source == "parser" and e.source == "encoder"
    assert e.clauses and e.clauses[0].predicate.token_index == 2


def test_parser_labels_carried_through(reader):
    # the parser's own relation labels are kept verbatim (nothing renamed or dropped)
    c = reader.read("he was a favourite .").clauses[0]
    assert {"SUBJECT", "OBJECT", "DESCRIPTION"} <= {a.role for a in c.args}


def test_senses_uncommitted(reader):
    c = reader.read("the dog ate the bone .").clauses[0]
    dog = c.arg("SUBJECT")
    assert dog.senses and dog.senses[0].startswith("dog.n.") and len(dog.senses) <= 5
    assert dog.sense is None and c.predicate.sense is None
    assert c.predicate.lemma == "eat" and c.predicate.senses


def test_cache_round_trip(tmp_path):
    p = tmp_path / "cache.jsonl"
    r1 = Reader(cache_path=str(p), use_encoder=False)
    a = r1.read("the dog ate the bone .")
    r1.save_cache()
    r2 = Reader(cache_path=str(p), use_encoder=False)
    b = r2.read("the dog ate the bone .")
    assert b.clauses == a.clauses
    assert r2._parser is None  # served from cache, parser never loaded


def test_mind_teach_and_ask_not_built():
    m = Mind(reader=Reader(use_encoder=False))
    counts = m.teach(["the dog ate the bone ."])
    assert counts["clauses"] >= 1 and "eat" in m.dump()
    with pytest.raises(NotImplementedError):
        m.ask("what did the dog eat ?")
