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


# ---- possessives / copulas / quantities / question shapes --------------------------
def test_possessive_kept_in_statement(reader):
    c = reader.read("jim's father was greg .").clauses[0]
    r = _roles(c)
    assert r["POSSESSOR"] == "jim" and r["SUBJECT"] == "father" and r["OBJECT"] == "greg"


def test_possessive_kept_in_question(reader):
    c = reader.read_question("Who was Jim's father?").clauses[0]
    assert c.hole().role == "SUBJECT"
    r = _roles(c)
    assert r["POSSESSOR"] == "jim" and r["OBJECT"] == "father"


def test_copula_complement_kept(reader):
    c = reader.read("he was not a favourite .").clauses[0]
    assert c.predicate.lemma == "be" and c.negated
    assert c.arg("ATTRIBUTE").lemma == "favourite"
    c = reader.read("the dog 's tail was long .").clauses[0]
    assert c.arg("ATTRIBUTE").lemma == "long" and c.arg("POSSESSOR").lemma == "dog"


def test_quantity_attached_to_noun(reader):
    c = reader.read("they stayed two days .").clauses[0]
    q = c.arg("QUANTITY")
    assert q is not None and q.lemma == "two"
    assert any(a.lemma.startswith("day") for a in c.args)


def test_how_many_how_long_holes(reader):
    c = reader.read_question("How many sons did the king have?").clauses[0]
    assert c.hole().role == "QUANTITY" and c.arg("SUBJECT").lemma == "king"
    assert any(a.lemma == "son" for a in c.args)
    c = reader.read_question("How long did they stay?").clauses[0]
    assert c.hole().role == "QUANTITY" and c.predicate.lemma == "stay"


def test_stranded_preposition_hole(reader):
    c = reader.read_question("Who did the people sing about?").clauses[0]
    assert c.predicate.lemma == "sing" and c.hole().role == "ABOUT"
    assert c.arg("SUBJECT").lemma == "people"


def test_attribute_nouns(reader):
    c = reader.read_question("What was the name of the fox?").clauses[0]
    assert c.hole().role == "ATTRIBUTE" and c.arg("KIND").lemma == "name"
    assert c.arg("SUBJECT").lemma == "fox"
    c = reader.read_question("What color was the dog?").clauses[0]
    assert c.hole().role == "ATTRIBUTE" and c.arg("KIND").lemma == "color"


def test_subordinate_clause_is_context(reader):
    c = reader.read_question("What did Tom see when he went home?").clauses[0]
    assert c.predicate.lemma == "see" and c.hole().role == "OBJECT"
    assert c.arg("SUBJECT").lemma == "tom"
    assert any(a.role == "CONTEXT" for a in c.args)
    assert not any(a.role == "SUBJECT" and a.lemma == "he" for a in c.args)


def test_what_do_with_trailing_words(reader):
    c = reader.read_question("What did Billy do every day after school?").clauses[0]
    assert c.predicate.is_hole and c.arg("SUBJECT").lemma == "billy"


def test_capitalised_name_is_not_a_verb(reader):
    c = reader.read_question("What did Mark have to write?").clauses[0]
    assert c.arg("SUBJECT").lemma == "mark" and c.hole().role == "OBJECT"


def test_long_sentence_is_split_into_clauses(reader):
    cs = reader.read("jenny rode in the car and her dad drove the truck toward utah .").clauses
    assert any(c.predicate.lemma == "drive" and c.arg("SUBJECT").lemma == "dad" for c in cs)
