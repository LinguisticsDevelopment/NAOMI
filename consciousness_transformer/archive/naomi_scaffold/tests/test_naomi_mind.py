"""Fast tests for naomi.mind (teach -> ask)."""
import pytest

from naomi.mind import Mind
from naomi.reader import Reader


@pytest.fixture(scope="module")
def reader():
    return Reader()


def _mind(reader, text):
    m = Mind(reader=reader)
    m.teach(text)
    return m


def test_basic_questions(reader):
    m = _mind(reader, "the dog ate the bone . mary went to the kitchen .")
    assert m.ask("what did the dog eat ?").text == "bone"
    assert m.ask("where did mary go ?").text == "kitchen"
    assert m.ask("who ate the bone ?").text == "dog"
    a = m.ask("what did the cat eat ?")
    assert a.text is None and m.say(a) == "not stated"


def test_negation(reader):
    m = _mind(reader, "the dog did not eat the bone .")
    a = m.ask("what did the dog eat ?")
    s = m.say(a)
    assert s == "not stated" or "negated" in s


def test_sleep_and_options(reader):
    m = _mind(reader, "the dog ate the bone . mary went to the kitchen .")
    assert m.sleep() >= 2
    assert not m.memory.clauses("current")
    assert m.ask("what did the dog eat ?").text == "bone"
    from naomi.recall import option_index
    a = m.ask("what did the dog eat ?", options=["a cake", "the bone", "a fish", "an apple"])
    assert option_index(a) == 1


def test_wrong_story_abstains(reader):
    m = _mind(reader, "the dog ate the bone . mary went to the kitchen .")
    m2 = Mind(reader=reader)
    m2.teach("the cat drank the milk .")
    assert m2.ask("where did mary go ?").text is None
    m.forget()
    assert m.ask("what did the dog eat ?").text is None


def test_possessive_copula_and_quantity_end_to_end(reader):
    m = _mind(reader, "Jim's father was Greg. The dog was named Rex. Anna's cat was black. "
                      "She bought three apples for fifty cents. They stayed two days.")
    assert m.ask("Who was Jim's father?").text == "greg"
    assert m.ask("What was the name of the dog?").text == "rex"
    assert m.ask("What color was Anna's cat?").text == "black"
    assert m.ask("How many apples did she buy?").text == "three apples"
    assert m.ask("How long did they stay?").text == "two days"
    assert m.ask("Who was Anna's father?").text is None


def test_pronoun_coreference_answers_with_the_name(reader):
    m = _mind(reader, "Tom went to the park. He ate the cake. The people sang about the king.")
    assert m.ask("Who ate the cake?").text == "tom"
    assert m.ask("Who did the people sing about?").text == "king"


def test_subordinate_question_clause_does_not_veto(reader):
    m = _mind(reader, "Tom saw a bird . Mary went home .")
    assert m.ask("What did Tom see when he went home?").text == "bird"
