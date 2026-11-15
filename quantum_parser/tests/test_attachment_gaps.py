"""Attachment gaps closed in the English grammar: every word becomes a node attached to what it
modifies (coordination, auxiliaries, negator + complement, possessive clitic, infinitive,
subordinators, adjuncts, numerals). Each test asserts the typed edges (type, parent word, child word)
of the best hypothesis for a small sentence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parser import QuantumParser, Tag
from parser.pos_tagger import tag_sentence

_GRAMMAR = str(Path(__file__).resolve().parent.parent / "grammars" / "english.json")
_PARSER = QuantumParser(_GRAMMAR)


def _edges(sentence):
    best = _PARSER.parse(tag_sentence(sentence)).best_hypothesis()
    assert best is not None
    word = lambda i: best.nodes[i].value.text  # noqa: E731
    return {(e.type.name, word(e.parent), word(e.child)) for e in best.edges}, best


def _has(sentence, *wanted):
    got, _ = _edges(sentence)
    missing = [w for w in wanted if w not in got]
    assert not missing, f"{sentence!r}: missing {missing}; got {sorted(got)}"


def test_comma_before_coordinator_links_clauses():
    _has("he ran , and she slept", ("COORDINATION", "ran", "and"), ("COORDINATION", "slept", "and"))


def test_do_support_attaches_to_main_verb():
    _has("the dog did not eat the bone", ("MODIFICATION", "did", "eat"), ("MODIFICATION", "not", "eat"),
         ("SUBJECT", "eat", "dog"), ("OBJECT", "eat", "bone"))


def test_inverted_auxiliary_attaches_to_main_verb():
    _has("did he eat the cake", ("MODIFICATION", "did", "eat"), ("SUBJECT", "eat", "he"))


def test_progressive_auxiliary_attaches_to_participle():
    _has("she was running", ("MODIFICATION", "was", "running"), ("SUBJECT", "running", "she"))
    _has("he was not running", ("MODIFICATION", "was", "running"), ("MODIFICATION", "not", "running"))


def test_modal_before_copula_and_negator():
    _has("the dog could not be found", ("MODIFICATION", "could", "be"), ("MODIFICATION", "not", "be"),
         ("MODIFICATION", "be", "found"))


def test_negator_attaches_to_predicate_complement():
    _has("he was not happy", ("SPECIFICATION", "not", "happy"), ("COMPLEMENT", "was", "happy"),
         ("SUBJECT", "was", "he"))


def test_possessive_clitic_nests_possessor():
    _has("jim 's father gave her three apples", ("MODIFICATION", "father", "'s"), ("PREPOSITION", "'s", "jim"),
         ("SUBJECT", "gave", "father"), ("INDIRECT_OBJECT", "gave", "her"), ("DESCRIPTION", "apples", "three"))


def test_clitic_after_third_person_pronoun_is_copula_not_possession():
    got, _ = _edges("it 's a dog")
    assert ("SUBJECT", "'s", "it") in got
    assert not any(t == "PREPOSITION" for t, _p, _c in got)


def test_numeral_is_tagged_as_a_name_like_number():
    words = tag_sentence("she gave her three apples")
    assert [w.pos for w in words if w.text == "three"] == [Tag.NUM]


def test_subordinators_attach_to_nearest_clause():
    _has("she said that he left because it rained", ("SUBORDINATION", "said", "that"),
         ("SUBORDINATION", "left", "because"), ("MODIFICATION", "because", "rained"))
    got, _ = _edges("she said that he left because it rained")
    assert ("SUBORDINATION", "said", "because") not in got


def test_fronted_subordinate_clause_attaches_to_main_clause():
    _has("when he came , she left", ("SUBORDINATION", "left", "when"))


def test_infinitive_takes_its_own_object_and_adverb():
    _has("he wants to eat the cake", ("MODIFICATION", "to", "eat"), ("OBJECT", "eat", "cake"))
    _has("he wants to go home", ("MODIFICATION", "to", "go"), ("SPECIFICATION", "home", "go"))


def test_sentence_final_adverb_and_fronted_phrase_attach():
    _has("the king never saw the princess again", ("SPECIFICATION", "again", "saw"),
         ("SPECIFICATION", "never", "saw"))
    _has("after school , lisa rushes to the store", ("MODIFICATION", "rushes", "after"))
