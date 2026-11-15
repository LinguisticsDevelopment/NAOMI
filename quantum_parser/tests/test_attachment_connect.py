"""Connecting the pieces of a sentence: reported speech, coordination across punctuation, subordinate
and participle clauses, stranded prepositions, auxiliaries left heading their own clause. The punctuation
subtypes (QUOTE_OPEN/QUOTE_CLOSE resolved by position, SEMICOLON, COLON, TERMINAL) come from the lexicon."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parser import QuantumParser, SubType
from parser.pos_tagger import tag_sentence

_GRAMMAR = str(Path(__file__).resolve().parent.parent / "grammars" / "english.json")
_PARSER = QuantumParser(_GRAMMAR)


def _best(sentence):
    best = _PARSER.parse(tag_sentence(sentence)).best_hypothesis()
    assert best is not None
    return best


def _edges(sentence):
    best = _best(sentence)
    word = lambda i: best.nodes[i].value.text  # noqa: E731
    return {(e.type.name, word(e.parent), word(e.child)) for e in best.edges}


def _has(sentence, *wanted):
    got = _edges(sentence)
    missing = [w for w in wanted if w not in got]
    assert not missing, f"{sentence!r}: missing {missing}; got {sorted(got)}"


def _pieces(sentence):
    """Unconsumed non-punctuation units of the best hypothesis (1 = one connected tree)."""
    best = _best(sentence)
    return [i for i in best.get_unconsumed() if best.nodes[i].type.name != "NIL"]


def _flags(sentence):
    return [[s.name for s in w.subtypes] for w in tag_sentence(sentence)]


def test_bare_quote_marks_get_a_direction_from_position():
    f = _flags('" go , " said he')
    assert "QUOTE_OPEN" in f[0] and "QUOTE_CLOSE" in f[3]
    assert "QUOTE_CLOSE" in _flags("he left . \"")[-1]          # lone closing mark of a longer quotation
    assert "QUOTE_OPEN" in _flags('" he left')[0]                # lone opening mark at the start


def test_sentence_marks_carry_their_identity():
    f = _flags("a ; b : c . d ? e !")
    assert "SEMICOLON" in f[1] and "COLON" in f[3] and "TERMINAL" in f[5] and "TERMINAL" in f[7] and "TERMINAL" in f[9]
    assert SubType.SEMICOLON in tag_sentence("a ; b")[1].subtypes


def test_quote_then_inverted_reporter():
    s = '" you must go , " said the little boy .'
    _has(s, ("SUBJECT", "said", "boy"), ("OBJECT", "said", "go"))
    assert len(_pieces(s)) == 1


def test_reporter_then_quote():
    s = 'jack said , " come here . " then he left .'
    _has(s, ("SUBJECT", "said", "jack"), ("OBJECT", "said", "come"))


def test_quotation_split_around_the_reporter_attaches_both_parts():
    s = '" it is very unpleasant , i am afraid of the police , " said pickles .'
    _has(s, ("OBJECT", "said", "is"), ("OBJECT", "said", "am"), ("SUBJECT", "said", "pickles"))
    assert len(_pieces(s)) == 1


def test_several_sentences_in_one_quotation_attach_to_the_reporter():
    _has('" i go . you stay , " said the king .', ("OBJECT", "said", "go"), ("OBJECT", "said", "stay"))


def test_semicolon_before_coordinator_and_between_clauses():
    _has("he left , and she wept", ("COORDINATION", "left", "and"), ("COORDINATION", "wept", "and"))
    _has("he boasted ; and she laughed", ("COORDINATION", "boasted", "and"), ("COORDINATION", "laughed", "and"))
    _has("he boasted ; she laughed", ("COORDINATION", "boasted", "laughed"))


def test_trailing_subordinate_clause_after_comma():
    _has("she wept , when he left", ("SUBORDINATION", "wept", "when"), ("MODIFICATION", "when", "left"))
    assert len(_pieces("she wept , when he left")) == 1


def test_auxiliary_that_headed_its_own_clause_takes_the_verb():
    _has("it might have been worse", ("MODIFICATION", "might", "have"), ("MODIFICATION", "have", "been"))


def test_infinitive_purpose_clause_is_one_tree():
    s = "he went to the well to fetch water"
    _has(s, ("MODIFICATION", "to", "fetch"), ("OBJECT", "fetch", "water"))
    assert len(_pieces(s)) == 1


def test_stranded_preposition_attaches_to_nearest_clause():
    s = "the jury wrote down on their slates"
    assert len(_pieces(s)) == 1


def test_participle_phrase_modifies_the_clause():
    s = "he sat there , looking at the sea"
    _has(s, ("MODIFICATION", "sat", "looking"))
    assert len(_pieces(s)) == 1
