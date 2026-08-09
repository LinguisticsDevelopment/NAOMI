"""Tests for the WordNet-generated open-class lexicon layer in pos_tagger.

These lock the M41 contract: closed-class hand entries keep precedence, the
lexicon fills open-class coverage with frequency-ordered multi-POS entries +
morphological subtypes, and everything degrades gracefully if the artifact is
missing (see _load_lexicon).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parser.data_structures import Word
from parser.enums import SubType, Tag
from parser.pos_tagger import (
    WORD_TAG_DICT,
    get_possible_tags,
    lexicon_entry,
    lexicon_subtypes,
    simple_tag,
    tag_words,
)


def test_lexicon_loads_and_has_open_class_coverage():
    assert lexicon_entry("zeppelin")[0][0] == Tag.NOUN
    assert lexicon_entry("wombat")[0][0] == Tag.NOUN


def test_past_participles_carry_the_passive_anchor_flag():
    for w in ("moved", "broken", "carried", "eaten"):
        assert SubType.PAST_PARTICIPLE in lexicon_subtypes(w, Tag.VERB), w


def test_third_person_verbs_get_agreement_subtypes():
    subs = lexicon_subtypes("thinks", Tag.VERB)
    assert SubType.THIRD_PERSON in subs and SubType.SINGULAR in subs


def test_plural_nouns_flagged():
    assert SubType.PLURAL in lexicon_subtypes("tables", Tag.NOUN)
    assert SubType.PLURAL in lexicon_subtypes("children", Tag.NOUN)  # irregular


def test_closed_class_dict_keeps_precedence():
    # "will" is a WordNet noun/verb but the hand dict says modal AUX — and wins.
    assert simple_tag("will") == Tag.AUX
    # Closed-class words never branch in the lattice.
    assert get_possible_tags(Word("behind", Tag.ADP, [])) == [Tag.ADP]


def test_new_prepositions_present():
    for p in ("behind", "beside", "above", "below", "onto", "against"):
        assert WORD_TAG_DICT[p] == Tag.ADP, p


def test_multi_pos_words_branch_in_the_lattice():
    tags = get_possible_tags(Word("shed", simple_tag("shed"), []))
    assert Tag.NOUN in tags and Tag.VERB in tags


def test_tag_words_merges_lexicon_subtypes():
    (w,) = tag_words(["broken"])
    assert w.pos == Tag.VERB
    assert SubType.PAST_PARTICIPLE in w.subtypes


def test_capitalized_names_stay_propn_even_if_lexicon_knows_them():
    # "Bill" the name must not become NOUN/VERB via the lexicon.
    assert simple_tag("Bill") == Tag.PROPN
