"""Tests for the OMW-es-generated open-class Spanish lexicon layer in
pos_tagger (Spanish Freeze Test, see consciousness_transformer/dev/
ROADMAP_LONG_TERM.md and scripts/build_parser_lexicon.py --lang spa).

Mirrors test_lexicon.py's English contract: closed-class hand entries
(SPANISH_WORD_TAG_DICT) keep precedence, the lexicon fills open-class
coverage, and everything degrades gracefully if the artifact is missing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parser.enums import Tag
from parser.pos_tagger import (
    SPANISH_WORD_TAG_DICT,
    es_lexicon_entry,
    es_lexicon_subtypes,
    simple_tag_spanish,
    tag_spanish_sentence,
    tag_spanish_words,
)


def test_es_lexicon_loads_and_has_open_class_coverage():
    assert es_lexicon_entry("jardín")[0][0] == Tag.NOUN
    assert es_lexicon_entry("cocina")[0][0] == Tag.NOUN
    assert es_lexicon_entry("pelota")[0][0] == Tag.NOUN


def test_es_lexicon_missing_word_returns_none():
    assert es_lexicon_entry("zzznotarealspanishword") is None
    assert es_lexicon_subtypes("zzznotarealspanishword", Tag.NOUN) == []


def test_closed_class_dict_keeps_precedence_over_lexicon():
    # "está" (estar) is hand-listed VERB; must not be overridden by any
    # lexicon entry even if OMW-es has other POS readings for related forms.
    assert simple_tag_spanish("está") == Tag.VERB
    assert SPANISH_WORD_TAG_DICT["en"] == Tag.ADP


def test_punctuation_guard_handles_inverted_marks():
    # "¿"/"¡" are not in string.punctuation (ASCII-only) -- the Spanish
    # tagger has its own guard (_ES_PUNCT_EXTRA) so these never fall through
    # to the open-class default (NOUN), which would corrupt parsing (see
    # pos_tagger.py's module-level note by _ES_PUNCT_EXTRA).
    assert simple_tag_spanish("¿") == Tag.PUNCT
    assert simple_tag_spanish("¡") == Tag.PUNCT
    assert simple_tag_spanish(".") == Tag.PUNCT
    assert simple_tag_spanish("?") == Tag.PUNCT


def test_al_del_contractions_tagged_adp():
    # a+el / de+el contractions must resolve to the SAME node type as the
    # bare preposition (ADP -> NodeType.PREP) so normPP1's PREP+NOUN rule
    # fires with no DET node in between (the contraction already ate it).
    assert simple_tag_spanish("al") == Tag.ADP
    assert simple_tag_spanish("del") == Tag.ADP


def test_interrogative_donde_tagged_adv_relative():
    from parser.pos_tagger import SPANISH_WORD_SUBTYPES
    from parser.enums import SubType

    assert simple_tag_spanish("dónde") == Tag.ADV
    assert SubType.RELATIVE in SPANISH_WORD_SUBTYPES["dónde"]


def test_tag_spanish_sentence_produces_verb_for_conjugated_forms():
    words = tag_spanish_sentence("mary está en el jardín .")
    by_text = {w.text: w for w in words}
    assert by_text["está"].pos == Tag.VERB
    assert by_text["el"].pos == Tag.DET
    assert by_text["jardín"].pos == Tag.NOUN
    assert by_text["."].pos == Tag.PUNCT


def test_tag_spanish_words_merges_lexicon_and_hand_dict():
    (w,) = tag_spanish_words(["jardín"])
    assert w.pos == Tag.NOUN
