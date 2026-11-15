"""Closed-class lexicon in USVS (proposal steps 1 and 3, owner decisions 2026-11-15).

Fast tier. Covers: 614 axes with the 607 base unchanged, lexicon-first grounding
("i" is not iodine), a handful of entries on the right dimensions, content words
byte-identical on the base axes, and the parser's tagger reading the same closed-class
data from quantum_parser/lexicons/<lang>.json as it did from its old Python tables.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_CT = Path(__file__).resolve().parent.parent
_QP = _CT.parent / "quantum_parser"
sys.path.insert(0, str(_QP / "src"))

from nsm_ct.ground.usvs import load_lexicon, load_usvs, save_usvs  # noqa: E402

_USVS_DIR = _CT / "data" / "usvs"
_BASE_FP = "e0daef638b640dd5"
_EXT = ["KNOWN", "UNKNOWN", "GOAL", "SOURCE", "AND", "MALE", "FEMALE"]


@pytest.fixture(scope="module")
def ext():
    return load_usvs(str(_USVS_DIR))


@pytest.fixture(scope="module")
def base():
    return load_usvs(str(_USVS_DIR), extended=False)


def _sig(u, word, pos=None, lang="en"):
    cands, _ = u.senses_of_surface(word, pos, lang=lang)
    assert cands, word
    return u.sense_signature(cands[0]), cands


# ---------------------------------------------------------------- axes / base
def test_614_axes_base_untouched(ext, base):
    assert len(base.axes) == 607 and len(ext.axes) == 614
    assert ext.axes[:607] == base.axes and ext.axes[607:] == _EXT
    assert ext.fingerprint == base.fingerprint == _BASE_FP
    assert ext.n_base_axes == 607
    assert np.array_equal(ext.core_coords[:, :607], base.core_coords)
    assert not ext.core_coords[:, 607:].any()          # content words: no value on extension axes


def test_ext_axes_are_data_and_nonnegative(ext):
    defs = json.loads((_USVS_DIR / "ext_axes.json").read_text())
    assert [a["name"] for a in defs["axes"]] == _EXT and all(a["meaning"] for a in defs["axes"])
    assert (ext.sense_axis_val >= 0).all()
    for lang in ("en", "es"):
        for word, e in load_lexicon(lang)["entries"].items():
            for r in e.get("readings", []):
                assert r["dims"] and all(d in ext.axes and 0 <= w <= 1 for d, w in r["dims"].items()), word


def test_content_words_byte_identical_on_base_axes(ext, base):
    for w in ("dog", "happy", "unhappy", "run", "justice", "bank", "iodine", "hot"):
        assert ext.senses_of(w) == base.senses_of(w)
        for sid in base.senses_of(w)[:4]:
            a, b = base.sense_dense(sid), ext.sense_dense(sid)
            assert a.tobytes() == b[:607].tobytes() and not b[607:].any()
        c = base.word_coord(w)
        if c is not None:
            assert c.tobytes() == ext.word_coord(w)[:607].tobytes()


def test_extended_usvs_cannot_overwrite_the_artifact(ext, tmp_path):
    with pytest.raises(ValueError):
        save_usvs(ext, tmp_path)


# ---------------------------------------------------------------- grounding
@pytest.mark.parametrize("word,pos,wn_homograph", [
    ("i", "PRON", "iodine.n.01"), ("he", "PRON", "helium.n.01"), ("a", "DET", "angstrom.n.01"),
    ("it", "PRON", "information_technology.n.01"), ("who", "PRON", "world_health_organization.n.01"),
    ("us", "PRON", "united_states.n.01"), ("am", "AUX", "americium.n.01"), ("at", "ADP", "astatine.n.01"),
    ("to", "ADP", "tho.n.01"), ("me", "PRON", "maine.n.01")])
def test_closed_class_words_are_not_homographs(ext, base, word, pos, wn_homograph):
    assert wn_homograph in base.senses_of_surface(word, pos)[0]      # the old behaviour
    cands, lemma = ext.senses_of_surface(word, pos)
    assert cands and all(c.startswith(f"fn.en.{word}.") for c in cands) and lemma == word
    assert not any(c == wn_homograph for c in cands)
    assert ext.senses_of_surface(word)[0][0].startswith("fn.en.")    # no POS hint: first reading


def test_other_readings_fall_through_to_wordnet(ext):
    assert ext.senses_of_surface("can", "NOUN")[0][0].startswith("can.n.")   # can of soup
    assert ext.senses_of_surface("can", "AUX")[0] == ["fn.en.can.1"]
    assert ext.senses_of_surface("dog", "NOUN")[0][0].startswith("dog.n.")


def test_language_selection(ext):
    assert ext.senses_of_surface("quién", "PRON", lang="es")[0][0].startswith("fn.es.quién.")
    assert not ext.senses_of_surface("el", "DET")[0][0].startswith("fn.es.")      # default lang is en
    assert ext.senses_of_surface("el", "DET", lang="es")[0] == ["fn.es.el.1"]


# ---------------------------------------------------------------- dimensions
@pytest.mark.parametrize("word,pos,dims", [
    ("not", "PART", {"NOT": 1.0}), ("n't", "PART", {"NOT": 1.0}),
    ("never", "ADV", {"NOT": 1.0, "WHEN": 1.0, "ALL": 1.0}),
    ("nobody", "PRON", {"SOMEONE": 1.0, "SOME": 1.0, "NOT": 1.0}),
    ("the", "DET", {"KNOWN": 1.0}), ("who", "PRON", {"SOMEONE": 1.0, "UNKNOWN": 1.0}),
    ("he", "PRON", {"SOMEONE": 1.0, "KNOWN": 1.0, "MALE": 1.0}),
    ("she", "PRON", {"SOMEONE": 1.0, "KNOWN": 1.0, "FEMALE": 1.0}),
    ("to", "ADP", {"GOAL": 1.0}), ("from", "ADP", {"SOURCE": 1.0}), ("in", "ADP", {"INSIDE": 1.0}),
    ("out", "ADV", {"INSIDE": 1.0, "SOURCE": 1.0}), ("and", "CCONJ", {"AND": 1.0}),
    ("will", "AUX", {"AFTER": 1.0}), ("did", "AUX", {"BEFORE": 1.0}), ("because", "SCONJ", {"BECAUSE": 1.0}),
    ("i", "PRON", {"I": 1.0}), ("you", "PRON", {"YOU": 1.0}), ("my", "DET", {"I": 1.0, "MINE": 1.0}),
])
def test_entries_land_on_the_right_dimensions(ext, word, pos, dims):
    sig, _ = _sig(ext, word, pos)
    assert sig == pytest.approx(dims)


def _cos(u, a, b):
    va, vb = u.sense_dense(_sig(u, *a)[1][0]), u.sense_dense(_sig(u, *b)[1][0])
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def test_formerly_indistinguishable_pairs_are_distinguishable(ext):
    for a, b in ((("in", "ADP"), ("out", "ADV")), (("to", "ADP"), ("from", "ADP")),
                 (("the", "DET"), ("a", "DET")), (("who", "PRON"), ("someone", "PRON")),
                 (("i", "PRON"), ("you", "PRON")), (("in", "ADP"), ("on", "ADP"))):
        assert _cos(ext, a, b) < 0.9, (a, b)
    assert _cos(ext, ("to", "ADP"), ("from", "ADP")) == 0.0 and _cos(ext, ("i", "PRON"), ("you", "PRON")) == 0.0
    n = ext.sense_dense(_sig(ext, "not", "PART")[1][0])
    assert np.linalg.norm(n) > 0 and (n[ext.axes.index("NOT")] == 1.0)    # not vs empty: a real point


def test_numerals_stay_names(ext):
    for w in ("one", "two", "three", "twenty"):
        assert "fn.en." not in "".join(ext.senses_of_surface(w)[0])


# ---------------------------------------------------------------- parser tables moved to data
_GOLD = json.loads((Path(__file__).parent / "data" / "pos_tagging_before.json").read_text())


def test_lexicon_tagging_equals_old_tagging_for_every_table_word():
    from parser import pos_tagger as P
    from parser.data_structures import Word
    for w, (tag, subs, possible) in _GOLD["words"].items():
        t = P.tag_words([w])[0]
        assert (P.simple_tag(w).name, [s.name for s in t.subtypes],
                [x.name for x in P.get_possible_tags(t)]) == (tag, subs, possible), w


def test_lexicon_tagging_equals_old_tagging_on_gold_sentences_and_spanish():
    from parser import pos_tagger as P
    for sent in _GOLD["sample"]:
        got = [[w.text, w.pos.name, [s.name for s in w.subtypes]] for w in P.tag_words([t[0] for t in sent])]
        assert got == sent
    got = {w.text: [w.pos.name, [s.name for s in w.subtypes]]
           for w in P.tag_spanish_words(list(_GOLD["es"]))}
    assert got == _GOLD["es"]


def test_tagger_tables_come_from_the_lexicon_files():
    from parser import pos_tagger as P
    for lang, tag_d, sub_d in (("en", P.WORD_TAG_DICT, P.WORD_SUBTYPES),
                               ("es", P.SPANISH_WORD_TAG_DICT, P.SPANISH_WORD_SUBTYPES)):
        entries = json.loads(Path(P.lexicon_path(lang)).read_text(encoding="utf-8"))["entries"]
        assert {w for w, e in entries.items() if "tag" in e} == set(tag_d)
        assert {w for w, e in entries.items() if "subtype" in e} == set(sub_d)
    src = (_QP / "src" / "parser" / "pos_tagger.py").read_text(encoding="utf-8")
    assert 'Tag.DET, "a"' not in src and '"the": Tag.DET' not in src and "SubType.NEGATIVE" not in src


def test_meaning_labels_are_gone_from_subtypes():
    from parser.enums import SubType
    names = {s.name for s in SubType}
    for old in ("NEGATIVE", "RELATIVE", "QUESTION", "MODAL", "PERFECT", "PROGRESSIVE",
                "POSSESSIVE", "INFINITIVE", "QUANTIFIER", "EXISTENTIAL"):
        assert old not in names, old
    for g in ("english", "spanish"):
        txt = (_QP / "grammars" / f"{g}.json").read_text(encoding="utf-8")
        for old in ("NEGATIVE", "RELATIVE", "QUESTION", "PROGRESSIVE"):
            assert f'"{old}"' not in txt
