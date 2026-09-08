"""Gate for the v4b fix (`dev/ENCODER_GOLD_V4B_SMALL_STATS.md`): v4-small's
`senses_of_surface` lemmatized every inflected surface form through
WordNet's `morphy` regardless of part of speech, which is POS-BLIND --
morphy has no notion of "this token is doing grammar, not naming a thing",
so a function word run through it grounds on whatever inflectional
coincidence WordNet's exception lists happen to produce (`was` -> morphy(n)
-> `wa` -> `washington.n.02`; `can` AUX -> its noun/container sense; `to`
ADP -> `tho.n.01`), and a content word run through it with the WRONG part
of speech can rank an off-POS sense first even when the raw surface already
grounds (`engraved` tagged VERB but the raw surface is itself only an
ADJECTIVE lemma, so it ranked `engraved.s.01` over any verb sense of
"engrave").

`senses_of_surface(word, pos_hint=...)` now takes the parser tagger's
universal-POS tag (`Tag` enum name) and:
  - never morphy-lemmatizes a FUNCTION tag (AUX/DET/ADP/PART/PRON/CCONJ/
    SCONJ/PUNCT/NUM/SYM/INTJ) -- raw surface only;
  - lemmatizes/ranks a content tag (NOUN/PROPN/VERB/ADJ/ADV) against the
    WordNet POS(es) that tag maps to, with candidates re-ordered (never
    dropped) so a POS-matched sense comes first;
  - with no pos_hint, keeps the legacy noun/verb/adj fallback order, plus a
    guard against an implausibly short morphy lemma recovered from a
    not-that-short surface (the `was` -> `wa` shape).
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

_USVS_DIR = _ROOT / "data" / "usvs"


def _skip_if_no_usvs():
    if not _USVS_DIR.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")


@pytest.fixture(scope="module")
def usvs():
    _skip_if_no_usvs()
    from nsm_ct.ground.usvs import load_usvs
    return load_usvs(str(_USVS_DIR))


def _wn_pos(sense_id: str) -> str:
    return sense_id.rsplit(".", 2)[1]


# ---------------------------------------------------------------------------
# Function words: never morphy-lemmatized
# ---------------------------------------------------------------------------

def test_was_aux_no_morphy(usvs):
    assert usvs.senses_of("was") == []  # raw surface: no direct WordNet lemma
    cands, lemma = usvs.senses_of_surface("was", pos_hint="AUX")
    # no morphy happened: either nothing grounds (entity fallback) or, if
    # something does, it's on the UNCHANGED raw surface -- never "wa".
    assert lemma == "was"
    assert cands == []


def test_can_aux_not_can_noun_first(usvs):
    cands, lemma = usvs.senses_of_surface("can", pos_hint="AUX")
    assert lemma == "can"
    assert cands  # "can" IS itself a raw WordNet lemma (container/verb)
    assert cands[0] != "can.n.01"


def test_to_adp_not_sense_grounded_via_morphy(usvs):
    """`to` raw-grounds to `tho.n.01` via the OMW Spanish-lemma widening
    (`to` is a Spanish lemma of that proper-noun synset -- see
    `spanish_lemmas`/`SPANISH_SWAP_FEASIBILITY.md`), NOT via morphy -- that
    is pre-existing, orthogonal behavior this fix does not touch. What v4's
    bug actually was: with pos_hint threaded through, an ADP token must
    never be run through morphy at all (the function-word guard), so
    `lemma_used` for "to" must stay "to" (the raw surface) under every POS
    hint -- never some morphy-recovered lemma."""
    cands, lemma = usvs.senses_of_surface("to", pos_hint="ADP")
    assert lemma == "to"
    assert cands == usvs.senses_of("to")  # raw surface only, no morphy detour


def test_function_tags_never_reach_morphy_even_when_raw_is_empty(usvs):
    from nsm_ct.ground.usvs import _FUNCTION_POS_TAGS
    for tag in sorted(_FUNCTION_POS_TAGS):
        cands, lemma = usvs.senses_of_surface("was", pos_hint=tag)
        assert lemma == "was", f"pos_hint={tag} morphy-lemmatized a function word"
        assert cands == []


# ---------------------------------------------------------------------------
# Content words: lemmatized/ranked against the tagger's own POS
# ---------------------------------------------------------------------------

def test_took_verb_lemmatizes_to_take_verb_first(usvs):
    assert usvs.senses_of("took") == []
    cands, lemma = usvs.senses_of_surface("took", pos_hint="VERB")
    assert lemma == "take"
    assert cands
    assert cands[0] == "take.v.01"
    assert cands == usvs.senses_of_surface("took", pos_hint="VERB")[0]


def test_beds_noun_lemmatizes_to_bed_noun_first(usvs):
    assert usvs.senses_of("beds") == []
    cands, lemma = usvs.senses_of_surface("beds", pos_hint="NOUN")
    assert lemma == "bed"
    assert cands
    assert cands[0] == "bed.n.01"


def test_engraved_verb_prefers_verb_sense_over_raw_adjective(usvs):
    """`engraved` IS its own raw WordNet lemma (`engraved.s.01`, an
    adjective) -- pre-fix this short-circuited to the adjective sense even
    when the tagger says VERB. The fix must pull in a POS-matched verb
    sense (recovered via morphy("engraved", "v") == "engrave") and rank it
    first, WITHOUT dropping the adjective candidate."""
    raw = usvs.senses_of("engraved")
    assert raw == ["engraved.s.01"]
    cands, lemma = usvs.senses_of_surface("engraved", pos_hint="VERB")
    assert lemma == "engraved"  # the raw surface is still the grounding entry point
    assert cands
    assert _wn_pos(cands[0]) == "v"
    assert cands[0] != "engraved.s.01"
    assert "engraved.s.01" in cands  # candidates-first: nothing dropped


def test_pos_hint_none_keeps_legacy_order(usvs):
    """No pos_hint: unchanged legacy behavior for a word whose raw surface
    already grounds (no reordering, no morphy)."""
    for w in ("closed", "cut", "wanted"):
        raw = usvs.senses_of(w)
        assert raw
        cands, lemma = usvs.senses_of_surface(w)
        assert lemma == w
        assert cands == raw


def test_pos_hint_none_was_still_guarded_against_wa(usvs):
    """Even with no pos_hint (the D5/legacy caller shape), the short-lemma
    guard still blocks `was` -> morphy(n) -> `wa` -> `washington.n.02`."""
    cands, lemma = usvs.senses_of_surface("was")
    assert lemma != "wa"
    assert "washington.n.02" not in cands


# ---------------------------------------------------------------------------
# Regression: dogs/walked still lemmatize (D5 path unaffected by pos_hint)
# ---------------------------------------------------------------------------

def test_dogs_and_walked_still_lemmatize_without_pos_hint(usvs):
    cands, lemma = usvs.senses_of_surface("dogs")
    assert lemma == "dog"
    assert cands == usvs.senses_of("dog")
    cands, lemma = usvs.senses_of_surface("walked")
    assert lemma == "walk"
    assert cands == usvs.senses_of("walk")


def test_dogs_noun_and_walked_verb_lemmatize_with_pos_hint(usvs):
    cands, lemma = usvs.senses_of_surface("dogs", pos_hint="NOUN")
    assert lemma == "dog"
    assert cands
    cands, lemma = usvs.senses_of_surface("walked", pos_hint="VERB")
    assert lemma == "walk"
    assert cands
    assert _wn_pos(cands[0]) == "v"
