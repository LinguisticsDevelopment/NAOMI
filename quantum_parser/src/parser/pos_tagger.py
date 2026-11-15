"""
Simple POS tagger for automatic word tagging.

Two layers:

1. ``WORD_TAG_DICT`` / ``WORD_SUBTYPES`` — CLOSED-CLASS entries (determiners,
   pronouns, auxiliaries, prepositions, conjunctions), loaded from
   ``quantum_parser/lexicons/<lang>.json`` (data, shared with USVS). These are
   finite classes WordNet doesn't cover; they keep top precedence.
2. ``data/en_lexicon.json.gz`` — the generated OPEN-CLASS lexicon (nouns,
   verbs, adjectives, adverbs + inflections with morphological subtypes),
   built from WordNet by consciousness_transformer/scripts/
   build_parser_lexicon.py. Multi-POS words expose every tag to the
   hypothesis lattice via get_possible_tags (frequency-ordered, entry [0] is
   the context-free default); the scorer picks the reading that completes a
   tree. If the file is absent, behavior degrades to the old suffix
   heuristics — never an error.

M58c (real-text parser round, dev/PROSE_FAILURE_TAXONOMY.md) adds three
MORE fallback tiers below the static lexicon, all inside
``lexicon_entry`` so every existing caller (this tagger's own
``simple_tag``/``get_possible_tags`` AND ``nsm_ct.corpus._is_known_word``,
which calls ``lexicon_entry`` directly) benefits with no other code
changes. Deliberately kept OUT of build_parser_lexicon.py / the static
``.json.gz`` file: tests/test_spanish_freeze.py pins the generated
English lexicon's exact fingerprint+count, so any change to what that
BUILD SCRIPT emits would be a regression by definition -- these tiers are
pure runtime fallback, consulted only when the static file (and the hand
dict) both miss:

  (a) hyphenated compounds -- split on "-", look the TAIL token up through
      this same chain (so "pine-tree" resolves via "tree", "brown-coated"
      via "coated" -- recursive but the tail never itself contains "-", so
      one level deep always terminates);
  (b) on-demand WordNet consult (``_wordnet_entry``) -- nltk is optional
      and imported lazily; a handful of cheap de-inflection guesses (plural
      -s/-es, verb -ed/-ing, comparative/superlative -er/-est/-ier/-iest)
      are tried against WordNet when the bare word isn't a lemma itself
      (WordNet only stores base forms) -- this is what recovers real
      inflected open-class vocabulary the static generator's own
      inflection rules didn't produce for (e.g. multi-syllable -en/-er
      verbs, where the generator's naive CVC-doubling heuristic is wrong:
      "happened"/"listened"/"clattered" -- a real, separate bug, left
      exactly as-is in build_parser_lexicon.py for the fingerprint-pin
      reason above; this runtime tier is the actual fix);
  (c) the NAME/NUM fallback (``is_bare_name_token`` / ``looks_like_number``)
      -- a token with NO coverage anywhere above AND no inflectional shape
      (doesn't end -ly/-ing/-ed/-s) is overwhelmingly a proper name in real
      narrative prose (measured: 9 of 11 residual unknown-word tokens in
      the Buster Bear sample are character names -- alice, sammy, blacky,
      joe, kate, ...), so it is treated as known/NAME rather than failing;
      a digit sequence is NUM. Genuine unknown vocabulary that DOES carry
      an inflectional suffix (rare archaic words, typos) still falls
      through to the old suffix heuristics/default-noun path untouched, so
      the "unknown-word" signal keeps catching that class.
"""

import gzip
import json
import os
import string
from typing import Dict, List, Optional, Tuple

from .data_structures import Word
from .enums import Tag, SubType

_LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "data", "en_lexicon.json.gz")
_lexicon: Optional[Dict[str, List[Tuple[Tag, List[SubType]]]]] = None

# Spanish open-class lexicon (M41 recipe pointed at OMW-es; see
# consciousness_transformer/scripts/build_parser_lexicon.py --lang spa).
# Separate cache/path from the English lexicon above -- loaded lazily and
# independently, so a process that never touches Spanish never pays for it
# and English behavior/timing is untouched.
_ES_LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "data", "es_lexicon.json.gz")
_es_lexicon: Optional[Dict[str, List[Tuple[Tag, List[SubType]]]]] = None


def _load_lexicon_from(path: str) -> Dict[str, List[Tuple[Tag, List[SubType]]]]:
    """Shared loader body for the English/Spanish open-class lexicons.

    Enum names unknown to this build (schema drift) are skipped per-entry
    rather than failing the load. Missing/corrupt file -> empty table (the
    caller's tagger degrades to its heuristics, never an error)."""
    table: Dict[str, List[Tuple[Tag, List[SubType]]]] = {}
    try:
        with gzip.open(path, "rb") as f:
            raw = json.load(f)
        for word, entries in raw.items():
            parsed = []
            for tag_name, subtype_names in entries:
                try:
                    parsed.append((Tag[tag_name],
                                   [SubType[s] for s in subtype_names]))
                except KeyError:
                    continue
            if parsed:
                table[word] = parsed
    except (OSError, ValueError):
        table = {}
    return table


def _load_lexicon() -> Dict[str, List[Tuple[Tag, List[SubType]]]]:
    """Lazy-load the generated English open-class lexicon (stdlib only, one time)."""
    global _lexicon
    if _lexicon is None:
        _lexicon = _load_lexicon_from(_LEXICON_PATH)
    return _lexicon


def _load_es_lexicon() -> Dict[str, List[Tuple[Tag, List[SubType]]]]:
    """Lazy-load the generated Spanish open-class lexicon (stdlib only, one time)."""
    global _es_lexicon
    if _es_lexicon is None:
        _es_lexicon = _load_lexicon_from(_ES_LEXICON_PATH)
    return _es_lexicon


def lexicon_entry(text_lower: str) -> Optional[List[Tuple[Tag, List[SubType]]]]:
    """The word's open-class lexicon entries (frequency-ordered), or None.

    M58c: three additive fallback tiers run when the exact word is absent
    from the static generated lexicon (see the module docstring's "M58c"
    section for why these live here rather than in the static file) --
    (1) hyphenated-compound tail lookup, (2) on-demand WordNet consult with
    light de-inflection. Each returns the SAME ``[(Tag, [SubType, ...]), ...]``
    shape as a real static-lexicon hit, so every caller (this tagger's
    ``simple_tag``/``get_possible_tags`` and ``nsm_ct.corpus._is_known_word``)
    is none the wiser. Tier (3), the bare NAME/NUM fallback, does NOT fit
    this shape (it isn't a WordNet POS reading) -- see
    :func:`is_bare_name_token`/:func:`looks_like_number`, consulted
    separately by ``simple_tag`` and by callers that want the "is this word
    covered at all" answer without a specific tag.
    """
    entry = _load_lexicon().get(text_lower)
    if entry:
        return entry
    if "-" in text_lower:
        head, _sep, tail = text_lower.rpartition("-")
        if tail and head:
            entry = lexicon_entry(tail)  # tail never contains "-": terminates in one recursion
            if entry:
                return entry
    return _wordnet_entry(text_lower)


# -- M58c tier (b): on-demand WordNet consult (lazy nltk import; see module
#    docstring) ----------------------------------------------------------
_wn_module = None          # lazily bound to nltk.corpus.wordnet, or False if unavailable
_wn_entry_cache: Dict[str, Optional[List[Tuple[Tag, List[SubType]]]]] = {}

_WN_POS_TO_TAG = {"n": Tag.NOUN, "v": Tag.VERB, "a": Tag.ADJ, "s": Tag.ADJ, "r": Tag.ADV}


def _get_wn():
    global _wn_module
    if _wn_module is None:
        try:
            from nltk.corpus import wordnet as wn  # local import: optional dependency
            wn.synsets("test")  # touch the data; raises if the corpus isn't installed
            _wn_module = wn
        except Exception:
            _wn_module = False
    return _wn_module or None


def _wn_direct(word: str) -> Optional[List[Tuple[Tag, List[SubType]]]]:
    """``word`` looked up AS-IS against WordNet (no de-inflection); tags
    ordered by total lemma frequency (SemCor ``lemma.count()``, same
    frequency signal build_parser_lexicon.py's static generation uses),
    highest first -- deterministic given a fixed WordNet install."""
    wn = _get_wn()
    if wn is None:
        return None
    try:
        synsets = wn.synsets(word)
    except Exception:
        return None
    freq: Dict[Tag, int] = {}
    for syn in synsets:
        tag = _WN_POS_TO_TAG.get(syn.pos())
        if tag is None:
            continue
        n = sum(lem.count() for lem in syn.lemmas() if lem.name().lower() == word)
        freq[tag] = freq.get(tag, 0) + n + 1  # +1: a sense with zero SemCor count still counts
    if not freq:
        return None
    ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0].name))
    return [(tag, []) for tag, _n in ordered]


# De-inflection guesses tried (in order) when the bare word isn't a WordNet
# lemma itself -- WordNet only stores base/dictionary forms. Each entry is
# (suffix, replacement, forced_tag): forced_tag overrides whatever POS the
# stem's OWN senses would rank highest, because the SUFFIX is the actual
# evidence for what part of speech this occurrence is (a comparative/
# superlative reads ADJ regardless of "deep"'s own noun senses).
_DEINFLECT = (
    ("iest", "y", Tag.ADJ), ("ier", "y", Tag.ADJ),
    ("est", "", Tag.ADJ), ("er", "", None),
    ("ies", "y", None), ("ied", "y", Tag.VERB),
    ("ing", "", Tag.VERB), ("ing", "e", Tag.VERB),
    ("ed", "", Tag.VERB), ("ed", "e", Tag.VERB),
    ("es", "", None), ("s", "", None),
)


def _wordnet_entry(word: str) -> Optional[List[Tuple[Tag, List[SubType]]]]:
    """M58c tier (b): the word as-is, then de-inflected, against WordNet."""
    if not word or not word.isalpha():
        return None
    if word in _wn_entry_cache:
        return _wn_entry_cache[word]
    entry = _wn_direct(word)
    if entry is None:
        for suffix, replacement, forced_tag in _DEINFLECT:
            if not word.endswith(suffix) or len(word) <= len(suffix) + 1:
                continue
            stem = word[: -len(suffix)] + replacement
            if len(stem) < 2:
                continue
            hit = _wn_direct(stem)
            if hit:
                entry = [(forced_tag, [])] if forced_tag is not None else hit
                break
    _wn_entry_cache[word] = entry
    return entry


def looks_like_number(text_lower: str) -> bool:
    """True for a bare digit sequence (optionally with internal ``,``/``.``/
    ``-``/``/`` -- plain numbers, decimals, simple dates) -- M58c's NUM tier."""
    core = text_lower.replace(",", "").replace(".", "").replace("-", "").replace("/", "")
    return core.isdigit() and len(core) > 0


def is_bare_name_token(text_lower: str) -> bool:
    """M58c tier (c): true for a word this tagger has NO real coverage for
    (hand dict, static lexicon, hyphen-compound, on-demand WordNet) AND
    that matches none of the inflectional-suffix heuristics either
    (-ly/-ing/-ed/-s) -- a bare root-looking token. In real narrative prose
    this is overwhelmingly a proper name (character names recur constantly;
    see the module docstring's measured count) rather than genuine unknown
    vocabulary, so callers may treat it as known/NAME rather than failing.
    Shared by :func:`simple_tag` (tags it PROPN) and
    ``nsm_ct.corpus._is_known_word`` (counts it as known) -- one source of
    truth for the boundary between the two.
    """
    w = text_lower
    if not w or not w.isalpha():
        return False
    if w in WORD_TAG_DICT or w in AMBIGUOUS_WORDS:
        return False
    if lexicon_entry(w):
        return False
    if w.endswith(("ly", "ing", "ed")):
        return False
    if w.endswith("s") and not w.endswith("ss"):
        return False
    return True


def lexicon_subtypes(text_lower: str, tag: Tag) -> List[SubType]:
    """Morphological subtypes the lexicon assigns to ``text_lower`` read as
    ``tag`` (e.g. broken/VERB -> [PAST_PARTICIPLE]); [] if unknown."""
    entry = lexicon_entry(text_lower)
    if entry:
        for etag, subs in entry:
            if etag == tag:
                return list(subs)
    return []


def es_lexicon_entry(text_lower: str) -> Optional[List[Tuple[Tag, List[SubType]]]]:
    """The word's Spanish open-class lexicon entries (sense-count-ordered), or None."""
    return _load_es_lexicon().get(text_lower)


def es_lexicon_subtypes(text_lower: str, tag: Tag) -> List[SubType]:
    """Morphological subtypes the Spanish lexicon assigns to ``text_lower`` read
    as ``tag``; [] if unknown. Mirrors :func:`lexicon_subtypes`."""
    entry = es_lexicon_entry(text_lower)
    if entry:
        for etag, subs in entry:
            if etag == tag:
                return list(subs)
    return []


# Closed-class tables (word -> tag / ambiguous tags / subtypes) are DATA: one lexicon file per
# language at quantum_parser/lexicons/<lang>.json, shared with USVS (its `readings[].dims` give
# each word a point on named USVS axes; the parser reads only tag/ambiguous/subtype). No word
# list lives in this module. A missing/corrupt lexicon is an error, not a silent fallback.
_LEXICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "lexicons")


def lexicon_path(lang: str) -> str:
    """Path of the closed-class lexicon file for *lang* ("en", "es")."""
    return os.path.join(_LEXICONS_DIR, f"{lang}.json")


def _load_closed_class(lang: str):
    """(tag dict, ambiguous-tags dict, subtype dict) from the language's lexicon file.

    ``tag``: the context-free tag (was WORD_TAG_DICT); ``ambiguous``: ordered tag list exposed to the
    hypothesis lattice (was AMBIGUOUS_WORDS); ``subtype``: closed-class subtypes (was WORD_SUBTYPES).
    Entries may carry only some of the three; ``readings`` (USVS dims) are ignored here."""
    with open(lexicon_path(lang), encoding="utf-8") as f:
        raw = json.load(f)
    tags: Dict[str, Tag] = {}
    amb: Dict[str, List[Tag]] = {}
    subs: Dict[str, List[SubType]] = {}
    for word, e in raw["entries"].items():
        if "tag" in e:
            tags[word] = Tag[e["tag"]]
        if "ambiguous" in e:
            amb[word] = [Tag[t] for t in e["ambiguous"]]
        if "subtype" in e:
            subs[word] = [SubType[s] for s in e["subtype"]]
    return tags, amb, subs


WORD_TAG_DICT, AMBIGUOUS_WORDS, WORD_SUBTYPES = _load_closed_class("en")
SPANISH_WORD_TAG_DICT, _ES_AMBIGUOUS_WORDS, SPANISH_WORD_SUBTYPES = _load_closed_class("es")

# Inverted question/exclamation marks (Spanish-only; not in string.punctuation,
# which is ASCII-only) -- must be recognized as PUNCT, same reason English's
# simple_tag has an all-punctuation guard (see its comment): an unrecognized
# "¿" would default through every heuristic below to NOUN and become a
# phantom NOMINAL that can wrongly satisfy a rule's argument slot.
_ES_PUNCT_EXTRA = "¿¡"


def get_possible_tags(word: Word) -> List[Tag]:
    """
    Get all possible POS tags for a word.

    Args:
        word: Word object to check

    Returns:
        List of possible POS tags (includes current tag if not in ambiguous dict)
    """
    text_lower = word.text.lower()

    # Check if word is in ambiguous dictionary
    if text_lower in AMBIGUOUS_WORDS:
        return AMBIGUOUS_WORDS[text_lower].copy()

    # Closed-class hand entries are authoritative: no branching
    if text_lower in WORD_TAG_DICT:
        return [word.pos]

    # Open-class lexicon: expose every WordNet POS to the lattice
    entry = lexicon_entry(text_lower)
    if entry and len(entry) > 1:
        tags = [t for t, _subs in entry]
        if word.pos not in tags:  # e.g. PROPN from capitalization heuristic
            tags.append(word.pos)
        return tags

    # Not ambiguous - return current tag as only option
    return [word.pos]


def simple_tag(text: str) -> Tag:
    """
    Tag a single word using simple rules.

    Args:
        text: Word to tag

    Returns:
        POS tag
    """
    text_lower = text.lower()

    # Punctuation (".", "?", "!", ",", etc.) -> PUNCT, never an open-class
    # fallback. Without this a sentence-final "." falls through every
    # heuristic below to the bare "Default: noun" case, becomes a phantom
    # NOMINAL, and can wrongly satisfy a rule's "NOMINAL after" pattern
    # (e.g. "is ... ." reading "." as its direct object) -- a spurious
    # match that collides with an unrelated anchor elsewhere in the
    # sentence and derails quantum branching (see aux1/predicate1 subord
    # interaction, M-round subordinate-clause fix).
    if text and all(ch in string.punctuation for ch in text):
        return Tag.PUNCT

    # Check dictionary
    if text_lower in WORD_TAG_DICT:
        return WORD_TAG_DICT[text_lower]

    # Capitalized words (not at start) → Proper noun. Checked BEFORE the
    # lexicon so names that are also common words ("Mark", "Bill") stay PROPN.
    if text[0].isupper() and text not in ["I", "A"]:
        return Tag.PROPN

    # Open-class lexicon (generated from WordNet, PLUS the M58c hyphen-
    # compound/on-demand-WordNet fallback tiers -- see lexicon_entry's own
    # docstring): entry [0] is the frequency-ordered context-free default;
    # the lattice can still branch to the other tags via get_possible_tags.
    entry = lexicon_entry(text_lower)
    if entry:
        return entry[0][0]

    # M58c: a bare digit sequence/date -> NUM, never an open-class fallback.
    if looks_like_number(text_lower):
        return Tag.NUM

    # Simple heuristics

    # Ends in -ly → probably adverb
    if text_lower.endswith("ly"):
        return Tag.ADV

    # Ends in -ing → verb or adjective
    if text_lower.endswith("ing"):
        return Tag.VERB

    # Ends in -ed → verb
    if text_lower.endswith("ed"):
        return Tag.VERB

    # Ends in -s (but not -ss) → could be verb or plural noun
    if text_lower.endswith("s") and not text_lower.endswith("ss"):
        # Default to noun (plural)
        return Tag.NOUN

    # M58c: a word with NO coverage anywhere above and no recognizable
    # inflectional shape either -> assume proper NAME (see
    # is_bare_name_token's docstring; this is the tail of the SAME check,
    # just re-derived here since we've already consumed the suffix cases
    # above rather than calling the standalone helper twice).
    if text_lower.isalpha():
        return Tag.PROPN

    # Default: noun
    return Tag.NOUN


def resolve_quote_direction(words: List[Word]) -> None:
    """Give each bare quotation mark (lexicon subtype QUOTE_MARK) a direction, in place.

    Language-neutral: no character is named here; a token is a quotation mark because its lexicon
    entry says so. Per mark string, marks alternate open / close in sentence order. With an odd
    count the sentence is a fragment of a longer quotation: if the first mark opens the sentence it
    is an opener, otherwise if the last mark is followed only by sentence-final marks it is a
    closer (so the run starts with a closer); otherwise the run starts with an opener."""
    by_mark: Dict[str, List[int]] = {}
    for i, w in enumerate(words):
        if SubType.QUOTE_MARK in w.subtypes:
            by_mark.setdefault(w.text, []).append(i)
    for idxs in by_mark.values():
        opens = True
        if len(idxs) % 2 == 1 and idxs[0] != 0:
            tail = words[idxs[-1] + 1:]
            if all(SubType.TERMINAL in t.subtypes for t in tail):
                opens = False
        for i in idxs:
            words[i].subtypes.append(SubType.QUOTE_OPEN if opens else SubType.QUOTE_CLOSE)
            opens = not opens


def tag_sentence(sentence: str) -> List[Word]:
    """
    Tag an entire sentence.

    Args:
        sentence: Input sentence as string

    Returns:
        List of Word objects with POS tags and subtypes
    """
    # Simple tokenization (split on whitespace)
    tokens = sentence.split()

    words = []
    for token in tokens:
        if token.strip():  # Skip empty tokens
            tag = simple_tag(token)

            # Look up subtypes from dictionary
            subtypes = WORD_SUBTYPES.get(token.lower(), []).copy()

            # Add suffix-based subtypes
            token_lower = token.lower()

            # Morphological subtypes from the open-class lexicon
            for st in lexicon_subtypes(token_lower, tag):
                if st not in subtypes:
                    subtypes.append(st)

            # -ing suffix → PARTICIPLE (for present participles)
            if token_lower.endswith("ing") and tag == Tag.VERB:
                if SubType.PARTICIPLE not in subtypes:
                    subtypes.append(SubType.PARTICIPLE)

            words.append(Word(token, tag, subtypes))

    resolve_quote_direction(words)
    return words


def tag_words(text_list: List[str]) -> List[Word]:
    """
    Tag a list of word strings.

    Args:
        text_list: List of word strings

    Returns:
        List of Word objects with POS tags and subtypes
    """
    words = []
    for text in text_list:
        tag = simple_tag(text)
        subtypes = WORD_SUBTYPES.get(text.lower(), []).copy()

        # Add suffix-based subtypes
        text_lower = text.lower()

        # Morphological subtypes from the open-class lexicon
        for st in lexicon_subtypes(text_lower, tag):
            if st not in subtypes:
                subtypes.append(st)

        if text_lower.endswith("ing") and tag == Tag.VERB:
            if SubType.PARTICIPLE not in subtypes:
                subtypes.append(SubType.PARTICIPLE)

        words.append(Word(text, tag, subtypes))

    return words


# ========== SPANISH TAGGING FUNCTIONS ==========

def simple_tag_spanish(text: str) -> Tag:
    """
    Tag a single Spanish word using simple rules.

    Args:
        text: Spanish word to tag

    Returns:
        POS tag
    """
    text_lower = text.lower()

    # Punctuation (incl. inverted ¿/¡) -> PUNCT, never an open-class fallback.
    # See the module-level guard note by _ES_PUNCT_EXTRA and simple_tag's own
    # comment for why this must run before every other branch.
    if text and all(ch in string.punctuation or ch in _ES_PUNCT_EXTRA for ch in text):
        return Tag.PUNCT

    # Check Spanish dictionary
    if text_lower in SPANISH_WORD_TAG_DICT:
        return SPANISH_WORD_TAG_DICT[text_lower]

    # Capitalized words (not sentence-initial in practice; mirrors English's
    # simple_tag PROPN guard) → Proper noun. Checked BEFORE the lexicon so
    # names that are also common words stay PROPN.
    if text[0].isupper():
        return Tag.PROPN

    # Open-class Spanish lexicon (generated from OMW-es): entry [0] is the
    # sense-count-ordered default (see build_parser_lexicon.py --lang spa's
    # docstring for why this is a weaker MFS proxy than English's frequency
    # order).
    entry = es_lexicon_entry(text_lower)
    if entry:
        return entry[0][0]

    # Spanish heuristics (unchanged fallback if the lexicon is absent/misses)
    # Ends in -mente → adverb (Spanish)
    if text_lower.endswith("mente"):
        return Tag.ADV

    # Ends in -ar, -er, -ir → verb infinitive
    if text_lower.endswith(("ar", "er", "ir")):
        return Tag.VERB

    # Ends in -ando, -iendo → gerund (participle)
    if text_lower.endswith(("ando", "iendo")):
        return Tag.VERB

    # Ends in -ado, -ido → past participle
    if text_lower.endswith(("ado", "ido")):
        return Tag.VERB

    # Default: noun
    return Tag.NOUN


def tag_spanish_sentence(sentence: str) -> List[Word]:
    """
    Tag a Spanish sentence.

    Args:
        sentence: Input Spanish sentence as string

    Returns:
        List of Word objects with POS tags and subtypes
    """
    # Simple tokenization (split on whitespace)
    tokens = sentence.split()

    words = []
    for token in tokens:
        if token.strip():  # Skip empty tokens
            tag = simple_tag_spanish(token)

            # Look up subtypes from Spanish dictionary
            subtypes = SPANISH_WORD_SUBTYPES.get(token.lower(), []).copy()

            # Add suffix-based subtypes for Spanish
            token_lower = token.lower()

            # Morphological subtypes from the open-class Spanish lexicon
            for st in es_lexicon_subtypes(token_lower, tag):
                if st not in subtypes:
                    subtypes.append(st)

            # -ando/-iendo suffix → PARTICIPLE (gerund)
            if token_lower.endswith(("ando", "iendo")) and tag == Tag.VERB:
                if SubType.PARTICIPLE not in subtypes:
                    subtypes.append(SubType.PARTICIPLE)

            # -ado/-ido suffix → PARTICIPLE (past participle)
            if token_lower.endswith(("ado", "ido")) and tag == Tag.VERB:
                if SubType.PARTICIPLE not in subtypes:
                    subtypes.append(SubType.PARTICIPLE)

            words.append(Word(token, tag, subtypes))

    return words


def tag_spanish_words(text_list: List[str]) -> List[Word]:
    """
    Tag a list of Spanish word strings.

    Args:
        text_list: List of Spanish word strings

    Returns:
        List of Word objects with POS tags and subtypes
    """
    words = []
    for text in text_list:
        tag = simple_tag_spanish(text)
        subtypes = SPANISH_WORD_SUBTYPES.get(text.lower(), []).copy()

        # Add suffix-based subtypes
        text_lower = text.lower()

        # Morphological subtypes from the open-class Spanish lexicon
        for st in es_lexicon_subtypes(text_lower, tag):
            if st not in subtypes:
                subtypes.append(st)

        if text_lower.endswith(("ando", "iendo", "ado", "ido")) and tag == Tag.VERB:
            if SubType.PARTICIPLE not in subtypes:
                subtypes.append(SubType.PARTICIPLE)

        words.append(Word(text, tag, subtypes))

    return words


# Optional: Try to import spaCy for better tagging
try:
    import spacy
    _nlp = None

    def tag_sentence_spacy(sentence: str) -> List[Word]:
        """
        Tag sentence using spaCy (if available).

        Args:
            sentence: Input sentence

        Returns:
            List of Word objects
        """
        global _nlp
        if _nlp is None:
            _nlp = spacy.load("en_core_web_sm")

        doc = _nlp(sentence)

        # Map spaCy tags to our Tag enum
        SPACY_TAG_MAP = {
            "DET": Tag.DET,
            "NOUN": Tag.NOUN,
            "PROPN": Tag.PROPN,
            "VERB": Tag.VERB,
            "AUX": Tag.AUX,
            "ADJ": Tag.ADJ,
            "ADV": Tag.ADV,
            "ADP": Tag.ADP,
            "CCONJ": Tag.CCONJ,
            "SCONJ": Tag.SCONJ,
            "PRON": Tag.PRON,
            "NUM": Tag.NUM,
            "PART": Tag.PART,
            "INTJ": Tag.INTJ,
            "PUNCT": Tag.PUNCT,
            "SYM": Tag.SYM,
            "X": Tag.X,
        }

        words = []
        for token in doc:
            tag = SPACY_TAG_MAP.get(token.pos_, Tag.NOUN)
            words.append(Word(token.text, tag))

        return words

    # Use spaCy by default if available
    tag_sentence = tag_sentence_spacy

except ImportError:
    # spaCy not available, use simple tagger
    pass
