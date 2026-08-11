"""Spanish word -> meaning tree, via OMW-es -> Princeton synset -> the SAME
English-gloss decomposition :class:`~nsm_ct.meaning.NSMMeaningResolver`
already does.

The Spanish Freeze Test's central mechanism (dev/ROADMAP_LONG_TERM.md): sense
signatures are keyed to SYNSET IDs, not English strings -- "jardín" and
"garden" both resolve (via OMW's Spanish lemma layer) to ``garden.n.01``, so
they should produce the SAME meaning tree. ``NSMMeaningResolver``'s WordNet
step (:func:`nsm_ct.wordnet.senses`) is however English-string-keyed
(``wn.synsets(word)``, no ``lang=``) -- handed a raw Spanish word it finds
nothing and falls to the generic SOMETHING/SOMEONE fallback, losing all
content. :class:`SpanishMeaningResolver`'s fix is a single translation step
BEFORE handing off to the untouched parent logic: look up the word's OMW-es
synsets, take the first (index 0 -- OMW's own MFS-like order, NOT
necessarily English WordNet's own MFS order for the same concept; this is
the "expected leak" dev/ROADMAP_LONG_TERM.md calls out, quantified by
``scripts/probe_spanish_freeze.py``), and resolve using that synset's own
first English lemma -- reusing 100% of ``NSMMeaningResolver``'s prime/
molecule/explication-store/gloss-decomposition machinery unchanged (no
clause.py, meaning.py, or wordnet.py edits; this is a NEW file).

Two things are deliberately left UNTRANSLATED by this resolver:

- **Entity names** (mary/john/...): the Spanish curriculum reuses the exact
  same English name strings as proper nouns (see curriculum2.py's Spanish
  templates), precisely so clause.py's English-only entity/pronoun tables
  (``is_entity``/``_PRONOUNS``/``_ENTITY_NAMES``) need no edit -- ``resolve``
  is never even called for them (clause.py routes entities through
  ``codec.filler_vec("var:"+name)``, not through the resolver).
- **Spanish verb forms** (está/fue/dio/tomó/encontró): these are inflected;
  OMW's Spanish lemma layer indexes infinitives, so ``wn.synsets("fue",
  lang="spa")`` returns nothing. A tiny hand lemmatizer (the same closed set
  of conjugated forms ``quantum_parser/src/parser/pos_tagger.py``'s
  ``SPANISH_WORD_TAG_DICT`` already hand-lists -- M41's "closed class, not an
  open lexicon" rationale) maps them to their infinitive before the synset
  lookup; unknown inflections fall through to the parent's ordinary fallback,
  exactly like an unknown English word would.
"""

from __future__ import annotations

from typing import Dict, Optional

from .data_structures import ParseTree
from .meaning import NSMMeaningResolver

# Closed-set conjugated-verb -> infinitive lemmatizer (mirrors the hand-listed
# forms in quantum_parser/src/parser/pos_tagger.py's SPANISH_WORD_TAG_DICT --
# the same "closed class, not an open lexicon" rationale M41 established for
# English). Only the forms the Spanish curriculum templates actually use.
_VERB_LEMMA: Dict[str, str] = {
    "fui": "ir", "fuiste": "ir", "fue": "ir",
    "fuimos": "ir", "fuisteis": "ir", "fueron": "ir",
    "encontré": "encontrar", "encontraste": "encontrar", "encontró": "encontrar",
    "encontramos": "encontrar", "encontrasteis": "encontrar", "encontraron": "encontrar",
    "di": "dar", "diste": "dar", "dio": "dar",
    "dimos": "dar", "disteis": "dar", "dieron": "dar",
    "tomé": "tomar", "tomaste": "tomar", "tomó": "tomar",
    "tomamos": "tomar", "tomasteis": "tomar", "tomaron": "tomar",
    "soy": "ser", "eres": "ser", "es": "ser",
    "somos": "ser", "sois": "ser", "son": "ser",
    "estoy": "estar", "estás": "estar", "está": "estar",
    "estamos": "estar", "estáis": "estar", "están": "estar",
}

# Closed-class function words with no translatable content (determiners;
# "al"/"del"/"a"/"de"/"en" are prepositions, resolved to roles by
# input_encoder._install_spanish_prep_relation, never passed to this
# resolver as a content word). Kept tiny and explicit rather than guessed.
_STOP = frozenset({"el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "no"})


class SpanishMeaningResolver(NSMMeaningResolver):
    """Resolves a Spanish surface word by translating it to its Princeton
    WordNet synset (via OMW-es) FIRST, then delegating entirely to the
    parent's unmodified English pipeline for that synset's own first lemma.
    See the module docstring for the full mechanism and its two documented
    gaps (entities, verb inflection).
    """

    def resolve(self, word: str, context: object = None) -> ParseTree:
        key = word.lower().strip()
        if key not in _STOP:
            eng = self._translate(key)
            if eng is not None and eng != key:
                return super().resolve(eng, context=context)
        return super().resolve(key, context=context)

    def _translate(self, key: str) -> Optional[str]:
        """``key`` (Spanish surface form) -> an English lemma sharing its
        OMW-es sense-0 synset, or ``None`` if OMW has no Spanish entry for it
        at all (reported, not silently swallowed, by
        ``scripts/probe_spanish_freeze.py``'s coverage table)."""
        from .wordnet import wordnet_available

        if not wordnet_available():
            return None
        try:
            from nltk.corpus import wordnet as wn
        except Exception:  # pragma: no cover - nltk always present if wordnet_available()
            return None
        lemma_key = _VERB_LEMMA.get(key, key)
        try:
            synsets = wn.synsets(lemma_key, lang="spa")
        except Exception:
            return None
        if not synsets:
            return None
        syn = synsets[0]  # OMW-es's own sense-0 order -- the MFS-ordering leak
        lemmas = syn.lemmas()
        if not lemmas:
            return None
        return lemmas[0].name().replace("_", " ").lower()
