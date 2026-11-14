"""Build quantum_parser's open-class tag lexicon from WordNet.

The parser's WORD_TAG_DICT was ~200 hand-listed words — a bug class, not a
lexicon (M39: its missing entries were the single biggest failure bucket).
This script generates the open-class lexicon (nouns/verbs/adjectives/adverbs)
from WordNet, which the project already owns as USVS's source vocabulary:

- every single-word lemma, every POS it can be, **ordered by corpus
  frequency** (SemCor lemma counts, sense-count fallback) so entry [0] is the
  best context-free tag;
- inflected forms generated from base lemmas (regular rules + WordNet's
  exception lists for irregulars), carrying morphological subtypes:
  plural nouns [PLURAL], 3sg verbs [THIRD_PERSON, SINGULAR], -ing verbs
  [PARTICIPLE] (matches the existing ger1/part1 machinery), past/-ed verbs
  [PAST_PARTICIPLE] (what the aux1 passive rule anchors on; for irregulars
  WordNet's exc files don't distinguish simple past from participle, so both
  get the flag — documented trade-off, "was came" is not English anyone
  writes).

Output: ``quantum_parser/data/en_lexicon.json.gz`` —
``{word: [[tag_name, [subtype_name, ...]], ...]}``. The parser loads it with
stdlib gzip+json only (no nltk at parse time) and degrades to the old
heuristics if the file is absent. Closed-class words (determiners, pronouns,
auxiliaries, prepositions — WordNet doesn't cover them) stay hand-authored in
WORD_TAG_DICT, which keeps runtime precedence.

Run:  python scripts/build_parser_lexicon.py [--out PATH]

--- Spanish (``--lang spa``), added for the Spanish Freeze Test ---------------

``--lang spa`` builds ``quantum_parser/data/es_lexicon.json.gz`` from OMW's
Spanish lemma layer (``wn.all_lemma_names(pos=p, lang="spa")`` /
``wn.synsets(w, pos=p, lang="spa")``) instead of the plain English lemma
layer -- same recipe (M41), same output schema, different source table. Two
deliberate scope-narrowings vs. the English build, both because the source
data doesn't support the English recipe's tricks and are documented here
rather than silently approximated:

1. **No corpus frequency.** OMW-es lemmas don't carry SemCor-style
   ``lemma.count()`` (it is always 0 for a non-English lemma), so entries
   are ordered by sense-count alone (still deterministic, just a weaker
   MFS proxy than English's frequency-weighted order).
2. **No verb-conjugation generation.** English's ``third_sg``/``ing_form``/
   ``past_form`` regular-inflection rules and the irregular ``exc`` map are
   English morphology; Spanish conjugation (person/number/tense/mood) is a
   different, much larger system this script does not attempt. Only a
   simple regular NOUN pluralization is generated for Spanish (``+s`` after
   a vowel, ``+es`` after a consonant — the standard rule, exceptions
   unhandled). The few conjugated verb forms the Spanish curriculum
   templates need stay hand-authored in ``pos_tagger.SPANISH_WORD_TAG_DICT``
   (closed-class precedence, same as English's determiners/pronouns/aux).

``_ok_word`` also drops the ``isascii()`` requirement for ``--lang spa`` (own
branch) so accented letters (á é í ó ú ñ ü ü) survive — English's ASCII-only
check is untouched.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nltk.corpus import wordnet as wn  # noqa: E402

_VOWELS = set("aeiou")

WN_POS_TO_TAG = {"n": "NOUN", "v": "VERB", "a": "ADJ", "s": "ADJ", "r": "ADV"}


def _ok_word(w: str, lang: str = "eng") -> bool:
    if lang == "spa":
        return w.isalpha() and len(w) > 1  # accented letters allowed
    return w.isalpha() and w.isascii() and len(w) > 1


def _es_plural(noun: str) -> str:
    """Regular Spanish noun pluralization (standard rule; exceptions unhandled,
    documented at module level): vowel-final -> +s, consonant-final -> +es."""
    return noun + "s" if noun[-1] in "aeiouáéíóú" else noun + "es"


def _doubles_final(base: str) -> bool:
    """Crude CVC test for consonant doubling (stop -> stopped)."""
    if len(base) < 3:
        return False
    a, b, c = base[-3], base[-2], base[-1]
    return (c not in _VOWELS and c not in "wxy" and b in _VOWELS and a not in _VOWELS)


def plural(noun: str) -> str:
    if noun.endswith("y") and noun[-2] not in _VOWELS:
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def third_sg(verb: str) -> str:
    return plural(verb)  # same orthographic rules


def ing_form(verb: str) -> str:
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and not verb.endswith("ee"):
        return verb[:-1] + "ing"
    if _doubles_final(verb):
        return verb + verb[-1] + "ing"
    return verb + "ing"


def past_form(verb: str) -> str:
    if verb.endswith("e"):
        return verb + "d"
    if verb.endswith("y") and verb[-2] not in _VOWELS:
        return verb[:-1] + "ied"
    if _doubles_final(verb):
        return verb + verb[-1] + "ed"
    return verb + "ed"


def build(lang: str = "eng") -> dict:
    # word -> tag -> [freq, set(subtypes)]
    acc: dict = defaultdict(lambda: defaultdict(lambda: [0, set()]))

    def add(word: str, tag: str, freq: int, subtypes: tuple = ()) -> None:
        cell = acc[word][tag]
        cell[0] += freq
        cell[1].update(subtypes)

    # -- base lemmas, frequency-weighted (eng) / sense-count-weighted (spa) ----
    for pos in ("n", "v", "a", "r"):
        tag = WN_POS_TO_TAG[pos]
        lemma_names = (wn.all_lemma_names(pos=pos, lang=lang) if lang != "eng"
                       else wn.all_lemma_names(pos=pos))
        for lemma_name in lemma_names:
            w = lemma_name.lower()
            if not _ok_word(w, lang):
                continue
            freq = 0
            n_senses = 0
            synsets = wn.synsets(w, pos=pos, lang=lang) if lang != "eng" else wn.synsets(w, pos=pos)
            for syn in synsets:
                lemmas = syn.lemmas(lang=lang) if lang != "eng" else syn.lemmas()
                for lem in lemmas:
                    if lem.name().lower() == w:
                        # lem.count() is SemCor English-only; always 0 for spa
                        # (documented at module level -- sense-count is the
                        # whole ordering signal for non-English lemmas).
                        freq += lem.count() if lang == "eng" else 0
                        n_senses += 1
            add(w, tag, freq * 10 + n_senses)  # counts dominate; senses break ties

    if lang == "eng":
        # -- irregular inflections from WordNet's exception lists --------------
        # exc maps inflected -> base ("came" -> "come", "children" -> "child")
        for pos, tag, subtypes in (("n", "NOUN", ("PLURAL",)),
                                   ("v", "VERB", ("PAST_PARTICIPLE",))):
            for inflected, bases in wn._exception_map[pos].items():
                w = inflected.lower()
                if not _ok_word(w):
                    continue
                base_freq = max(
                    (acc[b.lower()][tag][0] for b in bases if b.lower() in acc), default=1
                )
                add(w, tag, max(base_freq, 1), subtypes)

        # -- regular inflections generated from base lemmas ---------------------
        nouns = [(w, tags["NOUN"][0]) for w, tags in list(acc.items()) if "NOUN" in tags]
        verbs = [(w, tags["VERB"][0]) for w, tags in list(acc.items()) if "VERB" in tags]
        for w, freq in nouns:
            add(plural(w), "NOUN", max(freq, 1), ("PLURAL",))
        for w, freq in verbs:
            f = max(freq, 1)
            add(third_sg(w), "VERB", f, ("THIRD_PERSON", "SINGULAR"))
            add(ing_form(w), "VERB", f, ("PARTICIPLE",))
            add(past_form(w), "VERB", f, ("PAST_PARTICIPLE",))
    else:
        # Spanish: regular noun pluralization only (see module docstring for
        # why verb conjugation generation is out of scope for this script).
        nouns = [(w, tags["NOUN"][0]) for w, tags in list(acc.items()) if "NOUN" in tags]
        for w, freq in nouns:
            add(_es_plural(w), "NOUN", max(freq, 1), ("PLURAL",))

    # -- serialize: per word, tags ordered by descending frequency -------------
    out = {}
    for w, tags in acc.items():
        ordered = sorted(tags.items(), key=lambda kv: -kv[1][0])
        out[w] = [[tag, sorted(subs)] for tag, (freq, subs) in ordered]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["eng", "spa"], default="eng")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_name = "en_lexicon.json.gz" if args.lang == "eng" else "es_lexicon.json.gz"
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "quantum_parser", "data", out_name)
    out_path = args.out or default_out

    lex = build(args.lang)
    payload = json.dumps(lex, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.GzipFile(out_path, "wb", mtime=0) as f:  # mtime=0 -> reproducible bytes
        f.write(payload)

    fp = hashlib.sha256(payload).hexdigest()[:16]
    n_multi = sum(1 for v in lex.values() if len(v) > 1)
    print(f"lang: {args.lang}  entries: {len(lex)}  multi-POS: {n_multi}  "
          f"gz size: {os.path.getsize(out_path) / 1e6:.2f} MB  fingerprint: {fp}")
    probes = ("shed", "moved", "broken", "thinks", "knows", "zeppelin", "bank") \
        if args.lang == "eng" else ("jardín", "cocina", "pelota", "casa", "gato", "perro")
    for probe in probes:
        print(f"  {probe}: {lex.get(probe)}")


if __name__ == "__main__":
    main()
