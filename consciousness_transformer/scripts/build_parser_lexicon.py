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


def _ok_word(w: str) -> bool:
    return w.isalpha() and w.isascii() and len(w) > 1


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


def build() -> dict:
    # word -> tag -> [freq, set(subtypes)]
    acc: dict = defaultdict(lambda: defaultdict(lambda: [0, set()]))

    def add(word: str, tag: str, freq: int, subtypes: tuple = ()) -> None:
        cell = acc[word][tag]
        cell[0] += freq
        cell[1].update(subtypes)

    # -- base lemmas, frequency-weighted --------------------------------------
    for pos in ("n", "v", "a", "r"):
        tag = WN_POS_TO_TAG[pos]
        for lemma_name in wn.all_lemma_names(pos=pos):
            w = lemma_name.lower()
            if not _ok_word(w):
                continue
            freq = 0
            n_senses = 0
            for syn in wn.synsets(w, pos=pos):
                for lem in syn.lemmas():
                    if lem.name().lower() == w:
                        freq += lem.count()
                        n_senses += 1
            add(w, tag, freq * 10 + n_senses)  # counts dominate; senses break ties

    # -- irregular inflections from WordNet's exception lists ------------------
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

    # -- regular inflections generated from base lemmas ------------------------
    nouns = [(w, tags["NOUN"][0]) for w, tags in list(acc.items()) if "NOUN" in tags]
    verbs = [(w, tags["VERB"][0]) for w, tags in list(acc.items()) if "VERB" in tags]
    for w, freq in nouns:
        add(plural(w), "NOUN", max(freq, 1), ("PLURAL",))
    for w, freq in verbs:
        f = max(freq, 1)
        add(third_sg(w), "VERB", f, ("THIRD_PERSON", "SINGULAR"))
        add(ing_form(w), "VERB", f, ("PARTICIPLE",))
        add(past_form(w), "VERB", f, ("PAST_PARTICIPLE",))

    # -- serialize: per word, tags ordered by descending frequency -------------
    out = {}
    for w, tags in acc.items():
        ordered = sorted(tags.items(), key=lambda kv: -kv[1][0])
        out[w] = [[tag, sorted(subs)] for tag, (freq, subs) in ordered]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "quantum_parser", "data", "en_lexicon.json.gz")
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()

    lex = build()
    payload = json.dumps(lex, sort_keys=True, separators=(",", ":")).encode()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.GzipFile(args.out, "wb", mtime=0) as f:  # mtime=0 -> reproducible bytes
        f.write(payload)

    fp = hashlib.sha256(payload).hexdigest()[:16]
    n_multi = sum(1 for v in lex.values() if len(v) > 1)
    print(f"entries: {len(lex)}  multi-POS: {n_multi}  "
          f"gz size: {os.path.getsize(args.out) / 1e6:.2f} MB  fingerprint: {fp}")
    for probe in ("shed", "moved", "broken", "thinks", "knows", "zeppelin", "bank"):
        print(f"  {probe}: {lex.get(probe)}")


if __name__ == "__main__":
    main()
