"""Typed-slot templates for hard-case gold generation (lead directive,
2026-09-08): the 16 hand-authored drafts in `dev/HAND_GOLD_DRAFT.md` /
`scripts/build_hand_gold_draft.py` prove the v2 schema is human-writable for
the constructions the deterministic teacher can never produce gold for
(imperatives, interjections, elision/fragments, additive/focus, quantity
fragments, synthesized subjects) -- but 16 records cannot train anything.

This module is the GENERALIZATION of those 16 exemplars: each hand-authored
surface pattern becomes a template with typed slots (verb / noun / adjective
/ proper-name / pronoun / ...), filled from word pools built from the USVS
core vocabulary + WordNet POS, so `scripts/gen_hard_gold.py` can draw
hundreds of gated records per family instead of one.

Division of labour, same discipline as `scripts/hand_gold.py`:

  this module supplies          | `scripts/hand_gold.py` / `gen_hard_gold.py` fill
  -------------------------------|--------------------------------------------
  the surface PATTERN (typed     | the actual filler values (sampled from
    slots) + the authoring spec  |   pools), tokens/pos, candidates, indices,
    that references those slots  |   the gate (linearize + legality + round-
                                  |   trip + finite train loss)

Every pool is filtered so a filler actually GROUNDS: `usvs.senses_of(word)`
non-empty (or `interj.*` for the pure-interjection gloss senses) -- see
`build_pools`. Fixed template anchor words (e.g. "did", "sounds", "too",
"more") follow the precedent the 16 drafts already set (some of those also
ground as bare `entity` -- D6/D5 -- which the gate accepts) and are NOT
pool-filtered; only the SAMPLED typed slots are.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_CT_ROOT = _HERE.parent.parent          # .../consciousness_transformer
_SCRIPTS = _CT_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from hand_gold import (  # noqa: E402
    C, W, PRIME, CTX, MEM, PREDICATE, CLAUSE, context_entry,
)

from nltk.corpus import wordnet as wn  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Word pools
# ---------------------------------------------------------------------------

_CONCRETE_NOUN_LEXNAMES = {
    "noun.animal", "noun.artifact", "noun.body", "noun.food", "noun.object",
    "noun.plant", "noun.substance", "noun.person", "noun.location", "noun.shape",
}

# Curated boosts: put familiar/child-register words first so small --per-family
# runs still look natural; automated fill (below) supplies the long tail.
_NOUN_CURATED = [
    "dog", "cat", "box", "cup", "hat", "ball", "book", "table", "door",
    "window", "room", "bowl", "doll", "father", "mother", "park", "flour",
    "chair", "bed", "car", "tree", "house", "garden", "cake", "apple",
]
_ADJ_CURATED = [
    "good", "bad", "big", "small", "hot", "cold", "happy", "sad", "loud",
    "quiet", "soft", "hard", "wet", "dry", "clean", "dirty", "sweet", "nice",
    "kind", "tired", "strange", "funny", "dangerous", "safe", "broken",
    "full", "empty", "dark", "bright", "strong", "weak", "fast", "slow",
    "new", "old", "young", "tall", "short", "heavy", "light",
]

# Plurale-tantum / irregular plurals that (unlike regular "-s" plurals, D5)
# survive `senses_of` on their own surface form.
_NOUN_PL_CURATED = [
    "people", "men", "women", "children", "teeth", "feet", "mice", "geese",
    "oxen", "cattle", "sheep", "fish", "deer",
    "police", "folk", "youth", "clothes", "glasses", "scissors", "stairs",
    "pants", "jeans", "shorts", "clergy", "staff", "crew", "troops",
    "cavalry", "infantry", "militia", "personnel", "livestock", "poultry",
    "vermin", "offspring",
]

# Singular nouns (that may also appear in the auto-built N pool) whose
# REGULAR "-s" plural is grammatically wrong because English irregularly
# pluralizes them -- their correct plural is already in `_NOUN_PL_CURATED`
# above, so the regular-plural auto-fill below must skip these bases rather
# than mint "mans"/"womans"/"childs"/"foots"/"tooths"/"mouses"/"gooses"
# alongside the real irregular form.
_NOUN_PL_IRREGULAR_BASES = {
    "man", "woman", "child", "foot", "tooth", "mouse", "goose", "person",
}

# Canonical entity names (`nsm_ct.episode._NAMES` / `clause._ENTITY_NAMES`):
# these ground as `type:"entity"` via `is_entity`, not a sense lookup, so they
# are NOT filtered by `senses_of` -- deliberately kept in sync with the rest
# of the corpus rather than expanded, so a hand/teacher-gold union never sees
# two different name universes.
PROPN_POOL = ["mary", "john", "sandra", "daniel", "bill", "fred"]

# Bare 3rd-person pronouns (not "I"/"me"/"myself" -- those route to the
# resolved prime `I`, D3; not "you" -- reserved for the imperative addressee
# reading). Ground as `type:"reference"`/`memory` regardless of WordNet
# coverage (`hand_gold.ground_W`'s `_PRONOUNS` branch), so no senses_of filter.
PRON_POOL = ["he", "she", "it", "they", "we"]

QUANT_POOL = ["more", "less", "enough", "all"]

INTERJ_CONTENT_CURATED = [
    "nonsense", "thief", "shit", "damn", "rubbish", "garbage", "stupid",
    "crap", "bullshit", "hogwash", "baloney", "malarkey", "liar", "coward",
    "fool", "idiot", "moron", "hero", "traitor", "brute", "swine", "pig",
    "rat", "snake", "clown", "monster", "disgrace", "shame", "outrage",
    "madness", "fantastic", "wonderful", "brilliant", "awesome", "terrific",
    "excellent", "fabulous", "marvelous", "great", "perfect", "incredible",
    "unbelievable", "disaster", "nightmare", "tragedy", "hallelujah",
    "bravo", "congratulations",
]

# Transitive / intransitive verbs (base form): transitivity is curated by
# hand per the plan (WordNet subcat frames are not reliable enough to
# automate this cheaply) -- deliberately overlapping with the existing
# teacher/hand-gold vocabulary (put, wait, tell, ...) rather than diverging.
VT_BASE_CURATED = [
    "open", "close", "break", "want", "ask", "tell", "eat", "drink", "take",
    "give", "find", "make", "build", "buy", "sell", "bring", "carry",
    "catch", "chase", "clean", "cook", "cut", "drop", "fix", "grab", "hit",
    "hold", "kick", "hug", "kiss", "lift", "lock", "love", "meet", "move",
    "paint", "pull", "push", "read", "save", "send", "show", "teach",
    "throw", "touch", "wash", "watch", "wear", "win", "write", "help",
    "follow", "feed", "fill", "hide", "join", "keep", "pack", "pick",
    "plant", "protect", "raise", "remove", "share", "stir", "put", "stop",
    "wave", "wake", "turn",
]
VI_BASE_CURATED = [
    "come", "go", "run", "walk", "wait", "sit", "stand", "sleep", "arrive",
    "leave", "stay", "smile", "laugh", "cry", "jump", "dance", "sing",
    "swim", "fly", "fall", "rest", "hurry", "listen", "look", "shout",
    "whisper", "cough", "sneeze", "work", "play", "appear", "disappear",
    "wander", "travel", "return",
]

# Irregular-past table, hand curated (D5, dev/HAND_GOLD_DRAFT.md: regular
# "-ed" forms are almost never their own WordNet lemma, so a *generated*
# "wanted"/"asked"/"helped" pool would fail the ground-filter wholesale;
# irregular pasts that happen to double as a noun/adjective lemma --
# "broke" (penniless), "found" (a discovery), "won" (the currency) -- are
# the ones that actually pass `senses_of`). Filtered against the live USVS
# in `build_pools`; entries that don't pass are dropped and reported, not
# hidden.
PAST_IRREGULAR = {
    "come": "came", "go": "went", "run": "ran", "sit": "sat",
    "stand": "stood", "sleep": "slept", "arrive": "arrived", "leave": "left",
    "stay": "stayed", "smile": "smiled", "laugh": "laughed", "cry": "cried",
    "jump": "jumped", "dance": "danced", "sing": "sang", "swim": "swam",
    "fly": "flew", "fall": "fell", "rest": "rested", "hurry": "hurried",
    "listen": "listened", "look": "looked", "shout": "shouted",
    "whisper": "whispered", "cough": "coughed", "sneeze": "sneezed",
    "work": "worked", "play": "played", "appear": "appeared",
    "disappear": "disappeared", "wander": "wandered", "travel": "traveled",
    "return": "returned", "stir": "stirred", "open": "opened",
    "close": "closed", "break": "broke", "want": "wanted", "ask": "asked",
    "tell": "told", "eat": "ate", "drink": "drank", "take": "took",
    "give": "gave", "find": "found", "make": "made", "build": "built",
    "buy": "bought", "sell": "sold", "bring": "brought", "carry": "carried",
    "catch": "caught", "chase": "chased", "clean": "cleaned",
    "cook": "cooked", "cut": "cut", "drop": "dropped", "fix": "fixed",
    "grab": "grabbed", "hit": "hit", "hold": "held", "kick": "kicked",
    "hug": "hugged", "kiss": "kissed", "lift": "lifted", "lock": "locked",
    "love": "loved", "meet": "met", "move": "moved", "paint": "painted",
    "pull": "pulled", "push": "pushed", "read": "read", "save": "saved",
    "send": "sent", "show": "showed", "teach": "taught", "throw": "threw",
    "touch": "touched", "wash": "washed", "watch": "watched", "wear": "wore",
    "win": "won", "write": "wrote", "help": "helped", "follow": "followed",
    "feed": "fed", "fill": "filled", "hide": "hid", "join": "joined",
    "keep": "kept", "pack": "packed", "pick": "picked", "plant": "planted",
    "protect": "protected", "raise": "raised", "remove": "removed",
    "share": "shared", "put": "put", "stop": "stopped", "wave": "waved",
    "wake": "woke", "turn": "turned",
}

_POOL_CAP = 60


@dataclass
class PoolReport:
    """Per-slot-type pool stats for the director-facing report."""
    sizes: Dict[str, int] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    dropped: Dict[str, List[str]] = field(default_factory=dict)


def _auto_noun_pool(usvs) -> List[Tuple[int, str]]:
    out = []
    for w in usvs.core_words:
        if not w.isalpha() or len(w) < 3:
            continue
        syns = wn.synsets(w)
        if not syns or syns[0].pos() != "n":
            continue
        if syns[0].lexname() not in _CONCRETE_NOUN_LEXNAMES:
            continue
        if not usvs.senses_of(w):
            continue
        out.append((syns[0].lemmas()[0].count(), w))
    out.sort(reverse=True)
    return out


_ADJ_STOP = {
    "many", "all", "another", "less", "more", "much", "some", "any", "each",
    "every", "both", "several", "few", "own", "certain", "available",
    "likely", "able", "aware", "successful", "proper", "distant",
}


def _auto_adj_pool(usvs) -> List[Tuple[int, str]]:
    out = []
    for w in usvs.core_words:
        if not w.isalpha() or len(w) < 3:
            continue
        if w in _ADJ_STOP or w.endswith("er") or w.endswith("est"):
            continue
        syns = wn.synsets(w)
        if not syns or syns[0].pos() not in ("a", "s"):
            continue
        if not usvs.senses_of(w):
            continue
        out.append((syns[0].lemmas()[0].count(), w))
    out.sort(reverse=True)
    return out


def _dedup_keep_order(words: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


_VOWELS = set("aeiou")


def _regular_plural(noun: str) -> str:
    """Standard English regular plural (no WordNet involved -- just the
    surface-form rule; the ground-filter in `build_pools` is what decides
    whether the result actually survives, via `senses_of_surface`)."""
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if len(noun) >= 2 and noun[-1] == "y" and noun[-2] not in _VOWELS:
        return noun[:-1] + "ies"
    return noun + "s"


def _regular_past(verb: str) -> str:
    """Standard English regular past tense (`-ed`, with the usual
    e-drop/y-to-i/consonant-doubling spelling rules)."""
    if verb.endswith("e"):
        return verb + "d"
    if len(verb) >= 2 and verb[-1] == "y" and verb[-2] not in _VOWELS:
        return verb[:-1] + "ied"
    if (len(verb) >= 3 and verb[-1] not in _VOWELS and verb[-1] not in "wxy"
            and verb[-2] in _VOWELS and verb[-3] not in _VOWELS):
        return verb + verb[-1] + "ed"
    return verb + "ed"


def build_pools(usvs) -> Tuple[Dict[str, List[str]], PoolReport]:
    """Build every typed-slot filler pool, gated by `usvs.senses_of`
    non-empty (pronouns/names excepted -- they ground by grammar rule, not
    by sense lookup; see `PROPN_POOL`/`PRON_POOL` docstrings above)."""
    report = PoolReport()
    pools: Dict[str, List[str]] = {}

    auto_n = [w for _, w in _auto_noun_pool(usvs)]
    n_pool = _dedup_keep_order([w for w in _NOUN_CURATED if usvs.senses_of(w)] + auto_n)[:_POOL_CAP]
    pools["N"] = n_pool
    report.sources["N"] = "curated+auto (concrete WordNet lexnames, freq-ranked)"

    auto_adj = [w for _, w in _auto_adj_pool(usvs)]
    adj_pool = _dedup_keep_order([w for w in _ADJ_CURATED if usvs.senses_of(w)] + auto_adj)[:_POOL_CAP]
    pools["ADJ"] = adj_pool
    report.sources["ADJ"] = "curated+auto (adjective/satellite, freq-ranked)"

    npl_curated_ok = [w for w in _NOUN_PL_CURATED if usvs.senses_of_surface(w)[0]]
    npl_curated_bad = [w for w in _NOUN_PL_CURATED if w not in npl_curated_ok]
    npl_auto_candidates = [(w, _regular_plural(w)) for w in n_pool
                            if w not in _NOUN_PL_IRREGULAR_BASES]
    npl_auto_ok = [pl for _, pl in npl_auto_candidates if usvs.senses_of_surface(pl)[0]]
    npl_auto_bad = [f"{w}->{pl}" for w, pl in npl_auto_candidates
                     if not usvs.senses_of_surface(pl)[0]]
    npl_pool = _dedup_keep_order(npl_curated_ok + npl_auto_ok)[:_POOL_CAP]
    pools["N_pl"] = npl_pool
    report.sources["N_pl"] = ("curated irregular plural / plurale tantum + "
                               "regular plurals of the N pool, both filtered "
                               "by senses_of_surface (lemmatized ground-filter)")
    if npl_curated_bad or npl_auto_bad:
        report.dropped["N_pl"] = npl_curated_bad + npl_auto_bad

    pools["PROPN"] = list(PROPN_POOL)
    report.sources["PROPN"] = "canonical entity names (nsm_ct.episode._NAMES) -- entity-grounded, ungated"

    pools["PRON"] = list(PRON_POOL)
    report.sources["PRON"] = "3rd-person bare pronouns -- reference-grounded, ungated"

    quant_ok = [w for w in QUANT_POOL if usvs.senses_of(w)]
    pools["QUANT"] = quant_ok
    report.sources["QUANT"] = "curated (bare quantifier fragments)"

    ic_ok = [w for w in INTERJ_CONTENT_CURATED if usvs.senses_of(w)]
    ic_bad = [w for w in INTERJ_CONTENT_CURATED if w not in ic_ok]
    pools["INTERJ_CONTENT"] = ic_ok[:40]
    report.sources["INTERJ_CONTENT"] = "curated (content words used as stance exclamations)"
    if ic_bad:
        report.dropped["INTERJ_CONTENT"] = ic_bad

    from .ground.usvs import PURE_INTERJECTION_GLOSSES
    ip_ok = [w for w in PURE_INTERJECTION_GLOSSES if usvs.senses_of(w)]
    ip_bad = [w for w in PURE_INTERJECTION_GLOSSES if w not in ip_ok]
    pools["INTERJ_PURE"] = ip_ok
    report.sources["INTERJ_PURE"] = "PURE_INTERJECTION_GLOSSES (gloss-grounded, D2)"
    if ip_bad:
        report.dropped["INTERJ_PURE"] = ip_bad

    vt_ok = [w for w in VT_BASE_CURATED if usvs.senses_of(w)]
    vt_bad = [w for w in VT_BASE_CURATED if w not in vt_ok]
    pools["VT"] = vt_ok[:40]
    report.sources["VT"] = "curated (transitive verbs, base form)"
    if vt_bad:
        report.dropped["VT"] = vt_bad

    vi_ok = [w for w in VI_BASE_CURATED if usvs.senses_of(w)]
    vi_bad = [w for w in VI_BASE_CURATED if w not in vi_ok]
    pools["VI"] = vi_ok[:30]
    report.sources["VI"] = "curated (intransitive verbs, base form)"
    if vi_bad:
        report.dropped["VI"] = vi_bad

    # Every curated base verb gets a past form: the irregular table's entry
    # when it has one, else the regular "-ed" rule (D5 fix: previously only
    # the table's entries were tried, so a base verb missing from the table
    # -- "walk", "wait" -- never got a VI_past candidate at all). All ground-
    # filtered through `senses_of_surface`, so a regular "-ed" form now
    # grounds via its morphy-recovered base-verb lemma instead of needing to
    # double as its own noun/adjective WordNet lemma.
    vt_past, vi_past, past_bad = [], [], []
    for base in VT_BASE_CURATED + VI_BASE_CURATED:
        past = PAST_IRREGULAR.get(base) or _regular_past(base)
        if not usvs.senses_of_surface(past)[0]:
            past_bad.append(f"{base}->{past}")
            continue
        if base in VT_BASE_CURATED:
            vt_past.append(past)
        else:
            vi_past.append(past)
    pools["VT_past"] = _dedup_keep_order(vt_past)[:40]
    pools["VI_past"] = _dedup_keep_order(vi_past)[:30]
    report.sources["VT_past"] = ("irregular-past table + regular \"-ed\" fallback, "
                                  "filtered by senses_of_surface (D5 fix)")
    report.sources["VI_past"] = ("irregular-past table + regular \"-ed\" fallback, "
                                  "filtered by senses_of_surface (D5 fix)")
    if past_bad:
        report.dropped["VT_past/VI_past"] = past_bad

    for k, v in pools.items():
        report.sizes[k] = len(v)
    return pools, report


# ---------------------------------------------------------------------------
# 1b. Article agreement ("a"/"an" by the filler's first SOUND, not letter)
# ---------------------------------------------------------------------------

# Vowel-LETTER-initial words that are actually consonant-SOUND-initial (a
# "y"/"w"/"h"-glide or a "yoo" vowel) -- take "a", not "an".
_ARTICLE_A_EXCEPTIONS = {
    "unicorn", "unicycle", "unique", "unit", "union", "united", "universe",
    "university", "uniform", "usual", "user", "european", "one", "once",
    "one-eyed",
}
# Consonant-LETTER-initial words with a silent leading consonant (silent
# "h") -- take "an", not "a".
_ARTICLE_AN_EXCEPTIONS = {
    "hour", "honest", "honor", "honorable", "heir", "heiress",
}
_VOWEL_LETTERS = set("aeiou")


def article(word: str) -> str:
    """"a" or "an" for *word*, by its first SOUND rather than its first
    letter -- a small hand exceptions table covers the common
    letter/sound mismatches (`unicorn`/`hour`); every other filler is
    plain vowel-letter-initial vs. not."""
    w = word.lower()
    if w in _ARTICLE_A_EXCEPTIONS:
        return "a"
    if w in _ARTICLE_AN_EXCEPTIONS:
        return "an"
    return "an" if w[:1] in _VOWEL_LETTERS else "a"


# ---------------------------------------------------------------------------
# 2. Template DSL
# ---------------------------------------------------------------------------

@dataclass
class Rendered:
    text: str
    clauses: List[C]
    context: List[dict] = field(default_factory=list)


@dataclass
class Template:
    id: str
    family: str
    slots: Dict[str, str]          # placeholder name -> pool key
    build: Callable[[Dict[str, str], object], Rendered]
    distinct: Sequence[Sequence[str]] = ()  # groups of placeholders that must sample distinct values


ALL_TEMPLATES: List[Template] = []


def _reg(t: Template) -> Template:
    ALL_TEMPLATES.append(t)
    return t


# ---- A. imperative ---------------------------------------------------------

_reg(Template("imp_vt_obj", "imperative", {"VT": "VT", "N": "N"},
    lambda v, usvs: Rendered(f"{v['VT']} the {v['N']} .",
        [C(kind="imperative", predicate=W(v["VT"]),
           roles=[("SUBJECT", PRIME("YOU")), ("OBJECT", W(v["N"]))])])))

_reg(Template("imp_neg_vt_obj", "imperative", {"VT": "VT", "N": "N"},
    lambda v, usvs: Rendered(f"do n't {v['VT']} the {v['N']} .",
        [C(kind="imperative", predicate=W(v["VT"]),
           roles=[("SUBJECT", PRIME("YOU")), ("OBJECT", W(v["N"]))])])))

_reg(Template("imp_vi_bang", "imperative", {"VI": "VI"},
    lambda v, usvs: Rendered(f"{v['VI']} !",
        [C(kind="imperative", predicate=W(v["VI"]),
           roles=[("SUBJECT", PRIME("YOU"))])])))

_reg(Template("imp_please_it", "imperative", {"VT": "VT"},
    lambda v, usvs: Rendered(f"please {v['VT']} it .",
        [C(kind="imperative", predicate=W(v["VT"]),
           roles=[("SUBJECT", PRIME("YOU")), ("OBJECT", W("it"))])])))

_reg(Template("imp_vt_obj_place", "imperative", {"VT": "VT", "N": "N", "N2": "N"},
    lambda v, usvs: Rendered(f"put the {v['N']} in the {v['N2']} .",
        [C(kind="imperative", predicate=W("put"),
           roles=[("SUBJECT", PRIME("YOU")), ("OBJECT", W(v["N"])),
                  ("PLACE", W(v["N2"]))])]),
    distinct=[("N", "N2")]))

_reg(Template("imp_vi_here", "imperative", {"VI": "VI"},
    lambda v, usvs: Rendered(f"{v['VI']} here !",
        [C(kind="imperative", predicate=W(v["VI"]),
           roles=[("SUBJECT", PRIME("YOU")), ("PLACE", W("here"))])])))

_reg(Template("imp_tell_about", "imperative", {"N": "N"},
    lambda v, usvs: Rendered(f"tell me about the {v['N']} .",
        [C(kind="imperative", predicate=W("tell"),
           roles=[("SUBJECT", PRIME("YOU")), ("INDIRECT_OBJECT", W("me")),
                  ("ABOUT", W(v["N"]))])])))

_reg(Template("imp_vi_for_me", "imperative", {"VI": "VI"},
    lambda v, usvs: Rendered(f"{v['VI']} for me .",
        [C(kind="imperative", predicate=W(v["VI"]),
           roles=[("SUBJECT", PRIME("YOU")), ("FOR", W("me"))])])))

_reg(Template("imp_vt_obj_please", "imperative", {"VT": "VT", "N": "N"},
    lambda v, usvs: Rendered(f"{v['VT']} the {v['N']} , please .",
        [C(kind="imperative", predicate=W(v["VT"]),
           roles=[("SUBJECT", PRIME("YOU")), ("OBJECT", W(v["N"]))])])))


# ---- B. pure interjection ---------------------------------------------------

_reg(Template("interjp_bang", "pure_interjection", {"INTERJ_PURE": "INTERJ_PURE"},
    lambda v, usvs: Rendered(f"{v['INTERJ_PURE']} !",
        [C(kind="interjection", predicate=W(v["INTERJ_PURE"]), roles=[])])))

_reg(Template("interjp_dot", "pure_interjection", {"INTERJ_PURE": "INTERJ_PURE"},
    lambda v, usvs: Rendered(f"{v['INTERJ_PURE']} .",
        [C(kind="interjection", predicate=W(v["INTERJ_PURE"]), roles=[])])))

_reg(Template("interjp_double", "pure_interjection",
    {"INTERJ_PURE": "INTERJ_PURE", "INTERJ_PURE2": "INTERJ_PURE"},
    lambda v, usvs: Rendered(f"{v['INTERJ_PURE']} , {v['INTERJ_PURE2']} !",
        [C(kind="interjection", predicate=W(v["INTERJ_PURE"]), roles=[]),
         C(kind="interjection", predicate=W(v["INTERJ_PURE2"]), roles=[])]),
    distinct=[("INTERJ_PURE", "INTERJ_PURE2")]))

_reg(Template("interjp_then_clause_comma", "pure_interjection",
    {"INTERJ_PURE": "INTERJ_PURE", "PRON": "PRON", "VI_past": "VI_past"},
    lambda v, usvs: Rendered(f"{v['INTERJ_PURE']} , {v['PRON']} {v['VI_past']} .",
        [C(kind="interjection", predicate=W(v["INTERJ_PURE"]), roles=[]),
         C(predicate=W(v["VI_past"]), roles=[("SUBJECT", W(v["PRON"]))])])))

_reg(Template("interjp_then_clause_bang", "pure_interjection",
    {"INTERJ_PURE": "INTERJ_PURE", "PRON": "PRON", "VI_past": "VI_past"},
    lambda v, usvs: Rendered(f"{v['INTERJ_PURE']} ! {v['PRON']} {v['VI_past']} .",
        [C(kind="interjection", predicate=W(v["INTERJ_PURE"]), roles=[]),
         C(predicate=W(v["VI_past"]), roles=[("SUBJECT", W(v["PRON"]))])])))

_reg(Template("interjp_clause_then", "pure_interjection",
    {"INTERJ_PURE": "INTERJ_PURE", "PRON": "PRON", "VI_past": "VI_past"},
    lambda v, usvs: Rendered(f"{v['PRON']} {v['VI_past']} , {v['INTERJ_PURE']} !",
        [C(predicate=W(v["VI_past"]), roles=[("SUBJECT", W(v["PRON"]))]),
         C(kind="interjection", predicate=W(v["INTERJ_PURE"]), roles=[])])))

_reg(Template("interjp_then_clause_bang2", "pure_interjection",
    {"INTERJ_PURE": "INTERJ_PURE", "PRON": "PRON", "VI_past": "VI_past"},
    lambda v, usvs: Rendered(f"{v['INTERJ_PURE']} , {v['PRON']} {v['VI_past']} !",
        [C(kind="interjection", predicate=W(v["INTERJ_PURE"]), roles=[]),
         C(predicate=W(v["VI_past"]), roles=[("SUBJECT", W(v["PRON"]))])])))

_reg(Template("interjp_double_dot", "pure_interjection",
    {"INTERJ_PURE": "INTERJ_PURE", "INTERJ_PURE2": "INTERJ_PURE"},
    lambda v, usvs: Rendered(f"{v['INTERJ_PURE']} , {v['INTERJ_PURE2']} .",
        [C(kind="interjection", predicate=W(v["INTERJ_PURE"]), roles=[]),
         C(kind="interjection", predicate=W(v["INTERJ_PURE2"]), roles=[])]),
    distinct=[("INTERJ_PURE", "INTERJ_PURE2")]))


# ---- C. content interjection ------------------------------------------------

_reg(Template("interjc_bang", "content_interjection", {"INTERJ_CONTENT": "INTERJ_CONTENT"},
    lambda v, usvs: Rendered(f"{v['INTERJ_CONTENT']} !",
        [C(kind="interjection", predicate=W(v["INTERJ_CONTENT"]), roles=[])])))

_reg(Template("interjc_dot", "content_interjection", {"INTERJ_CONTENT": "INTERJ_CONTENT"},
    lambda v, usvs: Rendered(f"{v['INTERJ_CONTENT']} .",
        [C(kind="interjection", predicate=W(v["INTERJ_CONTENT"]), roles=[])])))

_reg(Template("interjc_what", "content_interjection", {"INTERJ_CONTENT": "INTERJ_CONTENT"},
    lambda v, usvs: Rendered(f"what {v['INTERJ_CONTENT']} !",
        [C(kind="interjection", predicate=W(v["INTERJ_CONTENT"]), roles=[])])))

_reg(Template("interjc_then_clause_comma", "content_interjection",
    {"INTERJ_CONTENT": "INTERJ_CONTENT", "PRON": "PRON", "VI_past": "VI_past"},
    lambda v, usvs: Rendered(f"{v['INTERJ_CONTENT']} , {v['PRON']} {v['VI_past']} .",
        [C(kind="interjection", predicate=W(v["INTERJ_CONTENT"]), roles=[]),
         C(predicate=W(v["VI_past"]), roles=[("SUBJECT", W(v["PRON"]))])])))

_reg(Template("interjc_then_clause_bang", "content_interjection",
    {"INTERJ_CONTENT": "INTERJ_CONTENT", "PRON": "PRON", "VI_past": "VI_past"},
    lambda v, usvs: Rendered(f"{v['INTERJ_CONTENT']} ! {v['PRON']} {v['VI_past']} .",
        [C(kind="interjection", predicate=W(v["INTERJ_CONTENT"]), roles=[]),
         C(predicate=W(v["VI_past"]), roles=[("SUBJECT", W(v["PRON"]))])])))

_reg(Template("interjc_clause_then", "content_interjection",
    {"INTERJ_CONTENT": "INTERJ_CONTENT", "PRON": "PRON", "VI_past": "VI_past"},
    lambda v, usvs: Rendered(f"{v['PRON']} {v['VI_past']} , {v['INTERJ_CONTENT']} !",
        [C(predicate=W(v["VI_past"]), roles=[("SUBJECT", W(v["PRON"]))]),
         C(kind="interjection", predicate=W(v["INTERJ_CONTENT"]), roles=[])])))

_reg(Template("interjc_double", "content_interjection",
    {"INTERJ_CONTENT": "INTERJ_CONTENT", "INTERJ_CONTENT2": "INTERJ_CONTENT"},
    lambda v, usvs: Rendered(f"{v['INTERJ_CONTENT']} , {v['INTERJ_CONTENT2']} !",
        [C(kind="interjection", predicate=W(v["INTERJ_CONTENT"]), roles=[]),
         C(kind="interjection", predicate=W(v["INTERJ_CONTENT2"]), roles=[])]),
    distinct=[("INTERJ_CONTENT", "INTERJ_CONTENT2")]))

_reg(Template("interjc_double_dot", "content_interjection",
    {"INTERJ_CONTENT": "INTERJ_CONTENT", "INTERJ_CONTENT2": "INTERJ_CONTENT"},
    lambda v, usvs: Rendered(f"{v['INTERJ_CONTENT']} , {v['INTERJ_CONTENT2']} .",
        [C(kind="interjection", predicate=W(v["INTERJ_CONTENT"]), roles=[]),
         C(kind="interjection", predicate=W(v["INTERJ_CONTENT2"]), roles=[])]),
    distinct=[("INTERJ_CONTENT", "INTERJ_CONTENT2")]))


# ---- D. elision (VP-ellipsis, stranded "did") -------------------------------

def _did_context(v: Dict[str, str]) -> List[dict]:
    return [C(predicate=W(v["VT_past"]),
              roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])]


_reg(Template("elision_n_did", "elision", {"N": "N", "N_pl": "N_pl", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"the {v['N']} did .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates", word="did"),
           roles=[("SUBJECT", W(v["N"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT_past']} the {v['N2']} .",
                                _did_context(v))]),
    distinct=[("N", "N2")]))

_reg(Template("elision_n_did_not", "elision", {"N": "N", "N_pl": "N_pl", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"the {v['N']} did n't .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates", word="did"),
           roles=[("SUBJECT", W(v["N"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT_past']} the {v['N2']} .",
                                _did_context(v))]),
    distinct=[("N", "N2")]))

_reg(Template("elision_propn_did", "elision", {"PROPN": "PROPN", "N_pl": "N_pl", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"{v['PROPN']} did .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates", word="did"),
           roles=[("SUBJECT", W(v["PROPN"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT_past']} the {v['N2']} .",
                                _did_context(v))])))

_reg(Template("elision_propn_did_not", "elision", {"PROPN": "PROPN", "PROPN2": "PROPN", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"{v['PROPN']} did n't .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates", word="did"),
           roles=[("SUBJECT", W(v["PROPN"]))])],
        context=[context_entry(usvs, f"{v['PROPN2']} {v['VT_past']} the {v['N2']} .",
                                [C(predicate=W(v["VT_past"]),
                                   roles=[("SUBJECT", W(v["PROPN2"])), ("OBJECT", W(v["N2"]))])])]),
    distinct=[("PROPN", "PROPN2")]))

_reg(Template("elision_n_did_too", "elision", {"N": "N", "N_pl": "N_pl", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"the {v['N']} did too .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates", word="did"),
           roles=[("SUBJECT", W(v["N"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("ADDITIVE", W("too"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT_past']} the {v['N2']} .",
                                _did_context(v))]),
    distinct=[("N", "N2")]))

_reg(Template("elision_propn_did_too", "elision", {"PROPN": "PROPN", "N_pl": "N_pl", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"{v['PROPN']} did too .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates", word="did"),
           roles=[("SUBJECT", W(v["PROPN"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("ADDITIVE", W("too"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT_past']} the {v['N2']} .",
                                _did_context(v))])))

_reg(Template("elision_n_did_not_either", "elision", {"N": "N", "N_pl": "N_pl", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"the {v['N']} did n't either .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates", word="did"),
           roles=[("SUBJECT", W(v["N"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("ADDITIVE", W("either"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT_past']} the {v['N2']} .",
                                _did_context(v))]),
    distinct=[("N", "N2")]))

_reg(Template("elision_and_n_did", "elision", {"N": "N", "N_pl": "N_pl", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"and the {v['N']} did .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates", word="did"),
           roles=[("SUBJECT", W(v["N"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT_past']} the {v['N2']} .",
                                _did_context(v))]),
    distinct=[("N", "N2")]))


# ---- E. additive / focus ----------------------------------------------------

_reg(Template("add_me_too", "additive_focus", {"N_pl": "N_pl", "N2": "N"},
    lambda v, usvs: Rendered("me too .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("SUBJECT", W("me")),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("ADDITIVE", W("too"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} want {article(v['N2'])} {v['N2']} .",
                                [C(predicate=W("want"),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))

_reg(Template("add_and_a_n_too", "additive_focus", {"N": "N", "N_pl": "N_pl", "N2": "N"},
    lambda v, usvs: Rendered(f"and {article(v['N'])} {v['N']} too .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("SUBJECT", CTX("reference", of=v["N_pl"], scope="roles")),
                  ("OBJECT", W(v["N"])),
                  ("ADDITIVE", W("too"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} want {article(v['N2'])} {v['N2']} .",
                                [C(predicate=W("want"),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])]),
    distinct=[("N", "N2")]))

_reg(Template("add_propn_too", "additive_focus", {"PROPN": "PROPN", "N_pl": "N_pl", "N2": "N"},
    lambda v, usvs: Rendered(f"{v['PROPN']} too .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("SUBJECT", W(v["PROPN"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("ADDITIVE", W("too"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} want {article(v['N2'])} {v['N2']} .",
                                [C(predicate=W("want"),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))

_reg(Template("add_and_propn_too", "additive_focus", {"PROPN": "PROPN", "N_pl": "N_pl", "N2": "N"},
    lambda v, usvs: Rendered(f"and {v['PROPN']} too .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("SUBJECT", W(v["PROPN"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("ADDITIVE", W("too"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} want {article(v['N2'])} {v['N2']} .",
                                [C(predicate=W("want"),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))

_reg(Template("add_me_also", "additive_focus", {"N_pl": "N_pl", "N2": "N"},
    lambda v, usvs: Rendered("me also .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("SUBJECT", W("me")),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("ADDITIVE", W("also"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} want {article(v['N2'])} {v['N2']} .",
                                [C(predicate=W("want"),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))

_reg(Template("add_and_a_n_also", "additive_focus", {"N": "N", "N_pl": "N_pl", "N2": "N"},
    lambda v, usvs: Rendered(f"and {article(v['N'])} {v['N']} also .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("SUBJECT", CTX("reference", of=v["N_pl"], scope="roles")),
                  ("OBJECT", W(v["N"])),
                  ("ADDITIVE", W("also"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} want {article(v['N2'])} {v['N2']} .",
                                [C(predicate=W("want"),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])]),
    distinct=[("N", "N2")]))

_reg(Template("add_propn_comma_too", "additive_focus", {"PROPN": "PROPN", "N_pl": "N_pl", "N2": "N"},
    lambda v, usvs: Rendered(f"{v['PROPN']} , too .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("SUBJECT", W(v["PROPN"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("ADDITIVE", W("too"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} want {article(v['N2'])} {v['N2']} .",
                                [C(predicate=W("want"),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))

_reg(Template("add_only_propn_too", "additive_focus", {"PROPN": "PROPN", "N_pl": "N_pl", "N2": "N"},
    lambda v, usvs: Rendered(f"only {v['PROPN']} too .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("SUBJECT", W(v["PROPN"])),
                  ("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("FOCUS", W("only"))])],
        context=[context_entry(usvs, f"the {v['N_pl']} want {article(v['N2'])} {v['N2']} .",
                                [C(predicate=W("want"),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))


# ---- F. quantity -------------------------------------------------------------

_reg(Template("quant_bang_ctx", "quantity", {"QUANT": "QUANT", "N_pl": "N_pl", "N2": "N", "VT": "VT"},
    lambda v, usvs: Rendered(f"{v['QUANT']} !",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("QUANTITY", W(v["QUANT"]))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT']} the {v['N2']} .",
                                [C(predicate=W(v["VT"]),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))

_reg(Template("quant_dot_ctx", "quantity", {"QUANT": "QUANT", "N_pl": "N_pl", "N2": "N", "VT": "VT"},
    lambda v, usvs: Rendered(f"{v['QUANT']} .",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("QUANTITY", W(v["QUANT"]))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT']} the {v['N2']} .",
                                [C(predicate=W(v["VT"]),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))

_reg(Template("quant_bang_past_ctx", "quantity", {"QUANT": "QUANT", "N_pl": "N_pl", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"{v['QUANT']} !",
        [C(predicate=CTX("elision", of=PREDICATE, scope="predicates"),
           roles=[("OBJECT", CTX("elision", of=v["N2"], scope="roles")),
                  ("QUANTITY", W(v["QUANT"]))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT_past']} the {v['N2']} .",
                                [C(predicate=W(v["VT_past"]),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))

_reg(Template("quant_n_pl_bang", "quantity", {"QUANT": "QUANT", "N_pl": "N_pl"},
    lambda v, usvs: Rendered(f"{v['QUANT']} {v['N_pl']} !",
        [C(predicate=MEM(gtype="elision", method="elision_no_antecedent"),
           roles=[("OBJECT", W(v["N_pl"])), ("QUANTITY", W(v["QUANT"]))])])))

_reg(Template("quant_n_pl_dot", "quantity", {"QUANT": "QUANT", "N_pl": "N_pl"},
    lambda v, usvs: Rendered(f"{v['QUANT']} {v['N_pl']} .",
        [C(predicate=MEM(gtype="elision", method="elision_no_antecedent"),
           roles=[("OBJECT", W(v["N_pl"])), ("QUANTITY", W(v["QUANT"]))])])))

_reg(Template("quant_not_bang", "quantity", {"QUANT": "QUANT"},
    lambda v, usvs: Rendered(f"not {v['QUANT']} !",
        [C(predicate=MEM(gtype="elision", method="elision_no_antecedent"),
           roles=[("QUANTITY", W(v["QUANT"]))])])))

_reg(Template("quant_of_n_pl", "quantity", {"QUANT": "QUANT", "N_pl": "N_pl"},
    lambda v, usvs: Rendered(f"{v['QUANT']} of the {v['N_pl']} !",
        [C(predicate=MEM(gtype="elision", method="elision_no_antecedent"),
           roles=[("QUANTITY", W(v["QUANT"])), ("OF", W(v["N_pl"]))])])))

_reg(Template("quant_n_bang", "quantity", {"QUANT": "QUANT", "N": "N"},
    lambda v, usvs: Rendered(f"{v['QUANT']} {v['N']} !",
        [C(predicate=MEM(gtype="elision", method="elision_no_antecedent"),
           roles=[("QUANTITY", W(v["QUANT"])), ("OBJECT", W(v["N"]))])])))


# ---- G. synth-subject fragments (propositional anaphora) -------------------

_reg(Template("synth_sounds", "synth_subject", {"ADJ": "ADJ", "PRON": "PRON", "N": "N"},
    lambda v, usvs: Rendered(f"sounds {v['ADJ']} .",
        [C(predicate=W("sounds"),
           roles=[("SUBJECT", CTX("reference", of=CLAUSE, scope="clauses",
                                  method="propositional_anaphora")),
                  ("COMPLEMENT", W(v["ADJ"]))])],
        context=[context_entry(usvs, f"{v['PRON']} go to the {v['N']} .",
                                [C(predicate=W("go"),
                                   roles=[("SUBJECT", W(v["PRON"])), ("PLACE", W(v["N"]))])])])))

_reg(Template("synth_sounds_bang", "synth_subject", {"ADJ": "ADJ", "PRON": "PRON", "N": "N"},
    lambda v, usvs: Rendered(f"sounds {v['ADJ']} !",
        [C(predicate=W("sounds"),
           roles=[("SUBJECT", CTX("reference", of=CLAUSE, scope="clauses",
                                  method="propositional_anaphora")),
                  ("COMPLEMENT", W(v["ADJ"]))])],
        context=[context_entry(usvs, f"{v['PRON']} go to the {v['N']} .",
                                [C(predicate=W("go"),
                                   roles=[("SUBJECT", W(v["PRON"])), ("PLACE", W(v["N"]))])])])))

_reg(Template("synth_looks", "synth_subject", {"ADJ": "ADJ", "PRON": "PRON", "VI": "VI"},
    lambda v, usvs: Rendered(f"looks {v['ADJ']} .",
        [C(predicate=W("looks"),
           roles=[("SUBJECT", CTX("reference", of=CLAUSE, scope="clauses",
                                  method="propositional_anaphora")),
                  ("COMPLEMENT", W(v["ADJ"]))])],
        context=[context_entry(usvs, f"{v['PRON']} {v['VI']} .",
                                [C(predicate=W(v["VI"]), roles=[("SUBJECT", W(v["PRON"]))])])])))

_reg(Template("synth_looks_bang", "synth_subject", {"ADJ": "ADJ", "PRON": "PRON", "VI": "VI"},
    lambda v, usvs: Rendered(f"looks {v['ADJ']} !",
        [C(predicate=W("looks"),
           roles=[("SUBJECT", CTX("reference", of=CLAUSE, scope="clauses",
                                  method="propositional_anaphora")),
                  ("COMPLEMENT", W(v["ADJ"]))])],
        context=[context_entry(usvs, f"{v['PRON']} {v['VI']} .",
                                [C(predicate=W(v["VI"]), roles=[("SUBJECT", W(v["PRON"]))])])])))

_reg(Template("synth_seems", "synth_subject", {"ADJ": "ADJ", "PRON": "PRON", "VT": "VT", "N": "N"},
    lambda v, usvs: Rendered(f"seems {v['ADJ']} .",
        [C(predicate=W("seems"),
           roles=[("SUBJECT", CTX("reference", of=CLAUSE, scope="clauses",
                                  method="propositional_anaphora")),
                  ("COMPLEMENT", W(v["ADJ"]))])],
        context=[context_entry(usvs, f"{v['PRON']} {v['VT']} the {v['N']} .",
                                [C(predicate=W(v["VT"]),
                                   roles=[("SUBJECT", W(v["PRON"])), ("OBJECT", W(v["N"]))])])])))

_reg(Template("synth_seems_bang", "synth_subject", {"ADJ": "ADJ", "PRON": "PRON", "VT": "VT", "N": "N"},
    lambda v, usvs: Rendered(f"seems {v['ADJ']} !",
        [C(predicate=W("seems"),
           roles=[("SUBJECT", CTX("reference", of=CLAUSE, scope="clauses",
                                  method="propositional_anaphora")),
                  ("COMPLEMENT", W(v["ADJ"]))])],
        context=[context_entry(usvs, f"{v['PRON']} {v['VT']} the {v['N']} .",
                                [C(predicate=W(v["VT"]),
                                   roles=[("SUBJECT", W(v["PRON"])), ("OBJECT", W(v["N"]))])])])))

_reg(Template("synth_feels", "synth_subject", {"ADJ": "ADJ", "N_pl": "N_pl", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"feels {v['ADJ']} .",
        [C(predicate=W("feels"),
           roles=[("SUBJECT", CTX("reference", of=CLAUSE, scope="clauses",
                                  method="propositional_anaphora")),
                  ("COMPLEMENT", W(v["ADJ"]))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT_past']} the {v['N2']} .",
                                [C(predicate=W(v["VT_past"]),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))

_reg(Template("synth_feels_bang", "synth_subject", {"ADJ": "ADJ", "N_pl": "N_pl", "N2": "N", "VT_past": "VT_past"},
    lambda v, usvs: Rendered(f"feels {v['ADJ']} !",
        [C(predicate=W("feels"),
           roles=[("SUBJECT", CTX("reference", of=CLAUSE, scope="clauses",
                                  method="propositional_anaphora")),
                  ("COMPLEMENT", W(v["ADJ"]))])],
        context=[context_entry(usvs, f"the {v['N_pl']} {v['VT_past']} the {v['N2']} .",
                                [C(predicate=W(v["VT_past"]),
                                   roles=[("SUBJECT", W(v["N_pl"])), ("OBJECT", W(v["N2"]))])])])))


# ---- H. speaker prime --------------------------------------------------------

_reg(Template("speaker_tell_about_n", "speaker_prime", {"N": "N"},
    lambda v, usvs: Rendered(f"tell me about the {v['N']} .",
        [C(kind="imperative", predicate=W("tell"),
           roles=[("SUBJECT", PRIME("YOU")), ("INDIRECT_OBJECT", W("me")),
                  ("ABOUT", W(v["N"]))])])))

_reg(Template("speaker_wait_for_me", "speaker_prime", {"VI": "VI"},
    lambda v, usvs: Rendered(f"{v['VI']} for me .",
        [C(kind="imperative", predicate=W(v["VI"]),
           roles=[("SUBJECT", PRIME("YOU")), ("FOR", W("me"))])])))

_reg(Template("speaker_tell_about_n_bang", "speaker_prime", {"N": "N"},
    lambda v, usvs: Rendered(f"tell me about the {v['N']} !",
        [C(kind="imperative", predicate=W("tell"),
           roles=[("SUBJECT", PRIME("YOU")), ("INDIRECT_OBJECT", W("me")),
                  ("ABOUT", W(v["N"]))])])))

_reg(Template("speaker_give_me_n", "speaker_prime", {"N": "N"},
    lambda v, usvs: Rendered(f"give me the {v['N']} .",
        [C(kind="imperative", predicate=W("give"),
           roles=[("SUBJECT", PRIME("YOU")), ("INDIRECT_OBJECT", W("me")),
                  ("OBJECT", W(v["N"]))])])))

_reg(Template("speaker_show_me_n", "speaker_prime", {"N": "N"},
    lambda v, usvs: Rendered(f"show me the {v['N']} .",
        [C(kind="imperative", predicate=W("show"),
           roles=[("SUBJECT", PRIME("YOU")), ("INDIRECT_OBJECT", W("me")),
                  ("OBJECT", W(v["N"]))])])))

_reg(Template("speaker_wait_for_me_please", "speaker_prime", {"VI": "VI"},
    lambda v, usvs: Rendered(f"{v['VI']} for me , please .",
        [C(kind="imperative", predicate=W(v["VI"]),
           roles=[("SUBJECT", PRIME("YOU")), ("FOR", W("me"))])])))

_reg(Template("speaker_vt_it_for_me", "speaker_prime", {"VT": "VT"},
    lambda v, usvs: Rendered(f"{v['VT']} it for me .",
        [C(kind="imperative", predicate=W(v["VT"]),
           roles=[("SUBJECT", PRIME("YOU")), ("OBJECT", W("it")), ("FOR", W("me"))])])))

_reg(Template("speaker_bring_me_n", "speaker_prime", {"N": "N"},
    lambda v, usvs: Rendered(f"bring me {article(v['N'])} {v['N']} .",
        [C(kind="imperative", predicate=W("bring"),
           roles=[("SUBJECT", PRIME("YOU")), ("INDIRECT_OBJECT", W("me")),
                  ("OBJECT", W(v["N"]))])])))

_reg(Template("speaker_tell_about_propn", "speaker_prime", {"PROPN": "PROPN"},
    lambda v, usvs: Rendered(f"tell me about {v['PROPN']} .",
        [C(kind="imperative", predicate=W("tell"),
           roles=[("SUBJECT", PRIME("YOU")), ("INDIRECT_OBJECT", W("me")),
                  ("ABOUT", W(v["PROPN"]))])])))


FAMILIES: List[str] = sorted({t.family for t in ALL_TEMPLATES})


def templates_by_family() -> Dict[str, List[Template]]:
    out: Dict[str, List[Template]] = {f: [] for f in FAMILIES}
    for t in ALL_TEMPLATES:
        out[t.family].append(t)
    return out
