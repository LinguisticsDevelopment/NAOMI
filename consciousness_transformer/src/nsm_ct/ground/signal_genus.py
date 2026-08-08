"""Genus-differentia gloss parsing -- an independent hypernym signal (M26+,
dev/SIGNALS_AUDIT.md #1).

WordNet glosses are near-formulaic: a noun gloss states its genus (the
immediate category) up front, then differentiates it --
    "a motor vehicle with four wheels"           -> genus "vehicle"
    "any of various plants of the genus Rosa"    -> genus "plant"
    "an act of stealing"                         -> genus "act"
-- while a verb gloss is dominantly a bare infinitive with no leading "to"
    "draw air into, and expel out of, the lungs" -> genus "draw"
(spot-checked against this repo's WordNet: 15/15 sampled verb glosses had no
leading "to"; a handful of older WordNet releases do use "to VERB ...", so
that form is handled too).

``genus_of`` extracts that leading head word deterministically: no learning,
no dependency beyond the stdlib plus the existing ``nsm_ct.wordnet.senses``
lookup used only to *validate* a candidate denotes something (any part of
speech counts -- see module docstring on ``extras`` below for why).

This is a second, INDEPENDENT hypernym-shaped signal on top of the
synset-pointer hypernym relation already carried by ``RelationGraph.is_a``:
pointer hypernyms come from WordNet's curated taxonomy edges; genus comes
from parsing the gloss text itself. The two agree often but not always --
partial disagreement is expected and is exactly what makes this signal worth
having (an independent signal that always agreed would carry no new
information; see the pointer-agreement number this module's tests report).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..wordnet import senses

# ---------------------------------------------------------------------------
# Fixed word lists (deterministic, no learning).
# ---------------------------------------------------------------------------

# Quantifier phrases that wrap the real genus: "any of various plants of the
# genus X" -- the genus is inside "various plants", not "any"/"one"/"member".
_WRAPPERS: Tuple[str, ...] = (
    "a member of",
    "a variety of",
    "any of",
    "one of",
)

# Determiners/quantifiers that can lead (or sit inside) a noun phrase but are
# never themselves the genus -- skipped when scanning right-to-left for the
# head noun.
_NON_CANDIDATE = frozenset({
    "a", "an", "the", "any", "one", "some", "several", "various", "numerous",
    "other", "another", "each", "every", "all", "both", "either", "neither",
    "no", "many", "much", "more", "most", "few", "little", "this", "that",
    "these", "those",
})

# Tokens that close the leading noun phrase -- the genus is the last content
# word seen before one of these (or before any stopping punctuation).
_STOP_WORDS = frozenset({
    "of", "with", "for", "in", "on", "at", "by", "from", "into", "onto",
    "upon", "that", "which", "who", "whom", "whose", "used", "having",
    "consisting", "containing", "made", "found", "living", "growing",
    "occurring", "belonging", "characterized", "distinguished", "marked",
    "seen", "designed", "equipped", "as", "to", "and", "or", "but",
})

_LEADING_PAREN = re.compile(r"^\(([^()]*)\)\s*")
_WORD_OR_PUNCT = re.compile(r"[A-Za-z']+|[^\sA-Za-z']")
_TO_VERB = re.compile(r"^to\s+([A-Za-z]+)\b")
_FIRST_WORD = re.compile(r"^([A-Za-z']+)")


def _valid(word: str) -> bool:
    """True if *word* denotes anything in WordNet, any part of speech.

    Deliberately POS-agnostic: the point is to reject noise tokens (stray
    gerunds, connective words) that happen to survive NP-scanning, not to
    assert the candidate is specifically a noun sense.
    """
    return bool(word) and bool(senses(word))


def _singularize(word: str) -> str:
    """Naive stdlib depluralization. Only ever used behind a ``_valid`` gate
    (see ``_canon``), so an over-eager strip on an already-singular word
    (e.g. "genus" -> "genu") is harmless: "genu" won't validate and the
    original is kept."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _canon(word: str) -> str:
    """Prefer the singular lemma form when it also validates (aligns the
    returned genus with WordNet's singular-lemma hypernym convention)."""
    sing = _singularize(word)
    if sing != word and _valid(sing):
        return sing
    return word


def _strip_leading_parens(text: str) -> str:
    while True:
        m = _LEADING_PAREN.match(text)
        if not m:
            return text
        text = text[m.end():]


_HYPHENS = frozenset("-‐‑‒–—")


def _leading_np_tokens(text: str) -> List[str]:
    """Lowercased word tokens of the leading noun phrase: stop at the first
    stop-word or the first (non-hyphen) punctuation character. Hyphens are
    skipped rather than treated as a boundary so compound modifiers
    ("fleet-footed", "cold-blooded") don't sever the phrase before its head."""
    tokens: List[str] = []
    for tok in _WORD_OR_PUNCT.findall(text):
        if tok.isalpha():
            low = tok.lower()
            if low in _STOP_WORDS:
                break
            tokens.append(low)
        elif tok in _HYPHENS:
            continue
        else:
            break  # comma / semicolon / paren / ... ends the NP
    return tokens


def _head_of_np(tokens: Sequence[str]) -> Optional[str]:
    """Right-to-left scan for the first token that is neither a bare
    determiner/quantifier nor invalid, canonicalizing to singular."""
    for tok in reversed(tokens):
        if tok in _NON_CANDIDATE:
            continue
        if _valid(tok):
            return _canon(tok)
    return None


def _genus_with_confidence(gloss: str) -> Tuple[Optional[str], bool]:
    """Core parse. Returns ``(genus, confident)``.

    ``confident`` is False only for the bare-first-word branch, which is
    genuinely ambiguous from text alone: a gloss with no leading
    determiner/quantifier is dominantly a bare-infinitive verb gloss ("draw
    air into ... the lungs") but occasionally a determiner-less noun gloss
    ("animal viruses belonging to the family ..."), and nothing in the
    surface text disambiguates the two. ``genus_of`` still returns its best
    guess either way (matching the spec: bare-infinitive -> head verb); the
    confidence split exists so ``extras`` can prefer the higher-precision
    edges (explicit article / quantifier-wrapper / "to VERB" patterns) when
    proposing close-edges for placement.
    """
    if not gloss:
        return None, False
    text = _strip_leading_parens(gloss.strip())
    if not text:
        return None, False
    text = text.split(";", 1)[0].strip()  # semicolon glosses: first clause only
    if not text:
        return None, False

    low = text.lower()

    # Quantifier wrappers: skip past them to the real genus.
    for wrapper in _WRAPPERS:
        if low.startswith(wrapper + " "):
            return _genus_with_confidence(text[len(wrapper):].strip())

    # Verb glosses spelled "to VERB ..." (rare in modern WordNet, but named
    # in spec / seen in some releases).
    m = _TO_VERB.match(text)
    if m:
        candidate = m.group(1).lower()
        return (candidate, True) if _valid(candidate) else (None, False)

    fw = _FIRST_WORD.match(text)
    if not fw:
        return None, False
    first = fw.group(1).lower()

    if first not in _NON_CANDIDATE:
        # No leading determiner/quantifier: WordNet verb glosses are
        # dominantly bare infinitives, so treat the first word as the head
        # verb -- but see the ambiguity note above.
        return (first, False) if _valid(first) else (None, False)

    tokens = _leading_np_tokens(text)
    return _head_of_np(tokens), True


def genus_of(gloss: str) -> Optional[str]:
    """Extract the genus: the head word of *gloss*'s leading phrase.

    Deterministic, stdlib-only, no learning. Returns ``None`` when nothing
    in the leading phrase validates against WordNet.
    """
    genus, _ = _genus_with_confidence(gloss)
    return genus


# ---------------------------------------------------------------------------
# Harness contract (scripts/ablate_signal.py, dev/SEMANTIC_MAPPING_PLAN.md).
# ---------------------------------------------------------------------------


# A genus that thousands of unrelated words all bottom out at ("act",
# "state", "person", "part", "thing", ...) is a near-vacuous top-level
# category, not a specific hypernym -- treating it as a close-edge hub
# collapses placement (everything gets pulled toward the hub, inflating
# random-pair cosine and hurting synonym/similar discrimination). Capping how
# many source words may cite the same genus target keeps the signal
# specific, deterministic post-filter, no hand-curated stoplist needed.
_MAX_GENUS_DEGREE = 3


def extras(vocab, graph) -> Dict[str, List[Tuple[str, str]]]:
    """``close_extra`` pairs ``(w, genus_of(gloss(w)))`` for every in-vocab,
    non-reflexive hit.

    Two quality filters keep the proposed edges specific rather than noisy
    (both empirically necessary -- see the ablation report: without them
    synonym_auc/similar_auc regress well beyond the noise band even though
    hypernym_cos_auc improves):

    - only the *confident* parses are used (explicit article / quantifier
      wrapper / "to VERB" patterns) -- the bare-first-word branch is dropped
      here because it conflates real bare-infinitive verb glosses with the
      rarer determiner-less noun gloss, and that ambiguity shows up as
      wrong edges;
    - hub genus targets (cited by more than ``_MAX_GENUS_DEGREE`` distinct
      words -- near-vacuous top-level categories like "act"/"state"/
      "person"/"part") are dropped, since pulling many unrelated words
      toward one hub collapses placement instead of adding specific signal.

    The M24 leakage rule (dropping pairs that collide with a held-out test
    pair) is enforced by the ablation harness itself, not here -- this just
    proposes candidate close edges.
    """
    wset = set(vocab)
    raw: List[Tuple[str, str]] = []
    for w, gloss in graph.gloss.items():
        g, confident = _genus_with_confidence(gloss)
        if confident and g and g in wset and g != w:
            raw.append((w, g))

    degree: Dict[str, int] = {}
    for _, g in raw:
        degree[g] = degree.get(g, 0) + 1
    pairs = [(w, g) for w, g in raw if degree[g] <= _MAX_GENUS_DEGREE]
    return {"close_extra": pairs}
