"""M58a: the CORPUS -> EPISODE CONVERTER (dev/AURORA_SPRINT.md, dev/NEXT_ARC_PLAN.md M57).

Turns real (or synthetic-but-wild) prose into :class:`~nsm_ct.episode.Episode`
objects with SELF-GENERATED comprehension questions -- no hand labeling. The
question/answer machinery is the SAME "queried role" table
(:func:`nsm_ct.clause_reactor._queried_role`) the curriculum already uses, so
converted episodes are consumable by :func:`nsm_ct.clause_reactor.build_clause_batch`'s
default (parser-based, "old") path without any new branch there.

Three pieces:

* :func:`iter_sentences` -- a simple deterministic sentence splitter +
  the tokenization the parser expects (lowercase, space-separated
  punctuation, contractions split off -- "mary is in the garden .",
  "don't" -> "do n't", matching every hand-written curriculum sentence).
* :func:`parse_passage` -- runs each sentence through the SAME
  parser path :func:`nsm_ct.clause_reactor._context_steps` uses
  (``parser._parse_topk_one`` -> :func:`nsm_ct.clause.extract_discourse`),
  classifying the outcome as ``ok`` (one :class:`ParsedClause` per
  extracted (entity, relation, value) triple -- a transfer clause yields
  several) or a :class:`ParseFailure` tagged with one of the six
  :data:`FAILURE_REASONS` codes. Every sentence gets EXACTLY one outcome
  for taxonomy purposes (see :func:`taxonomy_counts`): "ok" if it
  produced at least one :class:`ParsedClause`, else its failure reason.
* :func:`make_episodes` -- holds out one eligible clause (a relation the
  existing "queried role" table can actually ask about --
  :data:`_RELATION_QUESTION_TEMPLATE`'s PLACE/RECIPIENT/AGENT, the only
  ones :func:`nsm_ct.clause_reactor._queried_role` resolves), keeps
  EVERY sentence of the passage in context (nothing removed -- the
  question just targets the held-out clause's value), and builds MC
  options from the gold value + same-relation distractor values seen
  elsewhere in the passage (or an optional corpus-wide pool).

M58c (the parser round driven by dev/PROSE_FAILURE_TAXONOMY.md's measured
real-text failure histogram) adds four perception-side, additive fixes on
top of the above, none of which change the shape just described:

* **lexicon coverage** -- :func:`_is_known_word` now also accepts
  quantum_parser's new hyphen-compound / on-demand-WordNet / NAME-NUM
  fallback tiers (:mod:`src.parser.pos_tagger`'s ``lexicon_entry``,
  ``is_bare_name_token``, ``looks_like_number`` -- see that module's
  docstring for the tier breakdown);
* **parse ties** -- a sentence whose top-K hypotheses tie within
  :data:`_AMBIGUITY_MARGIN` is no longer an automatic failure: the top-1
  reading is still extracted normally, just tagged with a
  ``HypothesisCandidateSet`` (the M55a membrane shape) on
  :attr:`ParsedClause.hypotheses`, and :func:`taxonomy_counts` reports it
  under the new ``"parsed-ambiguous"`` outcome (a variant of "ok", not a
  failure reason -- it stays out of :data:`FAILURE_REASONS`);
* **quotation/fragment handling** -- quotation-mark tokens are stripped
  before the sentence is handed to the parser (they carry no grammar and
  can by themselves prevent a hypothesis from ever completing); a bare
  interjection/verbless fragment gets the new ``"fragment-skipped"``
  failure reason instead of ``"no-parse"``; when the full sentence still
  yields nothing, the quoted SPAN inside it (if any) is tried on its own,
  tagged :attr:`ParsedClause.source` ``"quoted"`` on success (dialogue
  ATTRIBUTION to a speaker is out of scope -- the quoted clause's own
  facts are extracted, nothing about who said it);
* **passage-level entity registry** -- :func:`parse_passage` now threads a
  simple :class:`_PassageRegistry` (most-recently-mentioned entity name)
  across its sentences, so a pronoun with a registered antecedent from an
  EARLIER sentence resolves to that name (upgrading what used to be an
  automatic ``"pronoun-unresolvable"`` into a real fact) instead of being
  flagged unconditionally.
"""

from __future__ import annotations

import gc
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .clause import _PRONOUNS, extract_discourse, is_entity, strip_attribution
from .clause_reactor import _TRANSFER_ROLE_MAP
from .episode import Episode
from .membrane import (NAME_GENDER, PRONOUN_MORPHOLOGY, HypothesisCandidateSet,
                        hypothesis_candidate_set)

# Diagnosed memory blowup (see dev/ RESEARCH_NOTES / the converter-memfix
# work): quantum_parser's chart-parsing loop (src/parser/quantum_parser.py)
# branches over the FULL Cartesian product of a ruleset's ambiguous-anchor
# alternatives (``itertools.product(*ambiguous_groups)``) before dedup/prune
# ever runs. A handful of ambiguous anchors in a long, multi-clause,
# parenthetical-heavy real-prose sentence (measured on
# data/corpus/real_gutenberg_alice.txt: several of its first ~50 sentences
# individually blew a 2GB address-space cap or ran past 15s of pure
# CPU-bound Hypothesis deep-copying, in complete isolation from every other
# sentence/file) multiplies out to tens of thousands of deep-copied
# Hypothesis objects for ONE ruleset pass of ONE sentence -- that's what hit
# ~14GB on the ~180KB corpus, not accumulation across files.
#
# These are the "named module-level dial" caps the task asks for: additive
# (quantum_parser's default ParserConfig()/every non-corpus caller is
# untouched -- see ParserConfig.max_ruleset_hypotheses/max_parse_seconds and
# QuantumParser.parse's config_override parameter), applied ONLY to the
# corpus-conversion parse calls below via ParserInputEncoder._parse_topk_one's
# optional max_hypotheses/max_seconds kwargs. A sentence that hits either cap
# raises ParseResourceExceeded, caught here and tagged with the honest
# "parse-resource-capped" failure reason (not a crash, not a silent drop --
# see FAILURE_REASONS and taxonomy_counts).
CORPUS_MAX_HYPOTHESES = 4000
CORPUS_MAX_PARSE_SECONDS = 10.0

# Sentinel default for _quoted_fallback/_attribution_fallback's `max_seconds`
# param (M58f2): a plain `= CORPUS_MAX_PARSE_SECONDS` default would bind the
# dial's value at function-DEFINITION time, so a caller that monkeypatches
# CORPUS_MAX_PARSE_SECONDS and then calls one of these fallbacks directly
# (no explicit max_seconds) would silently get the stale original value
# instead -- this module's other dials are read fresh from the global on
# every call (see this section's own docstring), and these two must match.
_MAX_SECONDS_UNSET = object()

# ---------------------------------------------------------------------------
# 1. sentence splitting + tokenization (matches the curriculum convention:
#    lowercase, space-separated punctuation -- see episode.py's own
#    f"{name} is in the {place} ." templates and input_encoder._split_sentences,
#    whose "sentence boundary is a bare ./?/! token" contract this produces).
# ---------------------------------------------------------------------------

_ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "no", "prof", "capt", "gen", "col"}
_SENT_END_CHARS = ".!?"
_CONTRACTION_SUFFIXES = ("n't", "'s", "'re", "'ve", "'ll", "'d", "'m")


def _split_raw_sentences(text: str) -> List[str]:
    """Deterministic sentence splitter over raw (un-tokenized) prose.

    Splits on ``.``/``!``/``?`` followed by whitespace, then merges a split
    back together when the token immediately before the punctuation is a
    common abbreviation (``mr``, ``dr``, ...) -- a cheap, good-enough guard
    for graded-reader/Gutenberg-style prose. Not a general-purpose sentence
    boundary detector; deterministic and dependency-free is the point.
    """
    normalized = " ".join(text.split())
    if not normalized:
        return []
    parts: List[str] = []
    buf = []
    i = 0
    n = len(normalized)
    while i < n:
        ch = normalized[i]
        buf.append(ch)
        if ch in _SENT_END_CHARS:
            # swallow a run of closing punctuation/quotes right after ("!?", '."')
            j = i + 1
            while j < n and normalized[j] in _SENT_END_CHARS + '"\'':
                buf.append(normalized[j])
                j += 1
            i = j
            # peek: is what precedes an abbreviation? if so, keep accumulating.
            word = "".join(buf).strip()
            last_word = word.rstrip(_SENT_END_CHARS + '"\'').split(" ")[-1].lower()
            at_end = i >= n
            next_is_space = i < n and normalized[i] == " "
            if last_word in _ABBREV and not at_end:
                continue
            if at_end or next_is_space:
                parts.append(word)
                buf = []
            continue
        i += 1
    if buf:
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
    return [p for p in parts if p]


def _tokenize_words(raw_sentence: str) -> List[str]:
    """Lowercase + split off contractions/punctuation as their own tokens."""
    text = raw_sentence.lower()
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("--", " -- ")
    for suf in _CONTRACTION_SUFFIXES:
        text = text.replace(suf, " " + suf)
    for ch in ".,!?;:()\"":
        text = text.replace(ch, f" {ch} ")
    return text.split()


def iter_sentences(text: str) -> List[str]:
    """Raw prose -> ``["mary is in the garden .", ...]`` (curriculum tokenization).

    Deterministic: the same ``text`` always yields the same list, in the
    same order.
    """
    out = []
    for raw in _split_raw_sentences(text):
        toks = _tokenize_words(raw)
        if toks:
            out.append(" ".join(toks))
    return out


# ---------------------------------------------------------------------------
# 2. parse_passage -- the SAME extraction shape as
#    nsm_ct.clause_reactor._context_steps, but returning raw (entity,
#    relation, value) string triples (+ provenance) instead of grounded
#    vectors, and classifying every failure with a taxonomy reason.
# ---------------------------------------------------------------------------

FAILURE_REASONS: Tuple[str, ...] = (
    "unknown-word",
    "no-parse",
    "multiple-parses-unresolved",
    "unsupported-construction",
    "pronoun-unresolvable",
    "no-relation-extracted",
    # M58c: a bare interjection/verbless fragment ("Thief!", "Woof, woof!")
    # -- distinguished from "no-parse" (a genuine grammar miss) because the
    # sentence was never expected to carry a fact in the first place; see
    # _no_verb_token.
    "fragment-skipped",
    # converter-memfix: quantum_parser's ParseResourceExceeded fired --
    # CORPUS_MAX_HYPOTHESES/CORPUS_MAX_PARSE_SECONDS capped this sentence's
    # ambiguous-anchor combinatorics before it could run away with memory or
    # time. An honest "we gave up on purpose" outcome, not "no-parse" (the
    # grammar may well have found a reading eventually) and not a crash.
    "parse-resource-capped",
)

# Ambiguity gate for "multiple-parses-unresolved": >=2 COMPLETE hypotheses
# (parser._parse_topk_one already keeps only complete parses when any exist)
# whose top1-top2 structural-score margin is at or below this -- i.e. the
# parser genuinely could not tell the readings apart. Kept tight (not a
# generic "top-2 are somewhat close" threshold) so this fires on genuine
# ties, not on ordinary score noise.
_AMBIGUITY_MARGIN = 1e-6


def _rerank_topk(graphs, scores):
    """M58c item 2's "cheap deterministic tie-breaker": among structural-
    score TIES in the top-K quantum_parser already returned, prefer (i)
    fewer unattached tokens, (ii) a shorter total dependency span (sum of
    ``|parent - child|`` over edges) -- both computed straight off the
    already-built :class:`~nsm_ct.quantum_adapter.HypGraph` view (``roots``
    is the SAME ``get_unconsumed()`` list quantum_parser's own scorer would
    use). Reorders ``graphs``/``scores`` in place of a full re-parse --
    scores stay attached to their (possibly moved) graph, so a caller's
    ``margin = scores[0] - scores[1]`` after this call reflects the
    tie-break's choice of "top1".

    Deliberately implemented HERE rather than inside quantum_parser's own
    ``completeness_key`` (where it was first tried): that function is
    consulted at MULTIPLE points during chart parsing, not just once at the
    end (hypotheses are pruned/re-sorted between grammar-rule steps), so
    even a strictly-refining extra tie-break component there can change
    which hypothesis SURVIVES an intermediate prune and, through that,
    which reading a later rule builds on -- a real, measured regression
    (test_bare_passive_yields_a_subject_only_clause: "the window was
    broken ." flipped predicate=broken -> predicate=was). This function
    instead re-ranks a FIXED, already-complete top-K list post-parse, which
    can never feed back into parsing/pruning -- it changes nothing about
    how any existing single-best-hypothesis caller (``_parse_graph``/
    ``_parse_tree``, what curriculum generation actually uses) behaves,
    since those never call ``_parse_topk_one``/this function at all.
    """
    from .quantum_adapter import HypGraph  # local import: avoid a hard import cycle at module load

    if len(graphs) < 2:
        return graphs, scores
    top = scores[0]
    tied = [i for i, s in enumerate(scores) if abs(s - top) <= _AMBIGUITY_MARGIN]
    if len(tied) < 2:
        return graphs, scores

    def _tie_key(i: int):
        g = graphs[i]
        if not isinstance(g, HypGraph):
            return (0, 0)
        unattached = len(g.roots)
        span = sum(abs(p - c) for _t, p, c in g.edges)
        return (-unattached, -span)

    tied_sorted = sorted(tied, key=_tie_key, reverse=True)
    if tied_sorted == tied:
        return graphs, scores  # already in tie-break order -- nothing moved
    order = list(range(len(graphs)))
    for slot, i in zip(tied, tied_sorted):
        order[slot] = i
    return [graphs[i] for i in order], [scores[i] for i in order]


_PUNCT_TOKENS = set(".,!?;:()\"'")

_UNSUPPORTED_CHECKS: List[Tuple[str, "callable"]] = [
    ("quotation", lambda toks: any(t in ('"', "``", "''") for t in toks)),
    ("question", lambda toks: "?" in toks),
    ("subordinate_clause", lambda toks: any(
        w in toks for w in ("who", "which", "that", "whom", "whose", "because", "although",
                             "since", "while", "if", "when", "before", "after", "though"))),
    ("coordination", lambda toks: "and" in toks or "or" in toks or "but" in toks),
    ("passive", lambda toks: "by" in toks and any(t.endswith(("ed", "en")) for t in toks)),
]


def _unsupported_signal(tokens: Sequence[str]) -> Optional[str]:
    for name, fn in _UNSUPPORTED_CHECKS:
        if fn(tokens):
            return name
    return None


# ---------------------------------------------------------------------------
# M58c item 4: quotation-mark stripping + quoted-span extraction. Scoped to
# the PARSE call only -- the sentence text kept for episode context/display
# (ParsedClause.sentence / ParseFailure.sentence) is always the ORIGINAL,
# untouched string; only the text handed to the parser is normalized. No
# curriculum sentence ever contains a quote-mark token, so this is a no-op
# on every existing curriculum caller.
# ---------------------------------------------------------------------------
_QUOTE_TOKENS = {'"', "``", "''", "'"}


def _strip_quotes(sent: str) -> str:
    """Drop quotation-mark tokens before feeding ``sent`` to the parser.

    Quotation marks are structurally inert punctuation this grammar doesn't
    model; an unconsumed PUNCT node can by itself prevent a hypothesis from
    ever completing (several of dev/PROSE_FAILURE_TAXONOMY.md's "no-parse"
    examples are otherwise perfectly parseable clauses with a stray leftover
    quote mark, e.g. the sentence splitter's cross-sentence quote leakage).
    """
    toks = [t for t in sent.split() if t not in _QUOTE_TOKENS]
    return " ".join(toks)


def _quoted_span(tokens: Sequence[str]) -> Optional[str]:
    """The token span between the first pair of quote-mark tokens, or (when
    only ONE quote-mark token is present -- the sentence splitter's cross-
    sentence quote leakage case) from that single mark to the nearer
    sentence boundary. ``None`` if ``tokens`` carries no quote mark at all.
    """
    idxs = [i for i, t in enumerate(tokens) if t in _QUOTE_TOKENS]
    if not idxs:
        return None
    if len(idxs) >= 2:
        span = tokens[idxs[0] + 1: idxs[-1]]
    elif idxs[0] == 0:
        span = tokens[1:]
    else:
        span = tokens[: idxs[0]]
    span = [t for t in span if t not in _QUOTE_TOKENS]
    return " ".join(span) if span else None


_lexicon_entry = None
_word_tag_dict = None
_is_bare_name = None
_looks_like_number = None
_tagger_load_failed = False


def _load_tagger_tables() -> bool:
    """Lazily load quantum_parser's open/closed-class word tables.

    Same ``qp_root`` resolution :class:`nsm_ct.input_encoder.ParserInputEncoder`
    uses (``src/nsm_ct/corpus.py`` -> ``../../../quantum_parser``). Returns
    ``False`` (and never raises) if quantum_parser isn't importable here --
    callers then skip unknown-word classification rather than erroring.
    """
    global _lexicon_entry, _word_tag_dict, _is_bare_name, _looks_like_number, _tagger_load_failed
    if _lexicon_entry is not None:
        return True
    if _tagger_load_failed:
        return False
    try:
        qp_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "quantum_parser"))
        if qp_root not in sys.path:
            sys.path.insert(0, qp_root)
        from src.parser.pos_tagger import (  # type: ignore
            WORD_TAG_DICT, lexicon_entry, is_bare_name_token, looks_like_number)

        _lexicon_entry = lexicon_entry
        _word_tag_dict = WORD_TAG_DICT
        _is_bare_name = is_bare_name_token
        _looks_like_number = looks_like_number
        return True
    except Exception:
        _tagger_load_failed = True
        return False


def _is_known_word(word: str) -> bool:
    """True if the parser's tagger has real (non-guessed) coverage for ``word``.

    Closed-class (``WORD_TAG_DICT``), a pronoun, present in the WordNet-
    derived open-class lexicon (``lexicon_entry`` -- which itself now also
    covers hyphenated compounds and an on-demand WordNet consult, M58c), a
    plain number/date, or a "bare name" token (M58c's NAME fallback tier --
    see ``is_bare_name_token``'s docstring) all count as "known". Anything
    else falls through to the tagger's blind suffix/default-noun heuristics
    -- the proxy this module uses for "unknown word" (archaic vocabulary,
    typos -- rare proper names are now covered by the NAME tier above).
    """
    if not _load_tagger_tables():
        return True  # tagger tables unavailable -- don't classify on a guess
    w = word.lower()
    if w in _word_tag_dict or w in _PRONOUNS:
        return True
    if _lexicon_entry(w):
        return True
    if _looks_like_number(w):
        return True
    return bool(_is_bare_name(w))


def _word_tag_candidates(word: str):
    """EVERY plausible POS tag for ``word`` -- closed-class dict entry, or
    every tag the (possibly multi-POS) lexicon entry exposes, mirroring
    quantum_parser's own "every option in every slot" philosophy
    (get_possible_tags) rather than just the frequency-default reading.
    Consulting the FULL tag set (not just entry[0]) matters here: some
    static-lexicon words carry a non-verb PRIMARY sense purely by SemCor
    frequency/sense-count ordering despite also being common verbs (e.g.
    "opened" ranks ADJ first in the generated lexicon -- a real WordNet
    adjective sense exists, from an entirely different lemma than the
    verb-inflection generator -- but is still very much also a verb)."""
    if not _load_tagger_tables():
        return ()
    w = word.lower()
    if w in _word_tag_dict:
        return (_word_tag_dict[w],)
    entry = _lexicon_entry(w)
    if entry:
        return tuple(tag for tag, _subs in entry)
    return ()


def _no_verb_token(content_tokens: Sequence[str]) -> bool:
    """True if NO content token has ANY plausible VERB/AUX reading -- a bare
    interjection or nominal fragment ("Thief!", "Woof, woof!") rather than a
    genuine parse failure (M58c's "fragment-skipped" taxonomy code).
    ``False`` (never a fragment) if the tagger tables aren't available,
    matching every other tagger-dependent classifier in this module's
    fail-open contract."""
    if not _load_tagger_tables():
        return False
    from src.parser.enums import Tag  # type: ignore  # local import: qp_root already on sys.path

    for t in content_tokens:
        if any(tag in (Tag.VERB, Tag.AUX) for tag in _word_tag_candidates(t)):
            return False
    return True


# Round 2 item 1: is_entity() alone (clause.is_entity) only ever recognizes
# the 6 FIXED curriculum names (episode._NAMES) plus pronouns -- on real
# prose, a character introduced by an ordinary proper name ("alice",
# "buster", "otter") NEVER matches it, so the passage registry it gates
# stays empty for almost every real passage. This is the measured root
# cause of the pronoun-unresolvable bucket's size on real text
# (RESEARCH_NOTES M58c): the registry existed (M58c item 5) but had almost
# nothing to register. quantum_parser's own M58c "bare name" fallback tier
# (``is_bare_name_token`` -- see ``_is_known_word``'s docstring) already
# identifies exactly this class of token for a DIFFERENT purpose (lexicon
# coverage): a word the tagger has no closed-class/WordNet/number coverage
# for, that doesn't match a common inflectional suffix either -- "in real
# narrative prose this is overwhelmingly a proper name" (that function's own
# docstring). Reusing it here widens the registry to real-text proper names
# too, purely additively (every is_entity() match still matches).
def _is_registrable_entity(word: str) -> bool:
    if is_entity(word):
        return True
    if not _load_tagger_tables():
        return False
    return bool(_is_bare_name(word))


# Round 2 item 1: gender compatibility between a pronoun and a candidate
# antecedent NAME, via membrane.PRONOUN_MORPHOLOGY (the pronoun's own
# gender) and membrane.NAME_GENDER (a name's gender -- ONLY known for the 6
# fixed curriculum names; real-text names carry no gender signal at all).
# Unknown on EITHER side never rules a candidate out (perception "never
# guesses" a contradiction it can't support) -- this only EXCLUDES a
# candidate when both sides are known and genuinely conflict (a curriculum
# name of the wrong gender re-appearing in prose). For the overwhelming
# majority of real-text names (gender unknown), every antecedent stays
# compatible; the real filtering job is left to the resolver (evidence
# attr:gender), per the LOCKED design.
def _gender_compatible(pronoun: str, name: str) -> bool:
    p_gender = PRONOUN_MORPHOLOGY.get(pronoun, ("unknown", "sg", "3"))[0]
    n_gender = NAME_GENDER.get(name)
    if p_gender == "unknown" or n_gender is None:
        return True
    return p_gender == n_gender


class _PassageRegistry:
    """A simple passage-level entity registry (M58c item 5): the most
    recently mentioned entity NAME, threaded through :func:`parse_passage`
    sentence by sentence, so a pronoun's antecedent PRESENCE can be checked
    before flagging "pronoun-unresolvable" -- the design's own wording
    ("an antecedent EARLIER in the passage"), not full gender/number
    agreement. :meth:`register` for sentence N is always called AFTER
    :meth:`nearest` was consulted for sentence N's own pronouns, so a name
    never counts as its own same-sentence antecedent.
    """

    def __init__(self) -> None:
        self._recent: List[str] = []

    def nearest(self) -> Optional[str]:
        return self._recent[-1] if self._recent else None

    def candidates(self) -> List[str]:
        """Every registered antecedent, MOST-RECENTLY-MENTIONED FIRST (round
        2 item 1's own candidate-set ordering) -- the reverse of
        ``_recent``'s append-order/move-to-end bookkeeping."""
        return list(reversed(self._recent))

    def register(self, clauses) -> None:
        for cl in clauses:
            for _rel, arg in cl.args:
                tok = (arg.token or "").lower()
                if tok and tok not in _PRONOUNS and _is_registrable_entity(tok):
                    if tok in self._recent:
                        self._recent.remove(tok)
                    self._recent.append(tok)


@dataclass
class ParsedClause:
    """One extracted ``(entity, relation, value)`` fact, grounded in its sentence."""

    sentence_index: int
    sentence: str
    entity: str
    relation: str
    value: str
    predicate: str = ""
    # M58c item 2: set (a HypothesisCandidateSet) when this sentence's top-K
    # parses tied within _AMBIGUITY_MARGIN -- the top-1 reading below is
    # still what's extracted; this is the "we weren't sure" flag
    # taxonomy_counts reads to report "parsed-ambiguous" instead of "ok".
    hypotheses: Optional[HypothesisCandidateSet] = field(default=None, repr=False)
    # M58c item 4: "text" (default), "quoted" (a bare quoted span, speaker
    # unknown), or (round 2 item 2) "quoted:<speaker>" -- set when this
    # clause came from a quoted SPAN inside a longer sentence rather than
    # the sentence as a whole; "quoted:<speaker>" additionally names the
    # attribution frame's speaker (see clause.strip_attribution).
    source: str = "text"
    # Round 2 item 1 (PROSE PRONOUNS TO THE RESOLVER): non-None marks this
    # clause's ENTITY as a personal pronoun resolved against the passage
    # registry rather than a literal name mentioned in this sentence --
    # the gender-compatible antecedent candidates (most-recent-first;
    # ``entity`` itself is just candidates[0], kept for backward-compatible
    # question/quality-filter use -- see make_episodes' own note on why
    # this is safe). ``None`` (the default) for every ordinary clause,
    # keeping this byte-identical for every pre-round-2 caller.
    pronoun_candidates: Optional[Tuple[str, ...]] = None


@dataclass
class ParseFailure:
    """One sentence that produced no usable fact, tagged with a taxonomy reason."""

    sentence_index: int
    sentence: str
    reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in FAILURE_REASONS:
            raise ValueError(f"unknown failure reason {self.reason!r}; must be one of {FAILURE_REASONS}")


ParseOutcome = Union[ParsedClause, ParseFailure]


def _pronoun_candidates_for(subj: str, registry: Optional["_PassageRegistry"]) -> List[str]:
    """Round 2 item 1: gender-compatible antecedents for pronoun ``subj``,
    most-recently-mentioned first -- ``[]`` (never ``None``) when there is
    no registry or no compatible candidate, so callers can test truthiness
    directly."""
    if registry is None:
        return []
    return [n for n in registry.candidates() if _gender_compatible(subj, n)]


def _extract_triples(idx: int, sent: str, clauses, registry: Optional["_PassageRegistry"],
                      source: str = "text") -> Tuple[List[ParsedClause], bool]:
    """Mirrors nsm_ct.clause_reactor._context_steps's own extraction exactly
    (OBJECT-bearing transfer clause -> one step per other role; plain
    SUBJECT+single-other-role clause -> one step). A pronoun-bearing role is
    resolved against ``registry``'s nearest antecedent (M58c item 5 -- an
    entity mentioned in an EARLIER sentence) when one is available;
    otherwise it's reported via the returned ``pronoun_hit`` flag exactly as
    before this registry existed (``registry=None`` reproduces the old
    behavior exactly: every pronoun role is unresolved).

    Round 2 item 1 (PROSE PRONOUNS TO THE RESOLVER): a pronoun filling the
    SUBJECT+single-other-role shape's SUBJECT (an ADDRESS-position slot --
    see nsm_ct.clause_reactor._prose_steps' own docstring for why only this
    shape, not the OBJECT-bearing one below, is redirect-eligible) is
    resolved via :func:`_pronoun_candidates_for` (gender-compatible,
    most-recent-first) rather than a single deterministic guess:
    ``entity`` is bound to the single MOST RECENT candidate (so question
    generation/quality-filtering/distractor-pooling downstream still see an
    ordinary named entity -- see make_episodes' own note on why this can
    never leak a pronoun token into a held-out question) but
    ``pronoun_candidates`` carries the FULL compatible list, which is what
    :func:`nsm_ct.clause_reactor._prose_steps` actually grounds through the
    resolver (no gold index -- prose carries no ground truth for which
    candidate is correct). Zero compatible candidates -> ``pronoun_hit``
    (the pre-existing "pronoun-unresolvable" outcome), unchanged.

    Round 2 item 3 (PLAIN-OBJECT QUESTIONS): the OBJECT-bearing branch's
    pronoun-bearing OTHER roles keep the OLD deterministic
    ``registry.nearest()`` substitution -- there, the unresolved slot is the
    triple's VALUE (e.g. "he ate the apple" -> entity=apple, value=he's
    antecedent), an address the resolver's collapse machinery has no
    contract for redirecting (only the ENTITY axis can be redirected); see
    module docstring. A "clean" transitive clause (SUBJECT + a single OBJECT
    argument, nothing else -- e.g. "the man ate the apple") additionally
    gets the COMPLEMENTARY fact in the other direction (entity=SUBJECT,
    relation=OBJECT, value=the object word), askable via the existing
    "what" -> OBJECT :func:`nsm_ct.clause_reactor._queried_role` convention
    -- purely additive (never replaces the AGENT-direction triple already
    extracted above), and skipped when the subject is itself a pronoun (out
    of scope -- see item 1's own address-vs-value note; the AGENT-direction
    triple above already covers that case).
    """
    triples: List[ParsedClause] = []
    pronoun_hit = False
    for cl in clauses:
        pred = (cl.predicate or "").lower()
        obj_tok = next(((arg.token or "").lower() for rel, arg in cl.args if rel == "OBJECT"), None)
        if obj_tok:
            if obj_tok in _PRONOUNS:
                # the transferred OBJECT itself is a pronoun ("gave it to
                # john") -- not covered by the (person-name) registry.
                pronoun_hit = True
                continue
            other_roles = [(rel, arg) for rel, arg in cl.args if rel != "OBJECT"]
            for rel, arg in other_roles:
                tok = (arg.token or "").lower()
                if not tok:
                    continue
                mapped = _TRANSFER_ROLE_MAP.get(rel, rel)
                resolved = tok
                if tok in _PRONOUNS:
                    resolved = registry.nearest() if registry is not None else None
                    if not resolved:
                        pronoun_hit = True
                        continue
                triples.append(ParsedClause(idx, sent, obj_tok, mapped, resolved, pred, source=source))
            # item 3: the complementary OBJECT-direction fact for a CLEAN
            # transitive clause (subject + this one object, nothing else).
            if len(other_roles) == 1 and other_roles[0][0] == "SUBJECT":
                subj_tok = (other_roles[0][1].token or "").lower()
                if subj_tok and subj_tok not in _PRONOUNS:
                    triples.append(ParsedClause(idx, sent, subj_tok, "OBJECT", obj_tok, pred, source=source))
            continue
        subj = place = None
        for rel, arg in cl.args:
            if rel == "SUBJECT":
                subj = (arg.token or "").lower()
            elif rel == "PLACE":
                place = (arg.token or "").lower()
        # item 3: a "clean" catch-all fact for a clause with a SUBJECT and
        # EXACTLY ONE other role that ISN'T "PLACE" (SOURCE, or any raw
        # preposition relation _prep_relation passes through unmapped --
        # "about"/"with"/"through"/... -- see module docstring) -- asked via
        # the existing "what" -> OBJECT queried-role convention. Scoped to
        # "PLACE absent, exactly one other role" so it never overrides or
        # duplicates the dedicated PLACE branch below, and never fires
        # alongside extra structure this module doesn't otherwise model.
        other_non_place = [(rel, arg) for rel, arg in cl.args if rel not in ("SUBJECT", "PLACE")]
        if place is None and subj and len(other_non_place) == 1:
            value = (other_non_place[0][1].token or "").lower()
            if value:
                if subj in _PRONOUNS:
                    compatible = _pronoun_candidates_for(subj, registry)
                    if compatible:
                        triples.append(ParsedClause(idx, sent, compatible[0], "OBJECT", value, pred,
                                                     source=source, pronoun_candidates=tuple(compatible)))
                    else:
                        pronoun_hit = True
                else:
                    triples.append(ParsedClause(idx, sent, subj, "OBJECT", value, pred, source=source))
            continue
        if subj and place:
            resolved_subj = subj
            pronoun_candidates = None
            if subj in _PRONOUNS:
                compatible = _pronoun_candidates_for(subj, registry)
                if not compatible:
                    pronoun_hit = True
                    continue
                resolved_subj = compatible[0]
                pronoun_candidates = tuple(compatible)
            triples.append(ParsedClause(idx, sent, resolved_subj, "PLACE", place, pred, source=source,
                                         pronoun_candidates=pronoun_candidates))
    return triples, pronoun_hit


def _hypothesis_candidates(idx: int, sent: str, graphs, scores) -> HypothesisCandidateSet:
    """M58c item 2: build the M55a membrane's ``HypothesisCandidateSet`` for
    a low-margin top-K (:data:`_AMBIGUITY_MARGIN`) -- one candidate per
    hypothesis graph, each carrying ITS OWN representative (query_entity,
    query_relation) address (the first extractable role, mirroring
    :func:`_extract_triples`'s own priority: an OBJECT-bearing transfer
    clause's non-OBJECT role, else a SUBJECT+PLACE fact) so a downstream
    resolver could in principle check which reading coheres with memory.
    Priors are each hypothesis's structural score, renormalized to sum 1
    (uniform 1/k if every score is <=0, degenerate but never a crash).
    """
    readings: List[Tuple[str, float, str, str]] = []
    for i, g in enumerate(graphs):
        clauses, _links = extract_discourse(g)
        qe = qr = ""
        for cl in clauses:
            obj_tok = next(((arg.token or "").lower() for rel, arg in cl.args if rel == "OBJECT"), None)
            if obj_tok:
                for rel, arg in cl.args:
                    if rel == "OBJECT":
                        continue
                    tok = (arg.token or "").lower()
                    if tok:
                        qe, qr = obj_tok, _TRANSFER_ROLE_MAP.get(rel, rel)
                        break
                if qe:
                    break
            else:
                subj = place = None
                for rel, arg in cl.args:
                    if rel == "SUBJECT":
                        subj = (arg.token or "").lower()
                    elif rel == "PLACE":
                        place = (arg.token or "").lower()
                if subj and place:
                    qe, qr = subj, "PLACE"
                    break
        readings.append((f"h{i}", max(float(scores[i]), 0.0), qe, qr))
    total = sum(prior for _k, prior, _qe, _qr in readings) or 1.0
    readings = [(k, prior / total, qe, qr) for k, prior, qe, qr in readings]
    return hypothesis_candidate_set(readings, provenance={"sentence_index": idx, "sentence": sent})


def _quoted_fallback(idx: int, sent: str, tokens: Sequence[str], parser,
                      registry: Optional["_PassageRegistry"],
                      max_seconds: Optional[float] = _MAX_SECONDS_UNSET,
                      ) -> Optional[List[ParsedClause]]:
    """M58c item 4: when the sentence AS A WHOLE yields nothing, try the
    quoted SPAN inside it (if any) on its own -- e.g. narration wrapped
    around a complete quoted clause. Facts from the quoted span are tagged
    ``source="quoted"``; attribution to the speaker is out of scope (the
    quoted clause's own facts are extracted, nothing about who said it).
    ``None`` (not a fallback failure list) if there's no quote mark, or the
    span itself doesn't parse/extract either -- callers fall through to
    their normal failure classification in that case.

    ``max_seconds`` (M58f2): the caller's REMAINING per-sentence time
    budget, not a fresh ``CORPUS_MAX_PARSE_SECONDS`` allowance -- see
    :func:`_parse_one_sentence_uncapped`'s docstring for why a fallback
    attempt must not reset the clock. Omit it (as every direct/test caller
    does) to fall back to the live ``CORPUS_MAX_PARSE_SECONDS`` global.
    """
    if max_seconds is _MAX_SECONDS_UNSET:
        max_seconds = CORPUS_MAX_PARSE_SECONDS
    span = _quoted_span(tokens)
    if not span:
        return None
    graphs, _scores, _margin = parser._parse_topk_one(
        span, k=4, max_hypotheses=CORPUS_MAX_HYPOTHESES, max_seconds=max_seconds)
    if not graphs:
        return None
    clauses, _links = extract_discourse(graphs[0])
    if not clauses:
        return None
    triples, _pronoun_hit = _extract_triples(idx, sent, clauses, registry, source="quoted")
    if not triples:
        return None
    if registry is not None:
        registry.register(clauses)
    return triples


# Round 2 item 2 (ATTRIBUTION-WRAPPED NARRATION): a strengthened cousin of
# _quoted_fallback above -- tries clause.strip_attribution's recognized
# quote-comma-said-X / X-said-quote frames FIRST (see that function's own
# docstring for the two shapes), which additionally identifies WHO said the
# quoted clause. Falls back to the plain quoted-SPAN behavior (speaker
# unknown) when no attribution-verb pattern is recognized at all -- callers
# try this BEFORE _quoted_fallback so a genuine attribution frame is always
# tagged with its speaker rather than silently downgraded to the generic
# "quoted" source.
def _attribution_fallback(idx: int, sent: str, tokens: Sequence[str], parser,
                           registry: Optional["_PassageRegistry"],
                           max_seconds: Optional[float] = _MAX_SECONDS_UNSET,
                           ) -> Optional[List[ParsedClause]]:
    """``max_seconds`` (M58f2): the caller's REMAINING per-sentence time
    budget -- see :func:`_quoted_fallback`'s docstring."""
    if max_seconds is _MAX_SECONDS_UNSET:
        max_seconds = CORPUS_MAX_PARSE_SECONDS
    stripped = strip_attribution(tokens)
    if stripped is None:
        return None
    core, speaker = stripped
    if not core:
        return None
    graphs, _scores, _margin = parser._parse_topk_one(
        " ".join(core), k=4, max_hypotheses=CORPUS_MAX_HYPOTHESES, max_seconds=max_seconds)
    if not graphs:
        return None
    clauses, _links = extract_discourse(graphs[0])
    if not clauses:
        return None
    tag = f"quoted:{speaker}" if speaker else "quoted"
    triples, _pronoun_hit = _extract_triples(idx, sent, clauses, registry, source=tag)
    if not triples:
        return None
    if registry is not None:
        registry.register(clauses)
    return triples


def _parse_one_sentence(idx: int, sent: str, parser,
                         registry: Optional["_PassageRegistry"] = None) -> List[ParseOutcome]:
    """Wraps :func:`_parse_one_sentence_uncapped` so a
    ``ParseResourceExceeded`` (CORPUS_MAX_HYPOTHESES/CORPUS_MAX_PARSE_SECONDS
    tripped by quantum_parser on this sentence's ambiguous-anchor
    combinatorics -- see this module's docstring for those dials) becomes an
    honest ``"parse-resource-capped"`` taxonomy outcome instead of an
    uncaught exception/OOM. Caught here (one place) rather than in each of
    ``_parse_one_sentence_uncapped``/``_quoted_fallback``/
    ``_attribution_fallback`` since all three share the same per-sentence
    call stack and the same fallback-exhausted meaning: this sentence's
    parse was too combinatorially expensive, full stop.
    """
    # Local import: quantum_parser's repo root only lands on sys.path once
    # ParserInputEncoder._init_adapter runs (module load time of
    # nsm_ct.corpus is too early to import it at the top of this file).
    from src.parser.quantum_parser import ParseResourceExceeded
    try:
        return _parse_one_sentence_uncapped(idx, sent, parser, registry=registry)
    except ParseResourceExceeded as exc:
        # Eager release (task requirement): a capped sentence can leave
        # thousands of deep-copied Hypothesis objects reachable only via the
        # exception's traceback frames until those frames unwind -- gc.collect()
        # here (cheap relative to the parse that just got aborted) guarantees
        # they're actually freed before the next sentence starts, rather than
        # drifting to whenever the cyclic collector next runs on its own.
        # Deliberately NOT a `finally` (measured hang, M58f2): this process's
        # live heap includes the multi-hundred-thousand-object USVS/WordNet
        # table loaded once at startup, so a full gc.collect() costs a fixed
        # ~0.3s no matter how little garbage this one sentence produced --
        # calling it unconditionally on every one of a passage's (mostly
        # uncapped, fast) sentences turns that into minutes of pure overhead
        # over a real corpus. Only a capped sentence actually needs it.
        gc.collect()
        return [ParseFailure(idx, sent, "parse-resource-capped", detail=str(exc))]


def _parse_one_sentence_uncapped(idx: int, sent: str, parser,
                                  registry: Optional["_PassageRegistry"] = None) -> List[ParseOutcome]:
    tokens = sent.split()
    content_tokens = [t for t in tokens if t not in _PUNCT_TOKENS and any(c.isalpha() for c in t)]
    unknown = sorted({t for t in content_tokens if not _is_known_word(t)})
    if unknown:
        return [ParseFailure(idx, sent, "unknown-word", detail=",".join(unknown[:5]))]

    if getattr(parser, "_parser", None) is None or not hasattr(parser, "_parse_topk_one"):
        return [ParseFailure(idx, sent, "no-parse", detail="parser unavailable")]

    # M58f2: one WALL-CLOCK DEADLINE shared by the primary attempt AND both
    # fallbacks below, instead of each of the (up to) three parse attempts
    # getting its own fresh CORPUS_MAX_PARSE_SECONDS allowance. A real-text
    # sentence that structurally fails to parse (most of them, per
    # dev/PROSE_FAILURE_TAXONOMY.md) pays the full cap on the primary
    # attempt and then tries attribution- and quoted-fallback in turn --
    # uncapped-per-attempt, that is up to 3x CORPUS_MAX_PARSE_SECONDS of
    # wall clock for ONE sentence, which is what actually made a full-corpus
    # pass (thousands of mostly-unparseable real sentences) run for tens of
    # minutes despite every individual parse call being "capped". Sharing
    # one deadline bounds one sentence's total cost to CORPUS_MAX_PARSE_SECONDS,
    # matching what CORPUS_MAX_PARSE_SECONDS's docstring already promises.
    # ``None`` (dial off) means no deadline at all -- every remaining()
    # call below returns None too, so behavior is byte-identical to before
    # this change when caps are disabled.
    deadline = (time.monotonic() + CORPUS_MAX_PARSE_SECONDS
                if CORPUS_MAX_PARSE_SECONDS is not None else None)

    def _remaining() -> Optional[float]:
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    # M58c item 4: quotation marks are stripped before the PARSE call only --
    # `sent`/`tokens` (kept for context/detail text and the unsupported-
    # construction/quoted-span checks) are untouched.
    graphs, scores, margin = parser._parse_topk_one(
        _strip_quotes(sent), k=4, max_hypotheses=CORPUS_MAX_HYPOTHESES, max_seconds=_remaining())
    # M58c item 2's tie-breaker (see _rerank_topk's own docstring for why it
    # lives here rather than in quantum_parser's scorer): re-picks "top1"
    # among score-tied candidates only; margin is recomputed since the
    # reordering can change which score sits at index 0/1.
    graphs, scores = _rerank_topk(graphs, scores)
    margin = (scores[0] - scores[1]) if len(scores) > 1 else margin
    if not graphs:
        if _no_verb_token(content_tokens):
            return [ParseFailure(idx, sent, "fragment-skipped")]
        fallback = _attribution_fallback(idx, sent, tokens, parser, registry, max_seconds=_remaining()) or \
            _quoted_fallback(idx, sent, tokens, parser, registry, max_seconds=_remaining())
        if fallback:
            return fallback
        return [ParseFailure(idx, sent, "no-parse")]

    # M58c item 2: a low-margin top-K is no longer an automatic failure --
    # extraction still runs on the top-1 reading; `ambiguous` only decides
    # the taxonomy tag (see taxonomy_counts) and whether a
    # HypothesisCandidateSet is attached below.
    ambiguous = len(graphs) >= 2 and margin <= _AMBIGUITY_MARGIN

    clauses, _links = extract_discourse(graphs[0])
    if not clauses:
        if _no_verb_token(content_tokens):
            return [ParseFailure(idx, sent, "fragment-skipped")]
        fallback = _attribution_fallback(idx, sent, tokens, parser, registry, max_seconds=_remaining()) or \
            _quoted_fallback(idx, sent, tokens, parser, registry, max_seconds=_remaining())
        if fallback:
            return fallback
        sig = _unsupported_signal(tokens)
        if sig:
            return [ParseFailure(idx, sent, "unsupported-construction", detail=sig)]
        return [ParseFailure(idx, sent, "no-relation-extracted")]

    triples, pronoun_hit = _extract_triples(idx, sent, clauses, registry)

    if registry is not None:
        registry.register(clauses)

    if triples:
        if ambiguous:
            hyp_set = _hypothesis_candidates(idx, sent, graphs, scores)
            for t in triples:
                t.hypotheses = hyp_set
        return triples
    if pronoun_hit:
        return [ParseFailure(idx, sent, "pronoun-unresolvable")]
    fallback = _attribution_fallback(idx, sent, tokens, parser, registry, max_seconds=_remaining()) or \
        _quoted_fallback(idx, sent, tokens, parser, registry, max_seconds=_remaining())
    if fallback:
        return fallback
    sig = _unsupported_signal(tokens)
    if sig:
        return [ParseFailure(idx, sent, "unsupported-construction", detail=sig)]
    return [ParseFailure(idx, sent, "no-relation-extracted")]


def parse_passage(sentences: Sequence[str], parser) -> List[ParseOutcome]:
    """Parse every sentence of one passage through the real parser path.

    Returns a flat list mixing :class:`ParsedClause` (>=1 per sentence that
    yielded any fact -- a transfer sentence can yield several) and
    :class:`ParseFailure` (exactly 1 per sentence that yielded none). Use
    :func:`taxonomy_counts` to collapse this back to one outcome per
    sentence. M58c: threads one :class:`_PassageRegistry` across the whole
    passage (item 5) so a pronoun with an antecedent earlier in the SAME
    passage can resolve.
    """
    results: List[ParseOutcome] = []
    registry = _PassageRegistry()
    for idx, sent in enumerate(sentences):
        results.extend(_parse_one_sentence(idx, sent, parser, registry=registry))
    return results


def taxonomy_counts(results: Sequence[ParseOutcome]) -> Counter:
    """One outcome per ``sentence_index``: ``"ok"`` if it produced any
    (non-ambiguous, non-pronoun-resolved) :class:`ParsedClause`,
    ``"parsed-ambiguous"`` if its clause(s) came from a low-margin top-K
    (M58c item 2 -- :attr:`ParsedClause.hypotheses` set), ``"parsed-
    pronoun-resolved"`` if its clause's entity is a pronoun resolved
    against the passage registry (round 2 item 1 --
    :attr:`ParsedClause.pronoun_candidates` set; checked after ambiguity so
    a clause that happens to be BOTH is reported as ambiguous, the rarer
    and more informative of the two), else its (single, consistent) failure
    reason. Exhaustive over :data:`FAILURE_REASONS` +
    ``{"ok", "parsed-ambiguous", "parsed-pronoun-resolved"}``.
    """
    outcomes: Dict[int, str] = {}
    for r in results:
        if isinstance(r, ParsedClause):
            if r.hypotheses is not None:
                outcomes[r.sentence_index] = "parsed-ambiguous"
            elif r.pronoun_candidates is not None:
                outcomes[r.sentence_index] = "parsed-pronoun-resolved"
            else:
                outcomes[r.sentence_index] = "ok"
        elif r.sentence_index not in outcomes:
            outcomes[r.sentence_index] = r.reason
    return Counter(outcomes.values())


# ---------------------------------------------------------------------------
# 3. make_episodes -- hold out one eligible clause, ask it via the EXISTING
#    queried-role table (nsm_ct.clause_reactor._queried_role /
#    _question_entity); every sentence stays in context.
# ---------------------------------------------------------------------------

# Only relations nsm_ct.clause_reactor._queried_role can actually recover
# from question text get a template. Round 2 item 3 (PLAIN-OBJECT
# QUESTIONS) adds "OBJECT", using the "what" -> OBJECT keyword that table
# already reserves for exactly this ("not yet exercised by any curriculum
# level but resolvable here", its own comment says) -- no clause_reactor.py
# change needed. Every other raw preposition label _extract_triples's
# catch-all might pass through (SOURCE included) is grounded as "OBJECT" by
# that same catch-all, so this one template covers all of them; anything
# _extract_triples doesn't map to PLACE/RECIPIENT/AGENT/OBJECT at all is
# deliberately left un-askable rather than inventing a keyword the rest of
# the (frozen) pipeline doesn't know about.
_RELATION_QUESTION_TEMPLATE: Dict[str, str] = {
    "PLACE": "where is the {e} ?",
    "RECIPIENT": "who has the {e} ?",
    "AGENT": "who gave the {e} ?",
    "OBJECT": "what does the {e} have ?",
}


# M58d item 2 (the episode-quality filter, RESEARCH_NOTES "M58b"'s first
# diagnosed defect): a self-generated episode is only as good as the
# held-out clause it quizzes -- extraction NOISE surviving into the
# episode's entity/value turns the question into "where is the himself ?"
# or the gold answer into "got", testing memory of nonsense rather than
# comprehension. :func:`_quality_reject_reason` below is consulted ONLY
# from :func:`make_episodes`'s own candidate loop (never from the
# curriculum's own step-builders in clause_reactor.py -- this filter has
# no effect on any curriculum path). Reported under the new
# "episode-rejected-quality" taxonomy code (see :data:`make_episodes`'s
# ``reject_stats`` kwarg), a distinct namespace from :data:`FAILURE_REASONS`
# (a per-SENTENCE parse outcome; this is a per-CANDIDATE episode-quality
# gate over already-parsed clauses, so it does not join that tuple).
EPISODE_QUALITY_CODE = "episode-rejected-quality"

# himself/herself/itself/themselves: reflexives are NOT in clause.py's
# _PRONOUNS (a different grammatical role -- see that module's own note),
# so _extract_triples's pronoun-resolution branch never catches them; they
# pass straight through as a literal entity/value string instead of being
# resolved or flagged "pronoun-unresolvable". membrane.PRONOUN_MORPHOLOGY
# additionally catches every other unresolved pronominal form _extract_triples
# also lets through unresolved (possessives "his"/"her"/"their", Spanish
# clitics "le"/"la"/"lo"/"los"/"las") -- everything _PRONOUNS itself already
# resolves-or-drops never reaches here as a held-out entity/value at all.
_REFLEXIVES = {"himself", "herself", "itself", "themselves"}


def _is_pronoun_or_reflexive(word: str) -> bool:
    w = (word or "").lower()
    return w in PRONOUN_MORPHOLOGY or w in _REFLEXIVES


def _is_verb_tagged(word: str) -> bool:
    """True if ``word``'s PRIMARY tagger reading is VERB/AUX -- extraction
    noise surviving as a held-out value (e.g. gold="got", a mis-extracted
    verb, not a place/recipient/agent). Uses the PRIMARY (first) candidate
    tag, not "any" reading, so an ordinary place-noun that also happens to
    have a secondary verb sense ("garden", "house", "table" -- all NOUN-
    primary in the lexicon) is never rejected; only words whose tagger
    entry itself ranks VERB/AUX first (walked, ran, got, watched) are.
    """
    if not _load_tagger_tables():
        return False
    from src.parser.enums import Tag  # type: ignore  # local import: qp_root already on sys.path

    tags = _word_tag_candidates(word)
    return bool(tags) and tags[0] in (Tag.VERB, Tag.AUX)


def _is_closed_class(word: str) -> bool:
    """True if ``word`` is one of the tagger's ~300 closed-class function
    words (determiners, prepositions, conjunctions, ...) -- asking a
    question about one ("where is the to ?") means the entity token itself
    is extraction noise, not a real referring expression.
    """
    if not _load_tagger_tables():
        return False
    return (word or "").lower() in _word_tag_dict


def _quality_reject_reason(held: ParsedClause) -> Optional[str]:
    """None if ``held`` is fit to be a held-out episode clause, else a short
    reason code (for tests/debugging -- callers only need the None/not-None
    signal). See the three checks' own docstrings for what each catches.
    """
    if _is_pronoun_or_reflexive(held.entity):
        return "entity-pronoun"
    if _is_pronoun_or_reflexive(held.value):
        return "value-pronoun"
    if _is_verb_tagged(held.value):
        return "value-verb"
    if _is_closed_class(held.entity):
        return "entity-closed-class"
    return None


def make_episodes(passage_clauses: Sequence[ParseOutcome], *, holdout: str = "last",
                   seed: int = 0, doc_id: str = "",
                   distractor_pool: Optional[Sequence[ParsedClause]] = None,
                   reject_stats: Optional[Counter] = None) -> List[Episode]:
    """Hold out one eligible clause of the passage and ask a self-generated question.

    ``passage_clauses`` is one passage's full :func:`parse_passage` output
    (both :class:`ParsedClause` and :class:`ParseFailure` entries -- the
    failures are needed to reconstruct every sentence of the passage for
    context, even though only the successes are holdout candidates).

    Exactly one :class:`~nsm_ct.episode.Episode` is produced per passage (the
    LOCKED design's "hold out ONE parsed clause"): ``holdout="last"`` walks
    eligible clauses from the end of the passage backward and takes the
    first with >=2 real distractor values; ``holdout="random"`` shuffles
    (seeded) and does the same. A passage with no eligible clause with
    enough distractors yields ``[]`` ("skip the clause otherwise").

    Context = every sentence of the passage, UNCHANGED (nothing removed);
    the question just names the held-out clause's entity and asks for its
    value via the relation's template (:data:`_RELATION_QUESTION_TEMPLATE`),
    grounded through the SAME "queried role" keyword table
    :func:`nsm_ct.clause_reactor._queried_role`/``_question_entity`` use, so
    :func:`nsm_ct.clause_reactor.build_clause_batch` needs no new branch.

    ``distractor_pool``: extra :class:`ParsedClause` values (e.g. from other
    passages of the same corpus) to draw same-relation distractors from when
    the passage alone doesn't have enough ("elsewhere in the passage/corpus").

    ``reject_stats``: an optional :class:`collections.Counter` the caller
    passes in to accumulate the M58d episode-quality filter's rejections
    (:data:`EPISODE_QUALITY_CODE`, one increment per candidate clause
    :func:`_quality_reject_reason` flags, whether or not the passage goes
    on to yield an episode from a later candidate) -- corpus-wide honesty
    for a code that, unlike :data:`FAILURE_REASONS`, is per-CANDIDATE
    rather than per-sentence and so can't live in :func:`taxonomy_counts`.
    """
    if holdout not in ("last", "random"):
        raise ValueError(f"holdout must be 'last' or 'random', got {holdout!r}")

    by_idx: Dict[int, str] = {}
    for r in passage_clauses:
        by_idx.setdefault(r.sentence_index, r.sentence)
    if not by_idx:
        return []
    context = [by_idx[i] for i in sorted(by_idx)]

    clauses = [r for r in passage_clauses if isinstance(r, ParsedClause)]
    # Round 2 item 1: a pronoun-resolved clause's ``entity`` is the passage
    # registry's own MOST-RECENT gender-compatible candidate
    # (_extract_triples), never the literal pronoun token -- holding one
    # out asks "where is the {e} ?" naming a NAMED entity exactly as any
    # other clause would (no "where is the he ?" is possible), and
    # _quality_reject_reason's own entity-pronoun/entity-closed-class
    # checks below re-confirm this defensively. This is what makes holding
    # such a clause out safe despite the resolver's own gold_index being
    # withheld (pronoun_candidates carries the full candidate list for the
    # BATCH-BUILD resolver step, gold_antecedent=None; this module's own
    # "most-recent" pick is an ordinary heuristic used ONLY for question/
    # distractor bookkeeping here, never fed to the resolver as supervision
    # -- no gold leakage).
    eligible = [c for c in clauses if c.relation in _RELATION_QUESTION_TEMPLATE]
    if not eligible:
        return []

    pool = clauses + list(distractor_pool or [])
    rng = random.Random(seed)

    if holdout == "last":
        candidates = sorted(eligible, key=lambda c: c.sentence_index, reverse=True)
    else:
        candidates = list(eligible)
        rng.shuffle(candidates)

    for held in candidates:
        if _quality_reject_reason(held) is not None:
            if reject_stats is not None:
                reject_stats[EPISODE_QUALITY_CODE] += 1
            continue
        gold = held.value
        seen = {gold}
        distractor_values: List[str] = []
        for c in pool:
            if c is held or c.relation != held.relation or c.value in seen:
                continue
            seen.add(c.value)
            distractor_values.append(c.value)
        if len(distractor_values) < 2:
            continue
        chosen = distractor_values[:3]
        options = chosen + [gold]
        rng.shuffle(options)
        answer_idx = options.index(gold)
        question = _RELATION_QUESTION_TEMPLATE[held.relation].format(e=held.entity)
        parse_stats = dict(taxonomy_counts(passage_clauses))
        return [Episode(
            context=list(context),
            question=question,
            answer_text=gold,
            options=options,
            answer_idx=answer_idx,
            level=0,
            meta={
                "kind": "prose",
                "source_doc": doc_id,
                "sentence_index": held.sentence_index,
                "relation": held.relation,
                "held_out_entity": held.entity,
                "held_out_value": held.value,
                "held_out_predicate": held.predicate,
                "parse_stats": parse_stats,
            },
        )]
    return []
