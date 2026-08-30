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

import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .clause import _PRONOUNS, extract_discourse, is_entity
from .clause_reactor import _TRANSFER_ROLE_MAP
from .episode import Episode
from .membrane import HypothesisCandidateSet, hypothesis_candidate_set

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

    def register(self, clauses) -> None:
        for cl in clauses:
            for _rel, arg in cl.args:
                tok = (arg.token or "").lower()
                if tok and tok not in _PRONOUNS and is_entity(tok):
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
    # M58c item 4: "text" (default) or "quoted" -- set when this clause came
    # from a quoted SPAN inside a longer sentence rather than the sentence
    # as a whole (speaker attribution is out of scope -- see module docstring).
    source: str = "text"


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


def _extract_triples(idx: int, sent: str, clauses, registry: Optional["_PassageRegistry"],
                      source: str = "text") -> Tuple[List[ParsedClause], bool]:
    """Mirrors nsm_ct.clause_reactor._context_steps's own extraction exactly
    (OBJECT-bearing transfer clause -> one step per other role; plain
    SUBJECT+PLACE clause -> one step). A pronoun-bearing role is resolved
    against ``registry``'s nearest antecedent (M58c item 5 -- an entity
    mentioned in an EARLIER sentence) when one is available; otherwise it's
    reported via the returned ``pronoun_hit`` flag exactly as before this
    registry existed (``registry=None`` reproduces the old behavior exactly:
    every pronoun role is unresolved).
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
            for rel, arg in cl.args:
                if rel == "OBJECT":
                    continue
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
            continue
        subj = place = None
        for rel, arg in cl.args:
            if rel == "SUBJECT":
                subj = (arg.token or "").lower()
            elif rel == "PLACE":
                place = (arg.token or "").lower()
        if subj and place:
            resolved_subj = subj
            if subj in _PRONOUNS:
                resolved_subj = registry.nearest() if registry is not None else None
                if not resolved_subj:
                    pronoun_hit = True
                    continue
            triples.append(ParsedClause(idx, sent, resolved_subj, "PLACE", place, pred, source=source))
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
                      registry: Optional["_PassageRegistry"]) -> Optional[List[ParsedClause]]:
    """M58c item 4: when the sentence AS A WHOLE yields nothing, try the
    quoted SPAN inside it (if any) on its own -- e.g. narration wrapped
    around a complete quoted clause. Facts from the quoted span are tagged
    ``source="quoted"``; attribution to the speaker is out of scope (the
    quoted clause's own facts are extracted, nothing about who said it).
    ``None`` (not a fallback failure list) if there's no quote mark, or the
    span itself doesn't parse/extract either -- callers fall through to
    their normal failure classification in that case.
    """
    span = _quoted_span(tokens)
    if not span:
        return None
    graphs, _scores, _margin = parser._parse_topk_one(span, k=4)
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


def _parse_one_sentence(idx: int, sent: str, parser,
                         registry: Optional["_PassageRegistry"] = None) -> List[ParseOutcome]:
    tokens = sent.split()
    content_tokens = [t for t in tokens if t not in _PUNCT_TOKENS and any(c.isalpha() for c in t)]
    unknown = sorted({t for t in content_tokens if not _is_known_word(t)})
    if unknown:
        return [ParseFailure(idx, sent, "unknown-word", detail=",".join(unknown[:5]))]

    if getattr(parser, "_parser", None) is None or not hasattr(parser, "_parse_topk_one"):
        return [ParseFailure(idx, sent, "no-parse", detail="parser unavailable")]

    # M58c item 4: quotation marks are stripped before the PARSE call only --
    # `sent`/`tokens` (kept for context/detail text and the unsupported-
    # construction/quoted-span checks) are untouched.
    graphs, scores, margin = parser._parse_topk_one(_strip_quotes(sent), k=4)
    # M58c item 2's tie-breaker (see _rerank_topk's own docstring for why it
    # lives here rather than in quantum_parser's scorer): re-picks "top1"
    # among score-tied candidates only; margin is recomputed since the
    # reordering can change which score sits at index 0/1.
    graphs, scores = _rerank_topk(graphs, scores)
    margin = (scores[0] - scores[1]) if len(scores) > 1 else margin
    if not graphs:
        if _no_verb_token(content_tokens):
            return [ParseFailure(idx, sent, "fragment-skipped")]
        fallback = _quoted_fallback(idx, sent, tokens, parser, registry)
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
        fallback = _quoted_fallback(idx, sent, tokens, parser, registry)
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
    fallback = _quoted_fallback(idx, sent, tokens, parser, registry)
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
    (non-ambiguous) :class:`ParsedClause`, ``"parsed-ambiguous"`` if its
    clause(s) came from a low-margin top-K (M58c item 2 --
    :attr:`ParsedClause.hypotheses` set), else its (single, consistent)
    failure reason. Exhaustive over :data:`FAILURE_REASONS` +
    ``{"ok", "parsed-ambiguous"}``.
    """
    outcomes: Dict[int, str] = {}
    for r in results:
        if isinstance(r, ParsedClause):
            outcomes[r.sentence_index] = "parsed-ambiguous" if r.hypotheses is not None else "ok"
        elif r.sentence_index not in outcomes:
            outcomes[r.sentence_index] = r.reason
    return Counter(outcomes.values())


# ---------------------------------------------------------------------------
# 3. make_episodes -- hold out one eligible clause, ask it via the EXISTING
#    queried-role table (nsm_ct.clause_reactor._queried_role /
#    _question_entity); every sentence stays in context.
# ---------------------------------------------------------------------------

# Only relations nsm_ct.clause_reactor._queried_role can actually recover
# from question text get a template -- SOURCE (and any raw preposition
# label _context_steps might pass through unmapped) has no keyword in that
# table and is deliberately left un-askable rather than inventing a new
# keyword the rest of the (frozen) pipeline doesn't know about.
_RELATION_QUESTION_TEMPLATE: Dict[str, str] = {
    "PLACE": "where is the {e} ?",
    "RECIPIENT": "who has the {e} ?",
    "AGENT": "who gave the {e} ?",
}


def make_episodes(passage_clauses: Sequence[ParseOutcome], *, holdout: str = "last",
                   seed: int = 0, doc_id: str = "",
                   distractor_pool: Optional[Sequence[ParsedClause]] = None) -> List[Episode]:
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
