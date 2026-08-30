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
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .clause import _PRONOUNS, extract_discourse
from .clause_reactor import _TRANSFER_ROLE_MAP
from .episode import Episode

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
)

# Ambiguity gate for "multiple-parses-unresolved": >=2 COMPLETE hypotheses
# (parser._parse_topk_one already keeps only complete parses when any exist)
# whose top1-top2 structural-score margin is at or below this -- i.e. the
# parser genuinely could not tell the readings apart. Kept tight (not a
# generic "top-2 are somewhat close" threshold) so this fires on genuine
# ties, not on ordinary score noise.
_AMBIGUITY_MARGIN = 1e-6

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


_lexicon_entry = None
_word_tag_dict = None
_tagger_load_failed = False


def _load_tagger_tables() -> bool:
    """Lazily load quantum_parser's open/closed-class word tables.

    Same ``qp_root`` resolution :class:`nsm_ct.input_encoder.ParserInputEncoder`
    uses (``src/nsm_ct/corpus.py`` -> ``../../../quantum_parser``). Returns
    ``False`` (and never raises) if quantum_parser isn't importable here --
    callers then skip unknown-word classification rather than erroring.
    """
    global _lexicon_entry, _word_tag_dict, _tagger_load_failed
    if _lexicon_entry is not None:
        return True
    if _tagger_load_failed:
        return False
    try:
        qp_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "quantum_parser"))
        if qp_root not in sys.path:
            sys.path.insert(0, qp_root)
        from src.parser.pos_tagger import WORD_TAG_DICT, lexicon_entry  # type: ignore

        _lexicon_entry = lexicon_entry
        _word_tag_dict = WORD_TAG_DICT
        return True
    except Exception:
        _tagger_load_failed = True
        return False


def _is_known_word(word: str) -> bool:
    """True if the parser's tagger has real (non-guessed) coverage for ``word``.

    Closed-class (``WORD_TAG_DICT``), a pronoun, or present in the
    WordNet-derived open-class lexicon (``lexicon_entry``) all count as
    "known". Anything else falls through to the tagger's blind
    suffix/default-noun heuristics -- the proxy this module uses for
    "unknown word" (archaic vocabulary, rare proper names, typos).
    """
    if not _load_tagger_tables():
        return True  # tagger tables unavailable -- don't classify on a guess
    w = word.lower()
    if w in _word_tag_dict or w in _PRONOUNS:
        return True
    return bool(_lexicon_entry(w))


@dataclass
class ParsedClause:
    """One extracted ``(entity, relation, value)`` fact, grounded in its sentence."""

    sentence_index: int
    sentence: str
    entity: str
    relation: str
    value: str
    predicate: str = ""


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


def _parse_one_sentence(idx: int, sent: str, parser) -> List[ParseOutcome]:
    tokens = sent.split()
    content_tokens = [t for t in tokens if t not in _PUNCT_TOKENS and any(c.isalpha() for c in t)]
    unknown = sorted({t for t in content_tokens if not _is_known_word(t)})
    if unknown:
        return [ParseFailure(idx, sent, "unknown-word", detail=",".join(unknown[:5]))]

    if getattr(parser, "_parser", None) is None or not hasattr(parser, "_parse_topk_one"):
        return [ParseFailure(idx, sent, "no-parse", detail="parser unavailable")]

    graphs, scores, margin = parser._parse_topk_one(sent, k=4)
    if not graphs:
        return [ParseFailure(idx, sent, "no-parse")]
    if len(graphs) >= 2 and margin <= _AMBIGUITY_MARGIN:
        return [ParseFailure(idx, sent, "multiple-parses-unresolved",
                              detail=f"{len(graphs)} hyps, margin={margin:.6f}")]

    clauses, _links = extract_discourse(graphs[0])
    if not clauses:
        sig = _unsupported_signal(tokens)
        if sig:
            return [ParseFailure(idx, sent, "unsupported-construction", detail=sig)]
        return [ParseFailure(idx, sent, "no-relation-extracted")]

    # Mirrors nsm_ct.clause_reactor._context_steps's own extraction exactly
    # (OBJECT-bearing transfer clause -> one step per other role; plain
    # SUBJECT+PLACE clause -> one step), except pronoun-bearing roles are
    # filtered out (this converter does no cross-sentence coreference, so a
    # bare pronoun can't be turned into a groundable, nameable entity/value).
    triples: List[ParsedClause] = []
    pronoun_hit = False
    for cl in clauses:
        pred = (cl.predicate or "").lower()
        obj_tok = next(((arg.token or "").lower() for rel, arg in cl.args if rel == "OBJECT"), None)
        if obj_tok:
            for rel, arg in cl.args:
                if rel == "OBJECT":
                    continue
                tok = (arg.token or "").lower()
                if not tok:
                    continue
                mapped = _TRANSFER_ROLE_MAP.get(rel, rel)
                if obj_tok in _PRONOUNS or tok in _PRONOUNS:
                    pronoun_hit = True
                    continue
                triples.append(ParsedClause(idx, sent, obj_tok, mapped, tok, pred))
            continue
        subj = place = None
        for rel, arg in cl.args:
            if rel == "SUBJECT":
                subj = (arg.token or "").lower()
            elif rel == "PLACE":
                place = (arg.token or "").lower()
        if subj and place:
            if subj in _PRONOUNS:
                pronoun_hit = True
                continue
            triples.append(ParsedClause(idx, sent, subj, "PLACE", place, pred))

    if triples:
        return triples
    if pronoun_hit:
        return [ParseFailure(idx, sent, "pronoun-unresolvable")]
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
    sentence.
    """
    results: List[ParseOutcome] = []
    for idx, sent in enumerate(sentences):
        results.extend(_parse_one_sentence(idx, sent, parser))
    return results


def taxonomy_counts(results: Sequence[ParseOutcome]) -> Counter:
    """One outcome per ``sentence_index``: ``"ok"`` if it produced any
    :class:`ParsedClause`, else its (single, consistent) failure reason.
    Exhaustive over :data:`FAILURE_REASONS` + ``"ok"``.
    """
    outcomes: Dict[int, str] = {}
    for r in results:
        if isinstance(r, ParsedClause):
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
