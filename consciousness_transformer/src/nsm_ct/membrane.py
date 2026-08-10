"""The perception<->mind membrane (M53a): candidate-set types + the data.

dev/MIND_INTERFACE.md's v1 IN table says perception never guesses: anything
it cannot resolve deterministically (a pronoun's antecedent, a homograph's
sense, a low-margin parse) crosses the boundary as a **candidate set** --
a list of options with structural priors -- rather than a forced choice.
This module defines that shape (dataclasses + numpy only, **no torch**: the
membrane types are perception-side, the resolver that will consume them is
the next agent's model-side work, dev/RESOLVER_BUILD_PLAN.md Phase 2
"Agent 3") plus the deterministic pieces M53a needs to exercise the pipeline
end to end:

- :class:`CandidateSet` / :class:`Candidate` -- the generic v1 shape, dead
  simple, reused as-is by M54 (sense candidates) and M55 (parse hypotheses).
- :class:`EntityCandidateSet` -- the v1 IN table's "entity" row: a pronoun's
  unresolved antecedent slot.
- :func:`mention_feature_vector` -- a deterministic USVS + hand-specified
  feature vector for a mention (pronoun or name); see its docstring for the
  607-axis gender-axis finding that shaped this.
- :func:`entity_registry` / :func:`pronoun_entity_candidate_set` -- the
  discourse-level bookkeeping :mod:`nsm_ct.clause_reactor`'s batch path uses
  to emit one :class:`EntityCandidateSet` per pronoun-subject context
  sentence, candidates = entity atoms introduced EARLIER in the episode
  (STM order; M53a is STM-only by design, per MIND_INTERFACE.md's "LTM
  candidates for pronouns" out-of-scope note).

M53a binds every candidate set's gold index (from the curriculum's episode
meta) and ALSO carries it through the batch as a PLACEHOLDER: the reactor is
made to consume the gold antecedent directly (no resolver exists yet), while
the candidate set + gold index ride along in the batch so the resolver head
(M53b) has something to train against without a second data-plumbing pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from .clause import _PRONOUNS, extract_discourse
from .episode import _NAMES
from .wordnet import senses as _wn_senses

# ---------------------------------------------------------------------------
# Generic candidate-set shape (v1 contract) -- perception never guesses.
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """One membrane candidate: an opaque key (entity-atom name today; a
    sense handle / parse-hypothesis id for M54/M55) plus its STRUCTURAL
    prior. Priors are uniform in v1 (nothing here is learned -- perception
    stays untrained and honest about uncertainty)."""

    key: str
    prior: float = 1.0


@dataclass
class CandidateSet:
    """A slot perception couldn't resolve deterministically: candidates +
    priors + provenance (which clause/step, which surface token). Kept
    deliberately minimal so M54 (sense candidates) and M55 (parse
    hypotheses) can reuse it unchanged -- only :class:`EntityCandidateSet`
    below is populated in M53a."""

    candidates: List[Candidate] = field(default_factory=list)
    provenance: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def keys(self) -> List[str]:
        return [c.key for c in self.candidates]

    @property
    def priors(self) -> np.ndarray:
        return np.array([c.prior for c in self.candidates], dtype=np.float32)


@dataclass
class EntityCandidateSet(CandidateSet):
    """The v1 IN-table "entity" row: an unresolved pronoun's candidate
    antecedents (entity atoms introduced earlier in the episode, uniform
    structural prior), the mention's deterministic FEATURE vector (see
    :func:`mention_feature_vector`), and -- M53a placeholder only, not a
    learned resolution -- the gold index for the not-yet-built resolver
    (M53b) to train against."""

    surface: str = ""
    feature: Optional[np.ndarray] = None
    gold_index: Optional[int] = None


# ---------------------------------------------------------------------------
# Deterministic mention feature vectors.
#
# USVS axis check (607 named axes, data/usvs/usvs_meta.json): `attr:gender`
# and `attr:sex` EXIST as named axes but carry NO usable signal -- every
# probed word (man/woman/person/mary/john/ball) lands within float noise of
# zero on both (~1e-5..1e-6). This is because USVS grounds WORDS from
# WordNet glosses, and (a) pronouns aren't WordNet lemmas at all -- she/he/
# it/they/him/her/them have NO `word_coord` whatsoever -- and (b) even
# gendered common nouns don't populate `attr:gender`/`attr:sex` meaningfully
# (that axis pair fires on a handful of unrelated glosses, not sex-typing).
# `lex:noun.person` DOES carry real, useful signal instead (measured:
# woman=1.0, mary=0.30, man=0.34, person=0.16, ball=0.02) so it is the one
# USVS axis pulled into the feature vector below.
#
# Per MIND_INTERFACE.md's own escape hatch ("If gender axes don't exist
# usefully ... fall back to a tiny set of extra named feature dims appended
# in the membrane types, NOT in USVS itself"): gender/number are therefore
# represented as explicit EXTRA dims defined here, never written into
# ground/usvs.py. This is exactly the "closed class -> small explicit table
# is fine and transparent" case the task calls out: there are 7 pronouns and
# 6 curriculum names, not an open lexicon.
# ---------------------------------------------------------------------------
_USVS_FEATURE_AXES: Tuple[str, ...] = ("lex:noun.person",)

# Hand-specified extra dims (not USVS axes): person-hood, gender, plurality.
_EXTRA_DIMS: Tuple[str, ...] = ("PERSON", "GENDER_F", "GENDER_M", "NONPERSON", "PLURAL")

FEATURE_DIM = len(_USVS_FEATURE_AXES) + len(_EXTRA_DIMS)   # 1 + 5 = 6

# Closed-class pronoun feature profiles. Every English personal pronoun is
# listed (clause.py's full `_PRONOUNS` set), not just she/he/it/they, so the
# table never silently falls through to "unknown word" zeros for a pronoun.
_PRONOUN_EXTRA: Dict[str, Dict[str, float]] = {
    "she":  {"PERSON": 1.0, "GENDER_F": 1.0, "GENDER_M": 0.0, "NONPERSON": 0.0, "PLURAL": 0.0},
    "her":  {"PERSON": 1.0, "GENDER_F": 1.0, "GENDER_M": 0.0, "NONPERSON": 0.0, "PLURAL": 0.0},
    "he":   {"PERSON": 1.0, "GENDER_F": 0.0, "GENDER_M": 1.0, "NONPERSON": 0.0, "PLURAL": 0.0},
    "him":  {"PERSON": 1.0, "GENDER_F": 0.0, "GENDER_M": 1.0, "NONPERSON": 0.0, "PLURAL": 0.0},
    "it":   {"PERSON": 0.0, "GENDER_F": 0.0, "GENDER_M": 0.0, "NONPERSON": 1.0, "PLURAL": 0.0},
    "they": {"PERSON": 1.0, "GENDER_F": 0.0, "GENDER_M": 0.0, "NONPERSON": 0.0, "PLURAL": 1.0},
    "them": {"PERSON": 1.0, "GENDER_F": 0.0, "GENDER_M": 0.0, "NONPERSON": 0.0, "PLURAL": 1.0},
    "i":    {"PERSON": 1.0, "GENDER_F": 0.0, "GENDER_M": 0.0, "NONPERSON": 0.0, "PLURAL": 0.0},
    "you":  {"PERSON": 1.0, "GENDER_F": 0.0, "GENDER_M": 0.0, "NONPERSON": 0.0, "PLURAL": 0.0},
    "we":   {"PERSON": 1.0, "GENDER_F": 0.0, "GENDER_M": 0.0, "NONPERSON": 0.0, "PLURAL": 1.0},
    "us":   {"PERSON": 1.0, "GENDER_F": 0.0, "GENDER_M": 0.0, "NONPERSON": 0.0, "PLURAL": 1.0},
    "me":   {"PERSON": 1.0, "GENDER_F": 0.0, "GENDER_M": 0.0, "NONPERSON": 0.0, "PLURAL": 0.0},
}
assert set(_PRONOUN_EXTRA) == _PRONOUNS, "every clause.py pronoun needs a feature profile"

# Curriculum name genders (episode.py's fixed 6-name pool _NAMES). Same
# closed-class rationale as the pronoun table above: 6 names, hand-assigned
# once, transparent -- not derived from USVS (which has no per-name gender
# signal; "mary"/"john" are proper nouns, not WordNet-gendered common nouns).
NAME_GENDER: Dict[str, str] = {"mary": "F", "sandra": "F",
                                "john": "M", "daniel": "M", "bill": "M", "fred": "M"}
assert set(NAME_GENDER) == {n.lower() for n in _NAMES}


@lru_cache(maxsize=256)
def _feature_tuple(word: str) -> Tuple[float, ...]:
    """Cached (word -> feature tuple) so repeated mentions in a curriculum
    batch don't re-touch the USVS artifact. Cache key is the word alone
    (the USVS coordinate is a fixed artifact within a process), which is why
    :func:`mention_feature_vector` only hits this path for the DEFAULT usvs
    instance -- see its docstring."""
    from .usvs_bridge import default_usvs

    u = default_usvs()
    axis_idx = {a: i for i, a in enumerate(u.axes)}
    coord = u.word_coord(word)
    usvs_part = tuple(
        float(coord[axis_idx[a]]) if coord is not None and a in axis_idx else 0.0
        for a in _USVS_FEATURE_AXES
    )
    if word in _PRONOUN_EXTRA:
        extra = _PRONOUN_EXTRA[word]
    elif word in NAME_GENDER:
        g = NAME_GENDER[word]
        extra = {"PERSON": 1.0, "GENDER_F": 1.0 if g == "F" else 0.0,
                 "GENDER_M": 1.0 if g == "M" else 0.0, "NONPERSON": 0.0, "PLURAL": 0.0}
    else:
        extra = {d: 0.0 for d in _EXTRA_DIMS}   # unknown word: no cue either way
    return usvs_part + tuple(extra[d] for d in _EXTRA_DIMS)


def mention_feature_vector(word: str, *, usvs=None) -> np.ndarray:
    """Deterministic feature vector for a mention (pronoun or name).

    Shape ``(FEATURE_DIM,)``: the USVS ``lex:noun.person`` component,
    followed by the hand-specified extra dims (PERSON, GENDER_F, GENDER_M,
    NONPERSON, PLURAL) -- see the module-level USVS axis check above for why
    gender/number live here instead of in USVS itself. ``usvs`` is an
    injection point for tests only; the cached fast path is used when it's
    left as the default artifact.
    """
    w = (word or "").lower()
    if usvs is None:
        return np.array(_feature_tuple(w), dtype=np.float32)
    axis_idx = {a: i for i, a in enumerate(usvs.axes)}
    coord = usvs.word_coord(w)
    usvs_part = [float(coord[axis_idx[a]]) if coord is not None and a in axis_idx else 0.0
                 for a in _USVS_FEATURE_AXES]
    if w in _PRONOUN_EXTRA:
        extra = _PRONOUN_EXTRA[w]
    elif w in NAME_GENDER:
        g = NAME_GENDER[w]
        extra = {"PERSON": 1.0, "GENDER_F": 1.0 if g == "F" else 0.0,
                 "GENDER_M": 1.0 if g == "M" else 0.0, "NONPERSON": 0.0, "PLURAL": 0.0}
    else:
        extra = {d: 0.0 for d in _EXTRA_DIMS}
    return np.array(usvs_part + [extra[d] for d in _EXTRA_DIMS], dtype=np.float32)


# ---------------------------------------------------------------------------
# Entity registry + candidate-set emission (STM-only, per v1 scope).
# ---------------------------------------------------------------------------
def entity_registry(context_sentences: List[str], parser) -> List[str]:
    """Entity atoms (proper names) introduced by ``context_sentences``, in
    order of first mention -- the STM-side registry the v1 "entity" IN row
    needs ("candidates come from prior context sentences",
    dev/RESOLVER_BUILD_PLAN.md Phase 2). Pronoun subjects are never
    registered as candidates themselves (an unresolved mention can't stand
    in as someone else's antecedent). LTM candidates are out of scope for
    v1 by design (MIND_INTERFACE.md §5) -- this only ever looks at the
    sentences passed in.
    """
    seen: List[str] = []
    for sent in context_sentences:
        graph = parser._parse_graph(sent) if hasattr(parser, "_parse_graph") else None
        clauses, _links = extract_discourse(graph)
        for cl in clauses:
            for rel, arg in cl.args:
                if rel != "SUBJECT":
                    continue
                tok = (arg.token or "").lower()
                if tok and tok not in _PRONOUNS and tok not in seen:
                    seen.append(tok)
    return seen


def pronoun_entity_candidate_set(pronoun: str, registry: List[str], *,
                                  gold_antecedent: Optional[str] = None,
                                  provenance: Optional[Dict[str, object]] = None
                                  ) -> EntityCandidateSet:
    """The v1 "entity" IN-row for one pronoun mention: a uniform structural
    prior over every entity in ``registry`` (introduced earlier in the
    episode; STM order), the mention's deterministic feature vector, and
    (M53a placeholder) the gold index -- ``None`` if ``gold_antecedent``
    isn't among the candidates (an unresolvable/OPEN case; the real v1
    design's answer for that is the caution dial + abstain atoms, not
    built here)."""
    n = max(len(registry), 1)
    candidates = [Candidate(key=name, prior=1.0 / n) for name in registry]
    gold_index = registry.index(gold_antecedent) if gold_antecedent in registry else None
    return EntityCandidateSet(
        candidates=candidates,
        provenance=dict(provenance or {}),
        surface=pronoun,
        feature=mention_feature_vector(pronoun),
        gold_index=gold_index,
    )


# ---------------------------------------------------------------------------
# M54: sense candidate sets -- the v1 IN table's "value" row for a homograph
# (episode.generate_ambiguity_episodes). Reuses the generic CandidateSet
# shape exactly as M53a's own docstring promised M54 would.
# ---------------------------------------------------------------------------
@dataclass
class SenseCandidateSet(CandidateSet):
    """The v1 IN-table "value" row: a homograph's candidate senses.

    ``candidates`` come from :func:`nsm_ct.wordnet.senses` (MFS-ordered --
    index 0 IS the most-frequent sense by construction, matching
    ``episode._AMBIGUITY_FAMILIES``'s own ``fam["mfs"] == wn.synsets(word)[0]``
    convention: the SAME ordering source, not re-derived). Priors are
    MFS-RANK-derived (``1/(rank+1)``, normalized to sum to 1) -- perception
    has exactly one honest structural signal about which sense is likelier
    (how frequent it is in general, per WordNet), nothing about the CURRENT
    context, so that is the only prior it is allowed to expose (the v1
    membrane rule: perception never guesses).

    ``gold_index`` is the curriculum's gold sense's position in the
    candidate list (``None`` if somehow absent -- shouldn't happen for the
    M32 families, defensive only, mirrors :class:`EntityCandidateSet`'s own
    ``gold_index`` contract). ``context_word`` (M54 addition, not part of
    the generic candidate-set shape) is the SAME CLAUSE's other-role token,
    if any (e.g. "river" in "the river flowed past the bank ." when the
    homograph is "bank") -- membrane stays torch/codec-free, so this is a
    surface word string; :mod:`nsm_ct.clause_reactor`'s batch-build grounds
    it into a vector, mirroring how :class:`EntityCandidateSet`'s candidate
    KEYS (name strings) are grounded downstream, not here. It is the
    concrete form MIND_INTERFACE.md's "context IS the memory readout (+
    optionally the step's other-role vectors)" line takes for M54, since a
    same-name subject (e.g. "mary walked into the bank .") carries no
    lexical content at all (names ground to arbitrary identity atoms) and
    would add noise, not signal -- see :func:`sense_candidate_set`.
    """

    word: str = ""
    gold_index: Optional[int] = None
    context_word: Optional[str] = None


def sense_candidate_set(word: str, *, gold_sense: Optional[str] = None,
                         context_word: Optional[str] = None,
                         provenance: Optional[Dict[str, object]] = None
                         ) -> SenseCandidateSet:
    """The v1 "value" IN-row for one homograph mention.

    Candidates are ``nsm_ct.wordnet.senses(word)``'s sense ids, MFS-ordered
    (index 0 == MFS); priors are ``1/(rank+1)`` normalized. ``gold_index``
    indexes into that same list (``None`` if the curriculum's gold synset
    isn't among WordNet's candidates for this word -- shouldn't happen for
    the M32 families). If WordNet has no senses at all for ``word`` (should
    never happen for the hand-curated ambiguity families; defensive only),
    falls back to a single-candidate set built from ``gold_sense`` alone so
    downstream code never sees an empty candidate list, mirroring
    :func:`nsm_ct.sense_chooser.candidate_vectors`'s own empty-candidate
    fallback.
    """
    ids = tuple(s["sense_id"] for s in _wn_senses(word.lower()))
    if not ids and gold_sense:
        ids = (gold_sense,)
    raw_priors = [1.0 / (i + 1) for i in range(len(ids))] or [1.0]
    total = sum(raw_priors) or 1.0
    candidates = [Candidate(key=sid, prior=p / total) for sid, p in zip(ids, raw_priors)]
    gold_index = ids.index(gold_sense) if gold_sense in ids else None
    return SenseCandidateSet(
        candidates=candidates,
        provenance=dict(provenance or {}),
        word=word,
        gold_index=gold_index,
        context_word=context_word,
    )
