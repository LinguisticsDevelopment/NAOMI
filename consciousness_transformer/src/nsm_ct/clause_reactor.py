"""The token-free clause reactor: perception fixed, only the REACTION learned.

Perception is deterministic and grounded — each clause becomes a
``(entity, relation, value)`` triple of TPR/prime vectors (no token embedding). The
**only learned parameters** are a small GRU controller + heads that decide, per
clause, how to REACT: a write *gate* into the order-3 entity memory and a *respond*
weight; on responding it **generates** a response meaning-vector, scored
contrastively against the (fixed) option meaning-vectors. See plan / RESEARCH_NOTES §0h.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import entity_memory as em
from . import membrane
from .clause import extract_discourse
from .episode import _NAMES
from .membrane import FEATURE_DIM
from .resolver import Resolver, query_candidates, query_candidates_per_addr
from .tpr import TPRCodec
from .usvs_bridge import usvs_handle, usvs_sense_handle

_NAMESET = {n.lower() for n in _NAMES}

# Content-word meaning source: "usvs" (default since M31.1: the word's USVS
# handle, falling back to explication when the word isn't known to USVS) or
# "explication" (the depth-bounded subtree TPR — still exercised as the
# fallback path, and available explicitly via --meaning-source explication).
# Entity/variable atoms are NEVER affected — they never go through _content_vec.
MeaningSource = str  # "explication" | "usvs"


# ---------------------------------------------------------------------------
# Fixed perception: curriculum episode -> stream of grounded clause triples
# ---------------------------------------------------------------------------
def _content_vec(word: str, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                  meaning_source: MeaningSource = "usvs") -> np.ndarray:
    key = f"{meaning_source}:{word}"
    if key not in cache:
        vec = None
        if meaning_source == "usvs":
            vec = usvs_handle(word, codec.dim)
        if vec is None:                            # explication path (default / fallback)
            tree = resolver.resolve(word)
            vec = codec.contract(codec.encode_matrix(tree.root))
        cache[key] = vec
    return cache[key]


def _option_vec(word: str, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                 meaning_source: MeaningSource = "usvs") -> np.ndarray:
    """An MC option's meaning-vector — MAYBE/idk atoms for those, else content."""
    w = (word or "").lower()
    if w == "maybe":
        return codec.filler_vec("MAYBE")
    if w == "idk":
        return codec.filler_vec("idk")            # the abstain atom
    return _content_vec(word, resolver, codec, cache, meaning_source)


def _ent_vec(name: str, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
             meaning_source: MeaningSource = "usvs") -> np.ndarray:
    """Ground an entity/value: a person → its variable atom; a concept → its meaning."""
    return (codec.filler_vec("var:" + name) if name in _NAMESET
            else _content_vec(name, resolver, codec, cache, meaning_source))


def _reasoning_steps(ep, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                      meaning_source: MeaningSource = "usvs"):
    """Grounded stream for a reasoning episode (L9-L11).

    Perception is a deterministic grounding of the input's *meaning* (the oracle's
    structured premises — the same content the sentence carries; the derived answer
    is never input). A conditional rule streams its antecedent + consequent tagged
    with the IF atom on the ``coord`` channel; facts stream plainly; the question
    step carries the query (entity, relation). The model must CHAIN these to answer.
    Returns ``[(entity, relation, value, pred, coord, is_q)]`` like ``_context_steps``.
    """
    d = codec.dim
    z = np.zeros(d, np.float32)
    pred_is, pred_if = codec.filler_vec("pred:is"), codec.filler_vec("pred:if")
    q_pred, ifv = codec.filler_vec("pred:?"), codec.filler_vec("IF")
    steps = []
    # one conditional rule (meta["rule"]) or a cascade of them (meta["rules"]); each rule
    # streams its antecedent then consequent, tagged with the IF atom on the coord channel.
    rule_list = ep.meta.get("rules") or ([ep.meta["rule"]] if ep.meta.get("rule") else [])
    for rule in rule_list:
        for (e, r, v) in rule:                          # antecedent then consequent
            steps.append((_ent_vec(e, resolver, codec, cache, meaning_source), codec.filler_vec("rel:" + r),
                          _ent_vec(v, resolver, codec, cache, meaning_source), pred_if, ifv, 0))
    for (e, r, v) in ep.meta["facts"]:
        steps.append((_ent_vec(e, resolver, codec, cache, meaning_source), codec.filler_vec("rel:" + r),
                      _ent_vec(v, resolver, codec, cache, meaning_source), pred_is, z, 0))
    qe, qr = ep.meta["query"]
    steps.append((_ent_vec(qe, resolver, codec, cache, meaning_source), codec.filler_vec("rel:" + qr),
                  z, q_pred, z, 1))
    return steps


# M52: SUBJECT->AGENT, INDIRECT_OBJECT->RECIPIENT for OBJECT-bearing (transfer)
# clauses -- the roles the transferred OBJECT itself bears w.r.t. the event
# (see _context_steps). SOURCE/PLACE already read fine as relation names and
# pass through unchanged.
_TRANSFER_ROLE_MAP = {"SUBJECT": "AGENT", "INDIRECT_OBJECT": "RECIPIENT"}


def _context_steps(sent: str, parser, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                    meaning_source: MeaningSource = "usvs"):
    """Grounded steps for a context sentence: one per clause ROLE (M52).

    A disjunction ("A or B") yields one step per disjunct, each carrying the OR
    atom on the coord channel (so the controller can VOTE/superpose them); a
    negation ("not in A") yields one step carrying the NOT atom (so it can
    SUBTRACT). Plain facts carry a zero coord — identical to the old single-triple
    behaviour.

    M52 (multi-arg clauses, RESOLVER_BUILD_PLAN Phase 1): a clause carrying an
    OBJECT argument (a give/hand/pass/take transfer) is unrolled into ONE STEP
    PER OTHER ROLE, all sharing the transferred OBJECT as the entity -- so a
    later question can address the object directly ("where is the ball",
    "who has the ball") without chaining through the giver. SUBJECT maps to
    AGENT and INDIRECT_OBJECT to RECIPIENT for these steps (see
    ``_TRANSFER_ROLE_MAP``); SOURCE/PLACE pass through unchanged. A clause
    with NO OBJECT argument keeps the old single-step shape exactly
    (SUBJECT-is-the-entity, PLACE-is-the-value) -- byte-identical to pre-M52
    for every existing curriculum episode, none of which ever emits an
    OBJECT-labeled arg (see tests/test_multi_arg_transfer.py's regression
    test). Returns ``[(entity, relation, value, pred, coord, is_q=0)]``.
    """
    d = codec.dim
    graph = parser._parse_graph(sent) if hasattr(parser, "_parse_graph") else None
    clauses, links = extract_discourse(graph)
    if not clauses:
        return []
    prime = links[0].prime if links else None
    coordv = codec.filler_vec(prime) if prime else np.zeros(d, np.float32)
    steps = []
    for cl in clauses:
        pred_vec = codec.filler_vec("pred:" + (cl.predicate or "").lower())
        obj_tok = next(((arg.token or "").lower() for rel, arg in cl.args if rel == "OBJECT"), None)
        if obj_tok:
            entity_vec = _ent_vec(obj_tok, resolver, codec, cache, meaning_source)
            for rel, arg in cl.args:
                if rel == "OBJECT":
                    continue
                tok = (arg.token or "").lower()
                if not tok:
                    continue
                mapped = _TRANSFER_ROLE_MAP.get(rel, rel)
                steps.append((entity_vec, codec.filler_vec("rel:" + mapped),
                              _ent_vec(tok, resolver, codec, cache, meaning_source),
                              pred_vec, coordv, 0))
            continue
        subj = place = None
        for rel, arg in cl.args:
            if rel == "SUBJECT":
                subj = (arg.token or "").lower()
            elif rel == "PLACE":
                place = (arg.token or "").lower()
        if not (subj and place):
            continue
        steps.append((codec.filler_vec("var:" + subj), codec.filler_vec("rel:PLACE"),
                      _content_vec(place, resolver, codec, cache, meaning_source),
                      pred_vec, coordv, 0))
    return steps


# M53a (RESOLVER_BUILD_PLAN Phase 2, "Agent 2"): a pronoun-subject context
# sentence whose episode meta carries a gold antecedent binding
# (curriculum2.PronounCurriculumGenerator's "pronoun_binding" episodes only
# -- gated on ``ep.meta["pronoun_sentence_index"]``, a key no other
# generator sets). Perception cannot resolve the pronoun on its own (that is
# the resolver's job, M53b); M53a's PLACEHOLDER is to ground the sentence's
# TRUE meaning directly from the curriculum's own ground truth (the gold
# antecedent's place, already known at generation time) so the memory
# pipeline is exercised end to end, while the real v1 candidate set +
# gold index for the not-yet-built resolver rides along in the batch (see
# :mod:`nsm_ct.membrane` and ``build_clause_batch``'s ``cand_*`` fields).
def _pronoun_context_step(sent: str, ep, parser, resolver, codec: TPRCodec,
                           cache: Dict[str, np.ndarray], meaning_source: MeaningSource,
                           sentence_index: int):
    """One placeholder-bound step + its :class:`membrane.EntityCandidateSet`
    for a pronoun-subject context sentence ("she found the ball ."):
    entity = the OBJECT token (the thing found), relation = PLACE, value =
    the gold antecedent's place (``ep.meta["gold_place"]``) -- i.e. exactly
    the (object, PLACE, place) triple the pipeline already knows how to
    write, just resolved by the curriculum's ground truth instead of a
    trained resolver. Returns ``(step_or_None, candidate_set_or_None)``;
    ``None`` if the sentence doesn't have the expected pronoun+OBJECT shape
    (defensive -- every :mod:`nsm_ct.curriculum2` template is parser-
    verified, so this should never actually trigger).
    """
    d = codec.dim
    graph = parser._parse_graph(sent) if hasattr(parser, "_parse_graph") else None
    clauses, _links = extract_discourse(graph)
    for cl in clauses:
        roles = {rel: (arg.token or "").lower() for rel, arg in cl.args}
        pronoun = roles.get("SUBJECT")
        obj_tok = roles.get("OBJECT")
        if not (pronoun and obj_tok and pronoun in membrane._PRONOUNS):
            continue
        registry = membrane.entity_registry(ep.context[:sentence_index], parser)
        cand = membrane.pronoun_entity_candidate_set(
            pronoun, registry, gold_antecedent=ep.meta.get("gold_antecedent"),
            provenance={"sentence_index": sentence_index, "sentence": sent})
        gold_place = ep.meta["gold_place"]
        pred_vec = codec.filler_vec("pred:" + (cl.predicate or "").lower())
        step = (_ent_vec(obj_tok, resolver, codec, cache, meaning_source),
                codec.filler_vec("rel:PLACE"),
                _content_vec(gold_place, resolver, codec, cache, meaning_source),
                pred_vec, np.zeros(d, np.float32), 0)
        return step, cand
    return None, None


# M54 (RESOLVER_BUILD_PLAN Phase 3, "Agent 4"): an M32 ambiguity episode
# (episode.generate_ambiguity_episodes -- gated on ``ep.meta["homograph"]``,
# a key no other generator sets) whose context+question don't fit the
# old/transfer/pronoun shapes at all: the question is "what kind of X is it
# ?" (no NAME, no "the" -- _question_entity would return None and the whole
# episode would be silently DROPPED by build_clause_batch, which is exactly
# what happens today with no M54 code -- the "check they parse through"
# finding this function fixes at the batch-build level per the task, without
# touching _context_steps/_question_entity/_queried_role themselves).
#
# Design: every context sentence that literally contains the homograph token
# (in ANY clause role -- families place it as SUBJECT/OBJECT/PLACE
# inconsistently, see the M54 dev-agent's parser probe) gets ONE step whose
# ENTITY is the homograph's own (unambiguous, MFS-grounded) identity vector
# -- a stable address used consistently for every occurrence in the episode,
# exactly like _context_steps already uses a transfer clause's OBJECT token
# as an address -- on a FIXED "rel:SENSE" relation (not the sentence's real
# syntactic role: the question has no way to name a role to query, so the
# write/query address must be the same regardless of how any one sentence
# phrased it). The VALUE is the v1 IN table's homograph candidate SET
# (nsm_ct.membrane.SenseCandidateSet); in PLACEHOLDER mode (no sense_resolver
# installed) it is bound directly to either the GOLD sense (``sense_bind=
# "gold"``, the M32 oracle ceiling) or the MFS sense (``sense_bind="mfs"``,
# the M30 floor) -- mirrors _pronoun_context_step's gold-placeholder pattern
# exactly, just with a floor option added for the M30 rematch arm.
#
# Sentences that DON'T mention the homograph flow through _context_steps
# UNCHANGED (dropped if they don't fit its SUBJECT+PLACE shape, same as
# every other curriculum level -- not a regression introduced here). The
# ONE new signal this function extracts beyond that: the SAME CLAUSE's other
# role token (e.g. "river" in "the river flowed past the bank ."), carried
# as SenseCandidateSet.context_word -- membrane.py's docstring explains why
# (names carry no lexical content; this is the concrete, cheap form
# MIND_INTERFACE.md's "+ optionally the step's other-role vectors" takes
# here). A name-only clause (e.g. "mary walked into the bank .") yields
# ``context_word=None`` -- no signal to add, not a bug.
#
# M54b (RESOLVER_BUILD_PLAN follow-up, RESEARCH_NOTES M54b): the fixed
# (hom_vec, rel:SENSE) address above is entity-agnostic BY DESIGN. A first
# attempt at this curriculum swapped the sense step's OWN address to the
# homograph event's SUBJECT's (var:<name>, rel:PLACE) slot -- the same slot
# the entity's earlier disambiguating fact was written under -- reasoning
# that ClauseReactor.forward()'s generic ``mem_read = query(memory, e, r)``
# would then transparently retrieve it. MEASURED WRONG: reusing an address
# that ALREADY holds the entity's own (answer-revealing) content lets the
# model learn a zero-write-gate shortcut at the sense step -- writing
# (gate≈0) simply PRESERVES whatever was already there regardless of which
# sense gets placeholder-bound, so gold-ceiling and MFS-floor become
# indistinguishable (the exact gap M54b exists to restore). The sense step's
# address stays the OLD, untouched, entity-agnostic (hom_vec, rel:SENSE) --
# a slot NOTHING else ever writes to, so the placeholder bind_vec is the
# ONLY thing ever there (first-write-wins, no shortcut available). The
# genuine memory-mediated signal a REAL sense_resolver needs instead rides
# through NEW, separately-gated ``ClauseBatch.sense_cand_subject``/
# ``sense_cand_subject_rel`` fields, consumed by :meth:`ClauseReactor.
# _collapse`'s sense branch via an ACTUAL ``em.query`` call at collapse time
# (see that method's docstring) -- inert (``None``) for every episode this
# curriculum doesn't touch, including the M32 curriculum above.
def _ambiguity_steps(ep, parser, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                      meaning_source: MeaningSource, sense_bind: str = "gold"):
    """Grounded stream + per-step :class:`membrane.SenseCandidateSet` map for
    one ambiguity episode (M32's ``episode.generate_ambiguity_episodes`` OR
    M54b's ``curriculum2.generate_sense_binding_episodes``). Returns
    ``(steps, sense_cand_sets)`` -- the same shape ``_pronoun_context_step``'s
    caller already threads through ``build_clause_batch``'s ``cand_sets``
    dict, just for sense candidates.

    M54b addendum (RESEARCH_NOTES M54b; the ONE clause_reactor.py change that
    curriculum needs, gated on ``ep.meta["kind"] == "sense_binding"`` -- a
    marker only ``curriculum2.SenseBindingCurriculumGenerator`` sets, so the
    M32 curriculum above is completely byte-unaffected): every homograph-hit
    step's :class:`membrane.SenseCandidateSet` gets an EXTRA provenance key,
    ``"subject"`` -- ``ep.meta["entity"]``, the name whose own earlier
    context sentence carries the disambiguating fact (mirrors
    :func:`_pronoun_context_step`'s own pattern of reading a ground-truth
    binding straight out of ``ep.meta`` rather than re-deriving it here).
    ``build_clause_batch`` grounds this into the ``sense_cand_subject``/
    ``sense_cand_subject_rel`` batch fields (see their doc comment on
    :class:`ClauseBatch` for why a genuine per-collapse memory query, not a
    reused step address, is the fix). The sense/question steps' OWN
    address stays ``(hom_vec, rel:SENSE)`` -- unconditionally, for every
    homograph episode, old or new.
    """
    d = codec.dim
    homograph = ep.meta["homograph"]
    gold_sense = ep.meta.get("gold_sense")
    mfs_sense = ep.meta.get("mfs_sense")
    hom_vec = _content_vec(homograph, resolver, codec, cache, meaning_source)
    sense_rel = codec.filler_vec("rel:SENSE")
    pred_is = codec.filler_vec("pred:is")
    q_pred = codec.filler_vec("pred:?")
    z = np.zeros(d, np.float32)

    bind_sense = gold_sense if sense_bind == "gold" else mfs_sense
    bind_vec = usvs_sense_handle(bind_sense, d) if bind_sense else None
    if bind_vec is None:                       # defensive: shouldn't happen for the M32 families
        bind_vec = hom_vec

    # M54b: the SUBJECT whose own prior fact disambiguates this homograph
    # event, threaded through SenseCandidateSet.provenance only -- None for
    # the M32 curriculum (ep.meta has no "entity" key there).
    subj_name = ep.meta.get("entity") if ep.meta.get("kind") == "sense_binding" else None

    steps = []
    sense_cand_sets: Dict[int, "membrane.SenseCandidateSet"] = {}
    for si, sent in enumerate(ep.context):
        graph = parser._parse_graph(sent) if hasattr(parser, "_parse_graph") else None
        clauses, _links = extract_discourse(graph)
        hit_clause = None
        for cl in clauses:
            if any((arg.token or "").lower() == homograph for _rel, arg in cl.args):
                hit_clause = cl
                break
        if hit_clause is None:
            steps += _context_steps(sent, parser, resolver, codec, cache, meaning_source)
            continue
        context_word = None
        for _rel, arg in hit_clause.args:
            tok = (arg.token or "").lower()
            if tok and tok != homograph and tok not in _NAMESET:
                context_word = tok
                break
        cs = membrane.sense_candidate_set(
            homograph, gold_sense=gold_sense, context_word=context_word,
            provenance={"sentence_index": si, "sentence": sent, "subject": subj_name})
        sense_cand_sets[len(steps)] = cs
        steps.append((hom_vec, sense_rel, bind_vec, pred_is, z, 0))
    steps.append((hom_vec, sense_rel, z, q_pred, z, 1))
    return steps, sense_cand_sets


# M55a (RESOLVER_BUILD_PLAN successor, RESEARCH_NOTES M55a, dev/
# TRACK_C_DESIGN.md Sec 1.10): a garden-path episode
# (curriculum2.GardenPathCurriculumGenerator -- gated on
# ep.meta["garden_path"], a key no other generator sets). See
# scripts/probe_m55_hyp_survey.py for the empirical survey this curriculum
# is built from: quantum_parser's M41 WordNet-lexicon-backed tag lattice
# produces a genuine, EXACT structural-score TIE (margin 0.0) between two
# non-equivalent readings of "{name} can {homograph} ." for 14 verb/noun
# homographs (bear/book/date/duck/fish/flies/fly/park/rock/run/saw/time/
# train/watch) -- "can" read as a transitive VERB with the homograph as its
# OBJECT ("name CAN [get] homograph") vs the homograph read as the main
# VERB with "can" as a bare modal ("name CAN [ability] homograph", no
# object at all). Sec 1.10's finding is exactly what this collapse needs:
# the two readings disagree about WHICH (entity, relation) is worth
# querying to decide what's true -- the OBJECT reading's address is
# (homograph, PLACE) ("where is the thing name now has"), the VERB
# reading's is (name, PLACE) ("where was name already") -- a per-candidate
# Addr, not a clause-level one every candidate shares (unlike
# EntityCandidateSet/SenseCandidateSet).
def _garden_path_steps(ep, parser, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                        meaning_source: MeaningSource):
    """Grounded stream + the garden-path step's
    :class:`membrane.HypothesisCandidateSet` for one
    :class:`~nsm_ct.curriculum2.GardenPathCurriculumGenerator` episode.

    Episode shape (``ep.meta`` keys in parens):
        "{name_a} went to the {place_a} ."          -- the VERB reading's answer source
        "the {homograph} is in the {place_b} ."     -- the OBJECT reading's answer source
        "{name_a} can {homograph} ."                -- the garden-path sentence
        Q: "where is {name_a} ?"

    M55a is PLUMBING ONLY (no resolver trained yet, like M53a's pronoun
    placeholder): the third sentence's step is PLACEHOLDER-bound directly
    from ``ep.meta["gold_reading"]`` (``"object"`` | ``"verb"``) -- mirrors
    :func:`_pronoun_context_step`'s gold-antecedent pattern exactly -- while
    the REAL :class:`membrane.HypothesisCandidateSet` (top-K parse exposure,
    real structural-score priors, per-candidate query addresses) rides
    along in the batch's ``hyp_cand_*`` fields so a future resolver has
    something to train against without a second data-plumbing pass, same
    division of labor M53a established for entity candidates.
    """
    d = codec.dim
    name_a = ep.meta["name_a"]
    homograph = ep.meta["gp_homograph"]   # NOT "homograph" -- see curriculum2.py's note (M54 key collision)
    place_a_word = ep.meta["place_a"]
    place_b_word = ep.meta["place_b"]
    gold_reading = ep.meta["gold_reading"]        # "object" | "verb"

    place_rel = codec.filler_vec("rel:PLACE")
    pred_is = codec.filler_vec("pred:is")
    q_pred = codec.filler_vec("pred:?")
    z = np.zeros(d, np.float32)

    name_a_vec = _ent_vec(name_a, resolver, codec, cache, meaning_source)      # var:name_a (a name)
    hom_vec = _ent_vec(homograph, resolver, codec, cache, meaning_source)      # content vec (not a name)
    place_a_vec = _content_vec(place_a_word, resolver, codec, cache, meaning_source)
    place_b_vec = _content_vec(place_b_word, resolver, codec, cache, meaning_source)

    steps = []
    # C1: name_a's baseline place -- the VERB reading's answer source.
    steps.append((name_a_vec, place_rel, place_a_vec, pred_is, z, 0))
    # C2: the homograph's OWN place -- an independently-grounded fact, the
    # OBJECT reading's answer source, addressed the SAME way the collapse
    # step's OBJECT-reading candidate will query it (both via _ent_vec on
    # the homograph token).
    steps.append((hom_vec, place_rel, place_b_vec, pred_is, z, 0))

    # C3: the garden-path collapse step. Real top-K parse exposure (M55a
    # input_encoder addition) over the actual sentence text -- the candidate
    # set's priors are REAL structural scores from the parser, not invented.
    sentence = f"{name_a} can {homograph} ."
    graphs, scores = [], []
    if hasattr(parser, "_parse_topk_one"):
        graphs, scores, _margin = parser._parse_topk_one(sentence, k=2)
    if len(scores) >= 2:
        total = sum(scores) or 1.0
        priors = [s / total for s in scores]
    else:                       # defensive: parser unavailable/degenerate -- uniform prior
        priors = [0.5, 0.5]

    # Candidate 0 = OBJECT reading (query homograph's PLACE); candidate 1 =
    # VERB reading (query name_a's own PLACE) -- a FIXED assignment, not
    # derived from chart order: the survey shows this is an exact score tie,
    # so which of the two equally-scored hypotheses the parser happens to
    # emit first carries no semantic meaning; what matters (Sec 1.10) is
    # that each candidate carries its OWN address.
    gold_index = 0 if gold_reading == "object" else 1
    bind_vec = place_b_vec if gold_reading == "object" else place_a_vec
    steps.append((name_a_vec, place_rel, bind_vec, pred_is, z, 0))

    cand = membrane.hypothesis_candidate_set(
        readings=[("object_reading", priors[0], homograph, "PLACE"),
                  ("verb_reading", priors[1], name_a, "PLACE")],
        gold_index=gold_index,
        provenance={"sentence": sentence, "scores": scores},
    )
    hyp_cand_sets: Dict[int, "membrane.HypothesisCandidateSet"] = {len(steps) - 1: cand}

    # Question: "where is name_a ?" -- fixed form, always about name_a's
    # (possibly just-updated) PLACE.
    steps.append((name_a_vec, place_rel, z, q_pred, z, 1))
    return steps, hyp_cand_sets


# M52 v1 IN table ("queried role"): the relation a question is actually
# asking about. "has" -> RECIPIENT (who now holds it -- the transfer_who
# level); "where" -> PLACE, which is ALSO the fallback/default -- every
# question asked before M52 was "where is {name} ?", so this default
# preserves byte-identical behavior for all of them. "who"/"what" are the
# v1 IN table's other named cases (who-subject -> AGENT/SUBJECT, what ->
# OBJECT), not yet exercised by any curriculum level but resolvable here so
# a future one doesn't need to touch this function.
_QUESTION_ROLE_KEYWORDS = (("has", "RECIPIENT"), ("where", "PLACE"),
                           ("who", "AGENT"), ("what", "OBJECT"))


def _queried_role(question: str) -> str:
    words = set(question.lower().replace("?", " ").split())
    for kw, role in _QUESTION_ROLE_KEYWORDS:
        if kw in words:
            return role
    return "PLACE"


def _question_entity(question: str) -> Optional[str]:
    """The entity a question is about: a NAME if one appears (unchanged), else
    (M52) the noun immediately following "the" -- "where is the ball ?" /
    "who has the ball ?" ask about an OBJECT, not a person. Old-curriculum
    questions never contain "the" (always "where is {name} ?"), so this
    fallback is inert for them: behavior is byte-identical.
    """
    words = question.lower().replace("?", " ").split()
    for w in words:
        if w in _NAMESET:
            return w
    if "the" in words:
        idx = words.index("the")
        if idx + 1 < len(words):
            return words[idx + 1]
    return None


@dataclass
class ClauseBatch:
    entity: torch.Tensor    # [B, T, d]
    relation: torch.Tensor  # [B, T, d]
    value: torch.Tensor     # [B, T, d]
    pred: torch.Tensor      # [B, T, d]  predicate filler (the verb signal)
    is_q: torch.Tensor      # [B, T]  1 = question (respond) step
    mask: torch.Tensor      # [B, T]  1 = real step
    options: torch.Tensor   # [B, K, d]
    answer: torch.Tensor    # [B]
    coord: Optional[torch.Tensor] = None  # [B, T, d]  OR/NOT/IF atom (else zeros): the logical signal
    answerable: Optional[torch.Tensor] = None  # [B]  1 = derivable, 0 = should abstain (reasoning levels)
    # M53a (RESOLVER_BUILD_PLAN Phase 2): the v1 "entity" candidate set for
    # pronoun-subject steps, carried through so the not-yet-built resolver
    # (M53b) has data to train against. All None when the batch has no
    # pronoun episodes at all -- this is what keeps old batches
    # byte-identical (nothing above this line changes shape or values).
    cand_entity: Optional[torch.Tensor] = None   # [B, T, C, d]  candidate atoms' var-vectors (0-padded)
    cand_mask: Optional[torch.Tensor] = None     # [B, T, C]     1 = real candidate
    cand_prior: Optional[torch.Tensor] = None    # [B, T, C]     structural prior (uniform v1)
    cand_feature: Optional[torch.Tensor] = None  # [B, T, F]     mention feature vector (0 elsewhere)
    cand_gold: Optional[torch.Tensor] = None     # [B, T]        long; gold candidate index, -1 elsewhere
    # M56b (dev/TRACK_C_DESIGN.md §1.8 "the missing register"): each
    # CANDIDATE's own feature vector (membrane.EntityCandidateSet.
    # cand_features), as opposed to cand_feature above (the MENTION's,
    # broadcast). None whenever no EntityCandidateSet carried cand_features
    # (which is None for every episode -- old and new -- since only
    # pronoun_entity_candidate_set populates it), keeping this
    # byte-identical for every pre-M56b batch, exactly like cand_* above
    # stays None for a pronoun-free batch.
    cand_feature_per_candidate: Optional[torch.Tensor] = None    # [B, T, C, F]  per-candidate feature (0 elsewhere)
    # M54 (RESOLVER_BUILD_PLAN Phase 3): the v1 "value" candidate set for
    # homograph-bearing steps (nsm_ct.membrane.SenseCandidateSet), carried
    # through in a SEPARATE field group from the cand_* ones above rather
    # than folded into them -- see build_clause_batch's docstring for why
    # (byte-identity: an entity-candidate-free OR sense-candidate-free batch
    # must leave the OTHER group entirely None/untouched, and the two kinds
    # need different per-candidate semantics downstream -- cand_entity holds
    # ATOMS to look up in memory, sense_cand_entity holds MEANING VECTORS
    # used directly, not memory addresses).
    sense_cand_entity: Optional[torch.Tensor] = None   # [B, T, C, d]  candidate SENSE vectors (0-padded)
    sense_cand_mask: Optional[torch.Tensor] = None     # [B, T, C]     1 = real (USVS-groundable) candidate
    sense_cand_prior: Optional[torch.Tensor] = None    # [B, T, C]     MFS-rank-derived structural prior
    sense_cand_context: Optional[torch.Tensor] = None  # [B, T, d]     same-clause other-role vector (0 elsewhere)
    sense_cand_gold: Optional[torch.Tensor] = None     # [B, T]        long; gold candidate index, -1 elsewhere
    # M54b (RESEARCH_NOTES M54b, curriculum2.SenseBindingCurriculumGenerator
    # ONLY -- both None for every other episode, including the M32 ambiguity
    # curriculum above): a GENUINE memory-mediated collapse context, kept
    # SEPARATE from ``sense_cand_context`` (a static, batch-build-time
    # grounded word -- never used by this curriculum, which carries no
    # same-clause context word by construction) because this one requires an
    # actual live memory QUERY at collapse time, not a precomputed vector --
    # see ClauseReactor._collapse's sense branch and _ambiguity_steps's M54b
    # docstring for why the sense STEP's own (entity, relation) address
    # can't double as this signal (it must stay the OLD, entity-agnostic,
    # freshly-written (hom_vec, rel:SENSE) slot for the gold/MFS placeholder
    # arms to mean anything -- reusing an entity's own already-written PLACE
    # slot let a model learn a zero-write-gate shortcut that preserves the
    # entity's pre-existing (answer-revealing) content regardless of which
    # sense gets placeholder-bound, silently erasing the gold-vs-MFS gap;
    # MEASURED, not guessed -- see RESEARCH_NOTES M54b).
    sense_cand_subject: Optional[torch.Tensor] = None      # [B, T, d]  homograph event's SUBJECT var-atom (0 elsewhere)
    sense_cand_subject_rel: Optional[torch.Tensor] = None  # [B, T, d]  relation to query the subject under (0 elsewhere)
    # M55a (dev/TRACK_C_DESIGN.md Sec 1.10, RESEARCH_NOTES M55a): the v1
    # "parse hypothesis" candidate set (nsm_ct.membrane.HypothesisCandidateSet)
    # for a garden-path collapse step -- a THIRD, separate tensor group (same
    # rationale as sense_cand_* being separate from cand_*: a hypothesis
    # candidate's own asserted VALUE is not a memory address to look up like
    # an entity atom, and it needs a genuinely PER-CANDIDATE query address,
    # which neither cand_* nor sense_cand_* has). None whenever no episode in
    # the batch carries a garden-path step -- byte-identical to pre-M55a for
    # every existing batch, same guarantee M53a/M54 made for their own groups.
    hyp_cand_entity: Optional[torch.Tensor] = None            # [B, T, C, d]  each reading's OWN asserted value vector
    hyp_cand_mask: Optional[torch.Tensor] = None               # [B, T, C]     1 = real candidate
    hyp_cand_prior: Optional[torch.Tensor] = None               # [B, T, C]     normalized structural score
    hyp_cand_gold: Optional[torch.Tensor] = None                 # [B, T]        long; gold reading index, -1 elsewhere
    # The M55a per-candidate Addr register itself (Sec 1.10): WHICH (entity,
    # relation) reading i asserts is worth querying -- unlike cand_entity/
    # sense_cand_entity, this pair is genuinely per-candidate, not a single
    # clause-level address shared by every candidate in the set.
    hyp_cand_query_entity: Optional[torch.Tensor] = None          # [B, T, C, d]
    hyp_cand_query_relation: Optional[torch.Tensor] = None         # [B, T, C, d]

    def _coord(self) -> torch.Tensor:
        return self.coord if self.coord is not None else torch.zeros_like(self.entity)

    def _cand_fields(self, xform):
        return tuple(xform(t) if t is not None else None for t in
                     (self.cand_entity, self.cand_mask, self.cand_prior,
                      self.cand_feature, self.cand_gold, self.cand_feature_per_candidate))

    def _sense_cand_fields(self, xform):
        return tuple(xform(t) if t is not None else None for t in
                     (self.sense_cand_entity, self.sense_cand_mask, self.sense_cand_prior,
                      self.sense_cand_context, self.sense_cand_gold,
                      self.sense_cand_subject, self.sense_cand_subject_rel))

    def _hyp_cand_fields(self, xform):
        return tuple(xform(t) if t is not None else None for t in
                     (self.hyp_cand_entity, self.hyp_cand_mask, self.hyp_cand_prior,
                      self.hyp_cand_gold, self.hyp_cand_query_entity, self.hyp_cand_query_relation))

    def to(self, device):
        coord = self.coord.to(device) if self.coord is not None else None
        ans_ok = self.answerable.to(device) if self.answerable is not None else None
        cand = self._cand_fields(lambda t: t.to(device))
        scand = self._sense_cand_fields(lambda t: t.to(device))
        hcand = self._hyp_cand_fields(lambda t: t.to(device))
        return ClauseBatch(self.entity.to(device), self.relation.to(device),
                           self.value.to(device), self.pred.to(device),
                           self.is_q.to(device), self.mask.to(device),
                           self.options.to(device), self.answer.to(device), coord, ans_ok,
                           *cand, *scand, *hcand)

    def subset(self, idx) -> "ClauseBatch":
        """A minibatch over the leading (episode) dimension."""
        coord = self.coord[idx] if self.coord is not None else None
        ans_ok = self.answerable[idx] if self.answerable is not None else None
        cand = self._cand_fields(lambda t: t[idx])
        scand = self._sense_cand_fields(lambda t: t[idx])
        hcand = self._hyp_cand_fields(lambda t: t[idx])
        return ClauseBatch(self.entity[idx], self.relation[idx], self.value[idx],
                           self.pred[idx], self.is_q[idx], self.mask[idx],
                           self.options[idx], self.answer[idx], coord, ans_ok,
                           *cand, *scand, *hcand)


def build_clause_batch(episodes, parser, resolver, codec: TPRCodec,
                        meaning_source: MeaningSource = "usvs",
                        sense_bind: str = "gold") -> ClauseBatch:
    """Encode curriculum episodes into grounded clause-triple streams (fixed).

    Each step is ``(entity, relation, value, pred, coord, is_q)``; ``coord`` carries
    the OR/NOT atom for disjunction/negation steps (zeros otherwise) — the logical
    signal the controller reacts to. Options ground to content vectors, except
    "maybe" → the NSM MAYBE atom (so a disjunction can be answered "maybe").

    ``meaning_source`` (M31 consumer gate) selects how CONTENT-WORD meaning
    vectors are built: ``"usvs"`` (default since M31.1 — beats explication at
    both quick and full training budgets, see RESEARCH_NOTES) uses the word's
    USVS handle (:func:`nsm_ct.usvs_bridge.usvs_handle`), falling back to
    ``"explication"`` (the depth-bounded explication subtree TPR) for words
    USVS doesn't know. Entity/variable atoms (``var:<name>``) are never
    affected by this switch either way.

    M52 (RESOLVER_BUILD_PLAN Phase 1): multi-arg clauses (an OBJECT-bearing
    give/hand/pass/take transfer) unroll into one step per role instead of
    being dropped down to a single (SUBJECT, PLACE) pair — see
    :func:`_context_steps`. The question step's relation is now the
    QUERIED ROLE the question text actually names (:func:`_queried_role`)
    rather than a hardcoded PLACE; the default is still PLACE, so every
    pre-M52 "where is X ?" question is unaffected.

    M53a (RESOLVER_BUILD_PLAN Phase 2): an episode whose meta carries
    ``pronoun_sentence_index`` (curriculum2.PronounCurriculumGenerator only)
    has that ONE context sentence routed through
    :func:`_pronoun_context_step` instead of :func:`_context_steps` --
    PLACEHOLDER-bound to the gold antecedent (see that function's
    docstring) -- and its :class:`nsm_ct.membrane.EntityCandidateSet`
    carried through in the returned batch's ``cand_*`` fields (``None`` for
    every episode without a pronoun step, which is what keeps this
    byte-identical to pre-M53a for every existing episode/level).

    M54 (RESOLVER_BUILD_PLAN Phase 3): an episode whose meta carries
    ``homograph`` (``episode.generate_ambiguity_episodes`` only -- no other
    generator sets this key) is routed WHOLESALE through
    :func:`_ambiguity_steps` instead of the old/transfer/pronoun path above
    (its question shape, "what kind of X is it ?", doesn't fit
    :func:`_question_entity` at all -- see that function's docstring for why
    this has to be a top-level branch, not a per-sentence patch). Its
    :class:`nsm_ct.membrane.SenseCandidateSet` map rides through the
    returned batch's SEPARATE ``sense_cand_*`` fields (``None`` for every
    episode without a homograph step -- ambiguity-free batches are
    byte-identical to pre-M54, same guarantee M53a made for ``cand_*``).
    ``sense_bind`` (``"gold"`` default, or ``"mfs"``) selects which sense
    :func:`_ambiguity_steps` placeholder-binds to when NO sense_resolver is
    installed (the gold-ceiling / MFS-floor training arms); it is inert
    whenever a resolver IS installed, since :meth:`ClauseReactor._collapse`
    always overrides the placeholder for any step carrying real candidates.

    M55a: an episode whose meta carries ``garden_path``
    (``curriculum2.GardenPathCurriculumGenerator`` only) is routed WHOLESALE
    through :func:`_garden_path_steps` (its context/collapse/question shape
    doesn't fit the old/transfer/pronoun path, same reasoning as M54's
    ambiguity branch above). Its :class:`nsm_ct.membrane.HypothesisCandidateSet`
    rides through a THIRD, separate ``hyp_cand_*`` field group (``None`` for
    every episode without a garden-path step -- byte-identical to pre-M55a
    for every existing batch, same guarantee M53a/M54 made for their groups).
    """
    cache: Dict[str, np.ndarray] = {}
    d = codec.dim
    q_pred = codec.filler_vec("pred:?")             # the question's (unknown) predicate
    z = np.zeros(d, np.float32)
    rows = []
    for ep in episodes:
        cand_sets: Dict[int, "membrane.EntityCandidateSet"] = {}
        sense_cand_sets: Dict[int, "membrane.SenseCandidateSet"] = {}
        hyp_cand_sets: Dict[int, "membrane.HypothesisCandidateSet"] = {}
        if getattr(ep, "level", 0) >= 9 and ep.meta.get("query"):
            steps = _reasoning_steps(ep, resolver, codec, cache, meaning_source)   # L9-L11 reasoning stream
        elif ep.meta.get("homograph"):
            steps, sense_cand_sets = _ambiguity_steps(ep, parser, resolver, codec, cache,
                                                       meaning_source, sense_bind)   # M54 ambiguity stream
        elif ep.meta.get("garden_path"):
            steps, hyp_cand_sets = _garden_path_steps(ep, parser, resolver, codec, cache,
                                                        meaning_source)   # M55a garden-path stream
        else:
            steps = []
            pronoun_idx = ep.meta.get("pronoun_sentence_index")
            for si, sent in enumerate(ep.context):
                step_cand = (_pronoun_context_step(sent, ep, parser, resolver, codec, cache,
                                                    meaning_source, si)
                             if pronoun_idx is not None and si == pronoun_idx else (None, None))
                if step_cand[0] is not None:
                    cand_sets[len(steps)] = step_cand[1]
                    steps.append(step_cand[0])
                else:
                    steps += _context_steps(sent, parser, resolver, codec, cache, meaning_source)
            qent = _question_entity(ep.question)
            if qent is None:
                continue
            qrel = _queried_role(ep.question)
            steps.append((_ent_vec(qent, resolver, codec, cache, meaning_source),
                          codec.filler_vec("rel:" + qrel), z, q_pred, z, 1))
            for sent in getattr(ep, "post_context", []) or []:
                steps += _context_steps(sent, parser, resolver, codec, cache, meaning_source)
        opt = [_option_vec(o, resolver, codec, cache, meaning_source) for o in ep.options]
        rows.append((steps, opt, ep.answer_idx, 1.0 if getattr(ep, "answerable", True) else 0.0,
                     cand_sets, sense_cand_sets, hyp_cand_sets))

    b = len(rows)
    T = max(len(s) for s, _, _, _, _, _, _ in rows)
    K = max(len(o) for _, o, _, _, _, _, _ in rows)
    ent = torch.zeros(b, T, d); rel = torch.zeros(b, T, d); val = torch.zeros(b, T, d)
    prd = torch.zeros(b, T, d); crd = torch.zeros(b, T, d)
    is_q = torch.zeros(b, T); mask = torch.zeros(b, T)
    opts = torch.zeros(b, K, d); ans = torch.zeros(b, dtype=torch.long)
    ans_ok = torch.zeros(b)
    for i, (steps, opt, a, ok, _cs, _scs, _hcs) in enumerate(rows):
        for t, (e, r, v, p, c, q) in enumerate(steps):
            ent[i, t] = torch.from_numpy(e); rel[i, t] = torch.from_numpy(r)
            val[i, t] = torch.from_numpy(v); prd[i, t] = torch.from_numpy(p)
            crd[i, t] = torch.from_numpy(c); is_q[i, t] = q; mask[i, t] = 1.0
        for k, ov in enumerate(opt):
            opts[i, k] = torch.from_numpy(ov)
        ans[i] = a; ans_ok[i] = ok

    cand_entity = cand_mask = cand_prior = cand_feature = cand_gold = None
    cand_feature_per_candidate = None
    if any(cs for *_row, cs, _scs, _hcs in rows):
        Cmax = max((len(cs.candidates) for *_row, cs_map, _scs, _hcs in rows for cs in cs_map.values()),
                   default=1) or 1
        cand_entity = torch.zeros(b, T, Cmax, d)
        cand_mask = torch.zeros(b, T, Cmax)
        cand_prior = torch.zeros(b, T, Cmax)
        cand_feature = torch.zeros(b, T, membrane.FEATURE_DIM)
        cand_gold = torch.full((b, T), -1, dtype=torch.long)
        # M56b: only allocated when at least one EntityCandidateSet actually
        # carries cand_features (pronoun_entity_candidate_set always sets it
        # today, but this mirrors the sense_cand_subject "only if present"
        # pattern rather than assuming) -- an entity-candidate batch built
        # before M56b (or hand-built in a test without cand_features) leaves
        # this None exactly like every other optional field here.
        has_cand_features = any(
            cs.cand_features is not None for *_row, cs_map, _scs, _hcs in rows for cs in cs_map.values())
        if has_cand_features:
            cand_feature_per_candidate = torch.zeros(b, T, Cmax, membrane.FEATURE_DIM)
        for i, (steps, opt, a, ok, cs_map, _scs, _hcs) in enumerate(rows):
            for t, cs in cs_map.items():
                for j, c in enumerate(cs.candidates):
                    cand_entity[i, t, j] = torch.from_numpy(
                        _ent_vec(c.key, resolver, codec, cache, meaning_source))
                    cand_mask[i, t, j] = 1.0
                    cand_prior[i, t, j] = c.prior
                if cs.feature is not None:
                    cand_feature[i, t] = torch.from_numpy(cs.feature)
                if cs.gold_index is not None:
                    cand_gold[i, t] = cs.gold_index
                if has_cand_features and cs.cand_features is not None:
                    cand_feature_per_candidate[i, t, :len(cs.candidates)] = torch.from_numpy(cs.cand_features)

    # M54: SAME shape/pattern as the cand_* block above, but for
    # nsm_ct.membrane.SenseCandidateSet -- a wholly SEPARATE tensor group
    # (see ClauseBatch's field comment for why), so an ambiguity-free batch
    # leaves it None exactly like a pronoun-free batch leaves cand_* None.
    sense_cand_entity = sense_cand_mask = sense_cand_prior = None
    sense_cand_context = sense_cand_gold = None
    sense_cand_subject = sense_cand_subject_rel = None
    if any(scs for *_row, _cs, scs, _hcs in rows):
        Smax = max((len(scs.candidates) for *_row, _cs, scs_map, _hcs in rows for scs in scs_map.values()),
                   default=1) or 1
        sense_cand_entity = torch.zeros(b, T, Smax, d)
        sense_cand_mask = torch.zeros(b, T, Smax)
        sense_cand_prior = torch.zeros(b, T, Smax)
        sense_cand_context = torch.zeros(b, T, d)
        sense_cand_gold = torch.full((b, T), -1, dtype=torch.long)
        # M54b: only allocated when at least one SenseCandidateSet actually
        # carries a "subject" provenance key (curriculum2.
        # SenseBindingCurriculumGenerator episodes only -- _ambiguity_steps
        # sets this key exclusively for ep.meta["kind"] == "sense_binding",
        # so the M32 curriculum's batches never allocate these tensors at
        # all, exactly like sense_cand_context/etc. stay None for a
        # homograph-free batch).
        has_subject = any(
            scs.provenance.get("subject") for *_row, _cs, scs_map, _hcs in rows for scs in scs_map.values())
        if has_subject:
            sense_cand_subject = torch.zeros(b, T, d)
            sense_cand_subject_rel = torch.zeros(b, T, d)
        for i, (steps, opt, a, ok, _cs, scs_map, _hcs) in enumerate(rows):
            for t, scs in scs_map.items():
                for j, c in enumerate(scs.candidates):
                    sv = usvs_sense_handle(c.key, d)     # None -> masked out, never chosen (sense_chooser.py's rule)
                    sense_cand_entity[i, t, j] = torch.from_numpy(sv if sv is not None else np.zeros(d, np.float32))
                    sense_cand_mask[i, t, j] = 1.0 if sv is not None else 0.0
                    sense_cand_prior[i, t, j] = c.prior
                if scs.context_word:
                    sense_cand_context[i, t] = torch.from_numpy(
                        _content_vec(scs.context_word, resolver, codec, cache, meaning_source))
                if scs.gold_index is not None:
                    sense_cand_gold[i, t] = scs.gold_index
                subject_name = scs.provenance.get("subject")
                if subject_name and has_subject:
                    sense_cand_subject[i, t] = torch.from_numpy(
                        _ent_vec(subject_name, resolver, codec, cache, meaning_source))
                    sense_cand_subject_rel[i, t] = torch.from_numpy(codec.filler_vec("rel:PLACE"))

    # M55a: SAME shape/pattern again, for nsm_ct.membrane.HypothesisCandidateSet
    # -- a THIRD separate tensor group (see ClauseBatch's field comment), so a
    # garden-path-free batch leaves it None exactly like a pronoun/homograph-
    # free batch leaves cand_*/sense_cand_* None. The only group that also
    # fills a genuinely PER-CANDIDATE query address (Sec 1.10) -- every
    # candidate's own `query_entity`/`query_relation` token, grounded via
    # `_ent_vec` (matching how `_garden_path_steps` grounded the SAME tokens
    # when writing C1/C2) and `rel:` + the relation token respectively.
    hyp_cand_entity = hyp_cand_mask = hyp_cand_prior = hyp_cand_gold = None
    hyp_cand_query_entity = hyp_cand_query_relation = None
    if any(hcs for *_row, _cs, _scs, hcs in rows):
        Hmax = max((len(hcs.candidates) for *_row, _cs, _scs, hcs_map in rows for hcs in hcs_map.values()),
                   default=1) or 1
        hyp_cand_entity = torch.zeros(b, T, Hmax, d)
        hyp_cand_mask = torch.zeros(b, T, Hmax)
        hyp_cand_prior = torch.zeros(b, T, Hmax)
        hyp_cand_gold = torch.full((b, T), -1, dtype=torch.long)
        hyp_cand_query_entity = torch.zeros(b, T, Hmax, d)
        hyp_cand_query_relation = torch.zeros(b, T, Hmax, d)
        for i, (steps, opt, a, ok, _cs, _scs, hcs_map) in enumerate(rows):
            for t, hcs in hcs_map.items():
                for j, c in enumerate(hcs.candidates):
                    # A candidate's OWN identity/reading vector -- a scorer
                    # INPUT feature (mirrors the entity branch's cand_entity:
                    # the candidate atom's own vector, used alongside its
                    # memory readout, NOT the resolved value itself). Grounds
                    # the same way the per-candidate query address does
                    # (query_entity is always set for every reading this
                    # curriculum builds): candidate identity IS "the entity
                    # this reading's address is about."
                    if c.query_entity is not None:
                        qe_vec = torch.from_numpy(
                            _ent_vec(c.query_entity, resolver, codec, cache, meaning_source))
                        hyp_cand_entity[i, t, j] = qe_vec
                        hyp_cand_query_entity[i, t, j] = qe_vec
                    hyp_cand_mask[i, t, j] = 1.0
                    hyp_cand_prior[i, t, j] = c.prior
                    if c.query_relation is not None:
                        hyp_cand_query_relation[i, t, j] = torch.from_numpy(
                            codec.filler_vec("rel:" + c.query_relation))
                if hcs.gold_index is not None:
                    hyp_cand_gold[i, t] = hcs.gold_index

    return ClauseBatch(ent, rel, val, prd, is_q, mask, opts, ans, crd, ans_ok,
                       cand_entity, cand_mask, cand_prior, cand_feature, cand_gold,
                       cand_feature_per_candidate,
                       sense_cand_entity, sense_cand_mask, sense_cand_prior,
                       sense_cand_context, sense_cand_gold,
                       sense_cand_subject, sense_cand_subject_rel,
                       hyp_cand_entity, hyp_cand_mask, hyp_cand_prior, hyp_cand_gold,
                       hyp_cand_query_entity, hyp_cand_query_relation)


# ---------------------------------------------------------------------------
# Learned reaction policy (the ONLY parameters)
# ---------------------------------------------------------------------------
class ClauseReactor(nn.Module):
    """GRU controller over grounded clause triples + the order-3 entity memory.

    Learns: a write gate (commit/overwrite/trust), a respond weight (timing), and a
    generated response meaning-vector. No embeddings — input is fixed grounded vectors.

    M53b (RESOLVER_BUILD_PLAN Phase 2, "Agent 3"): an OPTIONAL ``resolver``
    (:class:`nsm_ct.resolver.Resolver` -- Track A :class:`~nsm_ct.resolver.CorefHead`
    or Track B :class:`~nsm_ct.resolver.SharedScorer`) may be installed. Default
    ``None`` -- forward is then BYTE-IDENTICAL to pre-M53b (regression-tested in
    tests/test_resolver.py's ``test_no_resolver_byte_identity_regression``): the
    resolver branch below never executes, so nothing about the per-step computation
    changes. When a resolver IS installed and the batch carries candidate sets
    (``batch.cand_mask is not None``), at every step whose row has a real candidate
    set (M53a's pronoun-subject steps today) the resolver's collapsed choice
    REPLACES the placeholder gold-bound ``value`` for that step before it enters the
    GRU / gets written to memory -- MIND_INTERFACE.md §3's "collapse candidates ->
    binding, before the write". Steps/rows with no candidate set are untouched.

    M54 (RESOLVER_BUILD_PLAN Phase 3, "Agent 4"): a SECOND, independent,
    OPTIONAL ``sense_resolver`` slot for the v1 "value" candidate set
    (homograph senses, ``batch.sense_cand_mask``) -- kept SEPARATE from
    ``resolver`` rather than reusing one slot for both kinds, because Track A
    needs TWO DIFFERENT specialist heads active in the SAME forward pass
    (:class:`~nsm_ct.resolver.CorefHead` for pronoun steps,
    :class:`~nsm_ct.resolver.SenseHead` for sense steps -- pronoun and
    ambiguity episodes coexist in one M54 batch), while Track B's whole
    argument is that ONE :class:`~nsm_ct.resolver.SharedScorer` instance can
    be passed to BOTH slots (``resolver is sense_resolver``, a genuine
    shared-weights setup -- ``scripts/train_resolver.py`` does exactly this).
    Default ``None`` -- when neither ``sense_resolver`` nor
    ``batch.sense_cand_mask`` is set, the sense-collapse branch never
    executes, so a batch/model with no sense involvement is untouched by
    M54 (verified alongside the M53 byte-identity regression).
    """

    def __init__(self, dim: int, hidden: int = 128, resolver: Optional[Resolver] = None,
                 sense_resolver: Optional[Resolver] = None,
                 hyp_resolver: Optional[Resolver] = None) -> None:
        super().__init__()
        self.dim = dim
        # (entity, relation, value, predicate, coord, mem_read)
        self.gru = nn.GRUCell(6 * dim, hidden)
        self.write_gate = nn.Linear(hidden, 1)            # how strongly to commit value
        self.overwrite_gate = nn.Linear(hidden, 1)        # replace (update) vs add (vote)
        self.decide_truth = nn.Linear(hidden + dim, 1)    # refute (subtract) this value? (the truth policy)
        self.respond = nn.Linear(hidden, 1)
        self.response = nn.Linear(hidden + dim, dim)      # generate the response meaning-vector
        self.resolver = resolver                          # M53b: optional, None = pre-M53b behavior
        self.sense_resolver = sense_resolver               # M54: optional, None = pre-M54 behavior
        # M55a: a THIRD, independent, OPTIONAL slot for the v1 "parse
        # hypothesis" candidate set (batch.hyp_cand_mask). Default None --
        # M55a is plumbing only (no resolver trained yet, mirrors M53a's
        # pronoun placeholder before M53b existed); the collapse branch
        # below never executes with no hyp_resolver installed, so every
        # batch/model without one is untouched (verified alongside the
        # M53/M54 byte-identity regressions).
        self.hyp_resolver = hyp_resolver

    @staticmethod
    def _collapse_weights(logits: torch.Tensor, training: bool) -> torch.Tensor:
        """Soft (training) or hard argmax (eval) collapse weights -- factored
        out of the pre-M54 ``_collapse`` body UNCHANGED so both the entity
        and (M54) sense branches share it byte-for-byte."""
        if training:
            return torch.softmax(logits, dim=-1)                                # soft collapse (gradients)
        C = logits.shape[-1]
        return F.one_hot(logits.argmax(-1), num_classes=C).to(logits.dtype)     # hard collapse at eval

    @staticmethod
    def _top2_margin(logits: torch.Tensor, has_cand: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """``top1 - top2`` logit margin per row, 0 where ``has_cand`` is False --
        factored out of the pre-M54 ``_collapse`` body UNCHANGED."""
        margin = torch.zeros(ref.shape[0], device=ref.device)
        if logits.shape[-1] >= 2:
            top2 = torch.topk(logits, k=2, dim=-1).values
            margin = torch.where(has_cand, top2[:, 0] - top2[:, 1], margin)
        return margin

    def _collapse(self, memory: torch.Tensor, state: torch.Tensor, mem_read: torch.Tensor,
                  r: torch.Tensor, v: torch.Tensor, batch: ClauseBatch, t: int):
        """M53b/M54 collapse step for step ``t``: resolves ENTITY candidates
        (M53b, arithmetic UNCHANGED from pre-M54) and then SENSE candidates
        (M54, new) against the (possibly entity-collapsed) value. The two
        kinds are always on DISJOINT (row, step) entries in every M54
        curriculum (an ambiguity episode's step never also carries a pronoun
        candidate set), so applying them in sequence is safe: each only
        touches rows where its own ``has_cand`` is true, via ``torch.where``.
        Returns ``(value, ent_logits, ent_margin, sense_logits, sense_margin,
        hyp_logits, hyp_margin)`` -- any of the last six is ``None`` exactly
        when the corresponding resolver/candidate data is absent (mirrors
        the pre-M54 ``(v, None, None)`` contract for the no-resolver case
        component-wise).

        M55a's hypothesis branch (new) is arithmetically the ENTITY branch's
        twin, not the sense branch's: each candidate's per-candidate ``Addr``
        (``batch.hyp_cand_query_entity``/``hyp_cand_query_relation`` --
        dev/TRACK_C_DESIGN.md Sec 1.10) is queried via
        :func:`~nsm_ct.resolver.query_candidates_per_addr` (the ENTITY
        branch's :func:`~nsm_ct.resolver.query_candidates` twin -- per-
        candidate relation instead of one clause-shared relation), and the
        RESOLVED value is the weighted sum of THAT memory readout (not the
        candidate vectors themselves, unlike the sense branch) -- a
        hypothesis's identity (``hyp_cand_entity``) is a SCORER INPUT
        feature, not a value in its own right; what it asserts is only
        knowable by actually querying its own address.

        M54's sense branch is the "thinnest possible projection" adapting
        :class:`~nsm_ct.resolver.SharedScorer` (Track B, UNCHANGED code) and
        the new :class:`~nsm_ct.resolver.SenseHead` (Track A) to sense
        candidates without changing the :class:`~nsm_ct.resolver.Resolver`
        contract's fixed slots at all:
        - ``cand_entity`` <- the candidate SENSE VECTORS directly (there is
          no entity address a meaning vector can stand in for, so unlike the
          entity branch there is no ``query_candidates`` call here).
        - ``cand_feature`` <- zero (``membrane.FEATURE_DIM`` width) -- senses
          carry no gender/person mention feature; zero keeps SharedScorer's
          fixed input width intact.
        - ``mem_read`` <- THIS step's own memory readout (the running
          content at the homograph's address) PLUS the same-clause
          other-role vector (``batch.sense_cand_context`` --
          ``nsm_ct.membrane.SenseCandidateSet.context_word``, grounded at
          batch-build time) PLUS (M54b addendum, ``batch.sense_cand_subject``
          only -- None for the M32 curriculum) a GENUINE live memory query
          ``em.query(memory, sense_cand_subject, sense_cand_subject_rel)``:
          the homograph event's SUBJECT's own prior fact, read from the SAME
          memory tensor this very step's generic read/write already uses,
          not a batch-build-time-grounded shortcut. Broadcast identically
          across every candidate slot -- MIND_INTERFACE.md §3's "context IS
          the memory readout (+ optionally the step's other-role vectors)"
          for M54/M54b.
        - the RESOLVED value is the weighted SUM OF THE CANDIDATE VECTORS
          THEMSELVES (not their memory readout, unlike the entity branch --
          a sense candidate's vector already IS the resolved meaning).
        """
        ent_logits = ent_margin = None
        if self.resolver is not None and batch.cand_mask is not None:
            ce_t, cf_t = batch.cand_entity[:, t], batch.cand_feature[:, t]
            cp_t, cm_t = batch.cand_prior[:, t], batch.cand_mask[:, t]
            cand_mem_read = query_candidates(memory, ce_t, r)                    # [B, C, d]
            # M56b: pass the per-candidate feature register (§1.8) ONLY to a
            # resolver that opted in (`use_cand_feature=True` -- CorefHead
            # only today); SharedScorer/SenseHead are never called with this
            # kwarg at all, so "Do NOT change SharedScorer" holds literally
            # (zero edits, zero new call shape) for every non-opted-in track.
            extra = {}
            if getattr(self.resolver, "use_cand_feature", False) and batch.cand_feature_per_candidate is not None:
                extra["cand_feature_per_candidate"] = batch.cand_feature_per_candidate[:, t]
            logits = self.resolver(ce_t, cf_t, cp_t, cm_t, cand_mem_read, state, **extra)  # [B, C]
            logits = logits.masked_fill(cm_t <= 0, -1e9)
            has_cand = cm_t.sum(-1) > 0                                          # [B]
            w = self._collapse_weights(logits, self.training)
            resolved_v = (w.unsqueeze(-1) * cand_mem_read).sum(1)                # [B, d]
            v = torch.where(has_cand.unsqueeze(-1), resolved_v, v)
            ent_logits, ent_margin = logits, self._top2_margin(logits, has_cand, v)

        sense_logits = sense_margin = None
        if self.sense_resolver is not None and batch.sense_cand_mask is not None:
            sc_t = batch.sense_cand_entity[:, t]                                 # [B, C, d]  sense VECTORS
            sp_t, sm_t = batch.sense_cand_prior[:, t], batch.sense_cand_mask[:, t]
            b, C, sd = sc_t.shape
            extra_ctx = mem_read
            if batch.sense_cand_subject is not None:                            # M54b addendum, see docstring
                subj_read = em.query(memory, batch.sense_cand_subject[:, t], batch.sense_cand_subject_rel[:, t])
                extra_ctx = extra_ctx + subj_read
            ctx = (extra_ctx + batch.sense_cand_context[:, t]).unsqueeze(1).expand(b, C, sd)
            feat0 = sc_t.new_zeros(b, FEATURE_DIM)          # thinnest projection, see docstring above
            logits = self.sense_resolver(sc_t, feat0, sp_t, sm_t, ctx, state)     # [B, C]
            logits = logits.masked_fill(sm_t <= 0, -1e9)
            has_cand = sm_t.sum(-1) > 0                                          # [B]
            w = self._collapse_weights(logits, self.training)
            resolved_v = (w.unsqueeze(-1) * sc_t).sum(1)                         # [B, d] -- candidates ARE the value
            v = torch.where(has_cand.unsqueeze(-1), resolved_v, v)
            sense_logits, sense_margin = logits, self._top2_margin(logits, has_cand, v)

        hyp_logits = hyp_margin = None
        if self.hyp_resolver is not None and batch.hyp_cand_mask is not None:
            hc_t = batch.hyp_cand_entity[:, t]                                   # [B, C, d]  reading identity vectors
            hp_t, hm_t = batch.hyp_cand_prior[:, t], batch.hyp_cand_mask[:, t]
            hqe_t, hqr_t = batch.hyp_cand_query_entity[:, t], batch.hyp_cand_query_relation[:, t]
            cand_mem_read = query_candidates_per_addr(memory, hqe_t, hqr_t)          # [B, C, d] -- Sec 1.10's per-candidate Addr
            feat0 = hc_t.new_zeros(hc_t.shape[0], FEATURE_DIM)
            logits = self.hyp_resolver(hc_t, feat0, hp_t, hm_t, cand_mem_read, state)   # [B, C]
            logits = logits.masked_fill(hm_t <= 0, -1e9)
            has_cand = hm_t.sum(-1) > 0                                          # [B]
            w = self._collapse_weights(logits, self.training)
            resolved_v = (w.unsqueeze(-1) * cand_mem_read).sum(1)                # [B, d]
            v = torch.where(has_cand.unsqueeze(-1), resolved_v, v)
            hyp_logits, hyp_margin = logits, self._top2_margin(logits, has_cand, v)

        return v, ent_logits, ent_margin, sense_logits, sense_margin, hyp_logits, hyp_margin

    def forward(self, batch: ClauseBatch) -> Dict[str, torch.Tensor]:
        b, T, d = batch.entity.shape
        device = batch.entity.device
        state = torch.zeros(b, self.gru.hidden_size, device=device)
        memory = em.init_memory(b, d, device)

        coord = batch._coord()
        have_resolver_data = self.resolver is not None and batch.cand_mask is not None
        have_sense_data = self.sense_resolver is not None and batch.sense_cand_mask is not None
        have_hyp_data = self.hyp_resolver is not None and batch.hyp_cand_mask is not None
        resp_logits, resp_vecs = [], []
        resolver_logits_all, resolver_margin_all = [], []
        sense_logits_all, sense_margin_all = [], []
        hyp_logits_all, hyp_margin_all = [], []
        for t in range(T):
            e, r, v = batch.entity[:, t], batch.relation[:, t], batch.value[:, t]
            p, c = batch.pred[:, t], coord[:, t]
            real, isq = batch.mask[:, t], batch.is_q[:, t]
            mem_read = em.query(memory, e, r)                          # [B, d]
            (v, res_logits_t, res_margin_t, sense_logits_t, sense_margin_t,
             hyp_logits_t, hyp_margin_t) = self._collapse(
                memory, state, mem_read, r, v, batch, t)
            state = self.gru(torch.cat([e, r, v, p, c, mem_read], dim=-1), state)
            stmt = real * (1.0 - isq)                                  # statement (write) step
            # write gate: statement steps only (questions carry no value)
            gate = torch.sigmoid(self.write_gate(state)).squeeze(-1) * stmt
            # overwrite gate: replace the old binding (update) vs accumulate (vote)
            owr = torch.sigmoid(self.overwrite_gate(state)).squeeze(-1) * gate
            # decide-truth: refute (SUBTRACT) this value — the learned NOT policy. Driven
            # by the coord=NOT signal but supervised only by the answer. value_gate may go
            # negative, so a negated value cancels a previously-voted one (A or B, not A → B).
            neg = torch.sigmoid(self.decide_truth(torch.cat([state, v], dim=-1))).squeeze(-1) * stmt
            memory = em.write(memory, e, r, v, gate - neg, overwrite=owr)
            # respond weight (timing) + generated response meaning-vector
            rl = self.respond(state).squeeze(-1)
            rl = rl.masked_fill(real <= 0, float("-inf"))
            resp_logits.append(rl)
            resp_vecs.append(self.response(torch.cat([state, mem_read], dim=-1)))  # [B, d]
            if have_resolver_data:
                resolver_logits_all.append(res_logits_t)
                resolver_margin_all.append(res_margin_t)
            if have_sense_data:
                sense_logits_all.append(sense_logits_t)
                sense_margin_all.append(sense_margin_t)
            if have_hyp_data:
                hyp_logits_all.append(hyp_logits_t)
                hyp_margin_all.append(hyp_margin_t)

        RL = torch.stack(resp_logits, dim=1)               # [B, T]
        RV = torch.stack(resp_vecs, dim=1)                 # [B, T, d]
        w = torch.softmax(RL, dim=1)                        # respond distribution over steps
        r = (w.unsqueeze(-1) * RV).sum(dim=1)              # [B, d] aggregated response

        # contrastive answer: cosine(generated r, option meaning-vectors)
        rn = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
        on = batch.options / (batch.options.norm(dim=-1, keepdim=True) + 1e-8)
        answer_logits = torch.einsum("bd,bkd->bk", rn, on) * 10.0   # temperature

        out = {"answer_logits": answer_logits, "response": r,
               "respond_gates": w, "respond_position": (w * batch.is_q).sum(1)}
        if have_resolver_data:
            out["resolver_logits"] = torch.stack(resolver_logits_all, dim=1)   # [B, T, C]
            out["resolver_margin"] = torch.stack(resolver_margin_all, dim=1)   # [B, T]
        if have_sense_data:
            out["sense_resolver_logits"] = torch.stack(sense_logits_all, dim=1)   # [B, T, C]
            out["sense_resolver_margin"] = torch.stack(sense_margin_all, dim=1)   # [B, T]
        if have_hyp_data:
            out["hyp_resolver_logits"] = torch.stack(hyp_logits_all, dim=1)   # [B, T, C]
            out["hyp_resolver_margin"] = torch.stack(hyp_margin_all, dim=1)   # [B, T]
        return out
