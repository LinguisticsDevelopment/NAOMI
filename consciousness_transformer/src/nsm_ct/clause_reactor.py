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
from .resolver import Resolver, query_candidates
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
def _ambiguity_steps(ep, parser, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                      meaning_source: MeaningSource, sense_bind: str = "gold"):
    """Grounded stream + per-step :class:`membrane.SenseCandidateSet` map for
    one M32 ambiguity episode. Returns ``(steps, sense_cand_sets)`` -- the
    same shape ``_pronoun_context_step``'s caller already threads through
    ``build_clause_batch``'s ``cand_sets`` dict, just for sense candidates.
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
            provenance={"sentence_index": si, "sentence": sent})
        sense_cand_sets[len(steps)] = cs
        steps.append((hom_vec, sense_rel, bind_vec, pred_is, z, 0))
    steps.append((hom_vec, sense_rel, z, q_pred, z, 1))
    return steps, sense_cand_sets


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

    def _coord(self) -> torch.Tensor:
        return self.coord if self.coord is not None else torch.zeros_like(self.entity)

    def _cand_fields(self, xform):
        return tuple(xform(t) if t is not None else None for t in
                     (self.cand_entity, self.cand_mask, self.cand_prior,
                      self.cand_feature, self.cand_gold))

    def _sense_cand_fields(self, xform):
        return tuple(xform(t) if t is not None else None for t in
                     (self.sense_cand_entity, self.sense_cand_mask, self.sense_cand_prior,
                      self.sense_cand_context, self.sense_cand_gold))

    def to(self, device):
        coord = self.coord.to(device) if self.coord is not None else None
        ans_ok = self.answerable.to(device) if self.answerable is not None else None
        cand = self._cand_fields(lambda t: t.to(device))
        scand = self._sense_cand_fields(lambda t: t.to(device))
        return ClauseBatch(self.entity.to(device), self.relation.to(device),
                           self.value.to(device), self.pred.to(device),
                           self.is_q.to(device), self.mask.to(device),
                           self.options.to(device), self.answer.to(device), coord, ans_ok,
                           *cand, *scand)

    def subset(self, idx) -> "ClauseBatch":
        """A minibatch over the leading (episode) dimension."""
        coord = self.coord[idx] if self.coord is not None else None
        ans_ok = self.answerable[idx] if self.answerable is not None else None
        cand = self._cand_fields(lambda t: t[idx])
        scand = self._sense_cand_fields(lambda t: t[idx])
        return ClauseBatch(self.entity[idx], self.relation[idx], self.value[idx],
                           self.pred[idx], self.is_q[idx], self.mask[idx],
                           self.options[idx], self.answer[idx], coord, ans_ok, *cand, *scand)


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
    """
    cache: Dict[str, np.ndarray] = {}
    d = codec.dim
    q_pred = codec.filler_vec("pred:?")             # the question's (unknown) predicate
    z = np.zeros(d, np.float32)
    rows = []
    for ep in episodes:
        cand_sets: Dict[int, "membrane.EntityCandidateSet"] = {}
        sense_cand_sets: Dict[int, "membrane.SenseCandidateSet"] = {}
        if getattr(ep, "level", 0) >= 9 and ep.meta.get("query"):
            steps = _reasoning_steps(ep, resolver, codec, cache, meaning_source)   # L9-L11 reasoning stream
        elif ep.meta.get("homograph"):
            steps, sense_cand_sets = _ambiguity_steps(ep, parser, resolver, codec, cache,
                                                       meaning_source, sense_bind)   # M54 ambiguity stream
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
                     cand_sets, sense_cand_sets))

    b = len(rows)
    T = max(len(s) for s, _, _, _, _, _ in rows)
    K = max(len(o) for _, o, _, _, _, _ in rows)
    ent = torch.zeros(b, T, d); rel = torch.zeros(b, T, d); val = torch.zeros(b, T, d)
    prd = torch.zeros(b, T, d); crd = torch.zeros(b, T, d)
    is_q = torch.zeros(b, T); mask = torch.zeros(b, T)
    opts = torch.zeros(b, K, d); ans = torch.zeros(b, dtype=torch.long)
    ans_ok = torch.zeros(b)
    for i, (steps, opt, a, ok, _cs, _scs) in enumerate(rows):
        for t, (e, r, v, p, c, q) in enumerate(steps):
            ent[i, t] = torch.from_numpy(e); rel[i, t] = torch.from_numpy(r)
            val[i, t] = torch.from_numpy(v); prd[i, t] = torch.from_numpy(p)
            crd[i, t] = torch.from_numpy(c); is_q[i, t] = q; mask[i, t] = 1.0
        for k, ov in enumerate(opt):
            opts[i, k] = torch.from_numpy(ov)
        ans[i] = a; ans_ok[i] = ok

    cand_entity = cand_mask = cand_prior = cand_feature = cand_gold = None
    if any(cs for *_row, cs, _scs in rows):
        Cmax = max((len(cs.candidates) for *_row, cs_map, _scs in rows for cs in cs_map.values()),
                   default=1) or 1
        cand_entity = torch.zeros(b, T, Cmax, d)
        cand_mask = torch.zeros(b, T, Cmax)
        cand_prior = torch.zeros(b, T, Cmax)
        cand_feature = torch.zeros(b, T, membrane.FEATURE_DIM)
        cand_gold = torch.full((b, T), -1, dtype=torch.long)
        for i, (steps, opt, a, ok, cs_map, _scs) in enumerate(rows):
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

    # M54: SAME shape/pattern as the cand_* block above, but for
    # nsm_ct.membrane.SenseCandidateSet -- a wholly SEPARATE tensor group
    # (see ClauseBatch's field comment for why), so an ambiguity-free batch
    # leaves it None exactly like a pronoun-free batch leaves cand_* None.
    sense_cand_entity = sense_cand_mask = sense_cand_prior = None
    sense_cand_context = sense_cand_gold = None
    if any(scs for *_row, _cs, scs in rows):
        Smax = max((len(scs.candidates) for *_row, _cs, scs_map in rows for scs in scs_map.values()),
                   default=1) or 1
        sense_cand_entity = torch.zeros(b, T, Smax, d)
        sense_cand_mask = torch.zeros(b, T, Smax)
        sense_cand_prior = torch.zeros(b, T, Smax)
        sense_cand_context = torch.zeros(b, T, d)
        sense_cand_gold = torch.full((b, T), -1, dtype=torch.long)
        for i, (steps, opt, a, ok, _cs, scs_map) in enumerate(rows):
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

    return ClauseBatch(ent, rel, val, prd, is_q, mask, opts, ans, crd, ans_ok,
                       cand_entity, cand_mask, cand_prior, cand_feature, cand_gold,
                       sense_cand_entity, sense_cand_mask, sense_cand_prior,
                       sense_cand_context, sense_cand_gold)


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
                 sense_resolver: Optional[Resolver] = None) -> None:
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
        Returns ``(value, ent_logits, ent_margin, sense_logits, sense_margin)``
        -- any of the last four is ``None`` exactly when the corresponding
        resolver/candidate data is absent (mirrors the pre-M54
        ``(v, None, None)`` contract for the no-resolver case component-wise).

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
          batch-build time), broadcast identically across every candidate
          slot -- MIND_INTERFACE.md §3's "context IS the memory readout (+
          optionally the step's other-role vectors)" for M54.
        - the RESOLVED value is the weighted SUM OF THE CANDIDATE VECTORS
          THEMSELVES (not their memory readout, unlike the entity branch --
          a sense candidate's vector already IS the resolved meaning).
        """
        ent_logits = ent_margin = None
        if self.resolver is not None and batch.cand_mask is not None:
            ce_t, cf_t = batch.cand_entity[:, t], batch.cand_feature[:, t]
            cp_t, cm_t = batch.cand_prior[:, t], batch.cand_mask[:, t]
            cand_mem_read = query_candidates(memory, ce_t, r)                    # [B, C, d]
            logits = self.resolver(ce_t, cf_t, cp_t, cm_t, cand_mem_read, state)  # [B, C]
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
            ctx = (mem_read + batch.sense_cand_context[:, t]).unsqueeze(1).expand(b, C, sd)
            feat0 = sc_t.new_zeros(b, FEATURE_DIM)          # thinnest projection, see docstring above
            logits = self.sense_resolver(sc_t, feat0, sp_t, sm_t, ctx, state)     # [B, C]
            logits = logits.masked_fill(sm_t <= 0, -1e9)
            has_cand = sm_t.sum(-1) > 0                                          # [B]
            w = self._collapse_weights(logits, self.training)
            resolved_v = (w.unsqueeze(-1) * sc_t).sum(1)                         # [B, d] -- candidates ARE the value
            v = torch.where(has_cand.unsqueeze(-1), resolved_v, v)
            sense_logits, sense_margin = logits, self._top2_margin(logits, has_cand, v)

        return v, ent_logits, ent_margin, sense_logits, sense_margin

    def forward(self, batch: ClauseBatch) -> Dict[str, torch.Tensor]:
        b, T, d = batch.entity.shape
        device = batch.entity.device
        state = torch.zeros(b, self.gru.hidden_size, device=device)
        memory = em.init_memory(b, d, device)

        coord = batch._coord()
        have_resolver_data = self.resolver is not None and batch.cand_mask is not None
        have_sense_data = self.sense_resolver is not None and batch.sense_cand_mask is not None
        resp_logits, resp_vecs = [], []
        resolver_logits_all, resolver_margin_all = [], []
        sense_logits_all, sense_margin_all = [], []
        for t in range(T):
            e, r, v = batch.entity[:, t], batch.relation[:, t], batch.value[:, t]
            p, c = batch.pred[:, t], coord[:, t]
            real, isq = batch.mask[:, t], batch.is_q[:, t]
            mem_read = em.query(memory, e, r)                          # [B, d]
            v, res_logits_t, res_margin_t, sense_logits_t, sense_margin_t = self._collapse(
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
        return out
