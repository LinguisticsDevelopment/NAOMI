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
from . import ops
from .clause import _PRONOUNS, extract_discourse
from .episode import _NAMES
from .instances import InstanceRegistry
from .ltm import mem_total
from .membrane import FEATURE_DIM
from .resolver import Resolver, evidence_interaction, query_candidates, query_candidates_per_addr
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


def _ground_evidence_target(target: str, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                             meaning_source: MeaningSource = "usvs") -> np.ndarray:
    """M57c.3: ground a :attr:`nsm_ct.membrane.EntityCandidateSet.evidence_target`
    key into the SAME vector the matching attribute fact was WRITTEN with,
    so :func:`nsm_ct.resolver.evidence_interaction`'s ``cos(readout, target)``
    is meaningful (RESEARCH_NOTES "M57c battery #2"'s diagnosis: the
    resolver never had a target to compare a candidate's evidence readout
    against). ``target`` carries its OWN grounding-convention prefix:
    ``"kind:doctor"``/``"gender:F"`` reproduce ``codec.filler_vec`` EXACTLY
    (the identical call :func:`nsm_ct.clause_reactor._instance_steps`/
    :func:`_rich_steps` make when WRITING that attribute's value, e.g.
    ``codec.filler_vec("kind:" + kinds[r])``); ``"name:mary"`` (the
    ambiguous-shared-name device, whose evidence_relation reads attr:kind
    but whose referring expression names no kind at all) grounds via the
    ordinary entity-vector convention (:func:`_ent_vec`) instead -- by
    construction this does NOT correlate with the attr:kind readout it is
    compared against, correctly reflecting that ambiguous-name resolution
    is NOT attribute-decidable (see curriculum2's own "discourse recency
    for ambiguous_name" note); it disambiguates via a genuinely different
    channel (discourse order), out of scope for this interaction feature.

    M57e (morphology signals): ``"number:sg"``/``"number:pl"`` targets (the
    plural-pronoun referring/question steps :func:`_rich_steps` builds for a
    :class:`~nsm_ct.curriculum2.RichEpisodeGenerator` group episode) fall
    straight through to the generic ``codec.filler_vec(target)`` branch
    below -- no new branch needed, since ``"number:"`` is just another
    attribute-value grounding prefix in the SAME convention as ``"kind:"``/
    ``"gender:"``, reproducing exactly the vector every candidate's own
    attr:number fact was written with.
    """
    if target.startswith("name:"):
        return _ent_vec(target[len("name:"):], resolver, codec, cache, meaning_source)
    return codec.filler_vec(target)


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
        steps.append((_ent_vec(subj, resolver, codec, cache, meaning_source), codec.filler_vec("rel:PLACE"),
                      _content_vec(place, resolver, codec, cache, meaning_source),
                      pred_vec, coordv, 0))
    return steps


# Round 2 item 1 (PROSE PRONOUNS TO THE RESOLVER): the batch-grounding half
# of nsm_ct.corpus's own registry-broadening + gender-compatibility logic
# (see that module's ``_is_registrable_entity``/``_gender_compatible``) --
# duplicated here in miniature rather than imported: nsm_ct.corpus already
# imports FROM this module (``_TRANSFER_ROLE_MAP``), so the reverse import
# would be circular. Every non-pronoun argument token mentioned by
# ``context_sentences`` (any role -- mirrors nsm_ct.corpus._PassageRegistry.
# register's own "any arg, not just SUBJECT" scope), most-recently-
# mentioned first, filtered to those gender-compatible with ``pronoun``
# (membrane.PRONOUN_MORPHOLOGY + membrane.NAME_GENDER -- unknown on either
# side never excludes a candidate).
def _prose_pronoun_candidates(pronoun: str, context_sentences, parser) -> List[str]:
    recent: List[str] = []
    for sent in context_sentences:
        graph = parser._parse_graph(sent) if hasattr(parser, "_parse_graph") else None
        clauses, _links = extract_discourse(graph)
        for cl in clauses:
            for _rel, arg in cl.args:
                tok = (arg.token or "").lower()
                if tok and tok not in _PRONOUNS:
                    if tok in recent:
                        recent.remove(tok)
                    recent.append(tok)
    candidates = list(reversed(recent))
    p_gender = membrane.PRONOUN_MORPHOLOGY.get(pronoun, ("unknown", "sg", "3"))[0]

    def _compat(name: str) -> bool:
        n_gender = membrane.NAME_GENDER.get(name)
        return p_gender == "unknown" or n_gender is None or p_gender == n_gender

    return [n for n in candidates if _compat(n)]


def _prose_steps(ep, parser, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                  meaning_source: MeaningSource):
    """Round 2 item 1: the ``kind == "prose"`` (``nsm_ct.corpus.
    make_episodes`` only) counterpart of :func:`_context_steps`.

    Every clause shape :func:`_context_steps` already grounds is grounded
    HERE IDENTICALLY: an OBJECT-bearing transfer clause (unrolled one step
    per other role, byte-for-byte the same as :func:`_context_steps` --
    including a pronoun-bearing OTHER role, which stays on
    :func:`_context_steps`'s own naive grounding: there the unresolved slot
    is the triple's VALUE, an address the collapse machinery has no
    contract for redirecting, only the ENTITY axis can be -- see
    :func:`nsm_ct.corpus._extract_triples`'s own note on this exact
    asymmetry); a SUBJECT+PLACE clause, or (round 2 item 3) a SUBJECT+
    single-other-role clause more generally, whose SUBJECT is a NAMED
    entity.

    The one behavioral difference: a SUBJECT+single-other-role clause whose
    SUBJECT is a PERSONAL PRONOUN. Instead of naively grounding the pronoun
    as its own fixed content atom (what :func:`_context_steps` does for
    every caller -- correct there, since no curriculum generator ever
    routes a pronoun-subject sentence through it uninterceped), a
    gender-compatible antecedent candidate set
    (:class:`nsm_ct.membrane.EntityCandidateSet`, ``addr_redirect=True``,
    ``evidence_relation="gender"``, its target the pronoun's OWN gender
    atom) is built from the entities mentioned earlier in ``ep.context``
    (:func:`_prose_pronoun_candidates`, most-recent-first) -- mirroring
    :func:`_instance_steps`/:func:`_rich_steps`'s own "pronoun device"
    contract exactly: a placeholder (garbage) entity address, the clause's
    real (known) relation+value, and the candidate set's resolver-owned
    ``addr_redirect`` deciding which entity's node the write actually lands
    on. ``gold_antecedent`` is never supplied -- prose carries no ground
    truth for which candidate is correct (mirrors ``nsm_ct.corpus``'s own
    "parsed-pronoun-resolved" taxonomy code; this is that same design's
    batch-grounding half). A pronoun with NO gender-compatible antecedent
    yet falls back to :func:`_context_steps`'s own naive grounding -- there
    is nothing better to offer the resolver.

    Returns ``(steps, cand_sets)`` -- ``cand_sets`` maps a row-local step
    index to its :class:`~nsm_ct.membrane.EntityCandidateSet`, ``{}`` (not
    ``None``) for a prose episode with no pronoun-redirect step at all,
    matching every other step-builder's "empty dict, not None" convention
    for :func:`build_clause_batch`'s per-episode ``cand_sets`` local.
    """
    z = np.zeros(codec.dim, np.float32)
    steps = []
    cand_sets: Dict[int, "membrane.EntityCandidateSet"] = {}
    mention_log: Dict[str, List[int]] = {}

    def _register_mention(name: str) -> None:
        mention_log.setdefault(name, []).append(len(steps))

    def _recency_fields(names: List[str]):
        ms = np.array([(mention_log[n][-1] if mention_log.get(n) else -1) for n in names], dtype=np.float32)
        mc = np.array([len(mention_log.get(n, [])) for n in names], dtype=np.float32)
        return ms, mc

    for si, sent in enumerate(ep.context):
        graph = parser._parse_graph(sent) if hasattr(parser, "_parse_graph") else None
        clauses, links = extract_discourse(graph)
        if not clauses:
            continue
        prime = links[0].prime if links else None
        coordv = codec.filler_vec(prime) if prime else z
        for cl in clauses:
            pred_vec = codec.filler_vec("pred:" + (cl.predicate or "").lower())
            obj_tok = next(((arg.token or "").lower() for rel, arg in cl.args if rel == "OBJECT"), None)
            if obj_tok:
                # Unchanged from _context_steps -- see docstring.
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
                    if tok not in _PRONOUNS:
                        _register_mention(tok)
                continue
            subj = place = None
            for rel, arg in cl.args:
                if rel == "SUBJECT":
                    subj = (arg.token or "").lower()
                elif rel == "PLACE":
                    place = (arg.token or "").lower()
            other_non_place = [(rel, arg) for rel, arg in cl.args if rel not in ("SUBJECT", "PLACE")]
            if subj and place:
                out_rel, other_val = "PLACE", place
            elif subj and place is None and len(other_non_place) == 1:
                out_rel = "OBJECT"
                other_val = (other_non_place[0][1].token or "").lower()
            else:
                continue
            if not other_val:
                continue
            value_vec = _content_vec(other_val, resolver, codec, cache, meaning_source)
            if subj in _PRONOUNS:
                candidates = _prose_pronoun_candidates(subj, ep.context[:si], parser)
                if candidates:
                    placeholder = _ent_vec(subj, resolver, codec, cache, meaning_source)
                    step_idx = len(steps)
                    steps.append((placeholder, codec.filler_vec("rel:" + out_rel), value_vec,
                                  pred_vec, coordv, 0))
                    cand = membrane.pronoun_entity_candidate_set(
                        subj, candidates, gold_antecedent=None,
                        provenance={"sentence_index": si, "sentence": sent, "kind": "prose"},
                        addr_redirect=True)
                    p_gender = membrane.PRONOUN_MORPHOLOGY.get(subj, ("unknown", "sg", "3"))[0]
                    cand.evidence_relation = "gender"
                    cand.evidence_target = "gender:" + p_gender
                    cand.mention_steps, cand.mention_counts = _recency_fields(candidates)
                    cand_sets[step_idx] = cand
                    continue
                # no compatible antecedent yet -- fall through to the same
                # naive grounding _context_steps would use.
            entity_vec = _ent_vec(subj, resolver, codec, cache, meaning_source)
            steps.append((entity_vec, codec.filler_vec("rel:" + out_rel), value_vec, pred_vec, coordv, 0))
            if subj not in _PRONOUNS:
                _register_mention(subj)
    return steps, cand_sets


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
                        meaning_source: MeaningSource, reading_bind: str = "gold"):
    """Grounded stream + the garden-path step's
    :class:`membrane.HypothesisCandidateSet` for one
    :class:`~nsm_ct.curriculum2.GardenPathCurriculumGenerator` episode.

    Episode shape (M55b, ``ep.meta`` keys in parens -- see that generator's
    docstring for the full binding-critical redesign):
        "{name_a_or_b} is in the {trait_word} ."       -- TRAIT cue (order per meta["cue_order"])
        "{name_b_or_a} is in the {other_trait_word} ." -- TRAIT cue (decoy)
        "{name_a} went to the {place_a} ."              -- the VERB reading's answer source
        "the {homograph} is in the {place_b} ."         -- the OBJECT reading's answer source
        "{name_a} can {homograph} ."                     -- the garden-path sentence
        Q: "where is {name_a} ?"

    M55a's placeholder mechanism is UNCHANGED (the collapse step's own value
    is still bound directly from a meta field so the memory pipeline is
    exercised end to end even with no resolver installed -- mirrors
    :func:`_pronoun_context_step`'s gold-antecedent pattern exactly), but
    M55b adds two things:

    1. Two new TRAIT steps (M55b), written under a DEDICATED relation
       (``rel:TRAIT``) that NEITHER reading's own PLACE query ever touches
       (see :class:`~nsm_ct.curriculum2.GardenPathCurriculumGenerator`'s
       docstring for why this can't reuse the per-candidate ``Addr``
       register), so the discriminating fact reaches the collapse step's
       resolver only through the controller's running GRU ``state`` -- see
       :class:`~nsm_ct.resolver.RankHead`'s docstring.
    2. ``reading_bind`` (mirrors ``_ambiguity_steps``'s ``sense_bind``):
       ``"gold"`` (default) placeholder-binds the collapse step's value to
       the TRUE gold reading's place (the M55a ceiling behavior, unchanged);
       ``"wrong"`` binds it to the OPPOSITE reading's place instead -- the
       M55b floor probe (``scripts/train_resolver.py --wrong-binding``):
       if forcing the wrong reading doesn't crater task accuracy, the
       reading isn't actually determining the answer, and the curriculum
       doesn't belong (RESEARCH_NOTES M53a/M54b's "capability curricula
       must make the capability NECESSARY" discipline, applied here as a
       gold-vs-wrong gap probe instead of a gold-vs-MFS one). The
       :class:`membrane.HypothesisCandidateSet`'s ``gold_index`` ALWAYS
       reflects the TRUE reading regardless of ``reading_bind`` -- mirrors
       ``_ambiguity_steps``'s ``sense_candidate_set(gold_sense=gold_sense,
       ...)`` always passing the true gold sense even under
       ``sense_bind="mfs"`` -- so a trained resolver's aux loss/eval never
       sees the forced-wrong value as its training target.

    The REAL :class:`membrane.HypothesisCandidateSet` (top-K parse
    exposure, real structural-score priors, per-candidate query addresses)
    still rides along in the batch's ``hyp_cand_*`` fields exactly as
    M55a built it -- unchanged.
    """
    d = codec.dim
    name_a = ep.meta["name_a"]
    homograph = ep.meta["gp_homograph"]   # NOT "homograph" -- see curriculum2.py's note (M54 key collision)
    place_a_word = ep.meta["place_a"]
    place_b_word = ep.meta["place_b"]
    true_reading = ep.meta["gold_reading"]        # "object" | "verb" -- ALWAYS the true gold
    if reading_bind == "wrong":
        write_reading = "verb" if true_reading == "object" else "object"
    else:
        write_reading = true_reading

    place_rel = codec.filler_vec("rel:PLACE")
    trait_rel = codec.filler_vec("rel:TRAIT")     # M55b: dedicated, never collides with rel:PLACE
    pred_is = codec.filler_vec("pred:is")
    q_pred = codec.filler_vec("pred:?")
    z = np.zeros(d, np.float32)

    name_a_vec = _ent_vec(name_a, resolver, codec, cache, meaning_source)      # var:name_a (a name)
    hom_vec = _ent_vec(homograph, resolver, codec, cache, meaning_source)      # content vec (not a name)
    place_a_vec = _content_vec(place_a_word, resolver, codec, cache, meaning_source)
    place_b_vec = _content_vec(place_b_word, resolver, codec, cache, meaning_source)

    steps = []
    # M55b T1/T2: the two entity-keyed TRAIT marker facts (gold cue +
    # decoy), in the curriculum's own randomized relative order
    # (meta["cue_order"]) -- mirrors SenseBindingCurriculumGenerator's
    # "cues first" discipline. Absent (pre-M55b episodes / defensive) ->
    # no marker steps at all, so a garden_path episode with no
    # "other_entity" key still builds (byte-compatible with any caller
    # constructing meta by hand without the new keys).
    other_entity = ep.meta.get("other_entity")
    if other_entity is not None:
        trait_word = ep.meta["trait_word"]
        other_trait_word = ep.meta["other_trait_word"]
        other_vec = _ent_vec(other_entity, resolver, codec, cache, meaning_source)
        trait_a_vec = _content_vec(trait_word, resolver, codec, cache, meaning_source)
        trait_b_vec = _content_vec(other_trait_word, resolver, codec, cache, meaning_source)
        marker_a = (name_a_vec, trait_rel, trait_a_vec, pred_is, z, 0)
        marker_b = (other_vec, trait_rel, trait_b_vec, pred_is, z, 0)
        cue_order = ep.meta.get("cue_order", ("a", "b"))
        steps += [marker_a, marker_b] if cue_order[0] == "a" else [marker_b, marker_a]

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
    gold_index = 0 if true_reading == "object" else 1        # ALWAYS the true reading (see docstring point 2)
    bind_vec = place_b_vec if write_reading == "object" else place_a_vec
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


# M57b (resolver-driven write-BACK, CLAUDE.md's M57 memory-schema decision):
# a WriteBackCurriculumGenerator episode (curriculum2.py -- gated on
# ep.meta["kind"] == "writeback", a key no other generator sets). Mirrors
# :func:`_reasoning_steps` in ONE respect that matters a lot here: it takes
# NO ``parser`` argument at all and never calls one -- every triple is built
# straight from ``ep.meta``, exactly like the L9-11 reasoning stream. Unlike
# ``_pronoun_context_step``/``_ambiguity_steps``/``_garden_path_steps`` (which
# all lean on the real parser to extract SUBJECT/OBJECT/PLACE roles from a
# sentence), a write-back episode's ``ep.context``/``ep.question`` strings
# exist for HUMAN READABILITY / provenance only -- they are never parsed.
# This sidesteps a real risk the pronoun-FIND template doesn't have: an
# attribute assertion's stated VALUE ("tall", "quiet", ...) is a genuine
# adjective, and quantum_parser's POS lexicon is WordNet-noun-biased (see
# curriculum2.py's own "fred" landmine note) -- there was no upside in
# betting a new sentence shape's parse reliability against the one thing
# this milestone is actually supposed to test (address redirection).
#
# The write-back triple itself (v2 -- see WriteBackCurriculumGenerator's own
# docstring for the full context shape: every entity gets a named attribute
# statement first, the pronoun statement comes LAST):
#   entity   = a PLACEHOLDER mention atom (``_ent_vec(pronoun, ...)`` --
#              "she"/"he" are not curriculum names, so this grounds to their
#              own CONTENT vector, a fixed atom distinct from EVERY var:<name>
#              atom in the episode). This is the address BEFORE collapse --
#              deliberately garbage: nothing else in the episode ever writes
#              to or queries "she"/"he"'s own content vector, so if the
#              resolver fails to redirect it, the write lands nowhere useful
#              and the target entity's own rel:ATTR slot keeps its EARLIER
#              (stale) named-attribute value instead of the overwrite --
#              exactly the signal the M57b v2 gate checks for.
#   relation = a NEW, dedicated ``rel:ATTR`` atom (never shared with
#              rel:PLACE/rel:TRAIT/rel:SENSE/etc.) -- the referent's own
#              named-attribute step, the write-back step, and the question
#              step all address it, so a correct redirect is both necessary
#              AND sufficient for the question to recover the right value,
#              and the write-back step's redirected write can OVERWRITE the
#              referent's earlier named-attribute value at that SAME address
#              (the reactor's own learned overwrite gate does this).
#   value    = the literally-stated attribute word (``ep.meta
#              ["pronoun_attr"]``) -- UNTOUCHED by collapse (M57b's "the
#              clause says WHAT is asserted; the resolver decides WHO it is
#              about").
#
# The candidate set (:func:`nsm_ct.membrane.pronoun_entity_candidate_set`,
# ``addr_redirect=True``) is built over the SAME two-name registry the write
# steps below establish (``[name_a, name_b]``, matching
# ``ep.meta["registry_order"]`` -- :class:`nsm_ct.curriculum2.
# WriteBackCurriculumGenerator`'s own bookkeeping), so each candidate's own
# grounded atom (``cand_entity[..., j]`` in the batch) is LITERALLY the same
# ``var:<name>`` vector the entity's own PLACE fact used to register it --
# the collapse's address-redirect sum (``Σ w·cand_entity``) can therefore
# land EXACTLY on the true referent's node, not merely something correlated
# with it.
def _writeback_steps(ep, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                      meaning_source: MeaningSource, cheat: bool = False,
                      no_gold: bool = False, force: Optional[str] = None):
    """Grounded stream + the write-back step's
    :class:`membrane.EntityCandidateSet` (``addr_redirect=True``) for one
    :class:`~nsm_ct.curriculum2.WriteBackCurriculumGenerator` episode (v2
    shape -- see that class's docstring: every entity gets its own NAMED
    attribute statement first, the pronoun statement comes LAST and
    OVERWRITES the referent's).

    ``cheat`` (M57b cheat-baseline arm): when True, the candidate set is
    dropped entirely (``cand_sets`` returned empty) -- the write-back step
    still carries the SAME placeholder (garbage) address, but with no
    candidate data at all the resolver (even if installed) never fires for
    it, so nothing in the batch can tell the model which node to write to.
    ``no_gold`` (M57b no-gold eval mode): when True, the candidate set's
    ``gold_index`` is forced ``None`` (``gold_antecedent=None`` passed to
    :func:`membrane.pronoun_entity_candidate_set`) regardless of
    ``ep.meta`` -- candidates + priors + the trained resolver are still
    fully present, only the supervision target is withheld, mirroring how
    :func:`membrane.pronoun_entity_candidate_set` already returns
    ``gold_index=None`` for an unresolvable open case.

    ``force`` (``"gold"`` | ``"wrong"`` | ``None``, replaces v1's
    ``wrong_binding`` aux-gold-corruption arm): when set, the pronoun step's
    candidate index to TEACHER-FORCE is recorded in the returned
    ``forced_map`` (``{step_idx: candidate_index}``) -- ``"gold"`` forces the
    TRUE referent's index, ``"wrong"`` forces the OTHER candidate's index
    (there are always exactly two candidates here), computed directly from
    ``ep.meta["true_antecedent"]``/the registry, independent of
    ``gold_antecedent`` (so it still works correctly under ``no_gold``).
    Consumed by :func:`build_clause_batch` to populate
    ``ClauseBatch.cand_forced_index`` -- see
    :meth:`ClauseReactor._collapse`'s teacher-forcing paragraph for why this
    replaces corrupting the resolver's own supervision target (task pressure
    simply overrode that corrupted target; forcing the COLLAPSE itself
    cannot be overridden that way). A no-op (``forced_map`` stays empty)
    when ``cheat`` is True (no candidate set exists to force an index into).
    """
    d = codec.dim
    name_a, name_b = ep.meta["name_a"], ep.meta["name_b"]
    place_a_word, place_b_word = ep.meta["place_a"], ep.meta["place_b"]
    pronoun = ep.meta["pronoun"]
    antecedent_name = ep.meta["true_antecedent"]
    other_name = ep.meta["other_entity"]
    stale_attr = ep.meta["stale_attr"]
    other_attr = ep.meta["other_attr"]
    pronoun_attr = ep.meta["pronoun_attr"]
    target_name = ep.meta["target_name"]
    gold_antecedent = None if no_gold else ep.meta.get("gold_antecedent")

    place_rel = codec.filler_vec("rel:PLACE")
    attr_rel = codec.filler_vec("rel:ATTR")          # M57b: dedicated, never collides with PLACE/TRAIT/SENSE
    pred_is = codec.filler_vec("pred:is")
    q_pred = codec.filler_vec("pred:?")
    z = np.zeros(d, np.float32)

    name_a_vec = _ent_vec(name_a, resolver, codec, cache, meaning_source)
    name_b_vec = _ent_vec(name_b, resolver, codec, cache, meaning_source)
    place_a_vec = _content_vec(place_a_word, resolver, codec, cache, meaning_source)
    place_b_vec = _content_vec(place_b_word, resolver, codec, cache, meaning_source)
    pronoun_vec = _ent_vec(pronoun, resolver, codec, cache, meaning_source)   # PLACEHOLDER address, see docstring
    antecedent_vec = _ent_vec(antecedent_name, resolver, codec, cache, meaning_source)
    other_name_vec = _ent_vec(other_name, resolver, codec, cache, meaning_source)
    stale_attr_vec = _ent_vec(stale_attr, resolver, codec, cache, meaning_source)
    other_attr_vec = _ent_vec(other_attr, resolver, codec, cache, meaning_source)
    pronoun_attr_vec = _ent_vec(pronoun_attr, resolver, codec, cache, meaning_source)
    target_vec = _ent_vec(target_name, resolver, codec, cache, meaning_source)

    steps = []
    # S0/S1: register both named entities with a distinguishing PLACE fact --
    # the same "own fact establishes the entity as a candidate referent"
    # convention PronounCurriculumGenerator/_pronoun_context_step already use.
    steps.append((name_a_vec, place_rel, place_a_vec, pred_is, z, 0))
    steps.append((name_b_vec, place_rel, place_b_vec, pred_is, z, 0))

    # S2: the referent's OWN named attribute statement ("mary is kind .") --
    # an ordinary write under rel:ATTR, the SAME relation the pronoun step
    # (S4) and the question both use, so S4's redirected write can OVERWRITE
    # this one at the SAME (entity, relation) address. Nothing here
    # special-cases the overwrite -- the reactor's own learned overwrite
    # gate (write_gate/overwrite_gate) does it, exactly like any other
    # repeated assertion about the same slot.
    steps.append((antecedent_vec, attr_rel, stale_attr_vec, pred_is, z, 0))

    # S3: the OTHER entity's OWN named attribute -- persists untouched
    # (nothing ever redirects a write to other_name's node), the
    # redirect-free control condition's answer source.
    steps.append((other_name_vec, attr_rel, other_attr_vec, pred_is, z, 0))

    # S4: the pronoun-SUBJECT overwrite statement ("she is tall .") -- the
    # collapse step this whole milestone is about. Placeholder (garbage)
    # address; the candidate set below carries the real redirect data.
    pronoun_step_idx = len(steps)
    steps.append((pronoun_vec, attr_rel, pronoun_attr_vec, pred_is, z, 0))

    # Q: "what is {target_name} like ?" -- queries target_name's OWN
    # rel:ATTR slot directly (a NAMED entity, never the pronoun's own
    # placeholder address). WriteBackCurriculumGenerator (v2) samples
    # target_name UNIFORMLY over both entities: when it's the referent,
    # answerable ONLY if S4's write actually redirected+overwrote there;
    # when it's the other entity, answerable from S3 alone (no redirect
    # needed) -- the control condition that keeps the question template
    # itself from being a fixed "echo the pronoun's value" shortcut.
    steps.append((target_vec, attr_rel, z, q_pred, z, 1))

    cand_sets: Dict[int, "membrane.EntityCandidateSet"] = {}
    forced_map: Dict[int, int] = {}
    if not cheat:
        registry = [name_a, name_b]   # matches ep.meta["registry_order"] (WriteBackCurriculumGenerator)
        cand = membrane.pronoun_entity_candidate_set(
            pronoun, registry, gold_antecedent=gold_antecedent,
            provenance={"sentence_index": pronoun_step_idx, "kind": "writeback"},
            addr_redirect=True)
        cand_sets[pronoun_step_idx] = cand
        if force is not None:
            true_idx = registry.index(antecedent_name)
            wrong_idx = 1 - true_idx     # exactly two candidates in this registry
            forced_map[pronoun_step_idx] = true_idx if force == "gold" else wrong_idx
    return steps, cand_sets, forced_map


# M57c (instance atoms + definite-description referring expressions,
# dev/MIND_INTERFACE.md's v2 addendum, CLAUDE.md's M57 memory-schema
# decision): an InstanceCurriculumGenerator episode (curriculum2.py --
# gated on ``ep.meta["kind"] == "instance"``, a key no other generator
# sets). Parser-free, sibling to :func:`_writeback_steps` for the SAME
# reason: an attribute assertion's value is a genuine content word (parser
# reliability was never the point), and the world here (3 instances, two
# sharing a name) has no existing sentence-role vocabulary to parse anyway.
#
# The one thing this function does that NOTHING before it did: entity
# atoms are minted by a FRESH :class:`nsm_ct.instances.InstanceRegistry`,
# seeded from ``ep.meta["instance_seed"]`` (deterministic given that seed,
# independent of dataset order/batching -- the whole point of M57a's
# registry), NOT grounded from the codec's ``var:<name>`` hash. Two
# instances sharing a name (the two-Marys premise) therefore mint to
# genuinely DIFFERENT atoms (``inst:mary#1`` != ``inst:mary#2``) -- the
# standing defect dev/MIND_INTERFACE.md's v2 addendum names explicitly.
#
# World (fixed roles "a"/"b"/"c", matching InstanceCurriculumGenerator's
# own bookkeeping so curriculum meta never needs to touch instance ids or
# atoms -- torch/codec stay out of curriculum2.py by that module's own
# constraint): "a" and "b" share ``ep.meta["shared_name"]`` (two Marys),
# "c" carries ``ep.meta["distinct_name"]``. Registration writes THREE
# ordinary attribute facts per instance -- kind, gender, a named-place --
# each addressed DIRECTLY via that instance's own minted atom (no
# ambiguity in the WRITE itself: the curriculum always knows exactly which
# instance a fact is about, even when two instances share a surface name;
# mirrors :func:`_writeback_steps`'s S0-S3 "own fact, own atom" pattern).
# Then every instance gets ONE baseline TRAIT statement (also direct-
# addressed, establishing the "stale" value the OVERWRITE below will
# clobber for exactly one of them -- the M57b overwrite shape, scaled from
# 2 entities to 3, same reactor-owned overwrite gate, no special-casing).
#
# The OVERWRITE step (mirrors _writeback_steps' S4 exactly): a placeholder
# (garbage) address; the real redirect data rides in the returned
# EntityCandidateSet, ``addr_redirect=True``, ``evidence_relation`` set to
# whichever attribute the referring device's evidence comes from:
#   - "definite_description" ("the {kind} is {attr} .") -- evidence
#     attr:kind, candidates = all three instances (kind is unique per
#     instance by construction, so this is always cleanly resolvable).
#   - "pronoun" ("she/he is {attr} .") -- evidence attr:gender, candidates
#     = all three instances (genuinely ambiguous when the referent's
#     gender ties with another instance's -- the harder case, left
#     unfiltered on purpose: "perception never guesses beyond what it can
#     support" would filter by gender, but the whole point of the evidence-
#     relation mechanism is that the RESOLVER, not perception, does the
#     matching against a LIVE memory read).
#   - "ambiguous_name" ("{shared_name} is {attr} .") -- evidence attr:kind,
#     candidates RESTRICTED to the two name-matched instances (a/b) only --
#     c's distinct name is never ambiguous, so it is never a candidate
#     here. The gold referent is whichever of a/b had the temporally
#     CLOSEST preceding baseline statement (the generator places that
#     instance's baseline last among the three) -- the SAME discourse-
#     recency convention this codebase already uses for pronoun antecedents
#     (WriteBackCurriculumGenerator's antecedent_first/antecedent_recency);
#     recency legitimately DETERMINES the referent (that is how ambiguous
#     repeated names actually get disambiguated in real discourse), it does
#     not make the FINAL ANSWER surface-guessable -- the answer still
#     requires correctly redirecting the write AND reading the right node's
#     attribute afterward (CLAUDE.md's design law is about the
#     answer-bearing statement's surface form, not about what legitimately
#     determines a referent).
#
# The QUESTION step: ``ep.meta["question_mode"] == "target"`` asks "what is
# {referring expression} like ?" about ONE of the three instances, phrased
# with an UNAMBIGUOUS referring expression (c's own unique name -- direct
# addressing, no candidates needed; a/b's definite description -- SAME
# evidence-relation/addr_redirect machinery as the overwrite step, just at
# a READ instead of a WRITE). This is safe to do at a question step with
# ZERO forward()/em.write changes: a question step's ``stmt = real*(1-isq)``
# is always 0, so ``gate`` is force-zeroed regardless of the (possibly
# redirected) entity -- the write portion is an unconditional no-op. The
# response-generation head does still read ``mem_read`` computed from the
# PRE-collapse placeholder address (forward()'s existing, unmodified
# ordering -- see ClauseReactor.forward's own comment, "ALWAYS from the
# pre-collapse address"), so the correct answer must flow through the
# controller's GRU STATE (which DOES incorporate the POST-collapse,
# resolved entity atom, since the GRU update happens after _collapse at
# the SAME step) rather than a literal re-read of the resolved node. This
# is a genuine, disclosed capability ceiling -- not a new mechanism, the
# SAME "recall via hidden state" capability this codebase already reports
# on via its cheat baselines (M57b's cheat arm: 0.6-0.9 depending on
# subset, RESEARCH_NOTES M57b) -- see this milestone's report for the seam
# this leaves for M57d (a genuine post-collapse mem_read recompute would
# need to be gated so it does not perturb M57b's already-proven writeback
# arithmetic, which reuses this same addr_redirect flag at WRITE steps).
#
# ``ep.meta["question_mode"] == "inverse"`` ("who is {trait} ?") is a
# DIFFERENT, deliberately simpler mechanism: no candidate/evidence-relation
# machinery at all -- a pure generation-plus-contrastive-answer step (the
# SAME mechanism every other MC curriculum in this codebase already uses),
# entity = a fixed "who" marker, relation = attr:trait, value = the queried
# trait word (so the controller sees WHAT is being asked directly, the
# same way a reasoning-level query step already carries its query
# (entity, relation) as the step's own address). The options are identity
# vectors, not attribute words -- see :func:`_instance_option_vec`.
#
# M57c.2 (RESEARCH_NOTES "M57c battery #1" -- inverse_query measured BELOW
# chance because no entity-axis read existed): the READ side now matches
# that "identity vectors" framing literally. :func:`build_clause_batch`
# grounds an inverse-query episode's options as the per-episode INSTANCE
# ATOMS themselves (``atom_lookup[iid] for iid in ep.meta["registry_order"]``
# -- the exact atoms :func:`_instance_steps` minted, in a/b/c order, which
# is also ``ep.options``'s order by construction), NOT
# :func:`_instance_option_vec`'s composite ``ent_vec(name) +
# content_vec(kind)`` sum below -- so the contrastive answer compares
# :func:`nsm_ct.entity_memory.query_entity`'s entity-axis readout against
# REAL candidate atoms, the same space it was unbound from, instead of a
# separately-grounded meaning-vector that only correlates with the right
# instance. :func:`_instance_option_vec` itself is UNCHANGED (still
# exercised by its own unit test) -- it is simply no longer the grounding
# path :func:`build_clause_batch` uses for inverse-query options.
def _instance_option_vec(opt: str, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                          meaning_source: MeaningSource) -> np.ndarray:
    """Ground one inverse-query MC option. Convention (curriculum2.
    InstanceCurriculumGenerator's own docstring documents this too):
    ``"{name}"`` for an unambiguous (unique-named) instance, ``"{name} the
    {kind}"`` for a name-matched-pair instance -- the kind suffix is what
    keeps two same-named options DISTINCT meaning-vectors (the same
    ``ent_vec(name)`` alone would collide for both Marys, exactly the
    identity-fusion defect this whole milestone fixes, here showing up on
    the ANSWER side instead of the memory side). Grounded as
    ``ent_vec(name) + content_vec(kind)`` -- an ordinary vector-space sum,
    the same "compose meanings by addition" convention TPR superposition
    already relies on everywhere else in this codebase."""
    if " the " in opt:
        name_part, kind_part = opt.split(" the ", 1)
        return (_ent_vec(name_part, resolver, codec, cache, meaning_source)
                + _content_vec(kind_part, resolver, codec, cache, meaning_source))
    return _ent_vec(opt, resolver, codec, cache, meaning_source)


def _instance_steps(ep, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                     meaning_source: MeaningSource, cheat: bool = False,
                     no_gold: bool = False, force: Optional[str] = None):
    """Grounded stream + candidate-set map(s) + instance-atom lookup for one
    :class:`~nsm_ct.curriculum2.InstanceCurriculumGenerator` episode. See
    the module comment immediately above for the full design. Returns
    ``(steps, cand_sets, forced_map, atom_lookup, inverse_step_idx)`` --
    ``atom_lookup`` (a
    NEW fourth element, absent from every earlier ``_*_steps`` helper's
    return shape) is ``{instance_id: atom_ndarray}`` for this episode's
    freshly-minted registry, consumed by :func:`build_clause_batch`'s
    candidate-grounding loop so a candidate KEY that is an instance id
    (``"inst:mary#1"``, never in ``_NAMESET``, never a plain content word)
    grounds to its OWN minted atom instead of falling through to
    :func:`_ent_vec`'s name/content-word branches, which have no idea what
    an instance id is.

    M57c.2 (RESEARCH_NOTES "M57c battery #1"): a FIFTH element,
    ``inverse_step_idx`` -- the step index of the "who is {trait} ?"
    question step for an inverse-query episode, ``None`` for every
    "target"-mode episode. :func:`build_clause_batch` uses it to populate
    ``ClauseBatch.inverse_mask`` (that step's ENTITY-axis read: see
    :func:`nsm_ct.entity_memory.query_entity` and
    :meth:`ClauseReactor.forward`'s inverse-read paragraph) -- a fixed
    "who" marker atom is not a memory address, so there is no candidate
    set to redirect here, just a flag on the one step whose read must use
    the entity-axis unbind instead of the ordinary value read.
    """
    d = codec.dim
    z = np.zeros(d, np.float32)
    pred_is = codec.filler_vec("pred:is")
    q_pred = codec.filler_vec("pred:?")
    kind_rel = codec.filler_vec("attr:kind")
    gender_rel = codec.filler_vec("attr:gender")
    place_rel = codec.filler_vec("attr:place")
    trait_rel = codec.filler_vec("attr:trait")

    registry = InstanceRegistry(dim=d, seed=ep.meta["instance_seed"])
    shared_name = ep.meta["shared_name"]
    distinct_name = ep.meta["distinct_name"]
    id_a, atom_a = registry.mint(shared_name)
    id_b, atom_b = registry.mint(shared_name)
    id_c, atom_c = registry.mint(distinct_name)
    ids = {"a": id_a, "b": id_b, "c": id_c}
    atoms = {"a": atom_a.numpy(), "b": atom_b.numpy(), "c": atom_c.numpy()}
    kinds = {r: ep.meta[f"kind_{r}"] for r in "abc"}
    genders = {r: ep.meta[f"gender_{r}"] for r in "abc"}
    places = {r: ep.meta[f"place_{r}"] for r in "abc"}
    baselines = {r: ep.meta[f"baseline_{r}"] for r in "abc"}
    overwrite_attr = ep.meta["overwrite_attr"]
    device = ep.meta["referring_device"]
    referent = ep.meta["referent_role"]

    steps = []
    # M60 (op-library integration -- RECENCY): row-local mention log, one
    # entry per step index at which a candidate's OWN atom was the step's
    # ENTITY (subject position) -- the "ambiguous_name" device's ONLY
    # disambiguating channel (its evidence_relation/evidence_target compare
    # against no useful signal at all, see EntityCandidateSet.evidence_target's
    # M57c.3 docstring paragraph and tests/test_evidence_interaction.py's
    # ``_train_short`` docstring, which excludes this device from the
    # interaction-feature test for exactly this reason). Built here (not
    # reconstructed downstream) because this function already knows, at
    # every ``steps.append`` call, exactly which candidate id that step's
    # entity is -- see :func:`_recency_fields` below and
    # membrane.EntityCandidateSet.mention_steps's own docstring.
    mention_log: Dict[str, List[int]] = {ids[r]: [] for r in ("a", "b", "c")}

    def _recency_fields(roles):
        ms = np.array([(mention_log[ids[r]][-1] if mention_log[ids[r]] else -1) for r in roles],
                       dtype=np.float32)
        mc = np.array([len(mention_log[ids[r]]) for r in roles], dtype=np.float32)
        return ms, mc

    # Registration: kind, gender, named-place -- direct addressing, own
    # atom (see module comment above).
    for r in ("a", "b", "c"):
        kind_vec = codec.filler_vec("kind:" + kinds[r])
        gender_vec = codec.filler_vec("gender:" + genders[r])
        place_vec = _content_vec(places[r], resolver, codec, cache, meaning_source)
        mention_log[ids[r]] += [len(steps), len(steps) + 1, len(steps) + 2]
        steps.append((atoms[r], kind_rel, kind_vec, pred_is, z, 0))
        steps.append((atoms[r], gender_rel, gender_vec, pred_is, z, 0))
        steps.append((atoms[r], place_rel, place_vec, pred_is, z, 0))

    # Baseline trait statements -- one per instance, direct addressing.
    # ``ambiguous_name`` places the referent's own baseline LAST (the
    # recency convention -- see module comment above); every other device
    # uses the fixed a/b/c order (irrelevant to those devices' resolution,
    # which reads attr:kind/attr:gender, not recency).
    baseline_order = ["a", "b", "c"]
    if device == "ambiguous_name":
        baseline_order = [r for r in baseline_order if r != referent] + [referent]
    for r in baseline_order:
        trait_vec = _content_vec(baselines[r], resolver, codec, cache, meaning_source)
        mention_log[ids[r]].append(len(steps))
        steps.append((atoms[r], trait_rel, trait_vec, pred_is, z, 0))

    cand_sets: Dict[int, "membrane.EntityCandidateSet"] = {}
    forced_map: Dict[int, int] = {}
    atom_lookup: Dict[str, np.ndarray] = {ids[r]: atoms[r] for r in ("a", "b", "c")}

    # Overwrite step.
    if device == "pronoun":
        pronoun = "she" if genders[referent] == "F" else "he"
        placeholder = _ent_vec(pronoun, resolver, codec, cache, meaning_source)
        evidence_rel_name = "gender"
        evidence_target_key = "gender:" + genders[referent]
        cand_roles = ["a", "b", "c"]
        mention_word = pronoun
    elif device == "definite_description":
        placeholder = _content_vec(kinds[referent], resolver, codec, cache, meaning_source)
        evidence_rel_name = "kind"
        evidence_target_key = "kind:" + kinds[referent]
        cand_roles = ["a", "b", "c"]
        mention_word = kinds[referent]
    else:  # "ambiguous_name"
        placeholder = _ent_vec(shared_name, resolver, codec, cache, meaning_source)
        evidence_rel_name = "kind"
        evidence_target_key = "name:" + shared_name
        cand_roles = ["a", "b"]
        mention_word = shared_name

    overwrite_vec = _content_vec(overwrite_attr, resolver, codec, cache, meaning_source)
    overwrite_step_idx = len(steps)
    steps.append((placeholder, trait_rel, overwrite_vec, pred_is, z, 0))

    if not cheat:
        candidates = [membrane.Candidate(key=ids[r], prior=1.0 / len(cand_roles)) for r in cand_roles]
        gold_index = None if no_gold else cand_roles.index(referent)
        rec_steps, rec_counts = _recency_fields(cand_roles)
        cs = membrane.EntityCandidateSet(
            candidates=candidates,
            provenance={"sentence_index": overwrite_step_idx, "kind": "instance", "device": device},
            surface=mention_word,
            feature=membrane.mention_feature_vector(mention_word),
            gold_index=gold_index,
            addr_redirect=True,
            evidence_relation=evidence_rel_name,
            evidence_target=evidence_target_key,
            mention_steps=rec_steps,
            mention_counts=rec_counts,
        )
        cand_sets[overwrite_step_idx] = cs
        if force is not None:
            true_idx = cand_roles.index(referent)
            wrong_idx = (true_idx + 1) % len(cand_roles)
            forced_map[overwrite_step_idx] = true_idx if force == "gold" else wrong_idx

    # Question step.
    inverse_step_idx: Optional[int] = None
    if ep.meta.get("question_mode") == "inverse":
        who_vec = _content_vec("who", resolver, codec, cache, meaning_source)
        query_vec = _content_vec(ep.meta["query_trait"], resolver, codec, cache, meaning_source)
        inverse_step_idx = len(steps)
        steps.append((who_vec, trait_rel, query_vec, q_pred, z, 1))
    else:
        target = ep.meta["target_role"]
        if target == "c":
            steps.append((atoms["c"], trait_rel, z, q_pred, z, 1))
        else:
            q_placeholder = _content_vec(kinds[target], resolver, codec, cache, meaning_source)
            q_step_idx = len(steps)
            steps.append((q_placeholder, trait_rel, z, q_pred, z, 1))
            if not cheat:
                q_candidates = [membrane.Candidate(key=ids[r], prior=1.0 / 3) for r in ("a", "b", "c")]
                q_gold_index = None if no_gold else ("a", "b", "c").index(target)
                q_rec_steps, q_rec_counts = _recency_fields(("a", "b", "c"))
                q_cs = membrane.EntityCandidateSet(
                    candidates=q_candidates,
                    provenance={"sentence_index": q_step_idx, "kind": "instance",
                                "device": "definite_description", "question": True},
                    surface=kinds[target],
                    feature=membrane.mention_feature_vector(kinds[target]),
                    gold_index=q_gold_index,
                    addr_redirect=True,
                    evidence_relation="kind",
                    evidence_target="kind:" + kinds[target],
                    mention_steps=q_rec_steps,
                    mention_counts=q_rec_counts,
                )
                cand_sets[q_step_idx] = q_cs
                if force is not None:
                    true_idx = ("a", "b", "c").index(target)
                    wrong_idx = (true_idx + 1) % 3
                    forced_map[q_step_idx] = true_idx if force == "gold" else wrong_idx

    return steps, cand_sets, forced_map, atom_lookup, inverse_step_idx


# RICH-EPISODE curriculum (the "stop requiring minimal episodes" priority,
# CLAUDE.md's 2026-08-30 reprioritization): a direct generalization of
# _instance_steps above from the fixed 3-role/1-overwrite world to N
# entities (curriculum-sampled 3-8) / K referring statements
# (curriculum-sampled 1-4) -- see nsm_ct.curriculum2.RichEpisodeGenerator's
# extensive module comment for the full design and its honesty machinery.
# A NEW, sibling function -- _instance_steps itself is UNCHANGED, still
# exclusively serving ep.meta["kind"] == "instance" -- matching this
# codebase's own established convention (M54/M55a/M57b/M57c each added
# their own dedicated _xxx_steps rather than editing an earlier, already-
# proven one). Every mechanism (kind/gender/named-place registration,
# DISTINCT-relation attribute facts, an EntityCandidateSet per referring
# statement with addr_redirect=True + evidence_relation, the entity-axis
# inverse read) is IDENTICAL to _instance_steps -- just looped over N
# roles / K statements read from ep.meta instead of the fixed a/b/c
# unrolling. Reuses EVERY existing ClauseBatch field (cand_entity's Cmax
# dimension already pads a variable per-(row, step) candidate-set size
# generically) -- no new tensor group needed.
def _rich_steps(ep, resolver, codec: TPRCodec, cache: Dict[str, np.ndarray],
                 meaning_source: MeaningSource, cheat: bool = False,
                 no_gold: bool = False, force: Optional[str] = None):
    """Grounded stream + candidate-set map + instance-atom lookup for one
    :class:`~nsm_ct.curriculum2.RichEpisodeGenerator` episode. Returns the
    SAME five-element shape as :func:`_instance_steps`: ``(steps,
    cand_sets, forced_map, atom_lookup, inverse_step_idx)``.

    Exactly one question step per episode (the reactor's own per-episode
    contract, preserved by :class:`~nsm_ct.curriculum2.RichEpisodeGenerator`
    regardless of ``n_entities``/K), so ``inverse_step_idx`` is either
    ``None`` (a "target"-mode episode) or a single step index --
    ``ClauseBatch.inverse_mask``'s "at most one inverse step per episode"
    seam holds by construction, never by convention alone.
    """
    d = codec.dim
    z = np.zeros(d, np.float32)
    pred_is = codec.filler_vec("pred:is")
    q_pred = codec.filler_vec("pred:?")
    kind_rel = codec.filler_vec("attr:kind")
    gender_rel = codec.filler_vec("attr:gender")
    place_rel = codec.filler_vec("attr:place")

    n = ep.meta["n_entities"]
    names = ep.meta["names"]
    kinds = ep.meta["kinds"]
    genders = ep.meta["genders"]
    places = ep.meta["places"]
    held_relations = ep.meta["held_relations"]
    initial_values = ep.meta["initial_values"]
    entity_order = ep.meta["entity_order"]
    name_groups = ep.meta["name_groups"]
    # M57e (morphology signals): True only for an episode
    # nsm_ct.curriculum2.RichEpisodeGenerator built with plural_frac > 0
    # THAT ALSO SAMPLED a group this episode -- absent/False for every
    # pre-M57e episode (and every plural_frac == 0.0 episode), so
    # everything gated on it below is a strict no-op then, byte-identical
    # to before this milestone.
    has_group = bool(ep.meta.get("has_group"))

    registry = InstanceRegistry(dim=d, seed=ep.meta["instance_seed"])
    ids: Dict[int, str] = {}
    atoms: Dict[int, np.ndarray] = {}
    for i in entity_order:
        iid, atom = registry.mint(names[i])
        ids[i] = iid
        atoms[i] = atom.numpy()

    # PLURAL group referent (see nsm_ct.curriculum2.RichEpisodeGenerator's
    # "optional PLURAL group" comment for the full design): minted AFTER
    # every individual, always the registry's first (only) "group" mint --
    # curriculum2's own group_instance_id prediction ("inst:group#1")
    # depends on this exact ordering.
    group_id: Optional[str] = None
    group_atom: Optional[np.ndarray] = None
    if has_group:
        group_id, group_atom_t = registry.mint("group")
        group_atom = group_atom_t.numpy()

    def _id_for(role) -> str:
        return group_id if role == "group" else ids[role]

    # M60 (op-library integration -- RECENCY): SAME row-local mention log
    # _instance_steps builds (see that function's own docstring paragraph),
    # generalized over N roles + the optional group id -- one entry per
    # step index at which a candidate's own atom was the step's ENTITY.
    mention_log: Dict[str, List[int]] = {ids[i]: [] for i in range(n)}
    if has_group:
        mention_log[group_id] = []

    def _recency_fields(roles):
        ms = np.array([(mention_log[_id_for(r)][-1] if mention_log[_id_for(r)] else -1)
                        for r in roles], dtype=np.float32)
        mc = np.array([len(mention_log[_id_for(r)]) for r in roles], dtype=np.float32)
        return ms, mc

    steps = []
    # Registration: kind, gender, named-place, then every held attribute
    # relation's baseline value -- direct addressing, own atom (mirrors
    # _instance_steps' registration block exactly, generalized to N roles
    # and a variable per-entity relation set). When this episode has a
    # PLURAL group, every individual ALSO gets an attr:number=sg fact here
    # (only then -- see module comment) so the group's own attr:number=pl
    # (written below) is genuinely discriminating: candidates_for/
    # evidence_interaction can only tell singular and plural instances
    # apart if EVERY candidate carries the fact, not just the plural one.
    number_rel = codec.filler_vec("attr:number") if has_group else None
    for i in entity_order:
        kind_vec = codec.filler_vec("kind:" + kinds[i])
        gender_vec = codec.filler_vec("gender:" + genders[i])
        place_vec = _content_vec(places[i], resolver, codec, cache, meaning_source)
        mention_log[ids[i]] += [len(steps), len(steps) + 1, len(steps) + 2]
        steps.append((atoms[i], kind_rel, kind_vec, pred_is, z, 0))
        steps.append((atoms[i], gender_rel, gender_vec, pred_is, z, 0))
        steps.append((atoms[i], place_rel, place_vec, pred_is, z, 0))
        for rel in held_relations[i]:
            val_vec = _content_vec(initial_values[i][rel], resolver, codec, cache, meaning_source)
            mention_log[ids[i]].append(len(steps))
            steps.append((atoms[i], codec.filler_vec("attr:" + rel), val_vec, pred_is, z, 0))
        if has_group:
            mention_log[ids[i]].append(len(steps))
            steps.append((atoms[i], number_rel, codec.filler_vec("number:sg"), pred_is, z, 0))

    atom_lookup: Dict[str, np.ndarray] = {ids[i]: atoms[i] for i in range(n)}
    if has_group:
        atom_lookup[group_id] = group_atom
    cand_sets: Dict[int, "membrane.EntityCandidateSet"] = {}
    forced_map: Dict[int, int] = {}

    # K referring/overwrite steps -- each mirrors _instance_steps' single
    # overwrite step exactly, just looped.
    for stmt in ep.meta["referring_statements"]:
        referent = stmt["referent"]
        rel = stmt["relation"]
        device = stmt["device"]
        attr_rel_vec = codec.filler_vec("attr:" + rel)
        overwrite_vec = _content_vec(stmt["new_value"], resolver, codec, cache, meaning_source)

        if device == "pronoun":
            pronoun = "she" if genders[referent] == "F" else "he"
            placeholder = _ent_vec(pronoun, resolver, codec, cache, meaning_source)
            evidence_rel_name = "gender"
            evidence_target_key = "gender:" + genders[referent]
            cand_roles = list(range(n))
        elif device == "definite_description":
            placeholder = _content_vec(kinds[referent], resolver, codec, cache, meaning_source)
            evidence_rel_name = "kind"
            evidence_target_key = "kind:" + kinds[referent]
            cand_roles = list(range(n))
        else:  # "ambiguous_name" -- candidates restricted to the name-matched pair
            placeholder = _ent_vec(names[referent], resolver, codec, cache, meaning_source)
            evidence_rel_name = "kind"
            evidence_target_key = "name:" + names[referent]
            cand_roles = list(name_groups[names[referent]])

        step_idx = len(steps)
        steps.append((placeholder, attr_rel_vec, overwrite_vec, pred_is, z, 0))

        if not cheat:
            mention_word = stmt["mention_word"]
            candidates = [membrane.Candidate(key=ids[r], prior=1.0 / len(cand_roles)) for r in cand_roles]
            gold_index = None if no_gold else cand_roles.index(referent)
            rec_steps, rec_counts = _recency_fields(cand_roles)
            cs = membrane.EntityCandidateSet(
                candidates=candidates,
                provenance={"sentence_index": step_idx, "kind": "rich", "device": device},
                surface=mention_word,
                feature=membrane.mention_feature_vector(mention_word),
                gold_index=gold_index,
                addr_redirect=True,
                evidence_relation=evidence_rel_name,
                evidence_target=evidence_target_key,
                mention_steps=rec_steps,
                mention_counts=rec_counts,
            )
            cand_sets[step_idx] = cs
            if force is not None:
                true_idx = cand_roles.index(referent)
                wrong_idx = (true_idx + 1) % len(cand_roles)
                forced_map[step_idx] = true_idx if force == "gold" else wrong_idx

    # PLURAL group facts + overwrite statement (see nsm_ct.curriculum2.
    # RichEpisodeGenerator's "optional PLURAL group" comment). The
    # coordination sentence's group mint carries attr:number=pl plus one
    # attr:member fact per member -- entity_memory write/overwrite is keyed
    # by (entity, relation), so two facts under the literal same relation
    # would clobber each other (the SECOND write's ``_last[key]`` subtract-
    # then-add erases the first); "member_0"/"member_1" (distinct relation
    # names) is this milestone's concrete answer to that, a deviation from
    # the design note's singular "attr:member" phrasing recorded in the
    # report, not silently reconciled. The plural-pronoun ("they are
    # {value} .") overwrite step then mirrors every K referring statement's
    # own evidence-relation mechanism exactly, with "number" standing in
    # for "kind"/"gender" and candidates = every individual PLUS the group.
    if has_group:
        m0, m1 = ep.meta["group_members"]
        mention_log[group_id] += [len(steps), len(steps) + 1, len(steps) + 2]
        steps.append((group_atom, number_rel, codec.filler_vec("number:pl"), pred_is, z, 0))
        steps.append((group_atom, codec.filler_vec("attr:member_0"), atoms[m0], pred_is, z, 0))
        steps.append((group_atom, codec.filler_vec("attr:member_1"), atoms[m1], pred_is, z, 0))

        group_relation = ep.meta["group_relation"]
        group_value = ep.meta["group_value"]
        group_attr_rel_vec = codec.filler_vec("attr:" + group_relation)
        group_overwrite_vec = _content_vec(group_value, resolver, codec, cache, meaning_source)
        group_placeholder = _ent_vec("they", resolver, codec, cache, meaning_source)
        group_cand_roles: List[object] = list(range(n)) + ["group"]
        group_step_idx = len(steps)
        steps.append((group_placeholder, group_attr_rel_vec, group_overwrite_vec, pred_is, z, 0))
        if not cheat:
            group_candidates = [membrane.Candidate(key=_id_for(r), prior=1.0 / len(group_cand_roles))
                                 for r in group_cand_roles]
            group_gold_index = None if no_gold else group_cand_roles.index("group")
            group_rec_steps, group_rec_counts = _recency_fields(group_cand_roles)
            group_cs = membrane.EntityCandidateSet(
                candidates=group_candidates,
                provenance={"sentence_index": group_step_idx, "kind": "rich", "device": "plural_pronoun"},
                surface="they",
                feature=membrane.mention_feature_vector("they"),
                gold_index=group_gold_index,
                addr_redirect=True,
                evidence_relation="number",
                evidence_target="number:pl",
                mention_steps=group_rec_steps,
                mention_counts=group_rec_counts,
            )
            cand_sets[group_step_idx] = group_cs
            if force is not None:
                true_idx = group_cand_roles.index("group")
                wrong_idx = (true_idx + 1) % len(group_cand_roles)
                forced_map[group_step_idx] = true_idx if force == "gold" else wrong_idx

    # Question step.
    inverse_step_idx: Optional[int] = None
    if ep.meta.get("question_mode") == "inverse":
        who_vec = _content_vec("who", resolver, codec, cache, meaning_source)
        query_vec = _content_vec(ep.meta["query_value"], resolver, codec, cache, meaning_source)
        query_rel_vec = codec.filler_vec("attr:" + ep.meta["query_relation"])
        inverse_step_idx = len(steps)
        steps.append((who_vec, query_rel_vec, query_vec, q_pred, z, 1))
    elif ep.meta.get("target_is_group"):
        group_relation = ep.meta["group_relation"]
        attr_rel_vec = codec.filler_vec("attr:" + group_relation)
        q_placeholder = _ent_vec("they", resolver, codec, cache, meaning_source)
        q_step_idx = len(steps)
        steps.append((q_placeholder, attr_rel_vec, z, q_pred, z, 1))
        if not cheat:
            q_cand_roles: List[object] = list(range(n)) + ["group"]
            q_candidates = [membrane.Candidate(key=_id_for(r), prior=1.0 / len(q_cand_roles))
                             for r in q_cand_roles]
            q_gold_index = None if no_gold else q_cand_roles.index("group")
            q_rec_steps, q_rec_counts = _recency_fields(q_cand_roles)
            q_cs = membrane.EntityCandidateSet(
                candidates=q_candidates,
                provenance={"sentence_index": q_step_idx, "kind": "rich",
                            "device": "plural_pronoun", "question": True},
                surface="they",
                feature=membrane.mention_feature_vector("they"),
                gold_index=q_gold_index,
                addr_redirect=True,
                evidence_relation="number",
                evidence_target="number:pl",
                mention_steps=q_rec_steps,
                mention_counts=q_rec_counts,
            )
            cand_sets[q_step_idx] = q_cs
            if force is not None:
                true_idx = q_cand_roles.index("group")
                wrong_idx = (true_idx + 1) % len(q_cand_roles)
                forced_map[q_step_idx] = true_idx if force == "gold" else wrong_idx
    else:
        target = ep.meta["target_entity"]
        target_rel = ep.meta["target_relation"]
        attr_rel_vec = codec.filler_vec("attr:" + target_rel)
        if names[target] not in name_groups:
            steps.append((atoms[target], attr_rel_vec, z, q_pred, z, 1))
        else:
            q_placeholder = _content_vec(kinds[target], resolver, codec, cache, meaning_source)
            q_step_idx = len(steps)
            steps.append((q_placeholder, attr_rel_vec, z, q_pred, z, 1))
            if not cheat:
                q_candidates = [membrane.Candidate(key=ids[r], prior=1.0 / n) for r in range(n)]
                q_gold_index = None if no_gold else target
                q_rec_steps, q_rec_counts = _recency_fields(range(n))
                q_cs = membrane.EntityCandidateSet(
                    candidates=q_candidates,
                    provenance={"sentence_index": q_step_idx, "kind": "rich",
                                "device": "definite_description", "question": True},
                    surface=kinds[target],
                    feature=membrane.mention_feature_vector(kinds[target]),
                    gold_index=q_gold_index,
                    addr_redirect=True,
                    evidence_relation="kind",
                    evidence_target="kind:" + kinds[target],
                    mention_steps=q_rec_steps,
                    mention_counts=q_rec_counts,
                )
                cand_sets[q_step_idx] = q_cs
                if force is not None:
                    wrong_idx = (target + 1) % n
                    forced_map[q_step_idx] = target if force == "gold" else wrong_idx

    return steps, cand_sets, forced_map, atom_lookup, inverse_step_idx


# ---------------------------------------------------------------------------
# M59b (CROSS-PASSAGE document curriculum for episodic LTM, M59a): a NEW,
# sibling function to _instance_steps/_rich_steps above -- see
# nsm_ct.curriculum2.DocumentGenerator's extensive module comment for the
# full design and its honesty machinery, and src/nsm_ct/ltm.py's module
# docstring ("Interface contract for the curriculum agent") for the binding
# contract this function's caller (build_clause_batch) must satisfy.
#
# UNLIKE _instance_steps/_rich_steps, this function does NOT construct its
# own InstanceRegistry -- it takes one, CALLER-SUPPLIED, threaded through
# EVERY passage of one document (the ltm.py contract's "one registry per
# document, constructed once"). The caller MUST build a document's passages'
# batches IN PASSAGE ORDER (passage 0 before passage 1, etc.) so this
# passage's ``registry.lookup``/id-prediction calls resolve atoms an EARLIER
# passage's build_clause_batch call already minted -- scripts/train_ltm.py's
# document-processing loop is the reference caller.
# ---------------------------------------------------------------------------
def _document_steps(ep, registry: InstanceRegistry, resolver, codec: TPRCodec,
                     cache: Dict[str, np.ndarray], meaning_source: MeaningSource,
                     cheat: bool = False, no_gold: bool = False, force: Optional[str] = None):
    """Grounded stream + candidate-set map for ONE PASSAGE of a
    :class:`~nsm_ct.curriculum2.DocumentGenerator` document. Returns the
    SAME five-element shape as :func:`_instance_steps`/:func:`_rich_steps`:
    ``(steps, cand_sets, forced_map, atom_lookup, inverse_step_idx)`` --
    ``inverse_step_idx`` is always ``None`` (this curriculum has no
    inverse-query mode).

    Three passage kinds, dispatched on ``ep.meta["passage_index"]``:
      - **passage 0** (registration): every passage-0 entity is minted and
        registered (kind/gender/place + its held attribute-relation facts),
        direct addressing only, no candidate sets -- mirrors
        :func:`_rich_steps`' own registration block.
      - **a filler passage** (``0 < passage_index < n_passages - 1``, only
        when ``n_passages == 3``): 1-2 unrelated distractor entities,
        registered the same way, no attribute facts, no candidate sets.
      - **the final passage** (``passage_index == n_passages - 1``): the
        mention (ONE :class:`~nsm_ct.membrane.EntityCandidateSet`,
        candidates = ``[referent's own LTM instance id, a freshly-minted NEW
        instance id]``, ``from_ltm=[1, 0]``, ``addr_redirect=True``) followed
        by ONE question, addressed DIRECTLY (never a second candidate set --
        see the module comment on :class:`~nsm_ct.curriculum2.
        DocumentGenerator` for why "the {kind}" is always globally
        unambiguous by construction).
    """
    d = codec.dim
    z = np.zeros(d, np.float32)
    pred_is = codec.filler_vec("pred:is")
    q_pred = codec.filler_vec("pred:?")
    kind_rel = codec.filler_vec("attr:kind")
    gender_rel = codec.filler_vec("attr:gender")
    place_rel = codec.filler_vec("attr:place")

    m = ep.meta
    passage_index = m["passage_index"]
    n_passages = m["n_passages"]
    steps = []
    cand_sets: Dict[int, "membrane.EntityCandidateSet"] = {}
    forced_map: Dict[int, int] = {}
    atom_lookup: Dict[str, np.ndarray] = {}

    def _register(name: str, kind: str, gender: str, place: str,
                   relations: Tuple[str, ...] = (), values: Tuple[str, ...] = ()) -> None:
        iid, atom = registry.mint(name)
        atom_np = atom.numpy()
        atom_lookup[iid] = atom_np
        place_vec = _content_vec(place, resolver, codec, cache, meaning_source)
        steps.append((atom_np, kind_rel, codec.filler_vec("kind:" + kind), pred_is, z, 0))
        steps.append((atom_np, gender_rel, codec.filler_vec("gender:" + gender), pred_is, z, 0))
        steps.append((atom_np, place_rel, place_vec, pred_is, z, 0))
        for rel, val in zip(relations, values):
            val_vec = _content_vec(val, resolver, codec, cache, meaning_source)
            steps.append((atom_np, codec.filler_vec("attr:" + rel), val_vec, pred_is, z, 0))

    if passage_index == 0:
        names, kinds, genders, places = m["names"], m["kinds"], m["genders"], m["places"]
        held_relations, initial_values = m["held_relations"], m["initial_values"]
        for i in m["entity_order"]:
            _register(names[i], kinds[i], genders[i], places[i],
                       tuple(held_relations[i]),
                       tuple(initial_values[i][r] for r in held_relations[i]))
        return steps, cand_sets, forced_map, atom_lookup, None

    if passage_index != n_passages - 1:
        # Filler passage (n_passages == 3 only) -- registration only.
        for fe in m["filler_entities"]:
            _register(fe["name"], fe["kind"], fe["gender"], fe["place"])
        return steps, cand_sets, forced_map, atom_lookup, None

    # Final passage: mention + question.
    referent_id = m["referent_instance_id"]
    shared_name = m["shared_name"]
    original_kind = m["original_kind"]
    condition = m["condition"]
    has_description = m["has_description"]
    new_kind = m["new_kind"]
    mention_relation = m["mention_relation"]
    mention_new_value = m["mention_new_value"]
    gold_id = m["gold_link"]

    referent_atom = registry.lookup(referent_id).numpy()
    atom_lookup[referent_id] = referent_atom
    new_id, new_atom_t = registry.mint(shared_name)   # ALWAYS minted -- uniform candidate-set shape
    new_atom = new_atom_t.numpy()
    atom_lookup[new_id] = new_atom

    if condition == "new":
        # Register the NEW instance's own kind fact -- "a doctor named
        # mary" introduces + attributes in one beat, so the mention step's
        # own evidence-relation (attr:kind) read sees it.
        steps.append((new_atom, kind_rel, codec.filler_vec("kind:" + new_kind), pred_is, z, 0))
        evidence_target_key = "kind:" + new_kind
    elif has_description:
        evidence_target_key = "kind:" + original_kind
    else:
        # Bare name, "same" condition: NO kind evidence available at all --
        # see DocumentGenerator's honesty invariant #2. "name:" grounds via
        # _ent_vec (nsm_ct.clause_reactor._ground_evidence_target), which by
        # construction does not correlate with either candidate's attr:kind
        # readout: the resolver must lean on cand_from_ltm/recency instead.
        evidence_target_key = "name:" + shared_name

    mention_word = shared_name
    placeholder = _ent_vec(shared_name, resolver, codec, cache, meaning_source)
    if has_description:
        placeholder = placeholder + _content_vec(original_kind, resolver, codec, cache, meaning_source)

    cand_roles = [referent_id, new_id]
    attr_rel_vec = codec.filler_vec("attr:" + mention_relation)
    overwrite_vec = _content_vec(mention_new_value, resolver, codec, cache, meaning_source)
    step_idx = len(steps)
    steps.append((placeholder, attr_rel_vec, overwrite_vec, pred_is, z, 0))

    if not cheat:
        candidates = [membrane.Candidate(key=cid, prior=0.5) for cid in cand_roles]
        gold_index = None if no_gold else cand_roles.index(gold_id)
        # from_ltm: [1, 0] -- the referent's own atom is the ONLY candidate
        # sourced from a prior (already-consolidated) passage; the freshly
        # minted NEW candidate is never from_ltm (see nsm_ct.ltm's module
        # docstring / nsm_ct.membrane.EntityCandidateSet.from_ltm).
        from_ltm_arr = np.array([1.0, 0.0], dtype=np.float32)
        cs = membrane.EntityCandidateSet(
            candidates=candidates,
            provenance={"sentence_index": step_idx, "kind": "document", "condition": condition},
            surface=mention_word,
            feature=membrane.mention_feature_vector(mention_word),
            gold_index=gold_index,
            addr_redirect=True,
            evidence_relation="kind",
            evidence_target=evidence_target_key,
            from_ltm=from_ltm_arr,
        )
        cand_sets[step_idx] = cs
        if force is not None:
            true_idx = cand_roles.index(gold_id)
            wrong_idx = 1 - true_idx
            forced_map[step_idx] = true_idx if force == "gold" else wrong_idx

    # Question: DIRECT address (never a second candidate set -- see module
    # comment above and DocumentGenerator's own docstring).
    question_type = m["question_type"]
    q_target_id = gold_id if question_type == "ii" else referent_id
    q_relation = m["untouched_relation"] if question_type == "i" else mention_relation
    q_atom = atom_lookup.get(q_target_id)
    if q_atom is None:
        q_atom = registry.lookup(q_target_id).numpy()
    q_rel_vec = codec.filler_vec("attr:" + q_relation)
    steps.append((q_atom, q_rel_vec, z, q_pred, z, 1))

    return steps, cand_sets, forced_map, atom_lookup, None


# ---------------------------------------------------------------------------
# M57d (PROVENANCE wiring into the live reactor, CLAUDE.md's M57
# memory-schema decision): STEP LABELS -- a sibling to _writeback_steps/
# _instance_steps/_rich_steps for EACH of those three functions, mirroring
# this codebase's own established convention (M54/M55a/M57b/M57c each added
# a dedicated sibling function rather than editing an earlier, already-
# proven one). These do NOT build grounded vectors at all -- they replay
# the EXACT SAME step-construction order as their vector-building sibling,
# emitting human-readable LABELS instead (entity_label, relation_label,
# value_label, candidate_ids, referring_device) for every STATEMENT step,
# in step order. build_clause_batch zips this label stream against the
# episode's own ``steps``/``cand_sets``/``ep.context`` to build
# ``ClauseBatch.step_meta`` -- see that field's own docstring for why this
# lives in build_clause_batch rather than as a new return value threaded
# through _writeback_steps/_instance_steps/_rich_steps themselves (their
# existing 3/5-tuple return shapes are unpacked positionally by
# tests/test_instance_curriculum.py and tests/test_writeback.py; adding a
# new element there would break every direct caller for a purely
# membrane-side bookkeeping concern that has nothing to do with the
# reactor's own arithmetic).
#
# Each helper returns ``[(entity_label_or_None, relation_label,
# value_label, candidate_ids, referring_device_or_None)]``, one tuple per
# STATEMENT step (the question step is never included -- it never writes,
# so it never appears in ``ClauseBatch.step_meta`` either). ``entity_label``
# is the statically-known address (an instance id, for instance/rich; a
# plain name, for writeback) when the step addresses directly, ``None``
# when it addresses via a referring expression instead (candidate_ids
# non-empty in that case).
# ---------------------------------------------------------------------------
def _writeback_step_labels(ep, cand_sets: Dict[int, "membrane.EntityCandidateSet"]):
    """Mirrors :func:`_writeback_steps`'s S0-S4 order exactly."""
    m = ep.meta
    labels = [
        (m["name_a"], "rel:PLACE", m["place_a"], [], None),
        (m["name_b"], "rel:PLACE", m["place_b"], [], None),
        (m["true_antecedent"], "rel:ATTR", m["stale_attr"], [], None),
        (m["other_entity"], "rel:ATTR", m["other_attr"], [], None),
    ]
    pronoun_idx = len(labels)
    cs = cand_sets.get(pronoun_idx)
    cand_ids = list(cs.keys) if cs is not None else [m["name_a"], m["name_b"]]
    labels.append((None, "rel:ATTR", m["pronoun_attr"], cand_ids, "pronoun"))
    return labels


def _instance_step_labels(ep, cand_sets: Dict[int, "membrane.EntityCandidateSet"]):
    """Mirrors :func:`_instance_steps`'s registration/baseline/overwrite
    order exactly (the question step, if any, is excluded -- see module
    comment above)."""
    m = ep.meta
    ids = {"a": m["registry_order"][0], "b": m["registry_order"][1], "c": m["registry_order"][2]}
    labels = []
    for r in ("a", "b", "c"):
        labels.append((ids[r], "attr:kind", m[f"kind_{r}"], [], None))
        labels.append((ids[r], "attr:gender", m[f"gender_{r}"], [], None))
        labels.append((ids[r], "attr:place", m[f"place_{r}"], [], None))
    baseline_order = ["a", "b", "c"]
    if m["referring_device"] == "ambiguous_name":
        ref = m["referent_role"]
        baseline_order = [r for r in baseline_order if r != ref] + [ref]
    for r in baseline_order:
        labels.append((ids[r], "attr:trait", m[f"baseline_{r}"], [], None))
    overwrite_idx = len(labels)
    cs = cand_sets.get(overwrite_idx)
    cand_ids = list(cs.keys) if cs is not None else []
    labels.append((None, "attr:trait", m["overwrite_attr"], cand_ids, m["referring_device"]))
    return labels


def _rich_step_labels(ep, cand_sets: Dict[int, "membrane.EntityCandidateSet"]):
    """Mirrors :func:`_rich_steps`'s registration/referring-statement/
    PLURAL-group order exactly (the question step, if any, is excluded --
    see module comment above)."""
    m = ep.meta
    entity_order = m["entity_order"]
    kinds, genders, places = m["kinds"], m["genders"], m["places"]
    held_relations, initial_values = m["held_relations"], m["initial_values"]
    ids = dict(zip(entity_order, m["registry_order"]))
    has_group = bool(m.get("has_group"))
    labels = []
    for i in entity_order:
        labels.append((ids[i], "attr:kind", kinds[i], [], None))
        labels.append((ids[i], "attr:gender", genders[i], [], None))
        labels.append((ids[i], "attr:place", places[i], [], None))
        for rel in held_relations[i]:
            labels.append((ids[i], "attr:" + rel, initial_values[i][rel], [], None))
        if has_group:
            labels.append((ids[i], "attr:number", "sg", [], None))
    for stmt in m["referring_statements"]:
        t = len(labels)
        cs = cand_sets.get(t)
        cand_ids = list(cs.keys) if cs is not None else []
        labels.append((None, "attr:" + stmt["relation"], stmt["new_value"], cand_ids, stmt["device"]))
    if has_group:
        group_id = m["group_instance_id"]
        m0, m1 = m["group_members"]
        labels.append((group_id, "attr:number", "pl", [], None))
        labels.append((group_id, "attr:member_0", ids[m0], [], None))
        labels.append((group_id, "attr:member_1", ids[m1], [], None))
        t = len(labels)
        cs = cand_sets.get(t)
        cand_ids = list(cs.keys) if cs is not None else []
        labels.append((None, "attr:" + m["group_relation"], m["group_value"], cand_ids, "plural_pronoun"))
    return labels


def _document_step_labels(ep, cand_sets: Dict[int, "membrane.EntityCandidateSet"]):
    """Mirrors :func:`_document_steps`'s per-passage step order exactly (the
    question step is excluded, like every other ``_*_step_labels`` sibling).
    Critical plumbing, not decoration: without this, :func:`nsm_ct.
    provenance.record_writes` produces zero records for every document
    passage, so :func:`nsm_ct.ltm.promote` never has anything to consolidate
    -- the whole cross-passage recall mechanism depends on this."""
    m = ep.meta
    passage_index = m["passage_index"]
    n_passages = m["n_passages"]
    labels = []

    def _reg(iid, kind, gender, place, relations=(), values=()):
        labels.append((iid, "attr:kind", kind, [], None))
        labels.append((iid, "attr:gender", gender, [], None))
        labels.append((iid, "attr:place", place, [], None))
        for rel, val in zip(relations, values):
            labels.append((iid, "attr:" + rel, val, [], None))

    if passage_index == 0:
        names, kinds, genders, places = m["names"], m["kinds"], m["genders"], m["places"]
        held_relations, initial_values = m["held_relations"], m["initial_values"]
        ids = dict(zip(m["entity_order"], m["registry_order"]))
        for i in m["entity_order"]:
            _reg(ids[i], kinds[i], genders[i], places[i],
                 held_relations[i], [initial_values[i][r] for r in held_relations[i]])
        return labels

    if passage_index != n_passages - 1:
        for fe in m["filler_entities"]:
            _reg(fe["id"], fe["kind"], fe["gender"], fe["place"])
        return labels

    # Final passage.
    if m["condition"] == "new":
        new_id = m["link_candidates"][1]
        labels.append((new_id, "attr:kind", m["new_kind"], [], None))
    t = len(labels)
    cs = cand_sets.get(t)
    cand_ids = list(cs.keys) if cs is not None else list(m["link_candidates"])
    labels.append((None, "attr:" + m["mention_relation"], m["mention_new_value"], cand_ids,
                    "cross_passage_mention"))
    return labels


_STEP_LABEL_BUILDERS = {
    "writeback": _writeback_step_labels,
    "instance": _instance_step_labels,
    "document": _document_step_labels,
    "rich": _rich_step_labels,
}


def _step_meta_for_row(ep, cand_sets: Dict[int, "membrane.EntityCandidateSet"],
                        episode_index: int) -> Optional[List[Optional[dict]]]:
    """``[Optional[dict]]`` for one episode's ROW-LOCAL step indices --
    ``None`` at every index this episode's kind doesn't label (including
    every question step) -- see ``ClauseBatch.step_meta``'s own docstring
    for the full field contract and :func:`nsm_ct.provenance.record_writes`
    for the consumer. Returns ``None`` (not a list) for a kind with no
    label builder at all (old/reasoning/ambiguity/garden-path episodes)."""
    builder = _STEP_LABEL_BUILDERS.get(ep.meta.get("kind"))
    if builder is None:
        return None
    labels = builder(ep, cand_sets)
    context = ep.context
    return [
        {
            "sentence_index": t,
            "surface": context[t] if t < len(context) else None,
            "relation_label": relation_label,
            "value_label": value_label,
            "entity_label": entity_label,
            "candidate_ids": candidate_ids,
            "referring_device": referring_device,
            "episode_index": episode_index,
        }
        for t, (entity_label, relation_label, value_label, candidate_ids, referring_device)
        in enumerate(labels)
    ]


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
    # M57b (resolver-driven write-BACK, CLAUDE.md's M57 memory-schema
    # decision): ONE new field, kept deliberately separate from every cand_*
    # tensor above (not folded into cand_gold/cand_mask) so a batch with NO
    # write-back episode leaves it None -- byte-identical to every pre-M57b
    # batch, same guarantee every prior addition in this dataclass made for
    # its own group. Truthy at (row, step) means: THIS step's ENTITY
    # candidate set (the SAME cand_entity/cand_mask/... group M53a/M53b
    # already populate -- write-back reuses that group wholesale, it is
    # not a fourth tensor family) resolves the WRITE ADDRESS ("she is
    # tall ." -- who does "she" address?), not the value ("she found the
    # ball ." -- M53a/M53b's existing pronoun-antecedent-for-a-VALUE
    # convention, untouched). See ClauseReactor._collapse's M57b paragraph
    # for the arithmetic this drives.
    cand_addr_mask: Optional[torch.Tensor] = None   # [B, T]  truthy = address-redirect, not value-redirect
    # M57b (v2 redesign, honest validity machinery replacing the
    # aux-gold-corruption "wrong_binding" arm): ``cand_forced_index`` --
    # per (row, step), the ENTITY candidate index to teacher-force the
    # collapse weights to, overriding the resolver's own logits entirely
    # (both train and eval) -- ``-1`` = "not forced" (the sentinel, mirrors
    # ``cand_gold``'s own -1-elsewhere convention). ``None`` (the default)
    # is byte-identical to an all-``-1`` tensor and to every pre-M57b-v2
    # batch -- see :meth:`ClauseReactor._collapse`'s teacher-forcing
    # paragraph. Generic machinery: not specific to write-back or to
    # address-redirect rows -- any curriculum's step-building code may
    # populate it (today only :func:`_writeback_steps`/``writeback_force``
    # does, via :func:`build_clause_batch`).
    cand_forced_index: Optional[torch.Tensor] = None  # [B, T]  long; forced candidate index, -1 = not forced
    # M57c (instance atoms + definite-description referring expressions,
    # dev/MIND_INTERFACE.md's v2 addendum): ONE new field, kept deliberately
    # separate from every cand_* tensor above (same "fourth optional field
    # to guard" discipline the M57b fields already established) -- a batch
    # with no evidence-relation candidate set (every pre-M57c batch, and
    # every M53a/M53b/M57b candidate set, which never populates
    # EntityCandidateSet.evidence_relation) leaves this None, byte-identical
    # to every field above it. Per (row, step): the RELATION vector to read
    # each candidate's LIVE memory slot under, REPLACING the step's own
    # relation ``r`` for that one collapse computation only ("the doctor"
    # reads attr:kind, "she" reads attr:gender -- see
    # membrane.EntityCandidateSet.evidence_relation's docstring and
    # ClauseReactor._collapse's entity branch for where this is consumed).
    cand_evidence_relation: Optional[torch.Tensor] = None  # [B, T, d]
    # M57c.3 (RESEARCH_NOTES "M57c battery #2" -- forced-gold PROVES the read
    # path but the trained resolver binds instance candidates at CHANCE: it
    # sees each candidate's evidence READOUT but never the referring
    # expression's own TARGET vector to compare it against): ONE new field,
    # the SAME "optional, None whenever no candidate set in the batch
    # populates EntityCandidateSet.evidence_target" discipline
    # cand_evidence_relation itself establishes -- a batch built entirely
    # from pre-M57c.3 candidate sets (every M53a/M53b/M57b/M57c set, and
    # every instance/rich set built before this milestone) leaves this
    # None, byte-identical to every field above it. Per (row, step): the
    # GROUNDED target vector ("doctor"'s kind vec, "F"'s gender atom, the
    # ambiguous shared name's own entity atom -- see
    # membrane.EntityCandidateSet.evidence_target's docstring and
    # :func:`_ground_evidence_target`) the resolver should compare each
    # candidate's evidence readout against
    # (:func:`nsm_ct.resolver.evidence_interaction`) -- see
    # ClauseReactor._collapse's entity branch for where this is consumed.
    cand_evidence_target: Optional[torch.Tensor] = None  # [B, T, d]
    # M57c.2 (RESEARCH_NOTES "M57c battery #1" -- inverse_query measured
    # BELOW chance because no entity-axis read existed): ONE new field, the
    # SAME "optional field defaults None, byte-identical for every batch
    # that doesn't populate it" discipline every field above establishes.
    # Truthy at (row, step) means: this step's read has NO memory address
    # to query at all (an inverse-query "who is {trait} ?" step's own
    # "entity" is a fixed marker atom, never written to) -- ClauseReactor.
    # forward must instead unbind the ENTITY axis from THIS step's own
    # (relation, value) via :func:`nsm_ct.entity_memory.query_entity`. Only
    # :func:`nsm_ct.clause_reactor._instance_steps` (inverse-query episodes)
    # ever sets this; every other curriculum leaves it ``None``.
    inverse_mask: Optional[torch.Tensor] = None  # [B, T]
    # M57d (PROVENANCE wiring into the live reactor, CLAUDE.md's M57
    # memory-schema decision -- "provenance is a membrane-side, append-only
    # log, one record per gated write"): a Python-side (NOT a tensor) field,
    # [B][T], entries ``None`` where no statement/write happens at that
    # step (every question step, and every episode kind this milestone
    # doesn't label -- old L1-6, reasoning, ambiguity, garden-path). Kept
    # OUT of every tensor group above deliberately: superposing labels into
    # the memory tensor is exactly what MIND_INTERFACE.md invariant #4 says
    # the audit trail must NOT do (see instances.py's own module
    # docstring). Populated by :func:`build_clause_batch` for writeback/
    # instance/rich episodes only; ``None`` (the default) for every batch
    # that doesn't populate it -- byte-identical to every pre-M57d batch,
    # same "optional field, guarded" discipline every field above
    # establishes. Consumed by :func:`nsm_ct.provenance.record_writes`,
    # never by :meth:`ClauseReactor.forward` itself (no arithmetic reads
    # this field -- it rides alongside the tensors, not through them).
    step_meta: Optional[List[List[Optional[dict]]]] = None  # [B][T] plain dicts
    # M59a (episodic LTM, CLAUDE.md's "LTM decisions" / dev/LTM_DESIGN_BRIEF.md
    # Sec.5 point 2 -- "identity linking through the EXISTING resolver
    # contract"): ONE new field, appended LAST (after step_meta) so every
    # pre-M59a POSITIONAL ``ClauseBatch(...)`` call site (build_clause_batch's
    # own return statement included) is untouched -- it simply defaults to
    # ``None``, byte-identical to every batch built before this milestone.
    # Per (row, step, candidate): 1 = this entity candidate's only source is
    # a prior passage's already-consolidated LTM (see
    # ``membrane.EntityCandidateSet.from_ltm``'s docstring for the full
    # cross-passage candidate-set contract), 0 = an ordinary same-passage
    # STM candidate. Consumed by :meth:`ClauseReactor._collapse`'s entity
    # branch as ONE MORE per-candidate feature column, appended onto the
    # SAME ``cand_feature_per_candidate`` register M57c.3's
    # ``evidence_interaction`` scalar already widens (``resolver.
    # cand_feature_extra`` grows to cover however many extra columns are
    # actually present this batch -- see ``_collapse`` for the concat
    # logic). Not yet populated by :func:`build_clause_batch` (M59b's
    # curriculum-generator job); ``None`` here for every batch this
    # milestone builds, same "hand this to the next milestone" contract
    # ``membrane.EntityCandidateSet.from_ltm`` documents on its own field.
    cand_from_ltm: Optional[torch.Tensor] = None  # [B, T, C]  0/1, 0 elsewhere
    # M60 (op-library integration, dev/OP_LIBRARY_MAP.md's ``recency`` row --
    # RESEARCH_NOTES "M57 battery #3": recency-only referent cases, e.g. the
    # "ambiguous_name" instance/rich device, sit at chance): ONE new field,
    # appended LAST (after cand_from_ltm) so every pre-M60 POSITIONAL
    # ``ClauseBatch(...)`` call site is untouched -- defaults to ``None``,
    # byte-identical to every batch built before this milestone. Per (row,
    # step, candidate, [steps_since, log_count, is_most_recent]): the THREE
    # :func:`nsm_ct.ops.recency` features, computed at batch-build time from
    # each candidate set's ``mention_steps``/``mention_counts`` (see
    # membrane.EntityCandidateSet's own docstring paragraph) against the
    # step's own index -- deterministic, no learned parameters (centering-
    # theory salience, dev/OP_INVENTORY.md's DNC temporal-link-style
    # ordering). ``None`` for every batch with no candidate set carrying
    # ``mention_steps`` at all -- the SAME "only if present" discipline
    # every optional field above establishes. Consumed by
    # :meth:`ClauseReactor._collapse`'s entity branch as a THIRD optional
    # extra-column group, appended onto ``cand_feature_per_candidate``
    # alongside ``evidence_interaction``'s scalar and ``cand_from_ltm``'s
    # flag (``resolver.cand_feature_extra`` widens to cover however many of
    # the three are actually present this batch).
    cand_recency: Optional[torch.Tensor] = None  # [B, T, C, 3]

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
        addr_mask = self.cand_addr_mask.to(device) if self.cand_addr_mask is not None else None
        forced = self.cand_forced_index.to(device) if self.cand_forced_index is not None else None
        evidence_rel = self.cand_evidence_relation.to(device) if self.cand_evidence_relation is not None else None
        evidence_target = self.cand_evidence_target.to(device) if self.cand_evidence_target is not None else None
        inv_mask = self.inverse_mask.to(device) if self.inverse_mask is not None else None
        from_ltm = self.cand_from_ltm.to(device) if self.cand_from_ltm is not None else None
        recency = self.cand_recency.to(device) if self.cand_recency is not None else None
        # M57d: step_meta is plain Python (no device), carried through
        # unchanged -- mirrors nothing else in this method living off-device.
        return ClauseBatch(self.entity.to(device), self.relation.to(device),
                           self.value.to(device), self.pred.to(device),
                           self.is_q.to(device), self.mask.to(device),
                           self.options.to(device), self.answer.to(device), coord, ans_ok,
                           *cand, *scand, *hcand, addr_mask, forced, evidence_rel, evidence_target,
                           inv_mask, self.step_meta, from_ltm, recency)

    def subset(self, idx) -> "ClauseBatch":
        """A minibatch over the leading (episode) dimension."""
        coord = self.coord[idx] if self.coord is not None else None
        ans_ok = self.answerable[idx] if self.answerable is not None else None
        cand = self._cand_fields(lambda t: t[idx])
        scand = self._sense_cand_fields(lambda t: t[idx])
        hcand = self._hyp_cand_fields(lambda t: t[idx])
        addr_mask = self.cand_addr_mask[idx] if self.cand_addr_mask is not None else None
        forced = self.cand_forced_index[idx] if self.cand_forced_index is not None else None
        evidence_rel = self.cand_evidence_relation[idx] if self.cand_evidence_relation is not None else None
        evidence_target = self.cand_evidence_target[idx] if self.cand_evidence_target is not None else None
        inv_mask = self.inverse_mask[idx] if self.inverse_mask is not None else None
        # M57d: step_meta is a plain Python list, not a tensor -- index it
        # by hand (torch fancy-indexing doesn't apply) so a minibatch
        # subset keeps its per-row alignment with every tensor field above.
        if self.step_meta is not None:
            idx_list = idx.tolist() if torch.is_tensor(idx) else list(idx)
            step_meta = [self.step_meta[i] for i in idx_list]
        else:
            step_meta = None
        from_ltm = self.cand_from_ltm[idx] if self.cand_from_ltm is not None else None
        recency = self.cand_recency[idx] if self.cand_recency is not None else None
        return ClauseBatch(self.entity[idx], self.relation[idx], self.value[idx],
                           self.pred[idx], self.is_q[idx], self.mask[idx],
                           self.options[idx], self.answer[idx], coord, ans_ok,
                           *cand, *scand, *hcand, addr_mask, forced, evidence_rel, evidence_target,
                           inv_mask, step_meta, from_ltm, recency)


def build_clause_batch(episodes, parser, resolver, codec: TPRCodec,
                        meaning_source: MeaningSource = "usvs",
                        sense_bind: str = "gold",
                        reading_bind: str = "gold",
                        writeback_cheat: bool = False,
                        writeback_no_gold: bool = False,
                        writeback_force: Optional[str] = None,
                        document_registry: Optional[InstanceRegistry] = None) -> ClauseBatch:
    """Encode curriculum episodes into grounded clause-triple streams (fixed).

    Each step is ``(entity, relation, value, pred, coord, is_q)``; ``coord`` carries
    the OR/NOT atom for disjunction/negation steps (zeros otherwise) — the logical
    signal the controller reacts to. Options ground to content vectors, except
    "maybe" → the NSM MAYBE atom (so a disjunction can be answered "maybe").

    ``reading_bind`` (M55b, mirrors ``sense_bind``): ``"gold"`` (default)
    placeholder-binds a garden-path episode's collapse step to the TRUE
    gold reading's place (the ceiling); ``"wrong"`` binds it to the
    OPPOSITE reading instead (``--wrong-binding``'s floor probe) -- see
    :func:`_garden_path_steps`. Inert for every episode without
    ``ep.meta["garden_path"]``.

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

    M57b (resolver-driven write-BACK, CLAUDE.md's M57 memory-schema
    decision): an episode whose meta carries ``kind == "writeback"``
    (``curriculum2.WriteBackCurriculumGenerator`` only) is routed WHOLESALE
    through :func:`_writeback_steps` (its own dedicated, parser-free stream
    -- see that function's docstring for why). Its
    :class:`nsm_ct.membrane.EntityCandidateSet` (``addr_redirect=True``)
    rides through the SAME ``cand_*`` fields M53a/M53b's pronoun-antecedent
    sets already use (NOT a fourth tensor family -- both kinds share one
    contract, they just differ in what the collapsed choice REDIRECTS), and
    the new ``cand_addr_mask`` field is set truthy at exactly those steps --
    ``None`` for every batch with no ``addr_redirect`` candidate set at all,
    keeping this byte-identical to pre-M57b for every existing batch/level
    (including plain pronoun-binding batches, which never set
    ``addr_redirect``). ``writeback_cheat``/``writeback_no_gold`` forward to
    :func:`_writeback_steps`'s own ``cheat``/``no_gold`` (the M57b
    cheat-baseline / no-gold-eval arms); both default ``False`` (inert,
    pre-M57b behavior) and are no-ops for every episode without
    ``ep.meta["kind"] == "writeback"``. ``writeback_force`` (``"gold"`` |
    ``"wrong"`` | ``None``, v2's honest validity machinery, replacing the
    curriculum-level ``wrong_binding`` aux-gold-corruption arm) forwards to
    :func:`_writeback_steps`'s own ``force`` -- populates the returned
    batch's ``cand_forced_index`` at the pronoun step, which
    :meth:`ClauseReactor._collapse` then uses to teacher-force the collapse
    weights regardless of the resolver's own logits. Default ``None``
    (inert) leaves ``cand_forced_index`` ``None`` for every batch, same
    byte-identity guarantee every other optional field here makes.

    M57c (instance atoms + definite-description referring expressions,
    dev/MIND_INTERFACE.md's v2 addendum): an episode whose meta carries
    ``kind == "instance"`` (``curriculum2.InstanceCurriculumGenerator``
    only) is routed WHOLESALE through :func:`_instance_steps` (see that
    function's own extensive docstring for the full design: instance atoms
    minted per-episode by a fresh :class:`nsm_ct.instances.InstanceRegistry`
    instead of ``var:<name>`` codec hashes, so two same-named instances
    genuinely differ; every referring expression -- definite description,
    pronoun, ambiguous shared name -- resolves via the SAME ``cand_*``
    fields M53a/M53b/M57b already use, plus the new
    ``cand_evidence_relation`` field, which names the attribute (kind/gender)
    each candidate's LIVE memory slot gets read under for scoring).
    ``writeback_cheat``/``writeback_no_gold``/``writeback_force`` are
    REUSED for instance episodes too (forwarded to :func:`_instance_steps`'s
    own ``cheat``/``no_gold``/``force`` -- one arm setting applies across a
    mixed writeback+instance batch in a single training run, matching how
    a real training script mixes curricula in ONE arm); all three remain
    complete no-ops for every episode without ``ep.meta["kind"] in
    ("writeback", "instance")``.

    M57c.2 (RESEARCH_NOTES "M57c battery #1"): an inverse-query instance
    episode's question step (``_instance_steps``'s ``inverse_step_idx``)
    populates the new ``inverse_mask`` field (``None`` for every batch
    without one -- the same "only if present" discipline every optional
    field above establishes), and its MC options are grounded as the
    per-episode INSTANCE ATOMS THEMSELVES (``atom_lookup`` in
    ``ep.meta["registry_order"]`` order) rather than
    :func:`_instance_option_vec`'s composite name+kind vectors -- see that
    function's own docstring addendum for why.

    M57c.3 (RESEARCH_NOTES "M57c battery #2" -- forced-gold PROVES the read
    path, but the TRAINED resolver binds instance candidates at CHANCE: it
    never had the referring expression's own TARGET vector to compare a
    candidate's evidence readout against): whenever an
    :class:`nsm_ct.membrane.EntityCandidateSet` carries an
    ``evidence_target`` (``_instance_steps``/``_rich_steps`` only), it is
    grounded via :func:`_ground_evidence_target` into the new
    ``cand_evidence_target`` field -- ``None`` for every batch built
    without one, the same "only if present" discipline
    ``cand_evidence_relation`` itself establishes. See
    :meth:`ClauseReactor._collapse`'s entity branch for how this drives
    :func:`nsm_ct.resolver.evidence_interaction`.

    RICH-EPISODE curriculum (CLAUDE.md's 2026-08-30 reprioritization, "stop
    requiring minimal episodes"): an episode whose meta carries ``kind ==
    "rich"`` (``curriculum2.RichEpisodeGenerator`` only) is routed WHOLESALE
    through :func:`_rich_steps` -- a direct N-entity/K-statement
    generalization of :func:`_instance_steps` (kept unchanged; see
    :func:`_rich_steps`'s own docstring). ``writeback_cheat``/
    ``writeback_no_gold``/``writeback_force`` are REUSED for rich episodes
    too, exactly as they already are for writeback/instance episodes mixed
    into the same batch. A rich inverse-query episode's MC options ground
    the same way an instance inverse-query episode's do (real per-episode
    instance atoms via ``atom_lookup``), keyed by ``ep.meta
    ["inverse_option_ids"]`` (a curriculum-CHOSEN subset of up to
    ``num_options`` entities, not every minted id -- ``n_entities`` may
    exceed ``num_options``, unlike the fixed-3-entity instance world).

    M59b (CROSS-PASSAGE document curriculum, episodic LTM): an episode whose
    meta carries ``kind == "document"`` (``curriculum2.DocumentGenerator``
    only) is routed WHOLESALE through :func:`_document_steps`, exactly like
    the instance/rich branches above -- EXCEPT its
    :class:`~nsm_ct.instances.InstanceRegistry` is NOT built internally: the
    caller passes it in as ``document_registry`` (``None`` by default,
    inert for every non-document batch), one registry PER DOCUMENT, threaded
    through this function once per passage of that document (see
    ``src/nsm_ct/ltm.py``'s module docstring, "Interface contract for the
    curriculum agent" -- a document's passages must be built, in passage
    order, via SEPARATE calls to this function sharing the same
    ``document_registry``; a single call never mixes rows from more than one
    document when ``document_registry`` is set). Its
    :class:`~nsm_ct.membrane.EntityCandidateSet` (``addr_redirect=True``,
    exactly 2 candidates: the referent's own instance id and a freshly
    minted "NEW" id) rides through the SAME ``cand_*`` fields every other
    ``EntityCandidateSet`` uses, PLUS the new ``cand_from_ltm`` field
    (populated here from ``EntityCandidateSet.from_ltm`` -- ``None`` for
    every batch with no ``from_ltm``-bearing candidate set, the same
    "only if present" discipline every optional field above establishes).
    ``writeback_cheat``/``writeback_no_gold``/``writeback_force`` are REUSED
    for document episodes too, forwarded to :func:`_document_steps`'s own
    ``cheat``/``no_gold``/``force``.
    """
    cache: Dict[str, np.ndarray] = {}
    d = codec.dim
    q_pred = codec.filler_vec("pred:?")             # the question's (unknown) predicate
    z = np.zeros(d, np.float32)
    rows = []
    forced_maps: List[Dict[int, int]] = []   # parallel to rows; see ClauseBatch.cand_forced_index
    # M57c: parallel to rows, {instance_id: atom_ndarray} for episodes routed
    # through _instance_steps, None for every other episode -- see that
    # function's docstring for why the generic cand_entity-grounding loop
    # below needs this (candidate keys are instance ids, not names/content
    # words _ent_vec knows how to ground).
    atom_lookups: List[Optional[Dict[str, np.ndarray]]] = []
    # M57c.2: parallel to rows, the (row-local) step index of an
    # inverse-query episode's "who is {trait} ?" step, None for every other
    # episode -- see ClauseBatch.inverse_mask's docstring for why this needs
    # its own tensor rather than folding into the cand_* groups (there is no
    # candidate set at this step at all).
    inverse_step_indices: List[Optional[int]] = []
    # M57d: parallel to rows, one row-local step-label list (or None) per
    # episode -- see ClauseBatch.step_meta's own docstring and
    # _step_meta_for_row's for why this is computed here (from cand_sets,
    # already built above) rather than threaded through
    # _writeback_steps/_instance_steps/_rich_steps themselves.
    step_metas: List[Optional[List[Optional[dict]]]] = []
    for ep in episodes:
        cand_sets: Dict[int, "membrane.EntityCandidateSet"] = {}
        sense_cand_sets: Dict[int, "membrane.SenseCandidateSet"] = {}
        hyp_cand_sets: Dict[int, "membrane.HypothesisCandidateSet"] = {}
        forced_map: Dict[int, int] = {}
        atom_lookup: Optional[Dict[str, np.ndarray]] = None
        inverse_step_idx: Optional[int] = None
        if getattr(ep, "level", 0) >= 9 and ep.meta.get("query"):
            steps = _reasoning_steps(ep, resolver, codec, cache, meaning_source)   # L9-L11 reasoning stream
        elif ep.meta.get("homograph"):
            steps, sense_cand_sets = _ambiguity_steps(ep, parser, resolver, codec, cache,
                                                       meaning_source, sense_bind)   # M54 ambiguity stream
        elif ep.meta.get("garden_path"):
            steps, hyp_cand_sets = _garden_path_steps(ep, parser, resolver, codec, cache,
                                                        meaning_source, reading_bind)   # M55a/M55b garden-path stream
        elif ep.meta.get("kind") == "writeback":
            steps, cand_sets, forced_map = _writeback_steps(
                ep, resolver, codec, cache, meaning_source,
                cheat=writeback_cheat, no_gold=writeback_no_gold,
                force=writeback_force)   # M57b write-back stream (v2)
        elif ep.meta.get("kind") == "instance":
            steps, cand_sets, forced_map, atom_lookup, inverse_step_idx = _instance_steps(
                ep, resolver, codec, cache, meaning_source,
                cheat=writeback_cheat, no_gold=writeback_no_gold,
                force=writeback_force)   # M57c instance-atom stream
        elif ep.meta.get("kind") == "rich":
            steps, cand_sets, forced_map, atom_lookup, inverse_step_idx = _rich_steps(
                ep, resolver, codec, cache, meaning_source,
                cheat=writeback_cheat, no_gold=writeback_no_gold,
                force=writeback_force)   # RICH-EPISODE curriculum stream (N entities, K statements)
        elif ep.meta.get("kind") == "document":
            steps, cand_sets, forced_map, atom_lookup, inverse_step_idx = _document_steps(
                ep, document_registry, resolver, codec, cache, meaning_source,
                cheat=writeback_cheat, no_gold=writeback_no_gold,
                force=writeback_force)   # M59b cross-passage document stream (episodic LTM)
        elif ep.meta.get("kind") == "prose":
            # round 2 item 1: prose pronoun stream (_context_steps' prose
            # counterpart) -- the QUESTION step reuses _question_entity/
            # _queried_role exactly as the old/default path below does
            # (corpus.py's own self-generated questions are plain "where is
            # the X ?"-style text, not a bespoke shape like writeback/
            # instance/rich/document's own internally-built question steps).
            steps, cand_sets = _prose_steps(ep, parser, resolver, codec, cache, meaning_source)
            qent = _question_entity(ep.question)
            if qent is None:
                continue
            qrel = _queried_role(ep.question)
            steps.append((_ent_vec(qent, resolver, codec, cache, meaning_source),
                          codec.filler_vec("rel:" + qrel), z, q_pred, z, 1))
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
        # M57c: inverse-query instance episodes answer with IDENTITY options
        # ("mary the doctor"), not content words. M57c.2: grounded as the
        # per-episode INSTANCE ATOMS THEMSELVES (``atom_lookup`` -- the exact
        # atoms _instance_steps just minted), keyed by ``ep.meta
        # ["registry_order"]`` (a/b/c mint order, which is also ep.options's
        # order by construction -- see the module comment above
        # _instance_option_vec) -- NOT _instance_option_vec's composite
        # ent_vec(name)+content_vec(kind) sum, so the contrastive answer
        # compares query_entity's entity-axis readout against REAL candidate
        # atoms, the same space it was unbound from. Every other episode
        # (including every "target"-mode instance episode, whose options are
        # plain trait words) is grounded exactly as before.
        if ep.meta.get("kind") == "instance" and ep.meta.get("question_mode") == "inverse":
            opt = [atom_lookup[iid] for iid in ep.meta["registry_order"]]
        elif ep.meta.get("kind") == "rich" and ep.meta.get("question_mode") == "inverse":
            # RICH's inverse options are a SAMPLED 4-of-N subset (never
            # "one option per entity" -- N may exceed num_options), keyed
            # by ep.meta["inverse_option_ids"] rather than the full
            # registry_order (mirrors the instance branch above exactly,
            # just over a caller-chosen subset instead of every minted id).
            opt = [atom_lookup[iid] for iid in ep.meta["inverse_option_ids"]]
        else:
            opt = [_option_vec(o, resolver, codec, cache, meaning_source) for o in ep.options]
        step_metas.append(_step_meta_for_row(ep, cand_sets, len(rows)))
        rows.append((steps, opt, ep.answer_idx, 1.0 if getattr(ep, "answerable", True) else 0.0,
                     cand_sets, sense_cand_sets, hyp_cand_sets))
        forced_maps.append(forced_map)
        atom_lookups.append(atom_lookup)
        inverse_step_indices.append(inverse_step_idx)

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
    cand_addr_mask = None
    cand_evidence_relation = None
    cand_evidence_target = None
    cand_from_ltm = None
    cand_recency = None
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
            # M57c: an instance episode's candidate KEYS are instance ids
            # ("inst:mary#1"), which _ent_vec has no branch for (not a
            # var:<name> atom, not a content word) -- atom_lookups[i], when
            # present, grounds those directly to their own minted atom;
            # every other episode's atom_lookups[i] is None, so this is a
            # no-op fallback straight to _ent_vec, byte-identical to pre-M57c.
            lookup = atom_lookups[i] or {}
            for t, cs in cs_map.items():
                for j, c in enumerate(cs.candidates):
                    vec = lookup[c.key] if c.key in lookup else _ent_vec(
                        c.key, resolver, codec, cache, meaning_source)
                    cand_entity[i, t, j] = torch.from_numpy(vec)
                    cand_mask[i, t, j] = 1.0
                    cand_prior[i, t, j] = c.prior
                if cs.feature is not None:
                    cand_feature[i, t] = torch.from_numpy(cs.feature)
                if cs.gold_index is not None:
                    cand_gold[i, t] = cs.gold_index
                if has_cand_features and cs.cand_features is not None:
                    cand_feature_per_candidate[i, t, :len(cs.candidates)] = torch.from_numpy(cs.cand_features)

        # M57b: only allocated when at least one EntityCandidateSet in this
        # batch actually carries addr_redirect=True (WriteBackCurriculumGenerator
        # episodes only) -- mirrors has_cand_features/has_subject's "only if
        # present" pattern exactly. A batch with only M53a/M53b
        # value-redirect pronoun sets (addr_redirect always False, the
        # dataclass default) leaves this None, byte-identical to pre-M57b.
        has_addr_redirect = any(
            cs.addr_redirect for *_row, cs_map, _scs, _hcs in rows for cs in cs_map.values())
        if has_addr_redirect:
            cand_addr_mask = torch.zeros(b, T)
            for i, (steps, opt, a, ok, cs_map, _scs, _hcs) in enumerate(rows):
                for t, cs in cs_map.items():
                    if cs.addr_redirect:
                        cand_addr_mask[i, t] = 1.0

        # M57c: only allocated when at least one EntityCandidateSet in this
        # batch actually carries an evidence_relation (InstanceCurriculumGenerator
        # episodes only, via _instance_steps) -- mirrors has_addr_redirect's
        # "only if present" pattern exactly. A batch with only M53a/M53b/M57b
        # candidate sets (evidence_relation always None, the dataclass
        # default) leaves this None, byte-identical to pre-M57c.
        has_evidence_relation = any(
            cs.evidence_relation for *_row, cs_map, _scs, _hcs in rows for cs in cs_map.values())
        if has_evidence_relation:
            cand_evidence_relation = torch.zeros(b, T, d)
            for i, (steps, opt, a, ok, cs_map, _scs, _hcs) in enumerate(rows):
                for t, cs in cs_map.items():
                    if cs.evidence_relation:
                        cand_evidence_relation[i, t] = torch.from_numpy(
                            codec.filler_vec("attr:" + cs.evidence_relation))

        # M57c.3: only allocated when at least one EntityCandidateSet in this
        # batch actually carries an evidence_target (instance/rich episodes
        # built by this milestone's _instance_steps/_rich_steps only) --
        # mirrors has_evidence_relation's "only if present" pattern exactly.
        # A batch with only pre-M57c.3 candidate sets (evidence_target
        # always None, the dataclass default) leaves this None,
        # byte-identical to every pre-M57c.3 batch.
        has_evidence_target = any(
            cs.evidence_target for *_row, cs_map, _scs, _hcs in rows for cs in cs_map.values())
        if has_evidence_target:
            cand_evidence_target = torch.zeros(b, T, d)
            for i, (steps, opt, a, ok, cs_map, _scs, _hcs) in enumerate(rows):
                for t, cs in cs_map.items():
                    if cs.evidence_target:
                        cand_evidence_target[i, t] = torch.from_numpy(
                            _ground_evidence_target(cs.evidence_target, resolver, codec, cache, meaning_source))

        # M59b (episodic LTM curriculum wiring): cand_from_ltm, only
        # allocated when at least one EntityCandidateSet in this batch
        # actually carries a from_ltm array (curriculum2.DocumentGenerator's
        # mention-step candidate sets, via _document_steps, only) -- mirrors
        # has_evidence_target's "only if present" pattern exactly. A batch
        # with only pre-M59b candidate sets (from_ltm always None, the
        # membrane dataclass default) leaves this None, byte-identical to
        # every pre-M59b batch.
        has_from_ltm = any(
            cs.from_ltm is not None for *_row, cs_map, _scs, _hcs in rows for cs in cs_map.values())
        if has_from_ltm:
            cand_from_ltm = torch.zeros(b, T, Cmax)
            for i, (steps, opt, a, ok, cs_map, _scs, _hcs) in enumerate(rows):
                for t, cs in cs_map.items():
                    if cs.from_ltm is not None:
                        cand_from_ltm[i, t, :len(cs.candidates)] = torch.from_numpy(cs.from_ltm)

        # M60 (op-library integration -- RECENCY, dev/OP_LIBRARY_MAP.md's
        # ``recency`` row): cand_recency, only allocated when at least one
        # EntityCandidateSet in this batch actually carries mention_steps
        # (_instance_steps/_rich_steps's overwrite/question candidate sets
        # only -- see membrane.EntityCandidateSet.mention_steps's own
        # docstring) -- mirrors has_from_ltm's "only if present" pattern
        # exactly. A batch with only pre-M60 candidate sets (mention_steps
        # always None, the membrane dataclass default) leaves this None,
        # byte-identical to every pre-M60 batch. :func:`nsm_ct.ops.recency`
        # is deterministic (no learned parameters); ``current_step`` is
        # this candidate set's own step index ``t``.
        has_recency = any(
            cs.mention_steps is not None for *_row, cs_map, _scs, _hcs in rows for cs in cs_map.values())
        if has_recency:
            cand_recency = torch.zeros(b, T, Cmax, 3)
            for i, (steps, opt, a, ok, cs_map, _scs, _hcs) in enumerate(rows):
                for t, cs in cs_map.items():
                    if cs.mention_steps is None:
                        continue
                    n_c = len(cs.candidates)
                    ms = torch.from_numpy(cs.mention_steps[:n_c]).unsqueeze(0)          # [1, C]
                    mc = (torch.from_numpy(cs.mention_counts[:n_c]).unsqueeze(0)
                          if cs.mention_counts is not None else None)
                    feats = ops.recency(ms, float(t), mention_counts=mc)
                    cand_recency[i, t, :n_c, 0] = feats.steps_since[0]
                    cand_recency[i, t, :n_c, 1] = feats.log_count[0]
                    cand_recency[i, t, :n_c, 2] = feats.is_most_recent[0].to(torch.float32)

    # M57b (v2, honest validity machinery): cand_forced_index, built from
    # forced_maps (parallel to rows, populated only by _writeback_steps'
    # ``force`` today -- see ClauseBatch's field docstring). Only allocated
    # when at least one episode's forced_map is non-empty, mirroring every
    # other "only if present" optional-field pattern in this function --
    # None (byte-identical to pre-v2/absent) otherwise.
    cand_forced_index = None
    if any(forced_maps):
        cand_forced_index = torch.full((b, T), -1, dtype=torch.long)
        for i, fm in enumerate(forced_maps):
            for t, idx in fm.items():
                cand_forced_index[i, t] = idx

    # M57c.2: inverse_mask, built from inverse_step_indices (parallel to
    # rows, populated only by _instance_steps for inverse-query episodes --
    # see ClauseBatch.inverse_mask's docstring). Only allocated when at
    # least one episode actually has an inverse-query step, same "only if
    # present" pattern as cand_forced_index above -- None (byte-identical to
    # absent) for every batch without one.
    inverse_mask = None
    if any(idx is not None for idx in inverse_step_indices):
        inverse_mask = torch.zeros(b, T)
        for i, idx in enumerate(inverse_step_indices):
            if idx is not None:
                inverse_mask[i, idx] = 1.0

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

    # M57d: only allocated when at least one episode actually labeled a row
    # (writeback/instance/rich) -- the same "only if present" pattern every
    # optional field above establishes; None (byte-identical to absent) for
    # a batch built entirely from old/reasoning/ambiguity/garden-path
    # episodes. Padded to T with None, mirroring the tensor fields' own
    # zero-padding (a padded step never has a candidate/gate to record).
    step_meta = None
    if any(m is not None for m in step_metas):
        step_meta = []
        for row_m in step_metas:
            row = list(row_m) if row_m is not None else []
            row += [None] * (T - len(row))
            step_meta.append(row)

    return ClauseBatch(ent, rel, val, prd, is_q, mask, opts, ans, crd, ans_ok,
                       cand_entity, cand_mask, cand_prior, cand_feature, cand_gold,
                       cand_feature_per_candidate,
                       sense_cand_entity, sense_cand_mask, sense_cand_prior,
                       sense_cand_context, sense_cand_gold,
                       sense_cand_subject, sense_cand_subject_rel,
                       hyp_cand_entity, hyp_cand_mask, hyp_cand_prior, hyp_cand_gold,
                       hyp_cand_query_entity, hyp_cand_query_relation,
                       cand_addr_mask, cand_forced_index, cand_evidence_relation, cand_evidence_target,
                       inverse_mask, step_meta, cand_from_ltm, cand_recency)


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

    M57b (resolver-driven write-BACK, CLAUDE.md's M57 memory-schema
    decision): the ENTITY branch above (``self.resolver``, M53b) now does
    TWO different things with its collapse weights depending on
    ``batch.cand_addr_mask``, instead of always redirecting the value. Per
    row, at a step with a real candidate set:
      - ``cand_addr_mask`` falsy (or the whole field ``None`` -- every
        pre-M57b batch, and M53a/M53b's own pronoun-antecedent-for-a-VALUE
        sets) -- UNCHANGED: ``v <- Σ w · cand_mem_read`` (the resolved
        VALUE replaces the placeholder value, e.g. "she found the ball ."
        resolves WHERE the ball now is).
      - ``cand_addr_mask`` truthy (write-back sets only) -- ``e <- Σ w ·
        cand_entity`` (the resolved ADDRESS replaces the placeholder
        entity -- the candidate atoms themselves, NOT their memory
        readout, since a candidate's identity IS the address, unlike a
        value) and ``v`` is left exactly as the batch stated it
        (the clause asserts WHAT is true; the resolver only decides WHO
        it is about).
    Same resolver call, same masking, same soft-train/hard-eval collapse
    weights either way -- ``cand_addr_mask`` only changes what the
    ALREADY-COMPUTED weights get applied to. ``_collapse`` now returns the
    (possibly redirected) entity alongside the value; ``forward`` threads
    it into both the GRU input and the ``em.write`` call, so an
    address-redirected write actually lands on the resolved node instead
    of the step's own placeholder address. Resolver logits/margins are
    recorded identically for address- and value-redirect steps (same
    output keys, ``resolver_logits``/``resolver_margin``) -- a gold_index
    aux loss supervises both the same way. A batch whose
    ``cand_addr_mask`` is ``None`` (every pre-M57b batch) or all-falsy
    reproduces the pre-M57b arithmetic exactly (regression-tested
    alongside the M53 byte-identity check, see tests/test_writeback.py).

    M57b v2 (RESEARCH_NOTES M57b, replacing the curriculum-level
    ``wrong_binding`` aux-gold-corruption arm, which task pressure simply
    overrode): ``batch.cand_forced_index`` -- per (row, step), ``-1`` = "not
    forced" -- TEACHER-FORCES the entity branch's collapse weights ``w`` to
    a one-hot at the given candidate index, REGARDLESS of the resolver's
    own logits, in BOTH train and eval mode, for addr-redirect AND
    value-redirect rows alike (generic machinery, not write-back-specific).
    The resolver's real logits/margin are still computed and recorded
    unchanged -- forcing only overrides what the collapse weights are
    applied to, never what the aux loss/eval diagnostics see. ``None``
    (every pre-v2 batch, and every batch built without
    ``writeback_force=``) is byte-identical to an all-``-1`` tensor and to
    this field being entirely absent.

    M57c.2 (RESEARCH_NOTES "M57c battery #1": instance episodes failed even
    under forced-gold collapse because a description/pronoun QUESTION step's
    read never followed the redirect -- only the WRITE did): two fixes,
    both gated so a batch that predates them is untouched.
      - Post-collapse read: on every (row, step) where ``_collapse``'s
        entity branch just redirected the address (its new ``addr_row``
        return value), :meth:`forward` recomputes ``mem_read`` at the
        RESOLVED node under the step's own relation and feeds THAT into
        both the GRU input and the response head, replacing the pre-collapse
        placeholder address's reading. Applies uniformly to statement and
        question steps. A row with no redirect this step (``addr_row``
        ``None``/all-``False`` -- every pre-M57c.2 batch) keeps the original
        pre-collapse ``mem_read``, byte-identical to before.
      - Entity-axis inverse read: ``batch.inverse_mask`` (truthy at an
        inverse-query "who is {trait} ?" step, which has no memory address
        at all) swaps ``mem_read`` for
        :func:`nsm_ct.entity_memory.query_entity`'s entity-axis unbind of
        THAT step's own (relation, value) instead of the ordinary
        address-keyed :func:`~nsm_ct.entity_memory.query`. ``None``/all-
        falsy (every batch without an instance inverse-query episode) is a
        no-op.

    M59a (episodic LTM, CLAUDE.md's "LTM decisions" / dev/LTM_DESIGN_BRIEF.md
    Sec.5): :meth:`forward`'s new optional ``ltm`` argument -- when set, every
    READ in this class (the step ``mem_read``, the post-collapse re-read
    above, the entity-axis inverse read, and every read INSIDE this method:
    ``query_candidates``'s candidate-evidence reads and
    ``query_candidates_per_addr``'s per-hypothesis reads) queries
    :func:`nsm_ct.ltm.mem_total`'s ``memory + ltm`` view instead of
    ``memory`` alone -- :meth:`forward` computes this ONCE per step and
    passes the COMBINED tensor into this method AS ``memory``, so nothing
    in this method's own body needs to change; every ``memory``-reading
    line above already reads the LTM-inclusive view for free. WRITES
    (``forward``'s own ``em.write`` call, downstream of this method) are
    UNCHANGED -- they still land only in STM, never in ``ltm`` itself, per
    the locked "STM-only writes" design. ``ltm=None`` (the default) makes
    ``mem_total`` return ``memory`` unchanged, so this method's behavior is
    byte-identical to pre-M59a whenever no LTM tensor is threaded in.

    M59a's SECOND, independent addition here: ``batch.cand_from_ltm`` (see
    that field's own docstring) widens the entity branch's per-candidate
    feature register with one more column (the **link** op's scorer
    feature) via the SAME ``resolver.cand_feature_extra``-driven concat
    path M57c.3's ``evidence_interaction`` scalar already uses -- see the
    ``extra_cols`` block below.
    """

    def __init__(self, dim: int, hidden: int = 128, resolver: Optional[Resolver] = None,
                 sense_resolver: Optional[Resolver] = None,
                 hyp_resolver: Optional[Resolver] = None,
                 evidence_prior_beta: Optional[float] = None,
                 cleanup: bool = False) -> None:
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
        # M57c.3 (RESEARCH_NOTES "M57c battery #2", CLAUDE.md invariant #6
        # "dials are explicit named scalars"): the DETERMINISTIC "perception
        # never guesses" structural-prior option -- when set, the entity
        # branch's ``cand_prior`` gets multiplied by
        # ``softmax(evidence_interaction(...) * evidence_prior_beta)`` before
        # the resolver ever sees it (see ``_collapse``'s own paragraph).
        # Default ``None`` -- byte-identical to every pre-M57c.3 forward:
        # the whole block is skipped whenever this is ``None``.
        self.evidence_prior_beta = evidence_prior_beta
        # M60 (op-library integration -- CLEANUP, dev/OP_LIBRARY_MAP.md's
        # ``cleanup`` row / dev/OP_INVENTORY.md's "caution never gates
        # anything" gap): when ``True`` AND the model is in EVAL mode
        # (``not self.training`` -- "at eval only", never during training),
        # ``forward`` runs :func:`nsm_ct.ops.cleanup` over the final
        # response vector against each row's own options codebook and
        # reports the top1-top2 margin + a :data:`nsm_ct.ops.CAUTION`-gated
        # abstain flag ALONGSIDE the existing ``answer_logits`` argmax --
        # never in place of it (the argmax over options already IS the
        # answer; ``cleanup`` never changes it, see ``forward``'s own
        # paragraph). Default ``False`` -- byte-identical to every
        # pre-M60 forward: the whole block below is skipped whenever this
        # is ``False``.
        self.cleanup = cleanup

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
                  e: torch.Tensor, r: torch.Tensor, v: torch.Tensor, batch: ClauseBatch, t: int):
        """M53b/M54/M57b collapse step for step ``t``: resolves ENTITY
        candidates (M53b/M57b) and then SENSE candidates (M54) against the
        (possibly entity-collapsed) value, plus M55a's hypothesis branch.
        The candidate kinds are always on DISJOINT (row, step) entries in
        every curriculum (an ambiguity episode's step never also carries a
        pronoun candidate set), so applying them in sequence is safe: each
        only touches rows where its own ``has_cand`` is true, via
        ``torch.where``. Returns ``(entity, value, ent_logits, ent_margin,
        sense_logits, sense_margin, hyp_logits, hyp_margin, addr_row,
        resolved_idx)`` -- any of the middle six is ``None`` exactly when the
        corresponding resolver/candidate data is absent (mirrors the pre-M54
        ``(v, None, None)`` contract for the no-resolver case component-wise);
        ``entity`` is ``e`` UNCHANGED whenever no address-redirect row
        applies (see the ENTITY branch below and the M57b class-docstring
        paragraph).
        ``addr_row`` (M57c.2, a ninth element) is the ``[B]`` bool mask
        of rows whose address was JUST redirected at this step (``None``
        whenever the entity branch didn't run at all, i.e. no resolver or no
        ``batch.cand_mask`` -- an ALL-``False`` tensor, not ``None``, when the
        branch ran but ``cand_addr_mask`` is absent/falsy) -- consumed by
        :meth:`ClauseReactor.forward`'s post-collapse read (RESEARCH_NOTES
        "M57c battery #1": a description/pronoun QUESTION step read memory
        at the PRE-collapse placeholder address, so the doctor's own node was
        never actually read even after a correct redirect).
        ``resolved_idx`` (M57d, PROVENANCE wiring, a tenth element) is the
        ``[B]`` long candidate index the entity branch's collapse weights
        ``w`` ACTUALLY APPLIED this step (``w.argmax(-1)`` where
        ``has_cand``, ``-1`` elsewhere) -- ``None`` under the same "entity
        branch didn't run" condition as ``addr_row``. Computed from ``w``
        AFTER ``cand_forced_index``'s override (unlike ``ent_logits``, which
        stays the resolver's raw, pre-force prediction by design -- see the
        M57b v2 paragraph above): coincides exactly with
        ``ent_logits.argmax(-1)`` whenever this step isn't forced (softmax,
        and the eval-mode hard-argmax collapse, are both order-preserving
        over logits), and reflects the FORCED index instead wherever
        forcing overrode ``w`` -- the audit trail
        (:mod:`nsm_ct.provenance`) needs what the write ACTUALLY did, not
        an untrained (or, under forcing, entirely bypassed) resolver's raw
        guess.

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
        addr_row_out = None    # M57c.2: [B] bool, which rows got address-redirected THIS step
        resolved_idx_out = None    # M57d: [B] long, the ACTUALLY APPLIED candidate index (post-force)
        if self.resolver is not None and batch.cand_mask is not None:
            ce_t, cf_t = batch.cand_entity[:, t], batch.cand_feature[:, t]
            cp_t, cm_t = batch.cand_prior[:, t], batch.cand_mask[:, t]
            # M57c: a per-(row, step) EVIDENCE relation (e.g. attr:kind for
            # "the doctor", attr:gender for "she") REPLACES the clause's own
            # step relation `r` for this read only, when present -- see
            # ClauseBatch.cand_evidence_relation's docstring. `r` unchanged
            # (None everywhere) reproduces the pre-M57c read exactly, the
            # fourth optional field guarded this way (cand_addr_mask,
            # cand_forced_index, and now this one). The fallback is PER
            # (row, step), not per batch: the build tensor is zero wherever a
            # candidate set carries no evidence relation (M53a/M53b/M57b
            # episodes mixed into the same batch as instance episodes), and a
            # zero relation would zero those rows' candidate readouts --
            # so rows without an evidence vector keep the step relation.
            if batch.cand_evidence_relation is not None:
                er_t = batch.cand_evidence_relation[:, t]
                has_er = er_t.norm(dim=-1, keepdim=True) > 0
                evidence_r = torch.where(has_er, er_t, r)
            else:
                evidence_r = r
            cand_mem_read = query_candidates(memory, ce_t, evidence_r)           # [B, C, d]
            # M57c.3 (RESEARCH_NOTES "M57c battery #2" -- the resolver never
            # had the referring expression's own TARGET to compare a
            # candidate's evidence readout against, so instance binding sat
            # at chance): per-candidate ``cos(cand_mem_read_c, target)`` --
            # see nsm_ct.resolver.evidence_interaction and
            # ClauseBatch.cand_evidence_target's docstring. ``s_c`` is
            # ``None`` (this whole feature inert) whenever the batch carries
            # NO evidence-target vectors at all -- byte-identical to
            # pre-M57c.3 for every such batch (every batch built before this
            # milestone, and every M53a/M53b/M57b pronoun/writeback-only
            # batch mixed alongside instance/rich episodes that DO carry
            # one -- the per-(row, step) zero-vector floor in
            # ``evidence_interaction`` handles rows without one).
            s_c = None
            if batch.cand_evidence_target is not None:
                et_t = batch.cand_evidence_target[:, t]                          # [B, d]
                s_c = evidence_interaction(cand_mem_read, et_t).unsqueeze(-1)     # [B, C, 1]
            # M57c.3 (--evidence-prior, scripts/train_instances.py): the
            # deterministic "perception never guesses" structural-prior
            # option -- multiplies ``cand_prior`` by
            # ``softmax(s_c * evidence_prior_beta)`` over the REAL
            # candidates only, BEFORE the resolver (either track) ever sees
            # it. The learned head can still override this (it is only ONE
            # of the resolver's inputs, never a hard mask). Inert (byte-
            # identical) unless BOTH ``s_c`` exists AND
            # ``self.evidence_prior_beta`` is set (``None`` by default).
            if s_c is not None and self.evidence_prior_beta is not None:
                boost_logits = (s_c.squeeze(-1) * self.evidence_prior_beta).masked_fill(cm_t <= 0, -1e9)
                cp_t = cp_t * torch.softmax(boost_logits, dim=-1)
            # M56b: pass the per-candidate feature register (§1.8) ONLY to a
            # resolver that opted in (`use_cand_feature=True` -- CorefHead
            # only today); SharedScorer/SenseHead are never called with this
            # kwarg at all, so "Do NOT change SharedScorer" holds literally
            # (zero edits, zero new call shape) for every non-opted-in track.
            extra = {}
            if getattr(self.resolver, "use_cand_feature", False) and batch.cand_feature_per_candidate is not None:
                extra["cand_feature_per_candidate"] = batch.cand_feature_per_candidate[:, t]
            # M57c.3: widen the register by ``resolver.cand_feature_extra``
            # columns (CorefHead only -- 0 for every other resolver/every
            # pre-M57c.3 CorefHead, so this is a no-op there) with the
            # interaction feature computed above. Building a fresh zero
            # register when no cand_feature_per_candidate exists yet mirrors
            # CorefHead.forward's own defensive-zeros fallback exactly.
            # M59a (episodic LTM): a SECOND optional extra column,
            # ``batch.cand_from_ltm`` (0/1 per candidate, "link" op's
            # scorer feature -- see nsm_ct.ltm's module docstring and
            # membrane.EntityCandidateSet.from_ltm) -- concatenated
            # alongside ``s_c`` (M57c.3's interaction scalar) rather than
            # replacing it, so a resolver can see both. Whichever of the
            # two extras is present this batch is gathered into
            # ``extra_cols`` and concatenated onto ``cfpc`` in ONE cat,
            # widened/truncated to ``resolver.cand_feature_extra`` (a plain
            # int, unchanged in resolver.py -- CorefHead's constructor
            # already takes an arbitrary width; this is the caller-side
            # generalization "keep it parameterized" asks for). A batch
            # with NEITHER extra (every pre-M59a batch, and every M57c.3
            # batch with no evidence_target) leaves ``extra_cols`` empty --
            # byte-identical no-op, same as before this milestone.
            extra_cols = []
            if s_c is not None:
                extra_cols.append(s_c)                                       # [B, C, 1]
            if batch.cand_from_ltm is not None:
                extra_cols.append(batch.cand_from_ltm[:, t].unsqueeze(-1).to(ce_t.dtype))  # [B, C, 1]
            # M60 (op-library integration -- RECENCY): a THIRD optional
            # extra-column group, the deterministic centering-theory
            # salience features (see ClauseBatch.cand_recency's own
            # docstring) -- concatenated alongside s_c/cand_from_ltm rather
            # than replacing either, same "gather whichever extras this
            # batch actually carries" discipline established above.
            if batch.cand_recency is not None:
                extra_cols.append(batch.cand_recency[:, t].to(ce_t.dtype))    # [B, C, 3]
            extra_width = getattr(self.resolver, "cand_feature_extra", 0)
            if extra_width > 0 and extra_cols:
                cfpc = extra.get("cand_feature_per_candidate")
                if cfpc is None:
                    b_, C_, _d_ = ce_t.shape
                    cfpc = ce_t.new_zeros(b_, C_, FEATURE_DIM)
                stacked_extra = torch.cat(extra_cols, dim=-1)                # [B, C, k]
                k = stacked_extra.shape[-1]
                if k < extra_width:
                    pad = ce_t.new_zeros(*stacked_extra.shape[:-1], extra_width - k)
                    stacked_extra = torch.cat([stacked_extra, pad], dim=-1)
                elif k > extra_width:
                    stacked_extra = stacked_extra[..., :extra_width]
                extra["cand_feature_per_candidate"] = torch.cat([cfpc, stacked_extra], dim=-1)
            logits = self.resolver(ce_t, cf_t, cp_t, cm_t, cand_mem_read, state, **extra)  # [B, C]
            logits = logits.masked_fill(cm_t <= 0, -1e9)
            has_cand = cm_t.sum(-1) > 0                                          # [B]
            w = self._collapse_weights(logits, self.training)
            # M57b (v2, honest validity machinery -- replaces corrupting the
            # curriculum's own gold_index, which task pressure could simply
            # override): ``cand_forced_index`` (per (row, step), ``-1`` =
            # "not forced") TEACHER-FORCES the collapse weights to a one-hot
            # at the given candidate index, REGARDLESS of the resolver's own
            # logits -- identically in train and eval mode, and identically
            # for addr-redirect and value-redirect rows (this override
            # happens before the addr/value split below, so both branches
            # see the SAME forced ``w``). ``logits``/``ent_margin`` below are
            # still computed from the resolver's REAL (unforced) prediction
            # -- forcing only changes what ``w`` gets applied to, not what is
            # recorded for the aux loss / eval diagnostics.
            if batch.cand_forced_index is not None:
                forced_t = batch.cand_forced_index[:, t]                        # [B] long, -1 = not forced
                forced_valid = has_cand & (forced_t >= 0)
                C = w.shape[-1]
                forced_onehot = F.one_hot(forced_t.clamp(min=0), num_classes=C).to(w.dtype)
                w = torch.where(forced_valid.unsqueeze(-1), forced_onehot, w)
            # M57d (PROVENANCE wiring): the candidate index THIS step's
            # collapse actually applied -- ``w``'s own argmax, AFTER the
            # forced-index override above, not the resolver's raw (possibly
            # forced-overridden-away) ``logits``. Coincides exactly with
            # ``logits.argmax(-1)`` whenever ``cand_forced_index`` is absent/
            # not-forced-here (softmax and the eval-mode hard-argmax collapse
            # are both order-preserving over ``logits``), and reflects the
            # FORCED index instead wherever forcing overrode ``w`` above --
            # an audit trail must record what the write actually did, not an
            # untrained resolver's raw (and, under forcing, entirely
            # bypassed) guess. See nsm_ct.provenance.record_writes, the sole
            # consumer (via ClauseReactor.forward's ``return_write_trace``).
            resolved_idx_out = torch.where(
                has_cand, w.argmax(-1), torch.full_like(has_cand, -1, dtype=torch.long))
            resolved_v = (w.unsqueeze(-1) * cand_mem_read).sum(1)                # [B, d]
            # M57b: the SAME weights ``w`` also collapse the candidate ATOMS
            # themselves (not their memory readout -- a candidate's identity
            # IS the address it stands for) into a candidate ADDRESS.
            # ``cand_addr_mask`` (per (row, step), None/falsy for every
            # pre-M57b batch and every M53a/M53b value-redirect row) decides,
            # per row, whether THIS step's collapse redirects the value
            # (unchanged) or the entity/address (new) -- never both, and
            # never neither when ``has_cand`` is true.
            resolved_e = (w.unsqueeze(-1) * ce_t).sum(1)                         # [B, d]
            if batch.cand_addr_mask is not None:
                addr_row = has_cand & (batch.cand_addr_mask[:, t] > 0)
            else:
                addr_row = torch.zeros_like(has_cand)
            value_row = has_cand & ~addr_row
            v = torch.where(value_row.unsqueeze(-1), resolved_v, v)
            e = torch.where(addr_row.unsqueeze(-1), resolved_e, e)
            ent_logits, ent_margin = logits, self._top2_margin(logits, has_cand, v)
            addr_row_out = addr_row

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

        return (e, v, ent_logits, ent_margin, sense_logits, sense_margin, hyp_logits, hyp_margin,
                addr_row_out, resolved_idx_out)

    def forward(self, batch: ClauseBatch, return_memory: bool = False,
                return_mem_read: bool = False, return_write_trace: bool = False,
                ltm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """See the class docstring's M59a paragraph for ``ltm``'s full
        contract. ``ltm`` is an optional ``[B, d, d, d]`` per-row long-term
        memory tensor (one document's persisted LTM slice per row --
        ``scripts._train_common.DocumentRunner`` is the intended caller).
        When set, every read this method performs -- directly, and inside
        :meth:`_collapse` -- queries :func:`nsm_ct.ltm.mem_total`'s ``memory
        + ltm`` view; every write still lands only in STM's ``memory``
        (unchanged, per the locked "STM-only writes" design). ``ltm=None``
        (the default) is byte-identical to pre-M59a ``forward()`` --
        regression-tested in ``tests/test_ltm.py``.
        """
        b, T, d = batch.entity.shape
        device = batch.entity.device
        state = torch.zeros(b, self.gru.hidden_size, device=device)
        memory = em.init_memory(b, d, device)

        coord = batch._coord()
        have_resolver_data = self.resolver is not None and batch.cand_mask is not None
        have_sense_data = self.sense_resolver is not None and batch.sense_cand_mask is not None
        have_hyp_data = self.hyp_resolver is not None and batch.hyp_cand_mask is not None
        # M60 (op-library integration -- inverse-read routing, LOCKED DESIGN
        # item 3): accumulates the entity-axis inverse readout at whichever
        # step ``batch.inverse_mask`` flags this row (see the entity-axis
        # inverse read below, unchanged) -- ``None`` whenever the batch has
        # no inverse-query step at all, byte-identical no-op then.
        inverse_readout = torch.zeros(b, d, device=device) if batch.inverse_mask is not None else None
        resp_logits, resp_vecs = [], []
        resolver_logits_all, resolver_margin_all = [], []
        sense_logits_all, sense_margin_all = [], []
        hyp_logits_all, hyp_margin_all = [], []
        mem_read_all = []
        # M57d (PROVENANCE wiring, CLAUDE.md's M57 memory-schema decision):
        # per-step write-trace accumulators, only ever consumed when
        # ``return_write_trace`` is True (see the end of this method) --
        # appending to a list nobody reads back is a no-op on every other
        # output, which is exactly what keeps ``return_write_trace=False``
        # (the default) byte-identical to pre-M57d forward().
        gate_trace: List[torch.Tensor] = []
        overwrite_trace: List[torch.Tensor] = []
        neg_trace: List[torch.Tensor] = []
        redirected_trace: List[torch.Tensor] = []
        resolved_idx_trace: List[torch.Tensor] = []
        for t in range(T):
            e, r, v = batch.entity[:, t], batch.relation[:, t], batch.value[:, t]
            p, c = batch.pred[:, t], coord[:, t]
            real, isq = batch.mask[:, t], batch.is_q[:, t]
            # M59a: the additive STM+LTM read view (nsm_ct.ltm.mem_total,
            # the "recall" op) -- computed ONCE per step and reused for
            # EVERY read this step (mem_read, the post-collapse re-read
            # below, and every read inside _collapse), never re-added per
            # read site. `ltm=None` (every pre-M59a call) makes this
            # exactly `memory`, byte-identical to before.
            mem_total_t = mem_total(memory, ltm)
            mem_read = em.query(mem_total_t, e, r)                     # [B, d] -- the pre-collapse address's reading
            (e, v, res_logits_t, res_margin_t, sense_logits_t, sense_margin_t,
             hyp_logits_t, hyp_margin_t, addr_row_t, resolved_idx_t) = self._collapse(
                mem_total_t, state, mem_read, e, r, v, batch, t)
            # M57c.2 (RESEARCH_NOTES "M57c battery #1" -- the measured gap:
            # a description/pronoun QUESTION step's read never reached the
            # resolved node, only the write did): `e` may now be the
            # resolver's redirected address (see ClauseReactor's own
            # docstring). On exactly the rows THIS step redirected
            # (`addr_row_t`), recompute `mem_read` AT that resolved node
            # under the step's own relation `r` and use THAT for the GRU
            # input and the response head below, instead of the pre-collapse
            # placeholder address's (almost always empty) reading -- applies
            # to statement AND question steps alike (the GRU should see the
            # content already at the resolved node before deciding whether
            # to overwrite it, too). Rows with no redirect this step keep the
            # original `mem_read` untouched: `addr_row_t` is `None` whenever
            # the entity/M57b branch never ran (no resolver, or no
            # `batch.cand_mask`), and an all-`False` tensor whenever it ran
            # but `cand_addr_mask` is absent/falsy -- both leave `mem_read`
            # byte-identical to before M57c.2.
            if addr_row_t is not None and bool(addr_row_t.any()):
                mem_read = torch.where(addr_row_t.unsqueeze(-1), em.query(mem_total_t, e, r), mem_read)
            # M57c.2: entity-axis inverse read. An inverse-query question
            # step ("who is tall ?") has no memory ADDRESS at all -- its own
            # "entity" is a fixed "who" marker atom, never written to -- so
            # there is nothing for `em.query` to usefully read. What it needs
            # instead is the ENTITY unbound from THIS step's own (relation,
            # value) via :func:`nsm_ct.entity_memory.query_entity`'s
            # entity-axis einsum. `batch.inverse_mask` is `None`/all-falsy
            # for every batch without an InstanceCurriculumGenerator
            # inverse-query episode, leaving `mem_read` untouched.
            if batch.inverse_mask is not None:
                inv_row = batch.inverse_mask[:, t] > 0
                if bool(inv_row.any()):
                    inv_readout_t = em.query_entity(mem_total_t, r, v)
                    mem_read = torch.where(inv_row.unsqueeze(-1), inv_readout_t, mem_read)
                    # M60 (inverse-read routing): stash the SAME entity-axis
                    # readout for the direct-similarity route below -- no
                    # extra memory access, just captured alongside the
                    # existing GRU-input override.
                    inverse_readout = torch.where(inv_row.unsqueeze(-1), inv_readout_t, inverse_readout)
            if return_mem_read:
                mem_read_all.append(mem_read)
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
            if return_write_trace:
                # M57d write trace: gate/overwrite/neg are exactly the
                # values just fed to em.write above (the write this step
                # ACTUALLY performed); redirected/resolved_index describe
                # the entity branch's collapse -- see forward()'s own
                # docstring paragraph, _collapse's resolved_idx_out
                # paragraph, and nsm_ct.provenance.record_writes (the sole
                # consumer). Both default to False/-1 whenever the entity
                # branch never ran at all (no resolver, or no
                # batch.cand_mask) -- _collapse returns None for both in
                # that case.
                gate_trace.append(gate)
                overwrite_trace.append(owr)
                neg_trace.append(neg)
                redirected_trace.append(
                    addr_row_t if addr_row_t is not None else torch.zeros(b, dtype=torch.bool, device=device))
                resolved_idx_trace.append(
                    resolved_idx_t if resolved_idx_t is not None
                    else torch.full((b,), -1, dtype=torch.long, device=device))
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
        # M60 (op-library integration -- inverse-read routing, LOCKED DESIGN
        # item 3): a SECOND, independent answer route for inverse-query
        # steps -- direct :func:`nsm_ct.ops.similarity` between the entity-
        # axis readout captured above and each row's own option atoms, NOT
        # threaded back into ``answer_logits``/the learned response head at
        # all (no change to the learned path). ``None`` whenever the batch
        # has no inverse-query step (``inverse_readout is None``), the same
        # "only if present" discipline every optional output here follows.
        if inverse_readout is not None:
            out["inverse_direct_logits"] = ops.similarity(
                inverse_readout.unsqueeze(1), batch.options).cosine     # [B, K]
        # M60 (op-library integration -- CLEANUP, LOCKED DESIGN item 2): at
        # EVAL only (``not self.training``), run :func:`nsm_ct.ops.cleanup`
        # per row (each row has its OWN options codebook, unlike ``cleanup``'s
        # single-shared-codebook signature) over the SAME response vector
        # ``r``/options ``batch.options`` the contrastive answer above
        # already scores -- the argmax over options IS the answer already
        # (``cleanup_index`` coincides with ``answer_logits.argmax(-1)`` up
        # to float rounding, asserted in tests/test_ops_integration.py); the
        # value here is ``cleanup_margin``/``cleanup_abstain``, the
        # :data:`nsm_ct.ops.CAUTION`-gated "the mind may abstain" dial made
        # real (dev/OP_INVENTORY.md's "caution never gates anything" gap).
        # Default ``self.cleanup=False`` skips this block entirely --
        # byte-identical to pre-M60 ``out`` for every existing caller.
        if self.cleanup and not self.training:
            idxs, margins, abstains = [], [], []
            for i in range(b):
                idx_i, _clean_i, margin_i, abstain_i = ops.cleanup(r[i], batch.options[i])
                idxs.append(idx_i)
                margins.append(margin_i)
                abstains.append(abstain_i)
            out["cleanup_index"] = torch.stack(idxs)
            out["cleanup_margin"] = torch.stack(margins)
            out["cleanup_abstain"] = torch.stack(abstains)
        # M57b test seam: the final post-episode memory tensor, for tests
        # that need to verify WHERE a write actually landed (e.g. an
        # address-redirect: querying the resolved node vs. the pronoun's own
        # placeholder address). Deliberately NOT part of the default output
        # dict (``return_memory`` defaults False) -- no training/eval script
        # reads it, this is purely a test-observability seam.
        if return_memory:
            out["_memory"] = memory
        # M57c.2 test seam (mirrors ``return_memory`` exactly): the per-step
        # ``mem_read`` actually fed into the GRU/response head, AFTER the
        # post-collapse recompute / entity-axis inverse override -- lets a
        # test check directly that a redirected question step reads the
        # RESOLVED node, not the pre-collapse placeholder. Deliberately NOT
        # part of the default output dict (``return_mem_read`` defaults
        # False) -- no training/eval script reads it.
        if return_mem_read:
            out["_mem_read"] = torch.stack(mem_read_all, dim=1)   # [B, T, d]
        # M57d test/provenance seam (mirrors return_memory/return_mem_read
        # exactly): per-step write bookkeeping -- gate/overwrite/neg are the
        # SAME values just fed to every em.write call this pass, redirected/
        # resolved_index describe the entity branch's collapse. Deliberately
        # NOT part of the default output dict (``return_write_trace``
        # defaults False, so this whole block is inert and the rest of
        # ``out`` -- and every arithmetic path above -- is byte-identical to
        # pre-M57d forward()); nsm_ct.provenance.record_writes is the sole
        # consumer.
        if return_write_trace:
            out["_write_trace"] = {
                "gate": torch.stack(gate_trace, dim=1),                    # [B, T] float
                "overwrite": torch.stack(overwrite_trace, dim=1),          # [B, T] float
                "neg": torch.stack(neg_trace, dim=1),                      # [B, T] float
                "redirected": torch.stack(redirected_trace, dim=1),        # [B, T] bool
                "resolved_index": torch.stack(resolved_idx_trace, dim=1),  # [B, T] long, -1 = no candidate set
            }
        return out
