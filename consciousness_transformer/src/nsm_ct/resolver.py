"""M53b: the resolver contract (dev/MIND_INTERFACE.md §3, dev/RESOLVER_BUILD_PLAN.md
Phase 2 "Agent 3") -- Track A (:class:`CorefHead`, a coref-specialized module) and
Track B (:class:`SharedScorer`, one shared collapse mechanism meant to generalize to
senses/parse-hypotheses in M54/M55) behind ONE interface, so
:class:`nsm_ct.clause_reactor.ClauseReactor` can swap between them with zero model
changes beyond the constructor argument.

Contract (:class:`Resolver`, both tracks implement it identically): given, for ONE
clause step ``t``, batched over episodes ``B`` with a small candidate-set size ``C``
(``nsm_ct.membrane``'s v1 "entity" row -- pronoun antecedents today, sense handles /
parse hypotheses reusing the same shape later):

  - ``cand_entity``  ``[B, C, d]``  candidate atoms' grounded vectors (0-padded)
  - ``cand_feature`` ``[B, F]``     the MENTION's deterministic feature vector
                                    (``membrane.FEATURE_DIM``; shared across every
                                    candidate in the set -- it describes the pronoun/
                                    sense-bearing word being resolved, not any one
                                    candidate)
  - ``cand_prior``   ``[B, C]``     structural prior (uniform in v1)
  - ``cand_mask``    ``[B, C]``     1 = real candidate, 0 = padding
  - ``mem_read``     ``[B, C, d]``  entity-memory readout PER CANDIDATE:
                                    ``em.query(memory, cand_entity[:, j], relation)``
                                    for the clause's current relation, stacked over
                                    ``j`` -- :func:`query_candidates` is the shared
                                    helper both the reactor and tests use for this
                                    loop (``C`` is a handful of discourse entities;
                                    a Python loop over it is fine, per the build plan)
  - ``state``        ``[B, H]``     controller state BEFORE this step's GRU update
                                    (the collapse happens before the write --
                                    MIND_INTERFACE.md §3 point 3)

and produce RAW logits ``[B, C]`` (unmasked -- the caller applies ``cand_mask`` and
softmaxes/argmaxes; keeping that outside the contract is what lets both tracks stay
agnostic to how invalid padding is neutralized).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import entity_memory as em
from .membrane import FEATURE_DIM

__all__ = ["Resolver", "CorefHead", "SharedScorer", "SenseHead", "query_candidates",
           "make_resolver", "make_sense_resolver", "shared_scorer_for_budget"]


def query_candidates(memory: torch.Tensor, cand_entity: torch.Tensor,
                      relation: torch.Tensor) -> torch.Tensor:
    """``em.query`` per candidate slot, looped over ``C`` (small -- a handful of
    discourse entities per v1 candidate set): ``[B, C, d]``.

    Shared by :class:`nsm_ct.clause_reactor.ClauseReactor` (the one call site that
    matters) and by tests, so the "loop over C is fine" design decision
    (dev/RESOLVER_BUILD_PLAN.md Phase 2) lives in exactly one place.
    """
    b, C, d = cand_entity.shape
    if C == 0:
        return cand_entity.new_zeros(b, 0, d)
    return torch.stack([em.query(memory, cand_entity[:, j], relation) for j in range(C)], dim=1)


class Resolver(nn.Module):
    """Abstract v1 resolver contract. Both tracks below implement this exact
    signature; :class:`nsm_ct.clause_reactor.ClauseReactor` calls it uniformly and
    never branches on which track is installed."""

    def forward(self, cand_entity: torch.Tensor, cand_feature: torch.Tensor,
                cand_prior: torch.Tensor, cand_mask: torch.Tensor,
                mem_read: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # M56b: CorefHead ALSO accepts an optional trailing
        # `cand_feature_per_candidate` kwarg ([B, C, FEATURE_DIM], each
        # candidate's own feature vector) when constructed with
        # `use_cand_feature=True` -- not part of this base signature (kept
        # untouched here, and SharedScorer/SenseHead are never called with
        # it) because the caller (ClauseReactor._collapse) only passes it to
        # resolvers that opt in via `getattr(resolver, "use_cand_feature",
        # False)`, so Track B and SenseHead are unaffected byte-for-byte.
        raise NotImplementedError


class CorefHead(Resolver):
    """Track A -- a coref-SPECIALIZED small MLP.

    Design: per candidate ``c``, score
    ``MLP([cand_entity_c ; mem_read_c ; mention_feature ; prior_c])`` -- one shared
    set of weights applied to every candidate slot (so param count is independent of
    ``C``: the candidate-set size can grow with no new parameters). Deliberately
    does NOT see the controller ``state`` -- Track A is the "distinct, local-match"
    functionality baseline: it only has to compare the mention's own feature vector
    (gender/person/number, ``nsm_ct.membrane.mention_feature_vector``) against each
    candidate's identity atom and what memory currently holds for that candidate.
    ``in_dim = 2*dim + FEATURE_DIM + 1``; with ``dim=32`` that's 71, hidden=24 ->
    1,753 params (well under the 20k budget; scales as O(dim), not O(dim^2)).

    M56b (`dev/TRACK_C_DESIGN.md` §1.8, RESEARCH_NOTES M56): the membrane's v1
    candidate-set contract carried exactly one feature slot -- the MENTION's,
    broadcast identically to every candidate -- so a coref head had geometric
    access to "who is being asked about" but NOT to "what is candidate c's own
    gender/person," and could only close that gap by memorizing a closed-world
    name->response lookup table (confirmed: held-out-name binding craters
    toward chance, see tests/test_resolver_cand_feature.py and
    scripts/probe_held_out_name_ablation.py). ``use_cand_feature=True`` (default
    ``False`` -- byte-identical to every pre-M56b call site, same submodule
    construction order, so seeded reproducibility is untouched) adds a SECOND
    ``FEATURE_DIM``-wide slot to the MLP's input, ``cand_feature_per_candidate``
    (each candidate's OWN feature vector, ``membrane.EntityCandidateSet.
    cand_features`` -- see that class's docstring), concatenated alongside the
    existing broadcast mention feature so the net's input is literally
    ``[cand_entity ; mem_read ; mention_feature ; cand_feature ; prior]`` --
    ``feature_match`` (§1.3) becomes geometrically POSSIBLE (the two ``Feat``
    vectors sit side by side in one linear layer's input) rather than a name
    the net has no way to compute. ``in_dim`` grows to
    ``2*dim + 2*FEATURE_DIM + 1`` only when the flag is set.
    """

    def __init__(self, dim: int, hidden: int = 24, use_cand_feature: bool = False) -> None:
        super().__init__()
        self.use_cand_feature = use_cand_feature
        in_dim = 2 * dim + FEATURE_DIM + 1 + (FEATURE_DIM if use_cand_feature else 0)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state,
                cand_feature_per_candidate=None):
        b, C, _d = cand_entity.shape
        feat = cand_feature.unsqueeze(1).expand(b, C, -1)      # [B, C, F]
        prior = cand_prior.unsqueeze(-1)                        # [B, C, 1]
        parts = [cand_entity, mem_read, feat]
        if self.use_cand_feature:
            cfpc = cand_feature_per_candidate
            if cfpc is None:                                    # defensive: no per-candidate data available
                cfpc = cand_entity.new_zeros(b, C, FEATURE_DIM)
            parts.append(cfpc)
        parts.append(prior)
        x = torch.cat(parts, dim=-1)
        return self.net(x).squeeze(-1)                          # [B, C]


class SharedScorer(Resolver):
    """Track B -- ONE shared collapse mechanism, MIND_INTERFACE.md §3's
    ``score(candidate_vec, mem_read, state) -> logit``.

    ``candidate_vec`` here is deliberately the GENERIC per-candidate
    representation the v1 contract already carries for any candidate kind
    (identity atom + the shared mention/context feature + structural prior) --
    ``cat([cand_entity_c, mention_feature, prior_c])`` -- not anything
    coref-specific, so the exact same module is meant to be reusable unchanged
    for M54 sense candidates and M55 parse hypotheses. Unlike Track A, this
    scorer DOES consult the controller ``state`` (projected down first to keep
    the parameter budget small) -- the architectural bet Track B is testing:
    does one context-conditioned mechanism reproduce Track A's coref behavior
    without a coref-specific input design? ``state`` is projected
    ``hidden -> state_proj`` (default 8) before concatenation, then one shared
    MLP scores every candidate. With ``dim=32, hidden=128, state_proj=8``:
    state projection 128*8+8=1,032 params, MLP in_dim=32+6+1+32+8=79 ->
    hidden=16 -> 1,281 params; total 2,313 (under the 20k budget).

    M54c adds ONE constructor flag, ``use_state`` (default ``True`` --
    byte-identical to every pre-M54c call site: the default path is the
    exact original code, same submodule construction order, so seeded
    reproducibility is untouched). ``use_state=False`` drops the
    ``state_proj`` submodule entirely and the MLP input narrows to
    ``[candidate_vec ; mem_read]`` -- the B-no-state diagnosis arm
    (RESEARCH_NOTES M54c) isolating whether Track B's task-accuracy damage
    (M53) comes from consulting the controller state at all, holding
    everything else about the "one shared collapse mechanism" bet fixed.
    See :func:`shared_scorer_for_budget` to pick ``mlp_hidden`` for a target
    total parameter count (capacity-matched arms) instead of inverting the
    step-function param count by hand.
    """

    def __init__(self, dim: int, hidden: int, state_proj: int = 8, mlp_hidden: int = 16,
                 use_state: bool = True) -> None:
        super().__init__()
        self.use_state = use_state
        cand_vec_dim = dim + FEATURE_DIM + 1
        if use_state:
            self.state_proj = nn.Linear(hidden, state_proj)
            in_dim = cand_vec_dim + dim + state_proj
        else:
            self.state_proj = None
            in_dim = cand_vec_dim + dim
        self.net = nn.Sequential(nn.Linear(in_dim, mlp_hidden), nn.Tanh(),
                                  nn.Linear(mlp_hidden, 1))

    def forward(self, cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state):
        b, C, _d = cand_entity.shape
        feat = cand_feature.unsqueeze(1).expand(b, C, -1)      # [B, C, F]
        prior = cand_prior.unsqueeze(-1)                        # [B, C, 1]
        candidate_vec = torch.cat([cand_entity, feat, prior], dim=-1)   # [B, C, cand_vec_dim]
        if self.use_state:
            s = self.state_proj(state).unsqueeze(1).expand(b, C, -1)    # [B, C, state_proj]
            x = torch.cat([candidate_vec, mem_read, s], dim=-1)
        else:
            x = torch.cat([candidate_vec, mem_read], dim=-1)
        return self.net(x).squeeze(-1)                          # [B, C]


def shared_scorer_for_budget(dim: int, hidden: int, target_params: int, *,
                              use_state: bool = True, state_proj: int = 8) -> SharedScorer:
    """M54c: build a :class:`SharedScorer` whose total parameter count is the
    closest achievable -- by varying ``mlp_hidden`` alone, everything else
    fixed -- to ``target_params``. Used to construct the capacity-matched B
    arms (RESEARCH_NOTES M54c): B-wide and B-nostate-wide are matched to
    Track A's CorefHead + SenseHead combined at whatever ``dim`` the caller
    is using; B-nostate (narrow) is matched to the original default
    ``SharedScorer(dim, hidden)``'s own param count. Total params are a step
    function of the integer ``mlp_hidden``, so exact equality generally isn't
    achievable -- this searches upward from ``mlp_hidden=1`` (param count is
    monotonically increasing in it) and returns whichever of the two widths
    straddling the target lands closer, so callers can just read the
    constructed module's own ``sum(p.numel() for p in ...parameters())``
    to report the exact count actually achieved rather than assuming the
    target was hit precisely.
    """
    def n_params(m: int) -> int:
        return sum(p.numel() for p in
                    SharedScorer(dim, hidden, state_proj=state_proj, mlp_hidden=m,
                                 use_state=use_state).parameters())

    m = 1
    prev = n_params(m)
    while prev < target_params:
        m += 1
        cur = n_params(m)
        if cur >= target_params:
            best_m = m if (cur - target_params) <= (target_params - prev) else m - 1
            return SharedScorer(dim, hidden, state_proj=state_proj, mlp_hidden=best_m,
                                 use_state=use_state)
        prev = cur
    return SharedScorer(dim, hidden, state_proj=state_proj, mlp_hidden=m, use_state=use_state)


def make_resolver(track: str, dim: int, hidden: int = 128, *, use_cand_feature: bool = False) -> Resolver:
    """Convenience factory: ``track`` is ``"A"`` (:class:`CorefHead`) or ``"B"``
    (:class:`SharedScorer`, needs the controller's ``hidden`` size for its state
    projection). Used by ``scripts/train_resolver.py``'s ``--track`` flag.
    ``use_cand_feature`` (M56b, Track A only -- ignored for "B", since "Do NOT
    change SharedScorer" is a hard constraint) forwards to
    :class:`CorefHead`'s own flag; default ``False`` keeps this factory's
    pre-M56b behavior exactly."""
    t = track.strip().upper()
    if t == "A":
        return CorefHead(dim, use_cand_feature=use_cand_feature)
    if t == "B":
        return SharedScorer(dim, hidden)
    raise ValueError(f"unknown track {track!r}, expected 'A' or 'B'")


# ---------------------------------------------------------------------------
# M54 -- sense collapse joins the same membrane (dev/RESOLVER_BUILD_PLAN.md
# Phase 3). Track A gets its OWN specialist (:class:`SenseHead`, the M34
# ``sense_chooser`` architecture ported onto this contract) since M53's
# CorefHead is coref-SPECIALIZED by design and not meant to generalize.
# Track B reuses :class:`SharedScorer` UNCHANGED -- no new class, no
# architecture change, not even a new constructor call site beyond
# ``make_sense_resolver``'s dispatch: the whole point of Track B is that the
# SAME weights (in practice, in
# ``scripts/train_resolver.py``, literally the SAME module instance --
# ``resolver is sense_resolver``) work for entities AND senses. See
# :meth:`nsm_ct.clause_reactor.ClauseReactor._collapse` for how the CALLER
# adapts sense-candidate tensors to this contract's fixed slots (the
# "thinnest possible projection" the module docstring above promises):
# ``cand_entity`` carries the candidate SENSE VECTORS directly (not entity
# atoms to look up in memory -- there is no memory address a meaning vector
# can stand in for), ``cand_feature`` is zero-filled (senses carry no
# gender/person mention feature; zero keeps :class:`SharedScorer`'s fixed
# input width intact without inventing a new one), and ``mem_read`` carries
# the memory readout at the homograph's own address PLUS (M54's reading of
# MIND_INTERFACE.md's "+ optionally the step's other-role vectors") the same
# clause's other-role token's vector, broadcast identically across every
# candidate slot -- exactly the shape both tracks already expect.
# ---------------------------------------------------------------------------
class SenseHead(Resolver):
    """Track A -- the M34 ``sense_chooser`` architecture, ported onto the
    membrane's :class:`Resolver` contract with ONE change: the M34 chooser's
    context was the mean USVS handle of the episode's other content words
    (a bag-of-words vector, no notion of ANY OTHER step); here ``mem_read``
    IS the context (the running memory readout at this homograph's address,
    plus the same-clause other-role token -- see the module note above),
    supplied already-broadcast to ``[B, C, d]`` by the caller so this class
    needs no special-casing at all.

    ``score = MLP([cand_sense_vec ; context ; cand_sense_vec * context])``
    -- literally :class:`nsm_ct.sense_chooser.SenseChooser`'s
    ``3*d -> hidden -> 1`` shape (see that module's docstring), scored per
    candidate with ONE shared set of weights (param count independent of
    candidate-set size ``C``, same as :class:`CorefHead`). Deliberately
    ignores ``cand_feature``, ``cand_prior``, and ``state`` -- exactly like
    the M34 chooser (candidate + context only, no frequency prior, no
    running thought-state) and like :class:`CorefHead`'s "no controller
    state" specialist design (Track A's whole argument is that a narrow,
    local-match head suffices; whatever it can't see is the point being
    tested, not an oversight -- see ``dev/RESOLVER_BUILD_PLAN.md`` Phase 3's
    A-vs-B framing). With ``dim=32, hidden=32``: ``in_dim=96`` ->
    ``96*32+32=3,104`` -> ``32*1+1=33``; total 3,137 params (well under the
    20k budget, and smaller than M34's own 24.7k because ``dim`` here is 32
    vs M34's ``d=64``).
    """

    def __init__(self, dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state):
        feats = torch.cat([cand_entity, mem_read, cand_entity * mem_read], dim=-1)
        return self.net(feats).squeeze(-1)                     # [B, C]


def make_sense_resolver(track: str, dim: int, hidden: int = 128) -> Resolver:
    """M54's factory, mirroring :func:`make_resolver`: ``track`` "A" ->
    :class:`SenseHead` (a NEW specialist, distinct from :class:`CorefHead`),
    "B" -> :class:`SharedScorer` (the SAME class Track B always was -- callers
    that want the literal "one shared scorer for everything" experiment pass
    the SAME instance to both ``ClauseReactor(resolver=..., sense_resolver=...)``
    slots; this factory just constructs one when a caller wants a fresh,
    independent one instead)."""
    t = track.strip().upper()
    if t == "A":
        return SenseHead(dim)
    if t == "B":
        return SharedScorer(dim, hidden)
    raise ValueError(f"unknown track {track!r}, expected 'A' or 'B'")
