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

__all__ = ["Resolver", "CorefHead", "SharedScorer", "query_candidates", "make_resolver"]


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
    """

    def __init__(self, dim: int, hidden: int = 24) -> None:
        super().__init__()
        in_dim = 2 * dim + FEATURE_DIM + 1
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state):
        b, C, _d = cand_entity.shape
        feat = cand_feature.unsqueeze(1).expand(b, C, -1)      # [B, C, F]
        prior = cand_prior.unsqueeze(-1)                        # [B, C, 1]
        x = torch.cat([cand_entity, mem_read, feat, prior], dim=-1)
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
    """

    def __init__(self, dim: int, hidden: int, state_proj: int = 8, mlp_hidden: int = 16) -> None:
        super().__init__()
        self.state_proj = nn.Linear(hidden, state_proj)
        cand_vec_dim = dim + FEATURE_DIM + 1
        in_dim = cand_vec_dim + dim + state_proj
        self.net = nn.Sequential(nn.Linear(in_dim, mlp_hidden), nn.Tanh(),
                                  nn.Linear(mlp_hidden, 1))

    def forward(self, cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state):
        b, C, _d = cand_entity.shape
        feat = cand_feature.unsqueeze(1).expand(b, C, -1)      # [B, C, F]
        prior = cand_prior.unsqueeze(-1)                        # [B, C, 1]
        candidate_vec = torch.cat([cand_entity, feat, prior], dim=-1)   # [B, C, cand_vec_dim]
        s = self.state_proj(state).unsqueeze(1).expand(b, C, -1)        # [B, C, state_proj]
        x = torch.cat([candidate_vec, mem_read, s], dim=-1)
        return self.net(x).squeeze(-1)                          # [B, C]


def make_resolver(track: str, dim: int, hidden: int = 128) -> Resolver:
    """Convenience factory: ``track`` is ``"A"`` (:class:`CorefHead`) or ``"B"``
    (:class:`SharedScorer`, needs the controller's ``hidden`` size for its state
    projection). Used by ``scripts/train_resolver.py``'s ``--track`` flag."""
    t = track.strip().upper()
    if t == "A":
        return CorefHead(dim)
    if t == "B":
        return SharedScorer(dim, hidden)
    raise ValueError(f"unknown track {track!r}, expected 'A' or 'B'")
