"""Tests for M53a: the membrane types (nsm_ct.membrane) + how
nsm_ct.clause_reactor.build_clause_batch carries pronoun candidate sets
through the batch. See dev/MIND_INTERFACE.md and
dev/RESOLVER_BUILD_PLAN.md Phase 2.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nsm_ct.clause_reactor import build_clause_batch
from nsm_ct.curriculum2 import generate_pronoun_episodes
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.membrane import (
    FEATURE_DIM,
    NAME_GENDER,
    Candidate,
    CandidateSet,
    EntityCandidateSet,
    entity_registry,
    mention_feature_vector,
    pronoun_entity_candidate_set,
)
from nsm_ct.tpr import TPRCodec


def _parser_env(dim=32):
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer

    eps = generate_pronoun_episodes(30, seed=0)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return parser, NSMMeaningResolver(), TPRCodec(dim=dim), eps


# ---------------------------------------------------------------------------
# Membrane type construction / determinism
# ---------------------------------------------------------------------------
def test_candidate_set_basic_shape():
    cs = CandidateSet(candidates=[Candidate("mary", 0.5), Candidate("john", 0.5)])
    assert len(cs) == 2
    assert cs.keys == ["mary", "john"]
    assert np.allclose(cs.priors, [0.5, 0.5])


def test_candidate_set_empty_is_fine():
    cs = CandidateSet()
    assert len(cs) == 0
    assert cs.keys == []
    assert cs.priors.shape == (0,)


def test_entity_candidate_set_fields():
    ec = EntityCandidateSet(
        candidates=[Candidate("mary", 0.5), Candidate("john", 0.5)],
        surface="she", feature=mention_feature_vector("she"), gold_index=0,
        provenance={"sentence_index": 2},
    )
    assert ec.surface == "she"
    assert ec.gold_index == 0
    assert ec.feature.shape == (FEATURE_DIM,)
    assert ec.provenance == {"sentence_index": 2}


def test_name_gender_covers_all_curriculum_names():
    from nsm_ct.episode import _NAMES

    assert set(NAME_GENDER) == {n.lower() for n in _NAMES}
    assert set(NAME_GENDER.values()) <= {"F", "M"}


# ---------------------------------------------------------------------------
# Feature-vector table
# ---------------------------------------------------------------------------
def test_feature_vector_shape_and_determinism():
    v1 = mention_feature_vector("she")
    v2 = mention_feature_vector("she")
    assert v1.shape == (FEATURE_DIM,)
    assert np.array_equal(v1, v2)


@pytest.mark.parametrize("pronoun,person,gender_f,gender_m,nonperson,plural", [
    ("she", 1.0, 1.0, 0.0, 0.0, 0.0),
    ("her", 1.0, 1.0, 0.0, 0.0, 0.0),
    ("he", 1.0, 0.0, 1.0, 0.0, 0.0),
    ("him", 1.0, 0.0, 1.0, 0.0, 0.0),
    ("it", 0.0, 0.0, 0.0, 1.0, 0.0),
    ("they", 1.0, 0.0, 0.0, 0.0, 1.0),
    ("them", 1.0, 0.0, 0.0, 0.0, 1.0),
])
def test_pronoun_feature_profiles(pronoun, person, gender_f, gender_m, nonperson, plural):
    v = mention_feature_vector(pronoun)
    # layout: [usvs_lex_noun_person, PERSON, GENDER_F, GENDER_M, NONPERSON, PLURAL]
    assert v[1] == person
    assert v[2] == gender_f
    assert v[3] == gender_m
    assert v[4] == nonperson
    assert v[5] == plural


def test_pronouns_carry_no_usvs_lexical_signal():
    """Pronouns aren't WordNet lemmas -- the USVS component of their feature
    vector must be exactly 0 (the whole reason gender/person live in the
    hand-specified extra dims instead)."""
    for w in ("she", "he", "it", "they", "him", "her", "them"):
        v = mention_feature_vector(w)
        assert v[0] == 0.0, f"{w!r} unexpectedly has USVS lex:noun.person signal"


def test_name_feature_uses_hand_specified_gender_not_usvs_gender_axes():
    """The documented finding this module's design rests on: USVS's own
    attr:gender / attr:sex axes carry no usable signal (measured ~1e-5..1e-6
    for man/woman/person/mary/john/ball -- indistinguishable from noise), so
    a name's gender comes from membrane.NAME_GENDER, not from projecting
    the word's real USVS coordinate onto those axes."""
    from nsm_ct.usvs_bridge import default_usvs

    u = default_usvs()
    axis_idx = {a: i for i, a in enumerate(u.axes)}
    assert "attr:gender" in axis_idx and "attr:sex" in axis_idx
    for w in ("man", "woman", "person", "mary", "john"):
        coord = u.word_coord(w)
        if coord is None:
            continue
        assert abs(float(coord[axis_idx["attr:gender"]])) < 1e-3
        assert abs(float(coord[axis_idx["attr:sex"]])) < 1e-3

    mary_v = mention_feature_vector("mary")
    john_v = mention_feature_vector("john")
    assert mary_v[2] == 1.0 and mary_v[3] == 0.0     # GENDER_F, GENDER_M
    assert john_v[2] == 0.0 and john_v[3] == 1.0


def test_unknown_word_feature_is_neutral():
    v = mention_feature_vector("garden")
    assert v[1:].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Candidate-set emission (registry + gold index)
# ---------------------------------------------------------------------------
def test_entity_registry_order_and_pronoun_exclusion():
    parser, _resolver, _codec, _eps = _parser_env()
    sents = ["mary went to the garden .", "john went to the kitchen .",
             "she found the ball ."]
    reg = entity_registry(sents, parser)
    assert reg == ["mary", "john"]           # order of first mention, pronoun excluded


def test_entity_registry_only_looks_at_sentences_given():
    parser, _resolver, _codec, _eps = _parser_env()
    sents = ["mary went to the garden .", "john went to the kitchen ."]
    assert entity_registry(sents[:1], parser) == ["mary"]
    assert entity_registry(sents[:0], parser) == []


def test_pronoun_entity_candidate_set_gold_index():
    registry = ["mary", "john"]
    cand = pronoun_entity_candidate_set("she", registry, gold_antecedent="john")
    assert cand.keys == ["mary", "john"]
    assert cand.gold_index == 1
    assert np.allclose(cand.priors, [0.5, 0.5])
    assert cand.surface == "she"
    assert np.array_equal(cand.feature, mention_feature_vector("she"))


def test_pronoun_entity_candidate_set_gold_not_in_registry_is_none():
    cand = pronoun_entity_candidate_set("she", ["mary"], gold_antecedent="nobody")
    assert cand.gold_index is None


# ---------------------------------------------------------------------------
# Batch carrying: candidate sets present for pronoun episodes, right shape
# ---------------------------------------------------------------------------
def test_batch_carries_candidate_sets_for_pronoun_episodes():
    parser, resolver, codec, eps = _parser_env()
    batch = build_clause_batch(eps, parser, resolver, codec)
    assert batch.cand_entity is not None
    b, T, d = batch.entity.shape
    _, _, C, d2 = batch.cand_entity.shape
    assert d2 == d
    assert batch.cand_feature.shape == (b, T, FEATURE_DIM)
    assert batch.cand_gold.shape == (b, T)

    for i, e in enumerate(eps):
        t = e.meta["pronoun_sentence_index"]
        registry = e.meta["registry_order"]
        expected_gold = registry.index(e.meta["gold_antecedent"])
        assert int(batch.cand_gold[i, t]) == expected_gold
        assert int(batch.cand_mask[i, t].sum()) == len(registry)
        # every OTHER step never carries a candidate set
        for tt in range(T):
            if tt == t:
                continue
            assert int(batch.cand_mask[i, tt].sum()) == 0
            assert int(batch.cand_gold[i, tt]) == -1


def test_batch_placeholder_binds_pronoun_step_to_gold_place():
    """M53a's PLACEHOLDER contract: the pronoun step's (entity, relation,
    value) is literally (transferred object, PLACE, gold antecedent's
    place) -- the reactor sees a resolved fact, not an unresolved pronoun,
    even though the resolver doesn't exist yet."""
    from nsm_ct.clause_reactor import _content_vec, _ent_vec

    parser, resolver, codec, eps = _parser_env()
    batch = build_clause_batch(eps, parser, resolver, codec)
    cache = {}
    for i, e in enumerate(eps):
        t = e.meta["pronoun_sentence_index"]
        obj = e.question.split()[-2]
        obj_vec = _ent_vec(obj, resolver, codec, cache)
        place_vec = _content_vec(e.meta["gold_place"], resolver, codec, cache)
        assert np.allclose(batch.entity[i, t].numpy(), obj_vec)
        assert np.allclose(batch.relation[i, t].numpy(), codec.filler_vec("rel:PLACE"))
        assert np.allclose(batch.value[i, t].numpy(), place_vec)


# ---------------------------------------------------------------------------
# Byte-identity regression: pronoun-free episodes are UNAFFECTED by M53a.
# ---------------------------------------------------------------------------
def test_batch_identity_regression_no_pronoun_episodes():
    parser, resolver, codec, _pron_eps = _parser_env()
    eps = CurriculumGenerator(max_level=8, seed=1).generate(30)
    batch = build_clause_batch(eps, parser, resolver, codec)
    # the whole point of the cand_* fields defaulting to None: nothing new
    # is allocated when there is nothing for the resolver to train on.
    assert batch.cand_entity is None
    assert batch.cand_mask is None
    assert batch.cand_prior is None
    assert batch.cand_feature is None
    assert batch.cand_gold is None

    # and the core tensors match a fresh, from-scratch computation exactly.
    batch2 = build_clause_batch(eps, parser, resolver, codec)
    assert torch.equal(batch.entity, batch2.entity)
    assert torch.equal(batch.relation, batch2.relation)
    assert torch.equal(batch.value, batch2.value)
    assert torch.equal(batch.pred, batch2.pred)
    assert torch.equal(batch.is_q, batch2.is_q)
    assert torch.equal(batch.mask, batch2.mask)
    assert torch.equal(batch.options, batch2.options)
    assert torch.equal(batch.answer, batch2.answer)


def test_to_and_subset_preserve_none_cand_fields():
    parser, resolver, codec, _pron_eps = _parser_env()
    eps = CurriculumGenerator(max_level=6, seed=3).generate(10)
    batch = build_clause_batch(eps, parser, resolver, codec)
    assert batch.to("cpu").cand_entity is None
    assert batch.subset(torch.tensor([0, 1])).cand_entity is None


def test_to_and_subset_carry_cand_fields_when_present():
    parser, resolver, codec, eps = _parser_env()
    batch = build_clause_batch(eps, parser, resolver, codec)
    moved = batch.to("cpu")
    assert moved.cand_entity is not None
    assert torch.equal(moved.cand_gold, batch.cand_gold)
    sub = batch.subset(torch.tensor([0, 1, 2]))
    assert sub.cand_entity is not None
    assert sub.cand_entity.shape[0] == 3
    assert torch.equal(sub.cand_gold, batch.cand_gold[[0, 1, 2]])


# ---------------------------------------------------------------------------
# M56b: the per-candidate feature register (dev/TRACK_C_DESIGN.md §1.8's
# "GAP: no such op/register exists today" -- EntityCandidateSet.cand_features
# + ClauseBatch.cand_feature_per_candidate).
# ---------------------------------------------------------------------------
def test_pronoun_entity_candidate_set_carries_cand_features():
    registry = ["mary", "john"]
    cand = pronoun_entity_candidate_set("she", registry, gold_antecedent="john")
    assert cand.cand_features is not None
    assert cand.cand_features.shape == (2, FEATURE_DIM)
    assert np.array_equal(cand.cand_features[0], mention_feature_vector("mary"))
    assert np.array_equal(cand.cand_features[1], mention_feature_vector("john"))
    # each candidate's OWN feature -- NOT the mention's ("she"'s feature is
    # FEMALE; "john"'s candidate feature must stay MALE, not copy the mention).
    assert not np.array_equal(cand.cand_features[1], cand.feature)


def test_pronoun_entity_candidate_set_empty_registry_cand_features_is_none():
    cand = pronoun_entity_candidate_set("she", [], gold_antecedent=None)
    assert cand.cand_features is None


def test_entity_candidate_set_default_cand_features_is_none():
    """Hand-built (no cand_features passed) mirrors every pre-M56b
    construction -- the field is purely additive."""
    ec = EntityCandidateSet(candidates=[Candidate("mary", 1.0)])
    assert ec.cand_features is None


def test_batch_carries_cand_feature_per_candidate_for_pronoun_episodes():
    parser, resolver, codec, eps = _parser_env()
    batch = build_clause_batch(eps, parser, resolver, codec)
    assert batch.cand_feature_per_candidate is not None
    b, T, C, d = batch.cand_entity.shape
    assert batch.cand_feature_per_candidate.shape == (b, T, C, FEATURE_DIM)

    for i, e in enumerate(eps):
        t = e.meta["pronoun_sentence_index"]
        registry = e.meta["registry_order"]
        for j, name in enumerate(registry):
            got = batch.cand_feature_per_candidate[i, t, j].numpy()
            assert np.array_equal(got, mention_feature_vector(name)), (i, j, name)
        # padding beyond the real candidate count stays zero
        for j in range(len(registry), C):
            assert np.array_equal(batch.cand_feature_per_candidate[i, t, j].numpy(),
                                   np.zeros(FEATURE_DIM, dtype=np.float32))
        # every OTHER step carries an all-zero per-candidate feature slab
        for tt in range(T):
            if tt == t:
                continue
            assert np.array_equal(batch.cand_feature_per_candidate[i, tt].numpy(),
                                   np.zeros((C, FEATURE_DIM), dtype=np.float32))


def test_batch_cand_feature_per_candidate_none_for_pronoun_free_episodes():
    """Byte-identity regression companion: an old-curriculum-only batch
    (no EntityCandidateSet at all) must leave the new field None exactly
    like every other cand_* field (test_batch_identity_regression_no_pronoun_episodes)."""
    parser, resolver, codec, _pron_eps = _parser_env()
    eps = CurriculumGenerator(max_level=8, seed=1).generate(30)
    batch = build_clause_batch(eps, parser, resolver, codec)
    assert batch.cand_feature_per_candidate is None


def test_to_and_subset_carry_cand_feature_per_candidate_when_present():
    parser, resolver, codec, eps = _parser_env()
    batch = build_clause_batch(eps, parser, resolver, codec)
    moved = batch.to("cpu")
    assert moved.cand_feature_per_candidate is not None
    assert torch.equal(moved.cand_feature_per_candidate, batch.cand_feature_per_candidate)
    sub = batch.subset(torch.tensor([0, 1, 2]))
    assert sub.cand_feature_per_candidate is not None
    assert torch.equal(sub.cand_feature_per_candidate, batch.cand_feature_per_candidate[[0, 1, 2]])
