"""Tests for M52: the reactor consumes full argument sets (multi-arg
clauses unroll into consecutive (entity, role, value) steps sharing the
entity) + the multi-arg TRANSFER curriculum (curriculum2.py).

See dev/MIND_INTERFACE.md and dev/RESOLVER_BUILD_PLAN.md Phase 1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nsm_ct.clause_reactor import _queried_role, _question_entity, build_clause_batch
from nsm_ct.curriculum2 import (
    TRANSFER_TEMPLATES,
    generate_transfer_episodes,
    verify_transfer_templates,
)
from nsm_ct.episode import CurriculumGenerator, Episode, _NAMES, _PLACES
from nsm_ct.tpr import TPRCodec


def _parser_env(dim=48):
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer

    texts = list(TRANSFER_TEMPLATES.values()) + list(_NAMES) + list(_PLACES) + \
        ["ball", "box", "key", "book", "letter", "coin",
         "where is mary ?", "where is the ball ?", "who has the ball ?"]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return parser, NSMMeaningResolver(), TPRCodec(dim=dim)


# ---------------------------------------------------------------------------
# Template verification (the new level's per-template table)
# ---------------------------------------------------------------------------
def test_transfer_templates_all_roles_correct():
    results = verify_transfer_templates()
    if not results:
        pytest.skip("quantum_parser unavailable in this environment")
    assert set(results) == set(TRANSFER_TEMPLATES)
    bad = {k: v for k, v in results.items() if not v["ok"]}
    assert not bad, f"transfer templates failed role verification: {bad}"


def test_dative_pp_landmine_is_real_not_assumed():
    """The literal spec template "gave the X to Y in the Z" USED TO mislabel
    Y as PLACE (clause.py's "to"->PLACE convention for "moved to the
    office"), which is why TRANSFER_TEMPLATES uses double-object phrasing
    instead. M58c (dev/PROSE_FAILURE_TAXONOMY.md's "Bug surfaced") fixed
    this: "to" after a TRANSFER verb with an entity object now resolves to
    RECIPIENT. TRANSFER_TEMPLATES still deliberately keeps the double-object
    phrasing (a harmless, still-valid design choice, not something this fix
    requires reverting -- see curriculum2.py's own landmine-avoided note);
    this test now confirms the underlying bug is FIXED rather than merely
    documenting the workaround."""
    from nsm_ct.clause import extract_discourse
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer

    sent = "mary gave the ball to john in the garden ."
    tok = SimpleTokenizer.build([sent, "mary", "john", "ball", "garden"],
                                 extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    graph = parser._parse_graph(sent)
    clauses, _links = extract_discourse(graph)
    roles = {rel: (arg.token or "").lower() for cl in clauses for rel, arg in cl.args}
    # the fix: "john" now comes out RECIPIENT, "garden" stays PLACE.
    assert roles.get("RECIPIENT") == "john"
    assert roles.get("PLACE") == "garden"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_transfer_deterministic_given_seed():
    a = generate_transfer_episodes(40, seed=7)
    b = generate_transfer_episodes(40, seed=7)
    assert [e.context for e in a] == [e.context for e in b]
    assert [e.question for e in a] == [e.question for e in b]
    assert [e.answer_text for e in a] == [e.answer_text for e in b]
    assert [e.options for e in a] == [e.options for e in b]


def test_transfer_different_seeds_differ():
    a = generate_transfer_episodes(40, seed=1)
    b = generate_transfer_episodes(40, seed=2)
    assert [e.context for e in a] != [e.context for e in b]


def test_transfer_episodes_structurally_valid():
    eps = generate_transfer_episodes(60, seed=3)
    assert len(eps) == 60
    assert {e.level for e in eps} == {7, 8}
    for e in eps:
        assert e.is_multiple_choice
        assert e.answer_text in e.options
        assert e.options[e.answer_idx] == e.answer_text
        assert len(e.context) == 1
        assert e.question.strip().endswith("?")
        assert e.meta.get("src") == "curriculum2"
        if e.level == 7:
            assert e.meta["kind"] == "transfer_place"
            assert e.answer_text in _PLACES
        else:
            assert e.meta["kind"] == "transfer_who"
            assert e.meta["verb"] in ("GIVE", "HAND", "PASS")
            assert e.answer_text in _NAMES


def test_transfer_who_never_uses_take_verb():
    eps = [e for e in generate_transfer_episodes(200, seed=9) if e.level == 8]
    assert eps
    assert all(e.meta["verb"] != "TAKE" for e in eps)


# ---------------------------------------------------------------------------
# Batch-identity regression (LOAD-BEARING): an old single-arg episode must
# produce EXACTLY the same batch as the documented pre-M52 shape.
# ---------------------------------------------------------------------------
def test_batch_identity_regression_old_single_arg_episode():
    parser, resolver, codec = _parser_env(dim=32)
    ep = CurriculumGenerator(max_level=1, seed=0).generate(1)[0]
    assert ep.level == 1
    name = ep.context[0].split()[0]
    place = ep.answer_text

    batch = build_clause_batch([ep], parser, resolver, codec)
    b, T, d = batch.entity.shape
    assert b == 1 and d == 32
    assert T == 2                      # one context step + one question step
    assert batch.mask[0].sum().item() == 2
    assert batch.is_q[0].tolist() == [0.0, 1.0]

    # step 0: the old (SUBJECT, PLACE) shape, byte-identical vectors.
    exp_ent = codec.filler_vec("var:" + name)
    exp_rel = codec.filler_vec("rel:PLACE")
    tree = resolver.resolve(place)
    exp_val = codec.contract(codec.encode_matrix(tree.root))
    from nsm_ct.usvs_bridge import usvs_handle
    usvs_val = usvs_handle(place, codec.dim)
    if usvs_val is not None:
        exp_val = usvs_val
    assert np.allclose(batch.entity[0, 0].numpy(), exp_ent)
    assert np.allclose(batch.relation[0, 0].numpy(), exp_rel)
    assert np.allclose(batch.value[0, 0].numpy(), exp_val)

    # step 1 (question): entity = var:name, relation = rel:PLACE (default,
    # unchanged), value = zeros, is_q = 1.
    assert np.allclose(batch.entity[0, 1].numpy(), exp_ent)
    assert np.allclose(batch.relation[0, 1].numpy(), exp_rel)
    assert np.allclose(batch.value[0, 1].numpy(), np.zeros(32, np.float32))


def test_batch_identity_regression_levels_1_to_8():
    """Broader sweep: every old-curriculum level (1-8, the ones that go
    through _context_steps rather than _reasoning_steps) must still produce
    a clause step per context sentence with entity=var:<subject>,
    relation=rel:PLACE -- never touching the M52 transfer path (no OBJECT
    arg is ever emitted by these templates)."""
    parser, resolver, codec = _parser_env(dim=32)
    eps = CurriculumGenerator(max_level=8, seed=1).generate(40)
    eps = [e for e in eps if e.level < 9]
    assert eps
    batch = build_clause_batch(eps, parser, resolver, codec)
    for i in range(len(eps)):
        # every relation on a real step must be rel:PLACE (the only role
        # old templates ever produce, on both context and question steps)
        for t in range(batch.mask[i].shape[0]):
            if batch.mask[i, t] == 0:
                continue
            rel = batch.relation[i, t].numpy()
            assert np.allclose(rel, codec.filler_vec("rel:PLACE"))


# ---------------------------------------------------------------------------
# Multi-arg unrolling: shape + shared entity
# ---------------------------------------------------------------------------
def test_multi_arg_clause_unrolls_sharing_entity():
    parser, resolver, codec = _parser_env(dim=48)
    ep = Episode(
        context=["mary gave john the ball in the garden ."],
        question="where is the ball ?",
        answer_text="garden", options=["garden", "kitchen", "office", "hallway"],
        answer_idx=0, level=7,
    )
    batch = build_clause_batch([ep], parser, resolver, codec)
    b, T, d = batch.entity.shape
    assert b == 1
    n_real = int(batch.mask[0].sum().item())
    assert n_real == 4          # AGENT + RECIPIENT + PLACE (context) + 1 question step
    assert batch.is_q[0].tolist().count(1.0) == 1
    assert batch.is_q[0].tolist()[-1] == 1.0    # question step is last

    from nsm_ct.clause_reactor import _ent_vec
    cache = {}
    ball_vec = _ent_vec("ball", resolver, codec, cache)
    mary_vec = _ent_vec("mary", resolver, codec, cache)
    john_vec = _ent_vec("john", resolver, codec, cache)
    garden_vec = _ent_vec("garden", resolver, codec, cache)

    ctx_steps = [(batch.entity[0, t].numpy(), batch.relation[0, t].numpy(), batch.value[0, t].numpy())
                 for t in range(3)]
    # all three context steps share the OBJECT (ball) as entity
    for e, _r, _v in ctx_steps:
        assert np.allclose(e, ball_vec)

    rel_agent = codec.filler_vec("rel:AGENT")
    rel_recipient = codec.filler_vec("rel:RECIPIENT")
    rel_place = codec.filler_vec("rel:PLACE")
    found = {"AGENT": False, "RECIPIENT": False, "PLACE": False}
    for _e, r, v in ctx_steps:
        if np.allclose(r, rel_agent):
            assert np.allclose(v, mary_vec)
            found["AGENT"] = True
        elif np.allclose(r, rel_recipient):
            assert np.allclose(v, john_vec)
            found["RECIPIENT"] = True
        elif np.allclose(r, rel_place):
            assert np.allclose(v, garden_vec)
            found["PLACE"] = True
    assert all(found.values()), found

    # question step: entity = ball (same as context), relation = PLACE (the
    # queried role for "where is ... ?")
    assert np.allclose(batch.entity[0, 3].numpy(), ball_vec)
    assert np.allclose(batch.relation[0, 3].numpy(), rel_place)


def test_take_variant_uses_source_role():
    parser, resolver, codec = _parser_env(dim=32)
    ep = Episode(
        context=["mary took the ball from john in the garden ."],
        question="where is the ball ?",
        answer_text="garden", options=["garden", "kitchen", "office", "hallway"],
        answer_idx=0, level=7,
    )
    batch = build_clause_batch([ep], parser, resolver, codec)
    from nsm_ct.clause_reactor import _ent_vec
    cache = {}
    rel_source = codec.filler_vec("rel:SOURCE")
    rel_agent = codec.filler_vec("rel:AGENT")
    john_vec = _ent_vec("john", resolver, codec, cache)
    mary_vec = _ent_vec("mary", resolver, codec, cache)
    seen_source = seen_agent = False
    for t in range(batch.mask.shape[1] - 1):     # exclude the question step
        if batch.mask[0, t] == 0:
            continue
        r, v = batch.relation[0, t].numpy(), batch.value[0, t].numpy()
        if np.allclose(r, rel_source):
            assert np.allclose(v, john_vec)
            seen_source = True
        elif np.allclose(r, rel_agent):
            assert np.allclose(v, mary_vec)
            seen_agent = True
    assert seen_source and seen_agent


# ---------------------------------------------------------------------------
# Queried role
# ---------------------------------------------------------------------------
def test_queried_role_mapping():
    assert _queried_role("where is mary ?") == "PLACE"
    assert _queried_role("where is the ball ?") == "PLACE"
    assert _queried_role("who has the ball ?") == "RECIPIENT"


def test_question_entity_extends_to_objects():
    assert _question_entity("where is mary ?") == "mary"
    assert _question_entity("where is the ball ?") == "ball"
    assert _question_entity("who has the ball ?") == "ball"


def test_who_has_question_queries_recipient_relation_in_batch():
    parser, resolver, codec = _parser_env(dim=32)
    ep_place = Episode(
        context=["mary gave john the ball in the garden ."],
        question="where is the ball ?", answer_text="garden",
        options=["garden", "kitchen", "office", "hallway"], answer_idx=0, level=7,
    )
    ep_who = Episode(
        context=["mary gave john the ball in the garden ."],
        question="who has the ball ?", answer_text="john",
        options=["john", "mary", "sandra", "daniel"], answer_idx=0, level=8,
    )
    batch = build_clause_batch([ep_place, ep_who], parser, resolver, codec)
    q_idx_place = int(batch.mask[0].sum().item()) - 1
    q_idx_who = int(batch.mask[1].sum().item()) - 1
    assert np.allclose(batch.relation[0, q_idx_place].numpy(), codec.filler_vec("rel:PLACE"))
    assert np.allclose(batch.relation[1, q_idx_who].numpy(), codec.filler_vec("rel:RECIPIENT"))


def test_old_where_question_unaffected_by_queried_role_change():
    """Every old-curriculum "where is {name} ?" question must still map to
    rel:PLACE with entity=var:<name> -- the queried-role change's default."""
    parser, resolver, codec = _parser_env(dim=32)
    eps = CurriculumGenerator(max_level=6, seed=2).generate(20)
    batch = build_clause_batch(eps, parser, resolver, codec)
    for i in range(len(eps)):
        # find the (single) question step -- NOT always the last position:
        # level 4 appends post_context distractors after it.
        q_positions = [t for t in range(batch.mask.shape[1])
                       if batch.mask[i, t] > 0 and batch.is_q[i, t] == 1.0]
        assert len(q_positions) == 1
        q_idx = q_positions[0]
        assert np.allclose(batch.relation[i, q_idx].numpy(), codec.filler_vec("rel:PLACE"))
