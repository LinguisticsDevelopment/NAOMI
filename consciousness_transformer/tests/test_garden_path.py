"""Tests for M55a: the membrane half of parse-hypothesis collapse
(dev/TRACK_C_DESIGN.md Sec 1.10, RESEARCH_NOTES M55a, dev/NEXT_ARC_PLAN.md
M55). Covers:

1. The per-candidate Addr register plumbing (membrane.Candidate.query_entity/
   query_relation, ClauseBatch.hyp_cand_*, resolver.query_candidates_per_addr,
   ClauseReactor._collapse's third branch) -- synthetic tensors, no parser.
2. input_encoder.ParserInputEncoder._parse_topk_one's top-K non-equivalence.
3. The survey shape's determinism (verify_garden_path_templates).
4. Both M55a honesty-gate baselines sitting at/near floor.
5. Gold-index packing (curriculum -> batch tensors).
6. Byte-identity: all three candidate kinds (entity/sense/hypothesis)
   coexisting in one batch, and old batches carrying none of them.

Parser-dependent tests skip cleanly (mirrors every other test_*.py in this
module) if quantum_parser isn't importable in this environment.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nsm_ct import entity_memory as em
from nsm_ct.clause_reactor import ClauseBatch, ClauseReactor
from nsm_ct.membrane import Candidate, HypothesisCandidateSet, hypothesis_candidate_set
from nsm_ct.resolver import SharedScorer, query_candidates_per_addr


# ---------------------------------------------------------------------------
# 1. Addr-register plumbing (synthetic tensors, no parser).
# ---------------------------------------------------------------------------
def test_candidate_query_addr_fields_default_none():
    c = Candidate(key="mary")
    assert c.query_entity is None
    assert c.query_relation is None


def test_hypothesis_candidate_set_builder():
    cs = hypothesis_candidate_set(
        readings=[("object_reading", 0.5, "watch", "PLACE"),
                  ("verb_reading", 0.5, "mary", "PLACE")],
        gold_index=0, provenance={"sentence": "mary can watch ."})
    assert isinstance(cs, HypothesisCandidateSet)
    assert len(cs) == 2
    assert cs.gold_index == 0
    assert cs.candidates[0].query_entity == "watch"
    assert cs.candidates[1].query_entity == "mary"
    assert cs.candidates[0].query_relation == cs.candidates[1].query_relation == "PLACE"
    assert cs.priors.tolist() == [0.5, 0.5]


def test_query_candidates_per_addr_matches_manual_loop():
    b, C, d = 3, 4, 8
    memory = em.init_memory(b, d, torch.device("cpu"))
    ent = F.normalize(torch.randn(b, 1, d), dim=-1)
    rel = F.normalize(torch.randn(b, d), dim=-1)
    val = F.normalize(torch.randn(b, 1, d), dim=-1)
    memory = em.write(memory, ent[:, 0], rel, val[:, 0], torch.ones(b), overwrite=torch.ones(b))

    cand_query_entity = F.normalize(torch.randn(b, C, d), dim=-1)
    cand_query_relation = F.normalize(torch.randn(b, C, d), dim=-1)
    got = query_candidates_per_addr(memory, cand_query_entity, cand_query_relation)
    ref = torch.stack([em.query(memory, cand_query_entity[:, j], cand_query_relation[:, j])
                        for j in range(C)], dim=1)
    assert torch.equal(got, ref)


def test_query_candidates_per_addr_zero_candidates_is_empty_not_error():
    b, d = 3, 8
    memory = em.init_memory(b, d, torch.device("cpu"))
    got = query_candidates_per_addr(memory, torch.zeros(b, 0, d), torch.zeros(b, 0, d))
    assert got.shape == (b, 0, d)


def _toy_hyp_batch(b=5, d=12, C=2, seed=0):
    """A minimal synthetic ClauseBatch carrying ONLY hyp_cand_* candidate
    data (no cand_*/sense_cand_*) over 3 steps: two write steps (each
    binding one candidate's own address) + one collapse+question step
    reusing ClauseBatch's own is_q=1 slot to keep this small. Mirrors
    tests/test_resolver.py's `_toy_batch_with_candidates` pattern."""
    g = torch.Generator().manual_seed(seed)
    K = 3
    opts = F.normalize(torch.randn(b, K, d, generator=g), dim=-1)
    ans = torch.randint(0, K, (b,), generator=g)

    query_entity = F.normalize(torch.randn(b, C, d, generator=g), dim=-1)
    query_relation = F.normalize(torch.randn(b, C, d, generator=g), dim=-1)
    hyp_identity = query_entity.clone()      # candidate identity == its own query entity (as in _garden_path_steps)
    gold_idx = torch.randint(0, C, (b,), generator=g)

    T = C + 2   # C write steps + 1 collapse step (value unused/placeholder) + 1 question step
    entity = torch.zeros(b, T, d); relation = torch.zeros(b, T, d); value = torch.zeros(b, T, d)
    pred = F.normalize(torch.randn(b, T, d, generator=g), dim=-1)
    is_q = torch.zeros(b, T); mask = torch.ones(b, T)
    for j in range(C):
        entity[:, j] = query_entity[:, j]
        relation[:, j] = query_relation[:, j]
        # each candidate's own write == one option's meaning; the GOLD
        # candidate's write is the episode's actual answer.
        value[:, j] = opts[torch.arange(b), torch.randint(0, K, (b,), generator=g)]
    value[torch.arange(b), gold_idx] = opts[torch.arange(b), ans]
    is_q[:, -1] = 1.0   # question step (last)

    hyp_cand_mask = torch.ones(b, T, C)
    hyp_cand_mask[:, C:, :] = 0.0
    hyp_cand_mask[:, C - 1, :] = 1.0   # the collapse step (index C-1 reused) carries the candidate set
    hyp_cand_prior = torch.full((b, T, C), 1.0 / C)
    hyp_cand_gold = torch.full((b, T), -1, dtype=torch.long)
    hyp_cand_gold[:, C - 1] = gold_idx
    hyp_cand_entity = torch.zeros(b, T, C, d)
    hyp_cand_query_entity = torch.zeros(b, T, C, d)
    hyp_cand_query_relation = torch.zeros(b, T, C, d)
    hyp_cand_entity[:, C - 1] = hyp_identity
    hyp_cand_query_entity[:, C - 1] = query_entity
    hyp_cand_query_relation[:, C - 1] = query_relation

    batch = ClauseBatch(entity, relation, value, pred, is_q, mask, opts, ans,
                         None, None,
                         None, None, None, None, None, None,          # cand_*
                         None, None, None, None, None, None, None,    # sense_cand_*
                         hyp_cand_entity, hyp_cand_mask, hyp_cand_prior, hyp_cand_gold,
                         hyp_cand_query_entity, hyp_cand_query_relation)
    return batch, gold_idx


def test_hyp_resolver_none_leaves_batch_untouched_byte_identical():
    """hyp_resolver=None (the M55a default, plumbing only) must reproduce the
    resolver-free / sense-resolver-free forward loop exactly and must not
    leak hyp_resolver_* keys, even though the batch carries real hyp_cand_*
    data -- mirrors test_resolver.py's own no-resolver regression."""
    torch.manual_seed(0)
    batch, _gold = _toy_hyp_batch()
    model = ClauseReactor(dim=12, hidden=16)   # resolver=None, sense_resolver=None, hyp_resolver=None
    out = model(batch)
    assert "hyp_resolver_logits" not in out
    assert "hyp_resolver_margin" not in out
    assert "resolver_logits" not in out
    assert "sense_resolver_logits" not in out


def test_hyp_resolver_installed_fires_and_shapes_correctly():
    torch.manual_seed(0)
    batch, gold_idx = _toy_hyp_batch(b=6, d=12, C=2)
    hyp_resolver = SharedScorer(12, 16)
    model = ClauseReactor(dim=12, hidden=16, hyp_resolver=hyp_resolver)
    out = model(batch)
    assert out["hyp_resolver_logits"].shape == (6, batch.entity.shape[1], 2)
    assert out["hyp_resolver_margin"].shape == (6, batch.entity.shape[1])
    assert "resolver_logits" not in out          # entity/sense slots stayed inert
    assert "sense_resolver_logits" not in out


def test_hyp_resolver_absent_on_hyp_candidate_free_batch_is_untouched():
    """A hyp_resolver CAN be installed, but a batch with no hyp_cand_mask at
    all (an old/pronoun/sense-only batch) must never fire it."""
    torch.manual_seed(0)
    batch, _ = _toy_hyp_batch()
    batch.hyp_cand_mask = None   # simulate an old batch
    hyp_resolver = SharedScorer(12, 16)
    model = ClauseReactor(dim=12, hidden=16, hyp_resolver=hyp_resolver)
    out = model(batch)
    assert "hyp_resolver_logits" not in out


def test_gradients_flow_through_hyp_collapse():
    torch.manual_seed(0)
    batch, _ = _toy_hyp_batch(b=4, d=10, C=2)
    hyp_resolver = SharedScorer(10, 16)
    model = ClauseReactor(dim=10, hidden=16, hyp_resolver=hyp_resolver)
    out = model(batch)
    loss = out["answer_logits"].sum() + out["hyp_resolver_logits"].sum()
    loss.backward()
    grads = [p.grad for p in hyp_resolver.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads)


# ---------------------------------------------------------------------------
# 2-4: parser-dependent (top-K exposure, survey-shape determinism, baselines)
# ---------------------------------------------------------------------------
def _make_parser():
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer
    from nsm_ct.curriculum2 import GARDEN_PATH_HOMOGRAPHS, _GARDEN_PATH_NAMES

    texts = (["mary can watch .", "the watch is in the garden .", "mary went to the garden ."]
             + list(GARDEN_PATH_HOMOGRAPHS) + list(_GARDEN_PATH_NAMES))
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    return ParserInputEncoder(tok)


def test_parse_topk_one_returns_non_equivalent_close_margin_hypotheses():
    parser = _make_parser()
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    graphs, scores, margin = parser._parse_topk_one("mary can watch .", k=2)
    assert len(graphs) == len(scores) == 2
    assert margin == pytest.approx(0.0, abs=1e-9)     # the survey's exact-tie finding
    from nsm_ct.clause import extract_discourse
    clauses0, _ = extract_discourse(graphs[0])
    clauses1, _ = extract_discourse(graphs[1])
    sig0 = frozenset((cl.predicate, rel, arg.token) for cl in clauses0 for rel, arg in cl.args)
    sig1 = frozenset((cl.predicate, rel, arg.token) for cl in clauses1 for rel, arg in cl.args)
    assert sig0 != sig1     # genuinely different readings, not a stray dedup miss


def test_parse_topk_one_default_off_path_unaffected():
    """_parse_topk_one is purely additive: _parse_graph/_parse_graph_one keep
    working exactly as before for the same sentence."""
    parser = _make_parser()
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    g1 = parser._parse_graph("mary went to the garden .")
    g2 = parser._parse_graph("mary went to the garden .")
    assert g1.nodes == g2.nodes and g1.edges == g2.edges


def test_parse_topk_one_multi_sentence_returns_empty():
    parser = _make_parser()
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    graphs, scores, margin = parser._parse_topk_one("mary went to the garden . she can watch .", k=2)
    assert graphs == [] and scores == [] and margin == 0.0


def test_garden_path_survey_shapes_all_verify_ok():
    from nsm_ct.curriculum2 import verify_garden_path_templates

    results = verify_garden_path_templates()
    if not results:
        pytest.skip("quantum_parser unavailable in this environment")
    bad = {k: v for k, v in results.items() if not v["ok"]}
    assert not bad, f"garden-path survey shapes should all verify: {bad}"


def test_garden_path_parser_top1_baseline_near_floor():
    from nsm_ct.curriculum2 import garden_path_parser_top1_baseline, generate_garden_path_episodes

    parser = _make_parser()
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    eps = generate_garden_path_episodes(120, seed=3)
    result = garden_path_parser_top1_baseline(eps, parser)
    assert result["n"] == 120
    assert result["accuracy"] == pytest.approx(0.5, abs=0.05)   # near-chance by construction


def test_garden_path_association_baseline_near_chance():
    from nsm_ct.curriculum2 import garden_path_association_baseline, generate_garden_path_episodes

    eps = generate_garden_path_episodes(200, seed=4)
    result = garden_path_association_baseline(eps)
    assert result["n"] == 200
    assert result["accuracy"] == pytest.approx(0.5, abs=0.15)


# ---------------------------------------------------------------------------
# 5. Gold-index packing (curriculum -> batch tensors).
# ---------------------------------------------------------------------------
def test_gold_index_packing_matches_meta():
    from nsm_ct.clause_reactor import build_clause_batch
    from nsm_ct.curriculum2 import generate_garden_path_episodes
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.tpr import TPRCodec

    eps = generate_garden_path_episodes(10, seed=5)
    texts = [s for e in eps for s in e.context] + [e.question for e in eps] + [o for e in eps for o in e.options]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    codec = TPRCodec(dim=16)
    resolver = NSMMeaningResolver()
    batch = build_clause_batch(eps, parser, resolver, codec)

    assert batch.hyp_cand_mask is not None
    for i, ep in enumerate(eps):
        expected = 0 if ep.meta["gold_reading"] == "object" else 1
        row_gold = batch.hyp_cand_gold[i]
        golds = row_gold[row_gold >= 0]
        assert golds.numel() == 1
        assert int(golds.item()) == expected
    # priors are the (real, near-tied) structural scores, normalized -- both
    # candidates present, summing to ~1 wherever the row is populated.
    populated = (batch.hyp_cand_gold >= 0)
    prior_sums = batch.hyp_cand_prior.sum(-1)[populated]
    assert torch.allclose(prior_sums, torch.ones_like(prior_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Byte-identity: old batches carry none of the M55a fields; all three
#    candidate kinds can coexist in one batch without interfering.
# ---------------------------------------------------------------------------
def test_old_batch_has_no_hyp_cand_fields():
    from nsm_ct.clause_reactor import build_clause_batch
    from nsm_ct.episode import CurriculumGenerator
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.tpr import TPRCodec

    eps = CurriculumGenerator(seed=0).generate(6)
    texts = [s for e in eps for s in e.context] + [e.question for e in eps] + [o for e in eps for o in e.options]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    codec = TPRCodec(dim=16)
    resolver = NSMMeaningResolver()
    batch = build_clause_batch(eps, parser, resolver, codec)
    assert batch.cand_mask is None
    assert batch.sense_cand_mask is None
    assert batch.hyp_cand_mask is None


def test_all_three_candidate_kinds_coexist_in_one_batch():
    """A batch mixing pronoun (entity), sense, AND garden-path episodes
    builds without error and each group's tensors are populated only on
    its OWN episodes' rows -- the three kinds never cross-contaminate."""
    from nsm_ct.clause_reactor import build_clause_batch, ClauseReactor
    from nsm_ct.curriculum2 import (generate_garden_path_episodes, generate_pronoun_episodes,
                                     generate_sense_binding_episodes)
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.tpr import TPRCodec

    pronoun_eps = generate_pronoun_episodes(4, seed=0)
    sense_eps = generate_sense_binding_episodes(4, seed=0)
    gp_eps = generate_garden_path_episodes(4, seed=0)
    eps = pronoun_eps + sense_eps + gp_eps

    texts = [s for e in eps for s in e.context] + [e.question for e in eps] + [o for e in eps for o in e.options]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    codec = TPRCodec(dim=16)
    resolver = NSMMeaningResolver()
    batch = build_clause_batch(eps, parser, resolver, codec)

    assert batch.cand_mask is not None
    assert batch.sense_cand_mask is not None
    assert batch.hyp_cand_mask is not None

    n_pronoun, n_sense = len(pronoun_eps), len(sense_eps)
    # pronoun rows carry cand_mask activity, no sense/hyp activity
    assert batch.cand_mask[:n_pronoun].sum() > 0
    assert batch.sense_cand_mask[:n_pronoun].sum() == 0
    assert batch.hyp_cand_mask[:n_pronoun].sum() == 0
    # sense rows carry sense_cand_mask activity, no pronoun/hyp activity
    assert batch.sense_cand_mask[n_pronoun:n_pronoun + n_sense].sum() > 0
    assert batch.cand_mask[n_pronoun:n_pronoun + n_sense].sum() == 0
    assert batch.hyp_cand_mask[n_pronoun:n_pronoun + n_sense].sum() == 0
    # garden-path rows carry hyp_cand_mask activity, no pronoun/sense activity
    assert batch.hyp_cand_mask[n_pronoun + n_sense:].sum() > 0
    assert batch.cand_mask[n_pronoun + n_sense:].sum() == 0
    assert batch.sense_cand_mask[n_pronoun + n_sense:].sum() == 0

    # forward pass with no resolvers installed at all: still byte-runs, no
    # resolver_* keys leak even though all three candidate groups are present.
    model = ClauseReactor(dim=16, hidden=16)
    out = model(batch)
    assert "resolver_logits" not in out
    assert "sense_resolver_logits" not in out
    assert "hyp_resolver_logits" not in out
