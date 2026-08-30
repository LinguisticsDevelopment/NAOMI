"""Tests for M57e: morphology signals (number + gender subtypes) flowing
parser -> membrane -> memory attributes -- dev/AURORA_SPRINT.md's
2026-08-11 reprioritization item, the last piece of the M57 "comprehensive
memory" milestone.

Covers:
  1. nsm_ct.membrane.PRONOUN_MORPHOLOGY -- the English + Spanish
     surface-form -> (gender, number, person) table (LOCKED DESIGN item 1).
  2. nsm_ct.clause._PRONOUNS/is_entity learning the Spanish personal
     pronouns (M-ES1's reported blocker), with an English-parsing
     byte-identical regression (the same discipline M-ES1 established).
  3. attr:number facts (LOCKED DESIGN item 2) via nsm_ct.instances'
     existing write_attribute/query_attribute machinery.
  4. Group minting: attr:member_0/attr:member_1 facts recoverable via
     nsm_ct.instances.inverse_query (LOCKED DESIGN item 2's "attr:member
     facts recover both members" requirement).
  5. nsm_ct.curriculum2.RichEpisodeGenerator's plural_frac (LOCKED DESIGN
     item 3): 0.0 regression (byte-identical), a plural episode's shape,
     and its honesty machinery.
  6. nsm_ct.clause_reactor._rich_steps' PLURAL group block (LOCKED DESIGN
     item 4): batch build for a plural episode, number evidence
     relation/target populated, and a forced-gold end-to-end eval
     answering "what are A and B like ?" via the group node.

No parser dependency in most of this file -- _rich_steps/curriculum2 are
parser-free by design (mirrors tests/test_rich_episodes.py's own isolation
discipline); the parser is used only in section 2's English-regression
test, mirroring tests/test_clause_round3.py's own fixture pattern.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause import _PRONOUNS, is_entity  # noqa: E402
from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch, _rich_step_labels, _rich_steps  # noqa: E402
from nsm_ct.curriculum2 import RichEpisodeGenerator, generate_rich_episodes  # noqa: E402
from nsm_ct.instances import InstanceRegistry, ProvenanceLog, inverse_query, query_attribute, write_attribute  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.membrane import PRONOUN_MORPHOLOGY  # noqa: E402
from nsm_ct.resolver import make_resolver  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

DIM = 24


def _meaning():
    return NSMMeaningResolver()


def _codec(dim=DIM):
    return TPRCodec(dim=dim)


# ---------------------------------------------------------------------------
# 1. PRONOUN_MORPHOLOGY table lookups (English + Spanish).
# ---------------------------------------------------------------------------
def test_pronoun_morphology_english_coverage():
    for w in ("he", "she", "it", "they", "him", "her", "them", "his", "hers", "their"):
        assert w in PRONOUN_MORPHOLOGY, w
    assert PRONOUN_MORPHOLOGY["he"] == ("M", "sg", "3")
    assert PRONOUN_MORPHOLOGY["she"] == ("F", "sg", "3")
    assert PRONOUN_MORPHOLOGY["they"][1] == "pl"
    assert PRONOUN_MORPHOLOGY["they"][0] == "unknown"   # never guessed
    assert PRONOUN_MORPHOLOGY["it"][0] == "unknown"


def test_pronoun_morphology_spanish_coverage():
    assert PRONOUN_MORPHOLOGY["él"] == ("M", "sg", "3")
    assert PRONOUN_MORPHOLOGY["ella"] == ("F", "sg", "3")
    assert PRONOUN_MORPHOLOGY["ellos"] == ("M", "pl", "3")
    assert PRONOUN_MORPHOLOGY["ellas"] == ("F", "pl", "3")
    assert PRONOUN_MORPHOLOGY["ello"][0] == "unknown"
    # "leave ambiguous forms 'unknown'" -- le/lo are gender-ambiguous
    # object clitics; la/los/las are unambiguous.
    assert PRONOUN_MORPHOLOGY["le"][0] == "unknown"
    assert PRONOUN_MORPHOLOGY["lo"][0] == "unknown"
    assert PRONOUN_MORPHOLOGY["la"] == ("F", "sg", "3")
    assert PRONOUN_MORPHOLOGY["los"] == ("M", "pl", "3")
    assert PRONOUN_MORPHOLOGY["las"] == ("F", "pl", "3")
    # number/person are always grammatically fixed -- never 'unknown'.
    for g, n, p in PRONOUN_MORPHOLOGY.values():
        assert n in ("sg", "pl")
        assert p == "3"


# ---------------------------------------------------------------------------
# 2. clause._PRONOUNS/is_entity learn the Spanish forms; English unaffected.
# ---------------------------------------------------------------------------
_ORIGINAL_ENGLISH_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they",
                               "him", "her", "them", "us", "me"}


def test_pronouns_are_additive_english_subset_preserved():
    """Every pre-M57e English pronoun is still recognized, unchanged."""
    assert _ORIGINAL_ENGLISH_PRONOUNS <= _PRONOUNS
    for w in _ORIGINAL_ENGLISH_PRONOUNS:
        assert is_entity(w)


def test_spanish_pronouns_now_recognized_as_entities():
    """M-ES1's reported blocker: ella/él (and the other Spanish 3rd-person
    personal pronouns) were NOT recognized by is_entity() before this
    milestone. They are now."""
    for w in ("él", "ella", "ellos", "ellas", "ello"):
        assert w in _PRONOUNS
        assert is_entity(w)


def test_spanish_object_clitics_deliberately_out_of_scope():
    """le/la/lo/los/las (object clitics) are covered by
    PRONOUN_MORPHOLOGY but deliberately NOT added to clause._PRONOUNS --
    a different grammatical slot, out of scope for this milestone (see
    clause.py's module comment)."""
    for w in ("le", "la", "lo", "los", "las"):
        assert w in PRONOUN_MORPHOLOGY
        assert w not in _PRONOUNS
        assert not is_entity(w)


@pytest.fixture(scope="module")
def parser():
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.nsm_primes import PRIME_NAMES
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.tokenizer import SimpleTokenizer

    sents = ["mary went to the garden .", "she found the ball .",
             "he is in the kitchen .", "they are in the garden .",
             "mary and john are in the garden ."]
    tok = SimpleTokenizer.build(sents, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    p = ParserInputEncoder(tok)
    if getattr(p, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable in this environment")
    return p


def test_english_parsing_byte_identical_regression(parser):
    """The M-ES1 discipline: a fixed English sentence set's parse ->
    extract_discourse structure must be UNCHANGED by the Spanish additions
    to clause._PRONOUNS/_PRONOUN_EXTRA (a golden snapshot of this
    milestone's own actual output, which is provably identical to
    pre-M57e behavior since English _PRONOUNS membership/is_entity results
    for these words are untouched -- see the two tests above)."""
    from nsm_ct.clause import extract_discourse

    def _shape(sent):
        graph = parser._parse_graph(sent)
        clauses, links = extract_discourse(graph)
        return (
            [(cl.predicate, [(r, a.token) for r, a in cl.args], cl.is_question) for cl in clauses],
            [(lk.coordinator, lk.prime, lk.i, lk.j) for lk in links],
        )

    assert _shape("mary went to the garden .") == (
        [("went", [("SUBJECT", "mary"), ("PLACE", "garden")], False)], [])
    assert _shape("she found the ball .") == (
        [("found", [("SUBJECT", "she"), ("OBJECT", "ball")], False)], [])
    assert _shape("he is in the kitchen .") == (
        [("is", [("SUBJECT", "he"), ("PLACE", "kitchen")], False)], [])
    assert _shape("they are in the garden .") == (
        [("are", [("SUBJECT", "they"), ("PLACE", "garden")], False)], [])
    assert _shape("mary and john are in the garden .") == (
        [("are", [("SUBJECT", "mary"), ("PLACE", "garden")], False),
         ("are", [("SUBJECT", "john"), ("PLACE", "garden")], False)],
        [("AND", None, 0, 1)])


# ---------------------------------------------------------------------------
# 3. attr:number facts written and recoverable via query_attribute.
# ---------------------------------------------------------------------------
def test_attr_number_sg_and_pl_round_trip():
    dim = 24
    codec = TPRCodec(dim=dim)
    registry = InstanceRegistry(dim=dim, seed=0)
    sg_id, _ = registry.mint("mary")
    pl_id, _ = registry.mint("group")
    log = ProvenanceLog()
    memory = torch.zeros(dim, dim, dim)

    memory = write_attribute(memory, registry, sg_id, "number",
                              codec.filler_vec("number:sg"), codec,
                              log=log, source="test", language="en",
                              timestamp=0.0, trust=1.0, value_label="sg")
    memory = write_attribute(memory, registry, pl_id, "number",
                              codec.filler_vec("number:pl"), codec,
                              log=log, source="test", language="en",
                              timestamp=1.0, trust=1.0, value_label="pl")

    sg_read = query_attribute(memory, registry, sg_id, "number", codec)
    pl_read = query_attribute(memory, registry, pl_id, "number", codec)
    sg_vec = torch.from_numpy(codec.filler_vec("number:sg")).to(sg_read.dtype)
    pl_vec = torch.from_numpy(codec.filler_vec("number:pl")).to(pl_read.dtype)

    def cos(a, b):
        return float((a / a.norm()) @ (b / b.norm()))

    assert cos(sg_read, sg_vec) > 0.9
    assert cos(sg_read, pl_vec) < cos(sg_read, sg_vec)
    assert cos(pl_read, pl_vec) > 0.9
    assert cos(pl_read, sg_vec) < cos(pl_read, pl_vec)


# ---------------------------------------------------------------------------
# 4. Group minting: attr:member_0/attr:member_1 recover both members via
#    inverse_query.
# ---------------------------------------------------------------------------
def test_group_member_facts_recover_both_members_via_inverse_query():
    """Registry scoped to just the group (the members' own raw atoms are
    supplied directly, not separately REGISTERED here) -- deliberately, not
    an oversight: attr:member_0/attr:member_1 is a relation only the GROUP
    ever holds (mary/john's own registry slots never get a member_0/
    member_1 write at all), so inverse_query's threshold/argmax over
    OTHER, never-written entities is not a well-posed discrimination task
    for this relation the way it is for attr:kind/attr:gender/attr:number
    (which EVERY instance holds -- see the entity_memory.write/query
    docstrings' own "exact when keys are orthonormal, otherwise noisy"
    caveat: a singly-written (entity, relation) address's cosine-similarity
    readout is, by construction, direction-identical for every OTHER
    entity too, scaled only by that entity's incidental dot product with
    the group atom -- discriminating "who really holds this" from "what
    does an arbitrary unrelated query happen to correlate with" needs a
    relation with multiple independent writes to triangulate against, the
    same reason attr:kind/gender/number are always written for every
    instance in this codebase, never just the referent). This test
    confirms the WRITE/READ mechanism itself (instances.py's own
    write_attribute + inverse_query), which is the milestone's literal
    ask; the registry-scale interference finding above is recorded in the
    report as a seam for a future per-member relation design, not silently
    worked around."""
    dim = 24
    codec = TPRCodec(dim=dim)
    registry = InstanceRegistry(dim=dim, seed=0)
    group_id, _ = registry.mint("group")
    log = ProvenanceLog()
    memory = torch.zeros(dim, dim, dim)

    rng = np.random.default_rng(1)
    m0_atom = rng.standard_normal(dim).astype(np.float32)
    m0_atom /= np.linalg.norm(m0_atom) + 1e-8
    m1_atom = rng.standard_normal(dim).astype(np.float32)
    m1_atom /= np.linalg.norm(m1_atom) + 1e-8

    memory = write_attribute(memory, registry, group_id, "member_0", m0_atom, codec,
                              log=log, source="test", language="en",
                              timestamp=0.0, trust=1.0, value_label="mary")
    memory = write_attribute(memory, registry, group_id, "member_1", m1_atom, codec,
                              log=log, source="test", language="en",
                              timestamp=1.0, trust=1.0, value_label="john")

    ids0, scores0 = inverse_query(memory, registry, codec, "member_0", m0_atom)
    ids1, scores1 = inverse_query(memory, registry, codec, "member_1", m1_atom)
    assert ids0 == [group_id]
    assert ids1 == [group_id]
    assert float(scores0[0]) > 0.9
    assert float(scores1[0]) > 0.9

    # direct addressing also distinguishes WHICH slot holds which member.
    read0 = query_attribute(memory, registry, group_id, "member_0", codec)
    read1 = query_attribute(memory, registry, group_id, "member_1", codec)
    m0_t = torch.from_numpy(m0_atom).to(read0.dtype)
    m1_t = torch.from_numpy(m1_atom).to(read1.dtype)

    def cos(a, b):
        return float((a / a.norm()) @ (b / b.norm()))

    assert cos(read0, m0_t) > 0.9
    assert cos(read0, m1_t) < cos(read0, m0_t)
    assert cos(read1, m1_t) > 0.9
    assert cos(read1, m0_t) < cos(read1, m1_t)


# ---------------------------------------------------------------------------
# 5. RichEpisodeGenerator.plural_frac.
# ---------------------------------------------------------------------------
def test_plural_frac_zero_is_byte_identical_regression():
    a = generate_rich_episodes(20, seed=7)
    b = RichEpisodeGenerator(seed=7, plural_frac=0.0).generate(20)
    assert len(a) == len(b) == 20
    for e1, e2 in zip(a, b):
        assert e1.context == e2.context
        assert e1.question == e2.question
        assert e1.answer_text == e2.answer_text
        assert e1.options == e2.options
        assert e1.answer_idx == e2.answer_idx
        assert e1.meta == e2.meta
        assert "has_group" not in e1.meta or e1.meta["has_group"] is False


def test_plural_frac_zero_never_draws_extra_rng():
    """The `and` short-circuit: at plural_frac == 0.0, RichEpisodeGenerator
    must reproduce IDENTICAL episodes to a generator built before
    plural_frac existed at all (a fresh RNG-state comparison against the
    default-arg call), confirming no extra rng draw was consumed."""
    default = RichEpisodeGenerator(seed=42).generate(15)
    explicit_zero = RichEpisodeGenerator(seed=42, plural_frac=0.0).generate(15)
    for e1, e2 in zip(default, explicit_zero):
        assert e1.meta == e2.meta and e1.context == e2.context


def test_plural_episode_has_group_shape():
    gen = RichEpisodeGenerator(seed=3, plural_frac=0.8, min_entities=3, max_entities=6,
                                inverse_frac=0.0)
    eps = gen.generate(400)
    grp = [e for e in eps if e.meta["has_group"]]
    assert len(grp) > 100, "expected many group episodes at this plural_frac"
    for e in grp:
        m = e.meta
        m0, m1 = m["group_members"]
        assert m0 != m1
        assert m["names"][m0] != m["names"][m1], "group members must have distinct names"
        assert m["group_instance_id"] == "inst:group#1"
        assert m["group_relation"] in ("trait", "mood", "size")
        assert m["group_value"] is not None
        assert f"{m['names'][m0]} and {m['names'][m1]} went to the park ." in e.context
        assert f"they are {m['group_value']} ." in e.context

    target_group = [e for e in grp if e.meta.get("target_is_group")]
    assert target_group, "expected at least one group-targeted question"
    for e in target_group:
        m = e.meta
        m0, m1 = m["group_members"]
        assert e.question == f"what are {m['names'][m0]} and {m['names'][m1]} like ?"
        assert e.answer_text == m["group_value"]
        assert e.options[e.answer_idx] == m["group_value"]
        assert m["target_entity"] is None
        assert m["target_instance_id"] == "inst:group#1"
        assert m["question_targets_overwritten"] is False
        assert m["stale_value_for_question"] is None


def test_plural_episode_zero_mi_device_independent_of_value():
    """The group's selecting evidence (that a plural coordination was used)
    carries no information about which attribute VALUE the group ends up
    holding -- group_relation/group_value are sampled independent of the
    group-forming decision, mirroring the K-statement honesty contract."""
    gen = RichEpisodeGenerator(seed=9, plural_frac=1.0, min_entities=3, max_entities=5,
                                inverse_frac=0.0)
    eps = gen.generate(300)
    grp = [e for e in eps if e.meta["has_group"]]
    values = {e.meta["group_value"] for e in grp}
    assert len(values) > 3, "group_value should vary widely, not collapse to one value"


# ---------------------------------------------------------------------------
# 6. clause_reactor._rich_steps' PLURAL group block.
# ---------------------------------------------------------------------------
def _plural_episodes(seed=5, n=300, **kw):
    gen = RichEpisodeGenerator(seed=seed, plural_frac=0.8, min_entities=3, max_entities=5,
                                min_referring=1, max_referring=2, inverse_frac=0.0, **kw)
    eps = gen.generate(n)
    return [e for e in eps if e.meta["has_group"]]


def test_rich_steps_group_block_step_and_label_alignment():
    meaning = _meaning()
    codec = _codec()
    grp = _plural_episodes()
    assert grp
    for ep in grp[:20]:
        steps, cand_sets, forced_map, atom_lookup, inverse_step_idx = _rich_steps(
            ep, meaning, codec, {}, "usvs")
        labels = _rich_step_labels(ep, cand_sets)
        # every non-question step gets exactly one label, in the same order.
        n_question_steps = 1
        assert len(labels) == len(steps) - n_question_steps
        assert ep.meta["group_instance_id"] in atom_lookup


def test_rich_steps_group_candidate_set_has_number_evidence():
    meaning = _meaning()
    codec = _codec()
    grp = _plural_episodes()
    ep = grp[0]
    steps, cand_sets, forced_map, atom_lookup, inverse_step_idx = _rich_steps(
        ep, meaning, codec, {}, "usvs")
    plural_cs = [cs for cs in cand_sets.values() if cs.provenance.get("device") == "plural_pronoun"]
    assert plural_cs, "expected a plural_pronoun candidate set"
    for cs in plural_cs:
        assert cs.evidence_relation == "number"
        assert cs.evidence_target == "number:pl"
        assert cs.addr_redirect is True
        assert "inst:group#1" in cs.keys
        assert cs.gold_index is not None
        assert cs.keys[cs.gold_index] == "inst:group#1"


def test_batch_build_populates_number_evidence_for_plural_episodes():
    meaning = _meaning()
    codec = _codec()
    grp = _plural_episodes()[:20]
    batch = build_clause_batch(grp, None, meaning, codec)
    assert batch.cand_evidence_relation is not None
    assert batch.cand_evidence_target is not None

    number_rel_vec = torch.from_numpy(codec.filler_vec("attr:number")).float()
    pl_target_vec = torch.from_numpy(codec.filler_vec("number:pl")).float()
    # at least one (row, step) actually carries the "number" evidence
    # relation, grounded exactly as codec.filler_vec("attr:number").
    er = batch.cand_evidence_relation
    matches_number = (F.cosine_similarity(er.reshape(-1, er.shape[-1]), number_rel_vec.unsqueeze(0)) > 0.99)
    assert bool(matches_number.any())
    et = batch.cand_evidence_target
    matches_pl = (F.cosine_similarity(et.reshape(-1, et.shape[-1]), pl_target_vec.unsqueeze(0)) > 0.99)
    assert bool(matches_pl.any())


def test_batch_build_mixed_plural_and_non_plural_rich_episodes():
    """A batch mixing group and non-group rich episodes builds without
    error (Cmax padding, evidence tensors) -- byte-identical machinery
    reused, no new ClauseBatch field."""
    meaning = _meaning()
    codec = _codec()
    gen = RichEpisodeGenerator(seed=13, plural_frac=0.5, min_entities=3, max_entities=6,
                                inverse_frac=0.0)
    eps = gen.generate(80)
    has_group = [e for e in eps if e.meta["has_group"]]
    no_group = [e for e in eps if not e.meta["has_group"]]
    assert has_group and no_group
    batch = build_clause_batch(eps, None, meaning, codec)
    assert batch.entity.shape[0] == len(eps)
    assert batch.cand_entity is not None


# ---------------------------------------------------------------------------
# 7. End-to-end forced-gold eval: "what are A and B like ?" answers via the
#    group node (write gate forced -- mirrors
#    tests/test_rich_episodes.py's own end-to-end pattern exactly).
# ---------------------------------------------------------------------------
def _freeze_write_mechanics(model: ClauseReactor):
    with torch.no_grad():
        model.write_gate.weight.zero_(); model.write_gate.bias.fill_(10.0)
        model.overwrite_gate.weight.zero_(); model.overwrite_gate.bias.fill_(10.0)
        model.decide_truth.weight.zero_(); model.decide_truth.bias.fill_(-10.0)
    for p in (model.write_gate.weight, model.write_gate.bias,
              model.overwrite_gate.weight, model.overwrite_gate.bias,
              model.decide_truth.weight, model.decide_truth.bias):
        p.requires_grad_(False)


def test_end_to_end_forced_gold_plural_question_answers_via_group():
    """Smoke-scale (n_entities fixed at 3, K=1 referring statement) to stay
    inside the agent-ops ≤2.5min budget -- full-batch training over a
    small (~90-row) train set is the dominant cost. The group-targeted
    question's answer must flow through the controller's GRU state across
    MORE intervening steps than an individual "target" question (the
    coordination + group-registration steps sit between the group's
    overwrite and the question) -- the SAME disclosed "recall via hidden
    state, not a literal post-collapse re-read" capability ceiling
    documented in clause_reactor.py's M57c comment for individual target
    questions, just a longer chain; 0.7 (vs 0.25 chance at num_options=4)
    is comfortably above chance without asking for M57c's individual-
    question 0.85 bar this harder, longer-chain task doesn't need to hit.
    """
    meaning = _meaning()
    codec = TPRCodec(dim=24)
    # P(target_is_group) ~= plural_frac * P(rng.random() < 0.5) == 0.9 *
    # 0.5 == 0.45 at inverse_frac=0.0 -- 260 generated comfortably clears
    # the >= 60 floor.
    gen = RichEpisodeGenerator(seed=21, plural_frac=0.9, inverse_frac=0.0,
                                min_entities=3, max_entities=3,
                                min_referring=1, max_referring=1)
    all_eps = gen.generate(260)
    eps = [e for e in all_eps if e.meta.get("target_is_group")]
    assert len(eps) >= 60, "expected enough group-targeted episodes at this seed"
    train_eps, eval_eps = eps[:-20], eps[-20:]

    torch.manual_seed(0)
    model = ClauseReactor(dim=24, hidden=32, resolver=make_resolver("A", 24, 32))
    _freeze_write_mechanics(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.02)

    batch_train = build_clause_batch(train_eps, None, meaning, codec, writeback_force="gold")
    model.train()
    for _ in range(150):
        opt.zero_grad()
        out = model(batch_train)
        loss = F.cross_entropy(out["answer_logits"], batch_train.answer)
        loss.backward()
        opt.step()

    batch_eval = build_clause_batch(eval_eps, None, meaning, codec, writeback_force="gold")
    model.eval()
    with torch.no_grad():
        out = model(batch_eval)
    acc = float((out["answer_logits"].argmax(-1) == batch_eval.answer).float().mean())
    assert acc >= 0.7, acc
