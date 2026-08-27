"""Tests for M57c: instance-atom candidate sets + definite-description
referring expressions + the two-Marys curriculum.

Integrates M57a's InstanceRegistry (nsm_ct.instances) with M57b's proven
resolver-driven write-back mechanism (nsm_ct.clause_reactor._writeback_steps /
ClauseReactor's address-redirect collapse). See dev/MIND_INTERFACE.md's "v2
addendum -- the entity-instance subsystem", CLAUDE.md's M57 memory-schema
decision, and nsm_ct.clause_reactor._instance_steps's own extensive
docstring for the full design.

No parser dependency anywhere in this file -- _instance_steps is parser-free
by design (mirrors tests/test_writeback.py's own isolation discipline), and
the mechanics tests build ClauseBatch objects by hand.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from nsm_ct import entity_memory as em
from nsm_ct.clause_reactor import (
    ClauseBatch,
    ClauseReactor,
    _content_vec,
    _instance_option_vec,
    _instance_steps,
    build_clause_batch,
)
from nsm_ct.curriculum2 import (
    InstanceCurriculumGenerator,
    generate_instance_episodes,
    generate_writeback_episodes,
)
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.instances import InstanceRegistry
from nsm_ct.meaning import NSMMeaningResolver
from nsm_ct.membrane import EntityCandidateSet
from nsm_ct.resolver import make_resolver
from nsm_ct.tpr import TPRCodec

DIM = 32


def _meaning():
    return NSMMeaningResolver()


def _codec(dim=DIM):
    return TPRCodec(dim=dim)


# ---------------------------------------------------------------------------
# 1. Byte-identity regression extension: cand_evidence_relation absent.
# ---------------------------------------------------------------------------
def test_cand_evidence_relation_is_none_for_every_pre_m57c_batch():
    """A batch built from ordinary (non-instance) episodes must leave
    ``cand_evidence_relation`` ``None`` -- the SAME "fourth optional field
    to guard" discipline cand_addr_mask/cand_forced_index already
    established. Covers old L1-6 AND writeback (M57b) episodes -- neither
    generator's candidate sets ever populate
    ``EntityCandidateSet.evidence_relation``."""
    meaning = _meaning()
    codec = _codec()
    old_eps = CurriculumGenerator(max_level=6, seed=0).generate(10)
    batch_old = build_clause_batch(old_eps, None, meaning, codec)
    assert batch_old.cand_evidence_relation is None

    wb_eps = generate_writeback_episodes(10, seed=0)
    batch_wb = build_clause_batch(wb_eps, None, meaning, codec)
    assert batch_wb.cand_evidence_relation is None


def test_forward_output_byte_identical_with_and_without_evidence_relation_field_when_none():
    """A hand-built batch with ``cand_evidence_relation`` explicitly absent
    (dataclass default), explicitly ``None``, must produce IDENTICAL model
    output to each other -- the no-op guard in ClauseReactor._collapse's
    entity branch (``r if batch.cand_evidence_relation is None else ...``)."""
    torch.manual_seed(0)
    b, d, C = 5, 16, 3
    g = torch.Generator().manual_seed(1)
    entity = F.normalize(torch.randn(b, 2, d, generator=g), dim=-1)
    relation = F.normalize(torch.randn(b, 2, d, generator=g), dim=-1)
    value = F.normalize(torch.randn(b, 2, d, generator=g), dim=-1)
    pred = F.normalize(torch.randn(b, 2, d, generator=g), dim=-1)
    is_q = torch.tensor([[0.0, 1.0]] * b)
    mask = torch.ones(b, 2)
    options = F.normalize(torch.randn(b, 4, d, generator=g), dim=-1)
    answer = torch.randint(0, 4, (b,), generator=g)
    cand_entity = F.normalize(torch.randn(b, 2, C, d, generator=g), dim=-1)
    cand_mask = torch.ones(b, 2, C)
    cand_prior = torch.full((b, 2, C), 1.0 / C)
    cand_feature = torch.zeros(b, 2, 6)
    cand_gold = torch.full((b, 2), -1, dtype=torch.long)
    cand_addr_mask = torch.zeros(b, 2)

    kwargs = dict(entity=entity, relation=relation, value=value, pred=pred, is_q=is_q,
                  mask=mask, options=options, answer=answer, cand_entity=cand_entity,
                  cand_mask=cand_mask, cand_prior=cand_prior, cand_feature=cand_feature,
                  cand_gold=cand_gold, cand_addr_mask=cand_addr_mask)

    model = ClauseReactor(dim=d, hidden=8, resolver=make_resolver("A", d, 8))
    model.eval()

    batch_absent = ClauseBatch(**kwargs)
    assert batch_absent.cand_evidence_relation is None
    with torch.no_grad():
        out_absent = model(batch_absent)

    batch_none = ClauseBatch(**kwargs, cand_evidence_relation=None)
    with torch.no_grad():
        out_none = model(batch_none)

    for k in out_absent:
        assert torch.equal(out_absent[k], out_none[k]), k


# ---------------------------------------------------------------------------
# 2. Evidence-relation mechanics: does cand_mem_read really come from the
#    evidence relation instead of the step's own relation?
# ---------------------------------------------------------------------------
def test_evidence_relation_replaces_step_relation_in_collapse():
    """Hand-built: candidate 0's attr:kind slot and its (unrelated) step
    relation slot hold DIFFERENT values in memory. Force the collapse
    weight onto candidate 0 (``cand_forced_index``) and compare the
    resolved value against BOTH: with no evidence relation, the resolved
    value must match the STEP relation's readout; with an evidence
    relation set, it must match THAT relation's readout instead -- a
    direct, resolver-behavior-independent check (forcing bypasses the
    resolver's own logits entirely, so this isolates exactly the one line
    ClauseReactor._collapse's entity branch changed)."""
    torch.manual_seed(0)
    d = 16
    cand0 = F.normalize(torch.randn(1, d), dim=-1)
    cand1 = F.normalize(torch.randn(1, d), dim=-1)
    evidence_rel = F.normalize(torch.randn(1, d), dim=-1)
    step_rel = F.normalize(torch.randn(1, d), dim=-1)
    vec_evidence = F.normalize(torch.randn(1, d), dim=-1)
    vec_step = F.normalize(torch.randn(1, d), dim=-1)

    memory = em.init_memory(1, d, "cpu")
    gate = torch.ones(1)
    memory = em.write(memory, cand0, evidence_rel, vec_evidence, gate)
    memory = em.write(memory, cand0, step_rel, vec_step, gate)

    model = ClauseReactor(dim=d, hidden=8, resolver=make_resolver("A", d, 8))
    state = torch.zeros(1, 8)
    e = torch.zeros(1, d)
    r = step_rel
    v = torch.zeros(1, d)
    mem_read = torch.zeros(1, d)

    cand_entity = torch.stack([cand0, cand1], dim=1)
    common = dict(
        entity=e.unsqueeze(1), relation=r.unsqueeze(1), value=v.unsqueeze(1),
        pred=torch.zeros(1, 1, d), is_q=torch.zeros(1, 1), mask=torch.ones(1, 1),
        options=torch.zeros(1, 1, d), answer=torch.zeros(1, dtype=torch.long),
        cand_entity=cand_entity.unsqueeze(1), cand_mask=torch.ones(1, 1, 2),
        cand_prior=torch.full((1, 1, 2), 0.5), cand_feature=torch.zeros(1, 1, 6),
        cand_gold=torch.full((1, 1), -1, dtype=torch.long),
        cand_forced_index=torch.tensor([[0]]),
    )

    batch_step = ClauseBatch(**common)
    _e2, v_step, *_ = model._collapse(memory, state, mem_read, e, r, v, batch_step, 0)
    assert F.cosine_similarity(v_step, vec_step).item() > 0.9
    assert F.cosine_similarity(v_step, vec_evidence).item() < 0.5

    batch_evidence = ClauseBatch(**common, cand_evidence_relation=evidence_rel.unsqueeze(1))
    _e2, v_ev, *_ = model._collapse(memory, state, mem_read, e, r, v, batch_evidence, 0)
    assert F.cosine_similarity(v_ev, vec_evidence).item() > 0.9
    assert F.cosine_similarity(v_ev, vec_step).item() < 0.5


# ---------------------------------------------------------------------------
# 3. Two-Marys minting: same name -> distinct atoms; candidate set for the
#    shared name contains exactly the two name-matched instances.
# ---------------------------------------------------------------------------
def test_two_marys_minting_distinct_atoms_and_restricted_candidate_set():
    meaning = _meaning()
    codec = _codec()
    gen = InstanceCurriculumGenerator(seed=3, inverse_frac=0.0)
    amb_eps = [e for e in gen.generate(80) if e.meta["referring_device"] == "ambiguous_name"]
    assert amb_eps, "expected at least one ambiguous_name episode at this seed"
    ep = amb_eps[0]

    steps, cand_sets, forced_map, atom_lookup = _instance_steps(ep, meaning, codec, {}, "usvs")
    t = next(iter(cand_sets))
    cs = cand_sets[t]
    assert isinstance(cs, EntityCandidateSet)
    assert cs.evidence_relation == "kind"
    assert cs.addr_redirect is True
    # restricted to EXACTLY the two name-matched instances -- never the
    # third, distinctly-named instance.
    assert len(cs.keys) == 2
    assert set(cs.keys) == {f"inst:{ep.meta['shared_name']}#1", f"inst:{ep.meta['shared_name']}#2"}

    atom1 = torch.from_numpy(atom_lookup[cs.keys[0]])
    atom2 = torch.from_numpy(atom_lookup[cs.keys[1]])
    cos = F.cosine_similarity(atom1.unsqueeze(0), atom2.unsqueeze(0)).item()
    assert abs(cos) < 0.5, f"mary#1/mary#2 should be near-orthogonal, cos={cos}"
    assert cs.gold_index in (0, 1)
    assert cs.keys[cs.gold_index] == ep.meta["gold_instance_id"]


def test_instance_registry_determinism_matches_curriculum_bookkeeping():
    """The registry _instance_steps mints, seeded from
    ``ep.meta["instance_seed"]``, must reproduce the EXACT ids
    InstanceCurriculumGenerator predicted in ``registry_order`` -- the
    torch/codec-free curriculum module never touches an actual atom, only
    the deterministic id-format string (mirrors InstanceRegistry.mint's own
    ``inst:<name>#<n>`` convention)."""
    meaning = _meaning()
    codec = _codec()
    eps = generate_instance_episodes(15, seed=7)
    for ep in eps:
        steps, cand_sets, forced_map, atom_lookup = _instance_steps(ep, meaning, codec, {}, "usvs")
        assert sorted(atom_lookup.keys()) == sorted(ep.meta["registry_order"])
        # independent re-mint with the SAME seed reproduces the SAME atoms.
        reg = InstanceRegistry(dim=codec.dim, seed=ep.meta["instance_seed"])
        id_a, atom_a = reg.mint(ep.meta["shared_name"])
        id_b, atom_b = reg.mint(ep.meta["shared_name"])
        id_c, atom_c = reg.mint(ep.meta["distinct_name"])
        assert [id_a, id_b, id_c] == ep.meta["registry_order"]
        assert torch.allclose(atom_a, torch.from_numpy(atom_lookup[id_a]))
        assert torch.allclose(atom_b, torch.from_numpy(atom_lookup[id_b]))
        assert torch.allclose(atom_c, torch.from_numpy(atom_lookup[id_c]))


# ---------------------------------------------------------------------------
# 4. Memory semantics with forcing (return_memory).
# ---------------------------------------------------------------------------
def _force_full_write_gate(model: ClauseReactor):
    with torch.no_grad():
        model.write_gate.weight.zero_(); model.write_gate.bias.fill_(10.0)
        model.overwrite_gate.weight.zero_(); model.overwrite_gate.bias.fill_(10.0)
        model.decide_truth.weight.zero_(); model.decide_truth.bias.fill_(-10.0)


def _referent_atoms(ep, dim):
    reg = InstanceRegistry(dim=dim, seed=ep.meta["instance_seed"])
    ids_atoms = {}
    id_a, atom_a = reg.mint(ep.meta["shared_name"])
    id_b, atom_b = reg.mint(ep.meta["shared_name"])
    id_c, atom_c = reg.mint(ep.meta["distinct_name"])
    ids_atoms["a"], ids_atoms["b"], ids_atoms["c"] = atom_a, atom_b, atom_c
    return ids_atoms


def test_forced_gold_and_wrong_eval_memory_semantics_for_definite_description_and_pronoun():
    """Forced-gold: the redirect lands on the TRUE referent's node, which
    then holds the OVERWRITE value; the referent's stale baseline is gone.
    Forced-wrong: the referent's node keeps its stale baseline (untouched);
    the OTHER candidate's node gets clobbered with the overwrite value
    instead. Restricted to definite_description/pronoun devices (3
    candidates, "wrong" = ``(true_idx + 1) % 3`` per _instance_steps) --
    ambiguous_name's 2-candidate "wrong" case is covered separately below."""
    meaning = _meaning()
    codec = _codec()
    gen = InstanceCurriculumGenerator(seed=11, inverse_frac=0.0)
    eps = [e for e in gen.generate(200) if e.meta["referring_device"] != "ambiguous_name"][:8]

    model = ClauseReactor(dim=DIM, hidden=16, resolver=make_resolver("A", DIM, 16))
    model.eval()
    _force_full_write_gate(model)
    trait_rel = torch.from_numpy(codec.filler_vec("attr:trait"))

    for force in ("gold", "wrong"):
        batch = build_clause_batch(eps, None, meaning, codec, writeback_force=force)
        with torch.no_grad():
            out = model(batch, return_memory=True)
        memory = out["_memory"]
        for i, ep in enumerate(eps):
            atoms = _referent_atoms(ep, DIM)
            ref = ep.meta["referent_role"]
            roles = ["a", "b", "c"]
            true_idx = roles.index(ref)
            wrong_role = roles[(true_idx + 1) % 3]

            overwrite_vec = torch.from_numpy(_content_vec(ep.meta["overwrite_attr"], meaning, codec, {}, "usvs"))
            stale_vec = torch.from_numpy(_content_vec(ep.meta["stale_attr"], meaning, codec, {}, "usvs"))

            read_referent = em.query(memory[i:i + 1], atoms[ref].unsqueeze(0), trait_rel.unsqueeze(0))
            if force == "gold":
                assert F.cosine_similarity(read_referent, overwrite_vec.unsqueeze(0)).item() > 0.9, (i, ep.meta)
            else:
                assert F.cosine_similarity(read_referent, stale_vec.unsqueeze(0)).item() > 0.9, (i, ep.meta)
                # the WRONG node got clobbered with the overwrite value instead
                # (mirrors tests/test_writeback.py's own convention: assert the
                # new value reads back strongly; with 13 facts superposed in one
                # dim-32 tensor here -- more than writeback's 5 -- interference
                # keeps the OLD value's cosine from cleanly vanishing, so this
                # does not also assert it goes to ~0, only that the overwrite
                # value now dominates).
                read_wrong = em.query(memory[i:i + 1], atoms[wrong_role].unsqueeze(0), trait_rel.unsqueeze(0))
                assert F.cosine_similarity(read_wrong, overwrite_vec.unsqueeze(0)).item() > 0.9, (i, ep.meta)


def test_forced_gold_and_wrong_eval_memory_semantics_for_ambiguous_name():
    """Same as above, restricted to ambiguous_name episodes (2-candidate
    "wrong" = the OTHER name-matched instance -- the two-Marys pair's own
    write-address confusability, the whole point of this milestone)."""
    meaning = _meaning()
    codec = _codec()
    gen = InstanceCurriculumGenerator(seed=13, inverse_frac=0.0)
    eps = [e for e in gen.generate(200) if e.meta["referring_device"] == "ambiguous_name"][:6]
    assert eps

    model = ClauseReactor(dim=DIM, hidden=16, resolver=make_resolver("A", DIM, 16))
    model.eval()
    _force_full_write_gate(model)
    trait_rel = torch.from_numpy(codec.filler_vec("attr:trait"))

    for force in ("gold", "wrong"):
        batch = build_clause_batch(eps, None, meaning, codec, writeback_force=force)
        with torch.no_grad():
            out = model(batch, return_memory=True)
        memory = out["_memory"]
        for i, ep in enumerate(eps):
            atoms = _referent_atoms(ep, DIM)
            ref = ep.meta["referent_role"]
            other = "b" if ref == "a" else "a"
            overwrite_vec = torch.from_numpy(_content_vec(ep.meta["overwrite_attr"], meaning, codec, {}, "usvs"))
            stale_vec = torch.from_numpy(_content_vec(ep.meta["stale_attr"], meaning, codec, {}, "usvs"))
            other_stale_vec = torch.from_numpy(
                _content_vec(ep.meta[f"baseline_{other}"], meaning, codec, {}, "usvs"))

            read_referent = em.query(memory[i:i + 1], atoms[ref].unsqueeze(0), trait_rel.unsqueeze(0))
            read_other = em.query(memory[i:i + 1], atoms[other].unsqueeze(0), trait_rel.unsqueeze(0))
            if force == "gold":
                assert F.cosine_similarity(read_referent, overwrite_vec.unsqueeze(0)).item() > 0.9
                assert F.cosine_similarity(read_other, other_stale_vec.unsqueeze(0)).item() > 0.9
            else:
                assert F.cosine_similarity(read_referent, stale_vec.unsqueeze(0)).item() > 0.9
                assert F.cosine_similarity(read_other, overwrite_vec.unsqueeze(0)).item() > 0.9


# ---------------------------------------------------------------------------
# 5. Generator sanity.
# ---------------------------------------------------------------------------
def test_generator_answer_key_and_option_shapes():
    eps = generate_instance_episodes(60, seed=4)
    for ep in eps:
        assert ep.meta["kind"] == "instance"
        assert ep.options is not None and ep.answer_idx is not None
        assert 0 <= ep.answer_idx < len(ep.options)
        assert ep.options[ep.answer_idx] == ep.answer_text
        if ep.meta["question_mode"] == "target":
            assert ep.meta["stale_attr"] in ep.options   # stale attr always among the options
            attrs = {ep.meta["baseline_a"], ep.meta["baseline_b"], ep.meta["baseline_c"], ep.meta["overwrite_attr"]}
            assert len(attrs) == 4                        # distinct attrs throughout
        else:
            assert len(ep.options) == 3
            assert len(set(ep.options)) == 3               # unambiguous: no duplicate option strings


def test_question_targets_and_referring_devices_are_roughly_uniform():
    eps = generate_instance_episodes(600, seed=5, inverse_frac=0.3)
    target_eps = [e for e in eps if e.meta["question_mode"] == "target"]
    inverse_eps = [e for e in eps if e.meta["question_mode"] == "inverse"]
    assert abs(len(inverse_eps) / len(eps) - 0.3) < 0.08

    targets = [e.meta["target_role"] for e in target_eps]
    for role in ("a", "b", "c"):
        frac = targets.count(role) / len(targets)
        assert 0.25 < frac < 0.42, (role, frac)   # ~1/3 uniform, generous tolerance

    devices = [e.meta["referring_device"] for e in eps]
    for device in ("definite_description", "pronoun", "ambiguous_name"):
        frac = devices.count(device) / len(devices)
        assert 0.25 < frac < 0.42, (device, frac)


def test_gender_tie_half_of_episodes():
    """a/b share gender in EXACTLY half of episodes (parity alternation,
    not a statistical draw)."""
    eps = generate_instance_episodes(40, seed=6, inverse_frac=0.0)
    same = sum(1 for e in eps if e.meta["same_gender_pair"])
    assert same == len(eps) // 2


def test_kind_always_disambiguates_gender_sometimes_does():
    """Every episode's three kinds are pairwise distinct (definite
    description is always unambiguous); gender ties (a/b sharing gender)
    occur in some but not all episodes."""
    eps = generate_instance_episodes(40, seed=8, inverse_frac=0.0)
    for ep in eps:
        kinds = {ep.meta["kind_a"], ep.meta["kind_b"], ep.meta["kind_c"]}
        assert len(kinds) == 3
    genders_a = [e.meta["gender_a"] for e in eps]
    genders_b = [e.meta["gender_b"] for e in eps]
    ties = sum(1 for ga, gb in zip(genders_a, genders_b) if ga == gb)
    assert 0 < ties < len(eps)


def test_zero_mi_referring_device_independent_of_answer_correctness():
    """Zero-MI-by-construction check (mirrors WriteBackCurriculumGenerator's
    own documented argument): which referring device was sampled must carry
    no information about whether the question targets the referent (the
    thing that determines answerability) -- both are independent draws."""
    eps = generate_instance_episodes(1200, seed=9, inverse_frac=0.0)
    by_device = {}
    for e in eps:
        by_device.setdefault(e.meta["referring_device"], []).append(e.meta["question_targets_referent"])
    overall = sum(e.meta["question_targets_referent"] for e in eps) / len(eps)
    for device, flags in by_device.items():
        frac = sum(flags) / len(flags)
        assert abs(frac - overall) < 0.12, (device, frac, overall)


def test_inverse_query_options_unambiguous_and_answer_correct():
    eps = generate_instance_episodes(200, seed=10, inverse_frac=1.0)
    for ep in eps:
        assert ep.meta["question_mode"] == "inverse"
        assert len(set(ep.options)) == 3   # no duplicate identity strings
        answer_role = ep.meta["answer_role"]
        shared, distinct = ep.meta["shared_name"], ep.meta["distinct_name"]
        if answer_role == "c":
            expected = distinct
        else:
            kind = ep.meta[f"kind_{answer_role}"]
            expected = f"{shared} the {kind}"
        assert ep.answer_text == expected
        # the query trait is never the referent's own overwritten-away stale value.
        assert ep.meta["query_trait"] != ep.meta["stale_attr"] or ep.meta["answer_role"] == ep.meta["referent_role"]


def test_instance_option_vec_disambiguates_two_marys():
    meaning = _meaning()
    codec = _codec()
    v_doctor = _instance_option_vec("mary the doctor", meaning, codec, {}, "usvs")
    v_teacher = _instance_option_vec("mary the teacher", meaning, codec, {}, "usvs")
    v_john = _instance_option_vec("john", meaning, codec, {}, "usvs")
    cos = np.dot(v_doctor, v_teacher) / (np.linalg.norm(v_doctor) * np.linalg.norm(v_teacher) + 1e-8)
    assert cos < 0.9   # kind-qualified options for the SAME name must be distinguishable
    assert not np.allclose(v_doctor, v_john)


# ---------------------------------------------------------------------------
# 6. Forced-index population (cand_forced_index) for instance episodes.
# ---------------------------------------------------------------------------
def test_instance_force_binding_gold_and_wrong_populate_cand_forced_index():
    meaning = _meaning()
    codec = _codec()
    eps = generate_instance_episodes(20, seed=12, inverse_frac=0.0)
    batch = build_clause_batch(eps, None, meaning, codec)
    batch_g = build_clause_batch(eps, None, meaning, codec, writeback_force="gold")
    batch_w = build_clause_batch(eps, None, meaning, codec, writeback_force="wrong")

    assert batch.cand_forced_index is None
    assert batch_g.cand_forced_index is not None
    has_cand = batch.cand_gold >= 0 if batch.cand_gold is not None else None
    # wherever a candidate set exists, forced-gold must equal cand_gold and
    # forced-wrong must differ from it.
    cand_gold = build_clause_batch(eps, None, meaning, codec).cand_gold
    mask = cand_gold >= 0
    assert torch.equal(batch_g.cand_forced_index[mask], cand_gold[mask])
    assert bool((batch_w.cand_forced_index[mask] != cand_gold[mask]).all())


def test_full_batch_forward_backward_smoke():
    """End-to-end: build a mixed instance-episode batch, run a full
    forward+backward pass through a resolver-installed ClauseReactor --
    catches any shape/dtype mismatch the narrower unit tests above miss."""
    meaning = _meaning()
    codec = _codec()
    eps = generate_instance_episodes(24, seed=14, inverse_frac=0.3)
    batch = build_clause_batch(eps, None, meaning, codec)
    model = ClauseReactor(dim=DIM, hidden=24, resolver=make_resolver("A", DIM, 24))
    out = model(batch)
    loss = F.cross_entropy(out["answer_logits"], batch.answer)
    loss.backward()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Mixed-kind batch regression (director, post-review): writeback episodes
# sharing a batch with instance episodes must behave EXACTLY as they do in a
# writeback-only batch. The build tensor is zero at candidate steps whose set
# carries no evidence_relation; the per-(row, step) fallback in _collapse must
# substitute the step relation there, NOT query memory with the zero vector.
# ---------------------------------------------------------------------------
def test_mixed_batch_writeback_rows_unaffected_by_evidence_relation_tensor():
    meaning = _meaning()
    codec = _codec()
    torch.manual_seed(0)
    wb_eps = generate_writeback_episodes(6, seed=3)
    inst_eps = generate_instance_episodes(6, seed=3)

    model = ClauseReactor(DIM, hidden=32, resolver=make_resolver("A", DIM))
    model.eval()

    batch_wb = build_clause_batch(wb_eps, None, meaning, codec)
    assert batch_wb.cand_evidence_relation is None          # control premise
    with torch.no_grad():
        out_wb = model(batch_wb)

    batch_mix = build_clause_batch(wb_eps + inst_eps, None, meaning, codec)
    assert batch_mix.cand_evidence_relation is not None     # instance rows present
    with torch.no_grad():
        out_mix = model(batch_mix.subset(torch.arange(len(wb_eps))))

    # The subset keeps the (now non-None) evidence tensor, so this compares
    # "field present but zero at writeback steps" against "field absent".
    # The mixed batch pads candidates to the instance episodes' C=3; the
    # writeback rows' real candidates live in the first 2 slots (the padding
    # slot is masked to -1e9 and never wins), so compare those slots only.
    # (Same for T: the mixed batch pads steps to the instance episodes'
    # longer length; the writeback rows' real steps are the first T_wb.)
    T_wb, C = out_wb["resolver_logits"].shape[1], out_wb["resolver_logits"].shape[-1]
    assert torch.allclose(out_wb["resolver_logits"],
                          out_mix["resolver_logits"][:, :T_wb, :C], atol=1e-6)
    assert torch.allclose(out_wb["answer_logits"], out_mix["answer_logits"], atol=1e-6)
