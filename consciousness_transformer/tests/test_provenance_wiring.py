"""Tests for M57d: wiring PROVENANCE into ClauseReactor's live reactor
writes (dev/MIND_INTERFACE.md invariant #4, CLAUDE.md's M57 memory-schema
decision). See src/nsm_ct/provenance.py's own module docstring for the
full design: ClauseReactor.forward's ``return_write_trace`` flag,
ClauseBatch's ``step_meta`` field, and this module's record_writes/
explain/overwrites_for read/write pair over nsm_ct.instances.ProvenanceLog.
"""

from __future__ import annotations

import torch

from nsm_ct.clause_reactor import ClauseBatch, ClauseReactor, build_clause_batch
from nsm_ct.curriculum2 import InstanceCurriculumGenerator, RichEpisodeGenerator
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.instances import ProvenanceLog
from nsm_ct.meaning import NSMMeaningResolver
from nsm_ct.provenance import explain, overwrites_for, record_writes
from nsm_ct.resolver import make_resolver
from nsm_ct.tpr import TPRCodec

DIM = 24


def _meaning():
    return NSMMeaningResolver()


def _codec(dim=DIM):
    return TPRCodec(dim=dim)


def _force_full_write_gate(model: ClauseReactor):
    """Force the write gate to ~1 (full commit) and the decide-truth
    (refute) gate to ~0 -- mirrors tests/test_instance_curriculum.py's own
    helper of the same name: every statement step then writes with
    trust~1.0, deterministically, with no training required."""
    with torch.no_grad():
        model.write_gate.weight.zero_(); model.write_gate.bias.fill_(10.0)
        model.overwrite_gate.weight.zero_(); model.overwrite_gate.bias.fill_(10.0)
        model.decide_truth.weight.zero_(); model.decide_truth.bias.fill_(-10.0)


# ---------------------------------------------------------------------------
# 1. Byte-identity: return_write_trace defaults False and is a complete
#    no-op on every other output key.
# ---------------------------------------------------------------------------
def test_return_write_trace_default_false_byte_identical():
    meaning = _meaning()
    codec = _codec()
    eps = InstanceCurriculumGenerator(seed=1, inverse_frac=0.3).generate(20)
    batch = build_clause_batch(eps, None, meaning, codec)
    model = ClauseReactor(dim=DIM, hidden=16, resolver=make_resolver("A", DIM, 16))
    model.eval()
    with torch.no_grad():
        out_default = model(batch)
        out_explicit_false = model(batch, return_write_trace=False)
        out_true = model(batch, return_write_trace=True)

    assert "_write_trace" not in out_default
    assert "_write_trace" not in out_explicit_false
    assert "_write_trace" in out_true

    for k in out_default:
        assert torch.equal(out_default[k], out_explicit_false[k]), k
        assert torch.equal(out_default[k], out_true[k]), k


# ---------------------------------------------------------------------------
# 2. Write-trace shapes/dtypes.
# ---------------------------------------------------------------------------
def test_write_trace_shapes_and_dtypes():
    meaning = _meaning()
    codec = _codec()
    eps = InstanceCurriculumGenerator(seed=2, inverse_frac=0.0).generate(10)
    batch = build_clause_batch(eps, None, meaning, codec)
    model = ClauseReactor(dim=DIM, hidden=16, resolver=make_resolver("A", DIM, 16))
    model.eval()
    with torch.no_grad():
        out = model(batch, return_write_trace=True)
    trace = out["_write_trace"]
    b, T = batch.entity.shape[0], batch.entity.shape[1]
    for key in ("gate", "overwrite", "neg", "redirected", "resolved_index"):
        assert trace[key].shape == (b, T), key
    assert trace["gate"].is_floating_point()
    assert trace["overwrite"].is_floating_point()
    assert trace["neg"].is_floating_point()
    assert trace["redirected"].dtype == torch.bool
    assert trace["resolved_index"].dtype == torch.long
    assert bool((trace["gate"] >= 0).all()) and bool((trace["gate"] <= 1).all())
    assert bool((trace["resolved_index"] >= -1).all())


# ---------------------------------------------------------------------------
# 3. record_writes is a no-op (returns 0) whenever step_meta or the write
#    trace is absent -- the "absent optional data" discipline every other
#    ClauseBatch/forward() field in this codebase already follows.
# ---------------------------------------------------------------------------
def test_record_writes_noop_without_step_meta_or_trace():
    meaning = _meaning()
    codec = _codec()

    old_eps = CurriculumGenerator(max_level=6, seed=0).generate(5)
    batch_old = build_clause_batch(old_eps, None, meaning, codec)
    assert batch_old.step_meta is None
    model = ClauseReactor(dim=DIM, hidden=16)
    model.eval()
    with torch.no_grad():
        out = model(batch_old, return_write_trace=True)   # trace present, but no step_meta anywhere
    log = ProvenanceLog()
    assert record_writes(batch_old, out, log, source="s") == 0

    inst_eps = InstanceCurriculumGenerator(seed=10, inverse_frac=0.0).generate(5)
    batch_inst = build_clause_batch(inst_eps, None, meaning, codec)
    assert batch_inst.step_meta is not None
    model2 = ClauseReactor(dim=DIM, hidden=16, resolver=make_resolver("A", DIM, 16))
    model2.eval()
    with torch.no_grad():
        out2 = model2(batch_inst)   # no return_write_trace -- no "_write_trace" key
    log2 = ProvenanceLog()
    assert record_writes(batch_inst, out2, log2, source="s") == 0


# ---------------------------------------------------------------------------
# 4. Forced-gold eval on a 2-Marys ("ambiguous_name") instance episode, write
#    gate forced to ~1: exactly one record per statement step; the redirected
#    overwrite step carries the RESOLVED (gold referent) instance id; the
#    gold referent's own attr:trait records show the named baseline THEN the
#    overwrite in time order; overwrites_for reproduces that same sequence;
#    explain() contains the overwrite sentence's own surface text.
# ---------------------------------------------------------------------------
def test_forced_gold_two_marys_audit_trail():
    meaning = _meaning()
    codec = _codec()
    gen = InstanceCurriculumGenerator(seed=3, inverse_frac=0.0)
    amb_eps = [e for e in gen.generate(80) if e.meta["referring_device"] == "ambiguous_name"]
    assert amb_eps, "expected at least one ambiguous_name episode at this seed"
    ep = amb_eps[0]

    model = ClauseReactor(dim=DIM, hidden=16, resolver=make_resolver("A", DIM, 16))
    model.eval()
    _force_full_write_gate(model)

    # writeback_force="gold" TEACHER-FORCES the overwrite step's collapse to
    # the TRUE referent's candidate index (nsm_ct.clause_reactor._instance_
    # steps) -- and M57d's resolved_idx_out (ClauseReactor._collapse) is
    # defined from the collapse weights ACTUALLY APPLIED (post-force), so
    # this is deterministic with no training required: the recorded
    # resolved instance MUST be the gold referent by construction.
    batch = build_clause_batch([ep], None, meaning, codec, writeback_force="gold")
    with torch.no_grad():
        out = model(batch, return_write_trace=True)

    log = ProvenanceLog()
    n = record_writes(batch, out, log, source="test")

    n_statement_steps = sum(1 for m in batch.step_meta[0] if m is not None)
    assert n_statement_steps == 13   # 3 roles * (kind+gender+place) + 3 baselines + 1 overwrite
    assert n == n_statement_steps    # exactly one record per statement step (gate forced ~1 > 0)

    referent = ep.meta["referent_role"]
    gold_id = ep.meta["gold_instance_id"]
    assert gold_id == f"inst:{ep.meta['shared_name']}#{1 if referent == 'a' else 2}"

    all_records = log.records_for(gold_id)
    assert [r.relation for r in all_records] == ["attr:kind", "attr:gender", "attr:place",
                                                   "attr:trait", "attr:trait"]

    trait_records = [r for r in all_records if r.relation == "attr:trait"]
    assert len(trait_records) == 2
    assert trait_records[0].value_label == ep.meta[f"baseline_{referent}"]        # the named baseline...
    assert trait_records[1].value_label == ep.meta["overwrite_attr"]              # ...THEN the overwrite
    assert trait_records[1].candidate_ids is not None
    assert set(trait_records[1].candidate_ids) == {
        f"inst:{ep.meta['shared_name']}#1", f"inst:{ep.meta['shared_name']}#2"}

    assert overwrites_for(log, gold_id, "attr:trait") == [
        ep.meta[f"baseline_{referent}"], ep.meta["overwrite_attr"]]

    trail = explain(log, gold_id)
    assert ep.context[-1] in trail          # the overwrite sentence's own surface text
    assert "resolved from [" in trail       # the redirected overwrite line names its candidates


# ---------------------------------------------------------------------------
# 5. Rich episode (8 entities / 4 referring statements): record count ==
#    number of statement steps.
# ---------------------------------------------------------------------------
def test_rich_episode_record_count_matches_statement_steps():
    meaning = _meaning()
    codec = _codec()
    gen = RichEpisodeGenerator(seed=6, min_entities=8, max_entities=8,
                                min_referring=4, max_referring=4, inverse_frac=0.0)
    eps = gen.generate(5)
    ep = eps[0]
    assert ep.meta["n_entities"] == 8
    assert ep.meta["n_referring_statements"] == 4

    model = ClauseReactor(dim=DIM, hidden=16, resolver=make_resolver("A", DIM, 16))
    model.eval()
    _force_full_write_gate(model)

    batch = build_clause_batch([ep], None, meaning, codec, writeback_force="gold")
    with torch.no_grad():
        out = model(batch, return_write_trace=True)

    log = ProvenanceLog()
    n = record_writes(batch, out, log, source="rich-test")
    n_statement_steps = sum(1 for m in batch.step_meta[0] if m is not None)
    assert n_statement_steps > 0
    assert n == n_statement_steps


# ---------------------------------------------------------------------------
# 6. trust_threshold filters low-gate writes; language round-trips.
# ---------------------------------------------------------------------------
def _synthetic_batch_and_trace():
    step_meta = [[
        {"sentence_index": 0, "surface": "the thing is tall .", "relation_label": "attr:trait",
         "value_label": "tall", "entity_label": "inst:x#1", "candidate_ids": [],
         "referring_device": None, "episode_index": 0},
        {"sentence_index": 1, "surface": "the thing is short .", "relation_label": "attr:trait",
         "value_label": "short", "entity_label": "inst:x#1", "candidate_ids": [],
         "referring_device": None, "episode_index": 0},
    ]]
    batch = ClauseBatch(
        entity=torch.zeros(1, 2, 4), relation=torch.zeros(1, 2, 4), value=torch.zeros(1, 2, 4),
        pred=torch.zeros(1, 2, 4), is_q=torch.zeros(1, 2), mask=torch.ones(1, 2),
        options=torch.zeros(1, 1, 4), answer=torch.zeros(1, dtype=torch.long),
        step_meta=step_meta,
    )
    out = {"_write_trace": {
        "gate": torch.tensor([[0.9, 0.1]]),
        "overwrite": torch.tensor([[0.9, 0.1]]),
        "neg": torch.zeros(1, 2),
        "redirected": torch.zeros(1, 2, dtype=torch.bool),
        "resolved_index": torch.full((1, 2), -1, dtype=torch.long),
    }}
    return batch, out


def test_trust_threshold_filters_low_gate_writes():
    batch, out = _synthetic_batch_and_trace()

    log_low = ProvenanceLog()
    assert record_writes(batch, out, log_low, source="s", trust_threshold=0.0) == 2

    log_high = ProvenanceLog()
    assert record_writes(batch, out, log_high, source="s", trust_threshold=0.5) == 1
    assert log_high.records[0].value_label == "tall"   # the gate=0.9 step only


def test_language_field_round_trips():
    batch, out = _synthetic_batch_and_trace()
    log = ProvenanceLog()
    record_writes(batch, out, log, source="s", language="es")
    assert len(log.records) == 2
    assert all(r.language == "es" for r in log.records)


def test_explain_no_records_placeholder():
    log = ProvenanceLog()
    msg = explain(log, "inst:nobody#1")
    assert "inst:nobody#1" in msg


# ---------------------------------------------------------------------------
# 7. subset() preserves step_meta row alignment; to() carries it through
#    unchanged.
# ---------------------------------------------------------------------------
def test_subset_and_to_preserve_step_meta_alignment():
    meaning = _meaning()
    codec = _codec()
    eps = InstanceCurriculumGenerator(seed=9, inverse_frac=0.0).generate(6)
    batch = build_clause_batch(eps, None, meaning, codec)
    assert batch.step_meta is not None

    idx = torch.tensor([2, 0, 4])
    sub = batch.subset(idx)
    assert sub.step_meta is not None
    for j, i in enumerate(idx.tolist()):
        assert sub.step_meta[j] == batch.step_meta[i]

    moved = batch.to("cpu")
    assert moved.step_meta == batch.step_meta
