"""Tests for M59a: episodic LONG-TERM MEMORY (nsm_ct.ltm +
scripts._train_common.DocumentRunner). See CLAUDE.md's "LTM decisions" and
dev/LTM_DESIGN_BRIEF.md Sec.5 (locked design) for the contract these tests
hold the implementation to: a separate LTM tensor, additive reads
(``memory + ltm``), STM-only writes, a tier-generic ``promote`` op, and the
wind-down/consolidate substate machine.

No curriculum generator exists yet for multi-passage documents (M59b's
job) -- every multi-passage/document test here hand-builds its own
:class:`~nsm_ct.clause_reactor.ClauseBatch` objects directly, the same
"synthetic-tensor test" discipline ``tests/test_instances.py`` and
``tests/test_provenance_wiring.py``'s ``_synthetic_batch_and_trace`` helper
already use.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from nsm_ct import entity_memory as em
from nsm_ct.clause_reactor import ClauseBatch, ClauseReactor, build_clause_batch
from nsm_ct.curriculum2 import InstanceCurriculumGenerator
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.instances import InstanceRegistry, ProvenanceLog, ProvenanceRecord
from nsm_ct.ltm import LINK_THRESHOLD, NEW, TRUST_LTM, link_decision, mem_total, promote
from nsm_ct.meaning import NSMMeaningResolver
from nsm_ct.resolver import make_resolver
from nsm_ct.tpr import TPRCodec

from _train_common import DocumentRunner  # noqa: E402  (path inserted above)

DIM = 16


def _codec(dim=DIM) -> TPRCodec:
    return TPRCodec(dim=dim)


def _meaning():
    return NSMMeaningResolver()


def _force_full_write_gate(model: ClauseReactor) -> None:
    """Force the write gate to ~1 (full commit) and the decide-truth
    (refute) gate to ~0 -- mirrors tests/test_instances.py's own helper of
    the same name: every statement step then writes with trust~1.0,
    deterministically, with no training required."""
    with torch.no_grad():
        model.write_gate.weight.zero_(); model.write_gate.bias.fill_(10.0)
        model.overwrite_gate.weight.zero_(); model.overwrite_gate.bias.fill_(10.0)
        model.decide_truth.weight.zero_(); model.decide_truth.bias.fill_(-10.0)


def _force_identity_response(model: ClauseReactor, dim: int, hidden: int) -> None:
    """Force ``response(cat[state, mem_read]) == mem_read`` exactly (zero
    the ``state`` half of the weight matrix, identity on the ``mem_read``
    half, zero bias) -- lets a test read the generated response vector
    directly off ``mem_read`` with NO training required, isolating the
    additive-read mechanism from the (otherwise untrained/random) response
    head."""
    with torch.no_grad():
        w = torch.cat([torch.zeros(dim, hidden), torch.eye(dim)], dim=1)
        model.response.weight.copy_(w)
        model.response.bias.zero_()


def _one_step_batch(entity_vec: torch.Tensor, rel_vec: torch.Tensor, val_vec: torch.Tensor, *,
                     is_q: bool, options: torch.Tensor, answer: int = 0,
                     step_meta_entry: dict = None) -> ClauseBatch:
    """A hand-built, single-(row, step) ClauseBatch -- no resolver, no
    candidate sets (mirrors ``tests/test_provenance_wiring.py``'s
    ``_synthetic_batch_and_trace``). ``options`` is ``[K, d]``."""
    d = entity_vec.shape[-1]
    ent = entity_vec.view(1, 1, d)
    rel = rel_vec.view(1, 1, d)
    val = val_vec.view(1, 1, d)
    pred = torch.zeros(1, 1, d)
    is_q_t = torch.tensor([[1.0 if is_q else 0.0]])
    mask = torch.ones(1, 1)
    opts = options.unsqueeze(0)                    # [1, K, d]
    ans = torch.tensor([answer], dtype=torch.long)
    step_meta = [[step_meta_entry]] if step_meta_entry is not None else None
    return ClauseBatch(entity=ent, relation=rel, value=val, pred=pred, is_q=is_q_t,
                        mask=mask, options=opts, answer=ans, step_meta=step_meta)


def _write_step_meta(instance_id: str, relation: str, value_label: str, surface: str) -> dict:
    return {
        "sentence_index": 0, "surface": surface, "relation_label": relation,
        "value_label": value_label, "entity_label": instance_id, "candidate_ids": [],
        "referring_device": None, "episode_index": 0,
    }


# ---------------------------------------------------------------------------
# 1. mem_total -- the additive read view, byte-identical when ltm=None.
# ---------------------------------------------------------------------------
def test_mem_total_none_returns_memory_unchanged():
    memory = torch.randn(2, DIM, DIM, DIM)
    assert mem_total(memory, None) is memory


def test_mem_total_adds_ltm():
    memory = torch.randn(2, DIM, DIM, DIM)
    ltm = torch.randn(2, DIM, DIM, DIM)
    out = mem_total(memory, ltm)
    assert torch.equal(out, memory + ltm)


# ---------------------------------------------------------------------------
# 2. forward(): ltm=None (default) is byte-identical to pre-M59a, with and
#    without a resolver installed (exercises the widened _collapse extra-
#    column path with cand_from_ltm absent).
# ---------------------------------------------------------------------------
def test_forward_ltm_none_byte_identical_no_resolver():
    old_eps = CurriculumGenerator(max_level=6, seed=0).generate(6)
    meaning, codec = _meaning(), _codec(24)
    batch = build_clause_batch(old_eps, None, meaning, codec)
    assert batch.cand_from_ltm is None
    model = ClauseReactor(dim=24, hidden=16)
    model.eval()
    with torch.no_grad():
        out_default = model(batch)
        out_explicit_none = model(batch, ltm=None)
    for k in out_default:
        if torch.is_tensor(out_default[k]):
            assert torch.equal(out_default[k], out_explicit_none[k]), k


def test_forward_ltm_none_byte_identical_with_resolver_and_evidence():
    meaning, codec = _meaning(), _codec(24)
    eps = InstanceCurriculumGenerator(seed=5, inverse_frac=0.3).generate(15)
    batch = build_clause_batch(eps, None, meaning, codec)
    assert batch.cand_from_ltm is None
    assert batch.cand_evidence_target is not None   # exercises the s_c-only extra_cols path
    resolver = make_resolver("A", 24, 16, use_cand_feature=True, cand_feature_extra=1)
    model = ClauseReactor(dim=24, hidden=16, resolver=resolver)
    model.eval()
    with torch.no_grad():
        out_default = model(batch)
        out_explicit_none = model(batch, ltm=None)
    for k in out_default:
        if torch.is_tensor(out_default[k]):
            assert torch.equal(out_default[k], out_explicit_none[k]), k

    # Sanity: a REAL ltm tensor actually changes the output (mem_total
    # isn't silently ignored when set).
    b = batch.entity.shape[0]
    ltm = torch.randn(b, 24, 24, 24) * 0.1
    with torch.no_grad():
        out_ltm = model(batch, ltm=ltm)
    assert not torch.equal(out_ltm["answer_logits"], out_default["answer_logits"])


# ---------------------------------------------------------------------------
# 3. ClauseBatch.cand_from_ltm: field-position alignment through to()/
#    subset(), and it actually reaches (and moves) the resolver's logits.
# ---------------------------------------------------------------------------
def test_cand_from_ltm_to_and_subset_alignment():
    meaning, codec = _meaning(), _codec(24)
    eps = InstanceCurriculumGenerator(seed=6, inverse_frac=0.0).generate(6)
    batch = build_clause_batch(eps, None, meaning, codec)
    b, T, Cmax = batch.cand_mask.shape
    from_ltm = (torch.rand(b, T, Cmax) > 0.5).float() * batch.cand_mask
    batch = dataclasses.replace(batch, cand_from_ltm=from_ltm)

    moved = batch.to("cpu")
    assert torch.equal(moved.cand_from_ltm, batch.cand_from_ltm)

    idx = torch.tensor([2, 0])
    sub = batch.subset(idx)
    assert torch.equal(sub.cand_from_ltm, batch.cand_from_ltm[idx])


def test_cand_from_ltm_widens_resolver_feature_register():
    torch.manual_seed(0)
    meaning, codec = _meaning(), _codec(24)
    eps = InstanceCurriculumGenerator(seed=7, inverse_frac=0.3).generate(10)
    batch = build_clause_batch(eps, None, meaning, codec)
    assert batch.cand_evidence_target is not None
    b, T, Cmax = batch.cand_mask.shape

    resolver = make_resolver("A", 24, 16, use_cand_feature=True, cand_feature_extra=2)
    model = ClauseReactor(dim=24, hidden=16, resolver=resolver)
    model.eval()

    zeros = torch.zeros(b, T, Cmax)
    ones = batch.cand_mask.clone()   # 1 on every real candidate, 0 on padding
    batch_zero = dataclasses.replace(batch, cand_from_ltm=zeros)
    batch_one = dataclasses.replace(batch, cand_from_ltm=ones)
    with torch.no_grad():
        out_zero = model(batch_zero)
        out_one = model(batch_one)
    assert not torch.allclose(out_zero["resolver_logits"], out_one["resolver_logits"])


# ---------------------------------------------------------------------------
# 4. Additive read: a fact written ONLY via a passage that's now gone (fresh
#    STM this passage), consolidated into LTM, is read at a question step.
# ---------------------------------------------------------------------------
def test_additive_read_ltm_only_fact_via_document_runner():
    codec = _codec()
    registry = InstanceRegistry(dim=DIM, seed=0)
    iid, atom = registry.mint("mary")
    rel = torch.from_numpy(codec.filler_vec("attr:kind")).float()
    doctor = torch.from_numpy(codec.filler_vec("val:doctor")).float()

    model = ClauseReactor(dim=DIM, hidden=8)
    _force_full_write_gate(model)
    model.eval()

    meta = _write_step_meta(iid, "attr:kind", "doctor", "mary is a doctor .")
    batch1 = _one_step_batch(atom, rel, doctor, is_q=False, options=doctor.unsqueeze(0),
                              step_meta_entry=meta)

    log = ProvenanceLog()
    runner = DocumentRunner(model, trust_ltm=0.5)
    reports, _ = runner.run_document([batch1], registry, log, codec, train=False)
    assert reports[0]["n_records"] == 1
    assert reports[0]["n_promoted"] == 1
    ltm_after_p1 = reports[0]["ltm"]

    # Passage 2: a question step at the SAME (entity, relation) address.
    # forward() always starts STM at zero -- this passage's own memory
    # carries NOTHING about mary's kind; only the additive ltm read can
    # supply it.
    batch2 = _one_step_batch(atom, rel, torch.zeros(DIM), is_q=True, options=doctor.unsqueeze(0))
    with torch.no_grad():
        out2 = model(batch2, ltm=ltm_after_p1, return_mem_read=True)
    mem_read = out2["_mem_read"][0, 0]
    cos = F.cosine_similarity(mem_read.unsqueeze(0), doctor.unsqueeze(0)).item()
    assert cos > 0.99


# ---------------------------------------------------------------------------
# 5. promote: trust gate, last-write-wins dedup, provenance tier tag.
# ---------------------------------------------------------------------------
def _write_fact(mem, entity, relation, value):
    return em.write(mem.unsqueeze(0), entity.unsqueeze(0), relation.unsqueeze(0),
                     value.unsqueeze(0), torch.tensor([1.0])).squeeze(0)


def test_promote_trust_gate_and_provenance_tag():
    codec = _codec()
    registry = InstanceRegistry(dim=DIM, seed=1)
    mary_id, mary = registry.mint("mary")
    john_id, john = registry.mint("john")

    rel_trait = torch.from_numpy(codec.filler_vec("attr:trait")).float()
    rel_kind = torch.from_numpy(codec.filler_vec("attr:kind")).float()
    old_val = torch.from_numpy(codec.filler_vec("val:old")).float()
    doctor = torch.from_numpy(codec.filler_vec("val:doctor")).float()

    source = torch.zeros(DIM, DIM, DIM)
    source = _write_fact(source, mary, rel_trait, old_val)     # mary's LAST (only) trait write
    source = _write_fact(source, john, rel_kind, doctor)

    rec_mary = ProvenanceRecord(instance_id=mary_id, relation="attr:trait", value_label="old",
                                 source="reactor", language="en", timestamp=1.0, trust=0.2, step=1)
    rec_john = ProvenanceRecord(instance_id=john_id, relation="attr:kind", value_label="doctor",
                                 source="reactor", language="en", timestamp=0.0, trust=0.8, step=0)

    target = torch.zeros(DIM, DIM, DIM)
    log = ProvenanceLog()
    new_target, n = promote(source, target, registry, log, records=[rec_mary, rec_john],
                             dial=TRUST_LTM, dial_name="trust_ltm", codec=codec, timestamp=5.0)

    assert n == 1   # mary's trust (0.2) < TRUST_LTM (0.5) -- not promoted
    assert len(log) == 1
    assert log.records[0].source == "promote:trust_ltm"
    assert log.records[0].instance_id == john_id
    assert log.records[0].value_label == "doctor"

    q_john = em.query(new_target.unsqueeze(0), john.unsqueeze(0), rel_kind.unsqueeze(0)).squeeze(0)
    assert F.cosine_similarity(q_john.unsqueeze(0), doctor.unsqueeze(0)).item() > 0.99
    # mary was never promoted -- her slot reads back only cross-term
    # interference from john's real write (entity_memory's documented
    # "exact only when keys are orthonormal" caveat, non-negligible at this
    # test's small dim), never anything resembling her (unpromoted) value.
    q_mary = em.query(new_target.unsqueeze(0), mary.unsqueeze(0), rel_trait.unsqueeze(0)).squeeze(0)
    assert F.cosine_similarity(q_mary.unsqueeze(0), old_val.unsqueeze(0)).item() < 0.3


def test_promote_last_write_wins():
    codec = _codec()
    registry = InstanceRegistry(dim=DIM, seed=2)
    iid, atom = registry.mint("mary")
    rel = torch.from_numpy(codec.filler_vec("attr:trait")).float()
    young = torch.from_numpy(codec.filler_vec("val:young")).float()
    old = torch.from_numpy(codec.filler_vec("val:old")).float()

    source = torch.zeros(DIM, DIM, DIM)
    source = _write_fact(source, atom, rel, young)
    source = _write_fact(source, atom, rel, old)   # the memory's CURRENT slot content is "old"

    rec_young = ProvenanceRecord(instance_id=iid, relation="attr:trait", value_label="young",
                                  source="reactor", language="en", timestamp=0.0, trust=0.9, step=0)
    rec_old = ProvenanceRecord(instance_id=iid, relation="attr:trait", value_label="old",
                                source="reactor", language="en", timestamp=1.0, trust=0.6, step=1)

    target = torch.zeros(DIM, DIM, DIM)
    log = ProvenanceLog()
    new_target, n = promote(source, target, registry, log, records=[rec_young, rec_old],
                             dial=0.5, dial_name="trust_ltm", codec=codec, timestamp=2.0)

    assert n == 1                                    # ONE (entity, relation) slot, deduplicated
    assert log.records[0].value_label == "old"        # the LAST write's label, not the first
    assert log.records[0].trust == 0.6

    q = em.query(new_target.unsqueeze(0), atom.unsqueeze(0), rel.unsqueeze(0)).squeeze(0)
    assert F.cosine_similarity(q.unsqueeze(0), old.unsqueeze(0)).item() > 0.99
    assert F.cosine_similarity(q.unsqueeze(0), young.unsqueeze(0)).item() < 0.5


# ---------------------------------------------------------------------------
# 6. Genericity: the SAME promote() serves a later tier (LTM->Truth) with a
#    stateful corroboration-COUNT criterion instead of the trust default.
# ---------------------------------------------------------------------------
def test_promote_genericity_counting_criterion():
    codec = _codec()
    registry = InstanceRegistry(dim=DIM, seed=3)
    iid, atom = registry.mint("mary")
    rel = torch.from_numpy(codec.filler_vec("attr:kind")).float()
    doctor = torch.from_numpy(codec.filler_vec("val:doctor")).float()

    source = torch.zeros(DIM, DIM, DIM)
    source = _write_fact(source, atom, rel, doctor)
    rec = ProvenanceRecord(instance_id=iid, relation="attr:kind", value_label="doctor",
                            source="reactor", language="en", timestamp=0.0, trust=0.1, step=0)

    counts: Counter = Counter()
    need = 2

    def corroboration_criterion(record: ProvenanceRecord) -> bool:
        key = (record.instance_id, record.relation)
        counts[key] += 1
        return counts[key] >= need

    target = torch.zeros(DIM, DIM, DIM)
    log = ProvenanceLog()

    # First "sighting" -- trust is low (0.1, would fail the DEFAULT
    # trust-gate criterion too) and the counting criterion hasn't yet seen
    # this fact twice.
    target, n1 = promote(source, target, registry, log, records=[rec], dial=need,
                          dial_name="trust_truth", criterion=corroboration_criterion,
                          codec=codec, timestamp=0.0)
    assert n1 == 0
    assert len(log) == 0

    # Second sighting (e.g. a later passage corroborating the same fact) --
    # SAME criterion closure, so its internal count is now 2 -- promoted.
    target, n2 = promote(source, target, registry, log, records=[rec], dial=need,
                          dial_name="trust_truth", criterion=corroboration_criterion,
                          codec=codec, timestamp=1.0)
    assert n2 == 1
    assert log.records[-1].source == "promote:trust_truth"

    q = em.query(target.unsqueeze(0), atom.unsqueeze(0), rel.unsqueeze(0)).squeeze(0)
    assert F.cosine_similarity(q.unsqueeze(0), doctor.unsqueeze(0)).item() > 0.99


# ---------------------------------------------------------------------------
# 7. DocumentRunner.ltm_detach bounds autograd.
# ---------------------------------------------------------------------------
def test_document_runner_ltm_detach_bounds_grad():
    codec = _codec()
    rel = torch.from_numpy(codec.filler_vec("attr:kind")).float()
    doctor = torch.from_numpy(codec.filler_vec("val:doctor")).float()

    def _one_passage_batch(registry):
        iid, atom = registry.mint("mary")
        meta = _write_step_meta(iid, "attr:kind", "doctor", "mary is a doctor .")
        return _one_step_batch(atom, rel, doctor, is_q=False, options=doctor.unsqueeze(0),
                                step_meta_entry=meta)

    model = ClauseReactor(dim=DIM, hidden=8)

    reg_detach = InstanceRegistry(dim=DIM, seed=0)
    runner_detach = DocumentRunner(model, ltm_detach=True)
    reports_detach, _ = runner_detach.run_document(
        [_one_passage_batch(reg_detach)], reg_detach, ProvenanceLog(), codec, train=True)
    assert reports_detach[0]["ltm"].requires_grad is False

    reg_attached = InstanceRegistry(dim=DIM, seed=0)
    runner_attached = DocumentRunner(model, ltm_detach=False)
    reports_attached, _ = runner_attached.run_document(
        [_one_passage_batch(reg_attached)], reg_attached, ProvenanceLog(), codec, train=True)
    assert reports_attached[0]["ltm"].requires_grad is True


# ---------------------------------------------------------------------------
# 8. link_decision: deterministic threshold semantics.
# ---------------------------------------------------------------------------
def test_link_decision_threshold_semantics():
    probs = torch.tensor([[0.6, 0.4], [0.3, 0.3], [0.5, 0.1]])
    idx = link_decision(probs, threshold=0.5)
    assert idx.tolist() == [0, NEW, 0]   # row 0 links; row 1 abstains (NEW); row 2 is AT threshold (>=)


def test_link_decision_default_threshold_is_module_dial():
    probs = torch.tensor([0.5, 0.1])
    assert int(link_decision(probs)) == 0
    assert LINK_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# 9. Cross-passage recall end to end, plus the cheat control (ltm zeroed).
# ---------------------------------------------------------------------------
def test_cross_passage_recall_e2e_and_cheat_control():
    codec = _codec()
    registry = InstanceRegistry(dim=DIM, seed=4)
    iid, atom = registry.mint("mary")
    rel = torch.from_numpy(codec.filler_vec("attr:kind")).float()
    doctor = torch.from_numpy(codec.filler_vec("val:doctor")).float()
    teacher = torch.from_numpy(codec.filler_vec("val:teacher")).float()

    model = ClauseReactor(dim=DIM, hidden=8)
    _force_full_write_gate(model)
    _force_identity_response(model, DIM, 8)
    model.eval()

    meta = _write_step_meta(iid, "attr:kind", "doctor", "mary is a doctor .")
    batch1 = _one_step_batch(atom, rel, doctor, is_q=False, options=doctor.unsqueeze(0),
                              step_meta_entry=meta)

    log = ProvenanceLog()
    runner = DocumentRunner(model, trust_ltm=0.5)
    reports, _ = runner.run_document([batch1], registry, log, codec, train=False)
    assert reports[0]["n_promoted"] == 1
    real_ltm = reports[0]["ltm"]

    options = torch.stack([doctor, teacher], dim=0)   # gold = doctor, index 0
    batch2 = _one_step_batch(atom, rel, torch.zeros(DIM), is_q=True, options=options, answer=0)

    with torch.no_grad():
        out_real = model(batch2, ltm=real_ltm)
    r_real = out_real["response"][0]
    cos_real = F.cosine_similarity(r_real.unsqueeze(0), doctor.unsqueeze(0)).item()
    assert cos_real > 0.99
    assert int(out_real["answer_logits"][0].argmax()) == 0   # correctly answers "doctor"

    # Cheat control: same passage-2 question, LTM zeroed out -- no STM
    # (fresh this passage) and no LTM either, so there is nothing to
    # recall the fact from.
    zero_ltm = torch.zeros_like(real_ltm)
    with torch.no_grad():
        out_cheat = model(batch2, ltm=zero_ltm)
    r_cheat = out_cheat["response"][0]
    cos_cheat = F.cosine_similarity(r_cheat.unsqueeze(0), doctor.unsqueeze(0)).item()
    assert cos_cheat < 0.3
    assert cos_real - cos_cheat > 0.5


# ---------------------------------------------------------------------------
# 10. Smoke (optional, cheap): a hand-built 2-passage loop, loss decreases.
# ---------------------------------------------------------------------------
def test_smoke_two_passage_loss_decreases():
    torch.manual_seed(0)
    codec = _codec(12)
    model = ClauseReactor(dim=12, hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    registry = InstanceRegistry(dim=12, seed=0)
    iid, atom = registry.mint("mary")
    rel = torch.from_numpy(codec.filler_vec("attr:kind")).float()
    doctor = torch.from_numpy(codec.filler_vec("val:doctor")).float()
    teacher = torch.from_numpy(codec.filler_vec("val:teacher")).float()
    options = torch.stack([doctor, teacher], dim=0)
    meta = _write_step_meta(iid, "attr:kind", "doctor", "mary is a doctor .")

    def loss_fn(out, batch):
        return F.cross_entropy(out["answer_logits"], batch.answer)

    losses = []
    log = ProvenanceLog()
    runner = DocumentRunner(model, trust_ltm=0.0)   # dial=0.0: every non-negative gate promotes
    for _ in range(50):
        batch1 = _one_step_batch(atom, rel, doctor, is_q=False, options=options, answer=0,
                                  step_meta_entry=meta)
        batch2 = _one_step_batch(atom, rel, torch.zeros(12), is_q=True, options=options, answer=0)
        reports, total_loss = runner.run_document(
            [batch1, batch2], registry, log, codec, train=True, loss_fn=loss_fn)
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        losses.append(float(total_loss.detach()))

    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    assert late < early
