"""Tests for M59b: the CROSS-PASSAGE curriculum for episodic LTM (M59a).

:class:`nsm_ct.curriculum2.DocumentGenerator` (N passage-0 entities,
one cross-passage NAME mention under a 50/50 "same"/"new" condition, one of
three question types) and its reactor-side batch-build path
(:func:`nsm_ct.clause_reactor._document_steps` + ``build_clause_batch``'s
``kind == "document"`` dispatch + ``cand_from_ltm``/``cand_gold`` wiring).
See ``nsm_ct.curriculum2``'s own extensive module comment immediately above
:class:`DocumentGenerator` for the full design and its honesty machinery,
and ``src/nsm_ct/ltm.py``'s module docstring ("Interface contract for the
curriculum agent") for the binding contract this generator's output must
satisfy.

No parser dependency anywhere in this file -- ``_document_steps`` is
parser-free by design (mirrors tests/test_rich_episodes.py's own isolation
discipline).
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch
from nsm_ct.curriculum2 import DocumentGenerator, _verify_unique_referent, generate_document_episodes
from nsm_ct.instances import InstanceRegistry, ProvenanceLog
from nsm_ct.meaning import NSMMeaningResolver
from nsm_ct.resolver import make_resolver
from nsm_ct.tpr import TPRCodec

from _train_common import DocumentRunner, epoch_minibatches  # noqa: E402  (path inserted above)
from test_ltm import _force_full_write_gate  # noqa: E402

DIM = 24


def _meaning():
    return NSMMeaningResolver()


def _codec(dim=DIM):
    return TPRCodec(dim=dim)


def _group_by_document(episodes):
    docs: dict = {}
    for ep in episodes:
        docs.setdefault(ep.meta["doc_id"], []).append(ep)
    for passages in docs.values():
        passages.sort(key=lambda e: e.meta["passage_index"])
    return list(docs.items())


def _build_document(passages, dim, meaning, codec, **kwargs):
    registry = InstanceRegistry(dim=dim, seed=passages[0].meta["instance_seed"])
    batches = [build_clause_batch([ep], None, meaning, codec, document_registry=registry, **kwargs)
               for ep in passages]
    return registry, batches


def _question_step_idx(batch) -> int:
    idx = (batch.is_q[0] > 0).nonzero()
    assert idx.numel() == 1, "the document curriculum's final passage carries exactly one question step"
    return int(idx[0, 0])


# ---------------------------------------------------------------------------
# 1. Generator invariants.
# ---------------------------------------------------------------------------
def test_condition_balance_roughly_50_50():
    eps = generate_document_episodes(400, seed=0)
    docs = _group_by_document(eps)
    conds = Counter(passages[-1].meta["condition"] for _, passages in docs)
    frac_same = conds["same"] / len(docs)
    assert 0.35 < frac_same < 0.65, f"condition split not roughly 50/50: {conds}"


def test_link_uniqueness_assertion_fires_on_a_constructed_collision():
    """_verify_unique_referent (reused from the RICH module's own honesty
    contract) must RAISE on a genuine tie -- the enforcement mechanism
    DocumentGenerator._episode_group calls at generation time, not
    decoration."""
    evidence = {"inst:mary#1": "doctor", "inst:mary#2": "doctor"}
    raised = False
    try:
        _verify_unique_referent("kind", "inst:mary#1", evidence)
    except AssertionError:
        raised = True
    assert raised


def test_link_evidence_uniquely_determines_gold_by_construction():
    """A large sample must generate without ever tripping
    DocumentGenerator's own link-evidence-uniqueness assertion (honesty
    invariant #2) -- i.e. generation itself is the enforcement."""
    generate_document_episodes(300, seed=1)  # would raise AssertionError on a genuine collision


def test_zero_mi_condition_vs_answer_value_empirical():
    """Honesty invariant #1: the realized ANSWER value must not be
    predictable from the CONDITION alone -- checked empirically as
    substantial overlap between "same" and "new" documents' answer-value
    vocabularies (a leak would show up as two near-disjoint vocabularies,
    one condition always answering with values the other never uses)."""
    eps = generate_document_episodes(500, seed=2)
    docs = _group_by_document(eps)
    same_answers = [p[-1].answer_text for _, p in docs if p[-1].meta["condition"] == "same"]
    new_answers = [p[-1].answer_text for _, p in docs if p[-1].meta["condition"] == "new"]
    overlap = set(same_answers) & set(new_answers)
    assert len(overlap) >= 5, "condition and answer value look suspiciously separable"


def test_options_contain_the_competing_value_for_type_ii_and_iii():
    """Type (ii)/(iii) questions must offer the OTHER candidate's own value
    as a genuine "wrong identity" distractor among the 4 options (the
    stale/competing-value contract) -- see DocumentGenerator's module
    comment."""
    eps = generate_document_episodes(200, seed=3)
    docs = _group_by_document(eps)
    checked = 0
    for _, passages in docs:
        m = passages[-1].meta
        opts = passages[-1].options
        if m["question_type"] == "ii":
            assert m["value_before"] in opts
            checked += 1
        elif m["question_type"] == "iii":
            assert m["mention_new_value"] in opts
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# 2. Batch-build satisfies src/nsm_ct/ltm.py's "Interface contract".
# ---------------------------------------------------------------------------
def test_document_batches_share_registry_ids_across_passages():
    meaning, codec = _meaning(), _codec()
    eps = generate_document_episodes(20, seed=4)
    doc_id, passages = _group_by_document(eps)[0]
    registry, batches = _build_document(passages, DIM, meaning, codec)
    for ep in passages:
        assert ep.meta["doc_id"] == doc_id
    for ref_id in passages[0].meta["registry_order"]:
        assert ref_id in registry
    final_meta = passages[-1].meta
    for cid in final_meta["link_candidates"]:
        assert cid in registry, f"{cid} not resolvable in the document's shared registry"


def test_new_candidate_present_and_from_ltm_flags_correct():
    meaning, codec = _meaning(), _codec()
    eps = generate_document_episodes(30, seed=5)
    for doc_id, passages in _group_by_document(eps)[:10]:
        registry, batches = _build_document(passages, DIM, meaning, codec)
        final_batch = batches[-1]
        assert final_batch.cand_mask is not None
        assert final_batch.cand_from_ltm is not None
        _, _, C = final_batch.cand_mask.shape
        assert C == 2, "exactly [referent, NEW] candidates, uniformly, by construction"
        t = int(final_batch.cand_mask[0].sum(-1).nonzero()[0])
        assert final_batch.cand_from_ltm[0, t].tolist() == [1.0, 0.0]


def test_cand_gold_set_and_matches_condition():
    meaning, codec = _meaning(), _codec()
    eps = generate_document_episodes(30, seed=6)
    for doc_id, passages in _group_by_document(eps)[:10]:
        registry, batches = _build_document(passages, DIM, meaning, codec)
        final_batch = batches[-1]
        m = passages[-1].meta
        t = int(final_batch.cand_mask[0].sum(-1).nonzero()[0])
        gold_idx = int(final_batch.cand_gold[0, t])
        expected_idx = 0 if m["condition"] == "same" else 1   # cand_roles = [referent, NEW], always
        assert gold_idx == expected_idx


def test_no_gold_eval_strips_cand_gold():
    meaning, codec = _meaning(), _codec()
    eps = generate_document_episodes(10, seed=9)
    doc_id, passages = _group_by_document(eps)[0]
    registry, batches = _build_document(passages, DIM, meaning, codec, writeback_no_gold=True)
    assert bool((batches[-1].cand_gold < 0).all())


def test_cheat_mode_strips_every_candidate_set():
    meaning, codec = _meaning(), _codec()
    eps = generate_document_episodes(10, seed=10)
    doc_id, passages = _group_by_document(eps)[0]
    registry, batches = _build_document(passages, DIM, meaning, codec, writeback_cheat=True)
    assert all(b.cand_mask is None for b in batches)


def test_force_binding_sets_forced_index():
    meaning, codec = _meaning(), _codec()
    eps = generate_document_episodes(10, seed=11)
    doc_id, passages = _group_by_document(eps)[0]
    registry, batches = _build_document(passages, DIM, meaning, codec, writeback_force="gold")
    final_batch = batches[-1]
    assert final_batch.cand_forced_index is not None
    t = int(final_batch.cand_mask[0].sum(-1).nonzero()[0])
    forced_idx = int(final_batch.cand_forced_index[0, t])
    gold_idx = int(final_batch.cand_gold[0, t])
    assert forced_idx == gold_idx


def test_step_meta_populated_for_every_passage():
    """Critical plumbing (see _document_step_labels' own docstring):
    without step_meta, nsm_ct.provenance.record_writes produces zero
    records, and nsm_ct.ltm.promote never has anything to consolidate."""
    meaning, codec = _meaning(), _codec()
    eps = generate_document_episodes(10, seed=12)
    doc_id, passages = _group_by_document(eps)[0]
    registry, batches = _build_document(passages, DIM, meaning, codec)
    for b in batches:
        assert b.step_meta is not None


# ---------------------------------------------------------------------------
# 3. Forced-gold eval through DocumentRunner (write gates forced full-commit)
#    -- checks the additive mem_read AT the question step directly (return_
#    mem_read, the same "isolate the read mechanism from the untrained
#    respond-timing head" seam tests/test_ltm.py's own e2e test uses),
#    since the document curriculum's final passage carries more than one
#    real step (unlike test_ltm.py's hand-built T=1 batches), so the
#    aggregated `response`/`answer_logits` alone would conflate an untrained
#    respond-timing head with the read mechanism under test.
# ---------------------------------------------------------------------------
def test_forced_gold_type_i_answered_via_ltm_and_fails_with_ltm_zeroed():
    meaning, codec = _meaning(), _codec()
    model = ClauseReactor(dim=DIM, hidden=16)
    _force_full_write_gate(model)
    model.eval()

    eps = generate_document_episodes(100, seed=7)
    type_i_docs = [(doc_id, p) for doc_id, p in _group_by_document(eps)
                    if p[-1].meta["question_type"] == "i"]
    assert len(type_i_docs) >= 5

    cos_with_ltm, cos_zeroed = [], []
    for doc_id, passages in type_i_docs[:10]:
        registry, batches = _build_document(passages, DIM, meaning, codec, writeback_force="gold")
        gold_vec = batches[-1].options[0, int(batches[-1].answer[0])]
        q_idx = _question_step_idx(batches[-1])

        log = ProvenanceLog()
        runner = DocumentRunner(model, trust_ltm=0.0)   # every non-negative gate promotes
        reports, _ = runner.run_document(batches, registry, log, codec, train=False)
        mem_read = reports[-1]["out"]["_mem_read"][0, q_idx]
        cos_with_ltm.append(F.cosine_similarity(mem_read.unsqueeze(0), gold_vec.unsqueeze(0)).item())

        registry2, batches2 = _build_document(passages, DIM, meaning, codec, writeback_force="gold")
        log2 = ProvenanceLog()
        runner_zero = DocumentRunner(model, trust_ltm=0.0, zero_ltm=True)
        reports2, _ = runner_zero.run_document(batches2, registry2, log2, codec, train=False)
        mem_read2 = reports2[-1]["out"]["_mem_read"][0, q_idx]
        cos_zeroed.append(F.cosine_similarity(mem_read2.unsqueeze(0), gold_vec.unsqueeze(0)).item())

    # dim=24 with several entities x several attribute facts superposed in
    # one tensor carries non-negligible cross-term interference (entity_
    # memory's own documented "exact only when keys are orthonormal"
    # caveat, the same one tests/test_ltm.py's promote() test flags) -- 0.8
    # is comfortably above "no signal at all" while tolerating that noise.
    assert np.mean(cos_with_ltm) > 0.8, (
        f"type-(i) recall via the additive LTM read should be near-exact, got mean cos={np.mean(cos_with_ltm):.3f}")
    assert np.mean(cos_with_ltm) - np.mean(cos_zeroed) > 0.5, (
        "zeroing LTM must cost type-(i) recall: it is the ONLY channel carrying the passage-0 fact forward")


def test_forced_gold_type_iii_original_fact_intact():
    """Type-(iii) (condition "new" only): the mention statement's write
    lands on the NEW instance's own address (forced-gold), so the ORIGINAL
    referent's LTM slot under the SAME relation must read back its
    passage-0 value, untouched by the new person's statement."""
    meaning, codec = _meaning(), _codec()
    model = ClauseReactor(dim=DIM, hidden=16)
    _force_full_write_gate(model)
    model.eval()

    eps = generate_document_episodes(200, seed=8)
    type_iii_docs = [(doc_id, p) for doc_id, p in _group_by_document(eps)
                      if p[-1].meta["question_type"] == "iii"]
    assert len(type_iii_docs) >= 3

    for doc_id, passages in type_iii_docs[:8]:
        assert passages[-1].meta["condition"] == "new"
        registry, batches = _build_document(passages, DIM, meaning, codec, writeback_force="gold")
        gold_vec = batches[-1].options[0, int(batches[-1].answer[0])]
        q_idx = _question_step_idx(batches[-1])
        log = ProvenanceLog()
        runner = DocumentRunner(model, trust_ltm=0.0)
        reports, _ = runner.run_document(batches, registry, log, codec, train=False)
        mem_read = reports[-1]["out"]["_mem_read"][0, q_idx]
        cos = F.cosine_similarity(mem_read.unsqueeze(0), gold_vec.unsqueeze(0)).item()
        # same dim=24 interference floor as the type-(i) test above.
        assert cos > 0.8, f"doc {doc_id}: original referent's LTM slot should be intact (cos={cos:.3f})"


# ---------------------------------------------------------------------------
# 4. Short training: link accuracy > 0.8, type-(i) accuracy above chance;
#    the same with LTM zeroed stays at the chance floor.
# ---------------------------------------------------------------------------
def test_short_training_link_and_type_i_accuracy_vs_ltm_zeroed_floor():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../scripts")
    from train_ltm import build_documents, evaluate, run_documents_train

    torch.manual_seed(0)
    meaning, codec = _meaning(), _codec()
    resolver = make_resolver("A", DIM, 16, use_cand_feature=True, cand_feature_extra=2)
    model = ClauseReactor(dim=DIM, hidden=16, resolver=resolver)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    eps = generate_document_episodes(150, seed=0)
    doc_items = _group_by_document(eps)
    n_val = max(1, int(round(len(doc_items) * 0.2)))
    order = np.random.RandomState(0).permutation(len(doc_items))
    val_idx = set(order[:n_val].tolist())
    tr_items = [d for i, d in enumerate(doc_items) if i not in val_idx]
    va_items = [d for i, d in enumerate(doc_items) if i in val_idx]

    tr_docs = build_documents(tr_items, DIM, meaning, codec)
    va_docs = build_documents(va_items, DIM, meaning, codec)

    runner = DocumentRunner(model, trust_ltm=0.0)
    batch_size = 8
    n_steps = 0
    epoch = 0
    while n_steps < 150:
        for mb_idx in epoch_minibatches(len(tr_docs), batch_size, 0, epoch):
            mb_docs = [tr_docs[j] for j in mb_idx]
            run_documents_train(runner, mb_docs, codec, opt)
            n_steps += 1
            if n_steps >= 150:
                break
        epoch += 1

    metrics = evaluate(runner, va_docs, codec)
    assert metrics["link_acc"] is not None and metrics["link_acc"] > 0.8, (
        f"link accuracy should exceed 0.8 after ~150 steps, got {metrics['link_acc']}")
    i_acc, i_n = metrics["by_qtype"].get("i", (0.0, 0))
    assert i_n > 0
    assert i_acc > 0.25, f"type-(i) accuracy should sit above chance (0.25), got {i_acc:.3f} (n={i_n})"

    runner_zero = DocumentRunner(model, trust_ltm=0.0, zero_ltm=True)
    metrics_zero = evaluate(runner_zero, va_docs, codec)
    i_acc_zero, i_n_zero = metrics_zero["by_qtype"].get("i", (0.0, 0))
    assert i_n_zero > 0
    assert i_acc_zero < 0.45, (
        f"zeroing LTM should leave type-(i) at the chance floor, got {i_acc_zero:.3f} (n={i_n_zero})")
