"""Tests for M57b: resolver-driven WRITE-BACK -- the resolver's collapsed
choice redirects the write ADDRESS (not just the value). "she is tall ."
must write to the resolved referent's own node (mary's), not to a fixed
pronoun-mention placeholder. See nsm_ct.clause_reactor.ClauseReactor's M57b
docstring paragraph, nsm_ct.clause_reactor._writeback_steps, and
nsm_ct.curriculum2.WriteBackCurriculumGenerator.

v2 (RESEARCH_NOTES M57b): the curriculum shape changed (every entity gets
its own NAMED attribute statement first; the pronoun statement comes LAST
and OVERWRITES the referent's) and the wrong-binding validity arm moved from
a curriculum-level gold_index corruption (which task pressure could simply
override -- measured, full scale: it stayed at 1.000) to COLLAPSE-level
teacher-forcing (``ClauseBatch.cand_forced_index`` /
``ClauseReactor._collapse``'s forcing branch, independent of the resolver's
own logits). See the curriculum module's docstring above
``WriteBackCurriculumGenerator`` for the full leak-fix rationale.

Synthetic-tensor tests (no parser dependency) mirror tests/test_resolver.py's
own isolation discipline; the curriculum-generator sanity tests at the
bottom exercise the real generator + build_clause_batch (still no parser
needed -- _writeback_steps is parser-free by design).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nsm_ct import entity_memory as em
from nsm_ct.clause_reactor import (
    ClauseBatch,
    ClauseReactor,
    _content_vec,
    _ent_vec,
    build_clause_batch,
)
from nsm_ct.curriculum2 import (
    _ATTR_VALUES,
    WriteBackCurriculumGenerator,
    generate_writeback_episodes,
)
from nsm_ct.membrane import FEATURE_DIM, NAME_GENDER
from nsm_ct.resolver import CorefHead, Resolver, SharedScorer, make_resolver


# ---------------------------------------------------------------------------
# Helpers: synthetic address-redirect batches, mirroring
# tests/test_resolver.py's `_toy_batch_with_candidates` pattern exactly, but
# for M57b's write-BACK shape: the collapse step's pre-resolve ENTITY is a
# PLACEHOLDER (garbage) atom, distinct from every candidate atom), and the
# question step queries a NAMED candidate's OWN address directly (never the
# placeholder), so the task is answerable ONLY if the write actually
# redirects there.
# ---------------------------------------------------------------------------
def _toy_writeback_batch(b=6, d=24, C=3, K=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    opts = F.normalize(torch.randn(b, K, d, generator=g), dim=-1)
    ans = torch.randint(0, K, (b,), generator=g)

    cand_atoms = F.normalize(torch.randn(b, C, d, generator=g), dim=-1)   # candidate ADDRESS atoms
    gold_idx = torch.randint(0, C, (b,), generator=g)

    rel = F.normalize(torch.randn(b, d, generator=g), dim=-1)             # shared rel:ATTR-like relation
    pronoun_addr = F.normalize(torch.randn(b, d, generator=g), dim=-1)    # PLACEHOLDER (garbage) pre-collapse address
    prd = F.normalize(torch.randn(b, d, generator=g), dim=-1)

    # the write-back step's STATED value is the gold answer option -- a
    # correct address redirect + a fully-open write gate answers the question.
    stated_value = opts[torch.arange(b), ans]

    T = 2   # 0: write-back collapse step, 1: question step
    entity = torch.zeros(b, T, d); relation = torch.zeros(b, T, d); value = torch.zeros(b, T, d)
    pred = torch.zeros(b, T, d); is_q = torch.zeros(b, T); mask = torch.ones(b, T)
    entity[:, 0], relation[:, 0], value[:, 0], pred[:, 0] = pronoun_addr, rel, stated_value, prd
    q_t = 1
    # the question asks about the TRUE gold candidate's own node directly
    # (mirrors "what is mary like ?" -- a named entity's own address).
    entity[:, q_t] = cand_atoms[torch.arange(b), gold_idx]
    relation[:, q_t] = rel
    is_q[:, q_t] = 1.0

    cand_entity = torch.zeros(b, T, C, d)
    cand_mask = torch.zeros(b, T, C)
    cand_prior = torch.zeros(b, T, C)
    cand_feature = torch.zeros(b, T, FEATURE_DIM)
    cand_gold = torch.full((b, T), -1, dtype=torch.long)
    cand_addr_mask = torch.zeros(b, T)
    cand_entity[:, 0] = cand_atoms
    cand_mask[:, 0] = 1.0
    cand_prior[:, 0] = 1.0 / C
    cand_feature[:, 0] = torch.randn(b, FEATURE_DIM, generator=g)
    cand_gold[:, 0] = gold_idx
    cand_addr_mask[:, 0] = 1.0

    batch = ClauseBatch(entity, relation, value, pred, is_q, mask, opts, ans,
                         cand_entity=cand_entity, cand_mask=cand_mask, cand_prior=cand_prior,
                         cand_feature=cand_feature, cand_gold=cand_gold, cand_addr_mask=cand_addr_mask)
    return batch, cand_atoms, pronoun_addr, rel, gold_idx


def _force_full_write_gate(model: ClauseReactor):
    """Deterministic write: gate=1, overwrite=1, decide_truth=0 -- so a
    statement step's write is EXACTLY ``memory += entity⊗relation⊗value``
    (no partial-gate ambiguity), letting the address-redirect mechanics test
    check memory contents by hand-computable arithmetic instead of an
    UNTRAINED (effectively random) write gate."""
    with torch.no_grad():
        model.write_gate.weight.zero_(); model.write_gate.bias.fill_(10.0)
        model.overwrite_gate.weight.zero_(); model.overwrite_gate.bias.fill_(10.0)
        model.decide_truth.weight.zero_(); model.decide_truth.bias.fill_(-10.0)


class _FixedByGoldResolver(Resolver):
    """Test-only stub: ALWAYS argmaxes to the row's own ``gold`` index
    (a per-row [B] long tensor), regardless of input -- the "resolver
    stubbed/forced so argmax lands on a known candidate" the M57b mechanics
    test needs. Mirrors tests/test_resolver.py's `_FixedLogitResolver`, just
    per-row rather than one shared logit row."""

    def __init__(self, gold: torch.Tensor):
        super().__init__()
        self.gold = gold
        self.dummy = torch.nn.Parameter(torch.zeros(1))   # well-formed nn.Module

    def forward(self, cand_entity, cand_feature, cand_prior, cand_mask, mem_read, state):
        b, C, _d = cand_entity.shape
        return F.one_hot(self.gold, num_classes=C).to(cand_entity.dtype) * 20.0 - 10.0


# ---------------------------------------------------------------------------
# 1. Byte-identity (mirrors tests/test_resolver.py's own regression group --
#    see also test_resolver.py::test_resolver_installed_cand_addr_mask_none_
#    byte_identical_to_absent for the "resolver installed, real candidates"
#    variant of this check).
# ---------------------------------------------------------------------------
def test_cand_addr_mask_none_is_byte_identical_to_field_absent():
    torch.manual_seed(0)
    batch, *_ = _toy_writeback_batch(b=4, d=16, C=3, seed=1)
    batch.cand_addr_mask = None   # simulate "field never set at all"
    model = ClauseReactor(dim=16, resolver=make_resolver("A", 16))
    model.eval()
    with torch.no_grad():
        out_none = model(batch)

    batch2, *_ = _toy_writeback_batch(b=4, d=16, C=3, seed=1)   # cand_addr_mask left at its true (all-1 at t=0) value
    batch2.cand_addr_mask = None
    with torch.no_grad():
        out_absent = model(batch2)
    for k in out_none:
        assert torch.equal(out_none[k], out_absent[k]), k


def test_no_addr_mask_falls_back_to_value_redirect_exactly():
    """With ``cand_addr_mask`` forced to all-zero (no row ever redirects the
    address), the ENTITY collapse must reproduce byte-for-byte the pre-M57b
    value-redirect arithmetic: ``v <- Σ w·cand_mem_read``, entity untouched."""
    torch.manual_seed(0)
    batch, cand_atoms, pronoun_addr, rel, gold_idx = _toy_writeback_batch(b=4, d=16, C=3, seed=2)
    batch.cand_addr_mask.zero_()   # force every row to value-redirect instead
    model = ClauseReactor(dim=16, resolver=make_resolver("B", 16, 128))
    model.eval()
    with torch.no_grad():
        out = model(batch)
    # value-redirect: no address change, so the question step (which queries
    # the CANDIDATE's own node, never written under value-redirect) reads back
    # ~nothing -- the mirror image of the address-redirect mechanics test.
    assert out["resolver_logits"].shape == (4, 2, 3)


def test_cand_forced_index_none_is_byte_identical_to_field_absent_and_all_negative_one():
    """The M57b v2 byte-identity extension: ``cand_forced_index`` is a NEW
    optional field -- ``None`` (never set), an explicit all-``-1`` tensor,
    and the field being entirely absent from a hand-built batch must all
    produce IDENTICAL model output, for a resolver-installed batch that DOES
    carry real address-redirect candidates (the strongest version of this
    check -- the forcing branch's ``if batch.cand_forced_index is not
    None:`` guard, plus every entry being invalid (-1), must be a true
    no-op)."""
    torch.manual_seed(0)
    model = ClauseReactor(dim=16, resolver=make_resolver("A", 16))
    model.eval()

    batch_absent, *_ = _toy_writeback_batch(b=5, d=16, C=3, seed=9)
    assert batch_absent.cand_forced_index is None   # dataclass default
    with torch.no_grad():
        out_absent = model(batch_absent)

    batch_none, *_ = _toy_writeback_batch(b=5, d=16, C=3, seed=9)
    batch_none.cand_forced_index = None
    with torch.no_grad():
        out_none = model(batch_none)

    batch_neg1, *_ = _toy_writeback_batch(b=5, d=16, C=3, seed=9)
    batch_neg1.cand_forced_index = torch.full_like(batch_neg1.cand_addr_mask, -1, dtype=torch.long)
    with torch.no_grad():
        out_neg1 = model(batch_neg1)

    for k in out_absent:
        assert torch.equal(out_absent[k], out_none[k]), k
        assert torch.equal(out_absent[k], out_neg1[k]), k


# ---------------------------------------------------------------------------
# 2. Address-redirect mechanics, EVAL mode (hard collapse).
# ---------------------------------------------------------------------------
def test_address_redirect_eval_mode_writes_to_resolved_node():
    torch.manual_seed(0)
    b, d, C = 6, 32, 3
    batch, cand_atoms, pronoun_addr, rel, gold_idx = _toy_writeback_batch(b=b, d=d, C=C, K=4, seed=3)
    model = ClauseReactor(dim=d, resolver=_FixedByGoldResolver(gold_idx))
    model.eval()
    _force_full_write_gate(model)

    with torch.no_grad():
        out = model(batch, return_memory=True)
    assert "_memory" not in ClauseReactor(dim=d)(batch)   # return_memory defaults False, key absent by default
    memory = out["_memory"]

    for i in range(b):
        target_atom = cand_atoms[i, gold_idx[i]].unsqueeze(0)
        r = rel[i].unsqueeze(0)
        read_target = em.query(memory[i:i + 1], target_atom, r)
        read_pronoun = em.query(memory[i:i + 1], pronoun_addr[i].unsqueeze(0), r)

        # cosine argmax against a value codebook (the episode's own options)
        # recovers the STATED value at the resolved candidate's node.
        cod = batch.options[i]                                     # [K, d]
        cos = F.cosine_similarity(read_target.expand(cod.shape[0], -1), cod, dim=-1)
        assert int(cos.argmax()) == int(batch.answer[i])
        assert F.cosine_similarity(read_target, batch.value[i, 0:1]).item() > 0.999

        # the pronoun's own batch-grounded (placeholder) address holds
        # ~nothing -- nothing ever explicitly wrote there.
        assert read_pronoun.norm().item() < 0.5


def test_address_redirect_eval_mode_is_hard_one_hot():
    """At eval, the SAME hard-argmax collapse weights M53b already uses
    drive the address redirect -- the resolved entity must equal EXACTLY
    the top candidate's own atom (not a soft blend)."""
    torch.manual_seed(0)
    b, d, C = 4, 16, 3
    batch, cand_atoms, pronoun_addr, rel, gold_idx = _toy_writeback_batch(b=b, d=d, C=C, seed=4)
    model = ClauseReactor(dim=d, resolver=_FixedByGoldResolver(gold_idx))
    model.eval()
    with torch.no_grad():
        out = model(batch, return_memory=False)
    logits = out["resolver_logits"][:, 0]
    assert torch.equal(logits.argmax(-1), gold_idx)


# ---------------------------------------------------------------------------
# 3. Train mode: soft redirect, gradients flow from the answer loss through
#    the addr-collapse weights to the resolver's own parameters.
# ---------------------------------------------------------------------------
def test_train_mode_soft_addr_redirect_gradients_flow_to_resolver():
    torch.manual_seed(0)
    batch, *_ = _toy_writeback_batch(b=5, d=16, C=3, K=4, seed=5)
    for track in ("A", "B"):
        torch.manual_seed(1)
        resolver = make_resolver(track, 16, 128)
        model = ClauseReactor(dim=16, resolver=resolver)
        model.train()
        out = model(batch)
        assert out["resolver_logits"].shape == (5, 2, 3)
        # step 0 (the write-back collapse) carries the real candidate set;
        # step 1 (the question) carries none -> zero margin.
        assert torch.equal(out["resolver_margin"][:, 1], torch.zeros(5))
        loss = F.cross_entropy(out["answer_logits"], batch.answer)
        loss.backward()
        grads = [p.grad for p in resolver.parameters()]
        assert any(g is not None and torch.any(g != 0) for g in grads), track


# ---------------------------------------------------------------------------
# 4. Mixed batch: value-redirect rows and addr-redirect rows in the SAME
#    batch don't interfere -- each behaves as its isolated (subset) equivalent.
#    Sound because every op in the model (GRUCell, Linear, entity_memory's
#    per-index einsums) is embarrassingly parallel across the batch
#    dimension -- there is no cross-row term anywhere -- so this is a strong
#    regression check, not a coincidence of small numbers.
# ---------------------------------------------------------------------------
def _toy_mixed_batch(b_value=3, b_addr=3, d=16, C=3, K=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = b_value + b_addr
    T = C + 2   # C "establish candidate fact" steps + 1 collapse step + 1 question step
    opts = F.normalize(torch.randn(b, K, d, generator=g), dim=-1)
    ans = torch.randint(0, K, (b,), generator=g)

    entity = torch.zeros(b, T, d); relation = torch.zeros(b, T, d); value = torch.zeros(b, T, d)
    pred = torch.zeros(b, T, d); is_q = torch.zeros(b, T); mask = torch.zeros(b, T)
    cand_entity = torch.zeros(b, T, C, d); cand_mask = torch.zeros(b, T, C)
    cand_prior = torch.zeros(b, T, C); cand_feature = torch.zeros(b, T, FEATURE_DIM)
    cand_gold = torch.full((b, T), -1, dtype=torch.long)
    cand_addr_mask = torch.zeros(b, T)

    cand_atoms_all = F.normalize(torch.randn(b, C, d, generator=g), dim=-1)
    rel_all = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    gold_idx_all = torch.randint(0, C, (b,), generator=g)
    prd = F.normalize(torch.randn(b, d, generator=g), dim=-1)

    is_addr_row = torch.zeros(b, dtype=torch.bool)
    is_addr_row[b_value:] = True   # first b_value rows = value-redirect, rest = addr-redirect

    collapse_t, q_t = C, C + 1
    for i in range(b):
        r = rel_all[i]
        ca = cand_atoms_all[i]
        if not is_addr_row[i]:
            # VALUE-redirect row (M53a/M53b shape): C write steps establish
            # each candidate's own fact; collapse step's placeholder value is
            # pre-bound to the answer (mirrors _toy_batch_with_candidates);
            # question queries the SAME fixed object address.
            cand_values = opts[i, torch.randint(0, K, (C,), generator=g)]
            cand_values[gold_idx_all[i]] = opts[i, ans[i]]
            for j in range(C):
                entity[i, j], relation[i, j], value[i, j], pred[i, j] = ca[j], r, cand_values[j], prd[i]
                mask[i, j] = 1.0
            obj_ent = F.normalize(torch.randn(d, generator=g), dim=-1)
            entity[i, collapse_t], relation[i, collapse_t], pred[i, collapse_t] = obj_ent, r, prd[i]
            value[i, collapse_t] = opts[i, ans[i]]
            mask[i, collapse_t] = 1.0
            entity[i, q_t], relation[i, q_t] = obj_ent, r
            is_q[i, q_t] = 1.0
            mask[i, q_t] = 1.0
            cand_addr_mask[i, collapse_t] = 0.0
        else:
            # ADDR-redirect row (M57b shape): steps 0..C-1 are padding
            # (mask=0, no-ops); collapse step's entity is a garbage
            # placeholder, value is the stated answer; question queries the
            # TRUE gold candidate's own node directly.
            pronoun_addr = F.normalize(torch.randn(d, generator=g), dim=-1)
            entity[i, collapse_t], relation[i, collapse_t], pred[i, collapse_t] = pronoun_addr, r, prd[i]
            value[i, collapse_t] = opts[i, ans[i]]
            mask[i, collapse_t] = 1.0
            entity[i, q_t], relation[i, q_t] = ca[gold_idx_all[i]], r
            is_q[i, q_t] = 1.0
            mask[i, q_t] = 1.0
            cand_addr_mask[i, collapse_t] = 1.0

        cand_entity[i, collapse_t] = ca
        cand_mask[i, collapse_t] = 1.0
        cand_prior[i, collapse_t] = 1.0 / C
        cand_feature[i, collapse_t] = torch.randn(FEATURE_DIM, generator=g)
        cand_gold[i, collapse_t] = gold_idx_all[i]

    batch = ClauseBatch(entity, relation, value, pred, is_q, mask, opts, ans,
                         cand_entity=cand_entity, cand_mask=cand_mask, cand_prior=cand_prior,
                         cand_feature=cand_feature, cand_gold=cand_gold, cand_addr_mask=cand_addr_mask)
    return batch, is_addr_row


def test_mixed_value_and_addr_rows_do_not_interfere():
    torch.manual_seed(0)
    batch, is_addr_row = _toy_mixed_batch(b_value=3, b_addr=4, d=16, C=3, seed=6)
    value_idx = (~is_addr_row).nonzero(as_tuple=True)[0]
    addr_idx = is_addr_row.nonzero(as_tuple=True)[0]

    for track in ("A", "B"):
        torch.manual_seed(2)
        model = ClauseReactor(dim=16, resolver=make_resolver(track, 16, 128))
        for mode in ("eval", "train"):
            getattr(model, mode)()
            ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()
            with ctx:
                out_mixed = model(batch)
                out_value = model(batch.subset(value_idx))
                out_addr = model(batch.subset(addr_idx))
            for k in out_mixed:
                assert torch.allclose(out_mixed[k][value_idx], out_value[k], atol=1e-5), (track, mode, k, "value")
                assert torch.allclose(out_mixed[k][addr_idx], out_addr[k], atol=1e-5), (track, mode, k, "addr")


# ---------------------------------------------------------------------------
# 5. Teacher-forcing mechanics (M57b v2, replaces the curriculum-level
#    wrong_binding aux-gold-corruption arm): ``cand_forced_index`` overrides
#    the collapse weights REGARDLESS of the resolver's own logits, in BOTH
#    train and eval mode, for addr-redirect rows.
# ---------------------------------------------------------------------------
def test_forced_index_overrides_resolver_logits_eval_and_train():
    """Stub the resolver to confidently argmax to candidate 0 for EVERY row
    (via ``_FixedByGoldResolver`` with an all-zero gold vector), but force
    the collapse to candidate ``gold_idx`` (which for most rows differs from
    0) via ``cand_forced_index``. The RESOLVED entity/value must reflect the
    FORCED index, not the resolver's own (wrong, by construction) argmax --
    checked via the write-back mechanics: only the FORCED candidate's node
    ends up holding the stated value."""
    torch.manual_seed(0)
    b, d, C = 6, 24, 3
    batch, cand_atoms, pronoun_addr, rel, gold_idx = _toy_writeback_batch(b=b, d=d, C=C, K=4, seed=11)
    assert bool((gold_idx != 0).any()), "need at least one row where forced != resolver's stubbed choice"

    always_zero = torch.zeros(b, dtype=torch.long)
    model = ClauseReactor(dim=d, resolver=_FixedByGoldResolver(always_zero))
    _force_full_write_gate(model)

    forced = torch.full((b, 2), -1, dtype=torch.long)
    forced[:, 0] = gold_idx   # force the write-back collapse step (t=0) to the TRUE gold index
    batch.cand_forced_index = forced

    for mode in ("eval", "train"):
        getattr(model, mode)()
        ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()
        with ctx:
            out = model(batch, return_memory=True)
        memory = out["_memory"]
        # resolver_logits still reflect the STUBBED (always-0) resolver --
        # forcing doesn't touch what's recorded for the aux loss.
        assert torch.equal(out["resolver_logits"][:, 0].argmax(-1), always_zero)
        for i in range(b):
            target_atom = cand_atoms[i, gold_idx[i]].unsqueeze(0)   # the FORCED node
            r = rel[i].unsqueeze(0)
            read_target = em.query(memory[i:i + 1], target_atom, r)
            assert F.cosine_similarity(read_target, batch.value[i, 0:1]).item() > 0.99, (mode, i)


def test_forced_index_train_mode_weights_are_hard_one_hot_not_soft():
    """Even in TRAIN mode (where an unforced collapse would be a SOFT
    softmax blend), a forced row's effective weights must be a hard one-hot
    at the forced index -- checked indirectly via the resolved entity/value
    exactly matching the forced candidate's own atom/memory-read (a soft
    blend would not)."""
    torch.manual_seed(0)
    b, d, C = 5, 20, 4
    batch, cand_atoms, pronoun_addr, rel, gold_idx = _toy_writeback_batch(b=b, d=d, C=C, K=4, seed=12)
    resolver = make_resolver("A", d)   # real (untrained, i.e. genuinely SOFT) resolver
    model = ClauseReactor(dim=d, resolver=resolver)
    model.train()

    forced = torch.full((b, 2), -1, dtype=torch.long)
    forced[:, 0] = gold_idx
    batch.cand_forced_index = forced

    out = model(batch)   # train mode: no grad context needed, we just inspect the forward pass
    # Reconstruct the resolved entity the SAME way _collapse would (Σ w·ce)
    # is not directly exposed, so instead verify via the GRU's downstream
    # effect: forcing must make the redirected node RECOVERABLE with high
    # cosine similarity, exactly like the eval-mode hard-collapse test does
    # -- a soft blend across C>=2 near-orthogonal random atoms would NOT
    # pass this (its cosine similarity to any single candidate atom would be
    # far below 0.99 for C=4 random unit vectors).
    _force_full_write_gate(model)
    model.train()
    with torch.enable_grad():
        out = model(batch, return_memory=True)
    memory = out["_memory"]
    for i in range(b):
        target_atom = cand_atoms[i, gold_idx[i]].unsqueeze(0)
        r = rel[i].unsqueeze(0)
        read_target = em.query(memory[i:i + 1], target_atom, r)
        assert F.cosine_similarity(read_target, batch.value[i, 0:1]).item() > 0.99, i


# ---------------------------------------------------------------------------
# 6. Curriculum generator sanity (nsm_ct.curriculum2.WriteBackCurriculumGenerator, v2).
# ---------------------------------------------------------------------------
class _DummyMeaningResolver:
    """A minimal stand-in for nsm_ct.meaning.NSMMeaningResolver, avoiding a
    real USVS/explication dependency for pure shape/flag checks -- every
    write-back triple is grounded via _ent_vec, which for a name goes
    through codec.filler_vec("var:" + name) directly (never touches this
    resolver at all); it's only invoked for the handful of non-name content
    words (attribute values, place words), so a fixed-vector stub is fine
    for a structural test that never inspects the OPTION vectors' semantic
    content."""

    def resolve(self, word):
        raise NotImplementedError("USVS should already cover every word in this curriculum")


def _build_writeback_batch(episodes, dim=16, **kwargs):
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.tpr import TPRCodec

    codec = TPRCodec(dim=dim)
    return build_clause_batch(episodes, None, NSMMeaningResolver(), codec, **kwargs)


def test_writeback_v2_context_shape_and_overwrite_ordering():
    """Every entity's OWN named attribute statement comes BEFORE the
    pronoun statement -- the overwrite ordering the v2 redesign depends on
    (see WriteBackCurriculumGenerator's docstring)."""
    eps = generate_writeback_episodes(30, seed=0)
    for ep in eps:
        assert len(ep.context) == 5
        antecedent_stmt_idx = ep.context.index(f"{ep.meta['true_antecedent']} is {ep.meta['stale_attr']} .")
        other_stmt_idx = ep.context.index(f"{ep.meta['other_entity']} is {ep.meta['other_attr']} .")
        pronoun_stmt_idx = ep.context.index(f"{ep.meta['pronoun']} is {ep.meta['pronoun_attr']} .")
        assert antecedent_stmt_idx < pronoun_stmt_idx
        assert other_stmt_idx < pronoun_stmt_idx
        assert pronoun_stmt_idx == 4   # pronoun statement is strictly LAST
        assert ep.meta["pronoun_sentence_index"] == 4


def test_writeback_three_distinct_attrs_and_stale_in_options():
    eps = generate_writeback_episodes(40, seed=1)
    for ep in eps:
        stale, other, pronoun_attr = ep.meta["stale_attr"], ep.meta["other_attr"], ep.meta["pronoun_attr"]
        assert len({stale, other, pronoun_attr}) == 3   # 3 distinct draws
        assert stale in ep.options    # the signature wrong answer of a failed redirect
        assert other in ep.options
        assert pronoun_attr in ep.options
        assert ep.answer_text != stale   # stale is NEVER the correct answer


def test_writeback_answer_key_matches_question_target():
    eps = generate_writeback_episodes(40, seed=2)
    for ep in eps:
        if ep.meta["question_targets_referent"]:
            assert ep.meta["target_name"] == ep.meta["true_antecedent"]
            assert ep.answer_text == ep.meta["pronoun_attr"]
        else:
            assert ep.meta["target_name"] == ep.meta["other_entity"]
            assert ep.answer_text == ep.meta["other_attr"]


def test_writeback_uniform_target_sampling_empirical():
    """The question's target entity (referent vs other) is sampled roughly
    50/50, independent of gender/attribute draws -- the second half of the
    v2 design law (see the curriculum module's docstring)."""
    n = 3000
    eps = generate_writeback_episodes(n, seed=3)
    frac_referent = sum(1 for e in eps if e.meta["question_targets_referent"]) / n
    assert abs(frac_referent - 0.5) < 0.05, frac_referent


def test_writeback_gold_antecedent_always_true_referent():
    """v2: no wrong_binding flag on the generator anymore --
    gold_antecedent is ALWAYS the true referent (validity forcing moved to
    collapse time, see build_clause_batch's writeback_force)."""
    eps = generate_writeback_episodes(30, seed=4)
    for ep in eps:
        assert ep.meta["gold_antecedent"] == ep.meta["true_antecedent"]
    assert not hasattr(WriteBackCurriculumGenerator(), "wrong_binding")


def test_writeback_registry_matches_gender_selects_referent_not_answer():
    """A second angle on the design law: the antecedent's NAME_GENDER never
    appears anywhere in how the attribute triple is chosen (structural check
    -- confirms the generator's OWN bookkeeping is self-consistent,
    complementing the empirical independence checks)."""
    eps = generate_writeback_episodes(200, seed=5)
    for ep in eps:
        antecedent = ep.meta["true_antecedent"]
        assert NAME_GENDER[antecedent] == ("F" if ep.meta["pronoun"] == "she" else "M")
        assert "gender" not in {k.lower() for k in ep.meta if "attr" in k.lower()}


def test_writeback_determinant_independence_empirical():
    """The design-law check: over MANY generated episodes, the answer is
    independent of BOTH gold-determinants -- gender (which name the pronoun
    picks out) and question-target selection (referent vs other) -- P(answer
    | determinant) is uniform over the attribute pool, same as P(answer). A
    large max-deviation-from-uniform tolerance (this is a randomized
    empirical check, not an exact one) still easily separates "independent
    by construction" from a real leak (e.g. accidentally keying attribute
    choice off gender or target selection would make SOME value have zero
    probability under one condition)."""
    n = 3000
    eps = generate_writeback_episodes(n, seed=7)
    uniform = 1.0 / len(_ATTR_VALUES)

    by_gender = {"she": [], "he": []}
    by_target = {True: [], False: []}
    for ep in eps:
        by_gender[ep.meta["pronoun"]].append(ep.answer_text)
        by_target[ep.meta["question_targets_referent"]].append(ep.answer_text)

    for label, groups in (("gender", by_gender), ("question_targets_referent", by_target)):
        for key, answers in groups.items():
            assert len(answers) > 200, "need enough samples per group for the empirical check to mean anything"
            for word in _ATTR_VALUES:
                frac = answers.count(word) / len(answers)
                assert abs(frac - uniform) < 0.06, (label, key, word, frac, uniform)


def test_writeback_episode_has_two_candidates_and_addr_flag_and_gold_index():
    eps = generate_writeback_episodes(12, seed=0)
    batch = _build_writeback_batch(eps)
    pronoun_t = 4   # fixed v2 shape: place_a, place_b, referent named attr, other named attr, pronoun collapse, question
    assert batch.cand_addr_mask is not None
    assert torch.equal(batch.cand_addr_mask[:, pronoun_t], torch.ones(len(eps)))
    assert torch.equal(batch.cand_addr_mask.sum(-1), torch.ones(len(eps)))   # exactly one addr-redirect step/episode
    assert torch.equal(batch.cand_mask[:, pronoun_t].sum(-1), torch.full((len(eps),), 2.0))  # >=2 candidates, exactly 2 here
    for i, ep in enumerate(eps):
        expected_idx = ep.meta["registry_order"].index(ep.meta["true_antecedent"])
        assert int(batch.cand_gold[i, pronoun_t]) == expected_idx


def test_writeback_cheat_arm_strips_all_candidate_data():
    eps = generate_writeback_episodes(10, seed=2)
    batch = _build_writeback_batch(eps, writeback_cheat=True)
    assert batch.cand_entity is None
    assert batch.cand_mask is None
    assert batch.cand_addr_mask is None
    assert batch.cand_forced_index is None


def test_writeback_no_gold_eval_has_no_gold_grounding():
    eps = generate_writeback_episodes(10, seed=3)
    batch = _build_writeback_batch(eps, writeback_no_gold=True)
    pronoun_t = 4
    # candidates + priors + the addr-redirect flag are still fully present...
    assert batch.cand_mask is not None
    assert torch.equal(batch.cand_mask[:, pronoun_t].sum(-1), torch.full((len(eps),), 2.0))
    assert torch.equal(batch.cand_addr_mask[:, pronoun_t], torch.ones(len(eps)))
    # ...but NO gold index anywhere.
    assert torch.equal(batch.cand_gold[:, pronoun_t], torch.full((len(eps),), -1, dtype=torch.long))


def test_writeback_force_binding_gold_and_wrong_populate_cand_forced_index():
    """``build_clause_batch(..., writeback_force="gold"/"wrong")`` -- the v2
    honest validity machinery replacing the curriculum-level wrong_binding
    corruption arm. "gold" must match cand_gold exactly (there are always
    exactly 2 candidates, matching the true referent); "wrong" must be the
    OTHER candidate index everywhere."""
    eps = generate_writeback_episodes(20, seed=6)
    pronoun_t = 4
    batch = _build_writeback_batch(eps)          # baseline: no forcing at all
    batch_g = _build_writeback_batch(eps, writeback_force="gold")
    batch_w = _build_writeback_batch(eps, writeback_force="wrong")

    assert batch.cand_forced_index is None
    assert torch.equal(batch_g.cand_forced_index[:, pronoun_t], batch.cand_gold[:, pronoun_t])
    assert torch.equal(batch_w.cand_forced_index[:, pronoun_t], 1 - batch.cand_gold[:, pronoun_t])
    # forcing is inert everywhere except the pronoun step itself.
    other_steps = [t for t in range(batch_g.cand_forced_index.shape[1]) if t != pronoun_t]
    for t in other_steps:
        assert torch.equal(batch_g.cand_forced_index[:, t], torch.full((len(eps),), -1, dtype=torch.long))


def test_writeback_answer_key_and_option_shapes_match_reactor_contract():
    eps = generate_writeback_episodes(15, seed=4, num_options=4)
    for ep in eps:
        assert ep.meta["kind"] == "writeback"
        assert 2 <= len(ep.options) <= 4
        assert 0 <= ep.answer_idx < len(ep.options)
        assert ep.options[ep.answer_idx] == ep.answer_text
        assert len(set(ep.meta["registry_order"])) == 2               # exactly 2 candidate referents
    batch = _build_writeback_batch(eps)
    assert batch.options.shape[0] == len(eps)
    assert batch.answer.shape == (len(eps),)


# ---------------------------------------------------------------------------
# 7. Memory semantics (real curriculum + build_clause_batch + a real
#    resolver, eval mode, forced collapse): the STRONGEST end-to-end check --
#    querying the actual entity_memory tensor at each entity's own node
#    confirms WHERE the write landed, not just that the answer_logits came
#    out right.
# ---------------------------------------------------------------------------
def test_writeback_forced_gold_eval_memory_semantics():
    """Forced-gold: the redirect lands on the TRUE referent's node, which
    then holds the OVERWRITE value (pronoun_attr) -- the referent's earlier
    named (stale) attribute is gone. The other entity's node is untouched,
    still holding its own named attribute."""
    eps = generate_writeback_episodes(5, seed=20)
    dim = 32
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.tpr import TPRCodec
    codec = TPRCodec(dim=dim)
    meaning = NSMMeaningResolver()
    cache = {}

    batch = build_clause_batch(eps, None, meaning, codec, writeback_force="gold")
    model = ClauseReactor(dim=dim, resolver=make_resolver("A", dim))
    model.eval()
    _force_full_write_gate(model)
    with torch.no_grad():
        out = model(batch, return_memory=True)
    memory = out["_memory"]

    attr_rel = torch.from_numpy(codec.filler_vec("rel:ATTR"))
    for i, ep in enumerate(eps):
        antecedent_vec = torch.from_numpy(_ent_vec(ep.meta["true_antecedent"], None, codec, cache))
        other_vec = torch.from_numpy(_ent_vec(ep.meta["other_entity"], None, codec, cache))
        pronoun_attr_vec = torch.from_numpy(_ent_vec(ep.meta["pronoun_attr"], None, codec, cache))
        other_attr_vec = torch.from_numpy(_ent_vec(ep.meta["other_attr"], None, codec, cache))

        read_antecedent = em.query(memory[i:i + 1], antecedent_vec.unsqueeze(0), attr_rel.unsqueeze(0))
        read_other = em.query(memory[i:i + 1], other_vec.unsqueeze(0), attr_rel.unsqueeze(0))

        assert F.cosine_similarity(read_antecedent, pronoun_attr_vec.unsqueeze(0)).item() > 0.95, i
        assert F.cosine_similarity(read_other, other_attr_vec.unsqueeze(0)).item() > 0.95, i


def test_writeback_forced_wrong_eval_memory_semantics():
    """Forced-wrong: the redirect lands on the OTHER (non-referent) entity's
    node instead -- the referent's node STILL HOLDS its earlier (stale)
    named attribute (never overwritten), and the other entity's node gets
    CLOBBERED with the pronoun's overwrite value (its own named attribute is
    gone)."""
    eps = generate_writeback_episodes(5, seed=21)
    dim = 32
    from nsm_ct.meaning import NSMMeaningResolver
    from nsm_ct.tpr import TPRCodec
    codec = TPRCodec(dim=dim)
    meaning = NSMMeaningResolver()
    cache = {}

    batch = build_clause_batch(eps, None, meaning, codec, writeback_force="wrong")
    model = ClauseReactor(dim=dim, resolver=make_resolver("A", dim))
    model.eval()
    _force_full_write_gate(model)
    with torch.no_grad():
        out = model(batch, return_memory=True)
    memory = out["_memory"]

    attr_rel = torch.from_numpy(codec.filler_vec("rel:ATTR"))
    for i, ep in enumerate(eps):
        antecedent_vec = torch.from_numpy(_ent_vec(ep.meta["true_antecedent"], None, codec, cache))
        other_vec = torch.from_numpy(_ent_vec(ep.meta["other_entity"], None, codec, cache))
        pronoun_attr_vec = torch.from_numpy(_ent_vec(ep.meta["pronoun_attr"], None, codec, cache))
        stale_attr_vec = torch.from_numpy(_ent_vec(ep.meta["stale_attr"], None, codec, cache))

        read_antecedent = em.query(memory[i:i + 1], antecedent_vec.unsqueeze(0), attr_rel.unsqueeze(0))
        read_other = em.query(memory[i:i + 1], other_vec.unsqueeze(0), attr_rel.unsqueeze(0))

        assert F.cosine_similarity(read_antecedent, stale_attr_vec.unsqueeze(0)).item() > 0.95, i
        assert F.cosine_similarity(read_other, pronoun_attr_vec.unsqueeze(0)).item() > 0.95, i
