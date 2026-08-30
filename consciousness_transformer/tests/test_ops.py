"""Tests for the deterministic op library (nsm_ct.ops) -- the robustification
pass + Track C instruction set described in dev/OP_LIBRARY_MAP.md. Each op's
algebraic property gets its own test; a composed-program test exercises
several ops chained through a RegisterFile with a trace recorded, mirroring
the shape a future executor's gold program would take
(dev/TRACK_C_DESIGN.md Sec.1.5). No parser dependency -- synthetic-tensor
tests, mirroring tests/test_instances.py's own isolation discipline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nsm_ct import entity_memory as em
from nsm_ct import ops
from nsm_ct.instances import InstanceRegistry, ProvenanceLog, candidates_for, write_attribute
from nsm_ct.tpr import TPRCodec

DIM = 16


def _codec(dim: int = DIM) -> TPRCodec:
    return TPRCodec(dim=dim)


def _unit(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm()


def _unit_b(b: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(b, d, generator=g)
    return F.normalize(v, dim=-1)


# ---------------------------------------------------------------------------
# MEMORY: bind_write / unbind_query -- VSA bind/unbind roundtrip.
# ---------------------------------------------------------------------------
def test_bind_write_unbind_query_roundtrip_orthonormal_keys():
    b, d = 3, 32
    memory = em.init_memory(b, d, "cpu")
    entity = F.normalize(torch.randn(b, d), dim=-1)
    relation = F.normalize(torch.randn(b, d), dim=-1)
    value = F.normalize(torch.randn(b, d), dim=-1)
    gate = torch.ones(b)
    memory = ops.bind_write(memory, entity, relation, value, gate)
    got = ops.unbind_query(memory, entity, relation)
    assert F.cosine_similarity(got, value).mean().item() > 0.9


def test_bind_write_matches_entity_memory_write():
    b, d = 2, 12
    memory = torch.randn(b, d, d, d)
    entity = F.normalize(torch.randn(b, d), dim=-1)
    relation = F.normalize(torch.randn(b, d), dim=-1)
    value = F.normalize(torch.randn(b, d), dim=-1)
    gate = torch.rand(b)
    got = ops.bind_write(memory, entity, relation, value, gate)
    expected = em.write(memory, entity, relation, value, gate)
    assert torch.equal(got, expected)


def test_bind_write_unbatched_convenience():
    d = 10
    memory = em.init_memory(1, d, "cpu").squeeze(0)
    entity = _unit(d, 1)
    relation = _unit(d, 2)
    value = _unit(d, 3)
    memory = ops.bind_write(memory, entity, relation, value, 1.0)
    assert memory.shape == (d, d, d)
    got = ops.unbind_query(memory, entity, relation)
    assert got.shape == (d,)
    assert F.cosine_similarity(got.unsqueeze(0), value.unsqueeze(0)).item() > 0.9


def test_inverse_query_entity_matches_entity_memory():
    b, d = 2, 14
    memory = torch.randn(b, d, d, d)
    relation = F.normalize(torch.randn(b, d), dim=-1)
    value = F.normalize(torch.randn(b, d), dim=-1)
    got = ops.inverse_query_entity(memory, relation, value)
    expected = em.query_entity(memory, relation, value)
    assert torch.equal(got, expected)


def test_inverse_query_entity_recovers_writer():
    d = 20
    memory = em.init_memory(1, d, "cpu")
    entity = F.normalize(torch.randn(1, d), dim=-1)
    relation = F.normalize(torch.randn(1, d), dim=-1)
    value = F.normalize(torch.randn(1, d), dim=-1)
    memory = ops.bind_write(memory, entity, relation, value, torch.ones(1))
    recovered = ops.inverse_query_entity(memory, relation, value)
    assert F.cosine_similarity(recovered, entity).item() > 0.9


# ---------------------------------------------------------------------------
# MEMORY: superpose_vote -- accumulation, majority wins.
# ---------------------------------------------------------------------------
def test_superpose_vote_accumulates_and_majority_wins():
    d = 24
    memory = em.init_memory(1, d, "cpu")
    entity = F.normalize(torch.randn(1, d), dim=-1)
    relation = F.normalize(torch.randn(1, d), dim=-1)
    majority = F.normalize(torch.randn(1, d), dim=-1)
    minority = F.normalize(torch.randn(1, d), dim=-1)
    gate = torch.ones(1)
    # majority value voted 3x, minority voted once.
    for _ in range(3):
        memory = ops.superpose_vote(memory, entity, relation, majority, gate)
    memory = ops.superpose_vote(memory, entity, relation, minority, gate)
    readout = ops.unbind_query(memory, entity, relation)
    cos_majority = F.cosine_similarity(readout, majority).item()
    cos_minority = F.cosine_similarity(readout, minority).item()
    assert cos_majority > cos_minority


def test_superpose_vote_is_write_with_overwrite_zero():
    b, d = 2, 10
    memory = torch.randn(b, d, d, d)
    entity = F.normalize(torch.randn(b, d), dim=-1)
    relation = F.normalize(torch.randn(b, d), dim=-1)
    value = F.normalize(torch.randn(b, d), dim=-1)
    gate = torch.rand(b)
    got = ops.superpose_vote(memory, entity, relation, value, gate)
    expected = em.write(memory, entity, relation, value, gate, overwrite=torch.zeros(b))
    assert torch.equal(got, expected)


# ---------------------------------------------------------------------------
# MEMORY: erase -- clears a slot to ~0, matches write(gate=0, overwrite=1).
# ---------------------------------------------------------------------------
def test_erase_clears_slot_to_zero():
    d = 18
    memory = em.init_memory(1, d, "cpu")
    entity = F.normalize(torch.randn(1, d), dim=-1)
    relation = F.normalize(torch.randn(1, d), dim=-1)
    value = F.normalize(torch.randn(1, d), dim=-1)
    memory = ops.bind_write(memory, entity, relation, value, torch.ones(1))
    before = ops.unbind_query(memory, entity, relation)
    assert before.norm().item() > 0.5
    memory = ops.erase(memory, entity, relation)
    after = ops.unbind_query(memory, entity, relation)
    assert after.norm().item() < 1e-4


def test_erase_matches_write_gate_zero_overwrite_one():
    b, d = 2, 10
    memory = torch.randn(b, d, d, d)
    entity = F.normalize(torch.randn(b, d), dim=-1)
    relation = F.normalize(torch.randn(b, d), dim=-1)
    got = ops.erase(memory, entity, relation)
    dummy_value = torch.zeros(b, d)
    expected = em.write(memory, entity, relation, dummy_value, torch.zeros(b), overwrite=torch.ones(b))
    assert torch.allclose(got, expected, atol=1e-5)


def test_erase_leaves_other_slots_untouched():
    d = 16
    memory = em.init_memory(1, d, "cpu")
    e1, r1, v1 = (F.normalize(torch.randn(1, d), dim=-1) for _ in range(3))
    e2, r2, v2 = (F.normalize(torch.randn(1, d), dim=-1) for _ in range(3))
    memory = ops.bind_write(memory, e1, r1, v1, torch.ones(1))
    memory = ops.bind_write(memory, e2, r2, v2, torch.ones(1))
    memory = ops.erase(memory, e1, r1)
    other = ops.unbind_query(memory, e2, r2)
    assert F.cosine_similarity(other, v2).item() > 0.9


# ---------------------------------------------------------------------------
# MEMORY: cleanup -- correct pick, margin, abstain.
# ---------------------------------------------------------------------------
def test_cleanup_picks_the_right_code():
    d, n = 16, 5
    codebook = _unit_b(n, d, seed=10)
    target_index = 2
    noisy = codebook[target_index] + 0.01 * torch.randn(d)
    index, cleaned, margin, abstain_flag = ops.cleanup(noisy, codebook)
    assert index.item() == target_index
    assert torch.equal(cleaned, codebook[target_index])
    assert not abstain_flag.item()


def test_cleanup_abstains_below_margin():
    d = 16
    # two near-identical codebook entries -> tiny margin regardless of query.
    base = _unit(d, 1)
    near = F.normalize(base + 0.01 * torch.randn(d), dim=-1)
    codebook = torch.stack([base, near, _unit(d, 99)], dim=0)
    index, cleaned, margin, abstain_flag = ops.cleanup(base, codebook, margin_dial=0.5)
    assert margin.item() < 0.5
    assert abstain_flag.item()


def test_cleanup_no_ambiguity_single_row_codebook():
    d = 8
    codebook = _unit(d, 5).unsqueeze(0)
    vec = _unit(d, 5)
    index, cleaned, margin, abstain_flag = ops.cleanup(vec, codebook)
    assert index.item() == 0
    assert margin.item() == float("inf")
    assert not abstain_flag.item()


def test_cleanup_batched():
    d, n, b = 12, 4, 3
    codebook = _unit_b(n, d, seed=7)
    vecs = codebook[torch.tensor([0, 1, 3])] + 0.001 * torch.randn(b, d)
    index, cleaned, margin, abstain_flag = ops.cleanup(vecs, codebook)
    assert index.tolist() == [0, 1, 3]
    assert abstain_flag.shape == (b,)


def test_cleanup_dot_vs_cosine_mode_agree_on_unit_vectors():
    d, n = 10, 4
    codebook = _unit_b(n, d, seed=3)
    vec = codebook[1]
    idx_cos, _, _, _ = ops.cleanup(vec, codebook, mode="cosine")
    idx_dot, _, _, _ = ops.cleanup(vec, codebook, mode="dot")
    assert idx_cos.item() == idx_dot.item() == 1


# ---------------------------------------------------------------------------
# MEMORY: similarity -- cosine + dot, magnitude-aware.
# ---------------------------------------------------------------------------
def test_similarity_cosine_scale_invariant_dot_is_not():
    a = _unit(8, 1)
    b = a.clone()
    scaled = a * 0.1
    sim_full = ops.similarity(a.unsqueeze(0), b.unsqueeze(0))
    sim_scaled = ops.similarity(scaled.unsqueeze(0), b.unsqueeze(0))
    assert abs(sim_full.cosine.item() - sim_scaled.cosine.item()) < 1e-5
    assert sim_full.dot.item() > sim_scaled.dot.item() + 0.5


def test_similarity_orthogonal_vectors_zero():
    d = 6
    a = torch.zeros(1, d); a[0, 0] = 1.0
    b = torch.zeros(1, d); b[0, 1] = 1.0
    sim = ops.similarity(a, b)
    assert abs(sim.cosine.item()) < 1e-6
    assert abs(sim.dot.item()) < 1e-6


# ---------------------------------------------------------------------------
# MEMORY: permute / unpermute -- invert, compose, don't collide.
# ---------------------------------------------------------------------------
def test_permute_unpermute_invert():
    x = torch.randn(3, 16)
    for k in (0, 1, 2, 5, 11):
        y = ops.permute(x, k)
        back = ops.unpermute(y, k)
        assert torch.allclose(back, x)


def test_permute_composes():
    x = torch.randn(2, 20)
    lhs = ops.permute(ops.permute(x, 1), 1)
    rhs = ops.permute(x, 2)
    assert torch.equal(lhs, rhs)
    lhs2 = ops.permute(ops.permute(x, 3), -1)
    rhs2 = ops.permute(x, 2)
    assert torch.equal(lhs2, rhs2)


def test_permute_is_identity_at_k_zero():
    x = torch.randn(2, 10)
    assert torch.equal(ops.permute(x, 0), x)


def test_permuted_keys_dont_collide_with_unpermuted():
    # Sequence/position binding: encode two DIFFERENT fillers at two
    # DIFFERENT positions (via permute) sharing the SAME (entity,
    # relation) memory slot, then confirm reading back at the wrong
    # position does NOT recover the filler bound at the other position.
    d = 32
    memory = em.init_memory(1, d, "cpu")
    entity = F.normalize(torch.randn(1, d), dim=-1)
    relation = F.normalize(torch.randn(1, d), dim=-1)
    filler_pos0 = F.normalize(torch.randn(1, d), dim=-1)
    filler_pos1 = F.normalize(torch.randn(1, d), dim=-1)
    v0 = ops.permute(filler_pos0, 0)
    v1 = ops.permute(filler_pos1, 1)
    combined = F.normalize(v0 + v1, dim=-1)
    memory = ops.bind_write(memory, entity, relation, combined, torch.ones(1))
    readout = ops.unbind_query(memory, entity, relation)
    recovered_pos0 = ops.unpermute(readout, 0)
    recovered_pos1 = ops.unpermute(readout, 1)
    assert F.cosine_similarity(recovered_pos0, filler_pos0).item() > F.cosine_similarity(recovered_pos0, filler_pos1).item()
    assert F.cosine_similarity(recovered_pos1, filler_pos1).item() > F.cosine_similarity(recovered_pos1, filler_pos0).item()


# ---------------------------------------------------------------------------
# MEMORY: allocate -- wraps InstanceRegistry.mint.
# ---------------------------------------------------------------------------
def test_allocate_wraps_mint():
    reg = InstanceRegistry(dim=8, seed=0)
    iid, atom = ops.allocate(reg, "mary")
    assert iid == "inst:mary#1"
    assert torch.equal(reg.lookup(iid), atom)


# ---------------------------------------------------------------------------
# MEMORY: recency -- monotone features, is-most-recent correct.
# ---------------------------------------------------------------------------
def test_recency_steps_since_monotone_and_most_recent_correct():
    mention_steps = torch.tensor([[5.0, 8.0, -1.0]])   # candidate 2 never mentioned
    current_step = torch.tensor([10.0])
    feats = ops.recency(mention_steps, current_step)
    # candidate 1 (step 8) is more recent than candidate 0 (step 5)
    assert feats.steps_since[0, 1].item() < feats.steps_since[0, 0].item()
    # never-mentioned candidate gets the sentinel, the largest steps_since
    assert feats.steps_since[0, 2].item() == ops.RECENCY_NEVER
    assert feats.steps_since[0, 2] > feats.steps_since[0, 0]
    assert feats.is_most_recent[0].tolist() == [False, True, False]


def test_recency_no_mentions_all_false():
    mention_steps = torch.tensor([[-1.0, -1.0]])
    feats = ops.recency(mention_steps, 3)
    assert feats.is_most_recent[0].tolist() == [False, False]


def test_recency_log_count_monotone():
    mention_steps = torch.tensor([[1.0, 1.0, 1.0]])
    counts = torch.tensor([[0.0, 1.0, 10.0]])
    feats = ops.recency(mention_steps, 2, mention_counts=counts)
    assert feats.log_count[0, 0].item() == 0.0
    assert feats.log_count[0, 0] < feats.log_count[0, 1] < feats.log_count[0, 2]


def test_recency_scalar_current_step_broadcasts():
    mention_steps = torch.tensor([[0.0], [2.0]])
    feats = ops.recency(mention_steps, 5)
    assert feats.steps_since[0, 0].item() == 5.0
    assert feats.steps_since[1, 0].item() == 3.0


# ---------------------------------------------------------------------------
# MEMORY: temporal_link -- consistent with write order.
# ---------------------------------------------------------------------------
def test_temporal_link_basic_chain():
    order = ["a", "b", "c", "d"]
    pred, succ = ops.temporal_link(order)
    assert succ == {"a": "b", "b": "c", "c": "d"}
    assert pred == {"b": "a", "c": "b", "d": "c"}
    assert "a" not in pred
    assert "d" not in succ


def test_temporal_link_repeated_slot_uses_latest_occurrence():
    order = ["a", "b", "a", "c"]
    pred, succ = ops.temporal_link(order)
    # 'a' appears at position 0 and 2; predecessor/successor reflect the
    # LATEST occurrence (position 2): pred['a'] should be 'b' (from index1->2),
    # not undefined; and succ['a'] should be 'c' (index2->3), overwriting the
    # earlier b->a transition's contribution to succ['a'] request (none, since
    # 'a' as a *source* only appears once at index0 in this order -- so
    # succ.get('a') reflects the LAST time 'a' was the source, which is index2).
    assert succ["a"] == "c"
    assert pred["a"] == "b"


def test_temporal_link_single_element_empty():
    pred, succ = ops.temporal_link(["only"])
    assert pred == {}
    assert succ == {}


# ---------------------------------------------------------------------------
# MEMORY: forget_decay -- 1.0 is identity.
# ---------------------------------------------------------------------------
def test_forget_decay_default_is_identity():
    memory = torch.randn(2, 6, 6, 6)
    out = ops.forget_decay(memory)
    assert torch.equal(out, memory)


def test_forget_decay_scales():
    memory = torch.ones(1, 4, 4, 4)
    out = ops.forget_decay(memory, decay=0.5)
    assert torch.allclose(out, memory * 0.5)


# ---------------------------------------------------------------------------
# TIERS -- re-exports (smoke: same objects/behavior as nsm_ct.ltm).
# ---------------------------------------------------------------------------
def test_tiers_are_ltm_reexports():
    from nsm_ct import ltm
    assert ops.recall is ltm.mem_total
    assert ops.promote is ltm.promote
    assert ops.link is ltm.link_decision


def test_recall_additive():
    memory = torch.randn(1, 5, 5, 5)
    ltm_mem = torch.randn(1, 5, 5, 5)
    assert torch.equal(ops.recall(memory, None), memory)
    assert torch.equal(ops.recall(memory, ltm_mem), memory + ltm_mem)


# ---------------------------------------------------------------------------
# CONTROL: select -- hard/soft.
# ---------------------------------------------------------------------------
def test_select_soft_is_softmax():
    logits = torch.tensor([[1.0, 2.0, 0.5]])
    got = ops.select(logits, hard=False)
    assert torch.allclose(got, torch.softmax(logits, dim=-1))
    assert torch.allclose(got.sum(dim=-1), torch.ones(1))


def test_select_hard_is_one_hot_argmax():
    logits = torch.tensor([[1.0, 5.0, 0.5]])
    got = ops.select(logits, hard=True)
    assert torch.equal(got, torch.tensor([[0.0, 1.0, 0.0]]))


def test_select_respects_mask():
    # index 1 has the highest raw logit but is masked out -- the winner
    # among the ELIGIBLE candidates (0, 2) is index 0 (1.0 > 0.5).
    logits = torch.tensor([[1.0, 5.0, 0.5]])
    mask = torch.tensor([[True, False, True]])
    got = ops.select(logits, mask, hard=True)
    assert torch.equal(got, torch.tensor([[1.0, 0.0, 0.0]]))


# ---------------------------------------------------------------------------
# CONTROL: abstain.
# ---------------------------------------------------------------------------
def test_abstain_idk_when_no_candidates():
    assert ops.abstain(5.0, has_candidates=False) == "idk"


def test_abstain_maybe_below_dial():
    assert ops.abstain(0.01, dial=0.1) == "MAYBE"


def test_abstain_none_above_dial():
    assert ops.abstain(0.5, dial=0.1) is None


def test_abstain_batched():
    margins = torch.tensor([0.5, 0.01])
    has_cand = torch.tensor([True, True])
    got = ops.abstain(margins, dial=0.1, has_candidates=has_cand)
    assert got == [None, "MAYBE"]


# ---------------------------------------------------------------------------
# CONTROL: compare -- cosine, for branch.
# ---------------------------------------------------------------------------
def test_compare_is_cosine():
    a = _unit(8, 1).unsqueeze(0)
    b = _unit(8, 1).unsqueeze(0)
    got = ops.compare(a, b)
    assert torch.allclose(got, torch.ones(1), atol=1e-5)


# ---------------------------------------------------------------------------
# CONTROL: branch -- hard/soft.
# ---------------------------------------------------------------------------
def test_branch_hard_picks_then_above_threshold():
    cond = torch.tensor([0.9, 0.1])
    then_vec = torch.ones(2, 4)
    else_vec = torch.zeros(2, 4)
    got = ops.branch(cond, then_vec, else_vec, hard=True)
    assert torch.equal(got[0], then_vec[0])
    assert torch.equal(got[1], else_vec[1])


def test_branch_soft_interpolates():
    cond = torch.tensor([0.25])
    then_vec = torch.ones(1, 3)
    else_vec = torch.zeros(1, 3)
    got = ops.branch(cond, then_vec, else_vec, hard=False)
    assert torch.allclose(got, torch.full((1, 3), 0.25))


def test_branch_soft_clamps_condition():
    cond = torch.tensor([1.5, -0.5])
    then_vec = torch.ones(2, 2)
    else_vec = torch.zeros(2, 2)
    got = ops.branch(cond, then_vec, else_vec, hard=False)
    assert torch.equal(got[0], then_vec[0])
    assert torch.equal(got[1], else_vec[1])


# ---------------------------------------------------------------------------
# CONTROL: halt -- at budget.
# ---------------------------------------------------------------------------
def test_halt_at_budget():
    assert not ops.halt(ops.PATIENCE - 1)
    assert ops.halt(ops.PATIENCE)
    assert ops.halt(ops.PATIENCE + 1)


def test_halt_batched():
    steps = torch.tensor([1, ops.PATIENCE, ops.PATIENCE + 3])
    got = ops.halt(steps, budget=ops.PATIENCE)
    assert got.tolist() == [False, True, True]


# ---------------------------------------------------------------------------
# CONTROL: emit -- reference identity.
# ---------------------------------------------------------------------------
def test_emit_returns_mem_read_unchanged():
    state = torch.randn(2, 5)
    mem_read = torch.randn(2, 5)
    got = ops.emit(state, mem_read)
    assert torch.equal(got, mem_read)


# ---------------------------------------------------------------------------
# REGISTERS: read/write/trace format.
# ---------------------------------------------------------------------------
def test_register_file_create_read_write():
    rf = ops.RegisterFile.create(2, 6, {"G.rel": "Addr", "P.mem": "Vec"})
    assert rf.values.shape == (2, 2, 6)
    val = torch.randn(2, 6)
    rf.write("G.rel", val, step=0)
    assert torch.equal(rf.read("G.rel"), val)
    assert torch.equal(rf.read("P.mem"), torch.zeros(2, 6))


def test_register_file_scalar_broadcast_and_read_back():
    rf = ops.RegisterFile.create(3, 4, {"P.score": "Scalar"})
    scalars = torch.tensor([0.1, 0.5, 0.9])
    rf.write("P.score", scalars, step=1)
    assert torch.allclose(rf.values[:, 0, :], scalars.unsqueeze(-1).expand(3, 4))
    assert torch.allclose(rf.read("P.score"), scalars, atol=1e-6)


def test_register_file_trace_format():
    rf = ops.RegisterFile.create(1, 4, {"a": "Vec", "b": "Vec"})
    rf.write("a", torch.ones(1, 4), step=0, op_name="bind_write")
    rf.write("b", torch.zeros(1, 4), step=1, op_name="unbind_query", args=("a",))
    rf.record("halt", (), step=2)
    assert rf.trace == [
        ("bind_write", ("a",), 0),
        ("unbind_query", ("b", "a"), 1),
        ("halt", (), 2),
    ]


def test_register_file_write_does_not_mutate_prior_read():
    rf = ops.RegisterFile.create(1, 4, {"a": "Vec"})
    rf.write("a", torch.ones(1, 4), step=0)
    snapshot = rf.read("a")
    rf.write("a", torch.zeros(1, 4), step=1)
    assert torch.equal(snapshot, torch.ones(1, 4))
    assert torch.equal(rf.read("a"), torch.zeros(1, 4))


# ---------------------------------------------------------------------------
# Composed program: write 3 facts -> query -> cleanup -> compare -> branch,
# through a RegisterFile with a trace recorded (dev/TRACK_C_DESIGN.md's
# gold-program shape).
# ---------------------------------------------------------------------------
def test_composed_program_write_query_cleanup_compare_branch():
    d = 24
    codec = _codec(d)
    reg = InstanceRegistry(dim=d, seed=1)
    log = ProvenanceLog()
    memory = torch.zeros(d, d, d)

    mary, _ = reg.mint("mary")
    john, _ = reg.mint("john")
    sandra, _ = reg.mint("sandra")

    doctor_vec = torch.from_numpy(codec.filler_vec("kind:doctor")).float()
    teacher_vec = torch.from_numpy(codec.filler_vec("kind:teacher")).float()
    nurse_vec = torch.from_numpy(codec.filler_vec("kind:nurse")).float()

    # Fact 1, 2, 3.
    for iid, (label, vec) in [(mary, ("doctor", doctor_vec)), (john, ("teacher", teacher_vec)),
                               (sandra, ("nurse", nurse_vec))]:
        memory = write_attribute(memory, reg, iid, "kind", vec, codec, log=log, source="text",
                                  language="en", timestamp=0.0, trust=1.0, value_label=label)

    rf = ops.RegisterFile.create(1, d, {
        "G.rel": "Addr", "P.addr": "Addr", "P.readout": "Vec",
        "P.cleaned": "Vec", "P.compare_then": "Scalar", "P.compare_else": "Scalar",
        "P.answer": "Vec",
    })
    kind_rel = torch.from_numpy(codec.filler_vec("attr:kind")).float().unsqueeze(0)
    rf.write("G.rel", kind_rel, step=0, op_name="load_relation")

    mary_atom = reg.lookup(mary).unsqueeze(0)
    rf.write("P.addr", mary_atom, step=1, op_name="load_address", args=(mary,))

    # query: what is mary's kind?
    readout = ops.unbind_query(memory.unsqueeze(0), rf.read("P.addr"), rf.read("G.rel"))
    rf.write("P.readout", readout, step=2, op_name="unbind_query", args=("P.addr", "G.rel"))

    # cleanup against the {doctor, teacher, nurse} codebook.
    codebook = torch.stack([doctor_vec, teacher_vec, nurse_vec], dim=0)
    index, cleaned, margin, abstain_flag = ops.cleanup(rf.read("P.readout"), codebook)
    rf.write("P.cleaned", cleaned, step=3, op_name="cleanup", args=("P.readout",))
    assert index.item() == 0            # doctor
    assert not abstain_flag.item()

    # compare the cleaned answer against two candidate continuations.
    doctor_continuation = torch.randn(1, d)
    nurse_continuation = torch.randn(1, d)
    cmp_then = ops.compare(rf.read("P.cleaned"), doctor_vec.unsqueeze(0))
    cmp_else = ops.compare(rf.read("P.cleaned"), nurse_vec.unsqueeze(0))
    rf.write("P.compare_then", cmp_then, step=4, op_name="compare", args=("P.cleaned", "doctor_vec"))
    rf.write("P.compare_else", cmp_else, step=5, op_name="compare", args=("P.cleaned", "nurse_vec"))
    assert cmp_then.item() > cmp_else.item()

    # branch picks the continuation matching the cleaned-up fact (mary is a
    # doctor, not a nurse): cond = cmp_then (near 1.0) selects then_vec.
    answer = ops.branch(cmp_then, doctor_continuation, nurse_continuation, hard=True)
    rf.write("P.answer", answer, step=6, op_name="branch", args=("P.compare_then",))
    assert torch.equal(rf.read("P.answer"), doctor_continuation)

    # the trace records the whole program, in order.
    op_names = [t[0] for t in rf.trace]
    assert op_names == [
        "load_relation", "load_address", "unbind_query", "cleanup",
        "compare", "compare", "branch",
    ]
    assert [t[2] for t in rf.trace] == list(range(7))


# ---------------------------------------------------------------------------
# instances.py regression: dot breaks the single-writer cosine tie.
# ---------------------------------------------------------------------------
def test_dot_breaks_cosine_tie_single_writer():
    d = 8
    codec = _codec(d)
    reg = InstanceRegistry(dim=d, seed=0)
    log = ProvenanceLog()
    memory = torch.zeros(d, d, d)

    id_a, _ = reg.mint("a")
    id_b, _ = reg.mint("b")
    # Hand-construct atoms so atom_b has a KNOWN, moderate positive
    # correlation with atom_a (a real registry's random atoms could land
    # either sign) -- this is what reproduces the "scaled copy" tie
    # deterministically rather than by luck.
    atom_a = F.normalize(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]), dim=-1)
    atom_b = F.normalize(torch.tensor([0.3, 1.0, 0, 0, 0, 0, 0, 0]), dim=-1)  # atom_a.atom_b ~ 0.29
    reg._atoms[id_a] = atom_a
    reg._atoms[id_b] = atom_b

    doctor_vec = torch.from_numpy(codec.filler_vec("kind:doctor")).float()
    # ONLY id_a writes "kind" -- id_b's readout at "kind" is a scaled copy
    # of doctor_vec (interference term (atom_b . atom_a) * doctor_vec),
    # same direction, smaller magnitude.
    memory = write_attribute(memory, reg, id_a, "kind", doctor_vec, codec, log=log,
                              source="text", language="en", timestamp=0.0, trust=1.0,
                              value_label="doctor")

    ids_cos, _, scores_cos = candidates_for(memory, reg, codec, attr_name="kind",
                                             target_vec=doctor_vec, threshold=0.5, score="cosine")
    ids_dot, _, scores_dot = candidates_for(memory, reg, codec, attr_name="kind",
                                             target_vec=doctor_vec, threshold=0.5, score="dot")

    # cosine ties: both a (true writer) and b (scaled copy, same direction)
    # clear the threshold -- a false positive.
    assert set(ids_cos) == {id_a, id_b}, f"expected the cosine tie to reproduce, got {ids_cos}"
    # dot correctly discounts the scaled copy: only the true writer clears.
    assert ids_dot == [id_a], f"expected dot to break the tie, got {ids_dot}"


def test_candidates_for_default_is_dot():
    d = 8
    codec = _codec(d)
    reg = InstanceRegistry(dim=d, seed=2)
    log = ProvenanceLog()
    memory = torch.zeros(d, d, d)
    iid, _ = reg.mint("x")
    doctor_vec = torch.from_numpy(codec.filler_vec("kind:doctor")).float()
    memory = write_attribute(memory, reg, iid, "kind", doctor_vec, codec, log=log,
                              source="text", language="en", timestamp=0.0, trust=1.0,
                              value_label="doctor")
    ids_default, _, scores_default = candidates_for(memory, reg, codec, attr_name="kind",
                                                      target_vec=doctor_vec, threshold=0.5)
    ids_dot, _, scores_dot = candidates_for(memory, reg, codec, attr_name="kind",
                                             target_vec=doctor_vec, threshold=0.5, score="dot")
    assert ids_default == ids_dot
    assert torch.equal(scores_default, scores_dot)
