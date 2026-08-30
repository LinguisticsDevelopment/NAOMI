"""Tests for the RICH-EPISODE curriculum (CLAUDE.md's 2026-08-30
reprioritization, "the system must stop requiring MINIMAL episodes"):
:class:`nsm_ct.curriculum2.RichEpisodeGenerator` (N entities 3-8, K
referring/overwrite statements 1-4, three DISTINCT attribute relations) and
its reactor-side generalization :func:`nsm_ct.clause_reactor._rich_steps`
(a NEW sibling to :func:`nsm_ct.clause_reactor._instance_steps`, which is
kept UNCHANGED). See nsm_ct.curriculum2's own extensive module comment
immediately above :class:`RichEpisodeGenerator` for the full design and its
honesty machinery, and nsm_ct.clause_reactor._rich_steps's docstring for
the reactor-side mechanics.

No parser dependency anywhere in this file -- _rich_steps is parser-free by
design (mirrors tests/test_instance_curriculum.py's own isolation
discipline).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch
from nsm_ct.curriculum2 import (
    RichEpisodeGenerator,
    _verify_recency_referent,
    _verify_unique_referent,
    generate_instance_episodes,
    generate_rich_episodes,
)
from nsm_ct.meaning import NSMMeaningResolver
from nsm_ct.resolver import make_resolver
from nsm_ct.tpr import TPRCodec

DIM = 24


def _meaning():
    return NSMMeaningResolver()


def _codec(dim=DIM):
    return TPRCodec(dim=dim)


# ---------------------------------------------------------------------------
# 1. Generator invariants.
# ---------------------------------------------------------------------------
def test_uniqueness_assertion_fires_on_a_constructed_collision():
    """_verify_unique_referent must RAISE when handed a genuine tie (two
    entities sharing the evidence value) -- the honesty contract's
    enforcement mechanism, not decoration. A non-tied case must pass
    silently."""
    # a genuine collision: entities 0 and 1 both have gender "F".
    evidence = {0: "F", 1: "F", 2: "M"}
    try:
        _verify_unique_referent("pronoun", 0, evidence)
        raised = False
    except AssertionError:
        raised = True
    assert raised

    # no collision: passes silently.
    _verify_unique_referent("pronoun", 2, evidence)


def test_recency_assertion_fires_on_a_constructed_collision():
    """_verify_recency_referent must RAISE when the supposed referent is
    NOT the more recent of the pair."""
    mention_order = {0: 2, 1: 5}   # entity 1 is more recent (higher order)
    try:
        _verify_recency_referent(0, 1, mention_order)   # 0 claimed referent, but 1 is more recent
        raised = False
    except AssertionError:
        raised = True
    assert raised

    _verify_recency_referent(1, 0, mention_order)   # correct direction passes silently


def test_distinct_attribute_values_within_a_relation_per_episode():
    """For every relation, every entity CURRENTLY holding it (post-overwrite
    final state) must hold a DISTINCT value -- enforced at initial
    assignment and re-enforced on every overwrite (see the module comment's
    honesty argument #3: no relation ever has two entities tied on the same
    value, so an inverse query and a definite-description MC distractor set
    are never accidentally ambiguous)."""
    eps = generate_rich_episodes(300, seed=1, inverse_frac=0.3)
    for ep in eps:
        n = ep.meta["n_entities"]
        by_relation: dict = {}
        for i in range(n):
            for rel, val in ep.meta["final_values"][i].items():
                by_relation.setdefault(rel, []).append(val)
        for rel, vals in by_relation.items():
            assert len(vals) == len(set(vals)), (ep.meta["instance_seed"], rel, vals)


def test_options_contain_stale_value_when_question_targets_overwritten_slot():
    eps = generate_rich_episodes(300, seed=2, inverse_frac=0.0)
    target_eps = [e for e in eps if e.meta["question_mode"] == "target"]
    assert target_eps
    overwritten = [e for e in target_eps if e.meta["question_targets_overwritten"]]
    assert overwritten, "expected at least one overwritten-slot question at this seed"
    for ep in overwritten:
        assert ep.meta["stale_value_for_question"] in ep.options
        assert ep.meta["stale_value_for_question"] != ep.answer_text


def test_target_and_relation_sampling_uniform_empirically():
    """Target entity (normalized by n_entities) and target relation (among
    the entity's own held relations) must be sampled UNIFORMLY --
    empirical, generous tolerance given the varying n_entities."""
    eps = generate_rich_episodes(2000, seed=3, inverse_frac=0.0,
                                  min_entities=5, max_entities=5)
    target_eps = [e for e in eps if e.meta["question_mode"] == "target"]
    counts = [0] * 5
    for e in target_eps:
        counts[e.meta["target_entity"]] += 1
    total = sum(counts)
    for c in counts:
        frac = c / total
        assert 0.12 < frac < 0.28, counts   # ~1/5 uniform, generous tolerance

    rel_counts: dict = {}
    for e in target_eps:
        rel_counts[e.meta["target_relation"]] = rel_counts.get(e.meta["target_relation"], 0) + 1
    assert len(rel_counts) >= 2   # more than one relation actually gets asked about


def test_referring_devices_all_appear():
    """All three referring devices occur across a large sample (pronoun is
    the rarest -- only eligible when a referent's gender happens to be
    globally unique -- but must not be structurally impossible)."""
    eps = generate_rich_episodes(1500, seed=4, inverse_frac=0.0)
    devices = {s["device"] for e in eps for s in e.meta["referring_statements"]}
    assert devices == {"definite_description", "pronoun", "ambiguous_name"}


def test_zero_mi_device_independent_of_question_targeting():
    """Zero-MI-by-construction check (mirrors WriteBackCurriculumGenerator's
    /InstanceCurriculumGenerator's own documented argument): the device of
    an episode's first referring statement must carry no information about
    whether the eventual question targets an overwritten slot -- both are
    independent draws."""
    eps = generate_rich_episodes(1500, seed=5, inverse_frac=0.0)
    target_eps = [e for e in eps if e.meta["question_mode"] == "target"]
    overall = sum(e.meta["question_targets_overwritten"] for e in target_eps) / len(target_eps)
    by_device: dict = {}
    for e in target_eps:
        dev = e.meta["referring_statements"][0]["device"]
        by_device.setdefault(dev, []).append(e.meta["question_targets_overwritten"])
    for dev, flags in by_device.items():
        if len(flags) < 20:
            continue
        frac = sum(flags) / len(flags)
        assert abs(frac - overall) < 0.18, (dev, frac, overall)


def test_ambiguous_name_referent_is_provably_more_recent():
    """For every ambiguous_name referring statement, the referent's
    entity_order position must be strictly later than its name-sharing
    partner's -- the recency evidence that disambiguates the repeated
    name, verified directly against the episode's own recorded
    entity_order (not just trusting the generator's internal assertion)."""
    eps = generate_rich_episodes(600, seed=6, inverse_frac=0.0)
    checked = 0
    for e in eps:
        order = {ent: pos for pos, ent in enumerate(e.meta["entity_order"])}
        for stmt in e.meta["referring_statements"]:
            if stmt["device"] != "ambiguous_name":
                continue
            group = e.meta["name_groups"][stmt["mention_word"]]
            partner = next(j for j in group if j != stmt["referent"])
            assert order[stmt["referent"]] > order[partner]
            checked += 1
    assert checked > 0


def test_group_touched_once_invariant():
    """Once a name-sharing group is referred to by ANY statement, no LATER
    statement in the same episode may target either member -- protects the
    recency evidence an ambiguous_name statement commits to."""
    eps = generate_rich_episodes(600, seed=13, inverse_frac=0.0)
    for e in eps:
        seen_groups: set = set()
        for stmt in e.meta["referring_statements"]:
            referent = stmt["referent"]
            group_name = e.meta["names"][referent]
            group = e.meta["name_groups"].get(group_name)
            if group is None:
                continue
            assert group_name not in seen_groups, (e.meta["instance_seed"], group_name)
            seen_groups.add(group_name)


def test_at_most_one_question_step_per_episode():
    """The reactor's own per-episode contract (ClauseBatch.inverse_mask's
    "at most one inverse step" seam): every episode has exactly one
    question -- inverse XOR target, never both, never neither."""
    eps = generate_rich_episodes(300, seed=7, inverse_frac=0.5)
    for e in eps:
        assert e.meta["question_mode"] in ("inverse", "target")


def test_generator_answer_key_and_option_shapes():
    eps = generate_rich_episodes(300, seed=8, inverse_frac=0.3)
    for ep in eps:
        assert ep.meta["kind"] == "rich"
        assert ep.options is not None and ep.answer_idx is not None
        assert 0 <= ep.answer_idx < len(ep.options)
        assert ep.options[ep.answer_idx] == ep.answer_text
        if ep.meta["question_mode"] == "inverse":
            assert len(set(ep.options)) == len(ep.options)   # unambiguous identity strings
        else:
            assert len(set(ep.options)) == len(ep.options)


def test_n_entities_and_k_referring_within_configured_range():
    eps = generate_rich_episodes(500, seed=9, inverse_frac=0.0,
                                  min_entities=3, max_entities=8,
                                  min_referring=1, max_referring=4)
    for ep in eps:
        assert 3 <= ep.meta["n_entities"] <= 8
        assert 1 <= ep.meta["n_referring_statements"] <= 4
    ns = {e.meta["n_entities"] for e in eps}
    assert ns == {3, 4, 5, 6, 7, 8}   # the full configured range actually occurs


# ---------------------------------------------------------------------------
# 2. Batch-build shape sanity, n_entities=8, K=4.
# ---------------------------------------------------------------------------
def test_batch_build_shape_sanity_eight_entities_four_statements():
    meaning = _meaning()
    codec = _codec()
    eps = generate_rich_episodes(60, seed=10, inverse_frac=0.0,
                                  min_entities=8, max_entities=8,
                                  min_referring=4, max_referring=4)
    for ep in eps:
        assert ep.meta["n_entities"] == 8
        assert ep.meta["n_referring_statements"] == 4

    # a row whose target is a SOLO (non-name-sharing) entity, so the
    # question step carries no extra addr-redirect candidate set -- isolates
    # "exactly K addr-redirect steps" to the K referring statements alone.
    solo_eps = [e for e in eps if e.meta["names"][e.meta["target_entity"]] not in e.meta["name_groups"]]
    assert solo_eps, "expected at least one solo-target episode at this seed"

    batch = build_clause_batch(solo_eps, None, meaning, codec)
    T3 = generate_rich_episodes(60, seed=11, inverse_frac=0.0,
                                 min_entities=3, max_entities=3,
                                 min_referring=1, max_referring=1)
    batch3 = build_clause_batch(T3, None, meaning, codec)
    assert batch.entity.shape[1] > batch3.entity.shape[1]   # T grows with n_entities/K

    assert batch.cand_entity is not None
    assert batch.cand_entity.shape[2] >= 8   # Cmax padded to accommodate up to 8 candidates
    assert batch.cand_addr_mask is not None
    n_addr = (batch.cand_addr_mask > 0).sum(dim=1)
    assert torch.equal(n_addr, torch.full_like(n_addr, 4))   # exactly K=4 addr-redirect steps

    inv_mask = batch.inverse_mask
    if inv_mask is not None:
        assert bool((inv_mask.sum(dim=1) <= 1).all())   # at most one inverse step per row


def test_inverse_mode_batch_shape_and_option_grounding():
    meaning = _meaning()
    codec = _codec()
    eps = generate_rich_episodes(40, seed=12, inverse_frac=1.0,
                                  min_entities=6, max_entities=8)
    for e in eps:
        assert e.meta["question_mode"] == "inverse"
    batch = build_clause_batch(eps, None, meaning, codec)
    assert batch.inverse_mask is not None
    assert bool((batch.inverse_mask.sum(dim=1) == 1).all())   # exactly one inverse step each
    assert batch.options.shape[1] <= 4   # num_options cap, even though n_entities may exceed 4


# ---------------------------------------------------------------------------
# 3. Instance-episode identity regression: adding rich episodes to a mixed
#    batch must leave every INSTANCE-episode row's tensors byte-identical
#    (up to the shared batch's T/Cmax zero-padding) to an instance-only
#    batch -- _instance_steps itself is untouched, and this is the
#    behavioral proof that the new "rich" dispatch branch/grounding branch
#    in build_clause_batch never perturbs the pre-existing kind=="instance"
#    path.
# ---------------------------------------------------------------------------
def test_instance_episodes_byte_identical_in_batch_with_rich_episodes_mixed_in():
    meaning = _meaning()
    codec = _codec()
    inst_eps = generate_instance_episodes(20, seed=7, inverse_frac=0.3)
    rich_eps = generate_rich_episodes(15, seed=7, inverse_frac=0.3)

    b_solo = build_clause_batch(inst_eps, None, meaning, codec)
    b_mixed = build_clause_batch(inst_eps + rich_eps, None, meaning, codec)
    n = len(inst_eps)

    def pad_t(t, target_T, value=0.0):
        if t is None:
            return None
        pad_amt = target_T - t.shape[1]
        if pad_amt <= 0:
            return t
        pad_shape = list(t.shape)
        pad_shape[1] = pad_amt
        return torch.cat([t, torch.full(pad_shape, value, dtype=t.dtype)], dim=1)

    T_mixed = b_mixed.entity.shape[1]
    assert torch.equal(pad_t(b_solo.entity, T_mixed), b_mixed.entity[:n])
    assert torch.equal(pad_t(b_solo.relation, T_mixed), b_mixed.relation[:n])
    assert torch.equal(pad_t(b_solo.value, T_mixed), b_mixed.value[:n])
    assert torch.equal(pad_t(b_solo.mask, T_mixed), b_mixed.mask[:n])
    assert torch.equal(pad_t(b_solo.is_q, T_mixed), b_mixed.is_q[:n])
    assert torch.equal(b_solo.answer, b_mixed.answer[:n])
    assert torch.equal(b_solo.options, b_mixed.options[:n])

    Cmax_mixed = b_mixed.cand_entity.shape[2]

    def pad_cand(t, value=0.0):
        if t is None:
            return None
        pad_T = T_mixed - t.shape[1]
        out = t
        if out.dim() == 3:   # [b, T, C]
            pad_C = Cmax_mixed - out.shape[2]
            out = F.pad(out, (0, pad_C, 0, pad_T), value=value)
        elif out.dim() == 4:   # [b, T, C, d]
            pad_C = Cmax_mixed - out.shape[2]
            out = F.pad(out, (0, 0, 0, pad_C, 0, pad_T), value=value)
        elif out.dim() == 2:   # [b, T]
            out = F.pad(out, (0, pad_T), value=value)
        return out

    assert torch.equal(pad_cand(b_solo.cand_entity), b_mixed.cand_entity[:n])
    assert torch.equal(pad_cand(b_solo.cand_mask), b_mixed.cand_mask[:n])
    assert torch.equal(pad_cand(b_solo.cand_prior), b_mixed.cand_prior[:n])
    assert torch.equal(pad_cand(b_solo.cand_gold, value=-1.0).long(), b_mixed.cand_gold[:n])
    assert torch.equal(pad_cand(b_solo.cand_addr_mask), b_mixed.cand_addr_mask[:n])
    # cand_evidence_relation is [B, T, d] (fixed codec width, NOT padded by
    # Cmax) -- pad_t (T-only), not pad_cand ([B, T, C]/[B, T, C, d]).
    assert torch.equal(pad_t(b_solo.cand_evidence_relation, T_mixed), b_mixed.cand_evidence_relation[:n])
    assert torch.equal(pad_t(b_solo.inverse_mask, T_mixed), b_mixed.inverse_mask[:n])
    assert b_solo.cand_forced_index is None and b_mixed.cand_forced_index is None


# ---------------------------------------------------------------------------
# 4. End-to-end forced-gold eval: a question about an overwritten slot,
#    write gate forced, answers correctly via memory (return_memory /
#    return_mem_read seams available but not required for the answer_logits
#    check itself -- mirrors test_instance_curriculum.py's own end-to-end
#    pattern exactly).
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


def test_end_to_end_forced_gold_overwritten_question_answers_correctly():
    meaning = _meaning()
    codec = TPRCodec(dim=24)
    gen = RichEpisodeGenerator(seed=21, inverse_frac=0.0, min_entities=3, max_entities=5,
                                min_referring=1, max_referring=2)
    all_eps = gen.generate(1200)
    eps = [e for e in all_eps
           if e.meta["question_mode"] == "target"
           and e.meta["question_targets_overwritten"]
           and e.meta["question_device"] == "definite_description"]
    assert len(eps) >= 80, "expected enough matching episodes at this seed"
    train_eps, eval_eps = eps[:-30], eps[-30:]

    torch.manual_seed(0)
    model = ClauseReactor(dim=24, hidden=32, resolver=make_resolver("A", 24, 32))
    _freeze_write_mechanics(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)

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
    assert acc >= 0.85, acc


def test_full_batch_forward_backward_smoke():
    """End-to-end: build a mixed rich-episode batch (varying n_entities/K),
    run a full forward+backward pass through a resolver-installed
    ClauseReactor -- catches any shape/dtype mismatch the narrower unit
    tests above miss."""
    meaning = _meaning()
    codec = _codec()
    eps = generate_rich_episodes(24, seed=14, inverse_frac=0.3)
    batch = build_clause_batch(eps, None, meaning, codec)
    model = ClauseReactor(dim=DIM, hidden=24, resolver=make_resolver("A", DIM, 24))
    out = model(batch)
    loss = F.cross_entropy(out["answer_logits"], batch.answer)
    loss.backward()
    assert torch.isfinite(loss)
