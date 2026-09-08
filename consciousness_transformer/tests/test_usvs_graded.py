"""Gates for the USVS-GRADED scoring + soft-target loss (lead directive
2026-09-08; dev/USVS_GRADED_SCORING.md).

The three sanity controls the directive names (identity = 1.0, corrupted <
identity, a random other sentence's tree ~ the ambient floor), the role
partial-credit matrix behaving as documented, the soft-target rows being
proper distributions with maximum mass on the gold class, and -- the one
that guards everything else in the repo -- the DEFAULT loss path being
numerically unchanged on a fixed record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nsm_ct import encoder_model as em
from nsm_ct import usvs_graded as ug
from nsm_ct.tree_render import normalize_gold_tree

_ROOT = Path(__file__).resolve().parent.parent
_USVS_DIR = _ROOT / "data" / "usvs"
_GOLD = _ROOT / "runs" / "encoder_gold_v4b.jsonl"
_GOLD_FALLBACK = _ROOT / "runs" / "encoder_gold_v2.jsonl"


@pytest.fixture(scope="module")
def usvs():
    if not _USVS_DIR.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")
    from nsm_ct.ground.usvs import load_usvs
    return load_usvs(str(_USVS_DIR))


@pytest.fixture(scope="module")
def records():
    path = _GOLD if _GOLD.exists() else _GOLD_FALLBACK
    if not path.exists():
        pytest.skip("needs a gold file in runs/")
    out = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            # only records with at least two grounded role nodes are
            # informative for an alignment test
            tree = rec["lattice"]["trees"][0]
            if sum(len(c["roles"]) for c in tree["clauses"]) >= 2:
                out.append(rec)
            if len(out) >= 40:
                break
    if not out:
        pytest.skip("no usable gold records")
    return out


# ---------------------------------------------------------------------------
# S4 -- the three sanity controls
# ---------------------------------------------------------------------------

def test_identity_is_exactly_one(usvs, records):
    for rec in records[:15]:
        gold = rec["lattice"]["trees"][0]
        pred = normalize_gold_tree(rec, gold)
        r = ug.usvs_tree_similarity(pred, gold, rec, usvs, gold_is_lattice=True)
        for key in ("graded_p", "graded_r", "graded_f", "clause_struct", "graded_overall"):
            assert r[key] == pytest.approx(1.0, abs=1e-9), (
                f"{rec['text']!r}: gold vs itself must score 1.0 on {key}, got {r[key]}")


def test_corrupted_scores_below_identity(usvs, records):
    n_checked = 0
    for rec in records[:20]:
        gold = rec["lattice"]["trees"][0]
        corrupt = ug.corrupt_tree(rec, gold)
        r = ug.usvs_tree_similarity(corrupt, gold, rec, usvs, gold_is_lattice=True)
        assert r["graded_f"] < 1.0 - 1e-9, (
            f"{rec['text']!r}: SUBJECT/OBJECT swap + non-verb predicate + a phantom node "
            f"must cost something, got graded_f={r['graded_f']}")
        n_checked += 1
    assert n_checked > 0


def test_random_other_sentence_is_near_the_floor(usvs, records):
    """A different sentence's tree must score FAR below identity. It is not
    0 -- USVS gloss signatures are non-negative, so unrelated senses have a
    positive ambient cosine (design doc S2.1). This test pins the gap, which
    is the thing the metric actually claims."""
    scores = []
    for i, rec in enumerate(records[:20]):
        other = records[(i + 7) % len(records)]
        if other is rec:
            continue
        gold = rec["lattice"]["trees"][0]
        pred = normalize_gold_tree(other, other["lattice"]["trees"][0])
        r = ug.usvs_tree_similarity(pred, gold, rec, usvs, gold_is_lattice=True,
                                     pred_record=other)
        scores.append(r["graded_f"])
    assert scores
    mean = float(np.mean(scores))
    assert mean < 0.45, f"random-tree floor {mean:.3f} is too high to be a floor"


# ---------------------------------------------------------------------------
# S3.2 -- role partial credit behaves per the matrix
# ---------------------------------------------------------------------------

def test_role_partial_credit_matches_the_matrix():
    assert ug.role_weight("SUBJECT", "SUBJECT") == 1.0
    assert ug.role_weight("DESCRIPTION", "SPECIFICATION") == 0.5
    assert ug.role_weight("SPECIFICATION", "DESCRIPTION") == 0.5      # symmetric
    assert ug.role_weight("SUBJECT", "OBJECT") == 0.0                 # directive
    assert ug.role_weight("OBJECT", "SUBJECT") == 0.0
    assert ug.role_weight("COMPLEMENT", "SUBJECT_COMPLEMENT") == 0.5
    assert ug.role_weight("ADDITIVE", "FOCUS") == 0.5
    # PLACE vs a preposition role -> 0.5 (directive); two different PP roles -> 0.25
    assert ug.is_pp_role("WITH") and ug.is_pp_role("ON") and not ug.is_pp_role("PLACE")
    assert ug.role_weight("PLACE", "ON") == 0.5
    assert ug.role_weight("ON", "PLACE") == 0.5
    assert ug.role_weight("WITH", "ON") == 0.25
    # PREDICATE is structural, never partially credited against a role
    assert ug.role_weight("PREDICATE", "SUBJECT") == 0.0
    assert ug.role_weight("PREDICATE", "PREDICATE") == 1.0
    # unrelated core roles
    assert ug.role_weight("SUBJECT", "DESCRIPTION") == 0.0


def test_role_confusion_matrix_is_symmetric_and_bounded():
    for pair, w in ug.ROLE_CONFUSION.items():
        assert 0.0 <= w <= 1.0
        a, b = tuple(pair) if len(pair) == 2 else (next(iter(pair)),) * 2
        assert ug.role_weight(a, b) == ug.role_weight(b, a)


def test_node_weights():
    assert ug.node_weight("PREDICATE", True) == 2.0
    assert ug.node_weight("SUBJECT", False) == 1.0
    assert ug.node_weight("DESCRIPTION", False) == 0.5


def test_partial_role_credit_shows_up_in_a_tree_score(usvs, records):
    """Relabelling one DESCRIPTION as SPECIFICATION must score strictly
    between a phantom relabelling (SUBJECT->OBJECT elsewhere) and identity."""
    for rec in records:
        gold = rec["lattice"]["trees"][0]
        norm = normalize_gold_tree(rec, gold)
        target = None
        for clause in norm["clauses"]:
            for r in clause["roles"]:
                if r["relation"] == "DESCRIPTION":
                    target = r
                    break
        if target is None:
            continue
        import copy
        near = copy.deepcopy(norm)
        for clause in near["clauses"]:
            for r in clause["roles"]:
                if r["relation"] == "DESCRIPTION":
                    r["relation"] = "SPECIFICATION"
                    break
        far = copy.deepcopy(norm)
        for clause in far["clauses"]:
            for r in clause["roles"]:
                if r["relation"] == "DESCRIPTION":
                    r["relation"] = "SUBJECT"
                    break
        s_id = ug.usvs_tree_similarity(norm, gold, rec, usvs, gold_is_lattice=True)["graded_f"]
        s_near = ug.usvs_tree_similarity(near, gold, rec, usvs, gold_is_lattice=True)["graded_f"]
        s_far = ug.usvs_tree_similarity(far, gold, rec, usvs, gold_is_lattice=True)["graded_f"]
        assert s_far < s_near < s_id, (rec["text"], s_far, s_near, s_id)
        return
    pytest.skip("no DESCRIPTION role in the sampled gold records")


# ---------------------------------------------------------------------------
# S2 -- node vectors
# ---------------------------------------------------------------------------

def test_reserved_vectors_compare_exactly(usvs):
    rec = {"tokens": ["alice", "bob"], "token_sense_candidates": []}
    ent_a = {"token_index": 0, "grounding": {"type": "entity"}}
    ent_a2 = {"token_index": 0, "grounding": {"type": "entity"}}
    ent_b = {"token_index": 1, "grounding": {"type": "entity"}}
    ref_a = {"token_index": 0, "grounding": {"type": "reference"}}
    eli = {"token_index": None, "grounding": {"type": "elision"}}
    eli2 = {"token_index": None, "grounding": {"type": "elision"}}
    prime_you = {"token_index": None, "grounding": {"type": "prime", "prime": "YOU"}}
    prime_i = {"token_index": None, "grounding": {"type": "prime", "prime": "I"}}

    def cos(x, y):
        vx, vy = ug.node_vector(x, rec, usvs), ug.node_vector(y, rec, usvs)
        return float(np.dot(vx, vy))

    assert cos(ent_a, ent_a2) == pytest.approx(1.0)
    assert cos(ent_a, ent_b) == pytest.approx(ug.TYPE_W ** 2)
    assert cos(prime_you, prime_you) == pytest.approx(1.0)
    assert cos(prime_you, prime_i) == pytest.approx(ug.TYPE_W ** 2)
    assert cos(eli, eli2) == pytest.approx(1.0)
    # types the contract does NOT relate are orthogonal ...
    assert cos(ent_a, ref_a) == pytest.approx(0.0, abs=1e-12)
    assert cos(prime_you, ref_a) == pytest.approx(0.0, abs=1e-12)
    # ... but reference/elision are "one shape, three instances" (contract
    # S4), so they get GTYPE_CONFUSION-derived partial credit, never 1.0.
    assert 0.0 < cos(ref_a, eli) < 1.0


def test_sense_node_vector_is_the_candidate_set_mean(usvs):
    sid = usvs.sense_ids[0]
    rec = {"tokens": ["x"], "token_sense_candidates": []}
    node = {"token_index": 0, "grounding": {"type": "sense", "candidates": [sid]}}
    v = ug.node_vector(node, rec, usvs)
    dense = usvs.sense_dense(sid)
    expect = dense / np.linalg.norm(dense)
    assert np.allclose(v[:len(usvs.axes)], expect, atol=1e-9)
    assert np.allclose(v[len(usvs.axes):], 0.0)


# ---------------------------------------------------------------------------
# S3.4 -- the in-module Hungarian solver
# ---------------------------------------------------------------------------

def test_hungarian_beats_greedy_on_the_classic_trap():
    # greedy takes (0,0)=0.9 then is forced onto (1,1)=0.1 -> 1.0;
    # the optimum is (0,1)+(1,0) = 0.8+0.8 = 1.6
    m = np.array([[0.9, 0.8], [0.8, 0.1]])
    h = ug._hungarian_max(m)
    assert sum(m[i, j] for i, j in h) == pytest.approx(1.6)
    g = ug._greedy_max(m)
    assert sum(m[i, j] for i, j in g) == pytest.approx(1.0)


def test_hungarian_handles_rectangular_both_ways():
    m = np.array([[0.5, 0.9, 0.1]])
    assert ug._hungarian_max(m) == [(0, 1)]
    assert ug._hungarian_max(m.T) == [(1, 0)]


# ---------------------------------------------------------------------------
# S5 -- soft targets
# ---------------------------------------------------------------------------

def _role_vocab():
    labels = [em.UNK, "PREDICATE", "SUBJECT", "OBJECT", "DESCRIPTION", "SPECIFICATION", "WITH"]
    return {lab: i for i, lab in enumerate(labels)}


def test_soft_role_target_sums_to_one_and_peaks_on_gold():
    vocab = _role_vocab()
    for gold in ("SUBJECT", "DESCRIPTION", "WITH", "PREDICATE"):
        row = ug.soft_role_target(gold, vocab)
        assert len(row) == len(vocab)
        assert sum(row) == pytest.approx(1.0)
        assert all(v >= 0.0 for v in row)
        assert int(np.argmax(row)) == vocab[gold], gold


def test_soft_role_target_gives_related_roles_more_mass_than_unrelated():
    vocab = _role_vocab()
    row = ug.soft_role_target("DESCRIPTION", vocab)
    assert row[vocab["SPECIFICATION"]] > row[vocab["SUBJECT"]]
    assert row[vocab["SUBJECT"]] == pytest.approx(row[vocab["OBJECT"]])


def test_soft_role_target_temperature_sharpens_toward_one_hot():
    vocab = _role_vocab()
    hot = ug.soft_role_target("DESCRIPTION", vocab, temperature=0.01)
    warm = ug.soft_role_target("DESCRIPTION", vocab, temperature=1.0)
    assert hot[vocab["DESCRIPTION"]] > warm[vocab["DESCRIPTION"]]
    assert sum(hot) == pytest.approx(1.0)


def test_soft_gtype_and_source_targets():
    grow = ug.soft_gtype_target("reference", em.GROUNDING_TYPES)
    assert sum(grow) == pytest.approx(1.0)
    assert int(np.argmax(grow)) == em.GTYPE_INDEX["reference"]
    assert grow[em.GTYPE_INDEX["elision"]] > grow[em.GTYPE_INDEX["prime"]]

    srow = ug.soft_source_target("context", em.SOURCES)
    assert sum(srow) == pytest.approx(1.0)
    assert int(np.argmax(srow)) == em.SOURCE_INDEX["context"]
    assert srow[em.SOURCE_INDEX["memory"]] > srow[em.SOURCE_INDEX["lexicon"]]


# ---------------------------------------------------------------------------
# The regression that matters: the DEFAULT loss path is unchanged
# ---------------------------------------------------------------------------

def _fixed_model_and_item(usvs, records):
    rec = records[0]
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    torch.manual_seed(1234)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes),
                             hash_buckets=512, d_model=32, controller_hidden=32)
    feats = em.build_features(rec, usvs, pos_vocab, 512)
    steps = em.linearize_tree(rec, rec["lattice"]["trees"][0])
    return model, feats, steps


def test_default_loss_path_is_numerically_unchanged(usvs, records):
    """`soft_targets=None` (the default) must reproduce the pre-existing
    loss EXACTLY -- recomputed here from the same one-hot cross-entropies
    the original implementation used, independent of the new code path."""
    import torch.nn.functional as F

    model, feats, steps = _fixed_model_and_item(usvs, records)
    got = em.teacher_force_loss(model, feats, steps, terminal_weight=4.0)

    # -- an independent re-implementation of the ORIGINAL loss --------------
    enc = model.encode(feats)
    T = enc.shape[0]
    h = model.init_controller_state()
    open_clause, has_clause = False, False
    open_kind_id = model._none_clause_id
    prev_action_id = model._start_action_id
    i = 0
    w = torch.ones(len(em.ACTION_TYPES))
    for a in em.TERMINAL_ACTION_TYPES:
        w[em.ACTION_INDEX[a]] = 4.0
    losses = []
    for step in steps:
        i_clamped = min(i, T - 1) if T > 0 else 0
        enc_i = enc[i_clamped] if T > 0 else torch.zeros(model.d_model)
        h = model.controller_step(enc_i, open_kind_id, prev_action_id, h)
        legal = em.legal_action_types(open_clause, i, T, has_clause)
        logits = model.action_type_head(h).squeeze(0) + em._mask_vector(legal)
        losses.append(F.cross_entropy(logits.unsqueeze(0),
                                       torch.tensor([em.ACTION_INDEX[step.action]]), weight=w))
        if step.action == "OPEN_CLAUSE":
            losses.append(F.cross_entropy(model.kind_head(h),
                                           torch.tensor([em.KIND_INDEX.get(step.kind, 0)])))
            open_clause = True
            open_kind_id = em.KIND_INDEX.get(step.kind, 0)
        elif step.action == "CLOSE_CLAUSE":
            open_clause = False
            open_kind_id = model._none_clause_id
            has_clause = True
        elif step.action in ("GROUND", "ATTACH", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT"):
            losses.append(F.cross_entropy(model.role_head(h),
                                           torch.tensor([model.role_id(step.role)])))
        if step.action in ("GROUND", "EMIT_SYNTH_SLOT", "EMIT_UNRESOLVED_SLOT") and step.gtype is not None:
            losses.append(F.cross_entropy(model.gtype_head(h),
                                           torch.tensor([em.GTYPE_INDEX[step.gtype]])))
        if step.gtype in ("sense", "reference", "elision") and step.source is not None:
            losses.append(F.cross_entropy(model.source_head(h),
                                           torch.tensor([em.SOURCE_INDEX.get(step.source, 0)])))
        if step.action == "EMIT_SYNTH_SLOT" and step.prime is not None:
            losses.append(F.cross_entropy(model.prime_head(h),
                                           torch.tensor([em.PRIME_INDEX.get(step.prime,
                                                                             em.PRIME_INDEX["<UNK_PRIME>"])])))
        if step.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT", "EMIT_SYNTH_SLOT") \
                and step.token_index is not None:
            i = step.token_index + 1
        prev_action_id = em.ACTION_INDEX[step.action]
    expected = torch.stack(losses).sum()

    assert float(got) == pytest.approx(float(expected), abs=1e-9)


def test_soft_loss_differs_but_stays_finite_and_close(usvs, records):
    model, feats, steps = _fixed_model_and_item(usvs, records)
    hard = float(em.teacher_force_loss(model, feats, steps, terminal_weight=4.0))
    soft = float(em.teacher_force_loss(model, feats, steps, terminal_weight=4.0,
                                        soft_targets=em.SoftTargetConfig()))
    assert soft == soft and abs(soft) < float("inf")
    assert soft != hard, "the soft target must actually change the loss"
    assert abs(soft - hard) / max(hard, 1.0) < 0.5, "soft loss should be a perturbation, not a rewrite"


def test_soft_loss_at_tiny_temperature_approaches_the_hard_loss(usvs, records):
    model, feats, steps = _fixed_model_and_item(usvs, records)
    hard = float(em.teacher_force_loss(model, feats, steps, terminal_weight=4.0))
    near = float(em.teacher_force_loss(model, feats, steps, terminal_weight=4.0,
                                        soft_targets=em.SoftTargetConfig(temperature=1e-4)))
    assert near == pytest.approx(hard, rel=1e-6)


def test_soft_loss_backprops(usvs, records):
    model, feats, steps = _fixed_model_and_item(usvs, records)
    loss = em.teacher_force_loss(model, feats, steps, soft_targets=em.SoftTargetConfig())
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


# ---------------------------------------------------------------------------
# record / forest level plumbing
# ---------------------------------------------------------------------------

def test_score_record_graded_on_the_gold_forest_is_one(usvs, records):
    rec = records[0]
    forest = [normalize_gold_tree(rec, t) for t in rec["lattice"]["trees"]]
    out = ug.score_record_graded(rec, forest, usvs)
    assert out["graded_f"] == pytest.approx(1.0, abs=1e-9)


def test_score_record_graded_on_an_empty_forest_is_zero(usvs, records):
    out = ug.score_record_graded(records[0], [], usvs)
    assert out["graded_f"] == pytest.approx(0.0)


def test_evaluate_full_metric_flag_is_additive(usvs, records):
    from nsm_ct import encoder_train_util as etu
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    torch.manual_seed(7)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes),
                             hash_buckets=512, d_model=32, controller_hidden=32)
    recs = records[:4]
    base = etu.evaluate_full(model, recs, usvs, pos_vocab, 512, beam_width=1, k=1)
    graded = etu.evaluate_full(model, recs, usvs, pos_vocab, 512, beam_width=1, k=1,
                                metric="graded")
    for key, val in base.items():
        other = graded[key]
        assert (val == other) or (val != val and other != other), f"{key} changed: {val} -> {other}"
    for key in etu.GRADED_FIELDS:
        assert key in graded and "rank1_" + key in graded
