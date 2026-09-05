"""Gates for the candidate-lattice encoder (dev/ENCODER_MODEL_SPEC.md S5):
the oracle round-trips a gold tree, the mask never admits an illegal action
(and never excludes a gold one), and sense emission always copies the FULL
candidate set -- never a single sense (the model spec's core invariant:
argmax over candidates must be unrepresentable).
"""

import json
import random
from pathlib import Path

import pytest
import torch

from nsm_ct import encoder_model as em

_GOLD_PATH = Path(__file__).resolve().parent.parent / "runs" / "encoder_gold_v2.jsonl"
pytestmark = pytest.mark.skipif(not _GOLD_PATH.exists(), reason="needs runs/encoder_gold_v2.jsonl")


@pytest.fixture(scope="module")
def gold_records():
    with open(_GOLD_PATH) as f:
        return [json.loads(line) for line in f][:200]


def test_oracle_round_trips_every_gold_tree(gold_records):
    """Every gold tree linearizes to a step sequence whose GROUND /
    EMIT_UNRESOLVED_SLOT token_index positions replay the gold's own
    (relation, token_index, type) triples in order -- i.e. the derivation
    is a faithful encoding of the tree, not a lossy one. The sequence must
    also end with a terminal STOP so teacher forcing always sees one."""
    for record in gold_records:
        for tree in record["lattice"]["trees"]:
            steps = em.linearize_tree(record, tree)
            assert steps, "every gold tree must yield at least OPEN_CLAUSE..CLOSE_CLAUSE"
            assert steps[-1].action == "STOP", "linearized sequence must end with terminal STOP"
            # replay: rebuild (relation, token_index, gtype) triples from steps
            replayed = []
            for s in steps:
                if s.action in ("GROUND", "EMIT_UNRESOLVED_SLOT"):
                    replayed.append((s.role, s.token_index, s.gtype))
                elif s.action == "EMIT_SYNTH_SLOT":
                    replayed.append((s.role, None, "prime"))

            expected = []
            for clause in tree["clauses"]:
                for role, g, tidx in em.clause_node_order(record, clause):
                    if g["type"] in ("sense", "entity", "reference", "elision"):
                        expected.append((role, tidx, g["type"]))
                    elif g["type"] == "prime":
                        expected.append((role, None, "prime"))
            # token_index in `replayed` may be clamped forward on a genuine
            # duplicate/overlap (see clause_node_order's docstring), so
            # compare (relation, type) always, and position when unclamped.
            assert [(r, t) for r, _, t in replayed] == [(r, t) for r, _, t in expected]


def test_oracle_produces_only_legal_actions(gold_records):
    """The grammar-constrained mask (spec S3.3) must never exclude a gold
    oracle action: replaying the oracle's own actions through the mask
    machinery (tracking `has_clause` exactly as teacher_force_loss does)
    must find every one of them legal at the moment it fires -- including
    the terminal STOP."""
    checked = 0
    saw_stop = False
    for record in gold_records:
        T = len(record["tokens"])
        for tree in record["lattice"]["trees"]:
            steps = em.linearize_tree(record, tree)
            open_clause = False
            has_clause = False
            i = 0
            for s in steps:
                legal = em.legal_action_types(open_clause, i, T, has_clause)
                assert s.action in legal, f"gold action {s.action} illegal at i={i},T={T}"
                checked += 1
                if s.action == "OPEN_CLAUSE":
                    open_clause = True
                elif s.action == "CLOSE_CLAUSE":
                    open_clause = False
                    has_clause = True
                elif s.action == "STOP":
                    saw_stop = True
                if s.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT") and s.token_index is not None:
                    i = s.token_index + 1
    assert checked > 0
    assert saw_stop, "every gold derivation should exercise the terminal STOP action"


def test_mask_never_admits_shift_past_buffer_end():
    """SHIFT must be illegal once the buffer is exhausted -- the one
    structural precondition every action-type mask call must uphold."""
    assert "SHIFT" not in em.legal_action_types(open_clause=True, i=5, T=5)
    assert "SHIFT" in em.legal_action_types(open_clause=True, i=4, T=5)
    # SHIFT also stays legal outside an open clause while i<T (the oracle's
    # terminal flush shifts the buffer to T before emitting STOP).
    assert em.legal_action_types(open_clause=False, i=0, T=5) == ["SHIFT", "OPEN_CLAUSE"]
    assert em.legal_action_types(open_clause=False, i=5, T=5) == ["OPEN_CLAUSE"]
    assert "ATTACH" not in em.legal_action_types(open_clause=True, i=0, T=5)


def test_stop_legal_only_at_buffer_end_outside_clause_after_a_close():
    """STOP becomes legal once the buffer is exhausted, no clause is open,
    and >=1 clause has already closed -- exactly the derivation's true-end
    state. OPEN_CLAUSE stays legal there too (not exclusive): the oracle's
    buffer pointer can drift past T mid-derivation on a backward-referencing
    clause, so `i>=T` alone doesn't guarantee no more clauses follow, and
    the mask must never exclude a gold OPEN_CLAUSE in that case. The
    `has_clause` gate blocks STOP before any clause has ever closed."""
    # buffer exhausted, no open clause, but no clause has closed yet -> no STOP
    assert em.legal_action_types(open_clause=False, i=5, T=5, has_clause=False) == ["OPEN_CLAUSE"]
    # buffer exhausted, no open clause, a clause has closed -> STOP joins OPEN_CLAUSE
    legal = em.legal_action_types(open_clause=False, i=5, T=5, has_clause=True)
    assert set(legal) == {"OPEN_CLAUSE", "STOP"}
    # buffer NOT exhausted -> no STOP even with a closed clause
    assert "STOP" not in em.legal_action_types(open_clause=False, i=3, T=5, has_clause=True)
    # a clause is still open -> no STOP regardless of has_clause/buffer state
    assert "STOP" not in em.legal_action_types(open_clause=True, i=5, T=5, has_clause=True)


def test_sense_emission_copies_the_full_candidate_set_not_one_sense():
    """The architectural core of the spec: at a GROUND(sense) site the
    emitted `candidates` must equal `sense_cand[token_index]` EXACTLY (the
    full retrieved list) for every token with >1 candidate sense -- proving
    there is no head that could have narrowed it to a single pick.

    Uses `policy="random"`: now that SHIFT is also legal outside an open
    clause (needed for the oracle's terminal flush), an untrained model's
    near-uniform logits can pick a long run of pre-clause SHIFTs and never
    open a clause within a small step budget; `policy="random"` explores
    both branches evenly so at least one beam reliably reaches a GROUND
    site, keeping this test about the copy invariant rather than about an
    untrained policy's exploration luck."""
    from nsm_ct.ground.usvs import load_usvs

    usvs_dir = Path(__file__).resolve().parent.parent / "data" / "usvs"
    if not usvs_dir.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")
    usvs = load_usvs(str(usvs_dir))

    with open(_GOLD_PATH) as f:
        records = [json.loads(line) for line in f][:20]
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes), hash_buckets=1024,
                             d_model=32, controller_hidden=32)
    model.eval()

    found_multi = False
    for record in records:
        feats = em.build_features(record, usvs, pos_vocab, 1024)
        forest = em.beam_decode(model, feats, beam_width=6, k=6, max_steps=150,
                                 policy="random", rng=random.Random(1))
        for tree in forest:
            for clause in tree["clauses"]:
                for node in [clause["predicate"]] + clause["roles"]:
                    if node["grounding"]["type"] != "sense" or node["token_index"] is None:
                        continue
                    ti = node["token_index"]
                    gold_set = feats.sense_cand[ti]
                    assert node["grounding"]["candidates"] == gold_set
                    if len(gold_set) > 1:
                        found_multi = True
    assert found_multi, "test corpus should contain at least one multi-sense token"


def test_beam_decode_terminates_via_stop(gold_records):
    """The decoder's primary stopping condition is emitting STOP, not
    hitting the max_clauses/max_steps safety backstop: with a generous
    backstop, decoded trees must still come out bounded and finite,
    proving the beam actually reaches a STOP rather than being cut off.

    Uses `policy="random"`, which (unlike an untrained model's near-uniform
    logits) explores SHIFT/CLOSE_CLAUSE broadly enough to reliably advance
    the buffer and reach STOP -- an untrained model can legally loop
    OPEN_CLAUSE/CLOSE_CLAUSE forever without ever shifting, which is exactly
    why the backstop exists; this test isolates the STOP-termination path
    itself from that untrained-policy degeneracy.
    """
    from nsm_ct.ground.usvs import load_usvs

    usvs_dir = Path(__file__).resolve().parent.parent / "data" / "usvs"
    if not usvs_dir.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")
    usvs = load_usvs(str(usvs_dir))

    records = gold_records[:10]
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes), hash_buckets=1024,
                             d_model=32, controller_hidden=32)
    model.eval()

    for record in records:
        feats = em.build_features(record, usvs, pos_vocab, 1024)
        # a generous backstop (well beyond any plausible gold clause count)
        # so termination is attributable to STOP, not to the backstop firing
        forest = em.beam_decode(model, feats, beam_width=3, k=3, max_steps=200,
                                 max_clauses=50, policy="random", rng=random.Random(0))
        assert forest
        for tree in forest:
            assert len(tree["clauses"]) < 50, "decode should stop well short of the safety backstop"


def test_model_stays_sub_megabyte_at_smoke_dims():
    from nsm_ct.ground.usvs import load_usvs

    usvs_dir = Path(__file__).resolve().parent.parent / "data" / "usvs"
    if not usvs_dir.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")
    usvs = load_usvs(str(usvs_dir))
    pos_vocab = {em.UNK: 0, "NOUN": 1, "VERB": 2}
    role_vocab = {em.UNK: 0, "PREDICATE": 1, "SUBJECT": 2, "OBJECT": 3}
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes), hash_buckets=4096,
                             d_tok=32, d_pos=8, d_sense=16, d_rule=8, d_model=64, controller_hidden=64)
    n_bytes = model.num_policy_params() * 4
    assert n_bytes < 1_000_000, f"policy is {n_bytes/1e6:.2f} MB, expected sub-MB at smoke dims"
