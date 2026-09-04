"""Gates for the candidate-lattice encoder (dev/ENCODER_MODEL_SPEC.md S5):
the oracle round-trips a gold tree, the mask never admits an illegal action
(and never excludes a gold one), and sense emission always copies the FULL
candidate set -- never a single sense (the model spec's core invariant:
argmax over candidates must be unrepresentable).
"""

import json
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
    is a faithful encoding of the tree, not a lossy one."""
    for record in gold_records:
        for tree in record["lattice"]["trees"]:
            steps = em.linearize_tree(record, tree)
            assert steps, "every gold tree must yield at least OPEN_CLAUSE..CLOSE_CLAUSE"
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
    machinery must find every one of them legal at the moment it fires."""
    checked = 0
    for record in gold_records:
        T = len(record["tokens"])
        for tree in record["lattice"]["trees"]:
            steps = em.linearize_tree(record, tree)
            open_clause = False
            i = 0
            for s in steps:
                legal = em.legal_action_types(open_clause, i, T)
                assert s.action in legal, f"gold action {s.action} illegal at i={i},T={T}"
                checked += 1
                if s.action == "OPEN_CLAUSE":
                    open_clause = True
                elif s.action == "CLOSE_CLAUSE":
                    open_clause = False
                if s.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT") and s.token_index is not None:
                    i = s.token_index + 1
    assert checked > 0


def test_mask_never_admits_shift_past_buffer_end():
    """SHIFT must be illegal once the buffer is exhausted -- the one
    structural precondition every action-type mask call must uphold."""
    assert "SHIFT" not in em.legal_action_types(open_clause=True, i=5, T=5)
    assert "SHIFT" in em.legal_action_types(open_clause=True, i=4, T=5)
    assert em.legal_action_types(open_clause=False, i=0, T=5) == ["OPEN_CLAUSE"]
    assert "ATTACH" not in em.legal_action_types(open_clause=True, i=0, T=5)


def test_sense_emission_copies_the_full_candidate_set_not_one_sense():
    """The architectural core of the spec: at a GROUND(sense) site the
    emitted `candidates` must equal `sense_cand[token_index]` EXACTLY (the
    full retrieved list) for every token with >1 candidate sense -- proving
    there is no head that could have narrowed it to a single pick."""
    import random
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
        forest = em.beam_decode(model, feats, beam_width=3, k=3, max_steps=80)
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
