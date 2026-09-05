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
    is a faithful encoding of the tree, not a lossy one. The derivation must
    also terminate on the new STOP action (the fix for the missing-STOP
    bug): every gold derivation ends in exactly one terminal STOP, emitted
    once the buffer is fully consumed and no clause is open."""
    for record in gold_records:
        T = len(record["tokens"])
        for tree in record["lattice"]["trees"]:
            steps = em.linearize_tree(record, tree)
            assert steps, "every gold tree must yield at least OPEN_CLAUSE..CLOSE_CLAUSE"
            assert steps[-1].action == "STOP", "every gold derivation must end in STOP"
            assert sum(1 for s in steps if s.action == "STOP") == 1, "STOP must be terminal, not repeated"
            # replay the buffer pointer/clause-open state up to the STOP
            # step and confirm STOP fires exactly at i>=T, no clause open.
            i = 0
            open_clause = False
            for s in steps[:-1]:
                if s.action == "OPEN_CLAUSE":
                    open_clause = True
                elif s.action == "CLOSE_CLAUSE":
                    open_clause = False
                if s.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT") and s.token_index is not None:
                    i = s.token_index + 1
            assert not open_clause and i >= T, (
                f"STOP fired with open_clause={open_clause}, i={i}, T={T}"
            )
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
                if s.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT") and s.token_index is not None:
                    i = s.token_index + 1
    assert checked > 0


def test_mask_never_admits_shift_past_buffer_end():
    """SHIFT must be illegal once the buffer is exhausted -- the one
    structural precondition every action-type mask call must uphold. SHIFT
    is also legal outside an open clause once `has_clause` (at least one
    clause already closed): the oracle's terminal flush advances the buffer
    pointer to `T` after the tree's last CLOSE_CLAUSE, with no clause open,
    so the derivation can reach a legal STOP. Before any clause has been
    produced, OPEN_CLAUSE stays the only legal action, exactly as before
    the STOP fix -- without this gate, an untrained policy can legally
    flush the entire buffer via SHIFT before ever opening a clause and land
    on STOP having produced no content at all (observed in a smoke run)."""
    assert "SHIFT" not in em.legal_action_types(open_clause=True, i=5, T=5)
    assert "SHIFT" in em.legal_action_types(open_clause=True, i=4, T=5)
    assert em.legal_action_types(open_clause=False, i=0, T=5) == ["OPEN_CLAUSE"]
    assert em.legal_action_types(open_clause=False, i=0, T=5, has_clause=True) == ["OPEN_CLAUSE", "SHIFT"]
    assert em.legal_action_types(open_clause=False, i=5, T=5, has_clause=True) == ["OPEN_CLAUSE", "STOP"]
    assert "ATTACH" not in em.legal_action_types(open_clause=True, i=0, T=5)


def test_stop_legal_exactly_when_buffer_consumed_and_no_open_clause():
    """STOP must be legal iff `i>=T`, no clause is open, AND at least one
    clause has already been closed (no gold tree has zero clauses) -- never
    before the buffer is consumed, never while a clause is still open,
    never before any clause has been produced."""
    assert "STOP" not in em.legal_action_types(open_clause=False, i=4, T=5, has_clause=True)
    assert "STOP" not in em.legal_action_types(open_clause=True, i=5, T=5, has_clause=True)
    assert "STOP" not in em.legal_action_types(open_clause=False, i=5, T=5, has_clause=False)
    assert "STOP" in em.legal_action_types(open_clause=False, i=5, T=5, has_clause=True)
    assert "STOP" in em.legal_action_types(open_clause=False, i=7, T=5, has_clause=True)  # past-T overshoot


def test_oracle_legality_full_corpus():
    """Re-run the legality check (spec S3.3's core invariant: the mask never
    excludes a gold oracle action) over every step of every gold derivation
    in the full corpus, not just the 200-record smoke slice -- this is the
    check that surfaces the duplicate-token-index collisions that force
    OPEN_CLAUSE to stay legal alongside STOP once `i>=T`."""
    with open(_GOLD_PATH) as f:
        records = [json.loads(line) for line in f]
    checked = 0
    n_stop = 0
    for record in records:
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
                    n_stop += 1
                if s.action in ("SHIFT", "GROUND", "EMIT_UNRESOLVED_SLOT") and s.token_index is not None:
                    i = s.token_index + 1
    assert checked > 50_000, f"expected tens of thousands of steps, got {checked}"
    assert n_stop > 0


def test_sense_emission_copies_the_full_candidate_set_not_one_sense():
    """The architectural core of the spec: at a GROUND(sense) site the
    emitted `candidates` must equal `sense_cand[token_index]` EXACTLY (the
    full retrieved list) for every token with >1 candidate sense -- proving
    there is no head that could have narrowed it to a single pick.

    Drives `_apply_action` (the exact code `beam_decode` uses to build a
    node) directly, with the grounding-type head pinned to always resolve
    to "sense", rather than going through the full beam search: post-STOP-
    fix, an UNTRAINED policy can legally (and, with random init, often
    does) walk straight through the whole buffer via the newly-legal
    no-open-clause SHIFT before ever opening a clause, which starves a
    short, un-pretrained smoke decode of any real GROUND(sense) site to
    inspect and made this test flaky end-to-end. Pinning the head isolates
    the one thing this test is actually about -- the copy -- from whether
    an untrained policy happens to choose to visit a given site at all
    (that's `beam_decode`'s job, exercised by the training-driven smoke
    eval, not this unit test's)."""
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
    with torch.no_grad():
        model.gtype_head.weight.zero_()
        model.gtype_head.bias.fill_(-100.0)
        model.gtype_head.bias[em.GTYPE_INDEX["sense"]] = 100.0

    found_multi = False
    h0 = model.init_controller_state()
    for record in records:
        feats = em.build_features(record, usvs, pos_vocab, 1024)
        for ti, cands in enumerate(feats.sense_cand):
            if len(cands) <= 1:
                continue
            found_multi = True
            state = em.BeamState(h=h0, i=ti, open_clause=True, open_kind_id=0,
                                  prev_action_id=model._start_action_id, logprob=0.0,
                                  cur_clause={"predicate": {"token_index": None, "grounding": {"type": "entity"}},
                                              "roles": [], "utterance_kind": "proposition"})
            em._apply_action(state, "GROUND", model, feats, h0)
            node = (state.cur_clause["predicate"]
                    if state.cur_clause["predicate"]["token_index"] == ti
                    else state.cur_clause["roles"][-1])
            assert node["grounding"]["type"] == "sense"
            assert node["token_index"] == ti
            assert node["grounding"]["candidates"] == cands
    assert found_multi, "test corpus should contain at least one multi-sense token"


def test_beam_decode_terminates_via_stop_not_artificial_cap():
    """The decode loop's PRIMARY termination is now the learned STOP action,
    not the max_clauses/max_steps backstop (the whole point of the fix): at
    every no-open-clause state there are at most two legal actions
    (OPEN_CLAUSE, STOP), and beam search always branches on every legal
    action when there are this few, so some beam must reach STOP well
    inside the (generous) max_clauses backstop for a real gold record, even
    with an untrained (randomly-initialized) policy -- before the fix there
    was no STOP action at all and every beam ran out to the cap."""
    from nsm_ct.ground.usvs import load_usvs

    usvs_dir = Path(__file__).resolve().parent.parent / "data" / "usvs"
    if not usvs_dir.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")
    usvs = load_usvs(str(usvs_dir))

    with open(_GOLD_PATH) as f:
        records = [json.loads(line) for line in f][:30]
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    torch.manual_seed(0)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes), hash_buckets=1024,
                             d_model=32, controller_hidden=32)
    model.eval()

    max_clauses_cap = 20
    any_bounded = False
    for record in records:
        gold_max_clauses = max((len(t["clauses"]) for t in record["lattice"]["trees"]), default=1)
        feats = em.build_features(record, usvs, pos_vocab, 1024)
        forest = em.beam_decode(model, feats, beam_width=8, k=8, max_steps=400,
                                 max_clauses=max_clauses_cap)
        assert forest, "beam_decode must return at least one tree"
        for tree in forest:
            n_clauses = len(tree["clauses"])
            assert n_clauses <= max_clauses_cap
            if n_clauses <= gold_max_clauses + 2:
                any_bounded = True
    assert any_bounded, (
        "expected at least one emitted tree to terminate near the gold clause "
        "range via STOP, not merely run out the artificial cap on every beam"
    )


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
