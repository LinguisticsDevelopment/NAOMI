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


def test_strict_ground_default_false_is_byte_identical_to_original():
    """`strict_ground` is decode-time-only and defaults to False; confirm
    that default produces the exact same legal set as calling without the
    kwarg at all (i.e. adding the parameter changed nothing for existing
    callers). The empirical question of whether strict_ground=True ever
    masks a gold oracle action (it does, rarely -- see
    `scripts/check_strict_ground_oracle.py`) is deliberately NOT asserted
    here as a hard invariant: unlike the training-time mask, this is a
    decode-time knob explicitly allowed to trade a small amount of oracle
    unreachability for tighter generation, so it does not belong in this
    module's "never excludes a gold action" test family."""
    for open_clause in (True, False):
        for i, T in ((0, 5), (4, 5), (5, 5), (7, 5)):
            for has_clause in (True, False):
                assert (em.legal_action_types(open_clause, i, T, has_clause)
                        == em.legal_action_types(open_clause, i, T, has_clause, strict_ground=False))


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


def _tiny_gold_record() -> dict:
    """A hand-built 1-clause, 3-edge gold record ("cats chase mice"), just
    enough schema for `_gold_sites`/`clause_node_order` to walk: a real
    `tokens` list + a clause with a `predicate` surface token, a
    `predicate_grounding`, and two roles."""
    return {
        "tokens": ["cats", "chase", "mice"],
        "lattice": {"trees": [{"clauses": [{
            "predicate": "chase",
            "predicate_grounding": {"type": "entity"},
            "roles": [
                {"relation": "SUBJECT", "token_index": 0, "grounding": {"type": "sense"}},
                {"relation": "OBJECT", "token_index": 2, "grounding": {"type": "sense"}},
            ],
            "utterance_kind": "proposition",
        }]}]},
    }


def _emitted_clause(predicate_tidx, predicate_gtype, roles) -> dict:
    return {"predicate": {"token_index": predicate_tidx, "grounding": {"type": predicate_gtype}},
            "roles": [{"relation": r, "token_index": t, "grounding": {"type": g}} for r, t, g in roles]}


def test_edge_precision_recall_exact_match_forest():
    """An emitted forest whose one tree exactly reproduces the gold tree's
    3 edges: precision=1.0, recall=1.0, overgen=1.0 (right-sized)."""
    record = _tiny_gold_record()
    forest = [{"clauses": [_emitted_clause(1, "entity", [("SUBJECT", 0, "sense"), ("OBJECT", 2, "sense")])]}]
    score = em.score_record(record, forest)
    assert score.edge_precision == pytest.approx(1.0)
    assert score.edge_recall == pytest.approx(1.0)
    assert score.overgen_ratio == pytest.approx(1.0)


def test_edge_precision_recall_over_attachment_forest():
    """An emitted tree that recalls all 3 gold edges but ALSO dumps 3 extra
    (wrong) edges: recall stays 1.0 (nothing gold is missed), but precision
    drops below 1 and the over-generation ratio rises above 1 -- the exact
    signature the model-vs-dump comparison is meant to catch."""
    record = _tiny_gold_record()
    forest = [{"clauses": [_emitted_clause(1, "entity", [
        ("SUBJECT", 0, "sense"), ("OBJECT", 2, "sense"),          # correct
        ("PLACE", 0, "sense"), ("PLACE", 2, "reference"), ("SUBJECT", 1, "sense"),  # extra/wrong
    ])]}]
    score = em.score_record(record, forest)
    assert score.edge_recall == pytest.approx(1.0)
    assert score.edge_precision < 1.0
    assert score.overgen_ratio > 1.0
    # 6 emitted edges, 3 correct -> precision 0.5, overgen 2.0 exactly.
    assert score.edge_precision == pytest.approx(0.5)
    assert score.overgen_ratio == pytest.approx(2.0)


def test_dump_policy_high_recall_low_precision_high_overgen(gold_records):
    """The `policy="dump"` cheat baseline (spec: attaches maximally, never
    terminates early) must reproduce the historical failure mode: strong
    site recall (sense/slot -- the OLD metrics that made the over-generating
    run-1 checkpoint look "done") but weak edge precision and a >>1
    over-generation ratio (the NEW metrics this eval adds specifically to
    catch that). Compared against a random-legal-action baseline, dump
    should recall far more and over-generate far more."""
    from nsm_ct.ground.usvs import load_usvs

    usvs_dir = Path(__file__).resolve().parent.parent / "data" / "usvs"
    if not usvs_dir.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")
    usvs = load_usvs(str(usvs_dir))
    records = gold_records[:20]
    pos_vocab = em.build_pos_vocab(records)
    role_vocab = em.build_role_vocab(records)
    hash_buckets = 1024
    torch.manual_seed(0)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes), hash_buckets=hash_buckets,
                             d_model=32, controller_hidden=32)
    model.eval()

    dump_metrics = em.evaluate(model, records, usvs, pos_vocab, hash_buckets, policy="dump")
    random_metrics = em.evaluate(model, records, usvs, pos_vocab, hash_buckets, policy="random",
                                  rng=random.Random(0))

    assert dump_metrics["sense_recall"] > 0.9
    assert dump_metrics["sense_recall"] > random_metrics["sense_recall"]
    assert dump_metrics["overgen_ratio"] > 1.5
    assert dump_metrics["edge_precision"] < 0.5


def test_commit_margin_zero_matches_default_branching(gold_records):
    """`commit_margin=0.0` (the default) must reproduce the exact original
    always-branch-top-3 behavior -- confidence gating is strictly opt-in."""
    from nsm_ct.ground.usvs import load_usvs

    usvs_dir = Path(__file__).resolve().parent.parent / "data" / "usvs"
    if not usvs_dir.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")
    usvs = load_usvs(str(usvs_dir))
    record = gold_records[0]
    pos_vocab = em.build_pos_vocab(gold_records[:20])
    role_vocab = em.build_role_vocab(gold_records[:20])
    hash_buckets = 1024
    torch.manual_seed(0)
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes), hash_buckets=hash_buckets,
                             d_model=32, controller_hidden=32)
    model.eval()
    feats = em.build_features(record, usvs, pos_vocab, hash_buckets)

    torch.manual_seed(1)
    default_forest = em.beam_decode(model, feats, beam_width=8, k=8)
    torch.manual_seed(1)
    explicit_off_forest = em.beam_decode(model, feats, beam_width=8, k=8, commit_margin=0.0)
    assert [t["clauses"] for t in default_forest] == [t["clauses"] for t in explicit_off_forest]


def test_select_branch_actions_commits_when_confident():
    """A large gap between the best and 2nd-best legal action's log-prob,
    with a tight margin, must commit to just the top action."""
    legal_ranked = ["GROUND", "SHIFT", "CLOSE_CLAUSE"]
    logp = torch.full((len(em.ACTION_TYPES),), -100.0)
    logp[em.ACTION_INDEX["GROUND"]] = -0.01
    logp[em.ACTION_INDEX["SHIFT"]] = -5.0
    logp[em.ACTION_INDEX["CLOSE_CLAUSE"]] = -6.0
    assert em._select_branch_actions(legal_ranked, logp, commit_margin=0.5) == ["GROUND"]


def test_select_branch_actions_branches_when_unsure():
    """A near-tied best/2nd-best pair, within the margin, must branch into
    both -- but never beyond the top 2, regardless of how wide the margin
    or how many legal actions there are."""
    legal_ranked = ["GROUND", "SHIFT", "CLOSE_CLAUSE"]
    logp = torch.full((len(em.ACTION_TYPES),), -100.0)
    logp[em.ACTION_INDEX["GROUND"]] = -1.0
    logp[em.ACTION_INDEX["SHIFT"]] = -1.05
    logp[em.ACTION_INDEX["CLOSE_CLAUSE"]] = -50.0
    assert em._select_branch_actions(legal_ranked, logp, commit_margin=0.5) == ["GROUND", "SHIFT"]
    assert em._select_branch_actions(legal_ranked, logp, commit_margin=1e9) == ["GROUND", "SHIFT"]


def test_select_branch_actions_zero_margin_is_original_top3():
    """`commit_margin<=0.0` (the default/off) must reproduce the exact
    original always-branch-top-3 behavior, unconditionally."""
    legal_ranked = ["GROUND", "SHIFT", "CLOSE_CLAUSE", "STOP"]
    logp = torch.full((len(em.ACTION_TYPES),), -100.0)
    for i, a in enumerate(legal_ranked):
        logp[em.ACTION_INDEX[a]] = -float(i)
    assert em._select_branch_actions(legal_ranked, logp, commit_margin=0.0) == \
        ["GROUND", "SHIFT", "CLOSE_CLAUSE"]
