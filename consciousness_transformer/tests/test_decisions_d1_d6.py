"""Gates for the SIX locked gold/schema decisions (dev/CURRENT_STATE.md,
"DECISIONS LOCKED", 2026-09-07): D1 forest top-1 default (+ margin/all),
D2 interjection gloss-grounding, D3 prime I, D4 QUANTITY/ADDITIVE/FOCUS
roles, D6 elision surface-carrier retention.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nsm_ct import encoder_model as em  # noqa: E402
from nsm_ct.clause import FIRST_PERSON_SINGULAR, MODIFIER_RELATIONS  # noqa: E402

import build_encoder_gold_v2 as gold_v2  # noqa: E402
import hand_gold as hg  # noqa: E402

_USVS_DIR = _ROOT / "data" / "usvs"


@pytest.fixture(scope="module")
def usvs():
    if not _USVS_DIR.exists():
        pytest.skip("needs data/usvs (run scripts/build_usvs.py)")
    from nsm_ct.ground.usvs import load_usvs
    return load_usvs(str(_USVS_DIR))


# ---------------------------------------------------------------------------
# D3 -- add prime I
# ---------------------------------------------------------------------------

def test_primes_includes_i_and_you():
    assert em.PRIMES == ["YOU", "I", "<UNK_PRIME>"]
    assert em.PRIME_INDEX["I"] != em.PRIME_INDEX["YOU"]


def test_first_person_singular_set():
    assert FIRST_PERSON_SINGULAR == {"i", "me", "myself"}


def test_build_encoder_gold_v2_ground_word_first_person_resolves_to_prime_i():
    for w in ("me", "I", "myself", "Me", "MYSELF"):
        g = gold_v2.ground_word(None, w)  # usvs never touched on this path
        assert g == {"type": "prime", "prime": "I", "candidates": None}


def test_hand_gold_ground_w_first_person_resolves_to_prime_i(usvs):
    for w in ("me", "I", "myself"):
        g = hg.ground_W(usvs, hg.W(w))
        assert g == {"type": "prime", "prime": "I", "candidates": None}


def test_prime_with_real_surface_token_round_trips(usvs):
    """'tell me about the dog .': the INDIRECT_OBJECT 'me' grounds prime I
    WITH a real token_index (unlike the synthesized addressee YOU, which
    has none) -- the oracle must shift to and consume it, not silently
    flush it under a stale buffer position (the D3 fix to
    `linearize_tree`'s prime branch + `teacher_force_loss`'s i-advance)."""
    rec = hg.hand_gold_record(
        "tell me about the dog .",
        [hg.C(kind="imperative", predicate=hg.W("tell"),
              roles=[("SUBJECT", hg.PRIME("YOU")),
                     ("INDIRECT_OBJECT", hg.W("me")),
                     ("ABOUT", hg.W("dog"))])],
        usvs=usvs)
    report = hg.check_record(rec, usvs, strict=True)
    assert report["ok"], report["problems"]

    tree = rec["lattice"]["trees"][0]
    steps = em.linearize_tree(rec, tree)
    synth_by_role = {s.role: s for s in steps if s.action == "EMIT_SYNTH_SLOT"}
    assert synth_by_role["SUBJECT"].token_index is None       # synthesized addressee: no surface token
    assert synth_by_role["SUBJECT"].prime == "YOU"
    assert synth_by_role["INDIRECT_OBJECT"].token_index == 1  # 'me' -- real token, shifted + consumed
    assert synth_by_role["INDIRECT_OBJECT"].prime == "I"

    # teacher-forced loss must stay finite through the real training path
    pos_vocab = em.build_pos_vocab([rec])
    role_vocab = em.build_role_vocab([rec])
    model = em.EncoderModel(pos_vocab, role_vocab, d_axes=len(usvs.axes), hash_buckets=1024,
                             d_model=32, controller_hidden=32)
    feats = em.build_features(rec, usvs, pos_vocab, model.hash_buckets)
    import torch
    loss = em.teacher_force_loss(model, feats, steps)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# D4 -- QUANTITY / ADDITIVE / FOCUS role labels (structural only)
# ---------------------------------------------------------------------------

def test_modifier_relations_includes_new_roles():
    for r in ("QUANTITY", "ADDITIVE", "FOCUS"):
        assert r in MODIFIER_RELATIONS
    # pre-existing modifier roles must still be present (additive, not a rename)
    for r in ("DESCRIPTION", "SPECIFICATION", "COMPLEMENT", "SUBJECT_COMPLEMENT"):
        assert r in MODIFIER_RELATIONS


def test_hand_gold_accepts_quantity_and_additive_roles(usvs):
    rec = hg.hand_gold_record(
        "the boy wants more candy too .",
        [hg.C(predicate=hg.W("wants"),
              roles=[("SUBJECT", hg.W("boy")), ("OBJECT", hg.W("candy")),
                     ("QUANTITY", hg.W("more")), ("ADDITIVE", hg.W("too"))])],
        usvs=usvs)
    report = hg.check_record(rec, usvs, strict=True)
    assert report["ok"], report["problems"]


def test_hand_gold_accepts_focus_role(usvs):
    rec = hg.hand_gold_record(
        "only the boy runs .",
        [hg.C(predicate=hg.W("runs"),
              roles=[("SUBJECT", hg.W("boy")), ("FOCUS", hg.W("only"))])],
        usvs=usvs)
    report = hg.check_record(rec, usvs, strict=True)
    assert report["ok"], report["problems"]


# ---------------------------------------------------------------------------
# D2 -- interjections ground to a real sense, never bare `entity`
# ---------------------------------------------------------------------------

def test_ugh_resolves_to_nonempty_nonentity_sense_with_nonzero_vector(usvs):
    cands = usvs.senses_of("ugh")
    assert cands, "ugh must resolve to >=1 sense after D2's gloss-grounding"
    assert cands[0].startswith("interj."), "ugh has no WordNet synset -- must be the minted gloss sense"
    vec = usvs.sense_dense(cands[0])
    assert vec is not None
    assert float(np.abs(vec).sum()) > 0.0, "ugh's USVS coordinate must be non-zero"

    g = hg.ground_W(usvs, hg.W("ugh"))
    assert g["type"] == "sense", "ugh must ground as a real sense slot, never bare 'entity'"
    assert g["candidates"] == cands

    g2 = gold_v2.ground_word(usvs, "ugh")
    assert g2["type"] == "sense"
    assert g2["candidates"] == cands


def test_nonsense_resolves_to_its_wordnet_sense(usvs):
    cands = usvs.senses_of("nonsense")
    assert cands
    assert "nonsense.n.01" in cands, "content interjections keep their real WordNet sense"
    assert not cands[0].startswith("interj."), "a WordNet-covered word must not fall through to a minted gloss sense"

    g = hg.ground_W(usvs, hg.W("nonsense"))
    assert g["type"] == "sense"
    assert g["candidates"] == cands


def test_interjection_clause_has_no_appraisal_node(usvs):
    """utterance_kind is the ENTIRE extra signal -- no FEEL/valence/stance
    fields anywhere on the clause (the Appraisal grounding decision)."""
    rec = hg.hand_gold_record(
        "ugh .", [hg.C(kind="interjection", predicate=hg.W("ugh"), roles=[])], usvs=usvs)
    report = hg.check_record(rec, usvs, strict=True)
    assert report["ok"], report["problems"]
    clause = rec["lattice"]["trees"][0]["clauses"][0]
    assert clause["utterance_kind"] == "interjection"
    assert clause["predicate_grounding"]["type"] == "sense"
    assert set(clause.keys()) == {"predicate", "predicate_grounding", "predicate_token_index",
                                  "is_question", "utterance_kind", "roles"}


# ---------------------------------------------------------------------------
# D6 -- elision slots retain their surface carrier
# ---------------------------------------------------------------------------

def test_elision_predicate_keeps_surface_carrier_token_index(usvs):
    ctx = [hg.context_entry(usvs, "the boys break the window .",
                            [hg.C(predicate=hg.W("break"),
                                  roles=[("SUBJECT", hg.W("boys")), ("OBJECT", hg.W("window"))])])]
    rec = hg.hand_gold_record(
        "the dog did .",
        [hg.C(predicate=hg.CTX("elision", of=hg.PREDICATE, scope="predicates", word="did"),
              roles=[("SUBJECT", hg.W("dog")),
                     ("OBJECT", hg.CTX("elision", of="window", scope="roles"))])],
        usvs=usvs, context=ctx)
    report = hg.check_record(rec, usvs, strict=True)
    assert report["ok"], report["problems"]

    clause = rec["lattice"]["trees"][0]["clauses"][0]
    assert clause["predicate"] is None, "the elided predicate's MEANING is still inherited, not this word"
    assert clause["predicate_grounding"]["type"] == "elision"
    assert clause["predicate_token_index"] == 2, "'did' in ['the','dog','did','.']"

    steps = em.linearize_tree(rec, rec["lattice"]["trees"][0])
    pred_steps = [s for s in steps if s.role == "PREDICATE"]
    assert len(pred_steps) == 1
    assert pred_steps[0].action == "EMIT_UNRESOLVED_SLOT"
    assert pred_steps[0].token_index == 2, "the oracle must shift to and consume 'did', not flush it"


def test_elision_predicate_without_carrier_still_token_index_none(usvs):
    """Pre-D6 shape (no `word=` on the CTX predicate spec) must be
    byte-identical: `predicate_token_index` absent/None, oracle emits
    EMIT_UNRESOLVED_SLOT with token_index=None, same as before this
    field existed."""
    ctx = [hg.context_entry(usvs, "the children eat bread .",
                            [hg.C(predicate=hg.W("eat"),
                                  roles=[("SUBJECT", hg.W("children")), ("OBJECT", hg.W("bread"))])])]
    rec = hg.hand_gold_record(
        "more !",
        [hg.C(predicate=hg.CTX("elision", of=hg.PREDICATE, scope="predicates"),
              roles=[("OBJECT", hg.CTX("elision", of="bread", scope="roles")),
                     ("QUANTITY", hg.W("more"))])],
        usvs=usvs, context=ctx)
    report = hg.check_record(rec, usvs, strict=True)
    assert report["ok"], report["problems"]
    clause = rec["lattice"]["trees"][0]["clauses"][0]
    assert clause.get("predicate_token_index") is None
    steps = em.linearize_tree(rec, rec["lattice"]["trees"][0])
    pred_step = next(s for s in steps if s.role == "PREDICATE")
    assert pred_step.token_index is None


# ---------------------------------------------------------------------------
# D1 -- forest top-1 default (+ margin / all)
# ---------------------------------------------------------------------------

def _mk_tree(pred_tok, roles):
    sense_g = {"type": "sense", "candidates": [], "retrieval":
               {"source": "lexicon", "method": "lemma_senses", "ref": None}}
    return {"clauses": [{
        "predicate": pred_tok,
        "predicate_grounding": sense_g,
        "utterance_kind": "proposition",
        "roles": [{"relation": rel, "word": w, "token_index": idx, "is_entity": False,
                   "grounding": dict(sense_g)}
                  for rel, w, idx in roles],
    }]}


def test_prune_forest_top1_keeps_only_the_best_tree():
    trees = [_mk_tree("saw", [("SUBJECT", "he", 0)]),
             _mk_tree("saw", [("SUBJECT", "he", 0), ("OBJECT", "cat", 2)])]
    kept_t, kept_l = gold_v2.prune_forest(trees, [[], []], [0.9, 0.85], "top1", 0.02)
    assert kept_t == [trees[0]]
    assert len(kept_l) == 1


def test_prune_forest_margin_drops_exact_ties_as_spurious():
    trees = [_mk_tree("saw", [("SUBJECT", "he", 0)]), _mk_tree("went", [("SUBJECT", "he", 0)])]
    kept_t, _ = gold_v2.prune_forest(trees, [[], []], [0.9, 0.9], "margin", 0.02)
    assert len(kept_t) == 1


def test_prune_forest_margin_drops_modifier_attachment_wobble():
    t0 = _mk_tree("saw", [("SUBJECT", "he", 0)])
    t1 = _mk_tree("saw", [("SUBJECT", "he", 0), ("DESCRIPTION", "big", 3)])
    kept_t, _ = gold_v2.prune_forest([t0, t1], [[], []], [0.9, 0.89], "margin", 0.02)
    assert len(kept_t) == 1, "an extra DESCRIPTION-only role is wobble, not a genuine 2nd reading"


def test_prune_forest_margin_keeps_genuinely_distinct_close_tree():
    t0 = _mk_tree("saw", [("SUBJECT", "he", 0), ("OBJECT", "cat", 2)])
    t1 = _mk_tree("saw", [("SUBJECT", "he", 0), ("PLACE", "cat", 2)])
    kept_t, _ = gold_v2.prune_forest([t0, t1], [[], []], [0.9, 0.89], "margin", 0.02)
    assert len(kept_t) == 2, "a different CORE role on the same content node is a genuine 2nd reading"


def test_prune_forest_margin_drops_trees_far_below_top():
    t0 = _mk_tree("saw", [("SUBJECT", "he", 0)])
    t1 = _mk_tree("went", [("SUBJECT", "he", 0)])
    kept_t, _ = gold_v2.prune_forest([t0, t1], [[], []], [0.9, 0.5], "margin", 0.02)
    assert len(kept_t) == 1


def test_prune_forest_all_mode_disables_pruning():
    t0 = _mk_tree("saw", [("SUBJECT", "he", 0)])
    t1 = _mk_tree("went", [("SUBJECT", "he", 0)])
    kept_t, _ = gold_v2.prune_forest([t0, t1], [[], []], [0.9, 0.1], "all", 0.02)
    assert len(kept_t) == 2


def test_forest_mode_default_is_top1():
    assert gold_v2.FOREST_MODE_DEFAULT == "top1"
    assert gold_v2.FOREST_MARGIN_DEFAULT == pytest.approx(0.02)


@pytest.mark.skipif(not (_ROOT / "data" / "corpus").exists(), reason="needs data/corpus")
def test_top1_mode_yields_one_tree_per_sentence_over_a_corpus_sample(usvs):
    """The task's own unit-test requirement: building over ~20 corpus
    sentences yields trees-per-sentence == 1 for every emitted record in
    the (default) top1 mode."""
    sentences = gold_v2.load_corpus()[:20]
    tok = gold_v2.SimpleTokenizer.build(sentences, extra_tokens=list(gold_v2.PRIME_NAMES) + gold_v2.PARSE_LABELS)
    parser = gold_v2.ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        pytest.skip("quantum_parser unavailable")

    n_ok = 0
    for sentence in sentences:
        record, outcome, _n_raw = gold_v2.build_record(usvs, parser, sentence)  # defaults: forest_mode="top1"
        if outcome != "ok":
            continue
        n_ok += 1
        assert len(record["lattice"]["trees"]) == 1, f"{sentence!r}: expected 1 tree in top1 mode"
    assert n_ok > 0, "expected at least one parseable sentence in the 20-sentence sample"
