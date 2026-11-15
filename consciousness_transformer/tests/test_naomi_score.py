"""Fast tests for naomi.score on HAND-BUILT trees (parser output is never used here)."""
import pytest

from naomi.reader import make_node
from naomi.score import score_coverage, score_pair, tree_similarity
from naomi.types import REL_ROOT, REL_UNATTACHED, Tree

pytest.importorskip("nltk")


@pytest.fixture(scope="module")
def usvs():
    from nsm_ct.ground.usvs import load_usvs
    from naomi.reader import CT_DIR
    return load_usvs(str(CT_DIR / "data" / "usvs"))


def N(rel, tok, idx, pos, *kids):
    n = make_node(rel, tok, idx, pos)
    n.children = list(kids)
    return n


def T(root):
    return Tree(sentence="", root=root)


def she_was(adj_tok, *adj_kids, adj_pos="ADJ"):
    return T(N(REL_ROOT, "was", 1, "AUX", N("SUBJECT", "she", 0, "PRON"),
               N("COMPLEMENT", adj_tok, 3, adj_pos, *adj_kids)))


def dims(reason):
    d = reason["dimensions"]
    return set(d) if isinstance(d, dict) else {x["dimension"] for x in d}


def test_identical_is_100(usvs):
    t = she_was("happy", N("SPECIFICATION", "very", 2, "ADV"))
    ps = score_pair(t, she_was("happy", N("SPECIFICATION", "very", 2, "ADV")), usvs)
    assert ps.overall == pytest.approx(100.0) and ps.reasons == []


def test_reordered_children_100(usvs):
    a = T(N(REL_ROOT, "ate", 2, "VERB", N("SUBJECT", "dog", 1, "NOUN"), N("OBJECT", "bone", 4, "NOUN")))
    b = T(N(REL_ROOT, "ate", 0, "VERB", N("OBJECT", "bone", 1, "NOUN"), N("SUBJECT", "dog", 5, "NOUN")))
    assert score_pair(a, b, usvs).overall == pytest.approx(100.0)


def test_dropped_negator_is_folded(usvs):
    ref = she_was("happy", N("SPECIFICATION", "not", 2, "PART"))
    ps = score_pair(ref, she_was("happy"), usvs)
    assert ps.shape == pytest.approx(100.0) and ps.meaning < 100.0
    r = [x for x in ps.reasons if x["kind"] == "operator_folded_differs"]
    assert len(r) == 1 and "NOT" in dims(r[0])
    d = {x["dimension"]: x["difference"] for x in r[0]["dimensions"]}
    assert d["NOT"] == pytest.approx(-1.0)
    assert not any(x["kind"] in ("node_dropped", "node_extra") for x in ps.reasons)


def test_not_happy_vs_unhappy_shape_100(usvs):
    ps = score_pair(she_was("happy", N("SPECIFICATION", "not", 2, "PART")), she_was("unhappy"), usvs)
    assert ps.shape == pytest.approx(100.0) and 0 < ps.meaning < 100.0


def test_identical_both_100(usvs):
    ps = score_pair(she_was("happy", N("SPECIFICATION", "very", 2, "ADV")), she_was("happy", N("SPECIFICATION", "very", 2, "ADV")), usvs)
    assert ps.meaning == pytest.approx(100.0) and ps.shape == pytest.approx(100.0)


def test_reordered_both_100(usvs):
    a = T(N(REL_ROOT, "saw", 1, "VERB", N("SUBJECT", "he", 0, "PRON"), N("DESCRIPTION", "big", 2, "ADJ"), N("DESCRIPTION", "old", 3, "ADJ")))
    b = T(N(REL_ROOT, "saw", 1, "VERB", N("DESCRIPTION", "old", 3, "ADJ"), N("DESCRIPTION", "big", 2, "ADJ"), N("SUBJECT", "he", 0, "PRON")))
    ps = score_pair(a, b, usvs)
    assert ps.meaning == pytest.approx(100.0) and ps.shape == pytest.approx(100.0)


def test_folding_equals_separate_slots(usvs):
    """Folding adds the operator vector to the head's; with disjoint dimensions that equals keeping it as
    a separate component (dot products identical)."""
    import numpy as np
    from naomi.reader import make_node
    def v(tok, pos):
        return usvs.sense_dense(make_node("X", tok, 0, pos).senses[0]).astype(float)
    for op, opos in (("not", "PART"), ("very", "ADV"), ("slightly", "ADV")):
        for head, hpos in (("happy", "ADJ"), ("unhappy", "ADJ"), ("sad", "ADJ")):
            o, h = v(op, opos), v(head, hpos)
            assert float(np.dot(o, h)) == 0.0                      # disjoint dimensions
            for op2 in ("not", "very"):
                o2 = v(op2, "PART" if op2 == "not" else "ADV")
                folded = float(np.dot(h + o, h + o2))
                separate = float(np.dot(h, h)) + float(np.dot(o, o2))   # slots: head part + operator part
                assert folded == pytest.approx(separate)


def test_antonym_sense_level(usvs):
    from naomi.reader import make_node
    from naomi.score import sense_opposite, DEFAULT_ANTONYM_TIERS
    first = lambda w: make_node("X", w, 0, "ADJ").senses[0]
    expect = {("hot", "cold"): "direct", ("happy", "unhappy"): "direct", ("wet", "dry"): "direct",
              ("damp", "dry"): "satellite_head", ("happy", "happy"): None}
    for (a, b), tier in expect.items():
        assert sense_opposite(first(a), first(b), DEFAULT_ANTONYM_TIERS) == tier
        ps = score_pair(she_was(a), she_was(b), usvs)
        fired = [x for x in ps.reasons if x["kind"] == "opposite_meaning"]
        assert [x["tier"] for x in fired] == ([tier] if tier else [])
        if fired:
            assert ps.meaning < 100.0 and not any(x["kind"] == "meaning_shift" for x in ps.reasons)
    # tiers are a parameter: no tiers -> nothing fires; the word store is an optional extra (default off)
    off = score_pair(she_was("hot"), she_was("cold"), usvs, antonym_tiers=())
    assert not [x for x in off.reasons if x["kind"] == "opposite_meaning"]
    on = score_pair(she_was("hot"), she_was("cold"), usvs, antonym_tiers=(), word_store_tiers=DEFAULT_ANTONYM_TIERS)
    assert [x for x in on.reasons if x["kind"] == "opposite_meaning"]


def test_idf_weights_and_effect(usvs):
    from naomi.score import axis_weights
    w = dict(zip(usvs.axes, axis_weights(usvs)))
    assert w["SOMETHING"] < w["NOT"] and w["SOMETHING"] < w["GOAL"] and w["SOMETHING"] < w["KNOWN"]
    # negation moves the meaning number more with weighting than without
    a, b = she_was("happy"), she_was("happy", N("SPECIFICATION", "not", 2, "PART"))
    assert score_pair(a, b, usvs).meaning < score_pair(a, b, usvs, idf=False).meaning
    # swapped arguments are penalised more than a mere synonym-ish change
    d = lambda s, o: T(N(REL_ROOT, "ate", 2, "VERB", N("SUBJECT", s, 1, "NOUN"), N("OBJECT", o, 4, "NOUN")))
    assert score_pair(d("dog", "bone"), d("bone", "dog"), usvs).meaning < 70.0


def test_moved_the_meaning_high_shape_lower(usvs):
    a = T(N(REL_ROOT, "ate", 2, "VERB", N("SUBJECT", "dog", 1, "NOUN", N("DESCRIPTION", "the", 0, "DET")), N("OBJECT", "bone", 4, "NOUN")))
    b = T(N(REL_ROOT, "ate", 2, "VERB", N("SUBJECT", "dog", 1, "NOUN"), N("OBJECT", "bone", 4, "NOUN", N("DESCRIPTION", "the", 3, "DET"))))
    ps = score_pair(a, b, usvs)
    # folded "the" differs per head: a head-level operator difference, shape stays 100 (nodes are the same)
    assert ps.shape == pytest.approx(100.0) and ps.meaning < 100.0
    assert sum(1 for x in ps.reasons if x["kind"] == "operator_folded_differs") == 2
    # a moved non-folded subtree: shape drops, meaning stays high relative to the swapped-argument tree
    c = T(N(REL_ROOT, "saw", 1, "VERB", N("SUBJECT", "he", 0, "PRON"),
            N("OBJECT", "dog", 3, "NOUN", N("MODIFICATION", "near", 4, "ADP", N("PREPOSITION", "house", 5, "NOUN")))))
    d = T(N(REL_ROOT, "saw", 1, "VERB", N("SUBJECT", "he", 0, "PRON"), N("OBJECT", "dog", 3, "NOUN"),
            N("MODIFICATION", "near", 4, "ADP", N("PREPOSITION", "house", 5, "NOUN"))))
    pm = score_pair(c, d, usvs)
    assert pm.shape < 100.0 and pm.meaning < 100.0


def test_dropping_prep_phrase_lowers_both(usvs):
    full = T(N(REL_ROOT, "went", 1, "VERB", N("SUBJECT", "he", 0, "PRON"),
               N("MODIFICATION", "to", 2, "ADP", N("PREPOSITION", "house", 4, "NOUN"))))
    short = T(N(REL_ROOT, "went", 1, "VERB", N("SUBJECT", "he", 0, "PRON")))
    ps = score_pair(full, short, usvs)
    assert ps.meaning < 100.0 and ps.shape < 100.0
    assert [x["token"] for x in ps.reasons if x["kind"] == "node_dropped"] == ["to", "house"]


def test_to_from_shift(usvs):
    def go(prep):
        return T(N(REL_ROOT, "went", 1, "VERB", N("SUBJECT", "he", 0, "PRON"),
                   N("MODIFICATION", prep, 2, "ADP", N("PREPOSITION", "house", 4, "NOUN", N("DESCRIPTION", "the", 3, "DET")))))
    ps = score_pair(go("to"), go("from"), usvs)
    sh = [x for x in ps.reasons if x["kind"] == "meaning_shift"]
    assert len(sh) == 1 and {"GOAL", "SOURCE"} <= dims(sh[0])
    assert ps.shape == pytest.approx(100.0) and ps.meaning < 100.0
    assert ps.function_words.node_recall < 100.0 and ps.content_words.overall == pytest.approx(100.0)


def test_synonym_beats_unrelated(usvs):
    def sent(noun):
        return T(N(REL_ROOT, "saw", 1, "VERB", N("SUBJECT", "he", 0, "PRON"), N("OBJECT", noun, 2, "NOUN")))
    ref = sent("car")
    syn = score_pair(ref, sent("automobile"), usvs).overall
    unrelated = score_pair(ref, sent("idea"), usvs).overall
    assert syn > unrelated


def test_name_mismatch_and_relation_change(usvs):
    def sent(name, rel):
        return T(N(REL_ROOT, "saw", 1, "VERB", N("SUBJECT", name, 0, "PROPN"), N(rel, "dog", 2, "NOUN")))
    ps = score_pair(sent("jim", "OBJECT"), sent("tom", "OBJECT"), usvs)
    assert any(x["kind"] == "name_mismatch" for x in ps.reasons)
    ps = score_pair(sent("jim", "OBJECT"), sent("jim", "MODIFICATION"), usvs)
    rc = [x for x in ps.reasons if x["kind"] == "relation_changed"]
    assert len(rc) == 1 and rc[0]["token"] == "dog"


def test_subtree_moved(usvs):
    a = T(N(REL_ROOT, "saw", 1, "VERB", N("SUBJECT", "he", 0, "PRON"),
            N("OBJECT", "dog", 3, "NOUN", N("MODIFICATION", "near", 4, "ADP", N("PREPOSITION", "house", 5, "NOUN")))))
    b = T(N(REL_ROOT, "saw", 1, "VERB", N("SUBJECT", "he", 0, "PRON"), N("OBJECT", "dog", 3, "NOUN"),
            N("MODIFICATION", "near", 4, "ADP", N("PREPOSITION", "house", 5, "NOUN"))))
    ps = score_pair(a, b, usvs)
    assert [x["token"] for x in ps.reasons if x["kind"] == "subtree_moved"] == ["near"]


def test_symmetric_similarity(usvs):
    a, b = she_was("happy"), she_was("happy", N("SPECIFICATION", "not", 2, "PART"))
    assert tree_similarity(a, b, usvs) == pytest.approx(tree_similarity(b, a, usvs))
    assert tree_similarity(a, b, usvs) < 100.0


def test_coverage(usvs):
    t = she_was("happy")
    t.root.children.append(N(REL_UNATTACHED, "slowly", 4, "ADV", N("SPECIFICATION", "very", 5, "ADV")))
    t.root.children.append(N("X", "zzzqq", 6, "NOUN"))
    c = score_coverage(t)
    assert c.attached_by_pos["ADV"] == (0, 2)
    assert c.attached == pytest.approx(4 / 6)
    assert c.structure_ok
    assert any(d["kind"] == "ungrounded_word" and d["token"] == "zzzqq" for d in c.diagnostics)
    assert c.grounded < 1.0
    full = score_coverage(she_was("happy"))
    assert full.percent == pytest.approx(100.0) and c.percent < full.percent
