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


def test_dropped_negator(usvs):
    ref = she_was("happy", N("SPECIFICATION", "not", 2, "PART"))
    cand = she_was("happy")
    ps = score_pair(ref, cand, usvs)
    r = [x for x in ps.reasons if x["kind"] == "node_dropped"]
    assert len(r) == 1 and r[0]["function_node_dropped"] and "NOT" in dims(r[0])
    assert ps.function_words.overall < 100.0
    assert ps.content_words.overall == pytest.approx(100.0)
    assert ps.function_words.node_recall < 100.0


def test_to_from_shift(usvs):
    def go(prep):
        return T(N(REL_ROOT, "went", 1, "VERB", N("SUBJECT", "he", 0, "PRON"),
                   N("MODIFICATION", prep, 2, "ADP", N("PREPOSITION", "house", 4, "NOUN", N("DESCRIPTION", "the", 3, "DET")))))
    ps = score_pair(go("to"), go("from"), usvs)
    sh = [x for x in ps.reasons if x["kind"] == "meaning_shift"]
    assert len(sh) == 1 and {"GOAL", "SOURCE"} <= dims(sh[0])
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
