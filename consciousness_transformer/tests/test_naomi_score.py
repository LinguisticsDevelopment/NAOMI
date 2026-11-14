"""Fast tests for naomi.score (coverage scorer + reference scorer)."""
from naomi.score import score_against, score_tree
from naomi.types import ClauseTree, Filler


def F(role, tok, i):
    return Filler(role=role, token=tok, lemma=tok, token_index=i)


def C(pred, args):
    return ClauseTree(predicate=F("PREDICATE", *pred), args=[F(r, t, i) for r, t, i in args])


TOK = ["the", "dog", "did", "not", "eat", "the", "big", "bone", "."]
POS = ["DET", "NOUN", "AUX", "PART", "VERB", "DET", "ADJ", "NOUN", "PUNCT"]
GOOD = [C(("did", 2), [("SUBJECT", "dog", 1)]), C(("eat", 4), [("SUBJECT", "dog", 1), ("OBJECT", "bone", 7),
                                                            ("DESCRIPTION", "big", 6)])]


def kinds(s):
    return [d["kind"] for d in s.diagnostics]


def test_full_coverage_reports_dropped_particle():
    s = score_tree(TOK, POS, GOOD)
    assert s.percent == 100.0
    assert {"kind": "token_dropped", "token": "not", "index": 3, "pos": "PART"} in s.diagnostics
    assert "verb_without_clause" not in kinds(s)


def test_missing_verb_and_noun():
    s = score_tree(TOK, POS, [C(("did", 2), [("SUBJECT", "dog", 1)])])
    assert s.components["verb"] == 0.5 and s.components["noun"] == 0.5
    assert {"kind": "verb_without_clause", "token": "eat", "index": 4, "pos": "VERB"} in s.diagnostics
    assert "noun_not_in_tree" in kinds(s) and "modifier_not_in_tree" in kinds(s)
    assert 0 < s.percent < 100


def test_structural_problems():
    bad = [C(("dog", 1), [("OBJECT", "bone", 7), ("SUBJECT", "bone", 7)]), C(("eat", 99), [])]
    s = score_tree(TOK, POS, bad)
    k = kinds(s)
    assert "predicate_not_verb_like" in k and "duplicate_token_index" in k
    assert "phantom_token_index" in k and "clause_without_subject" in k
    assert s.components["structure"] == 0.0


def test_empty_tree_scores_zero():
    s = score_tree(TOK, POS, [])
    assert s.percent == 0.0 and "no_clause" in kinds(s)


def test_score_against_identical_and_diffs():
    assert score_against(GOOD, GOOD).percent == 100.0
    cand = [C(("eat", 4), [("OBJECT", "bone", 7), ("OBJECT", "big", 6), ("PLACE", "the", 5)])]
    r = score_against(GOOD, cand)
    k = kinds(r)
    assert "clause_dropped" in k and "role_wrong" in k and "filler_missing" in k and "filler_extra" in k
    assert r.predicate_recall == 0.5 and 0 < r.percent < 100
    r2 = score_against(GOOD, GOOD + [C(("see", 3), [])])
    assert "clause_extra" in kinds(r2)
    assert r.graded is None
