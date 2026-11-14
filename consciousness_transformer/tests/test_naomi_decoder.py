"""Fast tests for the grammar-backwards decoder prototype (no training, no corpus parse)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from naomi import decoder as D  # noqa: E402

TREE = {"clauses": [{"predicate": "chased", "roles": [
    {"relation": "OBJECT", "word": "cat", "token_index": 6},
    {"relation": "DESCRIPTION", "word": "old", "token_index": 1},
    {"relation": "SUBJECT", "word": "dog", "token_index": 2}]}]}
POS = {"chased": "VERB", "dog": "NOUN", "cat": "NOUN", "old": "ADJ"}


@pytest.fixture(scope="module")
def table():
    return D.derive_order_table(D.load_grammar())


def test_order_table_is_read_from_grammar(table):
    def one(rel, head):
        return {(e.side, e.idx) for e in table.entries if e.rel == rel and e.head == head and not e.extra}
    assert one("SUBJECT", "PREDICATE") == {("before", 0)}
    assert ("after", 0) in one("OBJECT", "VERBAL") and ("after", 1) in one("OBJECT", "VERBAL")
    assert one("DESCRIPTION", "NOUN") == {("before", -1)}
    assert table.serves("NOUN", "NOMINAL") and table.serves("VERBAL", "PREDICATE")
    # PP rules need a PREP anchor: a function-word slot the tree does not carry
    assert any(e.rel == "PREPOSITION" and e.head == "PREP" for e in table.entries)


def test_spanish_grammar_has_same_format():
    es = D.derive_order_table(D.load_grammar(D.REPO / "quantum_parser" / "grammars" / "spanish.json"))
    assert any(e.rel == "SUBJECT" and e.side == "before" for e in es.entries)


def test_candidates_use_only_tree_tokens_and_respect_derived_order(table):
    cands = D.linearize(TREE, table, POS)
    assert cands
    for c in cands:
        assert sorted(c) == sorted(["chased", "cat", "old", "dog"])      # nothing invented, nothing lost
        assert c.index("dog") < c.index("chased")                        # SUBJECT precedes predicate
        assert c.index("chased") < c.index("cat")                        # OBJECT follows predicate
    assert ["old", "dog", "chased", "cat"] in cands


def test_candidates_ignore_token_index_and_role_order(table):
    a = D.linearize(D.strip_positions(TREE, seed=1), table, POS)
    b = D.linearize(D.strip_positions(TREE, seed=2), table, POS)
    assert {tuple(x) for x in a} == {tuple(x) for x in b}
    assert all("token_index" in r for r in TREE["clauses"][0]["roles"])
    assert all("token_index" not in r for r in D.strip_positions(TREE)["clauses"][0]["roles"])


def test_pp_slot_label_is_only_source_of_preposition(table):
    assert D.default_slot_fill("PREP", "WITH", "x") == "with"     # open label == the preposition token
    assert D.default_slot_fill("PREP", "PLACE", "x") is None      # collapsed label: word is gone
    tree = {"clauses": [{"predicate": "sat", "roles": [{"relation": "SUBJECT", "word": "dog"},
                                                       {"relation": "WITH", "word": "cat"}]}]}
    cands = D.linearize(tree, table, {"sat": "VERB", "dog": "NOUN", "cat": "NOUN"})
    assert ["dog", "sat", "with", "cat"] in cands


def test_edge_f1_and_bleu():
    g = D.tree_edges(TREE)
    assert D.edge_f1(g, g)[2] == 1.0
    assert D.edge_f1(g, D.tree_edges({"clauses": [{"predicate": "chased", "roles": []}]}))[2] < 1.0
    assert D.bleu("a b c d".split(), "a b c d".split()) > 0.99
    assert D.bleu("a b c d".split(), "x y".split()) < 0.3


def test_decode_selects_by_reparse(table):
    judge = D.Judge()
    tree = {"clauses": [{"predicate": "chase", "roles": [
        {"relation": "OBJECT", "word": "cat"}, {"relation": "SUBJECT", "word": "dog"}]}]}
    res = D.decode(tree, table, judge, {"chase": "VERB", "dog": "NOUN", "cat": "NOUN"})
    assert res.parsed and res.f1 == 1.0 and res.tokens == ["dog", "chase", "cat"]
