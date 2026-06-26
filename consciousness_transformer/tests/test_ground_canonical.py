"""M17.0 gate: deterministic canonical form for meaning trees."""

from __future__ import annotations

from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.ground.canonicalization import canon_label, normalize, tree_key


def _t(root: ParseNode) -> ParseTree:
    return ParseTree(root=root)


def test_normalize_is_idempotent():
    tree = _t(ParseNode(label="EXPLICATION", children=[
        ParseNode(label="BAD"),
        ParseNode(label="FEEL"),
        ParseNode(label="BAD"),  # duplicate
    ]))
    once = normalize(tree)
    twice = normalize(once)
    assert tree_key(once) == tree_key(twice)


def test_allolex_folds_to_canonical_prime():
    # "me" is an allolex of the prime I; "thing" of SOMETHING.
    assert canon_label("me") == "I"
    a = _t(ParseNode(label="me"))
    b = _t(ParseNode(label="I"))
    assert tree_key(a) == tree_key(b)


def test_sibling_order_is_canonical():
    a = _t(ParseNode(label="EXPLICATION", children=[
        ParseNode(label="FEEL"), ParseNode(label="BAD")]))
    b = _t(ParseNode(label="EXPLICATION", children=[
        ParseNode(label="BAD"), ParseNode(label="FEEL")]))
    assert tree_key(a) == tree_key(b)


def test_duplicate_children_dedup():
    a = _t(ParseNode(label="EXPLICATION", children=[
        ParseNode(label="GOOD"), ParseNode(label="GOOD")]))
    b = _t(ParseNode(label="EXPLICATION", children=[ParseNode(label="GOOD")]))
    assert tree_key(a) == tree_key(b)


def test_token_is_ignored_for_equality():
    a = _t(ParseNode(label="GOOD", token="good"))
    b = _t(ParseNode(label="GOOD", token="fine"))
    assert tree_key(a) == tree_key(b)


def test_empty_wrapper_is_pruned():
    # An EXPLICATION wrapper with no children carries no meaning -> UNRESOLVED.
    tree = _t(ParseNode(label="EXPLICATION"))
    assert normalize(tree).root.label == "UNRESOLVED"
