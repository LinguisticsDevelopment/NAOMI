"""Fast tests for the tree contract and the language-neutral graph attachment (no parser needed)."""
from naomi import tree as T
from naomi.types import Node, Tree


def mk_factory(tokens):
    return lambda i: Node(rel="UNATTACHED", token=tokens[i], token_index=i, lemma=tokens[i], pos="X")


def build(tokens, edges, roots, pos=None):
    pos = pos or ["X"] * len(tokens)
    root, diags = T.attach_graph(tokens, pos, edges, roots, mk_factory(tokens), lambda i: pos[i] == "PUNCT")
    return Tree("s", root, "parser", diags)


def test_forward_reversed_and_orphans():
    # a=0 b=1 c=2 d=3 ; b -> a (SUBJECT); c -> b reversed (c is the modifier pointing at its head); d no edge
    t = build(list("abcd"), [("SUBJECT", 1, 0), ("SPECIFICATION", 2, 1)], [1, 2, 3])
    assert T.compact(t) == "b(SUBJECT:a SPECIFICATION:c UNATTACHED:d)"
    kinds = [d["kind"] for d in t.diagnostics]
    assert kinds.count("reversed_edge") == 1 and kinds.count("unattached_token") == 1


def test_second_head_keeps_first_edge_and_records():
    t = build(list("abc"), [("OBJECT", 0, 2), ("SUBJECT", 1, 2), ("MODIFICATION", 0, 1)], [0])
    assert T.compact(t) == "a(MODIFICATION:b OBJECT:c)"
    assert [d["kind"] for d in t.diagnostics] == ["extra_edge"]


def test_several_components_get_synthetic_root_in_token_order():
    t = build(list("abcd"), [("SUBJECT", 0, 1), ("OBJECT", 2, 3)], [0, 2])
    assert t.root.token_index is None
    assert [c.token for c in t.root.children] == ["a", "c"]
    assert [c.rel for c in t.root.children] == ["ROOT", "UNATTACHED"]
    assert any(d["kind"] == "multiple_components" for d in t.diagnostics)


def test_punctuation_is_not_a_node_and_all_others_are():
    toks, pos = ["a", "b", "."], ["X", "X", "PUNCT"]
    t = build(toks, [("SUBJECT", 1, 0), ("MODIFICATION", 1, 2)], [1], pos)
    assert [n.token for n in T.token_nodes(t)] == ["a", "b"]


def test_json_round_trip_and_renderings():
    t = build(list("abc"), [("SUBJECT", 1, 0)], [1])
    t.root.children[0].senses = ["x.n.01"]
    assert T.tree_from_json(T.tree_to_json(t)) == t
    lines = T.render(t).splitlines()
    assert len(lines) == 3 and lines[0].startswith("ROOT  b#1  b  X  word") and lines[1].startswith("  ")
    assert "\n" not in T.compact(t)
