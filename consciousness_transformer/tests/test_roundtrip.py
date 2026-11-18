"""Fast tests for the reformulated round-trip check (naomi.roundtrip): order discord, source order, substitution."""
import pytest

from naomi import roundtrip as rt
from naomi.types import REL_ROOT, Node, Tree


def N(tok, idx, rel="x", kids=()):
    return Node(rel=rel, token=tok, token_index=idx, lemma=tok, pos="p", children=list(kids))


def tree():
    # "the old door" cut from a sentence: door(5) <- the(3), old(4)
    return Tree(sentence="", root=N("door", 5, REL_ROOT, [N("the", 3), N("old", 4)]))


def test_source_order():
    t = tree()
    assert rt.source_text(t) == "the old door"
    assert [n.token_index for n in rt.source_nodes(t)] == [3, 4, 5]


def test_order_discord_true_and_scrambled():
    t = tree()
    src = rt.source_nodes(t)
    assert rt.order_discord(t, src) == 0.0
    # door the old: door-the and door-old inverted (parent after children in the source), the-old kept
    scr = [src[2], src[0], src[1]]
    assert rt.order_discord(t, scr) == pytest.approx(2 / 3)
    rev = list(reversed(src))
    assert rt.order_discord(t, rev) == pytest.approx(1.0)


def test_order_discord_single_word_is_none():
    t = Tree(sentence="", root=N("farmer", 13, REL_ROOT))
    assert rt.order_discord(t, rt.source_nodes(t)) is None


def test_substitute_and_passes():
    assert rt.substitute("a b c d e", [1, 3], ["y", "x"]) == "a y c x e"
    assert rt.passes(95.0, 90) and not rt.passes(80.0, 90) and not rt.passes(None, 0)
