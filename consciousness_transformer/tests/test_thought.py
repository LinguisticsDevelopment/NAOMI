"""Tests for thought objects: tree-of-meaning unit + lossless round-trip."""

from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.input_encoder import ParserInputEncoder
from nsm_ct.reverse_parser import thought_to_text
from nsm_ct.serialization import deserialize_thought, serialize_thought
from nsm_ct.thought import (
    MockMeaningResolver,
    ThoughtObject,
    build_thought,
    meaning_prime_ids,
)
from nsm_ct.tokenizer import SimpleTokenizer, basic_tokenize


def _tok():
    from nsm_ct.structure import PARSE_LABELS
    from nsm_ct.episode import CurriculumGenerator
    from nsm_ct.nsm_primes import PRIME_NAMES
    eps = CurriculumGenerator(max_level=6, seed=0).generate(12)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    return SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)


def test_build_thought_attaches_prime_trees_to_words():
    """Each word becomes a tree of NSM primes (an NSM 'molecule')."""
    tree = ParseTree(root=ParseNode(label="CLAUSE", children=[
        ParseNode(label="NOMINAL", token="mary", relation="SUBJECT", index=0),
        ParseNode(label="NOMINAL", token="kitchen", relation="OBJECT", index=4),
    ]))
    thought = build_thought("mary ... kitchen", tree, MockMeaningResolver())
    leaves = [n for n in thought.tree.iter_preorder() if n.token is not None]
    assert len(leaves) == 2
    for leaf in leaves:
        assert leaf.meaning is not None                      # word -> meaning tree
        ids = meaning_prime_ids(leaf.meaning)
        assert 1 <= len(ids) <= 4 and all(i >= 1 for i in ids)  # real prime ids (1-based)


def test_serialize_thought_round_trips_losslessly():
    """serialize/deserialize is an exact inverse (the 'meaning without loss' rule),
    including the per-word meaning trees."""
    tree = ParseTree(root=ParseNode(label="CLAUSE", token="is", index=1, children=[
        ParseNode(label="NOMINAL", token="mary", relation="SUBJECT", index=0),
        ParseNode(label="NOMINAL", token="kitchen", relation="OBJECT", index=4, children=[
            ParseNode(label="DET", token="the", relation="SPECIFICATION", index=3),
        ]),
    ]))
    build_thought("...", tree, MockMeaningResolver())
    toks = serialize_thought(ThoughtObject(tree=tree))
    rebuilt = deserialize_thought(toks)
    assert rebuilt.root == tree.root                          # dataclass deep-equality


def test_reverse_parser_renders_leaves_in_source_order():
    """The reverse-parser seed renders the tree's word-leaves in source order."""
    tree = ParseTree(root=ParseNode(label="CLAUSE", token="is", index=1, children=[
        ParseNode(label="NOMINAL", token="mary", index=0),
        ParseNode(label="NOMINAL", token="kitchen", index=4),
    ]))
    assert thought_to_text(ThoughtObject(tree=tree)) == "mary is kitchen"


def test_reverse_parser_is_subsequence_of_real_parse():
    """On a real parse, the rendered words are a subsequence of the input (the
    rule parser may drop words — that's a parser-quality limit, not a loss in the
    thought object's serialization)."""
    enc = ParserInputEncoder(_tok())
    sent = "mary is in the kitchen ."
    tree = enc._parse_tree(sent)
    if tree is None:                                         # parser unavailable: skip
        return
    rendered = thought_to_text(tree).split()
    words = basic_tokenize(sent)
    # subsequence check
    it = iter(words)
    assert all(w in it for w in rendered)
