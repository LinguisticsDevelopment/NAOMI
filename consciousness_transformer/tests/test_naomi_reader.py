"""Fast tests for naomi.reader: the parser -> Tree adapter, the eight proposal sentences, fallback, cache.

The eight-sentence assertions are a SUBSET check of what the parser genuinely produces today (edges
that must be present); they do not assert the ideal tree where the grammar cannot yet deliver it, so
grammar improvements do not break them. Missing attachments show up as UNATTACHED nodes instead.
"""
import pytest

from naomi import tree as T
from naomi.mind import Mind
from naomi.reader import Reader, normalize
from naomi.types import REL_UNATTACHED


@pytest.fixture(scope="module")
def reader():
    return Reader(use_encoder=False)


def edges(tree):
    """{(head token, relation, dep token)} over all real nodes."""
    out = set()
    for n, _d, parent in T.walk(tree.root):
        if parent is not None and parent.token_index is not None:
            out.add((parent.token, n.rel, n.token))
    return out


EIGHT = {
    "she was slightly unhappy": {("was", "SUBJECT", "she"), ("was", "COMPLEMENT", "unhappy"),
                                 ("unhappy", "SPECIFICATION", "slightly")},
    "he was not happy": {("was", "SUBJECT", "he")},
    "nobody came to the house": {("came", "SUBJECT", "nobody"), ("came", "MODIFICATION", "to"),
                                 ("to", "PREPOSITION", "house"), ("house", "DESCRIPTION", "the")},
    "the dog did not eat the bone": {("eat", "SUBJECT", "dog"), ("dog", "DESCRIPTION", "the"),
                                     ("eat", "OBJECT", "bone"), ("bone", "DESCRIPTION", "the")},
    "jim's father gave her three apples": {("gave", "SUBJECT", "father"), ("gave", "OBJECT", "apples")},
    "she said that he left because it rained": {("said", "SUBJECT", "she"), ("left", "SUBJECT", "he"),
                                                ("rained", "SUBJECT", "it"), ("said", "SUBORDINATION", "that"),
                                                ("left", "SUBORDINATION", "because")},
    "who ate the bone?": {("ate", "SUBJECT", "who"), ("ate", "OBJECT", "bone"), ("bone", "DESCRIPTION", "the")},
    "the king never saw the princess again": {("saw", "SUBJECT", "king"), ("king", "DESCRIPTION", "the"),
                                              ("saw", "SPECIFICATION", "never"), ("saw", "OBJECT", "princess"),
                                              ("princess", "DESCRIPTION", "the")},
}


def test_normalize_is_corpus_tokenisation():
    assert normalize("The dog ate.") == "the dog ate ."
    assert normalize("She couldn't go") == "she could n't go"


@pytest.mark.parametrize("sentence", list(EIGHT))
def test_eight_sentences_parser_edges_and_every_token_a_node(reader, sentence):
    rr = reader.read(sentence)
    assert rr.source == "parser" and rr.tree is not None
    assert EIGHT[sentence] <= edges(rr.tree)
    # invariant: every non-punctuation token is exactly one node
    toks, pos = reader.tag(normalize(sentence))
    want = [i for i, p in enumerate(pos) if p != "PUNCT"]
    assert [n.token_index for n in T.token_nodes(rr.tree)] == want
    assert [n.token for n in T.token_nodes(rr.tree)] == [toks[i] for i in want]
    # JSON lossless
    assert T.tree_from_json(T.tree_to_json(rr.tree)) == rr.tree
    assert T.render(rr.tree).count("\n") + 1 == len(list(T.walk(rr.tree.root)))


def test_nesting_and_prepositions_as_nodes(reader):
    t = reader.read("she said that he left because it rained").tree
    assert T.max_depth(t) >= 3                     # clauses nest under 'that' / 'because'
    t = reader.read("nobody came to the house").tree
    to = next(n for n, _, _ in T.walk(t.root) if n.token == "to")
    assert to.pos == "ADP" and [c.token for c in to.children] == ["house"]


def test_unattached_is_recorded_not_repaired(reader):
    t = reader.read("he was not happy").tree
    un = {n.token for n in T.token_nodes(t) if n.rel == REL_UNATTACHED}
    diag = {d["token"] for d in t.diagnostics if d["kind"] == "unattached_token"}
    assert un == diag                              # every UNATTACHED node has its diagnostic
    t = reader.read("the dog did not eat the bone").tree   # the grammar now links auxiliary and main verb
    assert not any(d["kind"] == "multiple_components" for d in t.diagnostics)
    assert t.root.token == "eat"


def test_names_and_senses(reader):
    t = reader.read("jim ate the bone .").tree
    jim = next(n for n in T.token_nodes(t) if n.token == "jim")
    assert jim.kind == "name" and jim.senses == [] and jim.sense is None
    bone = next(n for n in T.token_nodes(t) if n.token == "bone")
    assert bone.kind == "word" and bone.senses[0].startswith("bone.n.") and len(bone.senses) <= 5
    assert bone.sense is None
    eat = next(n for n in T.token_nodes(t) if n.lemma == "eat")
    assert eat.senses


def test_parser_failure_falls_back_to_legacy_encoder():
    r = Reader(parser_seconds=0.0001)
    s = "james did n't know what to do and was scared ."
    assert r.read_parser(s) is None
    rr = r.read(s)
    assert rr.source == "encoder_v6_legacy" and rr.tree.source == "encoder_v6_legacy"
    toks, pos = r.tag(normalize(s))
    assert len(T.token_nodes(rr.tree)) == sum(p != "PUNCT" for p in pos)   # still every token a node
    assert any(n.rel == REL_UNATTACHED for n in T.token_nodes(rr.tree))


def test_question_is_just_read(reader):
    assert reader.read_question("who ate the bone ?").tree == reader.read("who ate the bone ?").tree


def test_cache_round_trip(tmp_path):
    p = tmp_path / "cache.jsonl"
    r1 = Reader(cache_path=str(p), use_encoder=False)
    a = r1.read("the dog ate the bone .")
    r1.save_cache()
    r2 = Reader(cache_path=str(p), use_encoder=False)
    b = r2.read("the dog ate the bone .")
    assert b.tree == a.tree
    assert r2._parser is None  # served from cache, parser never loaded


def test_mind_teach_and_ask_not_built():
    m = Mind(reader=Reader(use_encoder=False))
    counts = m.teach(["the dog ate the bone ."])
    assert counts["trees"] == 1 and "ate" in m.dump()
    with pytest.raises(NotImplementedError):
        m.ask("what did the dog eat ?")
