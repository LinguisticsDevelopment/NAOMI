"""Fast tests for Mind v2 (thinker, answer trees, decoder (b) word choice, ROUGE-L) on a 3-sentence toy story."""
import numpy as np
import pytest
import torch

from naomi import mind_v1 as m1
from naomi import mind_v2 as m2
from naomi.reader import Reader
from naomi.tree import tree_to_dict, walk

STORY = ["tom fixed the old door .", "jim painted the wall .", "the cat slept on the bed ."]
QUESTION = "What did Tom fix?"


@pytest.fixture(scope="module")
def world():
    reader = Reader(use_encoder=False, cache_path=None)
    reads = {}
    for s in STORY + [QUESTION, "the old door"]:
        rr = reader.read_parser(s)
        assert rr is not None
        reads[s] = {"tree": tree_to_dict(rr.tree), "k": s}
    vocab = m2.Vocab.from_reads(reads)
    story = m2.Story(STORY, reads, vocab)
    return reads, vocab, story


def test_rouge_l():
    assert m2.rouge_l("the old door", "the old door") == pytest.approx(1.0)
    assert m2.rouge_l("old door", "the old door") == pytest.approx(0.8)
    assert m2.rouge_l("cat", "door") == 0.0
    assert m2.rouge_l("", "door") == 0.0
    assert m2.rouge_multi("a b", ["x", "a b"]) == pytest.approx(1.0)


def test_targets_find_the_reference_subtree(world):
    reads, vocab, story = world
    t = m2.build_targets(story, m1.tree_of(reads, "the old door"))
    assert t["best_f1"] >= 0.8
    a, kept, _f = t["anchors"][0]
    arr = story.arrays()
    assert set(m2.true_order_text(arr, set(kept)).split()) == {"the", "old", "door"}


def test_thinker_shapes_and_forced_abstain(world):
    reads, vocab, story = world
    arr = story.arrays()
    stories = {0: arr}
    qt = m1.tree_of(reads, QUESTION)
    pf = m2.pair_features(story, qt, vocab)
    pf.update(id="x", kind="normal", item="x", skey=0)
    tab = m2.SenseTable()
    torch.manual_seed(0)
    model = m2.Thinker(m1.Ctx.get().D, len(vocab.rels), len(vocab.poss)).eval()
    b = m2.collate([pf], stories, tab, with_targets=False)
    e, anch, res = m2.decide(model, b)
    assert e["alog"].shape[0] == story.N
    kept, _ = m2.keep_sets(model, e, b, stories, [pf], anch)
    assert (anch[0] < 0) == (kept[0] is None)
    # empty memory: nothing to anchor on, so it must abstain
    empty = m2.Story([], reads, vocab)
    pf0 = m2.pair_features(empty, qt, vocab)
    pf0.update(id="y", kind="emp", item="y", skey=1)
    b0 = m2.collate([pf0], {1: empty.arrays()}, tab, with_targets=False)
    _e, a0, _r = m2.decide(model, b0)
    assert a0 == [-1]


def test_answer_tree_is_connected_subtree(world):
    reads, vocab, story = world
    arr = story.arrays()
    g = next(g for g in range(story.N) if story.refs[g].node.token == "door")
    tree = m2.answer_tree(arr, g, {g})
    assert [n.token for n, _d, _p in walk(tree.root)] == ["door"]
    assert tree.root.rel == "ROOT"


def test_meaning_only_word_choice_recovers_words(world):
    reads, vocab, story = world
    arr = story.arrays()
    qn = m2.flat_tree(m1.tree_of(reads, QUESTION))[0]
    pool = m2.Pool(arr["nodes"], qn)
    g = next(g for g in range(story.N) if story.refs[g].node.token == "door")
    kept = {g} | set(arr["desc"][g].tolist())
    tree = m2.answer_tree(arr, g, kept)
    tb, log = m2.meaning_only(tree, pool)
    assert len(log) == len(kept)
    assert all(c == o for c, o, _hp in log)
