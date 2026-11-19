"""Fast tests for Mind v4 (stepping down the tree: HERE / DOWN / NEXT / PREV / ABSTAIN, one recursive merge, local state) on a toy story."""
import sys
from pathlib import Path

import pytest
import torch

from naomi import mind_v1 as m1
from naomi import mind_v2 as m2
from naomi import mind_v4 as m4
from naomi.reader import Reader
from naomi.tree import tree_to_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

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
    return reads, m2.Vocab.from_reads(reads)


def pair_of(reads, vocab, lines, key, with_target=True):
    story = m2.Story(lines, reads, vocab)
    pf = m2.pair_features(story, m1.tree_of(reads, QUESTION), vocab)
    pf.update(id=f"p{key}", kind="normal", item=f"p{key}", skey=key)
    if with_target:
        pf["tgt"] = m2.build_targets(story, m1.tree_of(reads, "the old door"))
    return story, pf


def model_for(vocab, **kw):
    torch.manual_seed(0)
    return m4.Thinker4(m1.Ctx.get().D, len(vocab.rels), len(vocab.poss), **kw).eval()


def test_walk_stays_in_the_tree_and_empty_memory_abstains(world):
    reads, vocab = world
    story, pf = pair_of(reads, vocab, STORY, 0)
    stories = {0: story.arrays()}
    tab = m2.SenseTable()
    model = model_for(vocab)
    b = m2.collate([pf], stories, tab, with_targets=False)
    e, anch, res = m4.decide(model, b)
    assert e["move"].shape == (story.N, 3) and e["in2"].shape == (story.N, 2) and res[0]["steps"] <= m4.BUDGET
    path = res[0]["path"]
    for u, v in zip(path, path[1:]):                      # every step is DOWN to a child or a move between sentence roots
        assert story.parent[v] == u or (story.parent[u] < 0 and story.parent[v] < 0)
    assert (anch[0] >= 0) == (not res[0]["exhausted"] and anch[0] != -1)
    kept, _ = m2.keep_sets(model, e, b, stories, [pf], anch)
    assert (anch[0] < 0) == (kept[0] is None)
    empty = m2.Story([], reads, vocab)
    pf0 = m2.pair_features(empty, m1.tree_of(reads, QUESTION), vocab)
    pf0.update(id="e", kind="emp", item="e", skey=1)
    b0 = m2.collate([pf0], {1: empty.arrays()}, tab, with_targets=False)
    assert m4.decide(model, b0)[1] == [-1]


def test_node_state_is_local(world):
    reads, vocab = world
    story, pf = pair_of(reads, vocab, STORY, 0)
    tab = m2.SenseTable()
    model = model_for(vocab)
    b = m2.collate([pf], {0: story.arrays()}, tab, with_targets=False)
    h1 = model.encode(b)["hs"].detach().clone()
    sent = story.sent
    s2 = torch.where(torch.from_numpy(sent) == 2, torch.zeros(()).long(), b["s_sense"])     # blank another sentence's words
    b2 = dict(b)
    b2["s_sense"] = s2
    h2 = model.encode(b2)["hs"].detach()
    same = torch.from_numpy(sent) != 2
    assert torch.allclose(h1[same], h2[same], atol=1e-5)          # no sentence-wide or story-wide pooling reaches a node state
    assert not torch.allclose(h1[~same], h2[~same], atol=1e-5)


def test_gold_states_and_training(world, tmp_path):
    reads, vocab = world
    story, pf = pair_of(reads, vocab, STORY, 0)
    stories = {0: story.arrays()}
    tab = m2.SenseTable()
    model = model_for(vocab)
    b = m2.collate([pf], stories, tab)
    e = model.encode(b)
    nodes, ws, acc4, acc_down, _ = m4.build_states([pf], b, stories, e, None, False)
    a = pf["tgt"]["anchors"][0][0]
    i = nodes.tolist().index(a)
    assert bool(acc4[i, m4.N_HERE]) and float(ws.sum()) == pytest.approx(1.0)       # HERE is acceptable at the anchor, one unit of weight per item
    assert bool(acc_down[a]) is False or story.parent[a] >= 0
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = None
    for _ in range(25):
        lp, ls, lk = m4.stage1_loss(model, b, [pf], stories, None, False)
        loss = lp + ls + lk
        first = float(loss) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first
    model.eval()
    _e, anch, res = m4.decide(model, b)
    assert anch[0] == a and res[0]["steps"] >= 2                      # the learned walk reaches the anchor by DOWN steps then HERE
    import train_mind_v4 as T4
    path = tmp_path / "t.pt"
    T4.save(model, vocab.to_json(), path)
    model2, _ = T4.load_model(str(path))
    assert isinstance(model2, m4.Thinker4) and m4.decide(model2, b)[1] == anch


def test_wrong_passage_state_is_abstain(world):
    reads, vocab = world
    story, pf = pair_of(reads, vocab, STORY, 0)
    pf["kind"] = "wrong"
    stories = {0: story.arrays()}
    model = model_for(vocab)
    b = m2.collate([pf], stories, m2.SenseTable())
    e = model.encode(b)
    nodes, ws, acc4, acc_down, _ = m4.build_states([pf], b, stories, e, None, False)
    assert acc4[:, m4.N_ABST].all() and not acc4[:, :3].any() and not acc_down.any()


def test_sentence_moves_cannot_loop(world):
    reads, vocab = world
    story, pf = pair_of(reads, vocab, STORY, 0)
    stories = {0: story.arrays()}
    b = m2.collate([pf], stories, m2.SenseTable(), with_targets=False)
    for seed in range(5):
        torch.manual_seed(seed)
        model = m4.Thinker4(m1.Ctx.get().D, len(vocab.rels), len(vocab.poss)).eval()
        _e, _a, res = m4.decide(model, b, budget=40)
        roots = [g for g in res[0]["path"] if story.parent[g] < 0]
        assert len(roots) == len(set(roots))                # a neighbour must beat the current sentence by a margin: no ping-pong
