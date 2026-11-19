"""Fast tests for Mind v3 (sentence step, sentence context, length-invariant shape, abstain from scores) on a toy story."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from naomi import mind_v1 as m1
from naomi import mind_v2 as m2
from naomi import mind_v3 as m3
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
    vocab = m2.Vocab.from_reads(reads)
    return reads, vocab


def pair_of(reads, vocab, story_lines, key, with_target=True):
    story = m2.Story(story_lines, reads, vocab)
    qt = m1.tree_of(reads, QUESTION)
    pf = m2.pair_features(story, qt, vocab)
    pf.update(id=f"p{key}", kind="normal", item=f"p{key}", skey=key)
    if with_target:
        pf["tgt"] = m2.build_targets(story, m1.tree_of(reads, "the old door"))
    return story, pf


def model_for(vocab, **kw):
    torch.manual_seed(0)
    return m3.Thinker3(m1.Ctx.get().D, len(vocab.rels), len(vocab.poss), **kw).eval()


def test_two_step_decision_and_abstain(world):
    reads, vocab = world
    story, pf = pair_of(reads, vocab, STORY, 0)
    stories = {0: story.arrays()}
    tab = m2.SenseTable()
    model = model_for(vocab)
    b = m2.collate([pf], stories, tab, with_targets=False)
    e, anch, res = m3.decide(model, b)
    assert e["alog"].shape[0] == story.N and e["sl_pad"].shape == (1, 3)
    if anch[0] >= 0:                        # the anchor lies inside the top-scoring sentence
        assert int(stories[0]["sent"][anch[0]]) == int(e["sl_pad"][0].argmax())
    kept, _ = m2.keep_sets(model, e, b, stories, [pf], anch)
    assert (anch[0] < 0) == (kept[0] is None)
    # empty memory: only the abstain option exists
    empty = m2.Story([], reads, vocab)
    pf0 = m2.pair_features(empty, m1.tree_of(reads, QUESTION), vocab)
    pf0.update(id="e", kind="emp", item="e", skey=1)
    b0 = m2.collate([pf0], {1: empty.arrays()}, tab, with_targets=False)
    assert m3.decide(model, b0)[1] == [-1]


def test_shape_features_are_sentence_relative(world):
    reads, vocab = world
    story, pf = pair_of(reads, vocab, STORY, 0)
    model = model_for(vocab)
    b = m2.collate([pf], {0: story.arrays()}, m2.SenseTable(), with_targets=False)
    uniq, gs = torch.unique(b["s_sent"], return_inverse=True)
    sc, maxd, ssz = model.shape(b, gs, uniq.numel())
    assert sc.shape == (story.N, m3.NSI)
    assert float(sc[:, 0].max()) <= 1.0 and float(sc[:, 1].max()) <= 1.0          # depth / sentence max depth, size / sentence size
    for s in range(uniq.numel()):                                                  # the sentence root holds the whole sentence
        sel = gs == s
        assert float(sc[sel, 1].max()) == pytest.approx(1.0)
    assert torch.equal(ssz, torch.bincount(gs).float()[gs])


def test_context_changes_a_sentence_score_only_with_context(world):
    reads, vocab = world
    other = ["the king sat on the old chair ."]
    reads2 = dict(reads)
    reader = Reader(use_encoder=False, cache_path=None)
    rr = reader.read_parser(other[0])
    reads2[other[0]] = {"tree": tree_to_dict(rr.tree), "k": other[0]}
    tab = m2.SenseTable()
    out = {}
    for ctx in (True, False):
        model = model_for(vocab, ctx=ctx)
        sc = []
        for lines in (STORY, STORY[:2] + other):                   # same first two sentences, a different third one
            story, pf = pair_of(reads2, vocab, lines, 0, with_target=False)
            b = m2.collate([pf], {0: story.arrays()}, tab, with_targets=False)
            sc.append(model.encode(b)["sl_pad"][0, :2].detach())
        out[ctx] = float((sc[0] - sc[1]).abs().max())
    assert out[False] < 0.2 * out[True]


def test_stage1_loss_trains_and_checkpoint_roundtrip(world, tmp_path):
    reads, vocab = world
    story, pf = pair_of(reads, vocab, STORY, 0)
    stories = {0: story.arrays()}
    tab = m2.SenseTable()
    model = model_for(vocab)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    b = m2.collate([pf], stories, tab)
    first = None
    for _ in range(15):
        ls, la, lk = m3.stage1_loss(model, b, [pf], stories, [1.0])
        loss = ls + la + lk
        first = float(loss) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first
    model.eval()
    e, anch, _ = m3.decide(model, b)
    assert anch[0] >= 0
    import train_mind_v3 as T3
    path = tmp_path / "t.pt"
    T3.save(model, vocab.to_json(), path)
    model2, _ = T3.load_model(str(path))
    assert isinstance(model2, m3.Thinker3)
    assert m3.decide(model2, b)[1] == anch
    # sampling returns an anchor and a differentiable log-prob
    model.train()
    e2 = model.encode(b)
    a, lps = m3.sample(e2, b)
    assert len(a) == 1 and lps[0].requires_grad


def test_ablation_flags_build(world):
    reads, vocab = world
    story, pf = pair_of(reads, vocab, STORY, 0)
    b = m2.collate([pf], {0: story.arrays()}, m2.SenseTable(), with_targets=False)
    for inv, ctx in ((False, True), (True, False), (False, False)):
        e = model_for(vocab, inv=inv, ctx=ctx).encode(b)
        assert torch.isfinite(e["ablog"]).all() and torch.isfinite(e["alog"]).all()
