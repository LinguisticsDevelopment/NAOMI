"""Fast tests for the stage-A encoder (naomi.encoder): features, the unknown mask, decoding, proposals,
a tiny overfit, and tree JSON round-trip. A stub USVS stands in for the 614-axis space."""
import itertools
import random

import numpy as np
import pytest
import torch

from naomi import encoder as E
from naomi.reader import Reader
from naomi.tree import compact, tree_from_json, tree_to_json, walk
from naomi.types import REL_ROOT, REL_UNATTACHED, Node, Tree


class StubUSVS:
    axes = [f"ax{i}" for i in range(8)]

    def sense_dense(self, sid):
        if sid.startswith("x"):
            return None
        v = np.zeros(8, dtype=np.float32)
        v[sum(map(ord, sid)) % 8] = 1.0
        return v


def _node(i, rel, children=()):
    return Node(rel=rel, token=f"w{i}", token_index=i, lemma=f"w{i}", pos="NOUN", children=list(children))


def _single_root_acyclic(heads):
    """heads: {token: 0|token+1}; exactly one root, every token reaches it without a cycle."""
    assert sum(1 for h in heads.values() if h == 0) == 1
    for i in heads:
        seen, cur = set(), i
        while heads[cur] != 0:
            assert cur not in seen
            seen.add(cur)
            cur = heads[cur] - 1
            assert cur in heads


def _random_net(hash_buckets=0, hidden=32):
    cfg = dict(d_in=32, hidden=hidden, layers=1, hash_buckets=hash_buckets, d_arc=32, d_rel=16, dropout=0.0)
    labels = ["ROOT", "A", "B"]
    pv, sv = {None: 0, "NOUN": 1, "VERB": 2, "PUNCT": 3, "DET": 4}, {"X": 0, "Y": 1}
    net = E.EncoderNet(len(pv), len(sv), len(labels), 8, **cfg)
    return E.NaomiEncoder(net, labels, E.FeatureBuilder(StubUSVS(), pv, sv, hash_buckets), cfg), cfg


def test_features_build_for_a_sentence():
    rd = Reader(use_encoder=False, cache_path=None)
    infos = E.infos_from_sentence(rd, "The dog saw a cat, and it ran.")
    assert [t.token for t in infos][:3] == ["the", "dog", "saw"]
    assert [t.is_node for t in infos].count(False) == 2          # the comma and the full stop stay as context
    pv = {None: 0, **{p: i + 1 for i, p in enumerate(sorted({t.pos for t in infos}))}}
    subs = sorted({s for t in infos for s in t.subtypes})
    fb = E.FeatureBuilder(StubUSVS(), pv, {s: i for i, s in enumerate(subs)}, 16)
    f = fb.build(infos)
    n = len(infos)
    assert f.pos.shape == (n,) and f.node.shape == (n,) and f.hashed.max() < 16 and len(f.subs) == n
    b = E.collate([f], fb.table, len(subs))
    assert b["meaning"].shape == (1, n, 8) and not b["node"][0, [i for i, t in enumerate(infos) if not t.is_node]].any()


def test_unknown_mask_excludes_exactly_unconnected_tops():
    # main component: root w1 with child w0; piece: top w3 (UNATTACHED) with child w4; loose token w5; w2 punctuation
    main = _node(1, REL_ROOT, [_node(0, "SUBJECT")])
    piece = _node(3, REL_UNATTACHED, [_node(4, "OBJECT")])
    loose = _node(5, REL_UNATTACHED)
    synth = Node(rel=REL_ROOT, token="ROOT", token_index=None, lemma="ROOT", pos="ROOT", children=[main, piece, loose])
    sv = E.supervision_from_tree(Tree("s", synth), 6)
    assert sv.unknown == [False, False, False, True, False, True]
    assert sv.sup == [True, True, False, False, True, False]
    assert sv.head[1] == 0 and sv.rel[1] == REL_ROOT       # main root -> ROOT
    assert sv.head[0] == 2 and sv.rel[0] == "SUBJECT"      # head is token 1 (+1)
    assert sv.head[4] == 4 and sv.rel[4] == "OBJECT"       # inside the piece: supervised for its head within the piece
    assert not sv.connected
    full = E.supervision_from_tree(Tree("s", _node(1, REL_ROOT, [_node(0, "SUBJECT")])), 2)
    assert full.connected and full.sup == [True, True] and not any(full.unknown)


def test_decode_single_root_acyclic_property():
    rng = np.random.RandomState(0)
    for n in list(range(1, 13)) * 6:
        W = rng.randn(n + 1, n + 1) * 3
        heads = E.decode_heads(W)
        _single_root_acyclic({i - 1: int(heads[i]) for i in range(1, n + 1)})   # slot s = token s-1


def test_decode_matches_brute_force_on_small_inputs():
    rng = np.random.RandomState(1)
    for _ in range(25):
        n = 4
        W = rng.randn(n + 1, n + 1)
        best = -1e18
        for hs in itertools.product(range(n + 1), repeat=n):
            h = {i: hs[i] for i in range(n)}                    # token i -> slot (0 = ROOT, s = token s-1)
            if any(h[i] == i + 1 for i in h) or sum(1 for v in h.values() if v == 0) != 1:
                continue
            try:
                _single_root_acyclic(h)
            except AssertionError:
                continue
            best = max(best, sum(W[h[i], i + 1] for i in h))
        got = E.decode_heads(W)
        assert abs(E.tree_score(W, got) - best) < 1e-9


def _infos(n):
    return [E.TokenInfo(f"t{i}", "NOUN" if i % 3 else "VERB", ["X"] if i % 2 else [], "word", [f"s{i}"], f"t{i}")
            for i in range(n)] + [E.TokenInfo(".", "PUNCT", [], "word", [], ".")]


def test_propose_returns_distinct_valid_trees():
    torch.manual_seed(0)
    enc, _ = _random_net()
    infos = _infos(7)
    props = enc.propose_infos("s", infos, k=5)
    assert 1 < len(props) <= 5
    keys = {tuple(sorted(p.heads.items())) for p in props}
    assert len(keys) == len(props)
    assert [p.score for p in props] == sorted((p.score for p in props), reverse=True)
    for p in props:
        _single_root_acyclic(p.heads)
        assert sorted(p.heads) == list(range(7))            # every non-punctuation token, no punctuation
        idx = sorted(n.token_index for n, _, _ in walk(p.tree.root))
        assert idx == list(range(7)) and p.tree.root.rel == REL_ROOT
    again = enc.propose_infos("s", infos, k=5)
    assert [p.heads for p in again] == [p.heads for p in props]   # deterministic


def test_tiny_overfit_and_tree_roundtrip():
    torch.manual_seed(0)
    rng = random.Random(0)
    cfg = dict(d_in=48, hidden=48, layers=1, hash_buckets=512, d_arc=48, d_rel=16, dropout=0.0)
    labels = ["ROOT", "A", "B"]
    pv, sv = {None: 0, "NOUN": 1, "VERB": 2, "PUNCT": 3, "DET": 4}, {"X": 0, "Y": 1}
    fb = E.FeatureBuilder(StubUSVS(), pv, sv, 512)
    items = []
    for s in range(20):
        n = rng.randint(5, 9)
        infos = [E.TokenInfo(f"w{s}_{i}", rng.choice(["NOUN", "VERB", "DET"]), [], "word", [f"s{rng.randint(0, 9)}"], "")
                 for i in range(n)]
        root = rng.randrange(n)
        head = [0 if i == root else None for i in range(n)]
        order = [root]
        for i in rng.sample([i for i in range(n) if i != root], n - 1):
            head[i] = rng.choice(order) + 1
            order.append(i)
        rel = [0 if h == 0 else rng.choice([1, 2]) for h in head]
        items.append({"feats": fb.build(infos), "head": np.array(head), "relid": np.array(rel),
                      "sup": np.ones(n, dtype=bool), "infos": infos})
    net = E.EncoderNet(len(pv), len(sv), 3, 8, **cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    n_sub = len(sv)

    def batch():
        b = E.collate([it["feats"] for it in items], fb.table, n_sub)
        B, T = b["pos"].shape
        h = torch.full((B, T), -1, dtype=torch.long)
        r = torch.zeros((B, T), dtype=torch.long)
        sp = torch.zeros((B, T), dtype=torch.bool)
        for k, it in enumerate(items):
            n = len(it["head"])
            h[k, :n], r[k, :n], sp[k, :n] = torch.from_numpy(it["head"]), torch.from_numpy(it["relid"]), True
        return b, h, sp

    b, h, sp = batch()
    for _ in range(300):
        H = net.encode(b["pos"], b["subs"], b["meaning"], b["name"], b["hashed"], b["lengths"])
        s = net.arc_scores(H).masked_fill(~E.head_mask(b["node"]), -1e9)
        bi, di = sp.nonzero(as_tuple=True)
        loss = torch.nn.functional.cross_entropy(s[bi, di], h[bi, di])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    acc = (s[bi, di].argmax(-1) == h[bi, di]).float().mean().item()
    assert acc > 0.9, acc
    enc = E.NaomiEncoder(net, labels, fb, cfg)
    prop, _ = enc.decode_infos("sent", items[0]["infos"])
    _single_root_acyclic(prop.heads)
    t2 = tree_from_json(tree_to_json(prop.tree))
    assert compact(t2) == compact(prop.tree) and t2.source == prop.tree.source
    assert tree_to_json(t2) == tree_to_json(prop.tree)
