"""Fast tests for decoder v1 (naomi.orderer): permutation + contiguity, position removal, ceiling test, tiny overfit,
lookup baseline. A stub USVS stands in for the 614-axis space."""
import random

import numpy as np
import torch
import torch.nn.functional as F

from naomi import orderer as O
from naomi.encoder import FeatureBuilder
from naomi.types import REL_ROOT, Node, Tree

POS = {None: 0, "NOUN": 1, "VERB": 2, "DET": 3, "ADJ": 4, "ADV": 5}
RELS = [REL_ROOT, "SUBJECT", "OBJECT", "DESCRIPTION", "MODIFICATION", O.UNK_REL]
RID = {r: i for i, r in enumerate(RELS)}


class StubUSVS:
    axes = [f"ax{i}" for i in range(8)]

    def sense_dense(self, sid):
        v = np.zeros(8, dtype=np.float32)
        v[sum(map(ord, sid)) % 8] = 1.0
        return v


def builder():
    return FeatureBuilder(StubUSVS(), POS, {"X": 0}, 0)


def mk(tok, rel, pos, kids=()):
    return Node(rel=rel, token=tok, token_index=None, lemma=tok, pos=pos, children=list(kids))


def synthetic(rng, k=0):
    """verb with SUBJECT noun (before), OBJECT noun (after), optional DET/ADJ before nouns, optional ADV after."""
    def noun(tag):
        kids = ([mk(f"{tag}d{rng.randrange(99)}", "DESCRIPTION", "DET")] if rng.random() < .7 else [])
        kids += ([mk(f"{tag}j{rng.randrange(99)}", "DESCRIPTION", "ADJ")] if rng.random() < .5 else [])
        return kids
    s = mk(f"s{k}", "SUBJECT", "NOUN", noun("s"))
    o = mk(f"o{k}", "OBJECT", "NOUN", noun("o"))
    kids = [s, o] + ([mk(f"a{k}", "MODIFICATION", "ADV")] if rng.random() < .5 else [])
    root = mk(f"v{k}", REL_ROOT, "VERB", kids)
    # linear order: DET < ADJ < noun; SUBJECT subtree < verb < OBJECT subtree < ADV
    def lin(n):
        pre = sorted((c for c in n.children if c.rel == "DESCRIPTION"), key=lambda c: c.pos != "DET")
        return [x for c in pre for x in lin(c)] + [n]
    seq = lin(s) + [root] + lin(o) + [c for c in kids if c.rel == "MODIFICATION"]
    for i, n in enumerate(seq):
        n.token_index = i
    return Tree("", root)


def sort_children(t):  # ground-truth trees come with position-sorted children, like the parser's
    def go(n):
        n.children.sort(key=lambda c: c.token_index)
        for c in n.children:
            go(c)
    go(t.root)
    return t


def _net(d=32):
    return O.OrdererNet(len(POS), 1, len(RELS), 8, d=d, rounds=2, d_pair=48, dropout=0.0)


def _orderer(net):
    return O.Orderer(net, builder(), RELS, {})


def _subtree_tokens(n):
    out = [n.token]
    for c in n.children:
        out += _subtree_tokens(c)
    return out


def test_output_is_a_permutation_with_contiguous_subtrees():
    torch.manual_seed(0)
    o = _orderer(_net())
    rng = random.Random(1)
    for k in range(10):
        t = sort_children(synthetic(rng, k))
        out = [n.token for n in o.order_tree(t)]
        assert sorted(out) == sorted(_subtree_tokens(t.root)) and len(set(out)) == len(out)
        pos = {w: i for i, w in enumerate(out)}

        def check(n):
            ps = sorted(pos[w] for w in _subtree_tokens(n))
            assert ps[-1] - ps[0] + 1 == len(ps)
            for c in n.children:
                check(c)
        check(t.root)


def test_position_information_is_removed_from_the_input():
    torch.manual_seed(1)
    o = _orderer(_net())
    rng = random.Random(2)
    for k in range(8):
        t = sort_children(synthetic(rng, k))
        a = [n.token for n in o.order_tree(t)]
        # same tree, children reversed and token_index scrambled: nothing positional may reach the model
        import copy
        t2 = copy.deepcopy(t)
        perm = list(range(len(a)))
        random.Random(k).shuffle(perm)

        def scramble(n):
            n.token_index = perm[n.token_index]
            n.children.reverse()
            for c in n.children:
                scramble(c)
        scramble(t2.root)
        assert [n.token for n in o.order_tree(t2, seed=5)] == a
    tin = O.tree_in(t, builder(), RID)
    assert set(O.collate_trees([tin], builder().table, 1)) == {"pos", "subs", "meaning", "name", "rel", "parent", "pa", "pb", "ph"}


def test_ceiling_detects_non_projective_trees():
    a = mk("a", "SUBJECT", "NOUN")
    b = mk("b", "OBJECT", "NOUN", [mk("c", "DESCRIPTION", "ADJ")])
    root = mk("r", REL_ROOT, "VERB", [a, b])
    for n, i in ((a, 0), (root, 1), (b, 2), (b.children[0], 3)):
        n.token_index = i
    assert O.contiguous(O.tree_in(Tree("", root), builder(), RID))
    b.children[0].token_index, a.token_index = 0, 3          # c jumps over everything: b's subtree is split
    root.token_index, b.token_index = 1, 2
    a.token_index = 4
    assert not O.contiguous(O.tree_in(Tree("", root), builder(), RID))


def test_tiny_overfit_and_baseline():
    torch.manual_seed(0)
    rng = random.Random(3)
    trees = [sort_children(synthetic(rng, k)) for k in range(40)]
    b = builder()
    tins = [O.tree_in(t, b, RID, seed=k) for k, t in enumerate(trees)]
    net = _net()
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    batch = O.collate_trees(tins, b.table, 1)
    y = torch.from_numpy(np.concatenate([(t.true_pos[t.pa] < t.true_pos[t.pb]).astype(np.float32) for t in tins]))
    for _ in range(250):
        loss = F.binary_cross_entropy_with_logits(net(batch), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    m = O.evaluate(tins, _orderer(net).group_orders, by_band=False)["all"]
    assert m["exact_order"] > 0.95 and m["pairwise_acc"] > 0.98
    base = O.LookupBaseline([], [])
    base.fit(tins)
    mb = O.evaluate(tins, lambda ts: [base.group_order(t) for t in ts], by_band=False)["all"]
    assert mb["exact_order"] > 0.9


def test_kendall_distance():
    assert O.kendall_distance([0, 1, 2, 3]) == 0.0
    assert O.kendall_distance([3, 2, 1, 0]) == 1.0


def test_sample_fragment_is_connected_with_gaps():
    rng = random.Random(1)
    parent = [-1, 0, 0, 1, 1, 2, 3, 3, 5, 0]
    sizes = set()
    for _ in range(200):
        keep = O.sample_fragment(parent, rng)
        ks = set(keep)
        tops = [i for i in keep if parent[i] not in ks]
        assert len(tops) == 1                                # one root, every other kept node's parent is kept
        assert 1 <= len(keep) <= 10 and keep == sorted(keep)
        sizes.add(len(keep))
    assert len(sizes) > 4
    assert len(O.sample_fragment(parent, rng, size=4)) == 4
