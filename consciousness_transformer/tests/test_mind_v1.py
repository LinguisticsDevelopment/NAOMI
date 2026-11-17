"""Fast tests for the Mind v1 one-step policy on a 3-sentence toy story (parser reading only)."""
import copy
import json

import numpy as np
import pytest
import torch

from naomi import mind_v1 as m1
from naomi.memory import Memory
from naomi.mind import Mind
from naomi.reader import Reader
from naomi.tree import walk

STORY = ["tom fixed the old door .", "jim painted the wall .", "the cat slept on the bed ."]
REL_VOCAB = None


@pytest.fixture(scope="module")
def reader():
    return Reader(use_encoder=False, cache_path=None)


@pytest.fixture(scope="module")
def rel_vocab(reader):
    reads = {}
    for s in STORY + ["What did Tom fix?"]:
        rr = reader.read_parser(s)
        assert rr is not None
        reads[s] = {"tree": __import__("naomi.tree", fromlist=["tree_to_dict"]).tree_to_dict(rr.tree)}
    return m1.make_rel_vocab(reads)


def make_policy(rel_vocab, abstain_bias=None):
    torch.manual_seed(0)
    R = len(rel_vocab)
    p = m1.Policy(m1.Ctx.get().D, m1.n_cand_features(R), m1.n_q_features(R)).eval()
    if abstain_bias is not None:
        with torch.no_grad():
            p.abst[-1].bias.fill_(abstain_bias)
    return p


def taught(reader, policy, rel_vocab):
    m = Mind(reader=reader, policy=policy, rel_vocab=rel_vocab)
    m.teach(STORY)
    return m


def test_candidates_include_reachable_nodes(reader, rel_vocab):
    m = taught(reader, make_policy(rel_vocab), rel_vocab)
    q = reader.read_question("What did Tom fix?").tree
    ex = m1.extract(m.memory, q, rel_vocab)
    refs, _ = m1.node_table(m.memory)
    ctx = m1.Ctx.get()
    seeds = [int(g) for g, f in zip(ex.cand_gid, ex.fs[:, 7]) if f == 1.0]
    assert seeds, "FIND found nothing"
    cands = set(int(g) for g in ex.cand_gid)
    for g in seeds:
        r = refs[g]
        for h in ([r.parent] if r.parent >= 0 else []) + r.children:
            if not ctx.is_function(refs[h].node):
                assert h in cands, "a content node one tree link from a FIND seed is missing"
    # the object of the matched verb is reachable
    assert any(refs[g].node.lemma == "door" for g in cands)


def test_abstain_is_a_valid_output(reader, rel_vocab):
    m = taught(reader, make_policy(rel_vocab, abstain_bias=100.0), rel_vocab)
    a = m.ask("What did Tom fix?")
    assert a.stated is False and a.text is None and a.support is None


def test_memory_severed_abstains_or_chance(reader, rel_vocab):
    m = Mind(reader=reader, policy=make_policy(rel_vocab), rel_vocab=rel_vocab)     # empty memory
    a = m.ask("What did Tom fix?")
    assert a.stated is False
    # only the question in memory: any answer can only come from the question itself (sentence index 0)
    q = reader.read_question("What did Tom fix?").tree
    m.memory.store(q, "current", {"source": "question", "sentence_index": 0})
    a = m.ask("What did Tom fix?")
    assert (not a.stated) or m.last_trace["sentence_ids"] == [0]


def test_policy_inputs_are_numeric_and_free_of_tokens(reader, rel_vocab):
    m = taught(reader, make_policy(rel_vocab), rel_vocab)
    q = reader.read_question("What did Tom fix?").tree
    ex = m1.extract(m.memory, q, rel_vocab)
    for name in ("cand_gid", "cand_sent", "fs", "cv", "sv", "wv", "qf", "itf"):
        a = getattr(ex, name)
        assert isinstance(a, np.ndarray) and a.dtype.kind in "fiu", name
    # surface tokens of ordinary words carry nothing: scrambling them leaves every feature unchanged
    mem2 = Memory()
    for st in sorted(m.memory.trees(), key=lambda s: s.nid):
        t = copy.deepcopy(st.tree)
        for n, _d, _p in walk(t.root):
            if n.kind == "word":
                n.token = "x" + str(n.token_index)
        mem2.store(t, "current", dict(st.provenance))
    q2 = copy.deepcopy(q)
    for n, _d, _p in walk(q2.root):
        if n.kind == "word":
            n.token = "y" + str(n.token_index)
    ex2 = m1.extract(mem2, q2, rel_vocab)
    for name in ("cand_gid", "fs", "cv", "sv", "wv", "qf", "itf"):
        assert np.allclose(getattr(ex, name).astype(float), getattr(ex2, name).astype(float)), name
    # the policy takes tensors only
    out = m1.make_batch_check(ex) if hasattr(m1, "make_batch_check") else m1.collate([ex])
    assert all(isinstance(v, torch.Tensor) for v in out.values())


def test_trace_json_and_end_to_end(reader, rel_vocab):
    m = taught(reader, make_policy(rel_vocab, abstain_bias=-100.0), rel_vocab)
    a = m.ask("What did Tom fix?")
    assert a.stated and isinstance(a.text, str) and a.text
    tr = m.last_trace
    json.dumps(tr)
    assert tr["sentence_ids"] and tr["facts"][0]["sentence"] in STORY
    assert tr["candidates_considered"] > 0 and tr["top"]
    assert json.loads(a.trace[0])["text"] == a.text
    assert a.support is not None


def test_labels_mark_the_matching_subtree(reader, rel_vocab):
    m = taught(reader, make_policy(rel_vocab), rel_vocab)
    q = reader.read_question("What did Tom fix?").tree
    ex = m1.extract(m.memory, q, rel_vocab)
    refs, _ = m1.node_table(m.memory)
    opts = [reader.read_parser(s).tree if reader.read_parser(s) else None for s in ("old door", "wall")]
    from naomi.encoder import build_tree, TokenInfo
    from naomi.types import REL_ROOT
    def one(tok):
        return build_tree(tok, [TokenInfo(tok, "NOUN", [], "word", [], tok)], {0: 0}, {0: REL_ROOT}, "parser")
    opts = [one("door"), one("wall")]
    for o in opts:                                    # give the one-word trees the lexicon's senses
        from naomi.reader import make_node
        n = make_node(REL_ROOT, o.root.token, 0, "NOUN")
        o.root.senses, o.root.lemma = n.senses, n.lemma
    meaning, overall, correct = m1.label_candidates(refs, ex.cand_gid, opts, 0)
    assert correct.any()
    assert any(refs[int(g)].node.lemma == "door" for g, c in zip(ex.cand_gid, correct) if c)
