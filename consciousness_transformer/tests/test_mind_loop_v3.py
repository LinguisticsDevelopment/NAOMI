"""Mind loop v3: the reward grade (+1 / +0.25 / -1), committing beats averaging, no copy/pointer head, memory unchanged."""
import numpy as np
import torch
import torch.nn.functional as F

from naomi import mind_loop as ML
from naomi import mind_loop_v3 as V3

D, R, P = 16, 5, 4


def tiny(B=2, n=3):
    torch.manual_seed(0)
    m = ML.Loop(D, R, P, H=32, nh=2, L=1)
    SM = F.normalize(torch.randn(10, D), dim=-1)
    SM[0] = 0
    mem = {"row": np.array([1, 2, 3]), "key": [None] * 3, "rel": np.array([1, 1, 1]), "pos": np.array([1, 1, 1]), "par": np.array([ML.NMEM, 0, 0]),
           "sent": np.zeros(3, dtype=np.int64), "tidx": np.arange(3), "n": 3}
    ref = {"row": np.array([2, 9]), "key": [None] * 2, "rel": np.array([1, 2]), "pos": np.array([1, 1]), "par": np.array([ML.K, 0]), "n": 2}
    q = {"row": np.array([4]), "key": [None], "rel": np.array([1]), "pos": np.array([1]), "par": np.array([ML.K]), "n": 1}
    return m, SM, mem, ref, q


def test_reward_values_as_declared():
    assert (V3.R_CORRECT, V3.R_NONE, V3.R_WRONG) == (1.0, 0.25, -1.0)
    t = V3.reward_table(4, correct=2)
    assert t.tolist() == [-1.0, -1.0, 1.0, -1.0, 0.25]
    assert V3.reward_table(4, -1).tolist() == [-1.0] * 4 + [0.25]
    # certain choices: committed correct +1, silence +0.25, committed wrong -1
    p = torch.tensor([[0.0, 1.0, 0.0]])
    assert float(V3.expected_reward(p, torch.zeros(1), torch.tensor([1]))) == 1.0
    assert float(V3.expected_reward(p, torch.zeros(1), torch.tensor([0]))) == -1.0
    assert float(V3.expected_reward(torch.zeros(1, 3), torch.ones(1), torch.tensor([1]))) == 0.25
    # break-even: answering beats silence only when P(correct) > 0.625
    for pc, better in ((0.7, True), (0.6, False)):
        pr = torch.tensor([[pc, 1 - pc]])
        assert (float(V3.expected_reward(pr, torch.zeros(1), torch.tensor([0]))) > 0.25) is better


def test_averaged_prediction_scores_lower_than_committed():
    # two plausible answers, 50/50: the blurred (averaged-meaning) slot spreads mass over both, the committed one picks one; silence is best of the three
    blur = torch.tensor([[0.5, 0.5, 0.0]])
    commit = torch.tensor([[1.0, 0.0, 0.0]])
    c = torch.tensor([0])
    r_blur = float(V3.expected_reward(blur, torch.zeros(1), c))
    r_commit_right = float(V3.expected_reward(commit, torch.zeros(1), c))
    assert r_blur < r_commit_right and r_blur == 0.0
    # sure of the answer: committing > silence > blurred; when only 50% sure silence beats the blur
    assert r_commit_right > 0.25 > r_blur
    # the real softmax: a slot whose meaning is the average of two candidates has s = 0.5 / 0.5 -> lower reward than a slot sitting on one of them
    SM = F.normalize(torch.randn(4, D), dim=-1)
    avg = F.normalize(SM[1] + SM[2], dim=-1)
    pool = {"vec": SM[None, 1:4], "code": torch.zeros(1, 3, ML.CODE), "pt": torch.ones(1, 3, dtype=torch.bool), "valid": torch.ones(1, 3, dtype=torch.bool)}
    out_avg = {"vec": avg[None, None], "idc": torch.zeros(1, 1, ML.CODE)}
    out_one = {"vec": SM[None, 1:2], "idc": torch.zeros(1, 1, ML.CODE)}
    s_avg = torch.softmax(V3.choice_logits(out_avg, pool), -1)[0, 0]
    s_one = torch.softmax(V3.choice_logits(out_one, pool), -1)[0, 0]
    ra = float(V3.expected_reward(s_avg[None], torch.zeros(1), torch.tensor([0])))
    ro = float(V3.expected_reward(s_one[None], torch.zeros(1), torch.tensor([0])))
    assert ro > ra


def test_pool_and_loss_paths():
    m, SM, mem, ref, q = tiny()
    books = [ML.CodeBook(1)] * 2
    pool = V3.build_pool([mem, None], [q, q], books, SM, [ref, None])
    assert pool["valid"][0].sum() == 4 and pool["valid"][1].sum() == 1          # memory words + question word; empty memory: question only
    assert pool["tidx"][0, 0] >= 0 and pool["tidx"][0, 1] == -1                  # ref row 2 is in the pool, ref row 9 is not (unreachable)
    qin, tg = ML.tree_tensors([q, q], books, SM, R, P)
    _, tg = ML.tree_tensors([ref, None], books, SM, R, P, None)
    mt = ML.memory_tensors([mem, None], books, SM, R, P)
    outs, hl = m.ask(qin, m.init_state(2), m.encode_memory(mt), 3)
    loss, stats = V3.tree_loss_v3(outs[-1], tg, pool)
    assert loss.shape == (2,) and bool(torch.isfinite(loss).all())
    assert float(stats["ref"]) == 2 and float(stats["unreach"]) == 1             # excluded and counted
    loss.sum().backward()
    assert m.o_ex.weight.grad is not None and float(m.o_ex.weight.grad.abs().sum()) > 0
    # commit warm-up: no gradient to the existence head
    m.zero_grad()
    outs, hl = m.ask(qin, m.init_state(2), m.encode_memory(mt), 1)      # one step: ex also feeds the next step's input, which is not the reward path
    loss, _ = V3.tree_loss_v3(outs[-1], tg, pool, commit=True)
    loss.sum().backward()
    assert m.o_ex.weight.grad is None or float(m.o_ex.weight.grad.abs().sum()) == 0.0
    # empty reference: every slot's correct choice is "no node" -> loss falls as existence falls
    for b in (1,):
        assert float(tg["ex"][b].sum()) == 0


def test_no_copy_pointer_head_and_memory_unchanged():
    m, SM, mem, ref, q = tiny()
    names = " ".join(n for n, _ in m.named_parameters())
    for bad in ("pick", "keep", "drop", "replace", "abstain", "anchor", "pointer", "copy", "sentence", "write", "erase", "pool", "cand"):
        assert bad not in names, bad
    books = [ML.CodeBook(1)]
    mt = ML.memory_tensors([mem], books, SM, R, P)
    before = {k: v.clone() for k, v in mt.items()}
    mem0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in mem.items()}
    qin, tg = ML.tree_tensors([q], books, SM, R, P)
    enc = m.encode_memory(mt)
    outs, hl = m.ask(qin, m.init_state(1), enc, 3)
    pool = V3.build_pool([mem], [q], books, SM, [ref])
    V3.tree_loss_v3(outs[-1], ML.tree_tensors([ref], books, SM, R, P, None)[1], pool)[0].sum().backward()
    assert all(torch.equal(before[k], mt[k]) for k in before)
    assert all(np.array_equal(mem0[k], mem[k]) if hasattr(mem[k], "shape") else mem0[k] == mem[k] for k in mem)
    assert V3.TAU == 0.1


def test_decoupled_variant_keeps_choice_gradient_when_silent():
    m, SM, mem, ref, q = tiny()
    books = [ML.CodeBook(1)]
    qin, _ = ML.tree_tensors([q], books, SM, R, P)
    tg = ML.tree_tensors([ref], books, SM, R, P, None)[1]
    pool = V3.build_pool([mem], [q], books, SM, [ref])
    mt = ML.memory_tensors([mem], books, SM, R, P)
    with torch.no_grad():
        m.o_ex.bias.fill_(-12.0)                                       # a silent model
    g = {}
    for dec in (False, True):
        m.zero_grad()
        outs, _ = m.ask(qin, m.init_state(1), m.encode_memory(mt), 1)
        V3.tree_loss_v3(outs[-1], tg, pool, decouple=dec)[0].sum().backward()
        g[dec] = float(m.o_vec.weight.grad.abs().sum())
    assert g[False] < 1e-4 * max(g[True], 1e-12) + 1e-9 and g[True] > 1e-6
