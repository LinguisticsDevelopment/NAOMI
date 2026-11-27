"""Mind loop: memory is pre-stored by the storing code and never changed by T; no action-menu heads; halting distribution; loss sanity."""
import numpy as np
import torch

from naomi import mind_loop as ML


def tiny():
    torch.manual_seed(0)
    return ML.Loop(16, 5, 4, H=32, nh=2, L=1)


def mem_tensors(n=7):
    m = {"row": np.arange(1, n + 1), "key": [None] * n, "rel": np.ones(n, dtype=np.int64), "pos": np.ones(n, dtype=np.int64),
         "par": np.array([ML.NMEM] + list(range(n - 1))), "sent": np.zeros(n, dtype=np.int64), "tidx": np.arange(n), "n": n}
    SM = torch.nn.functional.normalize(torch.randn(n + 2, 16), dim=-1)
    return ML.memory_tensors([m, None], [ML.CodeBook(1), ML.CodeBook(2)], SM, 5, 4), SM


def question(SM, B=2):
    f = {"row": np.array([1, 2]), "key": [None, None], "rel": np.array([1, 2]), "pos": np.array([1, 1]), "par": np.array([ML.K, 0]), "n": 2}
    return ML.tree_tensors([f] * B, [ML.CodeBook(1)] * B, SM, 5, 4)[0]


def test_memory_is_stored_by_code_and_unchanged_by_T():
    m = tiny()
    mt, SM = mem_tensors()
    mem = m.encode_memory(mt)
    before = (mem[0].clone(), mem[1].clone(), {k: v.clone() for k, v in mt.items()})
    assert not mt["valid"][1].any()                      # the second memory is empty: no stored slot at all
    m.ask(question(SM), m.init_state(2), mem, 3)
    assert torch.equal(before[0], mem[0]) and torch.equal(before[1], mem[1])
    assert all(torch.equal(before[2][k], mt[k]) for k in mt)
    assert not any("write" in n or "erase" in n for n, _ in m.named_parameters())
    assert not hasattr(m, "write") and not hasattr(m, "read")


def test_no_action_menu_heads():
    m = tiny()
    names = {n.split(".")[0] for n, _ in m.named_parameters()}
    assert names >= {"o_vec", "o_rel", "o_pos", "o_ex", "halt"}
    for bad in ("pick", "keep", "drop", "replace", "stop", "abstain", "anchor", "pointer", "copy", "sentence"):
        assert not any(bad in n for n in names), bad
    # every output head is a per-slot tree field; the only scalar-per-item head is the halt read from the state
    mt, SM = mem_tensors()
    outs, hl = m.ask(question(SM), m.init_state(2), m.encode_memory(mt), 3)
    assert hl.shape == (2, 3) and outs[0]["vec"].shape == (2, ML.K, 16) and outs[0]["par"].shape == (2, ML.K, ML.K + 1)


def test_halting_distribution_and_loss():
    hl = torch.randn(4, 6)
    p = ML.halting(hl)
    assert torch.allclose(p.sum(1), torch.ones(4), atol=1e-5)
    m = tiny()
    mt, SM = mem_tensors()
    qin = question(SM)
    f = {"row": np.array([3, 4, 5]), "key": [None] * 3, "rel": np.array([1, 1, 2]), "pos": np.array([1, 1, 1]), "par": np.array([ML.K, 0, 0]), "n": 3}
    _, tg = ML.tree_tensors([f, None], [ML.CodeBook(1)] * 2, SM, 5, 4)
    outs, hl = m.ask(qin, m.init_state(2), m.encode_memory(mt), 3)
    loss, Ls, pp = ML.ponder_loss(outs, hl, tg)
    loss.backward()
    assert torch.isfinite(loss) and Ls.shape == (3,)
    # a perfect output tree has (near) zero tree loss; an empty reference with all-zero existence is cheap
    out = {"vec": tg["vec"].clone(), "has": tg["has"].clone().clamp(1e-3, 1 - 1e-3), "idc": tg["idc"].clone(), "ex": tg["ex"].clone().clamp(1e-3, 1 - 1e-3),
           "rel_logit": torch.nn.functional.one_hot(tg["rel"], 5).float() * 30, "pos_logit": torch.nn.functional.one_hot(tg["pos"], 4).float() * 30,
           "par": torch.nn.functional.one_hot(tg["par"], ML.K + 1).float().clamp(1e-4, 1)}
    assert ML.tree_loss(out, tg)[0] < 0.2 < ML.tree_loss(out, tg)[1] + 1.0
