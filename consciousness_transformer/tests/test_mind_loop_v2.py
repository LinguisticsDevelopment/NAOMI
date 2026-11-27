"""Mind loop v2: halt prior warm-up schedule; counterfactual swap keeps the question fixed and changes only aligned slots; no action menu; T never writes memory."""
import random

import numpy as np
import torch

from naomi import mind_loop as ML
from naomi import mind_loop_v2 as V2
from dataclasses import dataclass


@dataclass
class NS:
    token: str
    rel: int


def story(n, base, pos):
    mem = {"row": np.arange(base, base + n), "key": [None] * n, "rel": np.ones(n, dtype=np.int64), "pos": np.full(n, pos, dtype=np.int64),
           "par": np.array([ML.NMEM] + list(range(n - 1))), "sent": np.zeros(n, dtype=np.int64), "tidx": np.arange(n), "n": n}
    return {"mem": mem}, {"nodes": [NS(token=f"w{base + i}", rel=1) for i in range(n)]}


def setup():
    sf, st = {}, {}
    for s, base in (("a", 1), ("b", 100), ("c", 200)):
        sf[s], st[s] = story(6, base, 3)
    sf["a"]["mem"]["row"][5] = 1          # slot 5 repeats identity of slot 0 (a second mention)
    return sf, st


def test_counterfactual_changes_only_aligned_slots_and_keeps_question():
    sf, st = setup()
    cf = V2.CF(sf, st, ["a", "b", "c"], max_df=0.9)
    q = {"row": np.array([2]), "key": [None], "n": 1}          # question mentions identity 2: must not be swapped
    ref = {"row": np.array([1, 2, 3]), "key": [None] * 3, "rel": np.ones(3, dtype=np.int64), "pos": np.ones(3, dtype=np.int64), "par": np.array([ML.K, 0, 0]), "n": 3}
    q0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in q.items()}
    orig = sf["a"]["mem"]["row"].copy()
    r = cf.build("a", q, ref, random.Random(0))
    assert r is not None
    assert all(np.array_equal(q[k], q0[k]) if hasattr(q[k], "shape") else q[k] == q0[k] for k in q)
    assert np.array_equal(sf["a"]["mem"]["row"], orig)                      # the original memory is untouched
    changed = np.where(r["mem"]["row"] != orig)[0]
    assert len(changed) > 0 and 2 not in set(orig[changed])                 # only slots whose identity is in the reference and not in the question
    swapped = {int(orig[j]) for j in changed}
    assert swapped <= {1, 3}
    # every slot with a swapped identity changed, consistently to one donor; the reference changed in the same places
    for i in swapped:
        sl = np.where(orig == i)[0]
        assert len(set(r["mem"]["row"][sl].tolist())) == 1 and r["mem"]["row"][sl[0]] not in orig.tolist()
        assert (r["ref"]["row"][ref["row"] == i] == r["mem"]["row"][sl[0]]).all()
    assert np.array_equal(r["ref"]["row"][ref["row"] == 2], ref["row"][ref["row"] == 2])
    assert np.array_equal(r["mem"]["pos"], sf["a"]["mem"]["pos"]) and np.array_equal(r["mem"]["par"], sf["a"]["mem"]["par"])


def test_swap_text():
    assert V2.swap_text("the Wolf, ran away.", [("wolf", "bear")]) == "the bear, ran away."


def test_halt_warmup_schedule_and_prior():
    assert [V2.is_warm(e, 4) for e in range(6)] == [True] * 4 + [False] * 2
    p0 = V2.halt_prior(6)
    assert abs(float(p0.sum()) - 1) < 1e-5 and abs(V2.LAM_PRIOR - 1 / 3) < 1e-9
    assert 2.5 < float((p0 * torch.arange(1, 7)).sum()) < 3.5
    # warm: halt logits get no gradient and every step has weight from the prior; released: halt gets gradient
    torch.manual_seed(0)
    m = ML.Loop(16, 5, 4, H=32, nh=2, L=1)
    n = 3
    mt = {"vec": torch.randn(2, n, 16), "has": torch.ones(2, n), "idc": torch.zeros(2, n, ML.CODE), "rel": torch.nn.functional.one_hot(torch.ones(2, n, dtype=torch.long), 5).float(),
          "pos": torch.nn.functional.one_hot(torch.ones(2, n, dtype=torch.long), 4).float(), "par": torch.full((2, n), -1), "sent": torch.zeros(2, n, dtype=torch.long),
          "tix": torch.arange(n).expand(2, n), "valid": torch.ones(2, n, dtype=torch.bool)}
    f = {"row": np.array([1, 2]), "key": [None] * 2, "rel": np.array([1, 2]), "pos": np.array([1, 1]), "par": np.array([ML.K, 0]), "n": 2}
    SM = torch.nn.functional.normalize(torch.randn(5, 16), dim=-1)
    qin, tg = ML.tree_tensors([f, f], [ML.CodeBook(1)] * 2, SM, 5, 4, None)
    outs, hl = m.ask(qin, m.init_state(2), m.encode_memory(mt), 4)
    loss, Ls, p = V2.ponder_loss_v2(outs, hl, tg, warm=True)
    loss.backward(retain_graph=True)
    assert m.halt.weight.grad is None or float(m.halt.weight.grad.abs().sum()) == 0.0
    assert torch.allclose(p[0], V2.halt_prior(4), atol=1e-6)
    m.zero_grad()
    loss, Ls, p = V2.ponder_loss_v2(outs, hl, tg, warm=False)
    loss.backward()
    assert float(m.halt.weight.grad.abs().sum()) > 0


def test_v2_has_no_action_menu_and_no_write_path():
    m = ML.Loop(16, 5, 4, H=32, nh=2, L=1)
    names = " ".join(n for n, _ in m.named_parameters())
    for bad in ("pick", "keep", "drop", "replace", "abstain", "anchor", "pointer", "copy", "sentence", "write", "erase"):
        assert bad not in names, bad
    assert not hasattr(V2, "Loop")                         # v2 adds no model: the v1 Loop is used unchanged
