"""Tests for read/write working memory."""

import torch

from nsm_ct.memory import WorkingMemory


def _mem(mem_dim=8, state_dim=6):
    torch.manual_seed(0)
    return WorkingMemory(mem_dim=mem_dim, state_dim=state_dim)


def test_empty_read_is_zero():
    m = _mem()
    state = torch.randn(3, 6)
    mem = m.init_state(batch_size=3, num_slots=4, device=torch.device("cpu"))
    read = m.read(mem, state)
    assert read.shape == (3, 8)
    assert torch.allclose(read, torch.zeros_like(read))  # nothing written yet


def test_gated_write_controls_occupancy():
    m = _mem()
    mem = m.init_state(2, 4, torch.device("cpu"))
    content = torch.randn(2, 8)
    # example 0 writes fully, example 1 writes with gate 0.
    gate = torch.tensor([1.0, 0.0])
    mem = m.write(mem, slot_idx=0, content=content, gate=gate)
    occ = m.occupancy(mem)
    assert occ[0].item() == 1.0
    assert occ[1].item() == 0.0


def test_read_after_write_is_nonzero_and_differentiable():
    m = _mem()
    mem = m.init_state(1, 4, torch.device("cpu"))
    content = torch.randn(1, 8, requires_grad=True)
    gate = torch.tensor([1.0])
    mem = m.write(mem, 0, content, gate)
    state = torch.randn(1, 6, requires_grad=True)
    read = m.read(mem, state)
    assert read.abs().sum() > 0
    read.sum().backward()
    assert content.grad is not None and torch.isfinite(content.grad).all()


def test_ungated_write_leaves_memory_inert():
    m = _mem()
    mem = m.init_state(1, 4, torch.device("cpu"))
    mem = m.write(mem, 0, torch.randn(1, 8), gate=torch.tensor([0.0]))
    read = m.read(mem, torch.randn(1, 6))
    assert torch.allclose(read, torch.zeros_like(read))
