"""End-to-end tests for the stateful reasoning loop (the thing that matters)."""

import torch

from nsm_ct import build_default_stack, load_config
from nsm_ct.dataset import EpisodeDataset, make_dataloader
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.losses import compute_losses
from nsm_ct.metrics import action_accuracy, answer_accuracy


def _small_cfg():
    cfg = load_config()
    cfg.model.d_model = 32
    cfg.model.num_layers = 1
    cfg.model.nhead = 2
    cfg.model.dim_feedforward = 32
    cfg.data.num_episodes = 16
    cfg.train.batch_size = 8
    return cfg


def _batch(cfg, episodes, stack, bs):
    ds = EpisodeDataset(episodes, stack.encoder, stack.tokenizer, stack.answer_vocab, cfg)
    loader = make_dataloader(ds, stack.tokenizer.pad_id, bs, shuffle=False)
    return next(iter(loader))


def test_one_full_unroll_and_training_step():
    cfg = _small_cfg()
    torch.manual_seed(0)
    episodes = CurriculumGenerator(max_level=3, seed=0).generate(cfg.data.num_episodes)
    stack = build_default_stack(cfg, episodes)
    batch = _batch(cfg, episodes, stack, cfg.train.batch_size)
    b = batch.answer_target.shape[0]

    out = stack.mind(batch)
    # shapes
    assert out["answer_logits"].shape == (b, batch.opt_ids.shape[1])
    assert out["ctx_action_logits"].shape[0] == b
    assert out["q_action_logits"].shape == (b, 3)
    # the state actually changes across the unroll (not a constant)
    states = out["states"]
    assert states.shape[0] == b and states.shape[1] >= 2
    assert (states[:, 0] - states[:, -1]).abs().sum() > 0
    # at least some memory got written during the episode
    assert out["memory_occupancy"].sum() > 0

    losses = compute_losses(out, batch, 1.0, 1.0, 0.05)
    assert torch.isfinite(losses.total)
    for c in (losses.answer, losses.action, losses.consistency):
        assert torch.isfinite(c)

    opt = torch.optim.AdamW(stack.mind.parameters(), lr=1e-3)
    opt.zero_grad()
    losses.total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in stack.mind.parameters())
    opt.step()


def test_loop_overfits_one_batch():
    """Training should drive answer + action accuracy up on a fixed batch."""
    cfg = _small_cfg()
    torch.manual_seed(0)
    episodes = CurriculumGenerator(max_level=2, seed=0).generate(16)
    stack = build_default_stack(cfg, episodes)
    batch = _batch(cfg, episodes, stack, 16)

    opt = torch.optim.AdamW(stack.mind.parameters(), lr=1e-2)
    first_loss = None
    for step in range(40):
        out = stack.mind(batch)
        losses = compute_losses(out, batch, 1.0, 1.0, 0.05)
        opt.zero_grad()
        losses.total.backward()
        opt.step()
        if step == 0:
            first_loss = float(losses.total.detach())
    last_loss = float(losses.total.detach())
    assert last_loss < first_loss

    # The action policy should be learnable to near-perfect (absorb/respond).
    out = stack.mind(batch)
    assert action_accuracy(out, batch) > 0.9
    # And it should answer the (tiny, memorized) batch better than chance.
    assert answer_accuracy(out, batch) > 0.25


def test_trace_reports_actions_and_answers():
    cfg = _small_cfg()
    episodes = CurriculumGenerator(max_level=1, seed=0).generate(4)
    stack = build_default_stack(cfg, episodes)
    batch = _batch(cfg, episodes, stack, 4)
    trace = stack.mind.trace(batch)
    assert len(trace["actions"]) == 4
    # each trace = one action per context step + one for the question
    assert all(isinstance(steps, list) and steps for steps in trace["actions"])
    assert len(trace["answers"]) == 4
