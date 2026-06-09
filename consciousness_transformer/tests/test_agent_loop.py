"""End-to-end tests for the emergent reasoning loop."""

import torch

from nsm_ct import build_default_stack, load_config
from nsm_ct.dataset import EpisodeDataset, make_dataloader
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.losses import compute_losses
from nsm_ct.metrics import answer_accuracy, mean_respond_position, trust_gap
from nsm_ct.model import NUM_ACTIONS


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
    episodes = CurriculumGenerator(max_level=4, seed=0).generate(cfg.data.num_episodes)
    stack = build_default_stack(cfg, episodes)
    batch = _batch(cfg, episodes, stack, cfg.train.batch_size)
    b = batch.answer_target.shape[0]
    t = batch.item_ids.shape[1]

    out = stack.mind(batch)
    assert out["answer_logits"].shape == (b, batch.opt_ids.shape[1])
    assert out["action_logits"].shape == (b, t, NUM_ACTIONS)   # 4-way repertoire
    assert out["respond_gates"].shape == (b, t)
    assert out["append_gates"].shape == (b, t)
    assert out["trust"].shape == (b, t)
    # state evolves across the stream
    states = out["states"]
    assert (states[:, 0] - states[:, -1]).abs().sum() > 0

    losses = compute_losses(out, batch, 1.0, 0.05)
    assert torch.isfinite(losses.total)
    for c in (losses.answer, losses.consistency):
        assert torch.isfinite(c)

    opt = torch.optim.AdamW(stack.mind.parameters(), lr=1e-3)
    opt.zero_grad()
    losses.total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in stack.mind.parameters())
    opt.step()


def test_no_action_supervision_in_loss():
    """The loss must not reference action labels (fully emergent)."""
    import inspect
    from nsm_ct import losses
    src = inspect.getsource(losses.compute_losses)
    assert "action" not in src.lower()  # no action-label supervision


def test_overfit_one_batch_learns_answer_and_timing():
    cfg = _small_cfg()
    torch.manual_seed(0)
    episodes = CurriculumGenerator(max_level=2, seed=0).generate(16)
    stack = build_default_stack(cfg, episodes)
    batch = _batch(cfg, episodes, stack, 16)

    init_answer = answer_accuracy(stack.mind(batch), batch)

    opt = torch.optim.AdamW(stack.mind.parameters(), lr=1e-2)
    for _ in range(80):
        out = stack.mind(batch)
        losses = compute_losses(out, batch, 1.0, 0.0)
        opt.zero_grad()
        losses.total.backward()
        opt.step()

    final = stack.mind(batch)
    # The answer is learned purely from answer loss (no action supervision), and
    # the response-position diagnostic stays well-defined in [0, 1].
    assert answer_accuracy(final, batch) > max(0.4, init_answer)
    assert 0.0 <= mean_respond_position(final, batch) <= 1.0


def test_emergent_timing_with_post_question_distractors():
    """A corrupting distractor follows the question; answering right requires the
    model to respond before the distractor rewrites memory — emergent timing."""
    cfg = _small_cfg()
    torch.manual_seed(0)
    gen = CurriculumGenerator(max_level=4, seed=1)
    episodes = [e for e in gen.generate(60) if e.level == 4][:12]
    stack = build_default_stack(cfg, episodes)
    batch = _batch(cfg, episodes, stack, len(episodes))
    # question is item index 1; the last item (index 2) is the corrupting move
    assert int(batch.question_index[0]) == 1 and batch.item_ids.shape[1] == 3

    init_answer = answer_accuracy(stack.mind(batch), batch)
    opt = torch.optim.AdamW(stack.mind.parameters(), lr=1e-2)
    for _ in range(150):
        out = stack.mind(batch)
        losses = compute_losses(out, batch, 1.0, 0.0)
        opt.zero_grad()
        losses.total.backward()
        opt.step()
    # The model solves the harder task — it cannot just read the final state.
    assert answer_accuracy(stack.mind(batch), batch) > max(0.4, init_answer)


def test_trust_emerges_on_corroboration():
    """Trained only to answer corroboration episodes correctly (no trust labels),
    the model learns to trust the corroborated fact over the contradiction."""
    cfg = _small_cfg()
    torch.manual_seed(0)
    gen = CurriculumGenerator(max_level=5, seed=2)
    episodes = [e for e in gen.generate(120) if e.level == 5][:16]
    stack = build_default_stack(cfg, episodes)
    batch = _batch(cfg, episodes, stack, len(episodes))
    assert (batch.trust_label == 0).any()  # contradictions present

    init_ans = answer_accuracy(stack.mind(batch), batch)
    opt = torch.optim.AdamW(stack.mind.parameters(), lr=1e-2)
    for _ in range(200):
        out = stack.mind(batch)
        losses = compute_losses(out, batch, 1.0, 0.0)
        opt.zero_grad()
        losses.total.backward()
        opt.step()

    final = stack.mind(batch)
    assert answer_accuracy(final, batch) > max(0.4, init_ans)
    # Trust is emergent (no labels); it should end up higher on corroborated
    # items than on the contradicting one.
    assert trust_gap(final, batch) > 0.0


def test_multihop_runs_and_default_is_single_hop():
    cfg = _small_cfg()
    episodes = CurriculumGenerator(max_level=3, seed=0).generate(cfg.data.num_episodes)
    assert build_default_stack(cfg, episodes).mind.reasoning_hops == 1

    cfg.model.reasoning_hops = 3
    stack = build_default_stack(cfg, episodes)
    assert stack.mind.reasoning_hops == 3
    batch = _batch(cfg, episodes, stack, cfg.train.batch_size)
    out = stack.mind(batch)
    losses = compute_losses(out, batch, 1.0, 0.05)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(p.grad is not None for p in stack.mind.parameters())


def test_trace_reports_actions_and_chosen_response_step():
    cfg = _small_cfg()
    episodes = CurriculumGenerator(max_level=2, seed=0).generate(4)
    stack = build_default_stack(cfg, episodes)
    batch = _batch(cfg, episodes, stack, 4)
    trace = stack.mind.trace(batch)
    assert len(trace["actions"]) == 4
    assert all(a in {"ABSORB", "APPEND", "RESPOND", "SKIP"} for steps in trace["actions"] for a in steps)
    assert len(trace["respond_step"]) == 4 and len(trace["answers"]) == 4
