"""Tests for chained-question answering (multi-question, one unreset run)."""

import torch

from nsm_ct import build_default_stack, load_config
from nsm_ct.dataset import EpisodeDataset, make_dataloader
from nsm_ct.episode import chained_question_episode, generate_chained_episodes
from nsm_ct.losses import compute_losses
from nsm_ct.metrics import answer_consistency


def test_chained_question_probe_runs_and_metric_well_defined():
    cfg = load_config()
    cfg.model.d_model = 32
    cfg.model.num_layers = 1
    cfg.model.nhead = 2
    cfg.model.dim_feedforward = 32
    cfg.input_encoder = "token"

    eps, positions, pairs = [], None, None
    for i in range(6):
        ep, positions, _gold, pairs = chained_question_episode(seed=i)
        eps.append(ep)
    stack = build_default_stack(cfg, eps)  # default = controlled loop
    ds = EpisodeDataset(eps, stack.encoder, stack.tokenizer, stack.answer_vocab, cfg)
    batch = next(iter(make_dataloader(ds, stack.tokenizer.pad_id, len(eps), shuffle=False)))

    pos = torch.tensor([positions] * len(eps), dtype=torch.long)
    answers = stack.psyche.answer_at_positions(batch, pos)
    # one answer per question, all valid option indices, answered in one run
    assert answers.shape == (len(eps), 3)
    n_opts = len(eps[0].options)
    assert ((answers >= 0) & (answers < n_opts)).all()
    assert 0.0 <= answer_consistency(answers, pairs) <= 1.0


def test_multi_question_loss_trains_chained_answering():
    """Training with the per-question loss makes one unreset run answer several
    questions: per-question accuracy beats chance and the repeat agrees with Q1."""
    cfg = load_config()
    cfg.model.d_model = 32
    cfg.model.num_layers = 1
    cfg.model.nhead = 2
    cfg.model.dim_feedforward = 32
    cfg.input_encoder = "token"
    torch.manual_seed(0)

    eps = generate_chained_episodes(12, seed=3)
    pairs = [(0, 2)]
    stack = build_default_stack(cfg, eps)
    ds = EpisodeDataset(eps, stack.encoder, stack.tokenizer, stack.answer_vocab, cfg)
    batch = next(iter(make_dataloader(ds, stack.tokenizer.pad_id, len(eps), shuffle=False)))
    assert batch.q_positions.shape[1] == 3  # three questions per stream

    opt = torch.optim.AdamW(stack.psyche.parameters(), lr=1e-2)
    for _ in range(300):
        out = stack.psyche(batch)
        loss = compute_losses(out, batch, 1.0, 0.0, weight_multi=1.0).total
        opt.zero_grad(); loss.backward(); opt.step()

    preds = stack.psyche(batch)["question_logits"].argmax(-1)
    acc = (preds == batch.q_targets).float().mean()
    assert float(acc) > 0.5  # chance = 0.25; all three questions, one unreset run
    assert answer_consistency(preds, pairs) > 0.6
