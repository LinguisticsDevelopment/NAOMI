"""Tests for the chained-question consistency probe (multi-question, no reset)."""

import torch

from nsm_ct import build_default_stack, load_config
from nsm_ct.dataset import EpisodeDataset, make_dataloader
from nsm_ct.episode import chained_question_episode
from nsm_ct.metrics import answer_consistency


def test_chained_question_probe_runs_and_metric_well_defined():
    cfg = load_config()
    cfg.model.d_model = 32
    cfg.model.num_layers = 1
    cfg.model.nhead = 2
    cfg.model.dim_feedforward = 32

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
