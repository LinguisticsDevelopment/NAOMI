"""Tests for persistent long-term memory + the lifelong loop."""

import torch

from nsm_ct import build_default_stack, load_config
from nsm_ct.dataset import EpisodeDataset, make_dataloader
from nsm_ct.episode import CurriculumGenerator
from nsm_ct.lifelong import run_lifelong
from nsm_ct.long_term_memory import LongTermMemory


def _ltm(mem_dim=8, state_dim=6, max_size=10000, overwrite=False):
    torch.manual_seed(0)
    return LongTermMemory(mem_dim=mem_dim, state_dim=state_dim, max_size=max_size,
                          overwrite=overwrite)


# -- store mechanics ---------------------------------------------------------
def test_empty_read_is_zero_and_grows_on_consolidate():
    ltm = _ltm()
    assert len(ltm) == 0
    read = ltm.read(torch.randn(3, 6))
    assert read.shape == (3, 8) and torch.allclose(read, torch.zeros_like(read))

    idxs = ltm.consolidate(torch.randn(4, 8))
    assert idxs == [0, 1, 2, 3]
    assert len(ltm) == 4
    read = ltm.read(torch.randn(3, 6))
    assert read.abs().sum() > 0


def test_gated_consolidation_drops_ungated_entries():
    ltm = _ltm()
    vecs = torch.randn(3, 8)
    gates = torch.tensor([1.0, 0.0, 1.0])  # middle entry gated off
    idxs = ltm.consolidate(vecs, gates=gates)
    assert len(idxs) == 2 and len(ltm) == 2


def test_connections_repo():
    ltm = _ltm()
    # consolidating a group links them pairwise (connect_within_episode=True)
    ltm.consolidate(torch.randn(3, 8))
    assert ltm.num_connections == 3  # C(3,2)
    ns = ltm.neighbors(0)
    assert {j for j, _ in ns} == {1, 2}


def test_pruning_caps_size_and_reindexes_edges():
    ltm = _ltm(max_size=5)
    for _ in range(4):
        ltm.consolidate(torch.randn(3, 8))  # 12 entries total -> capped at 5
    assert len(ltm) == 5
    # all edges must reference valid (in-range) entries after re-indexing
    assert all(0 <= a < 5 and 0 <= b < 5 for (a, b) in ltm.edges)


def test_consolidate_overwrites_near_duplicate_in_place():
    """Overwrite, not forget: a near-duplicate updates an entry in place; novel
    content still grows the repo."""
    ltm = _ltm(overwrite=True)  # threshold 0.9
    a = torch.zeros(1, 8); a[0, 0] = 1.0
    b = torch.zeros(1, 8); b[0, 1] = 1.0  # orthogonal to a
    assert ltm.consolidate(a.clone()) == [0] and len(ltm) == 1
    assert ltm.consolidate(a.clone()) == [0] and len(ltm) == 1  # identical -> overwrite
    assert ltm.consolidate(b.clone()) == [1] and len(ltm) == 2  # novel -> append


def test_save_load_roundtrip(tmp_path):
    ltm = _ltm()
    ltm.consolidate(torch.randn(5, 8))
    path = tmp_path / "repo.pt"
    ltm.save(str(path))

    fresh = _ltm()
    fresh.load(str(path))
    assert len(fresh) == 5
    assert fresh.num_connections == ltm.num_connections


# -- integration with the Mind loop -----------------------------------------
def _small_cfg():
    cfg = load_config()
    cfg.model.d_model = 32
    cfg.model.num_layers = 1
    cfg.model.nhead = 2
    cfg.model.dim_feedforward = 32
    cfg.model.loop_mode = "sequential"  # clean per-item consolidation for these tests
    cfg.model.use_long_term = True
    cfg.data.num_episodes = 16
    cfg.train.batch_size = 8
    return cfg


def _loader_batch(cfg, episodes, stack):
    ds = EpisodeDataset(episodes, stack.encoder, stack.tokenizer, stack.answer_vocab, cfg)
    loader = make_dataloader(ds, stack.tokenizer.pad_id, cfg.train.batch_size, shuffle=False)
    return next(iter(loader))


def test_mind_consolidates_overwrites_repeats_and_grows_on_new():
    cfg = _small_cfg()
    episodes = CurriculumGenerator(max_level=2, seed=0).generate(cfg.data.num_episodes)
    stack = build_default_stack(cfg, episodes)
    assert stack.long_term is not None
    batch = _loader_batch(cfg, episodes, stack)

    out = stack.psyche(batch)
    first = stack.psyche.consolidate(out, batch)
    assert first > 0
    size_after_first = len(stack.long_term)
    # entries carry their fact text (a readable "facts we know" repo)
    assert any(stack.long_term.facts())

    # Re-consolidating the SAME facts overwrites in place — no growth (not forgotten).
    stack.psyche.consolidate(stack.psyche(batch), batch)
    assert len(stack.long_term) == size_after_first

    # A genuinely different batch grows the persistent repo.
    new_eps = CurriculumGenerator(max_level=3, seed=7).generate(cfg.data.num_episodes)
    new_batch = _loader_batch(cfg, new_eps, stack)
    stack.psyche.consolidate(stack.psyche(new_batch), new_batch)
    assert len(stack.long_term) > size_after_first


def test_long_term_off_by_default():
    cfg = load_config()
    cfg.data.num_episodes = 8
    episodes = CurriculumGenerator(max_level=1, seed=0).generate(8)
    stack = build_default_stack(cfg, episodes)
    assert stack.long_term is None and stack.mind.long_term is None


# -- lifelong loop -----------------------------------------------------------
def test_run_lifelong_grows_repo_over_rounds():
    cfg = _small_cfg()
    cfg.data.num_episodes = 16
    _stack, history = run_lifelong(cfg, num_rounds=3, episodes_per_round=8, verbose=False)
    assert len(history) == 3
    # the long-term repo accumulates across rounds
    assert history[-1]["ltm_entries"] > history[0]["ltm_entries"]
    assert history[-1]["ltm_connections"] > 0
