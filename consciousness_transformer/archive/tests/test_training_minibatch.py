"""Tests for the M57c footprint fix (RESEARCH_NOTES tail, "M57c battery #1"):
minibatched training/eval in scripts/train_instances.py,
scripts/train_writeback.py, scripts/train_resolver.py via the shared
scripts/_train_common.py helper. Two things are gated here:

1. ``_train_common.eval_minibatched`` aggregates per-row model output over
   minibatches EXACTLY as a single full-batch forward pass would (same
   predictions, row for row) -- the thing that would silently break the
   per-kind/binding-accuracy reports downstream if minibatch stitching ever
   mixed up row order.
2. Each training script's ``run_arm`` completes and produces the expected
   summary keys with ``--batch-size 0`` (full-batch, the pre-fix behavior,
   kept reachable for measurement) AND a small nonzero batch size.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from nsm_ct.clause_reactor import ClauseBatch, ClauseReactor  # noqa: E402
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

from _train_common import epoch_minibatches, eval_minibatched, peak_rss_mb  # noqa: E402

import train_instances  # noqa: E402
import train_resolver  # noqa: E402
import train_writeback  # noqa: E402


def _toy_batch(b=20, d=16, K=4, seed=0):
    """Same shape ``tests/test_clause_reactor.py::_toy_batch`` builds --
    one statement step + one question step, statement value = the gold
    option -- a plain (no-resolver) batch is enough to exercise
    ``eval_minibatched``'s row-order-preservation contract, since it
    stitches every tensor key the model returns, not just resolver-specific
    ones."""
    g = torch.Generator().manual_seed(seed)
    opts = F.normalize(torch.randn(b, K, d, generator=g), dim=-1)
    ans = torch.randint(0, K, (b,), generator=g)
    ent = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    rel = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    val = opts[torch.arange(b), ans]
    prd = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    entity = torch.stack([ent, ent], dim=1)
    relation = torch.stack([rel, rel], dim=1)
    value = torch.stack([val, torch.zeros(b, d)], dim=1)
    pred = torch.stack([prd, prd], dim=1)
    is_q = torch.tensor([[0.0, 1.0]]).repeat(b, 1)
    mask = torch.ones(b, 2)
    return ClauseBatch(entity, relation, value, pred, is_q, mask, opts, ans)


# ---------------------------------------------------------------------------
# epoch_minibatches
# ---------------------------------------------------------------------------
def test_epoch_minibatches_full_batch_is_single_unshuffled_arange():
    idx_list = list(epoch_minibatches(37, 0, seed=5, epoch=3))
    assert len(idx_list) == 1
    assert idx_list[0].tolist() == list(range(37))


def test_epoch_minibatches_covers_every_index_exactly_once():
    for epoch in range(4):
        idx_list = list(epoch_minibatches(37, 8, seed=5, epoch=epoch))
        covered = sorted(i for chunk in idx_list for i in chunk.tolist())
        assert covered == list(range(37))
        # every chunk but the last is exactly batch_size
        for chunk in idx_list[:-1]:
            assert len(chunk) == 8


def test_epoch_minibatches_deterministic_given_seed_and_epoch():
    a = [c.tolist() for c in epoch_minibatches(50, 10, seed=1, epoch=2)]
    b = [c.tolist() for c in epoch_minibatches(50, 10, seed=1, epoch=2)]
    assert a == b


def test_epoch_minibatches_differ_across_epochs():
    a = [c.tolist() for c in epoch_minibatches(50, 10, seed=1, epoch=0)]
    b = [c.tolist() for c in epoch_minibatches(50, 10, seed=1, epoch=1)]
    assert a != b


# ---------------------------------------------------------------------------
# eval_minibatched aggregation == full-batch evaluation, for a FIXED model
# ---------------------------------------------------------------------------
def test_eval_minibatched_matches_full_batch_predictions():
    torch.manual_seed(0)
    batch = _toy_batch(b=23, d=16, K=4)  # 23 is not a multiple of the batch size below
    model = ClauseReactor(dim=16)
    model.eval()

    with torch.no_grad():
        full_out = model(batch)
    mb_out = eval_minibatched(model, batch, batch_size=7)

    assert set(mb_out.keys()) == set(full_out.keys())
    for k in full_out:
        assert torch.allclose(mb_out[k], full_out[k], atol=1e-5), f"mismatch on key {k!r}"

    full_pred = full_out["answer_logits"].argmax(-1)
    mb_pred = mb_out["answer_logits"].argmax(-1)
    assert torch.equal(full_pred, mb_pred)


def test_eval_minibatched_batch_size_zero_is_full_batch_forward():
    torch.manual_seed(1)
    batch = _toy_batch(b=9, d=16, K=4)
    model = ClauseReactor(dim=16)
    model.eval()
    with torch.no_grad():
        full_out = model(batch)
    zero_out = eval_minibatched(model, batch, batch_size=0)
    for k in full_out:
        assert torch.equal(zero_out[k], full_out[k])


def test_peak_rss_mb_is_a_positive_float():
    v = peak_rss_mb()
    assert isinstance(v, float)
    assert v > 0.0


# ---------------------------------------------------------------------------
# Tiny end-to-end runs of each script's run_arm: batch-size 0 (full-batch)
# vs a small nonzero batch size both complete and produce the summary keys.
# ---------------------------------------------------------------------------
def _tiny_episodes(n=40, seed=0):
    return CurriculumGenerator(max_level=6, seed=seed).generate(n)


def _quick_setup_check():
    """Mirrors every run_arm's own quantum_parser availability check --
    tests should skip cleanly (not fail) in an environment without it,
    exactly like the training scripts print-and-return."""
    tok = SimpleTokenizer.build(["hello world"], extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    return getattr(parser, "_parser", None) is not None


import pytest  # noqa: E402

_PARSER_AVAILABLE = _quick_setup_check()
pytestmark = pytest.mark.skipif(not _PARSER_AVAILABLE, reason="quantum_parser unavailable in this environment")


@pytest.mark.parametrize("batch_size", [0, 8])
def test_train_writeback_run_arm_completes(batch_size):
    episodes = train_writeback.build_writeback_curriculum(60, seed=0)
    result = train_writeback.run_arm("test", "A", episodes, dim=8, epochs=2, seed=0, hidden=16,
                                      batch_size=batch_size)
    assert "total_acc" in result
    assert "peak_rss_mb" in result and result["peak_rss_mb"] > 0.0
    assert "n_resolver_params" in result and result["n_resolver_params"] > 0
    assert "binding" in result


@pytest.mark.parametrize("batch_size", [0, 8])
def test_train_instances_run_arm_completes(batch_size):
    episodes = train_instances.build_instance_curriculum(60, seed=0)
    result = train_instances.run_arm("test", "A", episodes, dim=8, epochs=2, seed=0, hidden=16,
                                      batch_size=batch_size)
    assert "total_acc" in result
    assert "peak_rss_mb" in result and result["peak_rss_mb"] > 0.0
    assert "n_resolver_params" in result and result["n_resolver_params"] > 0
    assert "binding" in result


@pytest.mark.parametrize("batch_size", [0, 8])
def test_train_resolver_run_arm_completes(batch_size):
    meaning_resolver = NSMMeaningResolver()  # noqa: F841  (import-availability sanity only)
    episodes = train_resolver.build_mixed_curriculum(60, seed=0)
    result = train_resolver.run_arm("test", "A", episodes, dim=8, epochs=2, seed=0, hidden=16,
                                     batch_size=batch_size)
    assert result is not None
    assert "total_acc" in result
    assert "n_resolver_params" in result and result["n_resolver_params"] > 0


def test_train_writeback_small_batch_reaches_comparable_val_to_full_batch():
    """Minibatching changes optimization dynamics -- spot-check at a small
    scale (same seed, a few more epochs to give SGD noise time to settle)
    that --batch-size 64-equivalent-small doesn't collapse relative to
    full-batch."""
    episodes = train_writeback.build_writeback_curriculum(150, seed=0)
    full = train_writeback.run_arm("full", "A", episodes, dim=12, epochs=20, seed=0, hidden=24,
                                    batch_size=0)
    mini = train_writeback.run_arm("mini", "A", episodes, dim=12, epochs=20, seed=0, hidden=24,
                                    batch_size=32)
    assert full["total_acc"] > 0.0 and mini["total_acc"] > 0.0
    # not a strict gate (smoke scale, CLAUDE.md: never gates curriculum
    # validity) -- just a sanity floor that minibatching hasn't broken
    # learning entirely.
    assert mini["total_acc"] > 0.3
