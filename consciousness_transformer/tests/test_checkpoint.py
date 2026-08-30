"""Tests for M57 checkpointing (RESEARCH_NOTES M-ES1, "once checkpointing
lands"): src/nsm_ct/checkpoint.py's save_checkpoint/load_checkpoint,
scripts/_train_common.py's build_model, and the --save/--load flags on
scripts/train_instances.py and scripts/train_writeback.py.

Two tiers:
  1. Direct unit tests against ``build_model``/``save_checkpoint``/
     ``load_checkpoint`` on a small toy batch -- no quantum_parser
     dependency, always run.
  2. End-to-end CLI round trips (subprocess, small scale) through
     scripts/train_instances.py's/scripts/train_writeback.py's own
     --save/--load and scripts/eval_checkpoint.py -- skipped cleanly (not
     failed) when quantum_parser is unavailable, exactly like
     tests/test_training_minibatch.py's own skip contract.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
sys.path.insert(0, _SRC)
sys.path.insert(0, _SCRIPTS)

from nsm_ct.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from nsm_ct.clause_reactor import ClauseBatch  # noqa: E402

from _train_common import build_model  # noqa: E402


# ---------------------------------------------------------------------------
# toy batch -- same shape tests/test_training_minibatch.py's _toy_batch
# builds (one statement step + one question step, statement value = the
# gold option), enough to exercise a resolver-carrying forward pass without
# needing quantum_parser/build_clause_batch at all.
# ---------------------------------------------------------------------------
def _toy_batch(b=12, d=8, K=3, C=3, seed=0):
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
# 1. save -> load round trip: identical forward outputs on a fixed batch.
# ---------------------------------------------------------------------------
def test_save_load_round_trip_identical_forward_no_resolver(tmp_path):
    config = {"dim": 8, "hidden": 16, "track": None}
    torch.manual_seed(0)
    model = build_model(config)
    model.eval()
    batch = _toy_batch(d=8)
    with torch.no_grad():
        out_before = model(batch)

    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, config=config, extra={"note": "no-resolver arm"})
    loaded, loaded_config, extra = load_checkpoint(path)

    assert loaded.training is False  # loaded in eval mode
    with torch.no_grad():
        out_after = loaded(batch)
    assert set(out_after.keys()) == set(out_before.keys())
    for k in out_before:
        assert torch.equal(out_before[k], out_after[k]), f"mismatch on key {k!r}"
    assert loaded_config == config
    assert extra == {"note": "no-resolver arm"}


def test_save_load_round_trip_identical_forward_with_resolver(tmp_path):
    config = {"dim": 8, "hidden": 16, "track": "A", "use_cand_feature": True, "cand_feature_extra": 1}
    torch.manual_seed(3)
    model = build_model(config)
    model.eval()
    batch = _toy_batch(d=8)
    with torch.no_grad():
        out_before = model(batch)

    path = str(tmp_path / "ckpt_resolver.pt")
    save_checkpoint(path, model, config=config)
    loaded, _loaded_config, extra = load_checkpoint(path)

    with torch.no_grad():
        out_after = loaded(batch)
    for k in out_before:
        assert torch.equal(out_before[k], out_after[k]), f"mismatch on key {k!r}"
    assert extra == {}  # extra defaults to {} when omitted


# ---------------------------------------------------------------------------
# 2. config round-trips.
# ---------------------------------------------------------------------------
def test_config_round_trips_including_extra_informational_keys(tmp_path):
    config = {
        "dim": 8, "hidden": 16, "track": "B",
        "codec_dim": 8, "codec_max_pos": 64, "meaning_source": "usvs",
        "episodes": 40, "seed": 1, "git_commit": None, "argv": ["train_writeback.py", "--track", "B"],
    }
    model = build_model(config)
    path = str(tmp_path / "ckpt_config.pt")
    save_checkpoint(path, model, config=config, extra={"total_acc": 0.75})
    _model, loaded_config, extra = load_checkpoint(path)
    assert loaded_config == config
    assert extra == {"total_acc": 0.75}


# ---------------------------------------------------------------------------
# 3. cand_feature_extra=1 loads with the widened register intact.
# ---------------------------------------------------------------------------
def test_widened_cand_feature_register_survives_round_trip(tmp_path):
    config = {"dim": 10, "hidden": 12, "track": "A", "use_cand_feature": True, "cand_feature_extra": 1}
    model = build_model(config)
    resolver = model.resolver
    assert resolver.use_cand_feature is True
    assert resolver.cand_feature_extra == 1
    expected_in_dim = resolver.net[0].in_features

    path = str(tmp_path / "ckpt_widened.pt")
    save_checkpoint(path, model, config=config)
    loaded, loaded_config, _extra = load_checkpoint(path)

    assert loaded_config["cand_feature_extra"] == 1
    assert loaded.resolver.use_cand_feature is True
    assert loaded.resolver.cand_feature_extra == 1
    assert loaded.resolver.net[0].in_features == expected_in_dim
    # and the actual weights, not just the shape, made the trip:
    for p_before, p_after in zip(model.resolver.parameters(), loaded.resolver.parameters()):
        assert torch.equal(p_before, p_after)


def test_build_model_no_track_installs_no_resolver():
    model = build_model({"dim": 6, "hidden": 8, "track": None})
    assert model.resolver is None


# ---------------------------------------------------------------------------
# 4/5. End-to-end CLI round trips -- subprocess, small scale, skipped
# cleanly when quantum_parser is unavailable (mirrors
# tests/test_training_minibatch.py's own skip contract).
# ---------------------------------------------------------------------------
def _quick_setup_check() -> bool:
    from nsm_ct.input_encoder import ParserInputEncoder
    from nsm_ct.tokenizer import SimpleTokenizer
    tok = SimpleTokenizer.build(["hello world"])
    parser = ParserInputEncoder(tok)
    return getattr(parser, "_parser", None) is not None


_PARSER_AVAILABLE = _quick_setup_check()
pytestmark = pytest.mark.skipif(not _PARSER_AVAILABLE, reason="quantum_parser unavailable in this environment")

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")


def _run(args, timeout=120):
    result = subprocess.run([sys.executable] + args, cwd=_SCRIPTS_DIR,
                             capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout


def _val_total(stdout: str) -> str:
    """Just the ``val total=X.XXX`` token off the report line -- NOT the
    whole line, which also carries ``time=``/``peak_rss_mb=`` (genuinely
    different between the training run and a --load run, and even between
    two runs of the same process -- not part of the bit-for-bit metrics
    contract)."""
    for line in stdout.splitlines():
        if "val total=" in line:
            for tok in line.split():
                if tok.startswith("total="):
                    return tok
    raise AssertionError(f"no 'val total=' line in:\n{stdout}")


def test_train_instances_save_then_load_reproduces_val_total(tmp_path):
    ckpt = str(tmp_path / "instances.pt")
    save_out = _run(["train_instances.py", "--track", "A", "--episodes", "48", "--dim", "8",
                      "--hidden", "12", "--epochs", "2", "--batch-size", "16", "--threads", "2",
                      "--save", ckpt])
    assert os.path.exists(ckpt) and os.path.getsize(ckpt) > 0
    assert "saved checkpoint to" in save_out

    load_out = _run(["train_instances.py", "--track", "A", "--episodes", "48", "--dim", "8",
                      "--hidden", "12", "--threads", "2", "--load", ckpt])
    assert _val_total(save_out) == _val_total(load_out)


def test_train_writeback_save_then_load_reproduces_val_total(tmp_path):
    ckpt = str(tmp_path / "writeback.pt")
    save_out = _run(["train_writeback.py", "--track", "A", "--episodes", "48", "--dim", "8",
                      "--hidden", "12", "--epochs", "2", "--batch-size", "16", "--threads", "2",
                      "--save", ckpt])
    assert os.path.exists(ckpt) and os.path.getsize(ckpt) > 0

    load_out = _run(["train_writeback.py", "--track", "A", "--episodes", "48", "--dim", "8",
                      "--hidden", "12", "--threads", "2", "--load", ckpt])
    assert _val_total(save_out) == _val_total(load_out)


def test_save_and_load_together_errors(tmp_path):
    ckpt = str(tmp_path / "dummy.pt")
    result = subprocess.run(
        [sys.executable, "train_instances.py", "--save", ckpt, "--load", ckpt],
        cwd=_SCRIPTS_DIR, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_eval_checkpoint_runs_on_fresh_seed_and_prints_report_keys(tmp_path):
    ckpt = str(tmp_path / "for_eval.pt")
    _run(["train_instances.py", "--track", "A", "--episodes", "48", "--dim", "8", "--hidden", "12",
          "--epochs", "2", "--batch-size", "16", "--threads", "2", "--save", ckpt])

    out = _run(["eval_checkpoint.py", "--ckpt", ckpt, "--episodes", "40", "--seed", "7",
                "--batch-size", "16"])
    assert "eval_checkpoint" in out
    assert "val total=" in out
    assert "BINDING ACCURACY" in out
