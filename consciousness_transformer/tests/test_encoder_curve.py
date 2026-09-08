"""Tests for encoder-train-arms-v2 (scripts/train_encoder.py): periodic dev
eval (--eval-every), keep-best checkpoint selection (--keep-best), and
per-family --extra-eval -- plus the regression gate that all of this is a
true no-op on the unchanged (no new flags) path.

WHY (see nsm_ct.encoder_train_util's module docstring and
scripts/run_encoder_arms.sh): comparing arms at an equal --max-steps budget
is unfair when gold files differ in derivations-per-record (v3's ~1 vs v2's
~3.3), since the SAME step budget is then a very different number of
epochs over the SAME corpus. --eval-every/--keep-best let every arm be
scored at ITS OWN best point on a dev holdout instead of wherever
--max-steps happens to land."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
GOLD_V2 = ROOT / "runs" / "encoder_gold_v2.jsonl"
HOLDOUT_FILE = ROOT / "runs" / "holdout_sentences.txt"
HOLDOUT_DEV_FILE = ROOT / "runs" / "holdout_dev_sentences.txt"
HARD_GOLD_TEST_TEMPLATE = ROOT / "runs" / "hard_gold_test_template.jsonl"
TRAIN_ENCODER = ROOT / "scripts" / "train_encoder.py"

# Pinned to the commit this session (encoder-train-arms-v2) branched from --
# the last commit before --eval-every/--keep-best/--patience/--extra-eval
# existed, used as the "before" reference for the unchanged-path
# byte-identical regression below. It is an ancestor of this branch, so it
# travels with any push/fetch of this branch regardless of whether the
# branch it was originally read from still exists.
BASELINE_COMMIT = "d28d774ce1dd32f595d97977c1216088e1c5e43b"


def _skip_if_missing():
    if not GOLD_V2.exists() or not HOLDOUT_FILE.exists():
        pytest.skip("runs/encoder_gold_v2.jsonl or runs/holdout_sentences.txt not present in this checkout")


def _run(args, timeout=300):
    cmd = [sys.executable, str(TRAIN_ENCODER)] + args
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, f"train_encoder.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return proc.stdout


def test_curve_has_expected_rows_and_keep_best_matches_argmax(tmp_path):
    _skip_if_missing()
    out = tmp_path / "curve_smoke.pt"
    _run(["--smoke", "--max-steps", "60", "--eval-every", "20", "--keep-best",
          "--holdout-file", str(HOLDOUT_FILE), "--dev-holdout-file", str(HOLDOUT_DEV_FILE),
          "--gold", str(GOLD_V2), "--beam-width", "1", "--k", "1", "--out", str(out)])

    curve_path = Path(str(out) + ".curve.tsv")
    assert curve_path.exists()
    lines = curve_path.read_text().strip().splitlines()
    header, *rows = lines
    assert header == "step\ttrain_avg_loss\tdev_rank1_f1\tdev_edge_p\tdev_edge_r\twall_s"
    assert len(rows) == 3, f"expected 3 curve rows (--max-steps 60, --eval-every 20), got {len(rows)}"
    steps = [int(r.split("\t")[0]) for r in rows]
    assert steps == [20, 40, 60]

    def _score(row: str) -> float:
        v = float(row.split("\t")[2])
        return v if v == v else float("-inf")  # NaN dev F1 never counts as the argmax

    argmax_step = int(max(rows, key=_score).split("\t")[0])

    ckpt = torch.load(str(out), map_location="cpu", weights_only=False)
    assert ckpt["config"]["best_step"] == argmax_step, (
        f"--out's config best_step ({ckpt['config']['best_step']}) must match the curve's "
        f"argmax row (step={argmax_step})")

    last_path = Path(str(out) + ".last.pt")
    assert last_path.exists(), "--keep-best must also write <out>.last.pt"
    ckpt_last = torch.load(str(last_path), map_location="cpu", weights_only=False)
    assert ckpt_last["config"]["optimizer_steps"] == 60


def test_extra_eval_reports_per_family(tmp_path):
    _skip_if_missing()
    if not HARD_GOLD_TEST_TEMPLATE.exists():
        pytest.skip("runs/hard_gold_test_template.jsonl not present in this checkout")
    out = tmp_path / "extra_smoke.pt"
    _run(["--smoke", "--max-steps", "10", "--beam-width", "1", "--k", "1",
          "--holdout-file", str(HOLDOUT_FILE), "--gold", str(GOLD_V2),
          "--extra-eval", str(HARD_GOLD_TEST_TEMPLATE), "--out", str(out)])

    ckpt = torch.load(str(out), map_location="cpu", weights_only=False)
    extra = ckpt["extra_eval"]
    assert str(HARD_GOLD_TEST_TEMPLATE) in extra
    fam_metrics = extra[str(HARD_GOLD_TEST_TEMPLATE)]
    assert "additive_focus" in fam_metrics  # a known family in this file (see meta.family)
    assert len(fam_metrics) >= 5  # runs/hard_gold_test_template.jsonl has 8 families
    for fam, m in fam_metrics.items():
        assert m["n"] > 0, f"family {fam} has 0 records"
        assert "rank1_edge_f1" in m


def test_unchanged_path_byte_identical_to_pre_train_arms_v2(tmp_path):
    """(item E) Without --eval-every/--keep-best/--patience/--extra-eval,
    training must behave EXACTLY as it did before this module existed --
    same per-step/per-epoch progress lines (avg_loss is a deterministic
    function of the fixed seed + control flow on CPU) -- proving the
    on_optimizer_step hook is a true no-op when the caller never wires it
    up (etu.run_training_loop's on_optimizer_step=None default)."""
    _skip_if_missing()

    show = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:consciousness_transformer/scripts/train_encoder.py"],
        cwd=str(ROOT.parent), capture_output=True, text=True)
    if show.returncode != 0 or not show.stdout.strip():
        pytest.skip(f"cannot read baseline scripts/train_encoder.py from commit {BASELINE_COMMIT} "
                     "(not present in this checkout's git history)")
    baseline_script = tmp_path / "baseline_train_encoder.py"
    baseline_script.write_text(show.stdout)

    # The baseline script computes its own --usvs-dir default from its OWN
    # (tmp_path) location, which is wrong once copied out -- pass the real
    # one explicitly. PYTHONPATH covers the `sys.path.insert` the same way
    # (harmless, redundant, for the current script).
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    common_args = ["--smoke", "--n-train", "12", "--n-dev", "5", "--n-test", "5",
                   "--seed", "3", "--beam-width", "1", "--k", "1",
                   "--gold", str(GOLD_V2), "--usvs-dir", str(ROOT / "data" / "usvs")]

    def run_and_extract_progress(script_path: Path, out: Path) -> list:
        cmd = [sys.executable, str(script_path)] + common_args + ["--out", str(out)]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300, env=env)
        assert r.returncode == 0, f"{script_path} failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        progress = []
        for line in r.stdout.splitlines():
            body = line.split("] ", 1)[-1] if line.startswith("[") else line
            if body.startswith("epoch ") or body.startswith("=== epoch "):
                progress.append(body)  # strips the "[  12.3s]" wall-clock prefix only
        return progress

    baseline_progress = run_and_extract_progress(baseline_script, tmp_path / "baseline_out.pt")
    current_progress = run_and_extract_progress(TRAIN_ENCODER, tmp_path / "current_out.pt")

    assert baseline_progress, "baseline run produced no epoch-progress lines to compare"
    assert current_progress == baseline_progress
