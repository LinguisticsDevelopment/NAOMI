"""M57 model checkpointing (RESEARCH_NOTES M-ES1, "once checkpointing
lands"): save a trained reactor+resolver so it can be loaded back later for
FROZEN-model evaluation -- the M58 zero-shot prose test and the Spanish
Freeze Test's true frozen-model comparison both need this, and neither
existed as of this milestone (this module is plumbing only, no new
curriculum/eval logic).

Reconstruction goes through :func:`scripts._train_common.build_model` --
the SAME constructor calls scripts/train_instances.py and
scripts/train_writeback.py use to build the model in the first place (see
that function's own docstring) -- so there is exactly one code path from a
config dict to a model, not a second hand-rolled one here that could drift.

TPRCodec reconstructibility (verified against nsm_ct.tpr.TPRCodec): the
codec is NOT separately seeded. Every RNG draw inside
``TPRCodec.__post_init__``/``filler_vec``/``role_vec`` is derived from the
module-level constant ``tpr._GLOBAL_SEED = 7331`` XORed with a stable hash
of the label/relation name; ``dim`` and ``max_pos`` are the only
constructor fields that affect the result. So ``TPRCodec(dim=d)`` is
byte-identical every time it is built for the same ``d`` (any process, any
machine, this codebase unchanged) -- nothing about the codec's internal
vectors needs to be persisted; only ``dim``/``max_pos`` are recorded in
``config`` (for the record / for a future non-default ``max_pos``, not
because reconstruction needs anything else). A future change to
``tpr._GLOBAL_SEED`` or the codebook would break this guarantee for OLD
checkpoints -- out of scope for this milestone, noted for whoever touches
that constant next.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

import torch

# ``build_model`` lives in scripts/_train_common.py (script-side, not part
# of the installed nsm_ct package) -- see this module's own docstring for
# why. Add scripts/ to sys.path defensively here rather than relying on
# every caller to have done it first (train_instances.py/train_writeback.py
# already do, since they need _train_common's other helpers too, but
# eval_checkpoint.py and tests/test_checkpoint.py should not have to know
# this module's internal layout to use it correctly).
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def git_commit() -> Optional[str]:
    """``git rev-parse HEAD`` for the repo containing this file, or
    ``None`` if git isn't available / this isn't a git checkout / anything
    else goes wrong -- read-only, never raises. Recorded in a checkpoint's
    ``config`` so a saved model can be traced back to the code that
    produced it; failure to obtain it is not an error (a checkpoint saved
    outside a git checkout is still a valid checkpoint)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def save_checkpoint(path: str, model: torch.nn.Module, *, config: Dict[str, Any],
                     extra: Optional[Dict[str, Any]] = None) -> None:
    """Writes ``{state_dict, config, extra}`` to ``path`` via ``torch.save``.

    ``config`` must carry every key :func:`build_model <scripts._train_common.build_model>`
    reads to rebuild the model (``dim``, ``hidden``, ``track``,
    ``use_cand_feature``, ``cand_feature_extra``, ``evidence_prior_beta``)
    plus whatever else the caller wants on record for provenance -- codec
    ``dim``/``max_pos``, ``meaning_source``, curriculum flags, ``seed``,
    ``git_commit``, ``argv``. Extra keys beyond the reconstruction subset
    are carried through untouched; :func:`load_checkpoint` returns the
    whole dict back, ``build_model`` just ignores what it doesn't read.

    ``extra`` is free-form -- typically the run's final metrics dict
    (accuracy, binding stats, timing) -- and defaults to ``{}``.
    """
    payload = {
        "state_dict": model.state_dict(),
        "config": dict(config),
        "extra": dict(extra) if extra is not None else {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, *, map_location: str = "cpu") -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Loads a checkpoint written by :func:`save_checkpoint`.

    Rebuilds the model via ``scripts._train_common.build_model(config)`` --
    the SAME constructor path the training scripts use -- loads the state
    dict into it, and returns ``(model, config, extra)`` with
    ``model.eval()`` already called: a loaded checkpoint is for FROZEN
    evaluation (the whole point of this milestone); call ``model.train()``
    yourself first if you actually intend to resume training, which
    nothing built on this module does yet.
    """
    from _train_common import build_model  # noqa: E402 (deferred: needs scripts/ on sys.path, added above)

    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = dict(payload["config"])
    model = build_model(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, dict(payload.get("extra", {}))
