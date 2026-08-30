"""Shared minibatching + memory-footprint instrumentation
(M57c battery #1 footprint fix, RESEARCH_NOTES tail): a full-scale arm
(1500 episodes, dim 48, 60 epochs) ran ONE forward/backward over the
WHOLE training set every epoch. The per-step memory tensor is
order-3 -- ``[B, d, d, d]`` -- and autograd keeps every step's copy alive
(T up to ~14), so footprint is proportional to B*T*d^3; at B=1500,
dim=48 that is ~5-8GB RSS per arm, and 4 parallel arms OOM'd a 15GB
cloud box. This module factors out the minibatch shuffle/aggregate
logic so scripts/train_instances.py, scripts/train_writeback.py, and
scripts/train_resolver.py share ONE implementation rather than three
copies of the same shuffle-and-stitch code.

``--batch-size 0`` means full-batch -- the PRE-FIX behavior -- kept
reachable (rather than removed) so the old and new code paths can be
measured against each other from the same script, per CLAUDE.md's ops
rule against unverified "should be faster" claims.
"""

from __future__ import annotations

import resource

import numpy as np
import torch


def add_footprint_args(ap) -> None:
    """Adds ``--batch-size`` and ``--threads`` to an argparse parser.
    ``--batch-size 0`` = full-batch (one forward/backward over the entire
    train/val set per epoch, the behavior every training script had before
    this fix -- kept reachable for before/after measurement, NOT just as a
    historical note). ``--threads`` defaults to ``None`` (leave torch's own
    default untouched); on a shared cloud box, set this -- and the
    ``OMP_NUM_THREADS`` environment variable -- to at most the box's core
    count, since parallel arms otherwise each try to claim every core and
    starve each other (RESEARCH_NOTES: 4 parallel arms already OOM a
    4-core/15GB box on memory alone; thread contention compounds it)."""
    ap.add_argument("--batch-size", type=int, default=64,
                     help="Minibatch size for both training and evaluation steps. 0 = full-batch "
                          "(the pre-footprint-fix behavior: one forward/backward over the whole "
                          "train/val set per epoch). Default 64 keeps the order-3 memory tensor's "
                          "autograd footprint (~ batch_size * T * dim^3, not episodes * T * dim^3) "
                          "off the full-dataset scale.")
    ap.add_argument("--threads", type=int, default=None,
                     help="torch.set_num_threads(N). Default: leave torch's own default untouched. "
                          "On a shared cloud box, set this (and OMP_NUM_THREADS in the environment) "
                          "<= the box's core count, especially when running more than one arm at once.")


def apply_threads(args) -> None:
    """Call once, after argparse, before building any model/tensor."""
    if getattr(args, "threads", None) is not None:
        torch.set_num_threads(args.threads)


def epoch_minibatches(n: int, batch_size, seed: int, epoch: int):
    """Yields ``np.ndarray`` index arrays covering ``range(n)`` exactly
    once, for one epoch of minibatched training. ``batch_size <= 0`` (or
    ``None``, or ``>= n``) yields a single unshuffled ``arange(n)`` --
    the full-batch behavior, byte-identical to the pre-fix training loop
    (no shuffle applied, since there was never more than one "batch" to
    shuffle the order of). Otherwise shuffles a fresh
    ``RandomState(seed, epoch)`` permutation of ``range(n)`` and slices it
    into ``batch_size``-sized chunks (the last chunk may be smaller) --
    deterministic given ``(n, batch_size, seed, epoch)``, and different
    across epochs so minibatch composition varies run-to-run the way SGD
    expects."""
    idx = np.arange(n)
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        yield idx
        return
    rng = np.random.RandomState((seed * 1_000_003 + epoch) % (2 ** 31 - 1))
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]


def eval_minibatched(model, batch, batch_size) -> dict:
    """Runs ``model(batch)`` under ``torch.no_grad()`` in minibatches of
    ``batch_size`` (via ``batch.subset``), stitching each returned
    per-row tensor back into a full-length tensor in ORIGINAL row order --
    so the returned dict is the value-for-value equivalent of one
    full-batch ``model(batch)`` call (only peak memory differs, since no
    single forward pass ever materializes the whole dataset's order-3
    memory tensor at once). ``batch_size <= 0`` (or ``None``, or
    ``>= len(batch)``) runs the single full-batch forward pass directly --
    the pre-fix behavior. Skips any non-tensor entry in the model's output
    dict (there are none as of this writing, but this stays robust either
    way)."""
    n = batch.entity.shape[0]
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        with torch.no_grad():
            return model(batch)
    buffers: dict = {}
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = torch.arange(start, end)
        sub = batch.subset(idx)
        with torch.no_grad():
            out = model(sub)
        for k, v in out.items():
            if not torch.is_tensor(v):
                continue
            if k not in buffers:
                buffers[k] = torch.empty((n,) + tuple(v.shape[1:]), dtype=v.dtype)
            buffers[k][start:end] = v
    return buffers


def build_model(config: dict):
    """Builds a :class:`~nsm_ct.clause_reactor.ClauseReactor` (with its
    resolver installed, per ``config``) via the SAME constructor calls
    scripts/train_instances.py's and scripts/train_writeback.py's own
    ``run_arm`` already used inline before M57 checkpointing -- both scripts
    now call THIS function instead, and
    ``nsm_ct.checkpoint.load_checkpoint`` reconstructs a frozen model
    through it too, so there is exactly ONE code path from a config dict to
    a model (not a second hand-rolled one at load time that could drift
    from the training-time construction).

    Reads:
      - ``dim`` (required), ``hidden`` (default 128)
      - ``track`` (``"A"`` | ``"B"`` | falsy/absent -- no resolver
        installed, matching every script's own ``if track else None``)
      - ``use_cand_feature`` (default ``False``), ``cand_feature_extra``
        (default ``0``) -- forwarded to ``make_resolver`` (Track A only;
        ignored for Track B, same contract ``make_resolver`` itself has)
      - ``evidence_prior_beta`` (default ``None``)

    Any other config key (codec dim/max_pos, meaning_source, curriculum
    flags, seed, git commit, argv, ...) is ignored here -- this function
    builds only the ``nn.Module``, not the data pipeline around it.
    Construction order (resolver, THEN the reactor) matches every pre-M57
    call site exactly, so a fixed ``torch.manual_seed`` before calling this
    reproduces the same initial weights a training script's own inline
    construction did before this refactor.
    """
    # Deferred import (not module-level): keeps this module importable
    # before a caller has put ``src/`` on ``sys.path`` (every call site
    # does so before it actually CALLS this function, not necessarily
    # before it imports this module).
    from nsm_ct.clause_reactor import ClauseReactor
    from nsm_ct.resolver import make_resolver

    dim = config["dim"]
    hidden = config.get("hidden", 128)
    track = config.get("track")
    resolver = None
    if track:
        resolver = make_resolver(track, dim, hidden,
                                  use_cand_feature=config.get("use_cand_feature", False),
                                  cand_feature_extra=config.get("cand_feature_extra", 0))
    return ClauseReactor(dim=dim, hidden=hidden, resolver=resolver,
                          evidence_prior_beta=config.get("evidence_prior_beta"))


def peak_rss_mb() -> float:
    """Peak resident set size in MB for this process (RUSAGE_SELF) since
    it started. ``ru_maxrss`` is reported in KB on Linux (the target
    platform for the cloud training routine) -- this does NOT correct for
    macOS's bytes convention, since every training run this instruments
    happens on the Linux cloud box."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
