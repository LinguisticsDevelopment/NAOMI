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

M59a (episodic LTM, CLAUDE.md's "LTM decisions"): this module also hosts
:class:`DocumentRunner`, the wind-down/consolidate substate machine
(dev/OP_INVENTORY.md's "wind-down / consolidate" row) that drives
multi-passage documents through :class:`~nsm_ct.clause_reactor.
ClauseReactor` with a persisted LTM tensor -- see :mod:`nsm_ct.ltm`'s
module docstring for the full document/passage/registry contract, and
:class:`DocumentRunner`'s own docstring for the substate machine itself.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# M59a: the wind-down/consolidate substate machine (episodic LTM).
# ---------------------------------------------------------------------------
class DocumentRunner:
    """Drives ONE document (an ordered list of per-passage
    :class:`~nsm_ct.clause_reactor.ClauseBatch`\\ es, already grouped/built
    by the caller -- see :mod:`nsm_ct.ltm`'s module docstring, "Interface
    contract for the curriculum agent", for the full document/passage/
    registry contract) through a small, explicit substate machine per
    passage:

    - **READING**: run :meth:`nsm_ct.clause_reactor.ClauseReactor.forward`
      on this passage's batch, with the document's LTM tensor (accumulated
      from every EARLIER passage's consolidation, ``None`` for passage 0)
      threaded in as ``ltm=`` -- every read this passage's forward pass
      performs is therefore additive over STM+LTM (:func:`nsm_ct.ltm.
      mem_total`), while every write still lands only in this passage's own
      fresh STM.
    - **WIND_DOWN**: a NAMED event, fired unconditionally at the end of
      every passage's reading -- a v1 NO-OP (recorded in the passage's
      report under ``"events"``), the hook for a future ``patience`` dial
      (dev/OP_INVENTORY.md's dials table: "not set" today) that would let a
      passage's reading run stop early/late based on the controller's own
      state instead of always running every step of the batch.
    - **CONSOLIDATE**: :func:`nsm_ct.ltm.promote` copies this passage's OWN
      new provenance records (this passage's STM writes, gated by
      ``trust_ltm``) into the document's LTM tensor -- ``dial_name=
      "trust_ltm"`` (the tier tag LTM->Truth will later reuse this SAME
      function with ``dial_name="trust_truth"`` for). The registry and log
      passed to :meth:`run_document` PERSIST across every passage of the
      document (constructed once by the caller, per the interface
      contract) -- this is what lets a later passage's instance ids
      resolve against an EARLIER passage's minted atoms.

    Works identically for training (``train=True`` -- gradients flow,
    ``loss_fn`` accumulates a ``total_loss`` SUMMED over passages, a
    single tensor a script calls ``.backward()`` on once after the whole
    document) and eval (``train=False`` -- every passage's forward pass
    runs under ``torch.no_grad()``, ``loss_fn`` is typically omitted).
    ``run_document`` returns ``(reports, total_loss)`` -- see this class's
    module (:mod:`nsm_ct.ltm`'s docstring) for the exact per-passage report
    key list; a script computing per-passage answer accuracy reads
    ``report["out"]["answer_logits"]`` against that passage's own
    ``batch.answer``.
    """

    def __init__(self, model, *, trust_ltm: Optional[float] = None,
                 ltm_detach: Optional[bool] = None) -> None:
        # Deferred import (not module-level): mirrors build_model's own
        # "keeps this module importable before src/ is on sys.path" seam.
        from nsm_ct.ltm import LTM_DETACH, TRUST_LTM

        self.model = model
        self.trust_ltm = TRUST_LTM if trust_ltm is None else trust_ltm
        # M59a: detach the LTM tensor between passages -- bounds autograd.
        # Without this, a document's Nth passage's backward pass would walk
        # through EVERY earlier passage's promote() write (each one itself
        # built from that passage's own forward pass's full order-3 STM
        # tensor), so the retained graph would grow with the NUMBER OF
        # PASSAGES CONSOLIDATED SO FAR on top of the existing per-step
        # B*T*d^3 footprint this module's own docstring already flags as
        # the dominant training-memory cost (RESEARCH_NOTES tail,
        # dev/LTM_DESIGN_BRIEF.md Sec.2's capacity-probe note: "an LTM
        # tensor that accumulates across MANY episodes inside one training
        # run is a second, likely worse instance of the same footprint
        # problem"). Detaching ``ltm`` after each passage's CONSOLIDATE
        # step cuts the graph there: the NEXT passage's forward pass reads
        # ``ltm`` as a plain (non-differentiable) tensor value, so ITS
        # backward pass only ever walks that ONE passage's own STM
        # computation -- the model's SHARED weights still receive a
        # gradient from every passage's own loss (nothing about the
        # parameters is detached, only the LTM VALUE passed between
        # passages), so training still learns from the whole document, at
        # bounded (not passage-count-multiplied) peak memory.
        self.ltm_detach = LTM_DETACH if ltm_detach is None else ltm_detach

    def run_document(self, passages: Sequence, registry, log, codec, *,
                      train: bool, loss_fn: Optional[Callable] = None,
                      source: str = "reactor", language: str = "en",
                      timestamp_base: float = 0.0) -> Tuple[List[dict], Optional["torch.Tensor"]]:
        """Run every passage of ONE document in order. ``passages`` is the
        already-ordered, already-grouped list of that document's per-
        passage ``ClauseBatch``es (see this module's own docstring and
        :mod:`nsm_ct.ltm`'s "Interface contract" -- grouping episodes by
        ``doc_id``/``passage_index`` into this list is the CALLER's job,
        not this method's). ``registry``/``log`` are the ONE
        :class:`~nsm_ct.instances.InstanceRegistry`/:class:`~nsm_ct.
        instances.ProvenanceLog` pair for this document, constructed by
        the caller BEFORE the first passage and reused across every
        passage (so instance ids persist). ``codec`` is the
        :class:`~nsm_ct.tpr.TPRCodec` used to build the passages'
        batches (needed by :func:`nsm_ct.ltm.promote` to ground each
        promoted relation's filler vector).

        Each passage's batch may carry ``B > 1`` rows -- every row in
        EVERY passage of this call shares the SAME ``registry``/``log``
        (instance ids are looked up by string, so this is only correct
        when every row genuinely belongs to the one document these
        registry/log objects were built for; a training script processing
        MANY independent documents in parallel calls ``run_document`` once
        PER document, each with its own registry/log -- registries are
        document-scoped by construction, see :mod:`nsm_ct.instances`'s own
        "unbatched by design" module docstring).

        ``loss_fn(out, batch) -> Tensor`` (optional), called once per
        passage under the SAME grad/no-grad context as that passage's
        forward pass; its return values are summed into ``total_loss``
        (``None`` when ``loss_fn`` is omitted -- the eval-only path).
        Returns ``(reports, total_loss)`` -- see :mod:`nsm_ct.ltm`'s module
        docstring for ``reports``' exact per-passage key list.
        """
        from nsm_ct import provenance as provenance_mod
        from nsm_ct.ltm import promote

        self.model.train(train)
        ltm_tensor = None   # [B, d, d, d] or None before the first CONSOLIDATE
        reports: List[dict] = []
        total_loss = None

        for p_idx, batch in enumerate(passages):
            grad_ctx = torch.enable_grad() if train else torch.no_grad()
            with grad_ctx:
                out = self.model(batch, ltm=ltm_tensor, return_write_trace=True,
                                  return_memory=True, return_mem_read=True)
                passage_loss = loss_fn(out, batch) if loss_fn is not None else None
            if passage_loss is not None:
                total_loss = passage_loss if total_loss is None else total_loss + passage_loss

            events = ["reading", "wind_down"]   # WIND_DOWN: v1 no-op, see class docstring

            b = batch.entity.shape[0]
            final_memory = out["_memory"]        # [B, d, d, d] -- this passage's own STM
            passage_timestamp = timestamp_base + p_idx
            new_ltm_rows = []
            n_records_total = 0
            n_promoted_total = 0
            for i in range(b):
                # Per-row provenance for THIS passage only (never the whole
                # historical log) -- record_writes is called on a ONE-ROW
                # subset so ``records=`` below is exactly this row's new
                # writes, matching promote()'s "records from the passage"
                # contract.
                row_idx = torch.tensor([i])
                row_batch = batch.subset(row_idx)
                row_trace = {k: v[i:i + 1] for k, v in out["_write_trace"].items()}
                start = len(log)
                provenance_mod.record_writes(
                    row_batch, {"_write_trace": row_trace}, log,
                    source=source, language=language, timestamp_base=passage_timestamp)
                row_records = log.records[start:]
                n_records_total += len(row_records)

                target_i = ltm_tensor[i] if ltm_tensor is not None else torch.zeros_like(final_memory[i])
                new_target_i, n_i = promote(
                    final_memory[i], target_i, registry, log,
                    records=row_records, dial=self.trust_ltm, dial_name="trust_ltm",
                    codec=codec, timestamp=passage_timestamp,
                )
                new_ltm_rows.append(new_target_i)
                n_promoted_total += n_i

            new_ltm = torch.stack(new_ltm_rows, dim=0)
            events.append("consolidate")
            if self.ltm_detach:
                new_ltm = new_ltm.detach()
            ltm_tensor = new_ltm

            reports.append({
                "passage_index": p_idx,
                "out": out,
                "ltm": ltm_tensor,
                "n_records": n_records_total,
                "n_promoted": n_promoted_total,
                "events": events,
                "loss": passage_loss,
            })

        return reports, total_loss
