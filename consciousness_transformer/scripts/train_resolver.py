"""Resolver training script: M53's pronoun collapse (Track A CorefHead vs
Track B SharedScorer vs --gold-binding, the M53a placeholder ceiling) EXTENDED
for M54's sense collapse (dev/RESOLVER_BUILD_PLAN.md Phase 3): the SAME two
tracks now ALSO resolve homograph senses in the same forward pass (Track A
gets a SECOND, sense-specialist head -- SenseHead; Track B reuses its ONE
SharedScorer instance for both candidate kinds, ``resolver is sense_resolver``
literally), plus a new --mfs-floor arm (the M30 baseline: always bind the
most-frequent sense, no resolver) alongside the pre-existing --gold-binding
ceiling. Follows scripts/probe_m52_transfer.py's arm pattern.

Curriculum mix:
    --mix m53 (M53's original): 1/2 old L1-6 + 1/4 transfer + 1/4 pronoun.
    --mix m54 (default): 40% old L1-6 + 20% transfer + 20% pronoun + 20%
        ambiguity (episode.generate_ambiguity_episodes, M32).

Reports: val accuracy overall + per curriculum kind (old/transfer_*/
pronoun_binding/ambiguity), RESOLVER BINDING ACCURACY (pronoun antecedent,
M53's own metric) overall + anti-recency half vs the scripted nearest-entity
baseline, SENSE BINDING ACCURACY (M54, new) overall + the sense-FLIPPED half
(gold_sense != mfs_sense, the M32 metric) vs the MFS floor (0.000 on the
flipped half BY CONSTRUCTION -- MFS is always candidate index 0, and
"flipped" means gold != mfs), both resolvers' param counts, and both
margin distributions.

Usage:
    python scripts/train_resolver.py --track A
    python scripts/train_resolver.py --track B
    python scripts/train_resolver.py --gold-binding      # ceiling: gold sense + gold antecedent, no resolver
    python scripts/train_resolver.py --mfs-floor         # floor: MFS sense (+ gold antecedent), no resolver
    python scripts/train_resolver.py --track A --mix m53 # M53 reproduction (no ambiguity episodes)

M54c (RESEARCH_NOTES M54c) adds four diagnosis arms unpicking three
confounds in the A-vs-B history (Track A's two specialist heads beat Track
B's one shared scorer three rounds straight, but B fights at ~40% of A's
combined capacity, B's state-input was implicated in M53's task-accuracy
damage, and B has never been distilled from A's already-learned behavior).
All four still install ONE :class:`~nsm_ct.resolver.SharedScorer` instance
across pronoun and sense candidates (``resolver is sense_resolver``, same as
plain ``--track B``) -- see :func:`build_b_family_resolver` and
:func:`run_distilled_arm`:
    --track B-wide          capacity-matched: mlp_hidden picked so total
                             params ~= CorefHead+SenseHead combined at this
                             dim (see nsm_ct.resolver.shared_scorer_for_budget).
    --track B-nostate       drops the controller-state input (use_state=False),
                             params matched to the ORIGINAL narrow B.
    --track B-nostate-wide  no state input AND capacity-matched wide -- the
                             2x2 {state,no-state} x {narrow,wide} cell that
                             isolates architecture from capacity.
    --track B-distilled     stage 1 trains Track A normally; stage 2 inits a
                             fresh B-family resolver (--distill-b-track,
                             default B-nostate-wide) and trains it with task
                             + aux losses PLUS KL(B's candidate softmax ||
                             frozen Track-A's per-type candidate softmax) on
                             the same batches; stage 3 drops the distillation
                             term for a brief task-only fine-tune. Stage
                             epoch counts are configurable
                             (--distill-stage{1,2,3}-epochs, default to
                             --epochs / --epochs / --epochs//4).

Run both --mix m53 (pronoun-critical) and --mix m54b (sense-critical) for
each new arm -- the diagnosis needs both capabilities measured.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

torch.set_num_threads(1)

from nsm_ct.clause_reactor import ClauseReactor, build_clause_batch  # noqa: E402
from nsm_ct.curriculum2 import (  # noqa: E402
    _FEMALE_NAMES,
    _MALE_NAMES,
    association_only_baseline,
    garden_path_association_baseline,
    garden_path_parser_top1_baseline,
    generate_garden_path_episodes,
    generate_pronoun_episodes,
    generate_sense_binding_episodes,
    generate_transfer_episodes,
    nearest_entity_baseline,
)
from nsm_ct.episode import CurriculumGenerator, generate_ambiguity_episodes, split_episodes  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.resolver import (  # noqa: E402
    CorefHead,
    SenseHead,
    make_hyp_resolver,
    make_resolver,
    make_sense_resolver,
    shared_scorer_for_budget,
)
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402

AUX_WEIGHT = 0.5   # resolver cross-entropy weight added to the answer loss (training-script side, not the model)
DISTILL_WEIGHT = 1.0   # M54c: default weight on stage 2's B-vs-frozen-A KL term (--distill-weight overrides)

B_FAMILY_TRACKS = ("B-WIDE", "B-NOSTATE", "B-NOSTATE-WIDE")   # M54c: all install ONE shared instance, like "B"


def a_combined_params(dim: int) -> int:
    """Track A's combined capacity at this ``dim`` -- CorefHead + SenseHead,
    the two INDEPENDENT specialist heads plain ``--track A`` installs. M54c's
    "capacity-matched" B arms (B-wide, B-nostate-wide) target this number via
    :func:`nsm_ct.resolver.shared_scorer_for_budget` rather than a hardcoded
    constant, so the match stays correct at whatever ``--dim`` a run uses
    (the M53/M54b gate runs use dim=48; the M54c smokes use dim=32 -- the
    combined-capacity target is genuinely different at each)."""
    return (sum(p.numel() for p in CorefHead(dim).parameters()) +
            sum(p.numel() for p in SenseHead(dim).parameters()))


def build_b_family_resolver(track: str, dim: int, hidden: int):
    """M54c dispatch for the three capacity/architecture diagnosis arms
    (RESEARCH_NOTES M54c) -- always returns a single :class:`SharedScorer`
    instance (the caller installs it as BOTH ``resolver=`` and
    ``sense_resolver=``, mirroring plain ``--track B``'s
    ``resolver is sense_resolver``):
        B-WIDE          use_state=True,  mlp_hidden picked for
                         :func:`a_combined_params` (Track A's combined
                         capacity) -- capacity-matched, architecture
                         unchanged from the original B.
        B-NOSTATE       use_state=False, mlp_hidden picked to match the
                         ORIGINAL narrow B's own param count
                         (``SharedScorer(dim, hidden)``) -- architecture
                         changed, capacity held at B's original (small) scale.
        B-NOSTATE-WIDE  use_state=False, mlp_hidden picked for
                         :func:`a_combined_params` -- both changes at once,
                         the cell that isolates whether dropping state costs
                         anything once B is no longer capacity-starved.
    """
    t = track.strip().upper()
    if t == "B-WIDE":
        return shared_scorer_for_budget(dim, hidden, a_combined_params(dim), use_state=True)
    if t == "B-NOSTATE":
        narrow_target = sum(p.numel() for p in make_resolver("B", dim, hidden).parameters())
        return shared_scorer_for_budget(dim, hidden, narrow_target, use_state=False)
    if t == "B-NOSTATE-WIDE":
        return shared_scorer_for_budget(dim, hidden, a_combined_params(dim), use_state=False)
    raise ValueError(f"not a B-family track: {track!r}, expected one of {B_FAMILY_TRACKS}")


def build_resolvers(track: str, dim: int, hidden: int):
    """Track dispatch shared by :func:`run_arm` and :func:`run_distilled_arm`:
    "A" -> THREE INDEPENDENT specialist heads (CorefHead, SenseHead,
    M55b's RankHead); "B" and every M54c B-family track -> ONE
    :class:`SharedScorer` instance installed as all three slots (Track B's
    "one shared collapse mechanism" bet extends to the garden-path
    candidate kind with zero new code -- same reasoning M54 already applied
    to senses). Returns ``(resolver, sense_resolver, hyp_resolver)``.
    ``hyp_resolver`` is inert (never fires) on any batch without a
    garden-path episode -- exactly like ``sense_resolver`` already is on a
    batch with no ambiguity episodes -- so building it unconditionally
    changes nothing for ``--mix`` values that don't include garden-path
    episodes."""
    t = track.strip().upper()
    if t == "A":
        return (make_resolver("A", dim, hidden), make_sense_resolver("A", dim, hidden),
                make_hyp_resolver("A", dim, hidden))
    if t == "B":
        r = make_resolver("B", dim, hidden)
        return r, r, r
    if t in B_FAMILY_TRACKS:
        r = build_b_family_resolver(t, dim, hidden)
        return r, r, r
    raise ValueError(f"unknown track {track!r}")


def aux_loss_terms(out, batch, resolver, sense_resolver, hyp_resolver=None):
    """The resolver-aux cross-entropy terms (pronoun + sense + M55b's
    reading), each weighted by ``AUX_WEIGHT`` and summed -- factored out of
    :func:`run_arm`'s training loop so :func:`run_distilled_arm`'s three
    stages can reuse the exact same aux-loss definition without duplicating
    the masking logic. Returns ``0.0`` (a plain float, safe to add to a loss
    tensor) if no candidate kind is present in ``batch``. ``hyp_resolver``
    defaults ``None`` so every pre-M55b call site (there are none left in
    this file, but the default keeps the function callable the old way)
    stays valid."""
    total = 0.0
    if resolver is not None and "resolver_logits" in out:
        cg = batch.cand_gold
        has = cg >= 0
        if bool(has.any()):
            total = total + AUX_WEIGHT * F.cross_entropy(out["resolver_logits"][has], cg[has])
    if sense_resolver is not None and "sense_resolver_logits" in out:
        scg = batch.sense_cand_gold
        has = scg >= 0
        if bool(has.any()):
            total = total + AUX_WEIGHT * F.cross_entropy(out["sense_resolver_logits"][has], scg[has])
    if hyp_resolver is not None and "hyp_resolver_logits" in out:
        hg = batch.hyp_cand_gold
        has = hg >= 0
        if bool(has.any()):
            total = total + AUX_WEIGHT * F.cross_entropy(out["hyp_resolver_logits"][has], hg[has])
    return total


def _distill_kl(student_logits, teacher_logits, has_cand):
    """KL(B's candidate softmax || A's frozen candidate softmax) for ONE
    candidate kind, over rows with a real candidate set, averaged (M54c
    distillation term). Computed by hand rather than ``F.kl_div`` to keep the
    direction unambiguous: with ``p`` = student (B) softmax and ``q`` =
    teacher (A, frozen, detached) softmax, this is ``sum(p * (log p - log
    q))`` -- literally KL(B || A), the direction RESEARCH_NOTES M54c
    specifies. Padded candidate slots carry logit -1e9 (masked upstream by
    :meth:`~nsm_ct.clause_reactor.ClauseReactor._collapse`), so their
    softmax probability is exactly 0.0 in float32 and contributes exactly
    0.0 here (0.0 * finite, not 0 * inf -- no NaN risk)."""
    if not bool(has_cand.any()):
        return None
    s = student_logits[has_cand]
    t = teacher_logits[has_cand].detach()
    p = F.softmax(s, dim=-1)
    log_p = F.log_softmax(s, dim=-1)
    log_q = F.log_softmax(t, dim=-1)
    return (p * (log_p - log_q)).sum(-1).mean()


def distill_loss_terms(student_out, teacher_out, batch):
    """Sum of all per-type KL terms (pronoun-candidate rows, sense-candidate
    rows, M55b reading-candidate rows) present in ``batch`` -- ``0.0`` if no
    kind is present (e.g. an old-only batch). Mirrors :func:`aux_loss_terms`'s
    per-kind masking."""
    total = 0.0
    if "resolver_logits" in student_out and "resolver_logits" in teacher_out:
        kl = _distill_kl(student_out["resolver_logits"], teacher_out["resolver_logits"],
                          batch.cand_gold >= 0)
        if kl is not None:
            total = total + kl
    if "sense_resolver_logits" in student_out and "sense_resolver_logits" in teacher_out:
        kl = _distill_kl(student_out["sense_resolver_logits"], teacher_out["sense_resolver_logits"],
                          batch.sense_cand_gold >= 0)
        if kl is not None:
            total = total + kl
    if "hyp_resolver_logits" in student_out and "hyp_resolver_logits" in teacher_out:
        kl = _distill_kl(student_out["hyp_resolver_logits"], teacher_out["hyp_resolver_logits"],
                          batch.hyp_cand_gold >= 0)
        if kl is not None:
            total = total + kl
    return total


def build_mixed_curriculum(n_episodes: int, seed: int):
    """M53's original mix: 1/2 old L1-6 + 1/4 transfer + 1/4 pronoun,
    deterministic given (n_episodes, seed). Kept for --mix m53 (exact M53
    reproduction, no ambiguity episodes at all)."""
    n_pronoun = n_episodes // 4
    n_transfer = n_episodes // 4
    n_old = n_episodes - n_pronoun - n_transfer
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    transfer = generate_transfer_episodes(n_transfer, seed=seed + 1)
    pronoun = generate_pronoun_episodes(n_pronoun, seed=seed + 2)
    episodes = old + transfer + pronoun
    order = np.random.RandomState(seed + 3).permutation(len(episodes))
    return [episodes[i] for i in order]


def build_m54_curriculum(n_episodes: int, seed: int):
    """M54's mix (RESOLVER_BUILD_PLAN.md Phase 3, "Curriculum mix"): 40% old
    L1-6 + 20% transfer + 20% pronoun + 20% ambiguity, deterministic given
    (n_episodes, seed). Ambiguity episodes get ``meta["kind"] = "ambiguity"``
    set here (episode.generate_ambiguity_episodes doesn't set "kind" itself
    -- every other generator does) purely so the per-kind task-accuracy
    report below groups them under a real label instead of the "old"
    fallback; this never touches episode.py."""
    n_ambiguity = n_episodes // 5
    n_transfer = n_episodes // 5
    n_pronoun = n_episodes // 5
    n_old = n_episodes - n_ambiguity - n_transfer - n_pronoun
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    transfer = generate_transfer_episodes(n_transfer, seed=seed + 1)
    pronoun = generate_pronoun_episodes(n_pronoun, seed=seed + 2)
    ambiguity = generate_ambiguity_episodes(n_ambiguity, seed=seed + 3)
    for e in ambiguity:
        e.meta.setdefault("kind", "ambiguity")
    episodes = old + transfer + pronoun + ambiguity
    order = np.random.RandomState(seed + 4).permutation(len(episodes))
    return [episodes[i] for i in order]


def build_m54b_curriculum(n_episodes: int, seed: int):
    """M54b's mix (RESEARCH_NOTES M54b): 50% old L1-6 + 50% the NEW
    entity-keyed, binding-critical sense curriculum
    (curriculum2.generate_sense_binding_episodes) -- deliberately heavy in
    the new kind (unlike --mix m54's 20% old-M32-ambiguity slice) so the
    gold-ceiling-vs-MFS-floor gap probe has enough of the new signal to
    measure cleanly. No transfer/pronoun episodes here on purpose -- this
    mix isolates the ONE new capability M54b adds; --mix m54 already covers
    the full curriculum-mix regression. Deterministic given
    (n_episodes, seed). Episodes get ``meta["kind"] = "sense_binding"``
    already (the generator sets it itself, unlike the M32 generator)."""
    n_new = n_episodes // 2
    n_old = n_episodes - n_new
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    new = generate_sense_binding_episodes(n_new, seed=seed + 1)
    episodes = old + new
    order = np.random.RandomState(seed + 2).permutation(len(episodes))
    return [episodes[i] for i in order]


def build_m55a_curriculum(n_episodes: int, seed: int):
    """M55a's mix (RESEARCH_NOTES M55a, dev/TRACK_C_DESIGN.md Sec 1.10): 50%
    old L1-6 + 50% the NEW garden-path curriculum
    (curriculum2.generate_garden_path_episodes) -- mirrors
    :func:`build_m54b_curriculum`'s exact "isolate the one new capability"
    reasoning: heavy in the new kind, no transfer/pronoun/sense episodes,
    since --mix m54/m54b already cover the full regression. Deterministic
    given (n_episodes, seed). M55a itself trains NO resolver (placeholder
    mode, like M53a before M53b existed) -- this mix exists so a smoke run
    can exercise the garden-path batch-build/collapse plumbing end to end
    (see scripts/probe_garden_path_smoke.py) and so a FUTURE resolver-
    training round has a ready-made mix to consume without a second
    data-plumbing pass."""
    n_new = n_episodes // 2
    n_old = n_episodes - n_new
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    new = generate_garden_path_episodes(n_new, seed=seed + 1)
    episodes = old + new
    order = np.random.RandomState(seed + 2).permutation(len(episodes))
    return [episodes[i] for i in order]


def build_m55_curriculum(n_episodes: int, seed: int):
    """M55b's mix (RESEARCH_NOTES M55b): 40%% old L1-6 + 20%% pronoun + 20%%
    sense-binding (M54b's ``curriculum2.generate_sense_binding_episodes``)
    + 20%% garden-path (M55b's redesigned
    ``curriculum2.generate_garden_path_episodes``) -- the FULL-MEMBRANE mix:
    all three collapse types (pronoun antecedent, sense, parse-hypothesis
    reading) training in the SAME batch for the first time, unlike m54/m54b/
    m55a which each isolate one new capability at a time. Deterministic
    given (n_episodes, seed)."""
    n_pronoun = n_episodes // 5
    n_sense = n_episodes // 5
    n_garden = n_episodes // 5
    n_old = n_episodes - n_pronoun - n_sense - n_garden
    old = CurriculumGenerator(max_level=6, seed=seed).generate(n_old)
    pronoun = generate_pronoun_episodes(n_pronoun, seed=seed + 1)
    sense = generate_sense_binding_episodes(n_sense, seed=seed + 2)
    garden = generate_garden_path_episodes(n_garden, seed=seed + 3)
    episodes = old + pronoun + sense + garden
    order = np.random.RandomState(seed + 4).permutation(len(episodes))
    return [episodes[i] for i in order]


def resolver_binding_stats(out, batch, va_eps):
    """RESOLVER BINDING ACCURACY overall + anti-recency half, and the raw margin
    list, read off one resolver-carrying forward pass + the episodes' meta."""
    if "resolver_logits" not in out:
        return None
    cand_gold = batch.cand_gold
    has_cand = cand_gold >= 0
    pred_idx = out["resolver_logits"].argmax(-1)
    correct = (pred_idx == cand_gold) & has_cand
    margins = []
    hits = anti_hits = anti_total = total = 0
    for i, e in enumerate(va_eps):
        row_mask = has_cand[i]
        if not bool(row_mask.any()):
            continue
        t = int(row_mask.nonzero()[0, 0])
        total += 1
        is_hit = bool(correct[i, t])
        hits += int(is_hit)
        margins.append(float(out["resolver_margin"][i, t]))
        if e.meta.get("antecedent_recency") == "old":
            anti_total += 1
            anti_hits += int(is_hit)
    if total == 0:
        return None
    return {
        "overall_acc": hits / total, "n": total,
        "anti_recency_acc": (anti_hits / anti_total) if anti_total else float("nan"),
        "n_anti_recency": anti_total,
        "margins": margins,
    }


def sense_binding_stats(out, batch, va_eps):
    """M54's SENSE BINDING ACCURACY: overall + the sense-FLIPPED half (M32's
    own metric -- ``gold_sense != mfs_sense``, the half where the MFS floor
    is WRONG by construction), and the raw margin list. Mirrors
    :func:`resolver_binding_stats` exactly, just keyed on
    ``sense_resolver_logits``/``sense_cand_gold`` and the flip criterion
    instead of pronoun anti-recency."""
    if "sense_resolver_logits" not in out:
        return None
    cand_gold = batch.sense_cand_gold
    has_cand = cand_gold >= 0
    pred_idx = out["sense_resolver_logits"].argmax(-1)
    correct = (pred_idx == cand_gold) & has_cand
    margins = []
    hits = flip_hits = flip_total = total = 0
    for i, e in enumerate(va_eps):
        row_mask = has_cand[i]
        if not bool(row_mask.any()):
            continue
        # an ambiguity episode's homograph steps all carry the SAME gold
        # sense -- take the FIRST real step (mirrors resolver_binding_stats).
        t = int(row_mask.nonzero()[0, 0])
        total += 1
        is_hit = bool(correct[i, t])
        hits += int(is_hit)
        margins.append(float(out["sense_resolver_margin"][i, t]))
        if e.meta.get("gold_sense") != e.meta.get("mfs_sense"):
            flip_total += 1
            flip_hits += int(is_hit)
    if total == 0:
        return None
    return {
        "overall_acc": hits / total, "n": total,
        "flipped_acc": (flip_hits / flip_total) if flip_total else float("nan"),
        "n_flipped": flip_total,
        "margins": margins,
    }


def garden_path_binding_stats(out, batch, va_eps):
    """M55b's READING BINDING ACCURACY: overall accuracy + margin
    distribution, mirroring :func:`resolver_binding_stats`/
    :func:`sense_binding_stats` exactly, keyed on ``hyp_resolver_logits``/
    ``batch.hyp_cand_gold``. ``gold_reading`` is an EXACT 50/50 split by
    construction (``GardenPathCurriculumGenerator``'s own counter, not an
    MFS-style skew), so there is no "flipped half" the way sense binding
    has one -- instead this reports the object/verb split separately, a
    balance sanity check (both should be near the overall accuracy if the
    resolver isn't just learning one reading's shortcut)."""
    if "hyp_resolver_logits" not in out:
        return None
    cand_gold = batch.hyp_cand_gold
    has_cand = cand_gold >= 0
    pred_idx = out["hyp_resolver_logits"].argmax(-1)
    correct = (pred_idx == cand_gold) & has_cand
    margins = []
    hits = obj_hits = obj_total = verb_hits = verb_total = total = 0
    for i, e in enumerate(va_eps):
        row_mask = has_cand[i]
        if not bool(row_mask.any()):
            continue
        t = int(row_mask.nonzero()[0, 0])
        total += 1
        is_hit = bool(correct[i, t])
        hits += int(is_hit)
        margins.append(float(out["hyp_resolver_margin"][i, t]))
        if e.meta.get("gold_reading") == "object":
            obj_total += 1
            obj_hits += int(is_hit)
        elif e.meta.get("gold_reading") == "verb":
            verb_total += 1
            verb_hits += int(is_hit)
    if total == 0:
        return None
    return {
        "overall_acc": hits / total, "n": total,
        "object_acc": (obj_hits / obj_total) if obj_total else float("nan"), "n_object": obj_total,
        "verb_acc": (verb_hits / verb_total) if verb_total else float("nan"), "n_verb": verb_total,
        "margins": margins,
    }


def _held_out_name_pools(held_out_female: str, held_out_male: str):
    """M56b (dev/TRACK_C_DESIGN.md §1.8/Risk #4, RESEARCH_NOTES M56): split
    curriculum2's fixed pronoun name pools into a TRAIN pool (every name
    except the two held out) and a HELD-OUT pool (exactly the two held-out
    names) -- disjoint by construction. Raises loudly on a typo'd name
    (not silently falling back to the full pool) or on holding out the only
    name in a gender pool (would leave nothing to train that gender on)."""
    if held_out_female not in _FEMALE_NAMES:
        raise ValueError(f"{held_out_female!r} not in curriculum2._FEMALE_NAMES {_FEMALE_NAMES}")
    if held_out_male not in _MALE_NAMES:
        raise ValueError(f"{held_out_male!r} not in curriculum2._MALE_NAMES {_MALE_NAMES}")
    train_female = [n for n in _FEMALE_NAMES if n != held_out_female]
    train_male = [n for n in _MALE_NAMES if n != held_out_male]
    if not train_female or not train_male:
        raise ValueError("holding out the only name in a gender pool leaves nothing to train on")
    return (train_female, train_male), ([held_out_female], [held_out_male])


def run_held_out_name_ablation(n_episodes: int, epochs: int, dim: int, seed: int, hidden: int,
                                held_out_female: str = "sandra", held_out_male: str = "bill",
                                heads=("old", "fixed")) -> dict:
    """M56b: the held-out-name ablation THAT CONFIRMS the memorization
    finding, plus (when ``heads`` includes ``"fixed"``) the re-gate proving
    the per-candidate-feature fix closes the gap -- one call, the milestone's
    own 2x2 deliverable.

    Trains a fresh :class:`~nsm_ct.resolver.CorefHead` per ``heads`` entry
    (``"old"`` = ``use_cand_feature=False``, the pre-M56b head; ``"fixed"`` =
    ``use_cand_feature=True``, §1.8's per-candidate feature register) on
    PURE pronoun-binding episodes drawn ONLY from a TRAIN name pool (every
    curriculum name except one held-out female + one held-out male, see
    :func:`_held_out_name_pools`). No old/transfer/ambiguity episodes and no
    sense_resolver -- this ablation isolates the coref mechanism alone, at
    smoke scale. Evaluates RESOLVER BINDING ACCURACY on two DISJOINT
    batches: the usual 20%% held-back split of the TRAIN-name episodes
    (in-distribution), and a separate batch built entirely from the two
    held-out names (never seen in ANY training step). If M56's hypothesis
    is right, the old head's held-out-name binding collapses toward chance
    (0.5) while its train-name binding stays high; the fixed head should
    hold up on both, since ``feature_match`` no longer requires having
    memorized the specific name atom.

    Returns ``{head_label: {"train_names": acc, "held_out_names": acc,
    "n_train_names": n, "n_held_out_names": n}}``.
    """
    (train_female, train_male), (ho_female, ho_male) = _held_out_name_pools(held_out_female, held_out_male)

    train_episodes = generate_pronoun_episodes(n_episodes, seed=seed,
                                                female_names=train_female, male_names=train_male)
    n_held_out = max(200, n_episodes // 4)
    held_out_episodes = generate_pronoun_episodes(n_held_out, seed=seed + 999,
                                                   female_names=ho_female, male_names=ho_male)

    all_texts = [t for e in (train_episodes + held_out_episodes)
                 for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(all_texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping.")
        return {}
    meaning_resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    tr_eps, va_eps = split_episodes(train_episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, parser, meaning_resolver, codec)
    va = build_clause_batch(va_eps, parser, meaning_resolver, codec)
    ho = build_clause_batch(held_out_episodes, parser, meaning_resolver, codec)
    gold_tr = torch.tensor([e.answer_idx for e in tr_eps])

    results: dict = {}
    for label in heads:
        use_cand_feature = label == "fixed"
        torch.manual_seed(seed)
        resolver = CorefHead(dim, use_cand_feature=use_cand_feature)
        model = ClauseReactor(dim=dim, hidden=hidden, resolver=resolver, sense_resolver=None)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        model.train()
        for _ in range(epochs):
            out = model(tr)
            loss = F.cross_entropy(out["answer_logits"], gold_tr)
            cg = tr.cand_gold
            has_cand = cg >= 0
            if bool(has_cand.any()):
                loss = loss + AUX_WEIGHT * F.cross_entropy(out["resolver_logits"][has_cand], cg[has_cand])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            out_va, out_ho = model(va), model(ho)
        bind_va = resolver_binding_stats(out_va, va, va_eps)
        bind_ho = resolver_binding_stats(out_ho, ho, held_out_episodes)
        results[label] = {
            "train_names": bind_va["overall_acc"] if bind_va else float("nan"),
            "n_train_names": bind_va["n"] if bind_va else 0,
            "held_out_names": bind_ho["overall_acc"] if bind_ho else float("nan"),
            "n_held_out_names": bind_ho["n"] if bind_ho else 0,
        }
        r = results[label]
        print(f"  [{label} head] train-names binding={r['train_names']:.3f} (n={r['n_train_names']})  "
              f"held-out-names binding={r['held_out_names']:.3f} (n={r['n_held_out_names']})", flush=True)

    print("\n=== M56b held-out-name ablation: 2x2 table ===")
    print(f"{'head':<10} {'train names':>14} {'held-out names':>16}")
    for label in heads:
        r = results[label]
        print(f"{label:<10} {r['train_names']:>14.3f} {r['held_out_names']:>16.3f}")
    return results


def report_eval(name: str, model, va, va_eps, gold_va, n_resolver_params: int, n_sense_params: int,
                 shared: bool, elapsed_min: "float | None" = None, n_hyp_params: int = 0):
    """Shared eval + report tail for :func:`run_arm` and (M54c)
    :func:`run_distilled_arm`: task accuracy (overall + per curriculum kind),
    RESOLVER/SENSE/READING BINDING ACCURACY (M53/M54/M55b's own metrics) +
    margin distributions, and all resolvers' param counts. Factored out of
    :func:`run_arm`'s post-training tail UNCHANGED (same prints, same dict
    shape) purely so the M54c distillation arm's stage-3 output doesn't
    duplicate ~35 lines of reporting logic. ``n_hyp_params`` defaults 0 so
    callers that never install a hyp_resolver need no change."""
    model.eval()
    with torch.no_grad():
        out_va = model(va)
    pred = out_va["answer_logits"].argmax(-1)
    total_acc = float((pred == gold_va).float().mean())
    time_str = f"  time={elapsed_min:.2f} min" if elapsed_min is not None else ""
    print(f"  [{name}] val total={total_acc:.3f}  resolver_params={n_resolver_params} "
          f"sense_resolver_params={n_sense_params} hyp_resolver_params={n_hyp_params}"
          f"{' (SHARED, same instance)' if shared else ''}"
          f"{time_str}", flush=True)

    per_kind = {}
    for i, e in enumerate(va_eps):
        per_kind.setdefault(str(e.meta.get("kind", "old")), []).append(bool(pred[i] == gold_va[i]))
    for k in sorted(per_kind):
        w = per_kind[k]
        print(f"    kind {k}: {sum(w)}/{len(w)} = {sum(w)/len(w):.3f}")

    binding = resolver_binding_stats(out_va, va, va_eps)
    if binding is not None:
        m = np.array(binding["margins"]) if binding["margins"] else np.array([0.0])
        print(f"  [{name}] RESOLVER BINDING ACCURACY overall={binding['overall_acc']:.3f} "
              f"(n={binding['n']}) anti-recency={binding['anti_recency_acc']:.3f} "
              f"(n={binding['n_anti_recency']}) vs baseline 0.500/0.000", flush=True)
        print(f"  [{name}] pronoun margin distribution: min={m.min():.3f} p25={np.percentile(m, 25):.3f} "
              f"median={np.median(m):.3f} p75={np.percentile(m, 75):.3f} max={m.max():.3f}", flush=True)

    sense_binding = sense_binding_stats(out_va, va, va_eps)
    if sense_binding is not None:
        sm = np.array(sense_binding["margins"]) if sense_binding["margins"] else np.array([0.0])
        print(f"  [{name}] SENSE BINDING ACCURACY overall={sense_binding['overall_acc']:.3f} "
              f"(n={sense_binding['n']}) flipped-half={sense_binding['flipped_acc']:.3f} "
              f"(n={sense_binding['n_flipped']}) vs MFS floor 0.000 (by construction)", flush=True)
        print(f"  [{name}] sense margin distribution: min={sm.min():.3f} p25={np.percentile(sm, 25):.3f} "
              f"median={np.median(sm):.3f} p75={np.percentile(sm, 75):.3f} max={sm.max():.3f}", flush=True)

    hyp_binding = garden_path_binding_stats(out_va, va, va_eps)
    if hyp_binding is not None:
        hm = np.array(hyp_binding["margins"]) if hyp_binding["margins"] else np.array([0.0])
        print(f"  [{name}] READING BINDING ACCURACY overall={hyp_binding['overall_acc']:.3f} "
              f"(n={hyp_binding['n']}) object={hyp_binding['object_acc']:.3f} (n={hyp_binding['n_object']}) "
              f"verb={hyp_binding['verb_acc']:.3f} (n={hyp_binding['n_verb']}) vs baseline 0.500", flush=True)
        print(f"  [{name}] reading margin distribution: min={hm.min():.3f} p25={np.percentile(hm, 25):.3f} "
              f"median={np.median(hm):.3f} p75={np.percentile(hm, 75):.3f} max={hm.max():.3f}", flush=True)

    return {"total_acc": total_acc, "n_resolver_params": n_resolver_params,
            "n_sense_params": n_sense_params, "n_hyp_params": n_hyp_params,
            "binding": binding, "sense_binding": sense_binding, "hyp_binding": hyp_binding}


def run_arm(name: str, track, episodes, dim: int, epochs: int, seed: int, hidden: int = 128,
            sense_bind: str = "gold", reading_bind: str = "gold"):
    """``track``: "A" | "B" | None (None = --gold-binding / --mfs-floor /
    --wrong-binding, no resolver).

    M54/M55b: when ``track`` is set, installs a pronoun resolver
    (``resolver=``, unchanged from M53), a sense resolver
    (``sense_resolver=``, M54), AND a garden-path/reading resolver
    (``hyp_resolver=``, M55b) on the SAME model -- Track A gets THREE
    INDEPENDENT specialist heads (CorefHead + SenseHead + RankHead); Track B
    gets ONE SharedScorer instance passed to all three slots
    (``resolver is sense_resolver is hyp_resolver`` -- the literal "one
    shared scorer for everything" experiment, now covering all three
    collapse types). When ``track`` is None, no slot is installed and
    ``sense_bind``/``reading_bind`` control what :func:`build_clause_batch`
    placeholder-binds homograph/garden-path steps to (``sense_bind``:
    ``"gold"`` = ceiling, ``"mfs"`` = floor; ``reading_bind``: ``"gold"`` =
    ceiling, ``"wrong"`` = the M55b floor probe -- forces the OPPOSITE
    reading); pronoun steps always placeholder-bind to the gold antecedent
    regardless (there is no MFS/wrong-equivalent floor for coreference).
    """
    texts = [t for e in episodes for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping.")
        return None
    meaning_resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    tr_eps, va_eps = split_episodes(episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, parser, meaning_resolver, codec,
                             sense_bind=sense_bind, reading_bind=reading_bind)
    va = build_clause_batch(va_eps, parser, meaning_resolver, codec,
                             sense_bind=sense_bind, reading_bind=reading_bind)

    torch.manual_seed(seed)
    if track:
        resolver, sense_resolver, hyp_resolver = build_resolvers(track, dim, hidden)
    else:
        resolver = sense_resolver = hyp_resolver = None
    model = ClauseReactor(dim=dim, hidden=hidden, resolver=resolver,
                           sense_resolver=sense_resolver, hyp_resolver=hyp_resolver)

    def _params(m):
        return sum(p.numel() for p in m.parameters()) if m is not None else 0

    n_resolver_params = _params(resolver)
    n_sense_params = _params(sense_resolver)
    n_hyp_params = _params(hyp_resolver)
    shared = resolver is not None and resolver is sense_resolver
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gold_tr = torch.tensor([e.answer_idx for e in tr_eps])
    gold_va = torch.tensor([e.answer_idx for e in va_eps])

    t0 = time.time()
    model.train()
    for i in range(epochs):
        out = model(tr)
        loss = F.cross_entropy(out["answer_logits"], gold_tr)
        if resolver is not None and "resolver_logits" in out:
            cg = tr.cand_gold
            has_cand = cg >= 0
            if bool(has_cand.any()):
                aux = F.cross_entropy(out["resolver_logits"][has_cand], cg[has_cand])
                loss = loss + AUX_WEIGHT * aux
        if sense_resolver is not None and "sense_resolver_logits" in out:
            scg = tr.sense_cand_gold
            has_scand = scg >= 0
            if bool(has_scand.any()):
                saux = F.cross_entropy(out["sense_resolver_logits"][has_scand], scg[has_scand])
                loss = loss + AUX_WEIGHT * saux
        if hyp_resolver is not None and "hyp_resolver_logits" in out:
            hcg = tr.hyp_cand_gold
            has_hcand = hcg >= 0
            if bool(has_hcand.any()):
                haux = F.cross_entropy(out["hyp_resolver_logits"][has_hcand], hcg[has_hcand])
                loss = loss + AUX_WEIGHT * haux
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 20 == 0 or i == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(va)["answer_logits"].argmax(-1) == gold_va).float().mean()
            model.train()
            print(f"  [{name}] epoch {i+1:3d} loss={loss.item():.3f} val={acc:.3f}", flush=True)

    elapsed_min = (time.time() - t0) / 60
    return report_eval(name, model, va, va_eps, gold_va, n_resolver_params, n_sense_params,
                        shared, elapsed_min, n_hyp_params=n_hyp_params)


def run_distilled_arm(name: str, episodes, dim: int, seed: int, hidden: int = 128,
                       stage1_epochs: int = 60, stage2_epochs: int = 60, stage3_epochs: int = 15,
                       b_track: str = "B-NOSTATE-WIDE", distill_weight: float = DISTILL_WEIGHT):
    """M54c ``--track B-distilled``: three stages in ONE call, unpicking the
    third A-vs-B confound (B has never been given a head start from A's
    already-learned specialist behavior).

    Stage 1 -- train Track A normally (the exact two-specialist-head setup
      ``run_arm("A", ...)`` uses: task CE + both resolver aux losses).
    Stage 2 -- FREEZE stage 1's Track A model (eval mode, forward passes
      under ``torch.no_grad()`` -- never backprops into it) and initialize a
      FRESH B-family resolver (``b_track``, default B-nostate-wide per the
      milestone brief: "pick the best config from your smokes, default
      wide-no-state if smokes are ambiguous"), installed as ONE shared
      instance across both slots. Trains it on the SAME batches with task +
      aux losses PLUS :func:`distill_loss_terms` -- KL(B's candidate softmax
      || frozen Track-A's per-type candidate softmax), computed separately
      for pronoun-candidate rows (against A's CorefHead output) and
      sense-candidate rows (against A's SenseHead output) on THIS batch.
    Stage 3 -- drop the distillation term; brief task(+aux)-only fine-tune of
      the SAME B model (no reinitialization).

    All three stage epoch counts are independently configurable so a smoke
    can check the MECHANICS (loss decreases within each stage, stages
    actually transition) cheaply without running a full comparison.
    """
    texts = [t for e in episodes for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable in this environment; skipping.")
        return None
    meaning_resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=dim)

    tr_eps, va_eps = split_episodes(episodes, 0.2, seed=0)
    tr = build_clause_batch(tr_eps, parser, meaning_resolver, codec, sense_bind="gold")
    va = build_clause_batch(va_eps, parser, meaning_resolver, codec, sense_bind="gold")
    gold_tr = torch.tensor([e.answer_idx for e in tr_eps])
    gold_va = torch.tensor([e.answer_idx for e in va_eps])

    # ---- stage 1: Track A, trained normally ----
    torch.manual_seed(seed)
    a_resolver, a_sense_resolver, a_hyp_resolver = build_resolvers("A", dim, hidden)
    model_a = ClauseReactor(dim=dim, hidden=hidden, resolver=a_resolver, sense_resolver=a_sense_resolver,
                             hyp_resolver=a_hyp_resolver)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=3e-3)
    model_a.train()
    stage1_losses = []
    t0 = time.time()
    for i in range(stage1_epochs):
        out = model_a(tr)
        loss = (F.cross_entropy(out["answer_logits"], gold_tr)
                + aux_loss_terms(out, tr, a_resolver, a_sense_resolver, a_hyp_resolver))
        opt_a.zero_grad(); loss.backward(); opt_a.step()
        stage1_losses.append(float(loss.item()))
    model_a.eval()
    print(f"  [{name}] stage 1 (Track A) done: loss {stage1_losses[0]:.3f} -> {stage1_losses[-1]:.3f} "
          f"over {stage1_epochs} epochs", flush=True)

    # ---- stage 2: fresh B, distilling from frozen stage-1 A ----
    torch.manual_seed(seed + 1000)
    b_resolver = build_b_family_resolver(b_track, dim, hidden)
    model_b = ClauseReactor(dim=dim, hidden=hidden, resolver=b_resolver, sense_resolver=b_resolver,
                             hyp_resolver=b_resolver)
    n_resolver_params = sum(p.numel() for p in b_resolver.parameters())
    opt_b = torch.optim.Adam(model_b.parameters(), lr=3e-3)
    model_b.train()
    stage2_losses, stage2_kl = [], []
    for i in range(stage2_epochs):
        with torch.no_grad():
            teacher_out = model_a(tr)
        out = model_b(tr)
        kl = distill_loss_terms(out, teacher_out, tr)
        loss = (F.cross_entropy(out["answer_logits"], gold_tr)
                + aux_loss_terms(out, tr, b_resolver, b_resolver, b_resolver)
                + distill_weight * kl)
        opt_b.zero_grad(); loss.backward(); opt_b.step()
        stage2_losses.append(float(loss.item()))
        stage2_kl.append(float(kl.item()) if torch.is_tensor(kl) else float(kl))
    print(f"  [{name}] stage 2 (distill into {b_track}) done: loss {stage2_losses[0]:.3f} -> "
          f"{stage2_losses[-1]:.3f}, KL term {stage2_kl[0]:.3f} -> {stage2_kl[-1]:.3f} "
          f"over {stage2_epochs} epochs", flush=True)

    # ---- stage 3: task-only fine-tune, distillation dropped ----
    model_b.train()
    stage3_losses = []
    for i in range(stage3_epochs):
        out = model_b(tr)
        loss = (F.cross_entropy(out["answer_logits"], gold_tr)
                + aux_loss_terms(out, tr, b_resolver, b_resolver, b_resolver))
        opt_b.zero_grad(); loss.backward(); opt_b.step()
        stage3_losses.append(float(loss.item()))
    if stage3_losses:
        print(f"  [{name}] stage 3 (task-only fine-tune) done: loss {stage3_losses[0]:.3f} -> "
              f"{stage3_losses[-1]:.3f} over {stage3_epochs} epochs", flush=True)

    elapsed_min = (time.time() - t0) / 60
    result = report_eval(name, model_b, va, va_eps, gold_va, n_resolver_params, n_resolver_params,
                          True, elapsed_min, n_hyp_params=n_resolver_params)
    result["stage1_losses"] = stage1_losses
    result["stage2_losses"] = stage2_losses
    result["stage2_kl"] = stage2_kl
    result["stage3_losses"] = stage3_losses
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track",
                     choices=["A", "B", "B-wide", "B-nostate", "B-nostate-wide", "B-distilled"],
                     default=None,
                     help="A/B = M53/M54's original two tracks. M54c diagnosis arms (RESEARCH_NOTES "
                          "M54c), all installing ONE SharedScorer instance across pronoun+sense "
                          "candidates like B: B-wide (capacity-matched to A's CorefHead+SenseHead "
                          "combined), B-nostate (drops the controller-state input, params matched "
                          "to the original narrow B), B-nostate-wide (both at once), B-distilled "
                          "(stage 1 trains A, stage 2 distills a fresh B from frozen A, stage 3 "
                          "task-only fine-tune -- see --distill-* flags)")
    ap.add_argument("--gold-binding", action="store_true",
                     help="ceiling arm: no resolver, gold sense + gold antecedent + gold reading "
                          "placeholder binding")
    ap.add_argument("--mfs-floor", action="store_true",
                     help="floor arm: no resolver, MFS sense placeholder binding "
                          "(gold antecedent/reading still bound) -- the M30 baseline")
    ap.add_argument("--wrong-binding", action="store_true",
                     help="M55b floor arm: no resolver, garden-path reading steps placeholder-bound "
                          "to the OPPOSITE of the true gold reading (gold sense/antecedent still "
                          "bound) -- the gold-vs-wrong gap probe (RESEARCH_NOTES M55b)")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", choices=["m53", "m54", "m54b", "m55a", "m55"], default="m54",
                     help="m53 = the original 1/2 old + 1/4 transfer + 1/4 pronoun mix "
                          "(no ambiguity episodes, exact M53 reproduction); "
                          "m54 (default) = 40%% old / 20%% transfer / 20%% pronoun / 20%% ambiguity; "
                          "m54b = 50%% old / 50%% the NEW entity-keyed sense-binding curriculum "
                          "(RESEARCH_NOTES M54b's gold/MFS gap probe); "
                          "m55a = 50%% old / 50%% the OLD (counter-driven) garden-path curriculum "
                          "(RESEARCH_NOTES M55a, superseded by m55 below); "
                          "m55 = 40%% old / 20%% pronoun / 20%% sense-binding / 20%% the REDESIGNED "
                          "binding-critical garden-path curriculum (RESEARCH_NOTES M55b) -- the "
                          "full-membrane mix, all three collapse types together")
    ap.add_argument("--distill-b-track", choices=["B-wide", "B-nostate", "B-nostate-wide"],
                     default="B-nostate-wide",
                     help="--track B-distilled only: which B-family config stage 2 initializes "
                          "fresh and distills into (default wide-no-state, per the M54c brief)")
    ap.add_argument("--distill-stage1-epochs", type=int, default=None,
                     help="--track B-distilled only: Track-A pretraining epochs (default --epochs)")
    ap.add_argument("--distill-stage2-epochs", type=int, default=None,
                     help="--track B-distilled only: distillation epochs (default --epochs)")
    ap.add_argument("--distill-stage3-epochs", type=int, default=None,
                     help="--track B-distilled only: task-only fine-tune epochs (default --epochs // 4)")
    ap.add_argument("--distill-weight", type=float, default=DISTILL_WEIGHT,
                     help="--track B-distilled only: weight on stage 2's KL(B || frozen A) term")
    ap.add_argument("--held-out-name-ablation", action="store_true",
                     help="M56b: run the held-out-name ablation instead of any --track arm -- trains "
                          "CorefHead (old: use_cand_feature=False, and fixed: use_cand_feature=True) "
                          "on pronoun-only episodes from a TRAIN name pool, evaluates RESOLVER BINDING "
                          "ACCURACY on a train-name val split AND a disjoint held-out-name-only batch, "
                          "prints the 2x2 table (see dev/TRACK_C_DESIGN.md §1.8, RESEARCH_NOTES M56/M56b)")
    ap.add_argument("--held-out-female", default="sandra",
                     help="--held-out-name-ablation only: which curriculum2._FEMALE_NAMES entry to hold out")
    ap.add_argument("--held-out-male", default="bill",
                     help="--held-out-name-ablation only: which curriculum2._MALE_NAMES entry to hold out")
    ap.add_argument("--ablation-heads", default="old,fixed",
                     help="--held-out-name-ablation only: comma-separated subset of {old,fixed} to run")
    args = ap.parse_args()

    if args.held_out_name_ablation:
        heads = tuple(h.strip() for h in args.ablation_heads.split(",") if h.strip())
        print(f"=== M56b held-out-name ablation: {args.episodes} train eps, "
              f"held-out={args.held_out_female}/{args.held_out_male}, dim={args.dim}, "
              f"epochs={args.epochs}, heads={heads} ===", flush=True)
        run_held_out_name_ablation(args.episodes, args.epochs, args.dim, args.seed, args.hidden,
                                    args.held_out_female, args.held_out_male, heads=heads)
        return

    n_arms = sum([args.gold_binding, args.mfs_floor, args.wrong_binding, args.track is not None])
    if n_arms != 1:
        raise SystemExit("pass exactly one of --track A|B|B-wide|B-nostate|B-nostate-wide|B-distilled, "
                          "--gold-binding, --mfs-floor, --wrong-binding")

    if args.mix == "m54":
        episodes = build_m54_curriculum(args.episodes, args.seed)
    elif args.mix == "m54b":
        episodes = build_m54b_curriculum(args.episodes, args.seed)
    elif args.mix == "m55a":
        episodes = build_m55a_curriculum(args.episodes, args.seed)
    elif args.mix == "m55":
        episodes = build_m55_curriculum(args.episodes, args.seed)
    else:
        episodes = build_mixed_curriculum(args.episodes, args.seed)
    baseline = nearest_entity_baseline(episodes)
    print(f"nearest-entity baseline: overall={baseline['accuracy']:.3f} (n={baseline['n']}) "
          f"anti-recency={baseline['anti_recency_accuracy']:.3f} (n={baseline['n_anti_recency']})",
          flush=True)
    assoc = association_only_baseline(episodes)
    if assoc["n"]:
        print(f"association-only baseline (M54b sense-binding kind): "
              f"accuracy={assoc['accuracy']:.3f} (n={assoc['n']}) vs chance 0.500", flush=True)
    gp_assoc = garden_path_association_baseline(episodes)
    if gp_assoc["n"]:
        print(f"association-only baseline (M55b garden-path kind): "
              f"accuracy={gp_assoc['accuracy']:.3f} (n={gp_assoc['n']}) vs chance 0.500", flush=True)
        gp_texts = [t for e in episodes for t in e.context + [e.question] + (e.options or [])]
        gp_tok = SimpleTokenizer.build(gp_texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
        gp_parser = ParserInputEncoder(gp_tok)
        if getattr(gp_parser, "_parser", None) is not None:
            gp_top1 = garden_path_parser_top1_baseline(episodes, gp_parser)
            print(f"parser-top-1 baseline (M55b garden-path kind): "
                  f"accuracy={gp_top1['accuracy']:.3f} (n={gp_top1['n']}) vs chance 0.500", flush=True)

    if args.gold_binding:
        print(f"=== gold-binding ceiling: no resolver, {args.episodes} eps, dim={args.dim}, "
              f"mix={args.mix} ===", flush=True)
        run_arm("gold-binding", None, episodes, args.dim, args.epochs, args.seed, args.hidden,
                 sense_bind="gold", reading_bind="gold")
    elif args.mfs_floor:
        print(f"=== mfs-floor: no resolver, {args.episodes} eps, dim={args.dim}, "
              f"mix={args.mix} ===", flush=True)
        run_arm("mfs-floor", None, episodes, args.dim, args.epochs, args.seed, args.hidden,
                 sense_bind="mfs", reading_bind="gold")
    elif args.wrong_binding:
        print(f"=== wrong-binding floor: no resolver, {args.episodes} eps, dim={args.dim}, "
              f"mix={args.mix} ===", flush=True)
        run_arm("wrong-binding", None, episodes, args.dim, args.epochs, args.seed, args.hidden,
                 sense_bind="gold", reading_bind="wrong")
    elif args.track == "B-distilled":
        s1 = args.distill_stage1_epochs if args.distill_stage1_epochs is not None else args.epochs
        s2 = args.distill_stage2_epochs if args.distill_stage2_epochs is not None else args.epochs
        s3 = (args.distill_stage3_epochs if args.distill_stage3_epochs is not None
              else max(1, args.epochs // 4))
        print(f"=== track B-distilled (stage1={s1} stage2={s2} stage3={s3}, "
              f"b_track={args.distill_b_track}): {args.episodes} eps, dim={args.dim}, "
              f"mix={args.mix} ===", flush=True)
        run_distilled_arm("track-B-distilled", episodes, args.dim, args.seed, args.hidden,
                           stage1_epochs=s1, stage2_epochs=s2, stage3_epochs=s3,
                           b_track=args.distill_b_track, distill_weight=args.distill_weight)
    else:
        print(f"=== track {args.track}: {args.episodes} eps, dim={args.dim}, mix={args.mix} ===",
              flush=True)
        run_arm(f"track-{args.track}", args.track, episodes, args.dim, args.epochs, args.seed,
                 args.hidden)


if __name__ == "__main__":
    main()
