"""Synthetic-user environment for the learned drive (M15, L6).

The drive (:mod:`nsm_ct.mind.drive`) must learn *when* initiative is useful. This module
is its teacher: it generates short conversational decision points whose **gold action** is
set by a **latent usefulness the policy never sees**, so imitating it is a genuine learning
problem (not a hand-coded rule restated in weights).

Grounded by construction: each world is a real theory whose derivation closure
(``forward_chain``) supplies the candidate facts and their **real derivation depths**
(relevance/novelty). The latent usefulness layered on top is principled, not arbitrary:

* a volunteer/premise is **on the user's goal-path** with probability that *decreases with
  derivation depth* (relevant, close facts matter more) — a noisy signal, so the features
  are predictive but not deterministic;
* initiative only helps when the user is **not overloaded** (``pending <= FOCUS_CAPACITY``)
  — volunteering into a backlog is noise.

So usefulness = *relevant* **and** *focused* — an **interaction** the policy must learn by
combining two grounded features (depth × backlog). This is exactly what separates the right
answer from the two failure modes: **always-volunteer (yappy)** ignores both; **never
(Siri)** ignores that relevant+focused initiative is wanted.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import torch

from ..reasoning_oracle import Rule, forward_chain
from . import verbalize
from .drive import (ANSWER, ASK, QUIET, VOLUNTEER, DriveContext, feasible_mask,
                    features_vec)

FOCUS_CAPACITY = 2          # pending above this ⇒ the user is overloaded; extra initiative = noise


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _want_prob(depth: int) -> float:
    """P(a candidate at this derivation depth is on the user's goal-path) — relevance
    decays with depth (close facts matter more). Noisy, so depth is predictive not exact."""
    return _sigmoid(2.0 - 1.1 * depth)


# -- worlds (grounded depths from a real closure) --------------------------------
def sample_world(rng: random.Random, *, max_depth: int = 4, max_branch: int = 3):
    """A real theory about subject ``x``: a derivation chain r0→r1→…→rD (depths 1..D) plus
    a few depth-1 branches. ``forward_chain`` over it yields candidates with real depths."""
    subj = "x"
    facts: List[Tuple[str, str, str, str]] = [(subj, "r0", "v0", "+")]
    rules: List[Rule] = []
    depth = rng.randint(1, max_depth)
    pr, pv = "r0", "v0"
    for i in range(1, depth + 1):
        ri, vi = f"r{i}", f"v{i}"
        rules.append(Rule(((f"?x", pr, pv, "+"),), ("?x", ri, vi, "+"), name=f"c{i}"))
        pr, pv = ri, vi
    for j in range(rng.randint(0, max_branch)):
        rules.append(Rule((("?x", "r0", "v0", "+"),), ("?x", f"b{j}", f"bv{j}", "+"), name=f"b{j}"))
    return subj, facts, rules


def _candidates(subj, facts, rules):
    """The volunteer pool (derived, positive, unsaid facts about ``subj``) with real depths,
    ranked most-relevant (shallowest) first — the same selection the live conversation uses."""
    known, chain = forward_chain(list(facts), rules)
    said = set(facts)
    pool = []
    for f in known:
        s, r, v, pol = f
        if s == subj and pol == "+" and f not in said:
            pool.append((len(verbalize._relevant_steps(chain, f)), f))
    pool.sort(key=lambda c: (c[0], c[1][1], c[1][2]))
    return pool                                          # list of (depth, fact)


# -- decision points (context + grounded gold) -----------------------------------
def sample_example(rng: random.Random) -> Tuple[DriveContext, int]:
    """One labelled decision point: a :class:`DriveContext` + the gold action from the
    latent usefulness. Half answered-context (ANSWER vs VOLUNTEER), half blocked
    (ASK vs QUIET)."""
    pending = rng.randint(0, 5)
    gained = rng.randint(0, 3)
    focus = rng.randint(0, 4)
    overloaded = pending > FOCUS_CAPACITY

    if rng.random() < 0.5:                               # ---- answered context ----
        subj, facts, rules = sample_world(rng)
        pool = _candidates(subj, facts, rules)
        if not pool:                                     # nothing derivable to add
            ctx = DriveContext(True, False, 0, 0, rng.randint(1, 4), pending, gained, focus)
            return ctx, ANSWER
        answer_steps, _asked = pool[-1]                  # the user asked the deepest fact
        remaining = pool[:-1]
        if not remaining:
            ctx = DriveContext(True, False, 0, 0, answer_steps, pending, gained, focus)
            return ctx, ANSWER
        top_depth, _top = remaining[0]
        on_goal = rng.random() < _want_prob(top_depth)   # latent: is the best candidate wanted?
        useful = on_goal and not overloaded
        ctx = DriveContext(True, False, len(remaining), top_depth, answer_steps,
                           pending, gained, focus)
        return ctx, (VOLUNTEER if useful else ANSWER)

    # ---- blocked context: a missing premise, ask vs stay quiet ----
    premise_depth = rng.randint(1, 4)
    on_goal = rng.random() < _want_prob(premise_depth)   # latent: is the premise on-goal?
    useful = on_goal and not overloaded
    ctx = DriveContext(False, True, 0, premise_depth, 0, pending, gained, focus)
    return ctx, (ASK if useful else QUIET)


def generate(n: int, *, seed: int = 0) -> List[Tuple[DriveContext, int]]:
    rng = random.Random(seed)
    return [sample_example(rng) for _ in range(n)]


def build_batch(examples) -> Dict[str, torch.Tensor]:
    """``[(ctx, gold)]`` → tensors ``feats [B,8]``, ``gold [B]``, ``mask [B,4]``."""
    feats = torch.stack([features_vec(c) for c, _ in examples])
    gold = torch.tensor([g for _, g in examples], dtype=torch.long)
    mask = torch.stack([feasible_mask(c) for c, _ in examples])
    return {"feats": feats, "gold": gold, "mask": mask}


# -- baselines + calibration metrics (yappy vs Siri) -----------------------------
def baseline_action(ctx: DriveContext, policy: str) -> int:
    """Reference policies: ``always`` = maximal initiative (yappy), ``never`` = none (Siri)."""
    if ctx.derivable:
        if policy == "always" and ctx.n_candidates > 0:
            return VOLUNTEER
        return ANSWER
    return ASK if policy == "always" else QUIET


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def initiative_metrics(examples, preds) -> Dict[str, float]:
    """Accuracy + the calibrated-initiative metrics: precision/recall of VOLUNTEER (in
    answered contexts) and ASK (in blocked contexts) vs the gold useful action — the
    quantified line between yappy (low precision) and Siri (low recall)."""
    acc = sum(int(p == g) for (_c, g), p in zip(examples, preds)) / max(len(preds), 1)
    vtp = vfp = vfn = atp = afp = afn = 0
    for (ctx, gold), pred in zip(examples, preds):
        if ctx.derivable:
            vtp += int(pred == VOLUNTEER and gold == VOLUNTEER)
            vfp += int(pred == VOLUNTEER and gold != VOLUNTEER)
            vfn += int(pred != VOLUNTEER and gold == VOLUNTEER)
        else:
            atp += int(pred == ASK and gold == ASK)
            afp += int(pred == ASK and gold != ASK)
            afn += int(pred != ASK and gold == ASK)
    vp, vr, vf = _prf(vtp, vfp, vfn)
    ap, ar, af = _prf(atp, afp, afn)
    return {"acc": acc, "volunteer_p": vp, "volunteer_r": vr, "volunteer_f1": vf,
            "ask_p": ap, "ask_r": ar, "ask_f1": af}


__all__ = ["FOCUS_CAPACITY", "sample_world", "sample_example", "generate", "build_batch",
           "baseline_action", "initiative_metrics"]
