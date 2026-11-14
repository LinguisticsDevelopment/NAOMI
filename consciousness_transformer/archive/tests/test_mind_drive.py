"""M15 gates: the learned drive (L6) — calibrated initiative.

The drive learns *when* to answer-terse / volunteer / ask / stay-quiet, supervised by a
synthetic-user environment whose gold action is set by a latent usefulness the policy never
sees. These gates cover: (Stage 2) the environment teacher is well-formed and feasible;
(Stage 3) the masked-CE policy is learnable and respects grounded feasibility; and the live
conversation is gated by the drive (VOLUNTEER surfaces, ANSWER/QUIET stay terse) — while
drive-absent behaviour is exactly M14.
"""

from __future__ import annotations

import torch

from nsm_ct.mind import drive, drive_env
from nsm_ct.mind.conscious_loop import ConsciousLoop
from nsm_ct.mind.conversation import Conversation
from nsm_ct.mind.drive import ANSWER, ASK, QUIET, VOLUNTEER
from nsm_ct.mind.knowledge import KnowledgeGraph


class _ConstDrive:
    """A stub policy that always prefers one action (feasibility masking still applies) —
    lets the wiring be tested without training a real policy."""

    def __init__(self, action: int) -> None:
        self.action = action

    def eval(self):
        return self

    def __call__(self, feats):
        logits = torch.full((drive.N_ACTS,), -5.0)
        logits[self.action] = 5.0
        return logits


# -- Stage 2: the environment teacher is well-formed + grounded -------------------
def test_env_gold_is_feasible_and_well_typed():
    """Every generated gold action is feasible under its own mask, and typed to its
    context (answered ⇒ {ANSWER,VOLUNTEER}; blocked ⇒ {ASK,QUIET}); VOLUNTEER only when
    there is a real candidate to surface."""
    for ctx, gold in drive_env.generate(500, seed=3):
        assert drive.feasible_mask(ctx)[gold] == 1.0
        if ctx.derivable:
            assert gold in (ANSWER, VOLUNTEER)
            if gold == VOLUNTEER:
                assert ctx.n_candidates > 0
        else:
            assert gold in (ASK, QUIET)


def test_env_world_candidates_are_grounded():
    """The world's volunteer pool is the real derivation closure (grounded depths), not
    invented: every candidate is derivable and unsaid, with depth >= 1."""
    import random
    rng = random.Random(0)
    for _ in range(50):
        subj, facts, rules = drive_env.sample_world(rng)
        pool = drive_env._candidates(subj, facts, rules)
        said = set(facts)
        for depth, f in pool:
            assert depth >= 1 and f not in said and f[0] == subj and f[3] == "+"


# -- Stage 3: the policy is learnable and respects feasibility --------------------
def test_drive_loss_overfits_separable_set():
    """Masked CE drives a separable set to ~1.0 — the learnability unit. The golds require
    combining relevance (depth) AND backlog (pending): relevant+focused ⇒ act, else hold."""
    ex = [
        (drive.DriveContext(True, False, 2, 1, 1, 0, 0, 0), VOLUNTEER),   # relevant + focused
        (drive.DriveContext(True, False, 2, 1, 1, 5, 0, 0), ANSWER),      # relevant, overloaded
        (drive.DriveContext(True, False, 1, 4, 1, 0, 0, 0), ANSWER),      # focused, irrelevant
        (drive.DriveContext(False, True, 0, 1, 0, 0, 0, 0), ASK),         # premise relevant+focused
        (drive.DriveContext(False, True, 0, 1, 0, 5, 0, 0), QUIET),       # relevant, overloaded
        (drive.DriveContext(False, True, 0, 4, 0, 0, 0, 0), QUIET),       # irrelevant premise
    ]
    torch.manual_seed(0)
    policy = drive.DrivePolicy()
    batch = drive_env.build_batch(ex)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    for _ in range(600):
        opt.zero_grad()
        drive.drive_loss(policy(batch["feats"]), batch["gold"], batch["mask"]).backward()
        opt.step()
    preds = [drive.predict_action(policy, drive.features_vec(c), drive.feasible_mask(c))
             for c, _ in ex]
    assert preds == [g for _, g in ex]


def test_predict_action_respects_feasibility():
    """A masked-out action is never chosen, even when the policy maximally prefers it."""
    # answered context with NO candidate ⇒ VOLUNTEER infeasible ⇒ must be ANSWER.
    ctx = drive.DriveContext(True, False, 0, 0, 1, 0, 0, 0)
    pol = _ConstDrive(VOLUNTEER)
    assert drive.predict_action(pol, drive.features_vec(ctx), drive.feasible_mask(ctx)) == ANSWER
    # blocked context ⇒ ANSWER/VOLUNTEER infeasible ⇒ chosen ∈ {ASK, QUIET}.
    ctx2 = drive.DriveContext(False, True, 0, 2, 0, 0, 0, 0)
    assert drive.predict_action(_ConstDrive(ANSWER), drive.features_vec(ctx2),
                                drive.feasible_mask(ctx2)) in (ASK, QUIET)


# -- Stage 3: the live conversation is gated by the drive -------------------------
def _kitchen_two() -> ConsciousLoop:
    return ConsciousLoop(KnowledgeGraph(dim=32))


def test_drive_gates_volunteer_on_and_off():
    """With a drive that chooses VOLUNTEER the extra fact is surfaced; with one that
    chooses ANSWER the reply is terse — the same grounded candidate, the drive decides."""
    def conv(action):
        c = Conversation(_kitchen_two(), max_volunteer=1, drive=_ConstDrive(action))
        c.say("everyone who is in the kitchen can see the window .")
        c.say("everyone who is in the kitchen can reach the door .")
        c.say("mary is in the kitchen .")
        return c.say("what can mary see ?")

    on = conv(VOLUNTEER)
    assert on == ["Mary can see the window.", "Also, mary can reach the door."]
    assert conv(ANSWER) == ["Mary can see the window."]


def test_drive_gates_ask_vs_quiet():
    """A blocked query: an ASK-drive asks the missing premise (and pends it); a QUIET-drive
    stays terse with an honest abstain (no nag, no pending)."""
    def conv(action):
        c = Conversation(_kitchen_two(), drive=_ConstDrive(action))
        c.say("everyone who is in the kitchen can see the window .")
        return c, c.say("what can mary see ?")

    c_ask, ask = conv(ASK)
    assert "mary" in ask[0].lower() and len(c_ask.pending) == 1
    c_quiet, quiet = conv(QUIET)
    assert quiet == ["I don't know."] and c_quiet.pending == []
    assert c_quiet.log[-1].kind == "quiet"


def test_drive_features_from_live_conversation():
    """``drive_features`` reads a well-formed, grounded context off the live session."""
    c = Conversation(_kitchen_two(), max_volunteer=1)
    c.say("everyone who is in the kitchen can see the window .")
    c.say("everyone who is in the kitchen can reach the door .")
    c.say("mary is in the kitchen .")
    cands = c._volunteer_candidates("mary", exclude=("mary", "CAN_SEE", "window"))
    ctx = drive.drive_features(c, ("mary", "CAN_SEE"), "window", cands, derivable=True)
    assert ctx.derivable and not ctx.has_premise
    assert ctx.n_candidates == len(cands) and ctx.n_candidates >= 1
    assert drive.feasible_mask(ctx)[VOLUNTEER] == 1.0
