"""M14 gates: a stateful conversation — ask-when-blocked (L2) + cross-turn memory (L4).

Deterministic, no training. Turns auto-search into natural back-and-forth: the system
remembers across turns, asks for the exact missing premise when it can't answer, and
resolves the pending question once told — staying grounded throughout (it only asks for
a premise a rule needs, and only answers what it derives).
"""

from __future__ import annotations

from nsm_ct.mind import membrane
from nsm_ct.mind.conscious_loop import ConsciousLoop
from nsm_ct.mind.conversation import Conversation
from nsm_ct.mind.knowledge import KnowledgeGraph
from nsm_ct.reasoning_oracle import (INHERITANCE, Rule, conditional_rule,
                                     find_missing_premise)


def _fresh() -> Conversation:
    return Conversation(ConsciousLoop(KnowledgeGraph(dim=32)))


# -- L2: ask when blocked -----------------------------------------------------
def test_ask_when_blocked():
    """Teach the rule but NOT the fact, then ask: the reply is the correct polar
    question for the missing premise, and the query is now pending."""
    c = _fresh()
    assert c.say("everyone who is in the kitchen can see the window .") == []
    replies = c.say("what can mary see ?")
    assert len(replies) == 1
    assert membrane.render_polar_question("mary", "PLACE", "kitchen") in replies[0]
    assert len(c.pending) == 1
    blocked_query, premise = c.pending[0]
    assert blocked_query == ("query", "mary", "CAN_SEE")
    assert premise[:3] == ("mary", "PLACE", "kitchen")


def test_full_loop_cross_turn():
    """Turn 1 teaches a rule + asks (it asks back); turn 2 supplies the premise and the
    pending query resolves. Proves L2 (ask) and L4 (cross-turn) together."""
    c = _fresh()
    c.say("everyone who is in the kitchen can see the window .")
    ask = c.say("what can mary see ?")[0]
    assert membrane.render_polar_question("mary", "PLACE", "kitchen") in ask
    resolved = c.say("mary is in the kitchen .")
    assert resolved == ["Then yes — mary can see the window."]
    assert c.pending == []                       # nothing left waiting


# -- L4: continuity -----------------------------------------------------------
def test_persistence_across_turns():
    """A fact taught in an early turn is used to answer a query in a later turn."""
    c = _fresh()
    c.say("mary is in the kitchen .")
    c.say("everyone who is in the kitchen can see the window .")
    assert c.say("what can mary see ?") == ["Mary can see the window."]


def test_cross_turn_coreference():
    """'she' in a later turn resolves to the entity introduced earlier (not the most
    recent token) and the answer is derived."""
    c = _fresh()
    c.say("mary is in the garden .")
    c.say("everyone who is in the garden can hold the stove .")
    assert c.say("what can she hold ?") == ["Mary can hold the stove."]


# -- iterative grounding (deeper chains, one premise per turn) ----------------
def test_iterative_grounding():
    """A two-hop gap surfaces one premise per turn; supplying the base fact resolves
    every pending question at once."""
    c = _fresh()
    c.say("everyone who is in the kitchen can reach the key .")
    c.say("everyone who can reach the key can open the door .")
    ask1 = c.say("what can mary open ?")[0]       # asks the door-rule's premise
    assert membrane.render_polar_question("mary", "CAN_REACH", "key") in ask1
    ask2 = c.say("what can mary reach ?")[0]       # asks the key-rule's premise
    assert membrane.render_polar_question("mary", "PLACE", "kitchen") in ask2
    assert len(c.pending) == 2
    resolved = c.say("mary is in the kitchen .")   # grounds the whole chain
    assert "Then yes — mary can open the door." in resolved
    assert "Then yes — mary can reach the key." in resolved
    assert c.pending == []


# -- honest abstain (no spurious question) ------------------------------------
def test_honest_abstain():
    """With no rule or fact reaching the goal, the system says it doesn't know rather
    than inventing a premise to ask for."""
    c = _fresh()
    assert c.say("where is sandra ?") == ["I don't know."]
    assert c.pending == []


# -- the missing-premise primitive itself -------------------------------------
def test_find_missing_premise_grounded_rule():
    """One-hop backward returns the ground antecedent the goal is waiting on."""
    rule = conditional_rule("kitchen", "window")      # (?p PLACE kitchen) ⇒ (?p CAN_SEE window)
    assert find_missing_premise([], [rule], ("mary", "CAN_SEE")) == ("mary", "PLACE", "kitchen")


def test_find_missing_premise_none_when_satisfied():
    """No premise to ask once the antecedent is already known."""
    rule = conditional_rule("kitchen", "window")
    assert find_missing_premise([("mary", "PLACE", "kitchen")], [rule], ("mary", "CAN_SEE")) is None


def test_find_missing_premise_skips_unbound():
    """An antecedent that stays variable after grounding (free ?y) is not askable."""
    # inheritance: (?x IS_A ?y) ∧ (?y CAN ?z) ⇒ (?x CAN ?z); goal binds ?x,?z only.
    assert find_missing_premise([], [INHERITANCE], ("robin", "CAN", "fly")) is None


def test_find_missing_premise_no_rule():
    """No rule path → no premise (the honest-abstain case)."""
    assert find_missing_premise([], [], ("sandra", "PLACE")) is None


# -- L3 (M15): bounded volunteering — calibrated initiative over real knowledge ----
def _kitchen_two_abilities() -> Conversation:
    """Mary in the kitchen, where the rules derive TWO abilities (see window, reach door)."""
    c = Conversation(ConsciousLoop(KnowledgeGraph(dim=32)), max_volunteer=1)
    c.say("everyone who is in the kitchen can see the window .")
    c.say("everyone who is in the kitchen can reach the door .")
    c.say("mary is in the kitchen .")
    return c


def test_volunteer_one_relevant_unsaid_fact():
    """After answering, the system surfaces exactly ONE extra true, on-topic, unsaid
    fact — and logs it as a 'volunteered' turn outcome (the L6 telemetry)."""
    c = _kitchen_two_abilities()
    replies = c.say("what can mary see ?")
    assert replies[0] == "Mary can see the window."
    assert len(replies) == 2                                  # answer + ONE volunteer (budget)
    assert replies[1] == "Also, mary can reach the door."
    assert c.log[-1].kind == "volunteered"


def test_volunteer_budget_zero_is_m14():
    """Budget 0 ⇒ off ≡ M14: a single terse answer, no initiative."""
    c = Conversation(ConsciousLoop(KnowledgeGraph(dim=32)), max_volunteer=0)
    c.say("everyone who is in the kitchen can see the window .")
    c.say("everyone who is in the kitchen can reach the door .")
    c.say("mary is in the kitchen .")
    assert c.say("what can mary see ?") == ["Mary can see the window."]


def test_volunteer_is_grounded():
    """A volunteered fact is never invented: it is in the real derivation closure and
    was not already stated."""
    from nsm_ct.reasoning_oracle import forward_chain
    c = _kitchen_two_abilities()
    cands = c._volunteer_candidates("mary", exclude=("mary", "CAN_SEE", "window"))
    known, _ = forward_chain(list(c._facts()), c._rules())
    said = set(c._facts())
    for f in cands:
        assert f in known and f not in said and f[3] == "+"


def test_volunteer_nothing_to_add_stays_terse():
    """With only the answered fact derivable about the subject, there is nothing useful
    to add → no volunteer (terse), not filler."""
    c = Conversation(ConsciousLoop(KnowledgeGraph(dim=32)), max_volunteer=1)
    c.say("everyone who is in the kitchen can see the window .")
    c.say("mary is in the kitchen .")
    assert c.say("what can mary see ?") == ["Mary can see the window."]
