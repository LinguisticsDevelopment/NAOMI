"""M14 headline gate (compact) — the long-conversation depth test as a fast unit gate.

One growing dialogue session: teach a body of facts/rules, interleave questions, and
assert — as the KB grows — that the system stays CORRECT vs the symbolic oracle, ASKS
for a missing premise and RESOLVES once told, ABSTAINS when unsupported, and stays
USABLE (bounded per-turn latency). The full-scale version (1000s of facts, latency
curve) is ``scripts/stress_converse.py``.
"""

from __future__ import annotations

import random
import time

from nsm_ct.mind import membrane
from nsm_ct.mind.conscious_loop import ConsciousLoop
from nsm_ct.mind.conversation import Conversation
from nsm_ct.mind.knowledge import KnowledgeGraph
from nsm_ct.reasoning_oracle import Rule, forward_chain

_VERBS = {"see": "CAN_SEE", "hold": "CAN_HOLD", "open": "CAN_OPEN", "reach": "CAN_REACH"}
_PLACES = ["kitchen", "garden", "office", "bedroom"]
_OBJECTS = ["window", "stove", "clock", "key"]


def _name(i: int) -> str:
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return "p" + s


def _cap_fact(s, r, v):
    t = membrane.render_fact(s, r, v)[:-2] + "."
    return t[:1].upper() + t[1:]


def test_long_conversation_depth():
    """Grounded back-and-forth stays correct, asks/resolves, and abstains as it learns."""
    rng = random.Random(0)
    conv = Conversation(ConsciousLoop(KnowledgeGraph(dim=32)))
    grant, rules = {}, []
    for i, place in enumerate(_PLACES):
        verb = list(_VERBS)[i % len(_VERBS)]
        grant[place] = (verb, _VERBS[verb], _OBJECTS[i])
        conv.say(f"everyone who is in the {place} can {verb} the {_OBJECTS[i]} .")
        rules.append(Rule((("?p", "PLACE", place, "+"),),
                          ("?p", _VERBS[verb], _OBJECTS[i], "+"), name="r"))

    facts, entities, n = [], [], 0
    rounds, per = 6, 12
    max_latency = 0.0
    for _ in range(rounds):
        for _ in range(per):                            # teach (KB grows)
            place = rng.choice(_PLACES)
            name = _name(n); n += 1
            conv.say(f"{name} is in the {place} .")
            facts.append((name, "PLACE", place, "+"))
            entities.append((name, place))
        for _ in range(per):                            # ask the granted ability
            name, place = rng.choice(entities)
            verb, rel, obj = grant[place]
            t0 = time.perf_counter()
            reply = conv.say(f"what can {name} {verb} ?")[0]
            max_latency = max(max_latency, time.perf_counter() - t0)
            assert reply == _cap_fact(name, rel, obj), reply   # correct vs oracle

    # everything answerable was answered correctly; closure is non-trivial.
    assert len(forward_chain(facts, rules)[0]) > rounds * per     # KB actually grew
    mix = {}
    for o in conv.log:
        mix[o.kind] = mix.get(o.kind, 0) + 1
    assert mix["answered"] == rounds * per
    assert max_latency < 1.0                            # stays usable

    # abstain: where-is on a fresh entity (no rule derives PLACE) → honest "I don't know"
    assert conv.say(f"where is {_name(n)} ?")[0] == "I don't know."
    n += 1

    # ask → tell → resolve at scale (late in the session)
    place = _PLACES[0]; verb, rel, obj = grant[place]
    name = _name(n); n += 1
    assert conv.say(f"what can {name} {verb} ?")[0].startswith("I can't tell")
    resolved = conv.say(f"{name} is in the {place} .")
    assert resolved == [f"Then yes — {name} can {verb} the {obj}."]
    assert conv.pending == []
