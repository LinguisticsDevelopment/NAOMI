"""M14 headline gate — the LONG-CONVERSATION depth test.

Not a unit test: ONE growing dialogue session that teaches a steadily larger body of
facts (KB 10s → 100s → 1000s of derivable facts) with questions interleaved
throughout, and checks — *along the way*, as the knowledge base grows — that the system

  * stays CORRECT (every answer matches the symbolic oracle, the trust floor),
  * ASKS for the missing premise when blocked and RESOLVES once told,
  * ABSTAINS only when nothing supports the query,
  * stays USABLE (per-turn derive latency reported as the KB grows).

It also prints the per-turn ``TurnOutcome`` log mix (answered / asked / resolved /
abstained / learned) — the reward/telemetry stream a future learned drive (L6) would
optimize. Run: ``python scripts/stress_converse.py [--rounds N] [--per-round K]``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.mind import membrane                                    # noqa: E402
from nsm_ct.mind.conscious_loop import ConsciousLoop               # noqa: E402
from nsm_ct.mind.conversation import Conversation                  # noqa: E402
from nsm_ct.mind.knowledge import KnowledgeGraph                   # noqa: E402
from nsm_ct.reasoning_oracle import Rule, forward_chain            # noqa: E402

_VERBS = {"see": "CAN_SEE", "hold": "CAN_HOLD", "open": "CAN_OPEN", "reach": "CAN_REACH"}
_PLACES = ["kitchen", "garden", "office", "bedroom", "bathroom", "hallway", "attic", "cellar"]
_OBJECTS = ["window", "stove", "clock", "key", "door", "lamp", "book", "chair"]


def _name(i: int) -> str:
    """A distinct alphabetic-only entity name (the controlled tokenizer drops digits,
    so names must be letters): ``pa, pb, …, pz, paa, …`` — never a function word."""
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return "p" + s


def _cap_fact(s, r, v):
    t = membrane.render_fact(s, r, v)[:-2] + "."
    return t[:1].upper() + t[1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--per-round", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    conv = Conversation(ConsciousLoop(KnowledgeGraph(dim=32)))

    # A FIXED rule set: each place grants exactly one ability (constant rule-count keeps
    # the latency curve a clean function of *fact* growth). place -> (verb, rel, obj).
    grant = {}
    for i, place in enumerate(_PLACES):
        verb = list(_VERBS)[i % len(_VERBS)]
        obj = _OBJECTS[i]
        grant[place] = (verb, _VERBS[verb], obj)
        conv.say(f"everyone who is in the {place} can {verb} the {obj} .")

    facts = []                                                     # shadow oracle (4-tuples)
    rules = [Rule((("?p", "PLACE", p, "+"),), ("?p", g[1], g[2], "+"), name="r")
             for p, g in grant.items()]
    entities = []                                                  # (name, place)
    n = 0

    correct = wrong = asked_ok = abstained_ok = 0
    wrong_examples, latencies = [], []

    for rnd in range(args.rounds):
        # teach a batch of new entities — the KB grows
        for _ in range(args.per_round):
            place = rng.choice(_PLACES)
            name = _name(n); n += 1
            conv.say(f"{name} is in the {place} .")
            facts.append((name, "PLACE", place, "+"))
            entities.append((name, place))

        # ask each entity its GRANTED ability (answerable) — correctness + latency,
        # checked against the symbolic oracle; pending stays empty on this path.
        for _ in range(args.per_round):
            name, place = rng.choice(entities)
            verb, rel, obj = grant[place]
            t0 = time.perf_counter()
            reply = conv.say(f"what can {name} {verb} ?")[0]
            latencies.append((len(facts), time.perf_counter() - t0))
            want = _cap_fact(name, rel, obj)
            if reply == want:
                correct += 1
            else:
                wrong += 1
                if len(wrong_examples) < 5:
                    wrong_examples.append((name, verb, reply, want))

        # one abstain probe per round: where-is on a fresh, unplaced entity (no rule
        # derives PLACE) → honest "I don't know", not a spurious question.
        fresh = _name(n); n += 1
        if conv.say(f"where is {fresh} ?")[0] == "I don't know.":
            abstained_ok += 1

    # the ask -> tell -> resolve loop AT SCALE (a held-back premise late in the session)
    place = _PLACES[0]
    verb, rel, obj = grant[place]
    name = _name(n); n += 1
    ask = conv.say(f"what can {name} {verb} ?")[0]
    assert ask.startswith("I can't tell"), ask
    asked_ok += 1
    res = conv.say(f"{name} is in the {place} .")
    resolved = [r for r in res if r.startswith("Then yes")]

    # ---- report ----
    total = correct + wrong
    kb = len(forward_chain(facts, rules)[0])
    print(f"\n=== long-conversation depth test: {args.rounds} rounds × {args.per_round} ===")
    print(f"final KB size (derivable facts): {kb}   (taught statements: {len(conv.statements)})")
    print(f"answerable queries:  {correct}/{total} correct vs oracle  "
          f"({correct / max(1, total):.3f})")
    print(f"blocked queries:     asked-for-premise={asked_ok}  abstained={abstained_ok}")
    print(f"ask→tell→resolve at scale: {len(resolved)} pending resolved on the tell "
          f"(e.g. {resolved[0]!r})" if resolved else "ask→tell→resolve: NONE resolved")
    mix = Counter(o.kind for o in conv.log)
    print(f"TurnOutcome mix (L6 signal): {dict(mix)}")
    # latency curve: median turn time in the first vs last quartile of KB growth
    latencies.sort()
    q = max(1, len(latencies) // 4)
    early = sorted(t for _, t in latencies[:q])[q // 2]
    late = sorted(t for _, t in latencies[-q:])[q // 2]
    print(f"derive latency / query: early-KB(~{latencies[0][0]} facts) median={early * 1e3:.2f}ms"
          f"   late-KB(~{latencies[-1][0]} facts) median={late * 1e3:.2f}ms")
    for (nm, v, got, want) in wrong_examples:
        print(f"  WRONG  what can {nm} {v}? -> {got!r}  (oracle wanted {want!r})")

    ok = (total > 0 and wrong == 0 and resolved and abstained_ok == args.rounds)
    print("\nGATE", "OK" if ok else "FAIL",
          "— grounded back-and-forth stays correct, asks/resolves, and abstains as the KB grows.")


if __name__ == "__main__":
    main()
