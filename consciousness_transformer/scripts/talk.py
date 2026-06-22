"""M13 demo: talk to it in controlled English (teach, then ask).

No training, no controller needed for these wh + forward-chain cases: the recursive
grammar parses English → meaning clauses, ConsciousLoop.consume reasons over them, and
the answer is rendered back to English. Run: ``python scripts/talk.py``.
"""
import sys

sys.path.insert(0, "src")
from nsm_ct.mind.conscious_loop import ConsciousLoop
from nsm_ct.mind.knowledge import KnowledgeGraph


def _say(title, lines):
    loop = ConsciousLoop(KnowledgeGraph(dim=32))
    print(f"\n=== {title} ===")
    for ln in lines:
        print(f"  > {ln}")
    for reply in loop.converse(lines):
        print(f"  ⇒ {reply}")


_say("a universal + a fact, then a question", [
    "everyone who is in the kitchen can see the window .",
    "mary is in the kitchen .",
    "what can mary see ?",
])

_say("a grounded conditional (about mary specifically)", [
    "if mary is in the garden , mary can hold the stove .",
    "mary is in the garden .",
    "what can mary hold ?",
])

_say("a restrictive relative as a universal", [
    "the dog that is in the bedroom can reach the clock .",
    "fido is in the bedroom .",
    "what can fido reach ?",
])

_say("pronoun resolved by gender agreement (she = mary, not john)", [
    "john is in the office .",
    "mary is in the garden .",
    "everyone who is in the garden can hold the stove .",
    "what can she hold ?",
])

_say("abstain when nothing supports it", [
    "where is sandra ?",
])
