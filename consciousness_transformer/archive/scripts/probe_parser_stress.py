"""Parser stress battery — where does the real pipeline break on real-ish English?

Runs sentences through the SAME path the curriculum uses
(``ParserInputEncoder._parse_graph`` -> ``nsm_ct.clause.extract_discourse``)
and classifies each case:

  PASS       parsed, a clause came out, and every expected (role -> token)
             pair is present on one clause
  WRONG      parsed and produced clauses, but no clause matches the expected
             roles (structure came out wrong)
  NO_CLAUSE  parsed, but extract_discourse produced zero clauses
  NO_PARSE   quantum_parser produced no hypothesis at all

Categories target the known/suspected gaps (subordinate clauses, referents,
preposition long-tail, passives, open vocabulary) rather than re-proving the
20 verified curriculum templates — those appear only as a small sanity block.

Usage:  python scripts/probe_parser_stress.py [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.clause import extract_discourse
from nsm_ct.input_encoder import ParserInputEncoder
from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.structure import PARSE_LABELS
from nsm_ct.tokenizer import SimpleTokenizer

# (category, sentence, expected {role: token} or None for "any clause").
# Role names are clause.py relations (SUBJECT/OBJECT/PLACE/SOURCE/...) plus
# the pseudo-role "PRED" for the clause predicate token.
BATTERY = [
    # -- sanity: verified curriculum shapes ---------------------------------
    ("sanity", "mary is in the garden .", {"SUBJECT": "mary", "PLACE": "garden"}),
    ("sanity", "mary went to the garden .", {"SUBJECT": "mary", "PLACE": "garden"}),
    ("sanity", "mary entered the garden .", {"SUBJECT": "mary", "PLACE": "garden"}),

    # -- passives (M38 landed SubType.PASSIVE) ------------------------------
    ("passive", "the ball was found in the garden .", {"SUBJECT": "ball", "PLACE": "garden"}),
    ("passive", "the ball was moved to the shed .", {"SUBJECT": "ball", "PLACE": "shed"}),
    ("passive", "the ball was found by mary .", {"SUBJECT": "ball"}),  # agentive-by: PLACE mislabel is the documented landmine
    ("passive", "the window was broken .", {"SUBJECT": "window", "PRED": "broken"}),

    # -- preposition long-tail ----------------------------------------------
    ("prep", "the ball is on the table .", {"SUBJECT": "ball", "PLACE": "table"}),
    ("prep", "the ball is at the door .", {"SUBJECT": "ball", "PLACE": "door"}),
    ("prep", "the ball is inside the box .", {"SUBJECT": "ball", "PLACE": "box"}),
    ("prep", "the ball is near the door .", {"SUBJECT": "ball", "PLACE": "door"}),
    ("prep", "the ball is under the table .", {"SUBJECT": "ball", "UNDER": "table"}),
    ("prep", "the ball is behind the door .", {"SUBJECT": "ball", "BEHIND": "door"}),
    ("prep", "the ball is beside the box .", {"SUBJECT": "ball", "BESIDE": "box"}),
    ("prep", "the ball came from the shed .", {"SUBJECT": "ball", "SOURCE": "shed"}),

    # -- subordinate clauses -------------------------------------------------
    ("subord", "mary said that the ball is in the garden .", {"SUBJECT": "ball", "PLACE": "garden"}),
    ("subord", "mary knows that john is in the kitchen .", {"SUBJECT": "john", "PLACE": "kitchen"}),
    ("subord", "mary thinks the ball is in the shed .", {"SUBJECT": "ball", "PLACE": "shed"}),

    # -- relative clauses (rel2 rule exists) ---------------------------------
    ("relative", "the ball that mary found is in the garden .", {"SUBJECT": "ball", "PLACE": "garden"}),
    ("relative", "the man who came is in the kitchen .", {"SUBJECT": "man", "PLACE": "kitchen"}),

    # -- pronouns / referents -------------------------------------------------
    ("pronoun", "she is in the garden .", {"SUBJECT": "she", "PLACE": "garden"}),
    ("pronoun", "mary went to the garden . she found the ball .", {"SUBJECT": "she", "OBJECT": "ball"}),
    ("pronoun", "it is in the box .", {"SUBJECT": "it", "PLACE": "box"}),

    # -- conjunctions ----------------------------------------------------------
    ("conj", "mary and john are in the garden .", {"PLACE": "garden"}),
    ("conj", "the ball is in the garden and the bat is in the shed .", {"SUBJECT": "bat", "PLACE": "shed"}),
    ("conj", "mary went to the garden and found the ball .", {"SUBJECT": "mary"}),

    # -- negation ---------------------------------------------------------------
    ("negation", "the ball is not in the garden .", {"SUBJECT": "ball", "PLACE": "garden"}),
    ("negation", "mary did not go to the garden .", {"SUBJECT": "mary"}),

    # -- questions ---------------------------------------------------------------
    ("question", "where is the ball ?", {"SUBJECT": "ball"}),
    ("question", "is the ball in the garden ?", {"SUBJECT": "ball", "PLACE": "garden"}),

    # -- open vocabulary (words not in WORD_TAG_DICT) ----------------------------
    ("open_vocab", "the zeppelin is in the hangar .", {"SUBJECT": "zeppelin", "PLACE": "hangar"}),
    ("open_vocab", "the wombat is near the fountain .", {"SUBJECT": "wombat", "PLACE": "fountain"}),
    ("open_vocab", "gertrude wandered into the arboretum .", {"SUBJECT": "gertrude", "PLACE": "arboretum"}),
    ("open_vocab", "the quokka sat on the bench .", {"SUBJECT": "quokka", "PLACE": "bench"}),
]


def clause_roles(cl) -> dict:
    roles = {"PRED": (cl.predicate or "").lower()}
    for rel, arg in cl.args:
        roles.setdefault(rel, (arg.token or "").lower())
    return roles


def classify(parser, sentence: str, expect):
    graph = parser._parse_graph(sentence)
    if graph is None:
        return "NO_PARSE", []
    clauses, _links = extract_discourse(graph)
    seen = [clause_roles(c) for c in clauses]
    if not clauses:
        return "NO_CLAUSE", seen
    if expect is None:
        return "PASS", seen
    for roles in seen:
        if all(roles.get(k) == v for k, v in expect.items()):
            return "PASS", seen
    return "WRONG", seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print clause dumps for every case")
    args = ap.parse_args()

    texts = [s for _, s, _ in BATTERY]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    if getattr(parser, "_parser", None) is None:
        print("quantum_parser unavailable; aborting")
        return 1

    by_cat = defaultdict(lambda: defaultdict(int))
    failures = []
    for cat, sent, expect in BATTERY:
        outcome, seen = classify(parser, sent, expect)
        by_cat[cat][outcome] += 1
        mark = {"PASS": ".", "WRONG": "W", "NO_CLAUSE": "C", "NO_PARSE": "X"}[outcome]
        line = f"  [{mark}] {cat:<10} {sent}"
        print(line)
        if outcome != "PASS" or args.verbose:
            for roles in seen:
                print(f"        clause: {roles}")
            if outcome != "PASS":
                failures.append((cat, sent, expect, seen))

    print("\n=== per-category ===")
    print(f"{'category':<12} {'pass':>4} {'wrong':>5} {'no_clause':>9} {'no_parse':>8}  rate")
    total_pass = total = 0
    for cat in dict.fromkeys(c for c, _, _ in BATTERY):
        d = by_cat[cat]
        n = sum(d.values())
        p = d["PASS"]
        total += n
        total_pass += p
        print(f"{cat:<12} {p:>4} {d['WRONG']:>5} {d['NO_CLAUSE']:>9} {d['NO_PARSE']:>8}  {p}/{n}")
    print(f"{'TOTAL':<12} {total_pass:>4} {'':>5} {'':>9} {'':>8}  {total_pass}/{total} = {total_pass/total:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
