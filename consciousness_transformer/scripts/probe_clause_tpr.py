"""Token-free clause prototype: a clause as a TPR, and cross-clause correlation.

Shows (1) "mary is in the kitchen" assembled into ONE TPR matrix from a
variable(mary) + the prime-composed meaning of kitchen — no token embedding
anywhere — then decoded back; and (2) two clauses ("mary is in the kitchen . she
went to the office .") correlated through the shared mary-variable in an order-3
entity⊗relation⊗value memory, so "where is mary?" returns the UPDATED place.

Run:
    python scripts/probe_clause_tpr.py [--dim 256]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause import EntityMemory, EntityTracker, clause_tpr, decode_clause, extract_clauses  # noqa: E402
from nsm_ct.dataset import PARSE_LABELS  # noqa: E402
from nsm_ct.episode import CurriculumGenerator  # noqa: E402
from nsm_ct.input_encoder import ParserInputEncoder  # noqa: E402
from nsm_ct.meaning import NSMMeaningResolver  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.tokenizer import SimpleTokenizer  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def _cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()

    eps = CurriculumGenerator(max_level=6, seed=0).generate(12)
    texts = [t for e in eps for t in e.context + [e.question] + (e.options or [])]
    tok = SimpleTokenizer.build(texts, extra_tokens=list(PRIME_NAMES) + PARSE_LABELS)
    parser = ParserInputEncoder(tok)
    resolver = NSMMeaningResolver()
    codec = TPRCodec(dim=args.dim)

    print("=== (1) one clause, token-free ===")
    tree = parser._parse_tree("mary is in the kitchen .")
    clause = extract_clauses(tree)[0]
    print(f"  clause: predicate={clause.predicate!r} args="
          f"{[(r, a.token) for r, a in clause.args]}")
    m, triples = clause_tpr(clause, codec, resolver)
    print(f"  built {m.shape} TPR matrix from primes + variable(mary) — "
          f"no token embedding. ‖M‖={np.linalg.norm(m):.1f}")
    print("  decoded back:", decode_clause(m, clause, codec, resolver))

    print("\n=== (2) two clauses, correlated through the mary-variable ===")
    mem = EntityMemory(codec)
    tracker = EntityTracker()
    discourse = ["mary is in the kitchen .", "she went to the office ."]
    for sent in discourse:
        cl = extract_clauses(parser._parse_tree(sent))[0]
        _, trips = clause_tpr(cl, codec, resolver, tracker)
        for ent, rel, val in trips:
            mem.write(ent, rel, val)
            print(f"  wrote ({ent}, {rel}) from {sent!r}")

    # query "where is mary?" -> PLACE, decode by nearest place vector
    q = mem.query("mary", "PLACE")
    places = ["kitchen", "garden", "office", "bedroom", "hallway", "bathroom"]
    place_vecs = {p: codec.contract(codec.encode_matrix(resolver.resolve(p).root)) for p in places}
    ranked = sorted(places, key=lambda p: _cos(q, place_vecs[p]), reverse=True)
    print("  query(mary, PLACE) nearest places:",
          [(p, round(_cos(q, place_vecs[p]), 2)) for p in ranked[:3]])
    print(f"  -> answer: {ranked[0].upper()}  "
          f"(expected OFFICE — updated across clauses via the shared variable)")


if __name__ == "__main__":
    main()
