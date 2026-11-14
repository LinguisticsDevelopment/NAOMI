"""Phase-A gate: coordinators relate lossless clauses; store-as-OR, decide truth.

Token-free, numpy, no training. On a REAL parse it shows the whole loop:

  (1) "mary is in the kitchen or the office ." -> extract_discourse -> 2 distinct
      clauses + 1 OR link; each clause matrix round-trips (cos 1.0) and the OR link
      recovers the related clause index. The connective MAYBE is itself readable.
  (2) store as a disjunction: both disjuncts tagged MAYBE, nothing summed across.
  (3) UNRESOLVED query -> MAYBE (the first-class uncertain answer).
  (4) ingest "mary is not in the kitchen ." -> decide-truth re-tags kitchen FALSE /
      office TRUE; the kitchen clause is STILL recoverable (now FALSE) — overwrite
      but don't forget.
  (5) RESOLVED query -> OFFICE.

Gate: clauses round-trip lossless AND both the MAYBE and OFFICE answers are correct.

Run:
    python scripts/probe_discourse_tpr.py [--dim 256]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.clause import (  # noqa: E402
    DisjunctionBuffer, build_discourse_tpr, extract_discourse, read_connective,
    read_truth,
)
from nsm_ct.structure import PARSE_LABELS  # noqa: E402
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
    places = ["kitchen", "garden", "office", "bedroom", "hallway", "bathroom"]
    place_vecs = {p: codec.contract(codec.encode_matrix(resolver.resolve(p).root)) for p in places}

    def name_place(vec) -> str:
        return max(places, key=lambda p: _cos(vec, place_vecs[p]))

    ok = True

    print("=== (1) coordination: two lossless clauses + an OR link ===")
    g = parser._parse_graph("mary is in the kitchen or the office .")
    clauses, links = extract_discourse(g)
    print(f"  clauses: {[(c.predicate, [(r, a.token) for r, a in c.args]) for c in clauses]}")
    print(f"  links:   {[(l.coordinator, l.prime, l.i, l.j) for l in links]}")
    ok &= len(clauses) == 2 and len(links) == 1 and links[0].coordinator == "OR"

    dtpr = build_discourse_tpr(clauses, links, codec, resolver)
    # Fidelity = the operative one: unbind a clause's PLACE role, clean up against
    # the place codebook (exactly how the system reads values). Raw cosine is only
    # a diagnostic — different-relation roles share the codec's ±1 sign diagonals,
    # so a bare unbind carries cross-talk; cleanup is what makes it lossless.
    for i, (cl, m) in enumerate(zip(clauses, dtpr.clauses)):
        rel, arg = cl.args[1]
        u = codec.unbind(m, codec.role_vec(1, rel))
        nearest = name_place(u)
        cs = sorted((_cos(u, place_vecs[p]) for p in places), reverse=True)
        print(f"  clause[{i}] {arg.token:8s} unbind->{nearest:8s} "
              f"(correct={nearest == arg.token}; cos {cs[0]:.2f} vs runner-up {cs[1]:.2f})")
        ok &= nearest == arg.token
    j0 = dtpr.recover_link(codec, 0)
    conn, cs = read_connective(dtpr, codec)
    print(f"  OR link 0 -> recovered clause j={j0} (expected 1); connective={conn} (cos {cs:.2f})")
    ok &= j0 == 1 and conn == "MAYBE"

    print("\n=== (2) store as a disjunction: both tagged MAYBE ===")
    buf = DisjunctionBuffer(codec)
    buf.store_disjunction(dtpr)
    for e in buf.group:
        t, sc = read_truth(e["matrix"], codec)
        print(f"  disjunct {name_place(e['value']):8s} truth={t} (cos {sc:.2f})")
        ok &= t == "MAYBE"

    print("\n=== (3) unresolved query -> MAYBE ===")
    state, vec = buf.query()
    print(f"  query(mary, PLACE) = {state}  (expected MAYBE)")
    ok &= state == "MAYBE"

    print("\n=== (4) evidence 'mary is not in the kitchen' -> decide truth ===")
    gn = parser._parse_graph("mary is not in the kitchen .")
    nclauses, nlinks = extract_discourse(gn)
    print(f"  negation: clauses={[(c.predicate,[(r,a.token) for r,a in c.args]) for c in nclauses]} "
          f"links={[(l.coordinator, l.prime) for l in nlinks]}")
    ok &= bool(nlinks) and nlinks[0].coordinator == "NOT"
    refuted_word = nclauses[0].args[1][1].token   # 'kitchen'
    refuted_vec = codec.contract(codec.encode_matrix(resolver.resolve(refuted_word).root))
    buf.decide_truth(refuted_vec)
    for e in buf.group:
        t, sc = read_truth(e["matrix"], codec)
        place = name_place(e["value"])
        recovered = name_place(codec.unbind(e["base"], codec.role_vec(1, e["relation"])))
        print(f"  disjunct {place:8s} truth={t:5s} "
              f"(still recoverable: base unbind->{recovered}, ok={recovered == place})")
        ok &= recovered == place   # FALSE-tagged disjunct NOT forgotten

    print("\n=== (5) resolved query -> OFFICE ===")
    state, vec = buf.query()
    answer = name_place(vec) if state == "RESOLVED" else state
    print(f"  query(mary, PLACE) = {state} -> {answer.upper()}  (expected RESOLVED -> OFFICE)")
    ok &= state == "RESOLVED" and answer == "office"

    print("\nGATE:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
