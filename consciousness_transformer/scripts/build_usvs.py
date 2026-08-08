"""Build and publish USVS — the Universal Semantic Vector Space (M29 / Step C).

Produces the versioned artifact directory (data/usvs/) + the human-browsable
English sense->coordinate dictionary (data/usvs/dictionary.jsonl.gz).

Usage:
    python scripts/build_usvs.py [--n-core 10000] [--max-senses N] [--out data/usvs]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.usvs import build_usvs, export_dictionary, load_usvs, save_usvs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-core", type=int, default=10_000)
    ap.add_argument("--max-senses", type=int, default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "usvs"))
    ap.add_argument("--no-dictionary", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    u = build_usvs(n_core=args.n_core, max_senses=args.max_senses,
                   log=lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True))
    out = save_usvs(u, args.out)
    print(f"[{time.time()-t0:6.0f}s] saved -> {out}  fingerprint={u.fingerprint}")
    for k, v in u.meta["counts"].items():
        print(f"  {k}: {v}")

    # load-back sanity + timing
    t1 = time.time()
    u2 = load_usvs(out)
    assert u2.fingerprint == u.fingerprint
    print(f"load-back ok in {time.time()-t1:.1f}s")

    if not args.no_dictionary:
        n = export_dictionary(u, Path(args.out) / "dictionary.jsonl.gz")
        print(f"[{time.time()-t0:6.0f}s] dictionary.jsonl.gz: {n} senses")

    # spot checks
    for a, b in (("dog", "puppy"), ("dog", "justice"), ("hot", "cold")):
        print(f"  sim({a},{b}) = {u.similarity(a, b):.3f}")
    print(f"  antonyms(hot) = {u.antonyms_of('hot')[:6]}")
    print(f"  genus(dog) = {u.genus_of('dog')}")
    print(f"  senses(bank) = {u.senses_of('bank')[:4]}")


if __name__ == "__main__":
    main()
