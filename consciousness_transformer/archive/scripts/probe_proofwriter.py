"""Probe: does the mind substrate reproduce ProofWriter's gold labels? (M8)

Forward-chains (OWA) over real ProofWriter theories and compares the
True/False/Unknown verdict to the dataset's gold answer, broken down by reasoning
depth — the trust anchor that the architecture's reasoning matches a real, broad
dataset before any training. Requires ``scripts/fetch_proofwriter.py`` first.

Run:  python scripts/probe_proofwriter.py [--per-depth 150]
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.mind.datasets import proofwriter as pw  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-depth", type=int, default=150)
    args = ap.parse_args()
    base = pw.default_data_dir()

    ok = total = 0
    bydepth = collections.defaultdict(lambda: [0, 0])
    byans = collections.defaultdict(lambda: [0, 0])
    for depth in ("0", "1", "2", "3", "5"):
        path = os.path.join(base, f"owa-depth{depth}-test.jsonl")
        if not os.path.exists(path):
            print(f"missing {path} — run scripts/fetch_proofwriter.py")
            return
        for rec in pw.load_records(path, limit=args.per_depth):
            ex = pw.parse_record(rec)
            for (lit, gold, qd) in ex.questions:
                hit = pw.verify(ex.facts, ex.rules, lit) == gold
                ok += hit; total += 1
                bydepth[qd][0] += hit; bydepth[qd][1] += 1
                byans[gold][0] += hit; byans[gold][1] += 1

    print(f"ProofWriter OWA parity (forward-chain vs gold): {ok}/{total} = {ok/total:.4f}")
    print("by reasoning depth:")
    for d in sorted(bydepth):
        c, n = bydepth[d]
        print(f"  QDep {d}: {c}/{n} = {c/n:.3f}")
    print("by gold answer:")
    for a in (pw.TRUE, pw.FALSE, pw.UNKNOWN):
        c, n = byans[a]
        print(f"  {a:8}: {c}/{max(n,1)} = {c/max(n,1):.3f}")


if __name__ == "__main__":
    main()
