"""M42 diagnosis — WHY do ball/court/racket/yard fail even with gold senses?

M40 showed these families have benchmark ceiling 0.000 in RAW 607-axis space
(cell/mole/hood stuck ≈ 0.5): the gold sense's own signature ranks the WRONG
sense's answer word above its own. This probe prints, per family and sense:

- cos(sense signature, correct answer word) vs cos(sense, wrong answer word)
- the top named axes of the sense signature and of both answer words
- the axes driving the wrong-side overlap (largest elementwise product)

so we can see whether the failure is (a) shallow signatures (sense collapses
to generic axes like SOMETHING), (b) answer-word handles dominated by generic
axes, or (c) a genuinely misleading association test.

Usage: python scripts/probe_grounding_depth.py [--families ball,court,...]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.episode import _AMBIGUITY_FAMILIES  # noqa: E402
from nsm_ct.usvs_bridge import default_usvs, usvs_handle, usvs_sense_handle  # noqa: E402

DEAD = ("ball", "court", "racket", "yard")
STUCK = ("cell", "mole", "hood")


def top_axes(u, v: np.ndarray, k: int = 6) -> str:
    idx = np.argsort(-v)[:k]
    return ", ".join(f"{u.axes[i]}={v[i]:.2f}" for i in idx if v[i] > 1e-4)


def overlap_axes(u, a: np.ndarray, b: np.ndarray, k: int = 5) -> str:
    prod = a * b
    idx = np.argsort(-prod)[:k]
    return ", ".join(f"{u.axes[i]}={prod[i]:.3f}" for i in idx if prod[i] > 1e-5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(DEAD + STUCK))
    args = ap.parse_args()

    u = default_usvs()
    d = len(u.axes)  # raw named-axis mode

    for fam in args.families.split(","):
        spec = _AMBIGUITY_FAMILIES[fam]
        senses = spec["senses"]
        answers = {k: s["answer"] for k, s in senses.items()}
        print(f"\n=== {fam} (answers: {answers}) ===")
        ans_vecs = {}
        for key, s in senses.items():
            ans_vecs[key] = usvs_handle(s["answer"], d)
            if ans_vecs[key] is None:
                print(f"  !! answer word {s['answer']!r} UNKNOWN to USVS")
        for key, s in senses.items():
            sid = s["synset"]
            sv = usvs_sense_handle(sid, d)
            if sv is None:
                print(f"  {key} {sid}: NO SIGNATURE in USVS")
                continue
            other = [k for k in senses if k != key][0]
            print(f"  {key} {sid} (correct answer: {s['answer']!r})")
            print(f"     sense axes:  {top_axes(u, sv)}")
            for label, akey in (("CORRECT", key), ("WRONG  ", other)):
                av = ans_vecs.get(akey)
                if av is None:
                    continue
                c = float(np.dot(sv, av))
                print(f"     cos vs {label} {senses[akey]['answer']!r}: {c:.3f}"
                      f"   overlap: {overlap_axes(u, sv, av)}")


if __name__ == "__main__":
    main()
