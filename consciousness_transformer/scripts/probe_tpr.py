"""TPR fidelity/capacity probe on REAL meaning trees.

Measures whether Tensor Product Representations can hold our explication trees
non-flatteningly: encode → decode round-trip accuracy for

* the **matrix form** (one exact level, recursion via lossy contraction below), and
* the **fully contracted vector** (fixed d — what a model input would consume),

on (a) the gold + DeepNSM explication trees, (b) resolver meaning trees, and
(c) synthetic depth×branching sweeps. Also prints the memory cost of the *exact*
order-growing TPR — the empirical case for getting depth from the loop, not from
tensor order. The numbers gate model integration (see RESEARCH_NOTES).

Run:
    python scripts/probe_tpr.py [--dims 64 128 256]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nsm_ct.data_structures import ParseNode, ParseTree  # noqa: E402
from nsm_ct.explications import ExplicationStore  # noqa: E402
from nsm_ct.nsm_primes import PRIME_NAMES  # noqa: E402
from nsm_ct.tpr import TPRCodec  # noqa: E402


def roundtrip(codec: TPRCodec, tree: ParseTree, contracted: bool) -> float:
    """Label-recovery fraction for one tree (guided decode)."""
    m = codec.encode_matrix(tree.root)
    if contracted:
        m = codec.lift(codec.contract(m))
    correct, total = codec.decode_guided(m, tree.root)
    return correct / max(total, 1)


def synthetic(depth: int, branch: int, rng: np.random.Generator) -> ParseTree:
    """Random tree of prime-labelled nodes with given depth/branching."""
    def make(d: int) -> ParseNode:
        node = ParseNode(label=str(rng.choice(PRIME_NAMES)))
        if d > 0:
            node.children = [make(d - 1) for _ in range(branch)]
        return node
    return ParseTree(root=make(depth))


def main() -> None:
    ap = argparse.ArgumentParser(description="TPR fidelity/capacity probe")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    args = ap.parse_args()

    store = ExplicationStore.load()
    words = ["kill", "broke", "sad", "happy", "children",  # gold
             "snake", "egg", "water", "dog", "house"]      # DeepNSM (if store built)
    real_trees = []
    for w in words:
        hit = store.get(w)
        if hit:
            t = store.explication_to_tree(hit["explication"])
            real_trees.append((w, hit["provenance"].split(":")[0], t,
                               len(t.root.children)))

    for d in args.dims:
        codec = TPRCodec(dim=d)
        print(f"\n=== d={d} ===")
        if real_trees:
            print("real explication trees (label-recovery: matrix | contracted vec):")
            for w, prov, t, nch in real_trees:
                am = roundtrip(codec, t, contracted=False)
                av = roundtrip(codec, t, contracted=True)
                print(f"  {w:9s} [{prov:12s} children={nch:3d}]  matrix={am:.2f}  vec={av:.2f}")
        rng = np.random.default_rng(0)
        print("synthetic sweep (matrix | vec), 5 trees each:")
        for depth in (1, 2, 3):
            for branch in (2, 4, 8):
                ams, avs = [], []
                for _ in range(5):
                    t = synthetic(depth, branch, rng)
                    ams.append(roundtrip(codec, t, contracted=False))
                    avs.append(roundtrip(codec, t, contracted=True))
                print(f"  depth={depth} branch={branch}:  matrix={np.mean(ams):.2f}  vec={np.mean(avs):.2f}")

    print("\nexact order-growing TPR cost (float32, the depth problem):")
    for d in (16, 32, 64, 128):
        for depth in (1, 2, 3):
            size = 4 * d ** (depth + 1)
            human = (f"{size/1e9:.1f} GB" if size > 1e9 else
                     f"{size/1e6:.1f} MB" if size > 1e6 else f"{size/1e3:.0f} KB")
            print(f"  d={d:4d} depth={depth}: {human}")

    print("\nNOTE: 'matrix' = one exact level (recursion contracted below); "
          "'vec' = fully contracted fixed-d vector (model-input candidate).")


if __name__ == "__main__":
    main()
