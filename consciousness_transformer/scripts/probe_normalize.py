"""Probe: which normalization fixes the overlap + incommensurability (M20.0).

Builds the space once, then compares normalization transforms on held-out
synonym/antonym/random cosine and syn>ant discrimination. The target: random-pair
similarity drops toward 0 (unrelated pairs stop overlapping) while synonym >> random
and syn>ant discrimination holds or improves.

Usage: python scripts/probe_normalize.py [--n 3000] [--alpha 0.7]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.axes import MeaningAxes  # noqa: E402
from nsm_ct.ground.cache import DecompCache  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.ground.closeness import split_pairs  # noqa: E402
from nsm_ct.ground.normalize import normalize_matrix  # noqa: E402
from nsm_ct.ground.placement import place  # noqa: E402
from nsm_ct.ground.relations import RelationGraph  # noqa: E402


def _norm_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    return M / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=0.7)
    args = ap.parse_args()

    t = time.time()
    vocab = gloss_vocabulary(args.n)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    words = g.words()
    idx = {w: i for i, w in enumerate(words)}
    wset = set(words)

    def pairidx(pairs):
        return np.array([(idx[a], idx[b]) for a, b in pairs if a in idx and b in idx and a != b])

    # HELD-OUT (M24 leakage fix): propagate over TRAIN closeness edges; score on the
    # disjoint test_syn / test_ant. Previously this placed over ALL synonym pairs AND
    # scored on all synonym pairs, inflating syn>ant discrimination (tanh 0.94 -> ~0.76).
    syn_all = sorted({tuple(sorted((w, s))) for w in words for s in g.synonym.get(w, []) if s in wset and s != w})
    sim_all = sorted({tuple(sorted((w, s))) for w in words for s in g.similar.get(w, []) if s in wset and s != w})
    ant_all = sorted({tuple(sorted(p)) for p in g.typed_pairs("antonym")})
    train_syn, test_syn = split_pairs(syn_all, 0.5)
    train_sim, _ = split_pairs(sim_all, 0.5)
    _, test_ant = split_pairs(ant_all, 0.5)
    placed = place(words, g, ax, cache=cache, depth=3, alpha=args.alpha,
                   train_pairs=train_syn + train_sim)
    P = np.stack([placed[w] for w in words])

    syn = pairidx(test_syn)
    ant = pairidx(test_ant)
    rng = np.random.RandomState(0)
    rand = rng.randint(0, len(words), (4000, 2))

    print(f"=== M20.0 normalization comparison (|vocab|={len(words)}, {time.time()-t:.0f}s) ===")
    print(f"{'transform':12s} {'syn':>7s} {'ant':>7s} {'random':>7s} {'syn>ant':>8s}")
    for kind in ("raw", "center", "standardize", "minmax", "tanh"):
        N = _norm_rows(normalize_matrix(P, kind))

        def mc(pr):
            return float((N[pr[:, 0]] * N[pr[:, 1]]).sum(1).mean()) if len(pr) else 0.0

        ss = (N[syn[:, 0]] * N[syn[:, 1]]).sum(1)
        aa = (N[ant[:, 0]] * N[ant[:, 1]]).sum(1)
        disc = float((ss[:, None] > aa[None, :]).mean())
        print(f"{kind:12s} {mc(syn):7.3f} {mc(ant):7.3f} {mc(rand):7.3f} {disc:8.3f}")
    print("\ntarget: random -> ~0 (unrelated pairs stop overlapping), syn >> random, syn>ant high.")


if __name__ == "__main__":
    main()
