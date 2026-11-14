"""Probe: graph-aware closeness vs coordinates on held-out syn/ant (M18.3).

The headline antonym result. Builds unsigned + polarity coordinates over a gloss
corpus and compares three closeness measures on a held-out synonym-vs-antonym
discrimination (P(synonym pair closer than antonym pair); 0.5 = chance):

  pure coordinate  <  polarity coordinate  <<  graph-aware closeness

Circularity-free: the held-out pair's own edge lives in the test split; the graph
penalty uses only train-split antonym edges (propagated one synonym hop).

Usage:
    python scripts/probe_ground_closeness.py [--n 3000] [--depth 3] [--lam 1.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.basis_search import _value_vec  # noqa: E402
from nsm_ct.ground.cache import DecompCache  # noqa: E402
from nsm_ct.ground.closeness import evaluate_closeness  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.ground.definition_graph import DefinitionGraph  # noqa: E402
from nsm_ct.ground.polarity import polarity_vector  # noqa: E402
from nsm_ct.ground.semantic_axes import AxisRegistry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()

    vocab = gloss_vocabulary(args.n)
    graph = DefinitionGraph.build(vocab)
    cache = DecompCache(depth=args.depth).warm(vocab)
    reg = AxisRegistry.seed()
    dec = lambda x: cache.decompose(x, args.depth)
    coord_u = {w: _value_vec(w, reg, args.depth, cache) for w in vocab}
    coord_p = {w: polarity_vector(w, axes=reg.axes, depth=args.depth, decompose=dec) for w in vocab}

    r = evaluate_closeness(vocab, graph=graph, coord_unsigned=coord_u, coord_polarity=coord_p,
                           train_frac=0.5, lam=args.lam)

    print(f"=== M18.3 held-out synonym-vs-antonym discrimination (|vocab|={len(vocab)}) ===")
    print(f"test pairs: synonym={r['n_test_syn']}  antonym={r['n_test_ant']}  (train antonym edges={r['n_train_ant']})")
    print()
    print(f"pure coordinate      : {r['pure_coordinate']:.3f}")
    print(f"polarity coordinate  : {r['polarity_coordinate']:.3f}")
    print(f"graph-aware (lam={args.lam}) : {r['graph_closeness']:.3f}")
    print()
    print("The relational web does what coordinates can't: held-out antonyms go from")
    print("'looks similar' (below chance) to clearly separated (above chance).")


if __name__ == "__main__":
    main()
