"""Probe: M28.0 — the WordNet-only baseline audit (Step A, dev/SEMANTIC_MAPPING_PLAN.md).

One table, one split policy: where the CURRENT signal set (synonym, antonym,
similar, is_a, meronym, derivational, verb_group + lexname/attribute axes) tops
out — overall and by POS region — before any new signal lands. Every Step B
signal is judged as a held-out delta against this table (M24 rule throughout:
placement propagates only over train pairs; scoring is on the disjoint test split).

Usage:
    python scripts/probe_m28_baseline.py [--n 3000] [--depth 3] [--alpha 0.7]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.axes import MeaningAxes  # noqa: E402
from nsm_ct.ground.cache import DecompCache  # noqa: E402
from nsm_ct.ground.closeness import discrimination, split_pairs  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.ground.dictionary import evaluate_dictionary  # noqa: E402
from nsm_ct.ground.meaning_value import cosine  # noqa: E402
from nsm_ct.ground.placement import evaluate_placement, place  # noqa: E402
from nsm_ct.ground.relations import RelationGraph  # noqa: E402
from nsm_ct.wordnet import senses  # noqa: E402

_POS_CACHE: dict = {}


def dom_pos(word: str) -> str:
    """Dominant POS = the word's first (most frequent) synset; 's' folds into 'a'."""
    if word not in _POS_CACHE:
        ss = senses(word)
        p = ss[0]["pos"] if ss else "?"
        _POS_CACHE[word] = "a" if p == "s" else p
    return _POS_CACHE[word]


def bucket(pair) -> str:
    pa, pb = dom_pos(pair[0]), dom_pos(pair[1])
    return f"{pa}-{pb}" if pa == pb else "mixed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.7)
    args = ap.parse_args()

    t0 = time.time()
    vocab = gloss_vocabulary(args.n)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=args.depth).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    print(f"=== M28.0 WordNet-only baseline (|vocab|={len(vocab)}, "
          f"axes={len(ax.names)}, build {time.time()-t0:.0f}s) ===")

    # -- overall placement (held-out syn>ant, plain cosine) ------------------
    res = evaluate_placement(vocab, g, ax, cache=cache, depth=args.depth,
                             alpha=args.alpha)
    print(f"\n[placement, alpha={args.alpha}] anchored {res['anchored']:.3f} -> "
          f"placed {res['placed']:.3f}  (test syn={res['n_test_syn']} ant={res['n_test_ant']})")

    # -- dictionary reconstruction (held-out AUC per relation) ---------------
    for norm in ("raw", "tanh"):
        dic = evaluate_dictionary(vocab, g, ax, cache=cache, depth=args.depth,
                                  normalization=norm)
        row = "  ".join(f"{k}={v:.3f}" for k, v in sorted(dic.items())
                        if isinstance(v, float))
        print(f"[dictionary, {norm}] {row}")

    # -- POS-region breakdown (mirrors evaluate_placement's split exactly) ---
    words = list(vocab)
    wset = set(words)
    syn = sorted({tuple(sorted((w, s))) for w in words for s in g.synonym.get(w, [])
                  if s in wset and s != w})
    sim = sorted({tuple(sorted((w, s))) for w in words for s in g.similar.get(w, [])
                  if s in wset and s != w})
    ant = sorted({tuple(sorted(p)) for p in g.typed_pairs("antonym")})
    train_syn, test_syn = split_pairs(syn, 0.5)
    train_sim, _ = split_pairs(sim, 0.5)
    _, test_ant = split_pairs(ant, 0.5)
    placed = place(words, g, ax, cache=cache, depth=args.depth,
                   train_pairs=train_syn + train_sim, iters=20, alpha=args.alpha)

    by_pos: dict = {}
    for w in words:
        by_pos.setdefault(dom_pos(w), []).append(w)
    rng = random.Random(0)

    print(f"\n[POS regions, alpha={args.alpha}]  (syn>ant needs in-bucket antonyms; "
          f"syn>rand is the coverage-independent check)")
    print(f"{'bucket':7} {'#syn':>5} {'#ant':>5}  {'syn>ant':>7}  {'syn>rand':>8}")
    for b in ("n-n", "v-v", "a-a", "r-r", "mixed"):
        ts = [p for p in test_syn if bucket(p) == b]
        ta = [p for p in test_ant if bucket(p) == b]
        if not ts:
            continue
        d_ant, ns, na = discrimination(
            ts, ta, lambda a, c: cosine(placed[a], placed[c]), placed)
        if b == "mixed":
            pool_a = pool_b = words
        else:
            pool_a = pool_b = by_pos.get(b[0], [])
        rand_pairs = []
        while len(rand_pairs) < 2000 and len(pool_a) > 1:
            a, c = rng.choice(pool_a), rng.choice(pool_b)
            if a != c:
                rand_pairs.append((a, c))
        d_rand, _, _ = discrimination(
            ts, rand_pairs, lambda a, c: cosine(placed[a], placed[c]), placed)
        ant_s = f"{d_ant:7.3f}" if na else "     --"
        print(f"{b:7} {ns:5d} {na:5d}  {ant_s}  {d_rand:8.3f}")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
