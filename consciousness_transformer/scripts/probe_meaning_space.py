"""Probe: the unified meaning space, end to end (M19 headline).

One report tying the phases together: from every relational signal, find the
minimum NAMED axes, place words in that one space, and reconstruct the dictionary
as geometry — validated held-out.

Usage:
    python scripts/probe_meaning_space.py [--n 3000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.axes import MeaningAxes, dimensionality_spectrum  # noqa: E402
from nsm_ct.ground.cache import DecompCache  # noqa: E402
from nsm_ct.ground.corpus import gloss_vocabulary  # noqa: E402
from nsm_ct.ground.dictionary import evaluate_dictionary, novel_synonyms  # noqa: E402
from nsm_ct.ground.minimality import minimal_axes  # noqa: E402
from nsm_ct.ground.placement import evaluate_placement, place  # noqa: E402
from nsm_ct.ground.relations import RelationGraph  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    t = time.time()
    vocab = gloss_vocabulary(args.n)
    g = RelationGraph.build(vocab)
    cache = DecompCache(depth=3).warm(vocab)
    ax = MeaningAxes.assemble(g, min_attribute_freq=2)

    spec = dimensionality_spectrum(g, ax, cache=cache, depth=3)
    plc = evaluate_placement(g.words(), g, ax, cache=cache, depth=3, alpha=0.7)
    mn = minimal_axes(g.words(), g, ax, cache=cache, depth=3, alpha=0.7, keep_frac=0.95)
    dic = evaluate_dictionary(g.words(), g, ax, cache=cache, depth=3, alpha=0.7)
    placed = place(g.words(), g, ax, cache=cache, depth=3, alpha=0.7)
    dt = time.time() - t

    s = ax.summary()
    print(f"=== THE UNIFIED MEANING SPACE ({len(g.words())} words, {dt:.0f}s) ===\n")
    print("[1] every source of signal -> one relation store")
    print(f"    coverage: " + ", ".join(f"{k} {v:.2f}" for k, v in g.coverage().items()))
    print(f"\n[2] candidate NAMED axes: {s['total']} "
          f"(primes {s['prime']} + attribute {s['attribute']} + lexname {s['lexname']})")
    print(f"    intrinsic dim ~{spec['intrinsic_dim_mass']} (90% energy), "
          f"effective ~{spec['participation_ratio']:.0f}")
    print(f"\n[3] minimum axes that reproduce relations: {mn['minimal_k']} "
          f"(95% of full fidelity {mn['full_discrimination']:.2f})")
    print(f"    kept (named): {', '.join(mn['kept_axes'][:12])} ...")
    print(f"\n[4] placement folds relations into position (held-out, PLAIN cosine):")
    print(f"    syn-vs-ant  anchored {plc['anchored']:.3f} -> placed {plc['placed']:.3f}"
          f"   (M18.3 penalty baseline 0.64)")
    print(f"\n[5] dictionary reconstructed from grounded space (held-out AUC, 0.5=chance):")
    print(f"    synonym {dic['synonym_auc']:.3f} | similar {dic['similar_auc']:.3f} | "
          f"hypernym {dic['hypernym_auc']:.3f} | antonym(by distance) {dic['antonym_auc']:.3f}")
    print(f"    (antonyms are near-but-opposite; the proper antonym metric is [4] = {plc['placed']:.2f})")
    print(f"\n[6] novel relationships the space proposes (spot-check; mostly noise at this fidelity):")
    for a, b, sc in novel_synonyms(g.words(), g, placed, top=6):
        print(f"    {a} ~ {b}  ({sc:.2f})")

    # [7] M20 — normalization re-audit (fix spurious overlap; the trade-off is honest)
    from nsm_ct.ground.dictionary import evaluate_dictionary
    print(f"\n[7] M20 normalization (dictionary reconstruction, held-out AUC):")
    print(f"    {'norm':12s} {'synonym':>8s} {'similar':>8s} {'hypernym':>9s} {'antonym':>8s}")
    for norm in ("raw", "standardize", "tanh"):
        d = evaluate_dictionary(g.words(), g, ax, cache=cache, depth=3, alpha=0.7, normalization=norm)
        print(f"    {norm:12s} {d['synonym_auc']:8.3f} {d['similar_auc']:8.3f} "
              f"{d['hypernym_auc']:9.3f} {d['antonym_auc']:8.3f}")
    print("    raw has spurious overlap (random-cosine 0.32); standardize keeps synonym/")
    print("    similar and fixes overlap; tanh bounds axes to [-1,1] + best antonym, costs synonym.")


if __name__ == "__main__":
    main()
