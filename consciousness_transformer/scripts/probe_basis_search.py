"""Probe: MDL-driven basis discovery (M17.2).

Grows an interpretable basis from NSM-65 by Minimum Description Length over a
vocabulary, then reports the promoted axes, the MDL curve, and the relational
signals (grounding / antonym / synonym / hypernym) for seed vs. final.

By default the vocabulary is relationally closed (each seed word plus its
WordNet hypernym lemmas and antonyms) so the hypernym/antonym signals have data.

Usage:
    python scripts/probe_basis_search.py [--depth N] [--max-axes K] [--no-expand]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nsm_ct.ground.basis_search import search  # noqa: E402
from nsm_ct.ground.clause_self_consistency import SAMPLE_VOCAB  # noqa: E402
from nsm_ct.ground.definition_graph import DefinitionGraph  # noqa: E402
from nsm_ct.wordnet import antonyms, hypernyms  # noqa: E402


def _expand(seed):
    """Relationally close the vocab: add hypernym lemmas + antonyms (one hop)."""
    vocab = set(seed)
    for w in seed:
        for h in hypernyms(w):
            vocab.add(h.replace("_", " ").split()[0].lower())
        for a in antonyms(w):
            vocab.add(a.lower())
    return sorted(vocab)


def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--max-axes", type=int, default=15)
    ap.add_argument("--no-expand", action="store_true")
    args = ap.parse_args()

    vocab = SAMPLE_VOCAB if args.no_expand else _expand(SAMPLE_VOCAB)
    graph = DefinitionGraph.build(vocab)
    res = search(vocab, depth=args.depth, max_axes=args.max_axes, graph=graph)

    print(f"=== M17.2 basis discovery (depth={args.depth}, |vocab|={len(vocab)}) ===")
    print(f"seed axes: 65 (NSM)   promoted: {len(res.registry.beyond_seed())}")
    print()
    print(f"{'axis':14s} {'freq':>4s} {'mdl_gain':>9s}")
    for a, f, g in res.added:
        print(f"{a:14s} {f:4d} {g:9.1f}")
    print()
    print("MDL curve:", " -> ".join(f"{m:.0f}" for _, m in res.mdl_curve))
    print()
    sm, fm = res.seed_metrics, res.final_metrics
    print(f"{'metric':22s} {'seed':>8s} {'final':>8s}   n")
    print(f"{'grounding_rate':22s} {_fmt(sm['grounding_rate']):>8s} {_fmt(fm['grounding_rate']):>8s}")
    print(f"{'antonym_cos':22s} {_fmt(sm['antonym_cos']):>8s} {_fmt(fm['antonym_cos']):>8s}   {fm['n_antonym_pairs']}")
    print(f"{'synonym_cos':22s} {_fmt(sm['synonym_cos']):>8s} {_fmt(fm['synonym_cos']):>8s}   {fm['n_synonym_pairs']}")
    print(f"{'hypernym_containment':22s} {_fmt(sm['hypernym_containment']):>8s} {_fmt(fm['hypernym_containment']):>8s}   {fm['n_hypernym_pairs']}")
    print()
    print("Note: grounding + MDL improve by construction; the MDL-frequency basis")
    print("optimizes description length, not relatedness — antonym/synonym signals")
    print("are reported honestly and could weight selection in future work.")


if __name__ == "__main__":
    main()
