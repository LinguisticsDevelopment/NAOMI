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
    print(f"\n[7] M20 normalization (dictionary reconstruction, held-out AUC):")
    print(f"    {'norm':12s} {'synonym':>8s} {'similar':>8s} {'hypernym':>9s} {'antonym':>8s}")
    for norm in ("raw", "standardize", "tanh"):
        d = evaluate_dictionary(g.words(), g, ax, cache=cache, depth=3, alpha=0.7, normalization=norm)
        print(f"    {norm:12s} {d['synonym_auc']:8.3f} {d['similar_auc']:8.3f} "
              f"{d['hypernym_auc']:9.3f} {d['antonym_auc']:8.3f}")
    print("    raw has spurious overlap (random-cosine 0.32); standardize keeps synonym/")
    print("    similar and fixes overlap; tanh bounds axes to [-1,1] + best antonym, costs synonym.")

    # [8] M23 — sense nodes + close-edge sweep (held-out; corrects M22's leaked sim>ant)
    import numpy as np
    from nsm_ct.ground.sense_graph import SenseGraph, build_sense_sparse
    from nsm_ct.ground.fusion import propagate, fused_similarity, _cos
    from nsm_ct.ground.closeness import split_pairs
    sg = SenseGraph.build(gloss_vocabulary(min(args.n, 1500)), max_senses_per_word=3, cap=4000)
    ssp = build_sense_sparse(sg, depth=2)
    sidx = {w: i for i, w in enumerate(ssp.words)}
    _pi = lambda ps: np.array([(sidx[a], sidx[b]) for a, b in ps if a in sidx and b in sidx and a != b])
    _, te_s = split_pairs(sg.typed_pairs("similar"), 0.5)
    _, te_a = split_pairs(sg.typed_pairs("antonym"), 0.5)
    S, A = _pi(te_s), _pi(te_a)
    tset = {tuple(sorted(p)) for p in te_s}
    rng = np.random.RandomState(0); Rn = rng.randint(0, len(ssp.words), (4000, 2))
    print(f"\n[8] M23 sense-node close-edge sweep (held-out; threshold gate 0.15):")
    print(f"    (M22's 0.63 sim>ant was leaked — test pairs were also propagation edges; below is honest)")
    print(f"    {'close-edge set':34s} {'sim>ant':>8s} {'random':>8s}")
    for tag, edges in [
        ("{similar+cohyponym} (M22 base)", sg.cohyponym_pairs() + sg.typed_pairs("similar")),
        ("{similar+deriv+mero}", sg.close_edges("similar", "derivational", "meronym")),
        ("{+gloss_overlap}", sg.close_edges("similar", "derivational", "meronym", "gloss_overlap")),
    ]:
        train = [p for p in edges if tuple(sorted(p)) not in tset]
        P = propagate(ssp, train, iters=20, alpha=0.7)
        d = float((fused_similarity(ssp, P, S, threshold=0.15)[:, None]
                   > fused_similarity(ssp, P, A, threshold=0.15)[None, :]).mean())
        r = float(fused_similarity(ssp, P, Rn, threshold=0.15).mean())
        print(f"    {tag:34s} {d:8.3f} {r:8.3f}")
    print("    deriv+mero is the best sense-node close set (beats cohyponym); gloss-overlap is noise.")
    print("    But held-out sim>ant ~0.50 (near chance) << word-graph — propagation only helps")
    print("    CONNECTED pairs; the generalizing win is the gate cutting random overlap (0.18->0.12).")

    # [9] M25 — joint all-signals TRAINED placement vs propagation (held-out): the negative
    from nsm_ct.ground.joint_place import joint_place, split_all_relations
    from nsm_ct.ground.normalize import normalize_matrix
    tr9, te9 = split_all_relations(g.words(), g, train_frac=0.5)
    prop9 = place(g.words(), g, ax, cache=cache, depth=3, alpha=0.7, train_pairs=tr9["syn"] + tr9["sim"])
    trained9 = joint_place(g.words(), g, ax, cache=cache, depth=3, train_syn=tr9["syn"], train_sim=tr9["sim"],
                           train_ant=tr9["ant"], train_hyp=tr9["hyp"], train_rel=tr9["rel"], iters=300)
    widx = {w: i for i, w in enumerate(g.words())}
    _pi = lambda p: np.array([(widx[a], widx[b]) for a, b in p if a in widx and b in widx and a != b])
    te9i = {k: _pi(te9[k]) for k in te9}
    rng9 = np.random.RandomState(0); R9 = rng9.randint(0, len(g.words()), (4000, 2))

    def _sc9(coord):
        M = normalize_matrix(np.stack([coord[w] for w in g.words()]), "tanh")
        nn = np.linalg.norm(M, axis=1, keepdims=True); nn[nn < 1e-9] = 1.0; P = M / nn
        def c(pr): return (P[pr[:, 0]] * P[pr[:, 1]]).sum(1) if len(pr) else np.array([])
        rnd, s, a = c(R9), c(te9i["syn"]), c(te9i["ant"])
        return float((s[:, None] > rnd[None, :]).mean()), float((s[:, None] > a[None, :]).mean()), float(rnd.mean())

    print(f"\n[9] M25 retrain-on-all-signals (held-out, tanh) — the honest negative:")
    print(f"    {'method':14s} {'synAUC':>7s} {'syn>ant':>8s} {'random':>7s}")
    for nm, cd in (("propagation", prop9), ("joint-trained", trained9)):
        syn, sa, rd = _sc9(cd)
        print(f"    {nm:14s} {syn:7.3f} {sa:8.3f} {rd:7.3f}")
    print("    free per-word fitting underperforms propagation on every held-out metric; more")
    print("    training overfits. Propagation generalizes (spreads via the graph); training does not.")


if __name__ == "__main__":
    main()
