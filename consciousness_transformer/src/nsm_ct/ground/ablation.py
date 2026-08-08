"""The Step B ablation harness (dev/SEMANTIC_MAPPING_PLAN.md).

One honest, reusable judge for every candidate signal: compute the M28.0 metric
table with and without a signal's extras, across train-split jitters, and report
deltas against the noise band. A signal "moves" a metric only if |delta| exceeds
2x the baseline's jitter band.

M24 rule is enforced by construction: extra close edges are filtered against
every scored test split before they enter propagation, and held-out antonym
scoring always uses the ORIGINAL antonym pairs' test split (expanded antonym
pairs are reported separately, never mixed into the original score).

Signals plug in three ways (any combination):
- ``close_extra``   — word pairs unioned into the train-side propagation edges
                      (the lever that moved everything since M19.2).
- ``antonym_extra``  — word pairs added to the antonym relation (grows the signed
                      edge store; original held-out score must not regress).
- ``feature_extra`` — word -> [category] merged into the attribute relation, so
                      categories become candidate named axes (like lexname).

Signal modules follow the convention ``nsm_ct.ground.signal_<name>`` exposing
``extras(vocab, graph) -> dict`` with any of the three keys above (see
``scripts/ablate_signal.py``). Coordinates are scored raw (like-for-like deltas;
normalization is an orthogonal, already-settled choice — M20).
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .axes import MeaningAxes
from .closeness import discrimination, split_pairs
from .dictionary import _auc, _random_pairs
from .meaning_value import cosine
from .placement import anchored_coordinate, place
from .relations import RelationGraph

Pair = Tuple[str, str]

# The metrics a signal is judged on (deltas reported for each).
# hypernym_auc = anchored-feature containment (M19.4) — placement-INDEPENDENT
#   (only feature_extra signals can move it).
# hypernym_cos_auc = held-out is_a pairs vs random by PLACED cosine — the
#   placement-side hypernym readout (close_extra signals can move this one).
METRIC_KEYS = ("syn_ant", "synonym_auc", "similar_auc", "antonym_auc",
               "hypernym_auc", "hypernym_cos_auc", "random_cos")


def pair_sets(words: Sequence[str], graph: RelationGraph) -> Dict[str, List[Pair]]:
    """The canonical scored pair sets, exactly as the M19 evaluators build them."""
    wset = set(words)
    syn = sorted({tuple(sorted((w, s))) for w in words
                  for s in graph.synonym.get(w, []) if s in wset and s != w})
    sim = sorted({tuple(sorted((w, s))) for w in words
                  for s in graph.similar.get(w, []) if s in wset and s != w})
    ant = sorted({tuple(sorted(p)) for p in graph.typed_pairs("antonym")})
    isa = sorted(set(graph.typed_pairs("is_a", directed=True)))
    return {"syn": syn, "sim": sim, "ant": ant, "isa": isa}


def _in_vocab(pairs: Iterable[Pair], wset) -> List[Pair]:
    return [tuple(sorted(p)) for p in pairs
            if p[0] in wset and p[1] in wset and p[0] != p[1]]


def metric_table(
    words,
    graph: RelationGraph,
    axes: MeaningAxes,
    *,
    cache=None,
    depth: int = 3,
    train_frac: float = 0.5,
    alpha: float = 0.7,
    iters: int = 20,
    n_neg: int = 2000,
    seed: int = 0,
    close_extra: Sequence[Pair] = (),
    antonym_extra: Sequence[Pair] = (),
) -> Dict:
    """All M28.0 metrics from ONE placement pass, extras injected M24-safely."""
    words = list(words)
    wset = set(words)
    idx = {w: i for i, w in enumerate(words)}

    ps = pair_sets(words, graph)
    train_syn, test_syn = split_pairs(ps["syn"], train_frac)
    train_sim, test_sim = split_pairs(ps["sim"], train_frac)
    _, test_ant = split_pairs(ps["ant"], train_frac)
    _, test_isa = split_pairs(ps["isa"], train_frac)

    # M24: an extra close edge may not touch ANY scored held-out pair.
    scored = set(test_syn) | set(test_sim) | set(test_ant) | \
        {tuple(sorted(p)) for p in test_isa}
    extra_close = [p for p in _in_vocab(close_extra, wset) if p not in scored]
    n_dropped = len(_in_vocab(close_extra, wset)) - len(extra_close)

    placed = place(words, graph, axes, cache=cache, depth=depth,
                   train_pairs=train_syn + train_sim + extra_close,
                   iters=iters, alpha=alpha)
    anchor = anchored_coordinate(words, graph, axes, cache=cache, depth=depth)
    anchor_by_w = {w: anchor[i] for i, w in enumerate(words)}

    def cos(a: str, b: str) -> float:
        return cosine(placed[a], placed[b])

    def contain(a: str, b: str) -> float:  # a IS_A b: b's features contained in a
        av, bv = anchor_by_w[a] > 0, anchor_by_w[b] > 0
        nb = int(bv.sum())
        return float((av & bv).sum() / nb) if nb > 0 else 0.0

    exclude = set(ps["syn"]) | set(ps["sim"]) | set(ps["ant"]) | \
        {tuple(sorted(p)) for p in ps["isa"]} | set(extra_close)
    neg = _random_pairs(words, n_neg, seed, exclude)
    neg_cos = [cos(a, b) for a, b in neg]

    syn_ant, n_ts, n_ta = discrimination(test_syn, test_ant, cos, placed)
    out = {
        "syn_ant": syn_ant,
        "synonym_auc": _auc([cos(a, b) for a, b in test_syn], neg_cos),
        "similar_auc": _auc([cos(a, b) for a, b in test_sim], neg_cos),
        "antonym_auc": _auc([-cos(a, b) for a, b in test_ant],
                            [-c for c in neg_cos]),
        "hypernym_auc": _auc([contain(a, b) for a, b in test_isa],
                             [contain(a, b) for a, b in neg]),
        "hypernym_cos_auc": _auc([cos(a, b) for a, b in test_isa], neg_cos),
        "random_cos": float(np.mean(neg_cos)) if neg_cos else 0.0,
        "n": {"test_syn": n_ts, "test_ant": n_ta, "test_sim": len(test_sim),
              "test_isa": len(test_isa), "extra_close": len(extra_close),
              "extra_close_dropped_m24": n_dropped},
    }

    # Expanded antonymy: NEVER mixed into the original score. Report separately:
    # do the *new* pairs (beyond the original relation) already read as opposed?
    ant_x = [p for p in _in_vocab(antonym_extra, wset) if p not in set(ps["ant"])]
    if ant_x:
        out["ant_expanded"] = {
            "n_new_pairs": len(ant_x),
            "expanded_syn_ant": discrimination(test_syn, ant_x, cos, placed)[0],
        }
    return out


def _agg(tables: List[Dict]) -> Dict:
    mean = {k: float(np.mean([t[k] for t in tables])) for k in METRIC_KEYS}
    band = {k: float((max(t[k] for t in tables) - min(t[k] for t in tables)) / 2)
            for k in METRIC_KEYS}
    return {"mean": mean, "band": band, "per_jitter": tables}


def baseline_table(words, graph, axes, *, cache=None,
                   jitters: Sequence[float] = (0.45, 0.5, 0.55), **kw) -> Dict:
    """The no-extras table across split jitters -> mean + noise band."""
    return _agg([metric_table(words, graph, axes, cache=cache,
                              train_frac=j, **kw) for j in jitters])


def with_features(graph: RelationGraph, feature_extra: Dict[str, List[str]]) -> RelationGraph:
    """A graph copy with extra word->category features merged into ``attribute``
    (so they become candidate named axes via ``MeaningAxes.assemble``)."""
    g = copy.deepcopy(graph)
    for w, cats in feature_extra.items():
        if w in g.gloss:
            g.attribute[w] = sorted(set(g.attribute.get(w, [])) | set(cats))
    return g


def with_antonyms(graph: RelationGraph, antonym_extra: Sequence[Pair]) -> RelationGraph:
    """A graph copy with extra antonym pairs added (symmetric)."""
    g = copy.deepcopy(graph)
    for a, b in antonym_extra:
        if a in g.gloss and b in g.gloss:
            if b not in g.antonym.setdefault(a, []):
                g.antonym[a].append(b)
            if a not in g.antonym.setdefault(b, []):
                g.antonym[b].append(a)
    return g


def ablate(
    words,
    graph: RelationGraph,
    axes: MeaningAxes,
    *,
    cache=None,
    name: str = "",
    close_extra: Sequence[Pair] = (),
    antonym_extra: Sequence[Pair] = (),
    feature_extra: Dict[str, List[str]] | None = None,
    jitters: Sequence[float] = (0.45, 0.5, 0.55),
    **kw,
) -> Dict:
    """Baseline vs +signal across jitters; returns {baseline, signal, delta, verdict}."""
    base = baseline_table(words, graph, axes, cache=cache, jitters=jitters, **kw)

    g, ax = graph, axes
    if feature_extra:
        g = with_features(g, feature_extra)
        ax = MeaningAxes.assemble(g, min_attribute_freq=2)
    tables = [metric_table(words, g, ax, cache=cache, train_frac=j,
                           close_extra=close_extra, antonym_extra=antonym_extra,
                           **kw) for j in jitters]
    sig = _agg(tables)

    delta = {k: sig["mean"][k] - base["mean"][k] for k in METRIC_KEYS}
    moved = {k: d for k, d in delta.items()
             if abs(d) > 2 * max(base["band"][k], 1e-9)}
    return {"name": name, "baseline": base, "signal": sig,
            "delta": delta, "moved": moved,
            "extras_n": tables[0]["n"] | ({"ant_expanded": tables[0].get("ant_expanded")}
                                          if tables[0].get("ant_expanded") else {})}


def format_report(res: Dict) -> str:
    lines = [f"=== ablation: {res['name']} ==="]
    lines.append(f"{'metric':13} {'base':>7} {'band':>6} {'+signal':>8} {'delta':>8}  verdict")
    for k in METRIC_KEYS:
        b, bd = res["baseline"]["mean"][k], res["baseline"]["band"][k]
        s, d = res["signal"]["mean"][k], res["delta"][k]
        v = "MOVED" if k in res["moved"] else ""
        lines.append(f"{k:13} {b:7.3f} {bd:6.3f} {s:8.3f} {d:+8.3f}  {v}")
    lines.append(f"counts: {res['extras_n']}")
    return "\n".join(lines)
