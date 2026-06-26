"""Honest understanding evaluation (M17.3).

Brings the pieces together to answer, with numbers, "how does the system
understand word meaning?" — measured on **held-out words outside the DeepNSM/gold
dictionary** so we are scoring *derivation, not lookup*.

Reported, seed (NSM-65) vs. derived basis (M17.2):
- ``grounding_rate``         — fraction of decomposition leaves that reach an axis.
- ``convergence``            — prime-fixpoint stability (does deeper decomposition
                               stop changing the coordinate?).
- ``syn_ant_discrimination`` — does the system place synonyms closer than antonyms?
                               (the core understanding probe).
- ``hypernym_containment``   — is a hypernym's meaning contained in the word's?
- ``round_trip`` / ``perturbed`` — clause==word recovery (exact + paraphrase).
- ``deepnsm_agreement``      — independent external check on the held-out slice.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .basis_search import _value_vec, relational_metrics, search
from .canonicalization import normalize
from .clause_self_consistency import SAMPLE_VOCAB, deepnsm_primes, _default_store
from .definition_graph import DEFAULT_MAX_DEPTH, DefinitionGraph, naive_decompose
from .meaning_value import axis_vector, cosine, jaccard
from .reduction import ReducedDefinitionIndex, lexicalize, perturbed_clause, round_trip
from .semantic_axes import AxisRegistry
from ..wordnet import antonyms, hypernyms


def held_out_vocab(words, store=None) -> List[str]:
    """Words NOT covered by the DeepNSM/gold store — where we measure derivation."""
    if store is None:
        store = _default_store()
    out = []
    for w in words:
        wl = w.lower().strip()
        if store is None or store.is_empty() or store.get(wl) is None:
            out.append(wl)
    return out


def expand_vocab(seed) -> List[str]:
    """Relationally close the vocab with one-hop hypernym lemmas + antonyms."""
    vocab = set(w.lower().strip() for w in seed)
    for w in seed:
        for h in hypernyms(w):
            vocab.add(h.replace("_", " ").split()[0].lower())
        for a in antonyms(w):
            vocab.add(a.lower())
    return sorted(vocab)


def _convergence(word: str, reg: AxisRegistry, depth: int) -> float:
    extra = reg.extra_axes()
    a = axis_vector(normalize(naive_decompose(word, max_depth=depth, extra_axes=extra)), reg.axes)
    b = axis_vector(normalize(naive_decompose(word, max_depth=depth + 1, extra_axes=extra)), reg.axes)
    return cosine(a, b)


def syn_ant_discrimination(words, reg: AxisRegistry, depth: int, graph: DefinitionGraph) -> Dict:
    """Accuracy: for words with both a grounded synonym and antonym, is the mean
    similarity to synonyms greater than to antonyms?"""
    coord = {w: _value_vec(w, reg, depth) for w in words}
    correct = total = 0
    for w in words:
        syns = [s.lower() for s in graph.synonym.get(w, []) if s.lower() in coord and s.lower() != w]
        ants = [a.lower() for a in graph.antonym.get(w, []) if a.lower() in coord]
        if not syns or not ants:
            continue
        syn_m = float(np.mean([cosine(coord[w], coord[s]) for s in syns]))
        ant_m = float(np.mean([cosine(coord[w], coord[a]) for a in ants]))
        total += 1
        if syn_m > ant_m:
            correct += 1
    return {"accuracy": (correct / total if total else None), "n": total}


def _round_trip_rates(words, depth: int) -> Dict:
    index = ReducedDefinitionIndex.build(words, depth=depth)
    exact = pt = ph = 0
    for w in words:
        if round_trip(w, index)[0] == w:
            exact += 1
        pc = perturbed_clause(w)
        if pc is not None:
            pt += 1
            if lexicalize(pc, index)[0] == w:
                ph += 1
    return {
        "exact": exact / len(words) if words else None,
        "perturbed": ph / pt if pt else None,
        "n": len(words),
        "n_perturbed": pt,
    }


def _deepnsm_agreement_mean(words, reg: AxisRegistry, depth: int, store) -> Dict:
    vals = []
    for w in words:
        ext = deepnsm_primes(w, store=store)
        if not ext:
            continue
        ours = {a for a, x in zip(reg.axes, _value_vec(w, reg, depth)) if x > 0}
        vals.append(jaccard(ours, ext))
    return {"mean": (sum(vals) / len(vals) if vals else None), "n": len(vals)}


def _metrics_for(words, reg: AxisRegistry, depth: int, graph: DefinitionGraph) -> Dict:
    """Understanding metrics on the *derivation* set (words outside DeepNSM)."""
    rel = relational_metrics(words, reg, depth, graph)
    conv = [_convergence(w, reg, depth) for w in words]
    return {
        "grounding_rate": rel["grounding_rate"],
        "convergence": sum(conv) / len(conv) if conv else None,
        "syn_ant": syn_ant_discrimination(words, reg, depth, graph),
        "hypernym_containment": rel["hypernym_containment"],
        "n_hypernym_pairs": rel["n_hypernym_pairs"],
    }


def evaluate(
    words: Optional[List[str]] = None,
    *,
    depth: int = DEFAULT_MAX_DEPTH,
    max_axes: int = 15,
    expand: bool = True,
) -> Dict:
    """Full M17.3 comparison: seed NSM-65 vs. derived basis.

    Understanding probes are reported on the **non-DeepNSM** subset (so we are
    measuring derivation, not lookup); DeepNSM agreement is reported on the
    **covered** subset as an independent external check.
    """
    base = words or SAMPLE_VOCAB
    vocab = expand_vocab(base) if expand else [w.lower().strip() for w in base]
    store = _default_store()
    graph = DefinitionGraph.build(vocab)

    derivation_words = held_out_vocab(vocab, store)  # outside DeepNSM/gold
    covered = [w for w in vocab if w not in set(derivation_words)]  # in DeepNSM/gold

    seed_reg = AxisRegistry.seed()
    res = search(vocab, depth=depth, max_axes=max_axes, graph=graph)
    derived_reg = res.registry

    return {
        "depth": depth,
        "n_vocab": len(vocab),
        "n_derivation": len(derivation_words),
        "n_deepnsm_covered": len(covered),
        "promoted_axes": derived_reg.summary(),
        "mdl_curve": res.mdl_curve,
        "round_trip": _round_trip_rates(derivation_words, depth),
        "seed": _metrics_for(derivation_words, seed_reg, depth, graph),
        "derived": _metrics_for(derivation_words, derived_reg, depth, graph),
        "deepnsm_agreement_seed": _deepnsm_agreement_mean(covered, seed_reg, depth, store),
        "deepnsm_agreement_derived": _deepnsm_agreement_mean(covered, derived_reg, depth, store),
    }
