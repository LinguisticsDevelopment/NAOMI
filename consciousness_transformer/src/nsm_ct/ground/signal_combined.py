"""The combined Step B winners (M28.1) — run via ablate_signal --signal combined.

Composes exactly the signals that individually beat the noise band with no
disqualifying regression:
- also_see  -> close_extra   (+0.015 similar_auc solo)
- domain    -> feature_extra (-0.032 random_cos solo; watch hypernym_cos here)
- satellite -> antonym_extra (edge-store growth; placement-inert by design)

Excluded (documented negatives, modules kept): genus-as-close-edge (moves
hypernym_cos +0.010 but costs similar -0.021 / random +0.022 — hypernymy is
directional, belongs in the relational store), entailment (regresses similar),
pertainym / cause / verbgroup (null).
"""

from __future__ import annotations

from . import signal_also_see, signal_domain, signal_satellite


def extras(vocab, graph) -> dict:
    out: dict = {}
    out["close_extra"] = signal_also_see.extras(vocab, graph)["close_extra"]
    out["feature_extra"] = signal_domain.extras(vocab, graph)["feature_extra"]
    out["antonym_extra"] = signal_satellite.extras(vocab, graph)["antonym_extra"]
    return out
