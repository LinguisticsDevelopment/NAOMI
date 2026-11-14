"""M43 prototype — opt-in DEEPER EXPLICATIONS for sense signatures.

M42 diagnosed the mechanism, and this module targets it directly. Traced
end-to-end for ``hood.n.08`` (gloss "a headdress that protects the head and
face", RESEARCH_NOTES M42/M43):

1. ``build_usvs`` grounds a sense purely via ``sense_prime_weights(gloss)``
   (see ``usvs.py``), which calls ``naive_decompose`` on each gloss content
   word and keeps ONLY leaves whose canonical label is one of the ~65 literal
   NSM primes (``nsm_primes.PRIME_NAMES``) — everything else, including
   whole MOLECULE leaves (mid-level nodes like ``HEAD`` or ``FACE``, which
   ARE in the axis inventory as their own named axes) is silently dropped by
   the prime-only filter in ``_word_prime_counts``.
2. For hood.n.08: "headdress" bottoms out UNRESOLVED (depth budget), "head"
   and "face" each decompose in ONE hop straight to the molecule leaves
   ``HEAD`` / ``FACE`` — real, named axes that exist in ``usvs.axes`` — but
   those leaves are discarded because they are not in ``_PRIME_SET``. Only
   "protects" survives, via a coincidental 2-hop chain to the literal prime
   ``BODY``. The signature ends up ``{BODY: 1.0, lex:noun.artifact: 1.0}``
   even though HEAD/FACE axes exist and would have been the correct content.
3. Elsewhere (e.g. ball.n.09, "a lavish dance requiring formal attire") the
   surviving prime is ``SOMETHING``, reached only because one gloss word's
   OWN two-hop chain ("attire" -> "style" -> the literal word "something")
   happens to hit it — the actual content word "dance" is thrown away
   entirely (its subtree bottoms out UNRESOLVED). This is the M42-measured
   57% SOMETHING-saturation mechanism: shallow decomposition keeps whatever
   generic prime it stumbles into and discards named, on-topic axes that
   were reachable but not prime-labelled.

This module does NOT change ``build_usvs`` or the shipped artifact (house
rule: ``data/usvs/`` is never touched). It adds an opt-in, on-the-fly
enrichment: pull the sense's gloss content words back out, look up each
one's OWN placed-core coordinate (or, for words outside the 10k core, its
MFS sense signature) — both of which carry the full 607-axis vocabulary,
not just the prime subset — and blend that into the base signature with a
damping factor. Nothing here is wired into the default build or query path;
callers must call ``enriched_sense_dense`` explicitly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import FrozenSet, List, Optional

import numpy as np

from .definition_graph import content_words

__all__ = ["gloss_of", "enriched_sense_dense"]


@lru_cache(maxsize=200_000)
def gloss_of(sense_id: str) -> str:
    """The WordNet definition string for a synset name, cached (117k senses
    fit comfortably; identical cost profile to ``usvs.sense_prime_weights``'s
    own per-word cache)."""
    from ..wordnet import _wn
    wn = _wn()
    try:
        syn = wn.synset(sense_id)
    except Exception:  # pragma: no cover - malformed/unknown sense id
        return ""
    return syn.definition() or "" if syn is not None else ""


def _content_word_vec(usvs, word: str) -> Optional[np.ndarray]:
    """A gloss content word's own coordinate: its placed CORE coordinate if
    it's a core word (the artifact's strongest, relation-derived layer),
    else its own MFS sense signature. Both cover the full 607-axis
    inventory, unlike the prime-only decomposition used for grounding."""
    v = usvs.word_coord(word)
    if v is not None:
        return v
    sids = usvs.senses_of(word)
    if not sids:
        return None
    d = usvs.sense_dense(sids[0])
    return d if d is not None and d.any() else None


def enriched_sense_dense(
    usvs,
    sense_id: str,
    alpha: float = 0.35,
    *,
    exclude: FrozenSet[str] = frozenset(),
    max_words: int = 8,
) -> Optional[np.ndarray]:
    """Opt-in DEEPER EXPLICATION: blend a sense's base signature with its
    OWN gloss content words' USVS coordinates.

    ``alpha`` is the damping factor for the content blend (0 -> identical to
    ``usvs.sense_dense``; the default build path never calls this function,
    so ``alpha`` has no effect on ``data/usvs/`` or any existing consumer).

    Deterministic: gloss content words are taken in fixed surface order
    (``content_words``, already de-duplicated), each word's own vector is
    L2-normalized before averaging (so no single unusually large-magnitude
    content word dominates the blend), and the result is renormalized. No
    randomness, no cache mutation of the base artifact.

    ``exclude`` (M24 house rule): drop these words from the gloss
    content-word set BEFORE lookup. Evaluation callers scoring a sense
    against a specific answer word must pass that word (and, to be safe,
    every other candidate answer in the same family) here — otherwise a
    gloss that happens to literally mention the scored word would leak it
    straight into the sense's own vector, which would inflate the benchmark
    rather than test grounding depth.
    """
    base = usvs.sense_dense(sense_id)
    if base is None:
        return None
    if alpha <= 0.0:
        return base
    gloss = gloss_of(sense_id)
    words: List[str] = [w for w in content_words(gloss) if w not in exclude][:max_words]
    vecs = []
    for w in words:
        v = _content_word_vec(usvs, w)
        if v is None or not v.any():
            continue
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            vecs.append(v / n)
    if not vecs:
        return base
    content = np.mean(vecs, axis=0).astype(np.float32)

    bn = float(np.linalg.norm(base))
    base_unit = base / bn if bn > 1e-9 else base
    blended = (1.0 - alpha) * base_unit + alpha * content
    n = float(np.linalg.norm(blended))
    return (blended / n).astype(np.float32) if n > 1e-9 else base
