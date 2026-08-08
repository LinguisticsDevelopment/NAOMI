"""Satellite-cluster indirect antonymy: grow the antonym edge store (M26.x
follow-on to the WSD grounding work).

WordNet adjectives split into head synsets (pos ``'a'``) and satellite synsets
(pos ``'s'``) that orbit a head via the ``SIMILAR_TO`` pointer. Only head
synsets carry direct antonym pointers on their lemmas, so a satellite like
``damp`` (satellite of ``wet``) has no direct antonym edge to ``dry`` even
though the two are plainly opposed. This module derives that indirect
antonymy: if satellite s1 is similar_to head h1, and h1 is an antonym of head
h2, then s1 is an indirect antonym of h2 AND of every satellite s2 of h2
(``damp``~``wet``, ``wet``<->``dry`` => ``damp``<->``dry``, ``damp``<->``arid``).

The graph-relation wrappers in :mod:`nsm_ct.wordnet` collapse this to
lemma-level ``similar_tos()``/``antonyms()`` lists and lose the per-synset
pos/similar_to structure this needs, so this module walks
``nltk.corpus.wordnet`` synsets directly (per project convention: use nltk
directly when the shared wrappers don't expose the pointer you need).

Exposes the ``nsm_ct.ground.signal_<name>`` ablation contract
(``extras(vocab, graph) -> dict``, see scripts/ablate_signal.py and
nsm_ct.ground.ablation) with a single ``antonym_extra`` key. Antonym edges do
not currently feed placement, so plugging this in must leave the original
``syn_ant``/``*_auc`` metrics unchanged by construction (ablation.py never
mixes ``antonym_extra`` into the scored held-out antonym pairs) — the payoff
is reported on the side via ``ant_expanded`` (breadth: ``n_new_pairs``;
consistency: ``expanded_syn_ant``).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..wordnet import wordnet_available
from .relations import RelationGraph

Pair = Tuple[str, str]


def expanded_antonyms(word: str) -> List[str]:
    """Lemma names indirectly opposed to *word* via the satellite~head~antonym
    chain (deduped, lowercase, sorted, excludes *word* itself).

    For every adjective synset of *word* (head ``'a'`` or satellite ``'s'``):

    - a satellite synset walks its ``similar_tos()`` pointer(s) to its head(s);
      a head synset is already the head, so it is used directly.
    - each head's lemma-level antonyms give one or more opposed head synsets.
    - every lemma of an opposed head, plus every lemma of *that* head's own
      satellites, is an indirect antonym of *word*.

    Both the antonym pointer and the similar_to pointer are stored
    reciprocally in WordNet, so this expansion is symmetric: if ``y`` is in
    ``expanded_antonyms(x)`` then ``x`` is in ``expanded_antonyms(y)``.

    Returns ``[]`` if WordNet is unavailable or *word* has no adjective senses.
    """
    if not wordnet_available():
        return []
    try:
        from nltk.corpus import wordnet as wn

        wl = word.lower().replace(" ", "_")
        out = set()
        synsets = list(wn.synsets(word, wn.ADJ)) + list(wn.synsets(word, wn.ADJ_SAT))
        for syn in synsets:
            heads = syn.similar_tos() if syn.pos() == "s" else [syn]
            for h1 in heads:
                for lemma in h1.lemmas():
                    for ant_lemma in lemma.antonyms():
                        h2 = ant_lemma.synset()
                        for l2 in h2.lemmas():
                            name = l2.name().lower()
                            if name != wl:
                                out.add(name)
                        # every satellite s2 of h2 is also opposed to *word*
                        for sat in h2.similar_tos():
                            if sat.pos() != "s":
                                continue
                            for l3 in sat.lemmas():
                                name = l3.name().lower()
                                if name != wl:
                                    out.add(name)
        return sorted(out)
    except Exception:  # pragma: no cover
        return []


def extras(vocab: Sequence[str], graph: RelationGraph) -> Dict[str, List[Pair]]:
    """The ablation-harness contract (scripts/ablate_signal.py).

    Pairs ``(w, x)`` for ``w`` in ``graph.gloss``, ``x`` in
    ``expanded_antonyms(w)``, both in-vocab, excluding pairs already present
    in ``graph.antonym``.
    """
    vocab_set = set(vocab)
    existing = {tuple(sorted(p)) for p in graph.typed_pairs("antonym")}
    pairs = set()
    for w in graph.gloss:
        if w not in vocab_set:
            continue
        for x in expanded_antonyms(w):
            if x == w or x not in vocab_set:
                continue
            key = tuple(sorted((w, x)))
            if key in existing:
                continue
            pairs.add(key)
    return {"antonym_extra": sorted(pairs)}
