"""Sense (synset) graph — WSD by construction (M22.0).

Nodes are WordNet **synsets**, not words, so every relation is sense-correct and
synonymy (co-lemmas of a synset) is matched by construction. Because synonymy
collapses *into* a node, the meaningful "these mean nearly the same" relation
between distinct sense-nodes is ``similar_to`` (adjective near-synonym clusters,
~90% coverage) plus shared-hypernym co-hyponyms; antonyms are per-sense.

Each synset node is grounded from ITS OWN gloss (not a word's first-sense gloss):
active primes come from decomposing the synset's gloss content words; attribute /
lexname / hypernym / similar_to / antonym are all read off that synset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List

import numpy as np

from .canonicalization import canon_label
from .definition_graph import content_words, naive_decompose
from .polarity import POLARITY_PAIRS, gloss_polarity
from .sparse_value import SparseSpace, _PAIRED, _PRIME_SET


def _synset_gloss_primes(gloss: str, depth: int, max_children: int = 6) -> set:
    """Active NSM primes from decomposing a synset gloss's content words."""
    prims = set()
    for c in content_words(gloss)[:max_children]:
        for n in naive_decompose(c, max_depth=depth).iter_preorder():
            lab = canon_label(n.label)
            if lab in _PRIME_SET:
                prims.add(lab)
    return prims


@lru_cache(maxsize=8192)
def _gloss_prime_counts(gloss: str, depth: int, max_children: int) -> tuple:
    """Count activations of each NSM prime across a gloss's content-word decompositions."""
    counts: Dict[str, int] = {}
    for c in content_words(gloss)[:max_children]:
        for n in naive_decompose(c, max_depth=depth).iter_preorder():
            lab = canon_label(n.label)
            if lab in _PRIME_SET:
                counts[lab] = counts.get(lab, 0) + 1
    return tuple(sorted(counts.items()))


def gloss_prime_weights(gloss: str, *, depth: int = 2, max_children: int = 8,
                        normalize: bool = True) -> Dict[str, float]:
    """Grounded NSM-prime signature of a gloss: prime name -> weight (M26).

    Decomposes the gloss's content words into primes and weights each by its
    activation count (normalized so the strongest prime is 1.0). This is the M22
    per-sense grounding, exposed as the sense-signature the WSD layer consumes."""
    counts = dict(_gloss_prime_counts(gloss or "", depth, max_children))
    if not counts:
        return {}
    if normalize:
        m = float(max(counts.values()))
        return {k: v / m for k, v in counts.items()}
    return {k: float(v) for k, v in counts.items()}


@dataclass
class SenseGraph:
    senses: List[str] = field(default_factory=list)          # synset ids (nodes)
    gloss: Dict[str, str] = field(default_factory=dict)
    similar: Dict[str, List[str]] = field(default_factory=dict)   # similar_to synset ids
    hypernym: Dict[str, List[str]] = field(default_factory=dict)
    antonym: Dict[str, List[str]] = field(default_factory=dict)   # per-sense
    derivational: Dict[str, List[str]] = field(default_factory=dict)  # cross-POS family (M23)
    meronym: Dict[str, List[str]] = field(default_factory=dict)       # part/member/substance + holonyms (M23)
    gloss_overlap: Dict[str, List[str]] = field(default_factory=dict) # senses sharing >=K gloss words (M23)
    attribute: Dict[str, List[str]] = field(default_factory=dict)
    lexname: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        words,
        *,
        max_senses_per_word: int = 3,
        cap: int = None,
        gloss_overlap_k: int = 3,
        gloss_overlap_max_fanout: int = 25,
    ) -> "SenseGraph":
        """Nodes = the top synsets (by frequency) of each word, capped."""
        from ..wordnet import senses as wn_senses, wordnet_available
        if not wordnet_available():
            return cls()
        from nltk.corpus import wordnet as wn

        # collect candidate synset ids (top senses per word by frequency)
        chosen: List[str] = []
        seen = set()
        for raw in words:
            ss = sorted(wn_senses(raw), key=lambda d: -d.get("frequency", 0))[:max_senses_per_word]
            for s in ss:
                sid = s["sense_id"]
                if sid not in seen:
                    seen.add(sid)
                    chosen.append(sid)
        if cap:
            chosen = chosen[:cap]
        nodeset = set(chosen)

        g = cls()
        for sid in chosen:
            try:
                syn = wn.synset(sid)
            except Exception:
                continue
            g.senses.append(sid)
            g.gloss[sid] = syn.definition()
            g.similar[sid] = [t.name() for t in syn.similar_tos() if t.name() in nodeset]
            g.hypernym[sid] = [h.name() for h in syn.hypernyms() if h.name() in nodeset]
            ants = []
            for lemma in syn.lemmas():
                for a in lemma.antonyms():
                    aid = a.synset().name()
                    if aid in nodeset:
                        ants.append(aid)
            g.antonym[sid] = sorted(set(ants))
            # derivational: cross-POS family, target synset ids (lemma-level relation)
            der = []
            for lemma in syn.lemmas():
                for d in lemma.derivationally_related_forms():
                    did = d.synset().name()
                    if did in nodeset and did != sid:
                        der.append(did)
            g.derivational[sid] = sorted(set(der))
            # meronym / holonym: structural part-of, both directions
            mer = []
            for m in (
                syn.part_meronyms() + syn.member_meronyms() + syn.substance_meronyms()
                + syn.part_holonyms() + syn.member_holonyms() + syn.substance_holonyms()
            ):
                if m.name() in nodeset and m.name() != sid:
                    mer.append(m.name())
            g.meronym[sid] = sorted(set(mer))
            g.attribute[sid] = [a.name().split(".")[0] for a in syn.attributes()]
            g.lexname[sid] = syn.lexname()

        # gloss-overlap (the "use the definitions" edge): senses sharing >=K content words.
        # Invert content-word -> senses, then link senses that co-occur, capping fan-out.
        posting: Dict[str, List[str]] = {}
        content_of: Dict[str, set] = {}
        for sid in g.senses:
            cw = set(content_words(g.gloss.get(sid, "")))
            content_of[sid] = cw
            for w in cw:
                posting.setdefault(w, []).append(sid)
        for sid in g.senses:
            counts: Dict[str, int] = {}
            for w in content_of[sid]:
                bucket = posting.get(w, ())
                if len(bucket) > gloss_overlap_max_fanout:
                    continue  # skip ubiquitous words (weak signal, quadratic blowup)
                for other in bucket:
                    if other != sid:
                        counts[other] = counts.get(other, 0) + 1
            g.gloss_overlap[sid] = sorted(o for o, c in counts.items() if c >= gloss_overlap_k)
        return g

    def words(self) -> List[str]:  # node ids (keeps the RelationGraph-like interface)
        return list(self.senses)

    def _store(self, relation: str) -> Dict[str, List[str]]:
        return {
            "similar": self.similar,
            "hypernym": self.hypernym,
            "antonym": self.antonym,
            "derivational": self.derivational,
            "meronym": self.meronym,
            "gloss_overlap": self.gloss_overlap,
        }[relation]

    def typed_pairs(self, relation: str) -> List[tuple]:
        store = self._store(relation)
        nodeset = set(self.senses)
        pairs = set()
        for a, neigh in store.items():
            for b in neigh:
                if b in nodeset and b != a:
                    pairs.add(tuple(sorted((a, b))) if relation != "hypernym" else (a, b))
        return sorted(pairs)

    def cohyponym_pairs(self, *, max_per_group: int = 40) -> List[tuple]:
        """Senses sharing a hypernym (the M22 baseline close signal; note: siblings
        can include antonyms)."""
        groups: Dict[str, List[str]] = {}
        for sid, hs in self.hypernym.items():
            for h in hs:
                groups.setdefault(h, []).append(sid)
        pairs = set()
        for members in groups.values():
            members = members[:max_per_group]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pairs.add(tuple(sorted((members[i], members[j]))))
        return sorted(pairs)

    def close_edges(self, *types: str, exclude_antonyms: bool = True) -> List[tuple]:
        """Union the requested close-relation types into one undirected edge list,
        dropping known antonym pairs so contamination can't leak opposites into the
        'close' set. ``cohyponym`` is accepted as a pseudo-type."""
        edges = set()
        for t in types:
            if t == "cohyponym":
                edges.update(self.cohyponym_pairs())
            else:
                edges.update(tuple(sorted(p)) for p in self.typed_pairs(t))
        if exclude_antonyms:
            ban = {tuple(sorted(p)) for p in self.typed_pairs("antonym")}
            edges -= ban
        return sorted(edges)


def build_sense_sparse(g: SenseGraph, *, depth: int = 2, min_attribute_freq: int = 2) -> SparseSpace:
    """The Null-aware (value, mask) representation over synset nodes (M21 model,
    grounded per-sense from each synset's own gloss)."""
    from collections import Counter
    from ..nsm_primes import PRIME_NAMES

    nodes = g.senses
    N = len(nodes)
    bipolar = [name for name, _, _ in POLARITY_PAIRS]
    other_primes = [p for p in PRIME_NAMES if p not in _PAIRED]
    attr_freq: Counter = Counter()
    for dims in g.attribute.values():
        for d in dims:
            attr_freq[d] += 1
    attr_axes = [f"attr:{d}" for d, f in attr_freq.most_common() if f >= min_attribute_freq]
    lex_axes = [f"lex:{lx}" for lx in sorted(set(g.lexname.values()))]
    axes = bipolar + other_primes + attr_axes + lex_axes
    col = {a: i for i, a in enumerate(axes)}
    D = len(axes)

    value = np.zeros((N, D), dtype=np.float32)
    mask = np.zeros((N, D), dtype=np.float32)
    for r, sid in enumerate(nodes):
        active = _synset_gloss_primes(g.gloss.get(sid, ""), depth)
        for name, pos, neg in POLARITY_PAIRS:
            if pos in active or neg in active:
                j = col[name]
                mask[r, j] = 1.0
                value[r, j] = (1.0 if pos in active else 0.0) - (1.0 if neg in active else 0.0)
        for p in active:
            if p in col and p not in _PAIRED:
                mask[r, col[p]] = 1.0
                value[r, col[p]] = 1.0
        s = gloss_polarity_of(g.gloss.get(sid, ""))
        for d in g.attribute.get(sid, []):
            j = col.get(f"attr:{d}")
            if j is not None:
                mask[r, j] = 1.0
                value[r, j] = s
        lx = g.lexname.get(sid)
        j = col.get(f"lex:{lx}") if lx else None
        if j is not None:
            mask[r, j] = 1.0
            value[r, j] = 1.0

    df = mask.sum(axis=0)
    idf = np.log(N / (df + 1.0)).astype(np.float32)
    return SparseSpace(words=nodes, axes=axes, value=value, mask=mask, idf=idf)


def gloss_polarity_of(gloss: str, *, weight: float = 2.0) -> float:
    """Gloss-magnitude polarity read directly off a synset gloss (M18.1 cue lexicon)."""
    from .polarity import _POS_CUES, _NEG_CUES
    from ..tokenizer import basic_tokenize
    net = 0
    for tok in basic_tokenize(gloss):
        if tok in _POS_CUES:
            net += 1
        elif tok in _NEG_CUES:
            net -= 1
    return 0.0 if net == 0 else weight * (1.0 if net > 0 else -1.0)
