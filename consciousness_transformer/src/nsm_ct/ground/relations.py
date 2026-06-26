"""The unified relation store — every source of signal (M19.0).

M17/M18 used five relations (gloss-DEFINES, hypernym, synonym, antonym). M19 treats
*every* WordNet relationship as signal about the geometry of meaning, so this store
carries them all in one typed structure: the word-word relations (synonym, antonym,
similar-to, hypernym, meronym, derivational, verb-group) plus the two *feature*
relations that name axes directly — ``lexname`` (44 coarse categories) and
``attribute`` (the noun dimension an adjective is a value of, e.g. hot->temperature).

This is the single substrate the axis discovery (M19.1), placement (M19.2), and
dictionary reconstruction (M19.4) all read from. Every field is interpretable;
nothing here is an opaque embedding.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple

from ..wordnet import (
    antonyms,
    attributes,
    derivationally_related,
    hypernyms,
    lexname,
    meronyms,
    senses,
    similar_tos,
    synonyms,
    verb_groups,
)
from .definition_graph import content_words

# Word-word relation types carried as symmetric/directed pair sets.
WORD_RELATIONS = ("synonym", "antonym", "similar", "is_a", "meronym", "derivational", "verb_group")


def _lemma_head(name: str) -> str:
    """``canine.n.02`` -> ``canine``; ``domestic_animal`` -> ``domestic_animal``."""
    return name.split(".")[0]


@dataclass
class RelationGraph:
    """All relational signal over a vocabulary, one typed store."""

    gloss: Dict[str, str] = field(default_factory=dict)
    content: Dict[str, List[str]] = field(default_factory=dict)          # DEFINES (definition words)
    synonym: Dict[str, List[str]] = field(default_factory=dict)
    antonym: Dict[str, List[str]] = field(default_factory=dict)
    similar: Dict[str, List[str]] = field(default_factory=dict)          # adj near-synonym clusters
    is_a: Dict[str, List[str]] = field(default_factory=dict)             # hypernyms
    meronym: Dict[str, List[str]] = field(default_factory=dict)
    derivational: Dict[str, List[str]] = field(default_factory=dict)
    verb_group: Dict[str, List[str]] = field(default_factory=dict)
    lexname: Dict[str, str] = field(default_factory=dict)                # one of 44 categories
    attribute: Dict[str, List[str]] = field(default_factory=dict)        # adj -> noun dimension(s)

    @classmethod
    def build(cls, words) -> "RelationGraph":
        g = cls()
        for raw in words:
            w = raw.lower().strip()
            if not w or w in g.gloss:
                continue
            ss = senses(w)
            if not ss:
                continue
            g.gloss[w] = ss[0]["gloss"]
            g.content[w] = content_words(ss[0]["gloss"], drop=frozenset({w}))
            g.synonym[w] = [s.lower() for s in synonyms(w)]
            g.antonym[w] = [a.lower() for a in antonyms(w)]
            g.similar[w] = [s.lower() for s in similar_tos(w)]
            g.is_a[w] = [_lemma_head(h) for h in ss[0]["hypernyms"]]
            g.meronym[w] = [m.lower() for m in meronyms(w)]
            g.derivational[w] = [d.lower() for d in derivationally_related(w)]
            g.verb_group[w] = [v.lower() for v in verb_groups(w)]
            lx = lexname(w)
            if lx:
                g.lexname[w] = lx
            attr = [a.lower() for a in attributes(w)]
            if attr:
                g.attribute[w] = attr
        return g

    def words(self) -> List[str]:
        return list(self.gloss)

    def _field(self, relation: str) -> Dict[str, List[str]]:
        return getattr(self, "is_a" if relation == "is_a" else relation)

    def typed_pairs(self, relation: str, *, directed: bool = False) -> List[Tuple[str, str]]:
        """In-vocab (a, b) pairs for a word-word *relation* (deduped)."""
        store = self._field(relation)
        wset = set(self.gloss)
        pairs = set()
        for a, neigh in store.items():
            for b in neigh:
                b = b.lower()
                if b in wset and b != a:
                    pairs.add((a, b) if directed else tuple(sorted((a, b))))
        return sorted(pairs)

    def attribute_axes(self) -> List[str]:
        """The distinct noun dimensions named by the attribute relation (candidate axes)."""
        axes: Counter = Counter()
        for dims in self.attribute.values():
            for d in dims:
                axes[d] += 1
        return [d for d, _ in axes.most_common()]

    def lexnames(self) -> List[str]:
        """The distinct lexname categories present (candidate coarse axes)."""
        return sorted(set(self.lexname.values()))

    def coverage(self) -> Dict[str, float]:
        """Fraction of in-vocab words carrying each relation (for the gate/probe)."""
        n = len(self.gloss) or 1
        cov = {r: sum(1 for v in self._field(r).values() if v) / n for r in WORD_RELATIONS}
        cov["lexname"] = len(self.lexname) / n
        cov["attribute"] = len(self.attribute) / n
        return cov
