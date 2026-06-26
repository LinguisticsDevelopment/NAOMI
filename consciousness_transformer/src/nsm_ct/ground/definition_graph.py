"""The relational word-web + the baseline decomposition (M17.0).

Two things live here:

1. **A bounded decomposition to the prime floor** (``naive_decompose``) that uses
   ONLY the prime/molecule exponent maps and WordNet glosses — *our system*,
   never the DeepNSM/gold lookup. Unlike the legacy ``meaning.py`` decomposer
   (which truncates at depth 2 with a ``SOMETHING`` placeholder), this one
   recurses deeper and marks genuinely un-grounded leaves as ``UNRESOLVED`` so
   the "did it reach primes?" question is *measurable*.

2. **A materialized definition graph** (``DefinitionGraph``): word -> definition
   content words (DEFINES), hypernym lemmas (IS_A), synonyms, antonyms. This is
   the "web of relational words" the basis search (M17.2) reasons over.

The NSM primes are the axes; WordNet supplies the relationships between the
points. Nothing here writes to weights or to the legacy resolver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

from ..data_structures import ParseNode, ParseTree
from ..meaning import _STOPWORDS
from ..nsm_molecules import MOLECULES_BY_EXPONENT
from ..tokenizer import basic_tokenize
from ..wordnet import antonyms, hypernyms, senses, synonyms, wordnet_available
from .canonicalization import EXPONENT_TO_PRIME

# We deliberately recurse deeper than meaning.py's depth-2: the whole point is to
# reach the prime floor rather than truncate. Cost is bounded by max_children.
DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_CHILDREN = 3

# Distinct labels so a "couldn't reach a prime" leaf is never confused with the
# genuine SOMETHING substantive (which IS a prime), and so the gloss-head wrapper
# is pruned by canonicalization.
_UNRESOLVED = "UNRESOLVED"
_HEAD = "EXPLICATION"


def content_words(gloss: str, *, drop: FrozenSet[str] = frozenset()) -> List[str]:
    """Extract de-duplicated content words from a WordNet *gloss*.

    Drops stopwords/function words (reusing ``meaning._STOPWORDS``), one-letter
    tokens, and any word in *drop* (typically the headword itself).
    """
    out: List[str] = []
    seen = set()
    for tok in basic_tokenize(gloss):
        if tok in _STOPWORDS or len(tok) <= 1 or tok in drop:
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _prime_or_molecule(word: str) -> Optional[str]:
    """Return the axis/molecule label if *word* is a prime exponent or molecule."""
    p = EXPONENT_TO_PRIME.get(word)
    if p is not None:
        return p
    mol = MOLECULES_BY_EXPONENT.get(word)
    if mol is not None:
        return mol.name
    return None


def naive_decompose(
    word: str,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_children: int = DEFAULT_MAX_CHILDREN,
    _visited: FrozenSet[str] = frozenset(),
) -> ParseTree:
    """Bounded recursive decomposition of *word* toward the NSM prime floor.

    Base cases: a prime exponent -> single axis leaf; a molecule -> molecule leaf.
    Otherwise recurse into the first WordNet gloss's content words. When recursion
    bottoms out (depth, cycle, or no gloss) the leaf is ``UNRESOLVED`` — an honest
    marker that grounding did *not* reach a prime, not a fake SOMETHING.

    Deterministic: same inputs -> identical tree.
    """
    w = word.lower().strip()

    axis = _prime_or_molecule(w)
    if axis is not None:
        return ParseTree(root=ParseNode(label=axis, token=w))

    if max_depth <= 0 or w in _visited:
        return ParseTree(root=ParseNode(label=_UNRESOLVED, token=w))

    if wordnet_available():
        word_senses = senses(w)
        if word_senses:
            cwords = content_words(word_senses[0]["gloss"], drop=frozenset({w}))[:max_children]
            if cwords:
                head = ParseNode(label=_HEAD, token=w)
                nv = _visited | {w}
                for c in cwords:
                    child = naive_decompose(
                        c,
                        max_depth=max_depth - 1,
                        max_children=max_children,
                        _visited=nv,
                    )
                    head.children.append(child.root)
                return ParseTree(root=head)

    return ParseTree(root=ParseNode(label=_UNRESOLVED, token=w))


def definition_clause(word: str, *, max_children: int = DEFAULT_MAX_CHILDREN) -> Optional[ParseTree]:
    """The *definition* of *word* as a shallow clause of its gloss content WORDS.

    Leaves are the surface content words (un-decomposed), labelled ``WORD`` with
    the token carried — this is the clause the reduction operator (M17.1) is asked
    to collapse back toward *word*. Returns ``None`` if *word* has no usable gloss.
    """
    if not wordnet_available():
        return None
    word_senses = senses(word.lower().strip())
    if not word_senses:
        return None
    cwords = content_words(word_senses[0]["gloss"], drop=frozenset({word.lower().strip()}))[:max_children]
    if not cwords:
        return None
    head = ParseNode(label=_HEAD, token=word.lower().strip())
    for c in cwords:
        head.children.append(ParseNode(label="WORD", token=c))
    return ParseTree(root=head)


def _lemma_head(synset_name: str) -> str:
    """``"canine.n.02"`` -> ``"canine"`` (the bare lemma word of a synset name)."""
    return synset_name.split(".")[0]


@dataclass
class DefinitionGraph:
    """A materialized web of relational words over a vocabulary sample.

    Edge stores (all keyed by lower-cased word):
      ``content``  — DEFINES: the definition's content words.
      ``is_a``     — hypernym lemma words.
      ``synonym``  — synonym lemma names.
      ``antonym``  — antonym lemma names.
    """

    gloss: Dict[str, str] = field(default_factory=dict)
    content: Dict[str, List[str]] = field(default_factory=dict)
    is_a: Dict[str, List[str]] = field(default_factory=dict)
    synonym: Dict[str, List[str]] = field(default_factory=dict)
    antonym: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, words) -> "DefinitionGraph":
        """Build the graph for *words* (skips words WordNet doesn't know)."""
        g = cls()
        for raw in words:
            w = raw.lower().strip()
            if not w or w in g.gloss:
                continue
            ss = senses(w)
            if not ss:
                continue
            gloss = ss[0]["gloss"]
            g.gloss[w] = gloss
            g.content[w] = content_words(gloss, drop=frozenset({w}))
            g.is_a[w] = [_lemma_head(h) for h in ss[0]["hypernyms"]]
            g.synonym[w] = synonyms(w)
            g.antonym[w] = antonyms(w)
        return g

    def words(self) -> List[str]:
        return list(self.gloss)

    def antonym_pairs(self) -> List[tuple]:
        """Distinct (word, antonym) pairs where both words are in the graph."""
        pairs = set()
        for w, ants in self.antonym.items():
            for a in ants:
                a = a.lower()
                if a in self.gloss and w != a:
                    pairs.add(tuple(sorted((w, a))))
        return sorted(pairs)
