"""NSM Meaning Resolver: grounds a word into a meaning tree (tree of NSM primes).

This module implements :class:`NSMMeaningResolver`, the Stage-B replacement for
:class:`~nsm_ct.thought.MockMeaningResolver`. It grounds every input word in real
semantic data — no hallucinated semantics:

1. **Prime exponent** — word is (an allolex of) an NSM prime  -> single prime leaf.
2. **Molecule exponent** — word matches a registered NSM molecule  -> molecule node
   (or the molecule's real explication tree if one is available).
3. **WordNet** — look up senses via :func:`nsm_ct.wordnet.senses`:
   - Person sense (lexname ``noun.person`` OR a hypernym name contains "person")
     -> ``SOMEONE`` leaf.
   - Otherwise gloss-decomposition: tokenise the first sense's gloss, strip
     stopwords, recursively resolve content words up to ``max_depth`` (default 2)
     and nest their meaning roots under a head ``SOMETHING`` node.
4. **Fallback** — unknown word: ``SOMETHING`` (or ``SOMEONE`` when ``context``
   carries ``{"pos": "PROPN"}`` or ``{"pos": "PRON"}``).

All resolutions are cached on the resolver instance (dict keyed on the lower-cased
word) so repeated words are O(1).

Anti-hallucination guarantee: prime and molecule mappings come exclusively from
:mod:`nsm_ct.nsm_primes` and :mod:`nsm_ct.nsm_molecules`; gloss decompositions
derive only from real WordNet glosses; no prime is ever invented for an unknown
word (fallback is always SOMETHING / SOMEONE).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Set

from .data_structures import ParseNode, ParseTree
from .nsm_molecules import MOLECULES_BY_EXPONENT
from .nsm_primes import PRIMES
from .thought import AbstractMeaningResolver

# ---------------------------------------------------------------------------
# Stopwords / function words excluded from gloss decomposition
# ---------------------------------------------------------------------------
_STOPWORDS: FrozenSet[str] = frozenset({
    # articles, determiners
    "a", "an", "the", "this", "that", "these", "those",
    # pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    # conjunctions / prepositions / auxiliaries
    "and", "or", "but", "nor", "so", "for", "yet",
    "in", "on", "at", "by", "to", "of", "with", "from", "into",
    "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "as", "if", "while", "because", "since", "although",
    "about", "against", "along", "among", "around", "near", "up",
    # copulas / auxiliaries / modals
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could",
    # relative / interrogative pronouns and determiners
    "which", "who", "whom", "whose", "what", "where", "when", "how", "why",
    # misc function words
    "not", "no", "nor", "than", "too", "very", "just", "only",
    "also", "both", "each", "more", "most", "other", "some", "such",
    "any", "few", "own", "same",
    # punctuation tokens that basic_tokenize might emit
    "(", ")", ",", ".", ";", ":", "-", "'s", "\"", "'",
    # "especially" and similar hedge words in WordNet glosses
    "especially", "particularly", "used", "usually", "often", "also",
    "typically", "generally", "e", "g", "i", "e",
})

# Maximum number of gloss-derived children under the SOMETHING head node.
_MAX_GLOSS_CHILDREN = 3

# Hard maximum recursion depth for gloss decomposition.
_DEFAULT_MAX_DEPTH = 2

# ---------------------------------------------------------------------------
# Build prime exponent -> prime name map once at module load
# ---------------------------------------------------------------------------

def _build_prime_exponent_map() -> Dict[str, str]:
    """Return a dict mapping every prime exponent (lower-case) to its prime name.

    Allolexes are joined with "~" in ``NSMPrime.exponent``, e.g. ``"I~ME"``
    or ``"SOMETHING~THING"``.  This function splits on "~", strips any
    parenthetical disambiguation (e.g. ``"BE (SOMEWHERE)"`` -> ``"be"``), and
    lower-cases each form.
    """
    mapping: Dict[str, str] = {}
    for prime in PRIMES:
        # Split allolexes on "~"
        parts = prime.exponent.split("~")
        for part in parts:
            # Strip parenthetical disambiguation: "BE (SOMEWHERE)" -> "be"
            bare = part.split("(")[0].strip().lower()
            if bare:
                mapping[bare] = prime.name
    return mapping


_PRIME_EXPONENT_MAP: Dict[str, str] = _build_prime_exponent_map()


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class NSMMeaningResolver(AbstractMeaningResolver):
    """Grounds a word into a ParseTree of NSM prime / molecule-name nodes.

    Resolution order (first match wins):
    1. Prime exponent  -> single prime leaf
    2. Molecule exponent -> molecule node (or its explication if non-None)
    3. WordNet person sense -> SOMEONE leaf
    4. WordNet gloss -> recursive gloss decomposition up to ``max_depth``
    5. Fallback: SOMETHING (or SOMEONE if context indicates a proper noun)

    All results are cached per (lower-cased) word so repeated calls are O(1).

    Args:
        max_depth: Maximum recursion depth for gloss decomposition (default 2).
    """

    def __init__(self, max_depth: int = _DEFAULT_MAX_DEPTH) -> None:
        self._max_depth = max_depth
        # Cache key: (word_lower, ctx_pos_or_None) -> ParseTree
        self._cache: Dict[tuple, ParseTree] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def resolve(self, word: str, context: object = None) -> ParseTree:
        """Resolve *word* to a meaning tree.

        Args:
            word: A single surface word (lowercased internally).
            context: Optional dict; if ``context.get("pos")`` is ``"PROPN"``
                or ``"PRON"``, an unknown word returns SOMEONE instead of
                SOMETHING.

        Returns:
            A :class:`~nsm_ct.data_structures.ParseTree` whose nodes carry
            prime or molecule names.

        Caching note: the cache key includes the context POS tag so that the
        same word can resolve differently when tagged as a proper noun vs. an
        ordinary word.  For most words (primes, molecules, WordNet entries) the
        context has no effect; the context-sensitive path only applies to the
        fallback (unknown word), so the minor duplication is harmless.
        """
        key = word.lower().strip()
        # Derive a lightweight context key (only the POS tag affects resolution)
        ctx_key: Optional[str] = None
        if isinstance(context, dict):
            ctx_key = context.get("pos")
        cache_key = (key, ctx_key)

        if cache_key in self._cache:
            return self._cache[cache_key]

        tree = self._resolve_uncached(key, context=context, depth=0, visited=frozenset())
        self._cache[cache_key] = tree
        # Also cache under the no-context key when context doesn't change the result
        # (i.e., not a fallback case) so repeated in-sentence calls are O(1).
        if ctx_key is None:
            pass  # already stored above
        return tree

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_uncached(
        self,
        word: str,
        *,
        context: object,
        depth: int,
        visited: frozenset,
    ) -> ParseTree:
        """Core resolution logic (no caching layer here)."""

        # 1. Prime exponent?
        prime_name = _PRIME_EXPONENT_MAP.get(word)
        if prime_name is not None:
            return ParseTree(root=ParseNode(label=prime_name))

        # 2. Molecule exponent?
        molecule = MOLECULES_BY_EXPONENT.get(word)
        if molecule is not None:
            if molecule.explication is not None:
                return molecule.explication
            return ParseTree(root=ParseNode(label=molecule.name))

        # 3. WordNet
        from .wordnet import senses, wordnet_available  # local import — graceful

        if wordnet_available():
            word_senses = senses(word)
            if word_senses:
                first = word_senses[0]

                # 3a. Person sense?
                if self._is_person_sense(first):
                    return ParseTree(root=ParseNode(label="SOMEONE"))

                # 3b. Gloss decomposition (depth-limited)
                if depth < self._max_depth:
                    tree = self._gloss_decompose(
                        word, first["gloss"], depth=depth, visited=visited
                    )
                    if tree is not None:
                        return tree

                # No decomposition possible at this depth — return SOMETHING
                return ParseTree(root=ParseNode(label="SOMETHING"))

        # 4. Fallback
        return self._fallback(context)

    def _is_person_sense(self, sense_dict: dict) -> bool:
        """Return True if the sense dict represents a person / human role.

        Checks (in order):
        * WordNet lexname is ``"noun.person"`` (requires importing wn directly)
        * Any hypernym synset name contains "person"
        * The gloss starts with "a person" (fast string check as last resort)
        """
        # Fast heuristic: gloss starts with "a person" (e.g. teacher, doctor …)
        gloss = sense_dict.get("gloss", "").lower()
        if gloss.startswith("a person") or gloss.startswith("someone who"):
            return True

        # Hypernym check: common for noun.person synsets
        for h in sense_dict.get("hypernyms", []):
            if "person" in h.lower():
                return True

        # Lexname check via wn (most reliable but requires direct wn import)
        sense_id = sense_dict.get("sense_id", "")
        if sense_id:
            try:
                from nltk.corpus import wordnet as _wn  # type: ignore
                synset = _wn.synset(sense_id)
                if synset.lexname() == "noun.person":
                    return True
            except Exception:
                pass  # graceful fallback if wn call fails

        return False

    def _gloss_decompose(
        self,
        word: str,
        gloss: str,
        *,
        depth: int,
        visited: frozenset,
    ) -> Optional[ParseTree]:
        """Decompose *word* via its WordNet *gloss* into a meaning tree.

        Extracts content words from the gloss (drops stopwords and the input
        word itself), recursively resolves each up to ``_max_depth``, then
        nests their roots under a head ``SOMETHING`` node.

        Returns ``None`` when no content words can be extracted.
        """
        from .tokenizer import basic_tokenize  # local import

        tokens = basic_tokenize(gloss)
        content_words = [
            t for t in tokens
            if t not in _STOPWORDS and t != word and len(t) > 1
        ]
        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_content: list = []
        for t in content_words:
            if t not in seen:
                seen.add(t)
                unique_content.append(t)
        unique_content = unique_content[:_MAX_GLOSS_CHILDREN]

        if not unique_content:
            return None

        new_visited = visited | {word}
        head = ParseNode(label="SOMETHING")
        for cword in unique_content:
            if cword in new_visited:
                continue  # cycle guard
            # Use cached result if available, else recurse
            child_cache_key = (cword, None)
            if child_cache_key in self._cache:
                child_root = self._cache[child_cache_key].root
            else:
                child_tree = self._resolve_uncached(
                    cword,
                    context=None,
                    depth=depth + 1,
                    visited=new_visited,
                )
                # Cache the child resolution (no context)
                self._cache[child_cache_key] = child_tree
                child_root = child_tree.root
            head.children.append(ParseNode(label=child_root.label))

        return ParseTree(root=head)

    @staticmethod
    def _fallback(context: object) -> ParseTree:
        """Return SOMEONE for PROPN/PRON context, else SOMETHING."""
        if isinstance(context, dict):
            pos = context.get("pos")
            if pos in {"PROPN", "PRON"}:
                return ParseTree(root=ParseNode(label="SOMEONE"))
        return ParseTree(root=ParseNode(label="SOMETHING"))
