"""Canonical inventory of Natural Semantic Metalanguage (NSM) primes.

The Natural Semantic Metalanguage (Wierzbicka; Goddard & Wierzbicka) proposes
a small set of indefinable, universal "semantic primes" out of which all
complex meanings can be paraphrased. This module encodes the canonical
inventory as plain constants so the rest of the system has a stable, typed
reference for the atomic units of meaning.

Reference: Goddard, C. & Wierzbicka, A. (2014). *Words and Meanings: Lexical
Semantics Across Domains, Languages, and Cultures*. The list below follows
the ~65-prime inventory as standardly presented on the NSM Homepage.

TODO(canonical-list): This implementation follows the 2014 65-prime table to
the best of our knowledge. A few details are version-dependent and should be
verified against a primary source before any linguistic claims are made:
  * Whether DON'T WANT is counted as a distinct prime or an allolex of WANT.
  * The possession prime: recent NSM lists use "(IS) MINE"; older lists and the
    project brief use "HAVE". Both spellings are recorded below.
  * The exact set of allolexes (recorded here after a "~" in the exponent).
  * BE has two distinct primes (locational "be somewhere" vs. specificational
    "be someone/something"); they are listed separately.
Where the canonical form was uncertain we kept a reasonable approximation and
flagged it rather than inventing entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class PrimeCategory(str, Enum):
    """Top-level groupings of NSM primes, following the canonical table."""

    SUBSTANTIVES = "substantives"
    RELATIONAL_SUBSTANTIVES = "relational_substantives"
    DETERMINERS = "determiners"
    QUANTIFIERS = "quantifiers"
    EVALUATORS = "evaluators"
    DESCRIPTORS = "descriptors"
    MENTAL_PREDICATES = "mental_predicates"
    SPEECH = "speech"
    ACTIONS_EVENTS_MOVEMENT = "actions_events_movement"
    EXISTENCE_POSSESSION = "existence_possession"
    LIFE_AND_DEATH = "life_and_death"
    TIME = "time"
    SPACE = "space"
    LOGICAL_CONCEPTS = "logical_concepts"
    INTENSIFIER_AUGMENTOR = "intensifier_augmentor"
    SIMILARITY = "similarity"


@dataclass(frozen=True)
class NSMPrime:
    """A single semantic prime.

    Attributes:
        name: Canonical uppercase key used throughout the codebase (e.g. "SOMEONE").
        exponent: The English exponent(s). Allolexes are joined with "~"
            (e.g. "SOMETHING~THING"), and disambiguating glosses are in
            parentheses (e.g. "BE (SOMEWHERE)").
        category: The grouping this prime belongs to.
    """

    name: str
    exponent: str
    category: PrimeCategory

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


# ---------------------------------------------------------------------------
# The canonical inventory. Ordered by category to match the standard NSM table.
# ---------------------------------------------------------------------------
_PRIME_TABLE: Tuple[Tuple[str, str, PrimeCategory], ...] = (
    # Substantives
    ("I", "I~ME", PrimeCategory.SUBSTANTIVES),
    ("YOU", "YOU", PrimeCategory.SUBSTANTIVES),
    ("SOMEONE", "SOMEONE", PrimeCategory.SUBSTANTIVES),
    ("SOMETHING", "SOMETHING~THING", PrimeCategory.SUBSTANTIVES),
    ("PEOPLE", "PEOPLE", PrimeCategory.SUBSTANTIVES),
    ("BODY", "BODY", PrimeCategory.SUBSTANTIVES),
    # Relational substantives
    ("KIND", "KIND", PrimeCategory.RELATIONAL_SUBSTANTIVES),
    ("PART", "PART", PrimeCategory.RELATIONAL_SUBSTANTIVES),
    # Determiners
    ("THIS", "THIS", PrimeCategory.DETERMINERS),
    ("THE_SAME", "THE SAME", PrimeCategory.DETERMINERS),
    ("OTHER", "OTHER~ELSE~ANOTHER", PrimeCategory.DETERMINERS),
    # Quantifiers
    ("ONE", "ONE", PrimeCategory.QUANTIFIERS),
    ("TWO", "TWO", PrimeCategory.QUANTIFIERS),
    ("SOME", "SOME", PrimeCategory.QUANTIFIERS),
    ("ALL", "ALL", PrimeCategory.QUANTIFIERS),
    ("MUCH_MANY", "MUCH~MANY", PrimeCategory.QUANTIFIERS),
    ("LITTLE_FEW", "LITTLE~FEW", PrimeCategory.QUANTIFIERS),
    # Evaluators
    ("GOOD", "GOOD", PrimeCategory.EVALUATORS),
    ("BAD", "BAD", PrimeCategory.EVALUATORS),
    # Descriptors
    ("BIG", "BIG", PrimeCategory.DESCRIPTORS),
    ("SMALL", "SMALL", PrimeCategory.DESCRIPTORS),
    # Mental predicates
    ("THINK", "THINK", PrimeCategory.MENTAL_PREDICATES),
    ("KNOW", "KNOW", PrimeCategory.MENTAL_PREDICATES),
    ("WANT", "WANT", PrimeCategory.MENTAL_PREDICATES),
    # TODO(canonical-list): DON'T WANT may be an allolex of WANT rather than a
    # distinct prime; included separately here pending verification.
    ("DONT_WANT", "DON'T WANT", PrimeCategory.MENTAL_PREDICATES),
    ("FEEL", "FEEL", PrimeCategory.MENTAL_PREDICATES),
    ("SEE", "SEE", PrimeCategory.MENTAL_PREDICATES),
    ("HEAR", "HEAR", PrimeCategory.MENTAL_PREDICATES),
    # Speech
    ("SAY", "SAY", PrimeCategory.SPEECH),
    ("WORDS", "WORDS", PrimeCategory.SPEECH),
    ("TRUE", "TRUE", PrimeCategory.SPEECH),
    # Actions, events, movement
    ("DO", "DO", PrimeCategory.ACTIONS_EVENTS_MOVEMENT),
    ("HAPPEN", "HAPPEN", PrimeCategory.ACTIONS_EVENTS_MOVEMENT),
    ("MOVE", "MOVE", PrimeCategory.ACTIONS_EVENTS_MOVEMENT),
    # Existence and possession
    ("BE_SOMEWHERE", "BE (SOMEWHERE)", PrimeCategory.EXISTENCE_POSSESSION),
    ("THERE_IS", "THERE IS", PrimeCategory.EXISTENCE_POSSESSION),
    ("BE_SOMEONE_SOMETHING", "BE (SOMEONE/SOMETHING)", PrimeCategory.EXISTENCE_POSSESSION),
    # TODO(canonical-list): recent NSM lists use "(IS) MINE" for the possession
    # prime; the project brief uses "HAVE". Both names recorded.
    ("MINE", "(IS) MINE~HAVE", PrimeCategory.EXISTENCE_POSSESSION),
    # Life and death
    ("LIVE", "LIVE", PrimeCategory.LIFE_AND_DEATH),
    ("DIE", "DIE", PrimeCategory.LIFE_AND_DEATH),
    # Time
    ("WHEN", "WHEN~TIME", PrimeCategory.TIME),
    ("NOW", "NOW", PrimeCategory.TIME),
    ("BEFORE", "BEFORE", PrimeCategory.TIME),
    ("AFTER", "AFTER", PrimeCategory.TIME),
    ("A_LONG_TIME", "A LONG TIME", PrimeCategory.TIME),
    ("A_SHORT_TIME", "A SHORT TIME", PrimeCategory.TIME),
    ("FOR_SOME_TIME", "FOR SOME TIME", PrimeCategory.TIME),
    ("MOMENT", "MOMENT", PrimeCategory.TIME),
    # Space
    ("WHERE", "WHERE~PLACE", PrimeCategory.SPACE),
    ("HERE", "HERE", PrimeCategory.SPACE),
    ("ABOVE", "ABOVE", PrimeCategory.SPACE),
    ("BELOW", "BELOW", PrimeCategory.SPACE),
    ("FAR", "FAR", PrimeCategory.SPACE),
    ("NEAR", "NEAR", PrimeCategory.SPACE),
    ("SIDE", "SIDE", PrimeCategory.SPACE),
    ("INSIDE", "INSIDE", PrimeCategory.SPACE),
    ("TOUCH", "TOUCH", PrimeCategory.SPACE),
    # Logical concepts
    ("NOT", "NOT", PrimeCategory.LOGICAL_CONCEPTS),
    ("MAYBE", "MAYBE", PrimeCategory.LOGICAL_CONCEPTS),
    ("CAN", "CAN", PrimeCategory.LOGICAL_CONCEPTS),
    ("BECAUSE", "BECAUSE", PrimeCategory.LOGICAL_CONCEPTS),
    ("IF", "IF", PrimeCategory.LOGICAL_CONCEPTS),
    # Intensifier, augmentor
    ("VERY", "VERY", PrimeCategory.INTENSIFIER_AUGMENTOR),
    ("MORE", "MORE", PrimeCategory.INTENSIFIER_AUGMENTOR),
    # Similarity
    ("LIKE", "LIKE~AS~WAY", PrimeCategory.SIMILARITY),
)


PRIMES: Tuple[NSMPrime, ...] = tuple(
    NSMPrime(name=name, exponent=exponent, category=category)
    for name, exponent, category in _PRIME_TABLE
)
"""Immutable ordered tuple of every NSM prime in the canonical inventory."""

PRIME_NAMES: Tuple[str, ...] = tuple(p.name for p in PRIMES)
"""Just the canonical uppercase keys, in canonical order."""

PRIMES_BY_NAME: Dict[str, NSMPrime] = {p.name: p for p in PRIMES}
"""Lookup from canonical key -> :class:`NSMPrime`."""

# Sanity check: the canonical inventory is ~65 primes. If this assertion ever
# fires after an edit, re-check against the reference table rather than
# silently changing the constant.
NUM_PRIMES: int = len(PRIMES)
assert 60 <= NUM_PRIMES <= 70, f"Unexpected prime count {NUM_PRIMES}; verify table."


def primes_in_category(category: PrimeCategory) -> List[NSMPrime]:
    """Return all primes belonging to ``category``, in canonical order."""
    return [p for p in PRIMES if p.category == category]


def prime_index(name: str) -> int:
    """Return the canonical index of a prime by name.

    Useful for building fixed-width activation vectors over the prime set.

    Raises:
        KeyError: if ``name`` is not a canonical prime key.
    """
    try:
        return PRIME_NAMES.index(name)
    except ValueError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown NSM prime: {name!r}") from exc
