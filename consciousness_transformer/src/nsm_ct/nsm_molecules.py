"""Registry of NSM semantic molecules.

Semantic molecules are mid-level concepts that bridge the atomic semantic
primes (see :mod:`nsm_ct.nsm_primes`) and fully complex lexical meanings.
Unlike primes, molecules *can* be decomposed into primes, but they are
conventionally taken as chunks when explicating still-more-complex words.

Every entry in this registry carries a real ``source`` citation (the
anti-hallucination invariant: no entry may be registered without a citable
source). An ``explication`` (a :class:`~nsm_ct.data_structures.ParseTree`
whose nodes carry prime or molecule names) is included **only** when the
decomposition was transcribed verbatim from a fetched source; otherwise
``explication=None``.

Documented molecule lists are drawn from:
  * Wikipedia: Natural semantic metalanguage
    https://en.wikipedia.org/wiki/Natural_semantic_metalanguage
  * Levisen, C. & Waters, S. (Eds.) (2017). *Cultural Keywords in Discourse*.
    Amsterdam: John Benjamins.
  * Goddard, C. (2010). Semantic molecules and semantic complexity (with
    special reference to "environmental" molecules). *Review of Cognitive
    Linguistics*, 8(1), 123–155.  DOI: 10.1075/rcl.8.1.05god
  * Goddard, C. (2012). Semantic primes, semantic molecules, semantic
    templates: Key concepts in the NSM approach to lexical typology.
    *Linguistics*, 50(3), 711–743.
  * Goddard, C. & Wierzbicka, A. (2007). Semantic molecules. In
    *Proceedings of the 2006 Conference of the Australian Linguistic
    Society*.  Griffith University Research Repository.
  * Goddard, C. & Wierzbicka, A. (2014). *Words and Meanings: Lexical
    Semantics Across Domains, Languages, and Cultures*. OUP.

NOTE ON EXPLICATIONS: The full-text of Goddard (2010) and related papers is
paywalled; no verbatim explication text could be retrieved by the automated
fetch pipeline that built this registry. All explication fields are therefore
``None`` pending a future pass with authenticated access. Molecule *list
membership* (name + exponents + category + source) is factually grounded in
the cited Wikipedia article and the secondary sources listed above.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .data_structures import ParseTree


class MoleculeCategory(str, Enum):
    """Top-level groupings for NSM semantic molecules."""

    BODY_PARTS = "body_parts"
    PHYSICAL_QUALITIES = "physical_qualities"
    BIOSOCIAL = "biosocial"
    ENVIRONMENT = "environment"


# Short citation strings reused across many entries.
_CITE_WIKI_LEVISEN = (
    "Wikipedia: Natural semantic metalanguage "
    "(https://en.wikipedia.org/wiki/Natural_semantic_metalanguage); "
    "Levisen, C. & Waters, S. (Eds.) (2017). Cultural Keywords in Discourse. "
    "Amsterdam: John Benjamins."
)
_CITE_GODDARD_2010 = (
    "Goddard, C. (2010). Semantic molecules and semantic complexity "
    "(with special reference to 'environmental' molecules). "
    "Review of Cognitive Linguistics, 8(1), 123-155. "
    "DOI: 10.1075/rcl.8.1.05god"
)
_CITE_GODDARD_2012 = (
    "Goddard, C. (2012). Semantic primes, semantic molecules, semantic "
    "templates: Key concepts in the NSM approach to lexical typology. "
    "Linguistics, 50(3), 711-743."
)
_CITE_GODDARD_WIERZBICKA_2014 = (
    "Goddard, C. & Wierzbicka, A. (2014). Words and Meanings: Lexical "
    "Semantics Across Domains, Languages, and Cultures. Oxford University "
    "Press. "
    "Also: Wikipedia: Natural semantic metalanguage; "
    "Levisen, C. & Waters, S. (Eds.) (2017). Cultural Keywords in Discourse. "
    "Amsterdam: John Benjamins."
)


@dataclass(frozen=True)
class NSMMolecule:
    """A single NSM semantic molecule.

    Attributes:
        name: Canonical UPPER_CASE key used throughout the codebase
            (e.g. ``"HANDS"``).
        exponents: The English exponent word(s) for this molecule.  Multiple
            exponents are listed as allolexes in the same tuple slot (e.g.
            ``("hands",)`` or ``("man", "men")``).
        category: The grouping this molecule belongs to
            (:class:`MoleculeCategory`).
        source: Bibliographic citation(s) for the molecule's *list membership*.
            This field MUST NOT be empty — it is the anti-hallucination
            invariant for this registry.
        explication: Optional decomposition tree (a
            :class:`~nsm_ct.data_structures.ParseTree` whose node labels are
            prime or molecule names). Set to ``None`` unless the decomposition
            was transcribed verbatim from a retrieved source.
    """

    name: str
    exponents: Tuple[str, ...]
    category: MoleculeCategory
    source: str
    explication: Optional[ParseTree] = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError(
                f"NSMMolecule {self.name!r} has an empty source — "
                "every molecule MUST carry a real citation."
            )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


# ---------------------------------------------------------------------------
# The documented molecule inventory.
# Ordered by category then alphabetically within each category.
# Each entry: (name, exponents, category, source)
# ---------------------------------------------------------------------------
_MOLECULE_TABLE: Tuple[Tuple[str, Tuple[str, ...], MoleculeCategory, str], ...] = (
    # ------------------------------------------------------------------ #
    # BODY PARTS                                                          #
    # Source for all: Wikipedia NSM article (which cites Levisen &        #
    # Waters 2017) lists these as proposed universal/near-universal body- #
    # part molecules.  Goddard & Wierzbicka (2014) and the "Bodies and   #
    # their parts" paper (Goddard 2007, Griffith Univ. repository) also   #
    # document this set.                                                  #
    # ------------------------------------------------------------------ #
    (
        "BLOOD",
        ("blood",),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "BONES",
        ("bones", "bone"),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "EARS",
        ("ears", "ear"),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "EYES",
        ("eyes", "eye"),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "FACE",
        ("face",),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "FINGERS",
        ("fingers", "finger"),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "HANDS",
        ("hands", "hand"),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "HEAD",
        ("head",),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "LEGS",
        ("legs", "leg"),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "MOUTH",
        ("mouth",),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "NOSE",
        ("nose",),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "SKIN",
        ("skin",),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "TEETH",
        ("teeth", "tooth"),
        MoleculeCategory.BODY_PARTS,
        _CITE_WIKI_LEVISEN,
    ),
    # ------------------------------------------------------------------ #
    # PHYSICAL QUALITIES                                                  #
    # ------------------------------------------------------------------ #
    (
        "FLAT",
        ("flat",),
        MoleculeCategory.PHYSICAL_QUALITIES,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "HARD",
        ("hard",),
        MoleculeCategory.PHYSICAL_QUALITIES,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "HEAVY",
        ("heavy",),
        MoleculeCategory.PHYSICAL_QUALITIES,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "LONG",
        ("long",),
        MoleculeCategory.PHYSICAL_QUALITIES,
        _CITE_GODDARD_2012,
    ),
    (
        "ROUND",
        ("round",),
        MoleculeCategory.PHYSICAL_QUALITIES,
        _CITE_GODDARD_2012,
    ),
    (
        "SHARP",
        ("sharp",),
        MoleculeCategory.PHYSICAL_QUALITIES,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "SMOOTH",
        ("smooth",),
        MoleculeCategory.PHYSICAL_QUALITIES,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "SOFT",
        ("soft",),
        MoleculeCategory.PHYSICAL_QUALITIES,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "THIN",
        ("thin",),
        MoleculeCategory.PHYSICAL_QUALITIES,
        _CITE_WIKI_LEVISEN,
    ),
    # ------------------------------------------------------------------ #
    # BIOSOCIAL                                                           #
    # ------------------------------------------------------------------ #
    (
        "BE_BORN",
        ("be born", "born"),
        MoleculeCategory.BIOSOCIAL,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "CHILDREN",
        ("children", "child"),
        MoleculeCategory.BIOSOCIAL,
        _CITE_GODDARD_WIERZBICKA_2014,
    ),
    (
        "FATHER",
        ("father",),
        MoleculeCategory.BIOSOCIAL,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "HUSBAND",
        ("husband",),
        MoleculeCategory.BIOSOCIAL,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "MEN",
        ("men", "man"),
        MoleculeCategory.BIOSOCIAL,
        _CITE_GODDARD_WIERZBICKA_2014,
    ),
    (
        "MOTHER",
        ("mother",),
        MoleculeCategory.BIOSOCIAL,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "WIFE",
        ("wife",),
        MoleculeCategory.BIOSOCIAL,
        _CITE_WIKI_LEVISEN,
    ),
    (
        "WOMEN",
        ("women", "woman"),
        MoleculeCategory.BIOSOCIAL,
        _CITE_GODDARD_WIERZBICKA_2014,
    ),
    # ------------------------------------------------------------------ #
    # ENVIRONMENT                                                         #
    # Goddard (2010) is the primary reference for the environmental set.  #
    # Wikipedia NSM article (citing Levisen & Waters 2017 and Goddard     #
    # 2010) also lists these as potentially universal.                    #
    # ------------------------------------------------------------------ #
    (
        "DAY",
        ("day",),
        MoleculeCategory.ENVIRONMENT,
        _CITE_GODDARD_2010,
    ),
    (
        "FIRE",
        ("fire",),
        MoleculeCategory.ENVIRONMENT,
        _CITE_GODDARD_2010,
    ),
    (
        "GROUND",
        ("ground",),
        MoleculeCategory.ENVIRONMENT,
        _CITE_GODDARD_2010,
    ),
    (
        "NIGHT",
        ("night",),
        MoleculeCategory.ENVIRONMENT,
        _CITE_GODDARD_2010,
    ),
    (
        "SKY",
        ("sky",),
        MoleculeCategory.ENVIRONMENT,
        _CITE_GODDARD_2010,
    ),
    (
        "SUN",
        ("sun",),
        MoleculeCategory.ENVIRONMENT,
        _CITE_GODDARD_2010,
    ),
    (
        "WATER",
        ("water",),
        MoleculeCategory.ENVIRONMENT,
        _CITE_GODDARD_2010,
    ),
)


# ---------------------------------------------------------------------------
# Public registry objects  (mirror the nsm_primes.py API)
# ---------------------------------------------------------------------------

MOLECULES: Tuple[NSMMolecule, ...] = tuple(
    NSMMolecule(name=name, exponents=exponents, category=category, source=source)
    for name, exponents, category, source in _MOLECULE_TABLE
)
"""Immutable ordered tuple of every documented NSM molecule."""

MOLECULE_NAMES: Tuple[str, ...] = tuple(m.name for m in MOLECULES)
"""Just the canonical UPPER_CASE keys, in canonical order."""

MOLECULES_BY_NAME: Dict[str, NSMMolecule] = {m.name: m for m in MOLECULES}
"""Lookup from canonical key -> :class:`NSMMolecule`."""

MOLECULES_BY_EXPONENT: Dict[str, NSMMolecule] = {}
"""Lookup from any exponent word (lower-case) -> :class:`NSMMolecule`.

When a molecule has multiple exponents (e.g. ``("men", "man")``), every
exponent word maps to the same molecule.
"""
for _mol in MOLECULES:
    for _exp in _mol.exponents:
        MOLECULES_BY_EXPONENT[_exp.lower()] = _mol

NUM_MOLECULES: int = len(MOLECULES)
"""Total number of molecules in the registry."""

assert 30 <= NUM_MOLECULES <= 100, (
    f"Unexpected molecule count {NUM_MOLECULES}; verify the table."
)


def molecule_index(name: str) -> int:
    """Return the canonical index of a molecule by name.

    Raises:
        KeyError: if ``name`` is not a canonical molecule key.
    """
    try:
        return MOLECULE_NAMES.index(name)
    except ValueError as exc:
        raise KeyError(f"Unknown NSM molecule: {name!r}") from exc


def molecules_in_category(category: MoleculeCategory) -> List[NSMMolecule]:
    """Return all molecules belonging to ``category``, in canonical order."""
    return [m for m in MOLECULES if m.category == category]


# ---------------------------------------------------------------------------
# Helpers for recursive prime flattening (used by thought.meaning_prime_ids)
# ---------------------------------------------------------------------------

_MAX_FLATTEN_DEPTH = 10  # hard guard against runaway recursion


def flatten_molecule_to_prime_names(
    name: str,
    *,
    _visited: Optional[frozenset] = None,
    _depth: int = 0,
) -> List[str]:
    """Recursively expand a molecule name to a list of prime names.

    The expansion walks the molecule's ``explication`` ParseTree (if any).
    Each node label that is itself a molecule is expanded recursively; labels
    that are primes are collected as-is.  If a molecule has no explication the
    function returns an empty list (the caller should handle the ``None`` case
    gracefully — typically by skipping or falling back to the bag-of-ids path).

    Args:
        name: A canonical molecule key (e.g. ``"WATER"``).
        _visited: Internal cycle-guard set (callers should not pass this).
        _depth: Internal depth counter (callers should not pass this).

    Returns:
        A (possibly empty) list of prime name strings.

    Raises:
        KeyError: if ``name`` is not a known prime OR molecule.
    """
    from .nsm_primes import PRIME_NAMES  # local import to avoid circularity

    if _visited is None:
        _visited = frozenset()

    if _depth > _MAX_FLATTEN_DEPTH:
        return []  # depth-exceeded guard

    if name in _visited:
        return []  # cycle guard

    # Base case: it's a prime.
    if name in PRIME_NAMES:
        return [name]

    # Recursive case: it's a molecule.
    mol = MOLECULES_BY_NAME.get(name)
    if mol is None:
        raise KeyError(f"Unknown prime or molecule name: {name!r}")

    if mol.explication is None:
        return []  # no decomposition available

    new_visited = _visited | {name}
    result: List[str] = []
    for node in mol.explication.iter_preorder():
        result.extend(
            flatten_molecule_to_prime_names(
                node.label,
                _visited=new_visited,
                _depth=_depth + 1,
            )
        )
    return result
