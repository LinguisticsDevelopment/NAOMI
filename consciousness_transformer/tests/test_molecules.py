"""Tests for the NSM semantic-molecule registry.

Invariants checked:
  (a) Every molecule has a non-empty ``source`` (anti-hallucination invariant).
  (b) Any molecule with an ``explication`` has node labels that are all valid
      prime OR molecule names.
  (c) Molecule-to-prime flattening works correctly and is cycle-guarded (we
      construct a deliberate in-memory cycle and confirm it terminates).
  (d) ``MOLECULES_BY_EXPONENT`` lookups work.
"""

import pytest

from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.nsm_molecules import (
    MOLECULES,
    MOLECULES_BY_EXPONENT,
    MOLECULES_BY_NAME,
    MOLECULE_NAMES,
    MoleculeCategory,
    NSMMolecule,
    NUM_MOLECULES,
    flatten_molecule_to_prime_names,
    molecule_index,
    molecules_in_category,
)
from nsm_ct.nsm_primes import PRIME_NAMES, NUM_PRIMES
from nsm_ct.thought import MAX_MEANING_PRIMES, meaning_prime_ids


# ---------------------------------------------------------------------------
# (a) Anti-hallucination invariant: every molecule MUST have a real source
# ---------------------------------------------------------------------------

def test_every_molecule_has_nonempty_source():
    """Registry invariant: no hallucinated / unsourced molecule entries."""
    for mol in MOLECULES:
        assert mol.source, (
            f"Molecule {mol.name!r} has an empty source — "
            "add a real bibliographic citation."
        )


def test_molecule_dataclass_rejects_empty_source():
    """NSMMolecule.__post_init__ enforces the source invariant at construction."""
    with pytest.raises(ValueError, match="empty source"):
        NSMMolecule(
            name="FAKE",
            exponents=("fake",),
            category=MoleculeCategory.ENVIRONMENT,
            source="",
        )


# ---------------------------------------------------------------------------
# (b) Explication node labels are valid prime or molecule names
# ---------------------------------------------------------------------------

_VALID_LABELS = frozenset(PRIME_NAMES) | frozenset(MOLECULE_NAMES)


def test_explication_labels_are_valid():
    """Any molecule with an explication must use only known prime/molecule labels."""
    for mol in MOLECULES:
        if mol.explication is None:
            continue
        for node in mol.explication.iter_preorder():
            assert node.label in _VALID_LABELS, (
                f"Molecule {mol.name!r}: explication node label {node.label!r} "
                "is neither a known prime nor a known molecule."
            )


# ---------------------------------------------------------------------------
# (c) Flatten-to-primes + cycle guard
# ---------------------------------------------------------------------------

def test_flatten_prime_name_returns_itself():
    """A prime name is its own leaf in the flatten recursion."""
    assert flatten_molecule_to_prime_names("THINK") == ["THINK"]
    assert flatten_molecule_to_prime_names("GOOD") == ["GOOD"]


def test_flatten_molecule_with_no_explication_returns_empty():
    """A molecule without an explication contributes nothing to the prime list."""
    # All current molecules have explication=None, so this covers the default.
    for mol in MOLECULES:
        if mol.explication is None:
            result = flatten_molecule_to_prime_names(mol.name)
            assert result == [], (
                f"Expected [] for molecule {mol.name!r} (no explication), "
                f"got {result!r}"
            )


def test_flatten_molecule_with_explication():
    """A molecule with a two-prime explication flattens to those two primes."""
    tree = ParseTree(
        root=ParseNode(
            label="SOMETHING",
            children=[ParseNode(label="GOOD")],
        ),
        text="test explication",
    )
    # Build a temporary in-memory molecule with a real explication.
    mol = NSMMolecule(
        name="TEST_EXPLICATION_MOL",
        exponents=("testword",),
        category=MoleculeCategory.ENVIRONMENT,
        source="Unit test fixture — not a registry entry.",
        explication=tree,
    )
    # Temporarily register it so flatten can recurse.
    MOLECULES_BY_NAME["TEST_EXPLICATION_MOL"] = mol  # type: ignore[index]
    try:
        result = flatten_molecule_to_prime_names("TEST_EXPLICATION_MOL")
        assert result == ["SOMETHING", "GOOD"], result
    finally:
        del MOLECULES_BY_NAME["TEST_EXPLICATION_MOL"]  # type: ignore[attr-defined]


def test_cycle_guard_terminates():
    """Deliberately cyclic molecule explications must not cause infinite recursion."""
    # Molecule A's explication references molecule B; B's explication references A.
    # The cycle-guard in flatten_molecule_to_prime_names must break the loop.

    # We build the trees first (referencing names that we'll register), then
    # temporarily patch MOLECULES_BY_NAME.
    tree_a = ParseTree(
        root=ParseNode(label="CYCLE_B"),
        text="cycle-a explication",
    )
    tree_b = ParseTree(
        root=ParseNode(label="CYCLE_A"),
        text="cycle-b explication",
    )
    mol_a = NSMMolecule(
        name="CYCLE_A",
        exponents=("cyclea",),
        category=MoleculeCategory.ENVIRONMENT,
        source="Unit test fixture — not a registry entry.",
        explication=tree_a,
    )
    mol_b = NSMMolecule(
        name="CYCLE_B",
        exponents=("cycleb",),
        category=MoleculeCategory.ENVIRONMENT,
        source="Unit test fixture — not a registry entry.",
        explication=tree_b,
    )
    MOLECULES_BY_NAME["CYCLE_A"] = mol_a  # type: ignore[index]
    MOLECULES_BY_NAME["CYCLE_B"] = mol_b  # type: ignore[index]
    try:
        # Both should return [] because the cycle is detected and guarded.
        result_a = flatten_molecule_to_prime_names("CYCLE_A")
        result_b = flatten_molecule_to_prime_names("CYCLE_B")
        assert isinstance(result_a, list)
        assert isinstance(result_b, list)
        # With cycles and no primes at leaves, both should come back empty.
        assert result_a == [], f"Expected [] for CYCLE_A, got {result_a!r}"
        assert result_b == [], f"Expected [] for CYCLE_B, got {result_b!r}"
    finally:
        del MOLECULES_BY_NAME["CYCLE_A"]  # type: ignore[attr-defined]
        del MOLECULES_BY_NAME["CYCLE_B"]  # type: ignore[attr-defined]


def test_meaning_prime_ids_with_molecule_node():
    """meaning_prime_ids handles a tree with a molecule-name node gracefully.

    When a ParseTree node's label is a molecule name (with no explication),
    it contributes 0 prime ids — the function must not raise.
    """
    tree = ParseTree(
        root=ParseNode(
            label="WATER",  # a registered molecule with explication=None
            children=[ParseNode(label="THINK")],  # a real prime
        ),
        text="test",
    )
    ids = meaning_prime_ids(tree)
    # WATER has no explication -> contributes 0 ids; THINK contributes 1.
    assert ids == [PRIME_NAMES.index("THINK") + 1]


def test_meaning_prime_ids_unknown_label_is_skipped():
    """meaning_prime_ids silently skips labels that are neither prime nor molecule."""
    tree = ParseTree(
        root=ParseNode(label="NOT_A_PRIME_OR_MOLECULE"),
        text="test",
    )
    ids = meaning_prime_ids(tree)
    assert ids == []


def test_meaning_prime_ids_capped_at_max():
    """meaning_prime_ids honours the MAX_MEANING_PRIMES cap."""
    children = [ParseNode(label=name) for name in PRIME_NAMES[: MAX_MEANING_PRIMES + 5]]
    tree = ParseTree(root=ParseNode(label="THINK", children=children))
    ids = meaning_prime_ids(tree)
    assert len(ids) <= MAX_MEANING_PRIMES


# ---------------------------------------------------------------------------
# (d) MOLECULES_BY_EXPONENT lookups
# ---------------------------------------------------------------------------

def test_molecules_by_exponent_covers_all_exponents():
    """Every exponent string in every molecule appears in MOLECULES_BY_EXPONENT."""
    for mol in MOLECULES:
        for exp in mol.exponents:
            key = exp.lower()
            assert key in MOLECULES_BY_EXPONENT, (
                f"Exponent {exp!r} of molecule {mol.name!r} not in "
                "MOLECULES_BY_EXPONENT."
            )
            assert MOLECULES_BY_EXPONENT[key].name == mol.name


def test_molecules_by_exponent_spot_checks():
    """Spot-check a selection of common exponent words."""
    checks = {
        "water": "WATER",
        "fire": "FIRE",
        "sky": "SKY",
        "ground": "GROUND",
        "sun": "SUN",
        "day": "DAY",
        "night": "NIGHT",
        "hands": "HANDS",
        "hand": "HANDS",
        "mouth": "MOUTH",
        "eyes": "EYES",
        "eye": "EYES",
        "head": "HEAD",
        "men": "MEN",
        "man": "MEN",
        "women": "WOMEN",
        "woman": "WOMEN",
        "children": "CHILDREN",
        "child": "CHILDREN",
        "long": "LONG",
        "round": "ROUND",
        "mother": "MOTHER",
        "father": "FATHER",
    }
    for exponent, expected_name in checks.items():
        assert exponent in MOLECULES_BY_EXPONENT, (
            f"Expected exponent {exponent!r} in MOLECULES_BY_EXPONENT."
        )
        assert MOLECULES_BY_EXPONENT[exponent].name == expected_name


# ---------------------------------------------------------------------------
# Registry structure sanity checks
# ---------------------------------------------------------------------------

def test_molecule_count_in_expected_range():
    """The registry should have between 30 and 100 documented molecules."""
    assert 30 <= NUM_MOLECULES <= 100, f"Unexpected count: {NUM_MOLECULES}"


def test_molecule_names_are_unique():
    assert len(set(MOLECULE_NAMES)) == len(MOLECULE_NAMES)


def test_molecule_index_round_trip():
    for i, name in enumerate(MOLECULE_NAMES):
        assert molecule_index(name) == i
        assert MOLECULES_BY_NAME[name].name == name


def test_molecule_index_raises_for_unknown():
    with pytest.raises(KeyError):
        molecule_index("NOT_A_MOLECULE")


def test_every_category_has_at_least_one_molecule():
    for cat in MoleculeCategory:
        mols = molecules_in_category(cat)
        assert mols, f"Category {cat} has no molecules in the registry."


def test_flatten_raises_for_unknown_name():
    """flatten_molecule_to_prime_names raises KeyError for totally unknown names."""
    with pytest.raises(KeyError):
        flatten_molecule_to_prime_names("TOTALLY_UNKNOWN_XYZ")
