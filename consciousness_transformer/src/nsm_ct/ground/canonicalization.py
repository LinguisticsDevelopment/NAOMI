"""Deterministic canonical form for meaning trees (M17.0).

``normalize(tree)`` returns a canonicalized *copy* so that two trees which mean
the same thing up to (a) allolex variation, (b) sibling order, and (c) duplicate
or empty-wrapper nodes compare equal by :func:`tree_key`. This is the equality
substrate the clause==word reduction (M17.1) and the basis search (M17.2) build
on — "two equivalent clauses must canonicalize identically".

Scope / honest limits (v0):
- Sibling children are treated as an **unordered, de-duplicated set** (sorted by
  a canonical key). This is correct for conjunction-style NSM explications but
  collapses any meaning that depends on sibling *order*; role-sensitive ordering
  is a documented future refinement.
- ``token`` (surface provenance) is preserved on nodes but **ignored** by
  :func:`tree_key`, so equality is over *meaning labels*, not surface words.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..data_structures import ParseNode, ParseTree
from ..nsm_primes import PRIMES, PRIME_NAMES


def _build_exponent_to_prime() -> Dict[str, str]:
    """Map every prime exponent / allolex (lower-cased) -> canonical prime name.

    Allolexes are joined with ``~`` in ``NSMPrime.exponent`` (e.g. ``"I~ME"``);
    parenthetical disambiguation is stripped (``"BE (SOMEWHERE)"`` -> ``"be"``).
    """
    mapping: Dict[str, str] = {}
    for prime in PRIMES:
        for part in prime.exponent.split("~"):
            bare = part.split("(")[0].strip().lower()
            if bare:
                mapping[bare] = prime.name
    return mapping


EXPONENT_TO_PRIME: Dict[str, str] = _build_exponent_to_prime()

# Generic structural wrappers that carry no meaning as a *leaf* (gloss-head /
# explication-root nodes). Pruned when they have no surviving children.
_PADDING_LABELS = frozenset({"EXPLICATION"})


def canon_label(label: str) -> str:
    """Fold an allolex / exponent to its canonical NSM prime name.

    Identity for labels that are already canonical prime names or that are not
    NSM exponents at all (molecule names, ordinary words, ``UNRESOLVED``).
    """
    if label in PRIME_NAMES:
        return label
    return EXPONENT_TO_PRIME.get(label.lower(), label)


def _node_key(node: ParseNode) -> str:
    """A canonical, order-stable string key for a *normalized* node subtree.

    Includes the canonical label and relation but **not** the surface token, so
    equality is over meaning, not surface form.
    """
    inner = "".join(_node_key(c) for c in node.children)
    rel = node.relation or ""
    return f"({canon_label(node.label)}|{rel}:{inner})"


def _dedup_sorted(children):
    """Sort children by canonical key and drop exact-duplicate subtrees."""
    seen = set()
    out = []
    for c in sorted(children, key=_node_key):
        k = _node_key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _norm_node(node: ParseNode) -> Optional[ParseNode]:
    label = canon_label(node.label)
    children = []
    for c in node.children:
        nc = _norm_node(c)
        if nc is not None:
            children.append(nc)
    children = _dedup_sorted(children)
    # Prune a structural wrapper that lost all of its children.
    if label in _PADDING_LABELS and not children:
        return None
    new = ParseNode(label=label, token=node.token, relation=node.relation)
    new.children = children
    return new


def normalize(tree: ParseTree) -> ParseTree:
    """Return a canonicalized copy of *tree* (idempotent)."""
    root = _norm_node(tree.root)
    if root is None:
        root = ParseNode(label="UNRESOLVED")
    return ParseTree(root=root, text=tree.text)


def tree_key(tree: ParseTree) -> str:
    """Canonical serialized key for equality / hashing of a meaning tree.

    ``tree_key(a) == tree_key(b)`` iff *a* and *b* are equivalent under the
    canonical form (allolex, sibling order, duplicates, empty wrappers folded).
    """
    return _node_key(normalize(tree).root)
