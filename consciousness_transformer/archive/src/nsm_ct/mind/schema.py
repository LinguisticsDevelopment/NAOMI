"""The frozen meaning-object schema — the single contract (M0).

Everything that touches meaning — the membrane (text↔meaning), the substrate
(STM/LTM graphs), and the controller (ops over the graph) — binds to *this*
contract. It does not define new types; it **re-exports and pins** the vocabulary
that already lives in ``meaning_graph``/``nsm_primes``/``structure``, so there is
one authoritative place that says "this is what a meaning object is made of."

Three layers of vocabulary (all already enumerated and tested elsewhere):

* **Node kinds** — :class:`NodeKind` ``{CONCEPT, REFERENT, CLAUSE, OPERATOR}``.
* **Edge types** — ``{SLOT, ABOUT, COREF, OPERATES_ON, SUPERSEDES, DEFINES}``.
* **Meaning-operators** — ``{NOT, MAYBE, AND, OR, IF}`` as first-class nodes.
* **Content primes** — the ~62 NSM primes (:data:`PRIME_NAMES`).
* **Typed relations** — the ~17 grammatical/semantic relations (:data:`RELATIONS`)
  plus the small set of **reasoning relations** the oracle/curriculum use
  (:data:`REASONING_RELATIONS`), which are distinct from the grammatical ones.

The reserved **variable markers** are the substantive primes ``SOMEONE`` /
``SOMETHING``; a term whose label begins with ``?`` is the unification variable
form used inside rules (see :mod:`nsm_ct.mind.knowledge`).
"""

from __future__ import annotations

from typing import Tuple

# -- node kinds & edge types (the graph's structural contract) ----------------
from ..meaning_graph import (  # noqa: F401  (re-exported as the contract)
    ABOUT,
    COREF,
    DEFINES,
    OPERATES_ON,
    SLOT,
    SUPERSEDES,
    NodeKind,
)

# -- meaning vocabulary -------------------------------------------------------
from ..nsm_primes import NUM_PRIMES, PRIME_NAMES  # noqa: F401
from ..structure import STRUCTURE_LABELS

# The 5 prime-derived meaning-operators, realized as OPERATOR nodes via
# ``meaning_graph.apply_operator`` / ``read_operator``.
MEANING_OPERATORS: Tuple[str, ...] = ("NOT", "MAYBE", "AND", "OR", "IF")

# Every typed edge in the substrate (kept here so callers don't reach past the
# contract into ``meaning_graph`` internals).
EDGE_TYPES: Tuple[str, ...] = (SLOT, ABOUT, COREF, OPERATES_ON, SUPERSEDES, DEFINES)

# The grammatical/semantic relations the parser/structure layer emits (SUBJECT,
# OBJECT, DESCRIPTION, ...). These label SLOT edges for *parsed* clauses.
RELATIONS: Tuple[str, ...] = tuple(STRUCTURE_LABELS)

# The reasoning relations the curriculum/oracle reason over. These are a SEPARATE
# vocabulary from the grammatical RELATIONS above (RESEARCH_NOTES §0k): they are
# the edge labels of taxonomy/inference facts stored in the knowledge graph.
REASONING_RELATIONS: Tuple[str, ...] = (
    "IS_A",      # taxonomy / inheritance (robin IS_A bird)
    "KIND",      # explicit kind-of (alias of the NSM KIND prime relation)
    "CAN",       # ability / property (bird CAN fly)
    "CAN_SEE",   # modus-ponens consequent used by the conditional curriculum
    "PLACE",     # locational fact (mary PLACE kitchen)
    "HAS",       # possession / part-of
)

# Reserved structural roles.
SUBJECT_ROLE = "SUBJECT"     # the subject slot of a clause
PREDICATE_ROLE = "PREDICATE"  # the predicate slot of a clause
VARIABLE_PRIMES: Tuple[str, ...] = ("SOMEONE", "SOMETHING")  # NSM variable carriers
VARIABLE_PREFIX = "?"        # a term label beginning with '?' is a unification variable


def is_variable(label: str) -> bool:
    """True if ``label`` is the unification-variable form (``?x``, ``?place``)."""
    return isinstance(label, str) and label.startswith(VARIABLE_PREFIX)


__all__ = [
    "NodeKind",
    "SLOT", "ABOUT", "COREF", "OPERATES_ON", "SUPERSEDES", "DEFINES",
    "EDGE_TYPES", "MEANING_OPERATORS",
    "PRIME_NAMES", "NUM_PRIMES", "RELATIONS", "REASONING_RELATIONS",
    "SUBJECT_ROLE", "PREDICATE_ROLE", "VARIABLE_PRIMES", "VARIABLE_PREFIX",
    "is_variable",
]
