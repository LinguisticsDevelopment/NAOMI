"""Stage 4 gate — STM/LTM read-time resolution, proven symbolically (no training).

The core L8 scenario and friends, the thing the whole design is for: distinct
clause nodes sharing one referent, resolved by recency / negation / disjunction.
"""

import hashlib

from nsm_ct.clause_psyche_graph import LTM, MAYBE, RESOLVED, STM
from nsm_ct.data_structures import ParseNode, ParseTree
from nsm_ct.nsm_primes import PRIME_NAMES
from nsm_ct.tpr import TPRCodec


class StubResolver:
    def resolve(self, word, context=None):
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        return ParseTree(root=ParseNode(label=PRIME_NAMES[h % len(PRIME_NAMES)], token=word))


def _stm(dim=64):
    return STM(TPRCodec(dim=dim), StubResolver())


def _label(stm, status_val):
    status, val = status_val
    return status, stm.value_label(val)


def test_l8_negation_resolves_to_the_other_place():
    stm = _stm()
    k = stm.add_clause("mary", "PLACE", "kitchen")
    o = stm.add_clause("mary", "PLACE", "office")
    # mary IS in two places (contradiction kept as two distinct clauses)...
    assert len(stm.graph.clauses_about(stm.graph.referent_index["mary"])) == 2
    stm.negate(o)  # "mary is not in the office"
    assert _label(stm, stm.read("mary", "PLACE")) == (RESOLVED, "kitchen")


def test_one_referent_shared_across_clauses():
    stm = _stm()
    stm.add_clause("mary", "PLACE", "kitchen")
    stm.add_clause("mary", "PLACE", "office")
    assert len(stm.graph.referent_index) == 1  # one mary, not two


def test_recency_moved_wins_without_any_negation():
    stm = _stm()
    stm.add_clause("mary", "PLACE", "kitchen")
    stm.add_clause("mary", "PLACE", "office")  # "moved to the office"
    assert _label(stm, stm.read("mary", "PLACE")) == (RESOLVED, "office")


def test_unresolved_disjunction_is_maybe():
    stm = _stm()
    stm.add_disjunction("mary", "PLACE", ["kitchen", "office"])
    status, val = stm.read("mary", "PLACE")
    assert status == MAYBE and val is None


def test_disjunction_narrowed_by_negation_resolves():
    stm = _stm()
    _op, (k, o) = stm.add_disjunction("mary", "PLACE", ["kitchen", "office"])
    stm.negate(k)  # "mary is not in the kitchen" -> only office survives
    assert _label(stm, stm.read("mary", "PLACE")) == (RESOLVED, "office")


def test_everything_negated_is_maybe():
    stm = _stm()
    k = stm.add_clause("mary", "PLACE", "kitchen")
    stm.negate(k)
    status, val = stm.read("mary", "PLACE")
    assert status == MAYBE and val is None


def test_distinct_subjects_do_not_interfere():
    stm = _stm()
    stm.add_clause("mary", "PLACE", "kitchen")
    stm.add_clause("john", "PLACE", "office")
    assert _label(stm, stm.read("mary", "PLACE")) == (RESOLVED, "kitchen")
    assert _label(stm, stm.read("john", "PLACE")) == (RESOLVED, "office")


def test_ltm_consolidates_and_recalls_resolved_facts():
    stm = _stm()
    k = stm.add_clause("mary", "PLACE", "kitchen")
    o = stm.add_clause("mary", "PLACE", "office")
    stm.negate(o)
    ltm = LTM(stm.codec)
    assert ltm.consolidate_stm(stm) == 1
    assert ltm.recall("mary", "PLACE") == (RESOLVED, "kitchen")
    # a fresh STM with no fact falls back to LTM
    fresh = _stm()
    assert fresh.read("mary", "PLACE", ltm=ltm) == (RESOLVED, "kitchen")
