"""M2 gate: the deterministic executor solves L7-L11 from hand-written gold
op-traces, with faithful (provenance-carrying) traces and correct abstention.

Problems mirror the curriculum patterns in ``episode.py`` (``_level7``..``_level11``):
triples are lifted straight from how those levels build their gold answers via
``reasoning_oracle.derive``.
"""

from __future__ import annotations

from nsm_ct.mind import ops
from nsm_ct.mind.executor import Executor
from nsm_ct.mind.knowledge import KnowledgeGraph
from nsm_ct.reasoning_oracle import INHERITANCE, conditional_rule


def _ex(ltm: KnowledgeGraph | None = None) -> Executor:
    return Executor(ltm or KnowledgeGraph(dim=64))


# --------------------------------------------------------------------------- L8
def test_l8_negation_removal():
    """office, bathroom, NOT bathroom -> office (not the last positive statement)."""
    ex = _ex()
    out = ex.run_trace([
        ops.Op(ops.PERCEIVE, {"subject": "john", "relation": "PLACE", "value": "office"}),
        ops.Op(ops.PERCEIVE, {"subject": "john", "relation": "PLACE", "value": "bathroom"}),
        ops.Op(ops.PERCEIVE, {"subject": "john", "relation": "PLACE",
                              "value": "bathroom", "negate": True}),
        ops.Op(ops.RESPOND, {"subject": "john", "relation": "PLACE"}),
    ])
    assert out["answer"] == "office"
    assert not out["abstained"]


# --------------------------------------------------------------------------- L9
def test_l9_modus_ponens():
    """if (bill PLACE bathroom) then (bill CAN_SEE window); bill PLACE bathroom -> window."""
    ltm = KnowledgeGraph(dim=64)
    ltm.add_rule(conditional_rule("bathroom", "window"))
    ex = _ex(ltm)
    out = ex.run_trace([
        ops.Op(ops.PERCEIVE, {"subject": "bill", "relation": "PLACE", "value": "bathroom"}),
        ops.Op(ops.INFER, {}),
        ops.Op(ops.RESPOND, {"subject": "bill", "relation": "CAN_SEE"}),
    ])
    assert out["answer"] == "window"
    # Faithfulness: the INFER step's support chain actually derives the answer.
    infer_step = next(s for s in out["trace"] if s.op == ops.INFER)
    assert any(d.derived == ("bill", "CAN_SEE", "window") and d.rule == "modus_ponens"
               for d in infer_step.support)


# -------------------------------------------------------------------------- L10
def test_l10_inheritance():
    """trout IS_A fish; fish CAN swim -> trout CAN swim."""
    ltm = KnowledgeGraph(dim=64)
    ltm.add_rule(INHERITANCE)
    ex = _ex(ltm)
    out = ex.run_trace([
        ops.Op(ops.PERCEIVE, {"subject": "trout", "relation": "IS_A", "value": "fish"}),
        ops.Op(ops.PERCEIVE, {"subject": "fish", "relation": "CAN", "value": "swim"}),
        ops.Op(ops.INFER, {}),
        ops.Op(ops.RESPOND, {"subject": "trout", "relation": "CAN"}),
    ])
    assert out["answer"] == "swim"


# -------------------------------------------------------------------------- L11
def test_l11_abstain():
    """Rule needs place 'bedroom'; sandra is in 'office' -> antecedent never fires -> idk."""
    ltm = KnowledgeGraph(dim=64)
    ltm.add_rule(conditional_rule("bedroom", "stove"))
    ex = _ex(ltm)
    out = ex.run_trace([
        ops.Op(ops.PERCEIVE, {"subject": "sandra", "relation": "PLACE", "value": "office"}),
        ops.Op(ops.INFER, {}),
        ops.Op(ops.RESPOND, {"subject": "sandra", "relation": "CAN_SEE"}),
    ])
    assert out["abstained"]
    assert out["answer"] == ops.ABSTAIN


# --------------------------------------------------------------------------- L7
def test_l7_disjunction_unresolved_is_maybe():
    ex = _ex()
    out = ex.run_trace([
        ops.Op(ops.PERCEIVE, {"subject": "sandra", "relation": "PLACE",
                              "values": ["kitchen", "garden"]}),
        ops.Op(ops.RESPOND, {"subject": "sandra", "relation": "PLACE"}),
    ])
    assert out["answer"] == "maybe"


def test_l7_disjunction_resolved_by_negation():
    ex = _ex()
    out = ex.run_trace([
        ops.Op(ops.PERCEIVE, {"subject": "sandra", "relation": "PLACE",
                              "values": ["kitchen", "garden"]}),
        ops.Op(ops.PERCEIVE, {"subject": "sandra", "relation": "PLACE",
                              "value": "kitchen", "negate": True}),
        ops.Op(ops.RESPOND, {"subject": "sandra", "relation": "PLACE"}),
    ])
    assert out["answer"] == "garden"


# ---------------------------------------------------------------- other ops ---
def test_consolidate_promotes_to_ltm_and_recall():
    """CONSOLIDATE settles a fact into durable LTM; a fresh STM still recalls it."""
    ltm = KnowledgeGraph(dim=64)
    ex = _ex(ltm)
    ex.run_trace([
        ops.Op(ops.PERCEIVE, {"subject": "mary", "relation": "PLACE", "value": "kitchen"}),
        ops.Op(ops.CONSOLIDATE, {}),
    ])
    assert ("mary", "PLACE", "kitchen") in ltm.facts()
    # A brand-new episode (empty STM) can still answer from durable LTM.
    fresh = Executor(ltm)
    out = fresh.run_trace([ops.Op(ops.RESPOND, {"subject": "mary", "relation": "PLACE"})])
    assert out["answer"] == "kitchen"


def test_halt_stops_execution():
    ex = _ex()
    out = ex.run_trace([
        ops.Op(ops.HALT, {}),
        ops.Op(ops.PERCEIVE, {"subject": "x", "relation": "PLACE", "value": "y"}),
    ])
    assert len(out["trace"]) == 1 and out["trace"][0].op == ops.HALT


if __name__ == "__main__":  # pragma: no cover
    import pytest
    pytest.main([__file__, "-v"])
