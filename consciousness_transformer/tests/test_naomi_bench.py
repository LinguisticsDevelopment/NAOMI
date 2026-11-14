"""Fast tests for the bench helpers in scripts/naomi_run.py (no parsing)."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "naomi_run", Path(__file__).resolve().parents[1] / "scripts" / "naomi_run.py")
naomi_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(naomi_run)


def test_shuffle_questions_is_cyclic_deterministic_and_keeps_gold():
    def mk():
        return {"s1": [{"question": f"q{j}", "answer_idx": j, "options": ["a", "b"]} for j in range(3)],
                "s2": [{"question": "only", "answer_idx": 0, "options": ["a", "b"]}]}
    a, b = mk(), mk()
    naomi_run.shuffle_questions(a)
    naomi_run.shuffle_questions(b)
    assert [g["question"] for g in a["s1"]] == ["q1", "q2", "q0"] == [g["question"] for g in b["s1"]]
    assert [g["_orig_question"] for g in a["s1"]] == ["q0", "q1", "q2"]
    assert [g["answer_idx"] for g in a["s1"]] == [0, 1, 2]       # gold index untouched
    assert a["s2"][0]["question"] == "only"
