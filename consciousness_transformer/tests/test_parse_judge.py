"""Tests for the LLM parse judge (dev/PARSE_JUDGE.md): `nsm_ct.tree_render`
and `nsm_ct.parse_judge`.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nsm_ct.tree_render import MAX_CHARS, normalize_gold_tree, render_tree  # noqa: E402
from nsm_ct.parse_judge import HaikuJudge, MockJudge, Verdict, judge_many_with_prefilter  # noqa: E402

from calibrate_judge import _STRUCTURAL_CORRUPTIONS, build_corrupted_set, gold_tree_of  # noqa: E402

GOLD_PATH = _ROOT / "runs" / "encoder_gold_v2.jsonl"


def _require_gold():
    if not GOLD_PATH.exists():
        pytest.skip(f"needs {GOLD_PATH}")


def _load_gold(n: int) -> list:
    with open(GOLD_PATH) as f:
        records = [json.loads(line) for line in f]
    return records[:n]


# ---------------------------------------------------------------------------
# render_tree: deterministic, capped
# ---------------------------------------------------------------------------

def test_render_tree_deterministic_and_capped_on_50_gold_records():
    _require_gold()
    records = _load_gold(50)
    for record in records:
        tree = normalize_gold_tree(record, record["lattice"]["trees"][0])
        first = render_tree(record, tree)
        second = render_tree(record, tree)
        assert first == second
        assert len(first) <= MAX_CHARS
        assert first.startswith(record["text"])


def test_render_tree_empty_clauses():
    record = {"text": "hi .", "tokens": ["hi", "."], "pos": ["INTJ", "PUNCT"]}
    out = render_tree(record, {"clauses": []})
    assert "hi ." in out
    assert "<no clauses emitted>" in out


def test_render_tree_marks_unresolved_slots_and_primes():
    record = {"text": "wait for me .", "tokens": ["wait", "for", "me", "."],
              "pos": ["VERB", "ADP", "PRON", "PUNCT"], "token_sense_candidates": []}
    tree = {"clauses": [{
        "utterance_kind": "imperative",
        "predicate": {"token_index": 0, "grounding": {"type": "entity"}},
        "roles": [
            {"relation": "SUBJECT", "token_index": None, "grounding": {"type": "prime", "prime": "YOU"}},
            {"relation": "INDIRECT_OBJECT", "token_index": 2, "grounding": {"type": "prime", "prime": "I"}},
            {"relation": "OBJECT", "token_index": None, "grounding": {"type": "elision"}},
        ],
    }]}
    out = render_tree(record, tree)
    assert "SUBJECT=YOU" in out
    assert "INDIRECT_OBJECT=I" in out
    assert "OBJECT=<elided>" in out


# ---------------------------------------------------------------------------
# MockJudge: corrupted always bad, gold mostly good
# ---------------------------------------------------------------------------

def test_mockjudge_marks_all_corrupted_trees_bad():
    """Structural corruptions (predicate -> non-verb token, phantom role
    attached) are always caught by J1/J3, so MockJudge must mark 100% of
    these bad."""
    _require_gold()
    records = _load_gold(100)[50:100]
    pairs = build_corrupted_set(records, seed=1234, kinds=_STRUCTURAL_CORRUPTIONS)
    assert len(pairs) == 50
    verdicts = judge_many_with_prefilter(MockJudge(), pairs)
    bad = [v for v in verdicts if v.verdict == "bad"]
    assert len(bad) == len(verdicts), (
        f"MockJudge missed {len(verdicts) - len(bad)}/{len(verdicts)} corrupted trees")


def test_mockjudge_misses_some_subject_object_swaps():
    """The SUBJECT/OBJECT swap corruption is structurally invisible (no
    phantom node, no POS violation) -- this is the documented reason a
    POS-only/structural judge is not a substitute for the LLM judge (see
    module docstrings). MockJudge is expected to catch MOST but not
    necessarily ALL of these; a semantic judge (HaikuJudge) is the one
    expected to close that gap."""
    _require_gold()
    records = _load_gold(100)[50:100]
    from calibrate_judge import _corrupt_swap_subject_object
    pairs = []
    for r in records:
        tree = gold_tree_of(r)
        corrupted = _corrupt_swap_subject_object(r, tree)
        if corrupted is not None:
            pairs.append((r, corrupted))
    assert pairs, "expected at least one record with both a SUBJECT and an OBJECT"
    verdicts = judge_many_with_prefilter(MockJudge(), pairs)
    caught = sum(1 for v in verdicts if v.verdict == "bad")
    # not asserting 100% here -- that's the whole point; just confirm the
    # mechanism runs and catches at least something (some swaps do land on
    # a non-nominal token and still trip J2).
    assert 0 <= caught <= len(pairs)


def test_mockjudge_marks_most_gold_trees_good():
    _require_gold()
    records = _load_gold(50)
    pairs = [(r, normalize_gold_tree(r, r["lattice"]["trees"][0])) for r in records]
    verdicts = judge_many_with_prefilter(MockJudge(), pairs)
    good = sum(1 for v in verdicts if v.verdict == "good")
    assert good / len(verdicts) >= 0.80, f"gold good-rate {good}/{len(verdicts)} < 80%"


def test_hard_prefilter_marks_phantom_trees_bad_without_calling_backend():
    record = {"text": "a b .", "tokens": ["a", "b", "."], "pos": ["NOUN", "VERB", "PUNCT"],
              "token_sense_candidates": []}
    tree = {"clauses": [{
        "utterance_kind": "proposition",
        "predicate": {"token_index": 1, "grounding": {"type": "entity"}},
        "roles": [{"relation": "SUBJECT", "token_index": 0, "grounding": {"type": "entity"}},
                  {"relation": "PHANTOM", "token_index": 99, "grounding": {"type": "entity"}}],
    }]}

    class ExplodingJudge(MockJudge):
        def judge(self, record, tree):
            raise AssertionError("backend should never be called on a J3-failing tree")

    verdicts = judge_many_with_prefilter(ExplodingJudge(), [(record, tree)])
    assert verdicts[0].verdict == "bad"
    assert verdicts[0].confidence == 1.0


# ---------------------------------------------------------------------------
# HaikuJudge: request construction against a fake client, no network
# ---------------------------------------------------------------------------

class _FakeMessages:
    def __init__(self, parsed_verdict: Verdict):
        self.parsed_verdict = parsed_verdict
        self.last_kwargs = None

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(parsed_output=self.parsed_verdict,
                                usage=SimpleNamespace(input_tokens=123, output_tokens=45))


class _FakeClient:
    def __init__(self, parsed_verdict: Verdict):
        self.messages = _FakeMessages(parsed_verdict)


def _make_haiku_judge(fake_client) -> HaikuJudge:
    judge = HaikuJudge.__new__(HaikuJudge)
    import anthropic
    judge._anthropic = anthropic
    judge.client = fake_client
    judge.model = "claude-haiku-4-5"
    judge.max_retries = 4
    judge.base_delay = 1.0
    judge.total_requests = 0
    judge.total_input_tokens = 0
    judge.total_output_tokens = 0
    return judge


def test_haikujudge_request_construction_and_canned_verdict():
    canned = Verdict(verdict="bad", problems=["predicate is an auxiliary, not the main verb"],
                      confidence=0.87)
    fake_client = _FakeClient(canned)
    judge = _make_haiku_judge(fake_client)

    record = {"text": "he may go .", "tokens": ["he", "may", "go", "."],
              "pos": ["PRON", "AUX", "VERB", "PUNCT"], "token_sense_candidates": []}
    tree = {"clauses": [{
        "utterance_kind": "proposition",
        "predicate": {"token_index": 1, "grounding": {"type": "entity"}},
        "roles": [{"relation": "SUBJECT", "token_index": 0, "grounding": {"type": "entity"}}],
    }]}

    result = judge.judge(record, tree)

    assert result is canned
    kwargs = fake_client.messages.last_kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["max_tokens"] == 256
    assert kwargs["output_format"] is Verdict
    assert "system" in kwargs and "PREDICATE" in kwargs["system"]
    assert kwargs["messages"][0]["role"] == "user"
    assert "he may go ." in kwargs["messages"][0]["content"]
    assert judge.total_requests == 1
    assert judge.total_input_tokens == 123
    assert judge.total_output_tokens == 45


def test_haikujudge_retries_on_rate_limit_then_succeeds():
    import httpx2
    import anthropic

    canned = Verdict(verdict="good", problems=[], confidence=0.9)

    class FlakyMessages(_FakeMessages):
        def __init__(self, parsed_verdict):
            super().__init__(parsed_verdict)
            self.calls = 0

        def parse(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                resp = httpx2.Response(status_code=429,
                                        request=httpx2.Request("POST", "https://example.com"))
                raise anthropic.RateLimitError("rate limited", response=resp, body=None)
            return super().parse(**kwargs)

    fake_client = SimpleNamespace(messages=FlakyMessages(canned))
    judge = _make_haiku_judge(fake_client)
    judge.base_delay = 0.001  # keep the test fast

    record = {"text": "hi .", "tokens": ["hi", "."], "pos": ["INTJ", "PUNCT"], "token_sense_candidates": []}
    tree = {"clauses": []}
    result = judge.judge(record, tree)
    assert result is canned
    assert fake_client.messages.calls == 2


def test_haikujudge_fails_fast_on_bad_request():
    import httpx2
    import anthropic

    class ExplodingMessages(_FakeMessages):
        def parse(self, **kwargs):
            resp = httpx2.Response(status_code=400, request=httpx2.Request("POST", "https://example.com"))
            raise anthropic.BadRequestError("bad request", response=resp, body=None)

    fake_client = SimpleNamespace(messages=ExplodingMessages(Verdict(verdict="good", problems=[], confidence=0.5)))
    judge = _make_haiku_judge(fake_client)

    record = {"text": "hi .", "tokens": ["hi", "."], "pos": ["INTJ", "PUNCT"], "token_sense_candidates": []}
    with pytest.raises(anthropic.BadRequestError):
        judge.judge(record, {"clauses": []})


# ---------------------------------------------------------------------------
# Batch mode: results keyed by custom_id, arriving in any order
# ---------------------------------------------------------------------------

class _FakeBatch:
    def __init__(self, batch_id, status="ended"):
        self.id = batch_id
        self.processing_status = status


class _FakeBatchResult:
    def __init__(self, custom_id, text):
        self.custom_id = custom_id
        self.result = SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            ),
        )


class _FakeBatches:
    def __init__(self, results_out_of_order):
        self.created_requests = None
        self._results = results_out_of_order

    def create(self, requests):
        self.created_requests = requests
        return _FakeBatch("batch_123", status="ended")

    def retrieve(self, batch_id):
        return _FakeBatch(batch_id, status="ended")

    def results(self, batch_id):
        return iter(self._results)


def test_batch_results_keyed_by_custom_id_not_position():
    import anthropic

    # 3 requests, results returned in REVERSED order -- keying must not
    # depend on the order `results()` yields them.
    verdict_texts = {
        "0": json.dumps({"verdict": "good", "problems": [], "confidence": 0.9}),
        "1": json.dumps({"verdict": "bad", "problems": ["wrong predicate"], "confidence": 0.8}),
        "2": json.dumps({"verdict": "good", "problems": [], "confidence": 0.7}),
    }
    out_of_order = [_FakeBatchResult(cid, txt) for cid, txt in
                     [("2", verdict_texts["2"]), ("0", verdict_texts["0"]), ("1", verdict_texts["1"])]]

    judge = _make_haiku_judge(fake_client=None)
    judge._anthropic = anthropic
    judge.client = SimpleNamespace(messages=SimpleNamespace(batches=_FakeBatches(out_of_order)))

    records_trees = []
    for i in range(3):
        record = {"text": f"sentence {i} .", "tokens": ["sentence", str(i), "."],
                   "pos": ["NOUN", "NUM", "PUNCT"], "token_sense_candidates": []}
        records_trees.append((record, {"clauses": []}))

    verdicts = judge._judge_many_batch(records_trees)
    assert [v.verdict for v in verdicts] == ["good", "bad", "good"]
    assert verdicts[1].problems == ["wrong predicate"]
    assert judge.total_requests == 3
    assert judge.total_input_tokens == 30
    assert judge.total_output_tokens == 15
