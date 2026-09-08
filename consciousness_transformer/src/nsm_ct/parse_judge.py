"""LLM-as-judge for flattened parse trees (dev/PARSE_JUDGE.md).

WHY this exists (lead, 2026-09-08): once the encoder self-trains, its own
good parses on edge cases become gold for the next round. The admission
filter into that self-training loop must NOT be the model's own confidence
-- `scripts/probe_encoder_complement.py`'s measurement found the encoder's
best-of-8 tree scores edge-F1 0.70 but its own rank-1 (most-confident) tree
only 0.53, i.e. confidence is uncalibrated. Instead: flatten each candidate
tree to text (`nsm_ct.tree_render.render_tree`) and have a cheap LLM
(Claude Haiku 4.5) judge it good/bad.

Two backends behind one `Judge` interface:

- `HaikuJudge` -- the real judge, via the official `anthropic` SDK
  (`client.messages.parse` for a single record, the Message Batches API for
  volume).
- `MockJudge` -- a deterministic, reference-free heuristic (no network, no
  key) used in tests and for offline sanity checks: the structural J1/J2/J3
  checks from `probe_encoder_complement.judge_tree`, plus first-sense/POS
  agreement. It is intentionally NOT a substitute for the LLM judge -- it
  cannot see e.g. a SUBJECT/OBJECT swap between two equally nominal tokens,
  which is exactly the kind of semantic error POS-only judging is too
  lenient about (the motivation for this module in the first place).

`judge_many_with_prefilter` is the entry point both `scripts/judge_parses.py`
and `scripts/calibrate_judge.py` use: it applies the HARD PRE-FILTER (a tree
failing J3 -- phantom nodes -- is marked bad without ever calling the
backend) before delegating anything left to the judge.
"""

from __future__ import annotations

import json
import time
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .tree_render import pos_agrees_with_first_sense, render_tree

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class Verdict(BaseModel):
    verdict: Literal["good", "bad"]
    problems: List[str] = Field(default_factory=list)
    confidence: float


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

RUBRIC = """You are judging whether a flattened predicate-argument parse of an \
English sentence is a FAITHFUL reading of that sentence.

You will be shown the original sentence and, for each clause the parser \
extracted, a PREDICATE plus its roles (SUBJECT, OBJECT, and others), each \
with the surface word and, where available, a short gloss of the sense it \
was grounded to.

Judge STRICTLY, against these criteria:
- PREDICATE must be the clause's actual main verb, not an auxiliary/modal \
standing in for it (a bare "may", "was", "is", "has", "will", "do" as the \
predicate is WRONG unless it truly is the main verb, e.g. copular "is" in \
"he is a thief").
- SUBJECT / OBJECT / other core roles must be the sentence's real arguments \
of that predicate -- not an unrelated word, and not swapped with each other.
- No invented, duplicated, or misattached arguments -- every node must be \
licensed by the sentence.
- Sense glosses shown for content words should be plausible for how the \
word is used in THIS sentence, not a wildly wrong sense.
- The clause split should match the sentence's actual clause structure \
(not fusing two clauses into one, or splitting one clause into two).

A tree that gets the predicate right but has ANY wrong core role (or vice \
versa) is "bad" -- there is no partial credit. Only mark "good" when the \
whole rendered tree is a faithful reading of the sentence.

List every problem you find in `problems` (empty list if none). Set \
`confidence` to how sure you are of the verdict, from 0.0 to 1.0."""

QUESTION = ("Is this parse a faithful predicate-argument reading of the sentence? "
            "Judge strictly by the rubric.")

_JSON_INSTRUCTION = ("\n\nRespond with ONLY a single JSON object matching this schema, "
                      "no other text:\n")


# ---------------------------------------------------------------------------
# Reference-free structural checks (J1/J2/J3), lifted from
# `probe_encoder_complement.judge_tree` -- shared by MockJudge and the hard
# pre-filter. Operates on the normalized clause/node shape (see
# `tree_render`'s module docstring).
# ---------------------------------------------------------------------------

VERB_LIKE = {"VERB", "AUX"}
NOMINAL_LIKE = {"NOUN", "PROPN", "PRON"}


def _pos_at(pos_list: list, idx: Optional[int]) -> Optional[str]:
    if idx is None or not (0 <= idx < len(pos_list)):
        return None
    return pos_list[idx]


def _n_content_tokens(tokens: list) -> int:
    return sum(1 for t in tokens if any(c.isalnum() for c in t))


def j1_j2_j3(record: dict, tree: dict) -> dict:
    """-> {"j1", "j2", "j3", "usable", "reasons", "n_nodes"}. J1 predicate-ok
    (the predicate token is VERB/AUX by POS), J2 subject-ok (a real/synth
    SUBJECT of the kind the clause's `utterance_kind` requires), J3
    no-phantoms (no out-of-range or duplicate `token_index`, and total
    emitted nodes don't exceed the sentence's content-token count)."""
    pos_list = record.get("pos", [])
    tokens = record.get("tokens", [])
    clauses = tree.get("clauses", []) if tree else []
    if not clauses:
        return {"j1": False, "j2": False, "j3": False, "usable": False,
                "reasons": ["no clauses emitted"], "n_nodes": 0}

    j1, j2, j3 = True, True, True
    reasons: List[str] = []
    total_nodes = 0
    for ci, clause in enumerate(clauses):
        pred = clause.get("predicate", {}) or {}
        pred_idx = pred.get("token_index")
        if _pos_at(pos_list, pred_idx) not in VERB_LIKE:
            j1 = False
            reasons.append(f"clause {ci}: J1 fail (predicate token_index={pred_idx} "
                            f"pos={_pos_at(pos_list, pred_idx)})")

        kind = clause.get("utterance_kind", "proposition")
        roles = clause.get("roles", [])
        subj_roles = [r for r in roles if r.get("relation") == "SUBJECT"]
        if kind in ("proposition", "imperative"):
            # A SUBJECT role must be PRESENT; if it has a surface token
            # (token_index is not None), that token must be nominal. A
            # None-token_index SUBJECT (synthesized/elided/anaphoric -- an
            # implicit repeated subject in a coordinated clause, an
            # imperative's synthesized addressee, ...) can't be faulted on
            # POS -- there is no surface token to check.
            ok = bool(subj_roles) and all(
                r.get("token_index") is None or _pos_at(pos_list, r.get("token_index")) in NOMINAL_LIKE
                for r in subj_roles
            )
            if not ok:
                j2 = False
                reasons.append(f"clause {ci}: J2 fail (no NOUN/PROPN/PRON SUBJECT)")

        node_list = [pred] + roles
        total_nodes += len(node_list)
        seen = set()
        for n in node_list:
            idx = n.get("token_index")
            if idx is None:
                continue
            if not (0 <= idx < len(tokens)):
                j3 = False
                reasons.append(f"clause {ci}: J3 fail (token_index {idx} out of range)")
            elif idx in seen:
                j3 = False
                reasons.append(f"clause {ci}: J3 fail (duplicate token_index {idx})")
            else:
                seen.add(idx)

    content_n = _n_content_tokens(tokens)
    if total_nodes > content_n:
        j3 = False
        reasons.append(f"J3 fail (total nodes {total_nodes} > content tokens {content_n})")

    usable = j1 and j2 and j3
    return {"j1": j1, "j2": j2, "j3": j3, "usable": usable, "reasons": reasons, "n_nodes": total_nodes}


# ---------------------------------------------------------------------------
# Judge interface
# ---------------------------------------------------------------------------

RecordTree = Tuple[dict, dict]


class Judge:
    """Common interface both backends implement."""

    name: str = "base"
    model: str = ""

    def judge(self, record: dict, tree: dict) -> Verdict:
        raise NotImplementedError

    def judge_many(self, records_trees: List[RecordTree], batch: bool = False) -> List[Verdict]:
        """Default: sequential `judge` calls. `batch=True` is a hint only
        backends that support it (currently `HaikuJudge`) act on."""
        return [self.judge(r, t) for r, t in records_trees]


# ---------------------------------------------------------------------------
# MockJudge -- deterministic, offline, reference-free
# ---------------------------------------------------------------------------


class MockJudge(Judge):
    """J1/J2/J3 plus first-sense/POS agreement on every sense-grounded node.
    Deliberately NOT a stand-in for `HaikuJudge`'s semantic judgment (see
    module docstring) -- it is what tests and offline sanity checks run
    against when no API key is available."""

    name = "mock"
    model = "mock-heuristic"

    def judge(self, record: dict, tree: dict) -> Verdict:
        j = j1_j2_j3(record, tree)
        problems = list(j["reasons"])
        if not j["usable"]:
            return Verdict(verdict="bad", problems=problems or ["fails J1/J2/J3"], confidence=0.9)

        # First-sense/POS agreement: informational only (contributes to
        # `problems`/`confidence`, does not by itself flip a structurally
        # sound tree to "bad"). It is noisy at small N on real prose (a
        # short clause with 1-2 sense nodes swings from 0% to 100%
        # disagreement on a single ambiguous gloss), so gating the verdict
        # on it pushes the real-gold good-rate well below what the
        # structural checks alone support; it stays a confidence signal, not
        # a second bad-if-any-fails gate on top of J1/J2/J3.
        total, disagreements = 0, 0
        for clause in tree.get("clauses", []):
            nodes = [clause.get("predicate", {}) or {}] + clause.get("roles", [])
            for n in nodes:
                g = n.get("grounding") or {}
                if g.get("type") != "sense":
                    continue
                agree = pos_agrees_with_first_sense(record, n.get("token_index"))
                if agree is None:
                    continue
                total += 1
                if not agree:
                    disagreements += 1

        ratio = disagreements / total if total else 0.0
        if disagreements:
            problems.append(f"first-sense/POS disagreement on {disagreements}/{total} sense nodes")
        confidence = 0.85 - 0.3 * ratio
        return Verdict(verdict="good", problems=problems, confidence=round(confidence, 2))


# ---------------------------------------------------------------------------
# HaikuJudge -- the real judge
# ---------------------------------------------------------------------------


class HaikuJudge(Judge):
    """Claude Haiku 4.5 via the official Anthropic Python SDK. Retries on
    `RateLimitError` / `APIStatusError` (>=500) with exponential backoff;
    fails fast on a 4xx (bad request -- a prompt/schema bug, not transient)."""

    name = "haiku"

    def __init__(self, model: str = "claude-haiku-4-5", max_retries: int = 4,
                 base_delay: float = 1.0):
        import anthropic  # local import: MockJudge users never need this dependency

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _record_usage(self, usage) -> None:
        self.total_requests += 1
        if usage is None:
            return
        self.total_input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.total_output_tokens += getattr(usage, "output_tokens", 0) or 0

    def judge(self, record: dict, tree: dict) -> Verdict:
        content = render_tree(record, tree) + "\n\n" + QUESTION
        delay = self.base_delay
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.messages.parse(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": content}],
                    system=RUBRIC,
                    output_format=Verdict,
                )
                self._record_usage(getattr(response, "usage", None))
                return response.parsed_output
            except self._anthropic.RateLimitError as e:
                last_exc = e
            except self._anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_exc = e
                else:
                    raise
            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2
        assert last_exc is not None
        raise last_exc

    def judge_many(self, records_trees: List[RecordTree], batch: bool = False) -> List[Verdict]:
        if not batch:
            return super().judge_many(records_trees, batch=False)
        return self._judge_many_batch(records_trees)

    # -- batch mode -----------------------------------------------------

    def _batch_requests(self, renderings: List[str], use_output_config: bool):
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        schema = Verdict.model_json_schema()
        system = RUBRIC if use_output_config else RUBRIC + _JSON_INSTRUCTION + json.dumps(schema)
        requests = []
        for i, rendering in enumerate(renderings):
            content = rendering + "\n\n" + QUESTION
            params = {
                "model": self.model,
                "max_tokens": 256,
                "system": system,
                "messages": [{"role": "user", "content": content}],
            }
            if use_output_config:
                params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
            requests.append(Request(custom_id=str(i), params=MessageCreateParamsNonStreaming(**params)))
        return requests

    def _judge_many_batch(self, records_trees: List[RecordTree]) -> List[Verdict]:
        renderings = [render_tree(r, t) for r, t in records_trees]

        try:
            requests = self._batch_requests(renderings, use_output_config=True)
            message_batch = self.client.messages.batches.create(requests=requests)
        except self._anthropic.BadRequestError:
            requests = self._batch_requests(renderings, use_output_config=False)
            message_batch = self.client.messages.batches.create(requests=requests)

        while True:
            message_batch = self.client.messages.batches.retrieve(message_batch.id)
            if message_batch.processing_status == "ended":
                break
            time.sleep(20)

        verdicts_by_id = {}
        for result in self.client.messages.batches.results(message_batch.id):
            self.total_requests += 1
            if result.result.type != "succeeded":
                verdicts_by_id[result.custom_id] = Verdict(
                    verdict="bad", problems=[f"batch request {result.result.type}"], confidence=0.0)
                continue
            msg = result.result.message
            usage = getattr(msg, "usage", None)
            if usage is not None:
                self.total_input_tokens += getattr(usage, "input_tokens", 0) or 0
                self.total_output_tokens += getattr(usage, "output_tokens", 0) or 0
            text = next((b.text for b in msg.content if b.type == "text"), "")
            try:
                verdicts_by_id[result.custom_id] = Verdict.model_validate(json.loads(text))
            except Exception as e:  # malformed judge output -- don't crash the run
                verdicts_by_id[result.custom_id] = Verdict(
                    verdict="bad", problems=[f"unparseable judge output: {e}"], confidence=0.0)

        missing = Verdict(verdict="bad", problems=["missing batch result"], confidence=0.0)
        return [verdicts_by_id.get(str(i), missing) for i in range(len(renderings))]


# ---------------------------------------------------------------------------
# Hard pre-filter + entry point
# ---------------------------------------------------------------------------


def judge_many_with_prefilter(judge: Judge, records_trees: List[RecordTree],
                               batch: bool = False) -> List[Verdict]:
    """Apply the hard J3 pre-filter, then delegate whatever survives to
    `judge`. A tree with phantom nodes never reaches the backend -- no
    wasted API call on structurally-broken output."""
    prefiltered: List[Tuple[int, Verdict]] = []
    to_judge: List[RecordTree] = []
    to_judge_idx: List[int] = []
    for i, (record, tree) in enumerate(records_trees):
        j = j1_j2_j3(record, tree)
        if not j["j3"]:
            prefiltered.append((i, Verdict(verdict="bad",
                                            problems=j["reasons"] or ["J3 fail (phantom nodes)"],
                                            confidence=1.0)))
        else:
            to_judge_idx.append(i)
            to_judge.append((record, tree))

    judged = judge.judge_many(to_judge, batch=batch) if to_judge else []

    out: List[Optional[Verdict]] = [None] * len(records_trees)
    for i, v in prefiltered:
        out[i] = v
    for idx, v in zip(to_judge_idx, judged):
        out[idx] = v
    assert all(v is not None for v in out)
    return out  # type: ignore[return-value]
