# LLM parse judge

Status: NEW, 2026-09-08 (branch `llm-parse-judge`). Low priority per the
lead, but cheap -- built to run at volume via the Anthropic Batches API.

## Why

Once the encoder self-trains, its own good parses on edge cases become gold
for the next round. The admission filter into that loop must NOT be the
model's own confidence: `scripts/probe_encoder_complement.py`'s measurement
found the encoder's best-of-8 tree scores edge-F1 0.70 but its own rank-1
(most-confident) tree only 0.53 -- confidence is uncalibrated. Instead, this
module flattens each candidate tree to text and has a cheap LLM (Claude
Haiku 4.5) judge it `good`/`bad`.

`runs/complement_probe.txt`'s own top-5 "best-looking" trees (by its POS-only
J1/predicate-ok check) show exactly why a purely structural/POS judge is too
lenient: `may` accepted as a clause's PREDICATE (it's AUX, POS-legal, but
it's the modal, not the main verb) and `was` accepted as a `PLACE` role. A
POS category match is not the same thing as being *semantically* the right
predicate or the right argument -- that's what the LLM judge is for.

## Files

- `src/nsm_ct/tree_render.py` -- `render_tree(record, tree) -> str`: the
  compact, LLM-friendly flattening (sentence + one `[kind] PREDICATE=... |
  ROLE=...` line per clause, ~600 chars max, WordNet glosses inlined).
  `normalize_gold_tree(record, tree)` converts a gold-lattice-shaped tree
  (`runs/*.jsonl`'s `lattice.trees[i]`) to the same normalized clause shape
  `encoder_model.beam_decode` emits, which is all `render_tree` ever
  consumes. `scripts/probe_encoder_complement.py`'s own diagnostic renderer
  (`render_node`/`gold_top1_tree`) now calls into this module instead of
  duplicating the logic.
- `src/nsm_ct/parse_judge.py` -- the `Judge` interface, `MockJudge`
  (deterministic, offline), `HaikuJudge` (the real judge), the shared
  `j1_j2_j3` structural checks, and `judge_many_with_prefilter` (applies the
  hard J3 pre-filter before calling the backend at all).
- `scripts/judge_parses.py` -- judges a whole gold or encoder-decoded
  `.jsonl` file, writes back a `judge` field per record, optionally filters
  to `good` only (the self-training admission gate).
- `scripts/calibrate_judge.py` -- the calibration harness: gold vs.
  corrupted vs. encoder-complement good-rates, mock always / haiku if a key
  is set.
- `tests/test_parse_judge.py`.

## The rubric

The judge is asked whether the flattened tree is a **faithful
predicate-argument reading** of the sentence:

- PREDICATE must be the actual main verb, not an auxiliary/modal standing in
  for it (a bare `may`/`was`/`is`/`has`/`will`/`do` as predicate is wrong
  unless it truly is the main verb -- e.g. copular `is` in "he is a thief").
- SUBJECT/OBJECT/other core roles must be the sentence's real arguments of
  that predicate -- not unrelated, not swapped with each other.
- No invented, duplicated, or misattached arguments.
- Sense glosses should be plausible for how the word is used in *this*
  sentence.
- Clause split should match the sentence's actual clause structure.

**No partial credit**: a tree right on the predicate but wrong on any core
role (or vice versa) is `bad`. The full rubric text is `parse_judge.RUBRIC`.

## The hard pre-filter

A tree that fails J3 (phantom nodes -- an out-of-range or duplicate
`token_index`, or more emitted nodes than the sentence has content tokens)
is marked `bad` **without ever calling the judge backend** --
`judge_many_with_prefilter` is the entry point that applies this; both
`judge_parses.py` and `calibrate_judge.py` go through it, never call
`judge.judge`/`judge_many` directly.

## Backends

- **`MockJudge`** -- J1/J2/J3 (predicate-ok / subject-ok / no-phantoms,
  adapted from `probe_encoder_complement.judge_tree`) plus first-sense/POS
  agreement (informational -- contributes to `confidence`, does not by
  itself flip a structurally-sound tree to `bad`; it is too noisy at small N
  on real prose to gate on). Deterministic, no network, no key. It is
  **not** a substitute for `HaikuJudge`: it cannot see e.g. a SUBJECT/OBJECT
  swap between two equally-nominal tokens (no phantom, no POS violation) --
  see the calibration numbers below, this is exactly what a purely
  structural judge misses and an LLM judge should catch.
- **`HaikuJudge`** -- `claude-haiku-4-5` via the official `anthropic` SDK.
  - Single record: `client.messages.parse(..., output_format=Verdict)` ->
    `response.parsed_output`. Retries on `RateLimitError` /
    `APIStatusError >= 500` with exponential backoff; raises immediately on
    a 4xx (a prompt/schema bug, not transient).
  - Volume (`judge_many(..., batch=True)`): the Message Batches API
    (`client.messages.batches.create/retrieve/results`), keyed by
    `custom_id` (results arrive in any order). Tries `output_config`
    JSON-schema structured output first; on a `BadRequestError` falls back
    to a JSON-only system-prompt instruction and parses `response.text` as
    JSON.

## Running it

```bash
# offline sanity check (no API key needed)
python scripts/calibrate_judge.py --backend mock

# judge encoder_gold_v2's top-1 gold trees with the real judge, single-record
ANTHROPIC_API_KEY=... python scripts/judge_parses.py \
    --in runs/encoder_gold_v2.jsonl --out runs/judged_gold.jsonl \
    --tree-source gold --backend haiku

# admission gate for self-training: encoder-decoded trees, batch mode,
# only the ones the judge marks good
ANTHROPIC_API_KEY=... python scripts/judge_parses.py \
    --in runs/some_sentences.jsonl --out runs/admitted.jsonl \
    --tree-source encoder --checkpoint runs/encoder_colab.pt \
    --backend haiku --batch --filter good

# calibration with the real judge too (only runs if a key is set; otherwise
# prints this exact command and skips, per dev/PARSE_JUDGE.md's own policy
# of never asking for a key)
ANTHROPIC_API_KEY=... python scripts/calibrate_judge.py --backend haiku
```

## Cost per 1,000 judged trees

Measured `HaikuJudge` usage is printed by `judge_parses.py` after every run
(`total_requests`/`total_input_tokens`/`total_output_tokens` and the derived
dollar estimate). Absent a live run in this environment (no
`ANTHROPIC_API_KEY` set here), the estimate below is derived from the
rendering length actually produced on 100 real gold records plus the fixed
rubric/question text (`dev/PARSE_JUDGE.md`'s own numbers, not a live
measurement):

- Rubric + question: ~1,600 chars (~400 tokens, fixed per request --
  a strong prompt-caching candidate if judging moves to very high volume).
- Average rendered tree: ~365 chars (~90 tokens) over 100 gold records.
- Estimated ~490 input tokens / ~60 output tokens per judged tree.

At Haiku 4.5 standard pricing ($1/M input, $5/M output):

- **~$0.79 per 1,000 judged trees** (standard `messages.create`/`.parse`).
- **~$0.40 per 1,000 judged trees** (Batches API, 50% off).

## Calibration results (`--backend mock`, this branch)

Three sets built from `runs/encoder_gold_v2.jsonl` (+ the 77 encoder
complement trees reconstructed from `runs/complement_probe.txt`, best-effort
-- see `calibrate_judge.parse_complement_probe_txt`'s docstring for the
reconstruction's known lossiness; validated against that file's own
recorded J1/J2/J3 verdicts at 76/77 exact match):

| set | n | good | good-rate |
|---|---|---|---|
| (i) gold top-1 trees | 50 | 42 | 84.0% |
| (ii) corrupted gold trees | 50 | 8 | 16.0% (= 84.0% corrupted-detection) |
| (iii) encoder complement rank-1 trees | 77 | 13 | 16.9% |

Corrupted set = one of {SUBJECT/OBJECT token swap, predicate replaced by an
adjacent non-verb token, phantom role attached} per record. The SUBJECT/OBJECT
swap is corruption MockJudge is **not** expected to reliably catch (no
phantom node, no POS violation) -- `tests/test_parse_judge.py`'s
100%-detection assertion runs only against the two structurally-detectable
corruption kinds for that reason; `calibrate_judge.py`'s report above
includes all three, which is why its detection rate is 84% rather than
100% -- that gap is the reason this module needs `HaikuJudge`, not just
`MockJudge`, as the real admission gate.

Run `ANTHROPIC_API_KEY=... python scripts/calibrate_judge.py --backend
haiku` to get the same table from the real judge.
