# K-12 benchmark fix + cheat baselines (2026-10-15, branch `k12-bench-fix`)

Scope: dev/reports/AUDIT_2026-10-15.md sections C ("the planned K-12 benchmark can
be solved by string matching") and A.1 (the encoder's headline edge-F1 has
no cheat baseline beside it), worked per that audit's "F. Next two weeks,
item 1 (fix the benchmarks, ≈2h, no torch)". CLAUDE.md's "cheat-baseline
law": *every capability curriculum ships with its cheat-baseline at
floor* -- this file is that floor for the K-12 MC benchmark and the
encoder's rank-1 edge-F1.

## 1. What was broken

`scripts/convert_fairytaleqa.py::attach_entity_options` built MC
distractors for `entity`-type episodes by drawing other stories' gold
answers of the same normalized `attribute` (character/setting/action/...),
corpus-wide. The gold answer is, BY CONSTRUCTION, always a verbatim
substring of its own passage (`classify_answer_type`'s own definition of
"entity"); a cross-story distractor overwhelmingly is NOT. So "pick the
option that occurs verbatim in the passage" identifies the answer almost
every time -- not a comprehension signal at all. On the committed 584
entity episodes (`runs/fairytaleqa_episodes.jsonl`), that cheat scores
**97.3%** (553 unique-correct + 31 ties, 0 misses), confirming the audit's
hand count exactly.

Separately, `scripts/eval_prose.py` only ever reported random/majority
floors, and no encoder eval script printed any cheat baseline beside the
encoder's own rank-1 edge-F1 number.

## 2. The fix

### 2a. Distractor policy (`scripts/convert_fairytaleqa.py`)

`--distractors {same-passage,cross-story}` (default **same-passage**):

- **same-passage** (the fix): distractors are noun-phrase spans
  (`DET? (ADJ|NUM)* (NOUN|PROPN)+`, <=4 tokens, chunked via
  `quantum_parser`'s own POS tagger -- no new parser) drawn from the
  episode's OWN passage. Every option, gold and distractors alike, is now
  a verbatim passage substring, so passage-presence stops identifying the
  answer. Preference order: same-attribute spans first (`character` ->
  PROPN-headed; `setting` -> PLACE-like, i.e. preceded by a preposition or
  PROPN-headed), topped up with any other passage NP chunk when that pool
  runs short ("falling back to any passage NP"). An episode needs >=2
  distractors to get MC options at all -- same `options=None` contract as
  before.
- **cross-story** (the original, broken policy): kept, for comparison only.

`--regen-options-from PATH` re-applies either policy to an ALREADY-BUILT
episode jsonl (e.g. the committed `runs/fairytaleqa_episodes.jsonl`)
without re-fetching or re-parsing the source corpus -- `ep.context`/
`answer_text`/`meta` already carry everything either policy needs. This is
how `runs/fairytaleqa_episodes_samepassage.jsonl` was produced: from the
EXACT SAME underlying 3,000-episode / 584-entity-episode set as the
existing `runs/fairytaleqa_episodes.jsonl`, so the two policies are
compared on identical episodes, not a re-sampled corpus.

**Gate result: no iteration needed.** The same-passage policy passed the
gate (substring cheat <= random + 0.10) on the first run, with the SAME
584/584 entity episodes surviving (>=2 distractors) as the broken
cross-story policy -- same-passage NP material was already abundant enough
in these passages; no distractor-pool widening was required.

### 2b. `scripts/cheat_baselines_k12.py` (new)

Reports, for any Episode-schema jsonl: random (mean 1/n_options),
majority (modal gold-answer-text), passage-substring ("pick the option
that occurs verbatim in the passage"; ties -> random among tied, expected
credit `1/|tie|`; zero matches -> a miss, credit 0), word-overlap ("option
with max token overlap with the passage"; same tie-break). Only episodes
with `options is not None` are scored.

### 2c. MCTest conversion (`scripts/convert_mctest.py`, new)

mc160 train+dev (100 stories, 400 Q/A, from the already-committed
`data/k12_samples/mctest/` sample -- no network fetch needed) converted
into the SAME Episode schema, keeping MCTest's own in-story 4-way MC
options/gold letter and the `"one"`/`"multiple"` sentence tag
(`meta["sentence_tag"]`).

### 2d. Encoder cheat baseline (`scripts/cheat_encoder_baseline.py`, new)

The audit's ~30-line POS heuristic, scored through the SAME
`nsm_ct.encoder_model.score_record`/`aggregate_recall` code path the
learned encoder uses (no parallel metric):

- PREDICATE = first VERB, else first AUX.
- SUBJECT = nearest nominal (NOUN/PROPN/PRON) before the predicate; OBJECT
  = nearest nominal after it.
- DESCRIPTION = every ADJ/DET/ADV/NUM token.
- PLACE = the nearest nominal after an ADP, skipping intervening DET/ADJ/NUM
  (a bare `ADP -> immediate-next-token` check almost never fires on real
  prose, where a determiner sits between the preposition and its head noun).
- grounding-type = `"sense"` if the token has any sense candidates
  (`record["token_sense_candidates"]`), else `"entity"` for a PROPN, else
  `"reference"`.

POS tags come from `record["pos"]` (every gold record this project builds
already carries it, written by the project's own tagger at gold-build
time); nltk's tagger is a pure fallback for a record with no `pos` field.

`scripts/train_encoder.py`'s final report and `scripts/eval_encoder.py`
now print this heuristic's edge-P/R/F1 + struct-exact next to the random
baseline by default, on every future run -- not just on request.

## 3. Cheat-baseline tables

### 3a. FairytaleQA entity MC (584 scorable episodes, both from the SAME
underlying 3,000-episode set)

| cheat | cross-story (broken) | same-passage (fixed) |
|---|---|---|
| random | 0.250 | 0.250 |
| majority (`='happy'`) | 0.017 | 0.017 |
| **passage-substring** | **0.973** (553 unique + 31 tied, 0 miss) | **0.250** (0 unique + 584 tied, 0 miss) |
| word-overlap | 0.796 | 0.393 |
| gate (substring <= random+0.10) | **FAIL** (0.973 vs 0.350) | **PASS** (0.250 vs 0.350) |

Same-passage's substring cheat landing EXACTLY at the random floor is
expected, not a coincidence: once every option is a passage substring by
construction, "pick the one in the passage" degenerates to "pick one of
the tied options at random" for every episode (584/584 ties, 0 unique,
0 miss) -- i.e. plain random guessing. Word-overlap drops from 0.796 to
0.393 but stays above random; it is not gated by this task and is reported
as an honest residual signal (longer/more-central NPs still overlap more
words with the passage than short ones), not a benchmark defect to fix here.

### 3b. MCTest mc160 (train+dev, 400 episodes, native in-story distractors)

| cheat | value |
|---|---|
| random | 0.250 |
| majority (`='Jim'`) | 0.007 |
| passage-substring | 0.260 (59 unique + 163 tied + 159 miss) |
| word-overlap | 0.528 |
| gate (substring <= random+0.10) | **PASS** (0.260 vs 0.350) |

MCTest's human-authored in-story distractors were never the problem the
audit flagged -- confirmed here directly rather than assumed. Word-overlap
(0.528) is a real, substantial signal on this benchmark (plausible wrong
answers still share fewer passage words than the right one, on average) --
worth keeping in mind for any future "is the model actually reading"
comparison on MCTest, same caveat as 3a's word-overlap residual.

### 3c. Encoder heuristic cheat baseline (98-sentence v2 holdout, scored
against `encoder-gold-v4b-16k`'s gold records)

| | heuristic | encoder (rank-1, from dev/archive/AUDIT_2026-09-08.md) |
|---|---|---|
| edge_precision | 0.246 | -- |
| edge_recall | 0.295 | -- |
| **edge_F1** | **0.268** | **0.622** |
| struct-exact (structure_recall) | 0.000 | 0.214 |

Margin: encoder beats the heuristic by **+0.354** edge-F1.

## 4. Surviving-episode counts

- FairytaleQA: 584/584 (100%) of the entity episodes that had >=2
  cross-story distractors ALSO get >=2 same-passage distractors -- the fix
  cost zero MC-scorable episodes.
- MCTest mc160 train+dev: 400/400 episodes scorable (every MCTest question
  ships exactly 4 options natively).

## 5. Tests

`tests/test_distractor_policy.py` (4), `tests/test_cheat_baselines_k12.py`
(5), `tests/test_cheat_encoder_baseline.py` (5) -- 14 new tests, all
passing. A targeted run of these plus the existing
`test_fairytaleqa_accounting.py`/`test_eval_prose.py`/`test_encoder_model.py`/
`test_encoder_holdout.py` (47 tests total) gave 27 passed, 19 skipped
(quantum_parser-dependent fixtures this environment skips), 1 failed --
that one failure (`test_eval_prose.py::test_end_to_end_fresh_checkpoint_on_
five_prose_episodes`, an `argparse.Namespace` missing `answer_type`) is
**pre-existing and unrelated**: verified via `git diff` that neither
`scripts/eval_prose.py` nor its test were touched on this branch.

The full repo suite (1,315 collected tests) does **not** finish in the
10-minute cap `pytest -x -q tests` was run under -- it reached ~5% before
being killed, dominated by real `quantum_parser` parsing tests the repo's
own docstrings already flag as minutes-slow. This is a pre-existing
property of the suite's runtime, not something this branch changed.

## 6. Surprises worth flagging

- The audit's hand-estimated heuristic edge-F1 (0.40, explicitly
  "reproduced from `score_record`, not run against the code") came out
  noticeably higher than this script's actual measured 0.268 on the same
  98-sentence holdout, scored against `encoder-gold-v4b-16k` (10,225
  records, fetched fresh from that branch for this task). The gap is most
  likely gtype mismatches the hand estimate didn't account for (the
  heuristic's `sense`/`entity`/`reference` guess must match gold's gtype
  EXACTLY for an edge to count, and gold's actual gtype choices are not
  always what a bare POS-based guess would pick) -- this script's number
  is the one now wired into every future encoder report, superseding the
  hand estimate.
- The same-passage distractor fix needed no iteration at all: NP material
  was already abundant enough in FairytaleQA passages that the SAME 584
  entity episodes clear >=2 distractors under the new policy, with the
  gate landing exactly at the random floor on the first run.
