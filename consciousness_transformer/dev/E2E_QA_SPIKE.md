# E2E QA SPIKE (thesis test, teacher-built episodes) -- branch `e2e-qa-spike`

dev/AUDIT_2026-10-15.md section F item 2: "does memory-routed QA work on
real prose?" Teacher-built episodes (the deterministic parser +
`nsm_ct.clause.extract_discourse` path, NOT the learned encoder) over
FairytaleQA entity-MC items and MCTest mc160, through the existing
resolver/QA head (`scripts/train_prose.py`'s model), each dataset's OWN
story split (FairytaleQA train/val/test; MCTest train/dev), controls on
the same test items: random, the `scripts/cheat_baselines_k12.py` cheats,
the trained model, and a memory-severed ablation.

**Headline: both gates FAIL at the scale this spike could afford in its
time budget (see "Scale and honesty" below) -- inconclusive, not a
falsification.** The pipeline itself (parse -> write memory -> MC-score
via the response head) runs end to end on real prose for the first time,
which audit #2 flagged as completely untested. What it could not do in
budget is give the resolver enough training data to show whether it CAN
learn to read memory at all.

## 0. What had to be built first: the minimal adapter

`nsm_ct.clause_reactor.build_clause_batch`'s `kind == "prose"` branch
(added for `nsm_ct.corpus.make_episodes`'s own self-generated "where is
the X ?"-style questions) resolved the question's address via
`_question_entity` (a NAME from a fixed curriculum nameset, or the noun
after a literal "the") and `_queried_role`, and **silently dropped the
whole episode** (`continue`) whenever `_question_entity` returned `None`.
A real benchmark question ("Who was skilled in magic more than most
fishes?", "What city...") essentially never matches that shape -- so
every FairytaleQA/MCTest MC item would have scored ZERO rows, not a bad
row, confirming the audit's "never run" finding for a structural reason,
not just a scheduling one.

Fix (`src/nsm_ct/clause_reactor.py`, ~20 lines, additive): when `qent is
None`, append a GENERIC question step instead of dropping the episode --
a fixed, never-written `(var:QUERY, rel:ANSWER)` marker address. `mem_read`
at that one step stays ~0, but the GRU's recurrent `state` already
absorbed every passage write step before it, and the response head's
contrastive `cosine(response, options)` scoring (the SAME answer path
every other episode kind uses) runs unchanged. This is the literal
"score each option against memory, pick argmax" adapter the task asked
for if the pipeline couldn't consume MC options out of the box -- no new
model code, no new answer path, one new question-step shape. The old,
templated branch is untouched (regression-tested below) -- byte-identical
for every `corpus.py`-built episode.

Tests: `tests/test_e2e_qa_spike_adapter.py` (2 new tests) -- a free-form
question with no NAME/"the NOUN" now survives build_clause_batch and
scores a finite, correctly-shaped `answer_logits`; a templated question
still takes the OLD branch unchanged.

## 1. Data build: episodes, parse coverage, answerable-from-memory ceiling

Source: `runs/fairytaleqa_episodes_samepassage.jsonl` (entity-MC items,
`answer_type == "entity"` AND `options is not None`, same-passage
distractors per `k12-bench-fix`) and `runs/mctest_mc160_episodes.jsonl`
(every item -- all 400 already carry native in-story 4-way options).
Splits are each dataset's OWN (`meta["split"]`): FairytaleQA train/val/test
(439/63/82 entity-MC items, 62/7/10 stories); MCTest train/dev (280/120
items, 70/30 stories -- MCTest has no native val, so val was CARVED out of
TRAIN's own passages, never out of dev, so model SELECTION never touches
the reported test set -- see `carve_val_from_train` in
`scripts/e2e_qa_spike.py`).

"% of passage sentences producing >=1 clause" and "answerable-from-memory"
(gold answer token appears in the pooled predicate+argument tokens
`_prose_steps` actually writes for that passage) are measured through the
SAME `_prose_parse_graph` + `extract_discourse` call the model's own
memory-write path makes -- not a separate/different parse.

| dataset | split | items | passages | sentence-clause coverage | answerable-from-memory ceiling |
|---|---|---|---|---|---|
| FairytaleQA | train | 37 | 6 | 86.9% | 81.1% |
| FairytaleQA | val   | 30 | 3 | 93.7% | 90.0% |
| FairytaleQA | test  | 33 | 4 | 90.4% | 78.8% |
| MCTest      | train | 24 | 6 | 94.5% | 79.2% |
| MCTest      | val (carved from train) | 12 | 3 | 95.5% | 91.7% |
| MCTest      | test (= dev) | 16 | 4 | 94.0% | 93.8% |

Parse coverage and the answerable ceiling are both healthy (79-94%, no
sign of a parser/extraction bottleneck at this sample -- consistent with
`clause-extraction-fix`'s own measured gains). Build failures (an episode
dropped because one of its sentences hit the 10s parse-resource cap):
FairytaleQA train 2/37, val 5/30, test 8/33; MCTest 0/24, 0/12, 0/16.

## 2. Scale and honesty: why these numbers are tiny

Each sentence parse costs anywhere from well under 1s to the full 10s
`CORPUS_MAX_PARSE_SECONDS` resource cap, and **FairytaleQA items on the
same story do not all share identical context** (different questions
pull different section slices) -- so the real unit of parse cost is
"distinct context variant," not "distinct story," and that count is
large: an early full-scale timing probe (78 FairytaleQA stories, 619
sentences, deduped by STORY not variant) alone took 468s; the corrected,
variant-correct accounting on just 6+3+4=13 passages of FairytaleQA
training data took 99.4+176.0+164.5 = 440s. Extrapolated, the FULL
FairytaleQA entity set (584 items, ~79 stories, likely 300+ distinct
context variants) plus the full MCTest set (400 items, 100 stories) would
need on the order of HOURS of parse-build time alone -- far past this
spike's ≤30-min-per-run / ~90-min-total budget.

Per CLAUDE.md's ops rule ("shrink steps/data if a pilot projects more, and
say so"), this spike therefore ran on a SEEDED, passage-capped subsample
(6 train / 3 val / 4 test passages per dataset -- see
`scripts/e2e_qa_spike.py --*-passage-cap`), not the full corpus. **This is
a pipeline-integration validation at this scale, not a power-adequate test
of the comprehension thesis.** The infra to run it at full scale now
exists (the adapter, the negative-parse-cache, the context-tuple-keyed
coverage pass, the known-bad-sentence batch-build prefilter -- see
`scripts/e2e_qa_spike.py`'s module docstring) and the honest next step is
a dedicated multi-hour background job, not a re-read of these numbers as
the thesis verdict.

## 3. Controls + model, test split (teacher-built episodes, deterministic parser)

| metric | FairytaleQA (n=25 scored) | MCTest (n=16 scored) |
|---|---|---|
| random | 0.250 | 0.250 |
| majority (modal answer text) | 0.040 | 0.063 |
| passage-substring cheat | 0.250 | 0.302 |
| word-overlap cheat | 0.400 | 0.531 |
| **best cheat** | **0.400** | **0.531** |
| **model (trained resolver/QA head)** | **0.400** | **0.250** |
| **memory-severed (same model, passage writes masked out)** | **0.400** | **0.188** |
| accuracy on answerable-from-memory subset | 0.421 (19/25) | 0.200 (15/16) |
| best val/dev accuracy (keep-best) | 0.480 | 0.333 |

Gate (dev/AUDIT_2026-10-15.md): PASS iff `model >= best_cheat + 0.10` AND
`|memory_severed - random| <= 0.03`.

- **FairytaleQA: FAIL.** `model (0.400) < best_cheat + 0.10 (0.500)`. Also
  `|severed (0.400) - random (0.250)| = 0.150 > 0.03`.
- **MCTest: FAIL.** `model (0.250) < best_cheat + 0.10 (0.631)`. Also
  `|severed (0.188) - random (0.250)| = 0.063 > 0.03`.

## 4. Diagnosis (which link broke)

Checked in the order the audit's own gate implies:

1. **Parse coverage: not the bottleneck.** 86.9-95.5% of sentences
   produced >=1 clause in every split measured (section 1).
2. **Answer not in memory: not the bottleneck.** 78.8-93.8%
   answerable-from-memory ceiling (section 1) -- most gold answers DO have
   at least one token the memory-write path actually captures.
3. **Resolver not using memory: the FairytaleQA number's own strongest
   signal.** Model accuracy and memory-severed accuracy are IDENTICAL
   (0.400 == 0.400) on every one of the 25 scored test items -- masking
   out every passage write changed NOT ONE prediction. That is the
   textbook signature of a response head that learned to answer from
   option identity/structure alone, never from the memory read. (MCTest's
   two numbers differ, 0.250 vs 0.188, but both sit at-or-below random on
   n=16 -- too small to call either a real memory-reading signal or a real
   severed-ablation drop.)
4. **Head capacity / data starvation: the more likely ROOT cause for
   BOTH, given section 2.** 35-37 train items (FairytaleQA) and 24 train
   items (MCTest) is far below what a GRU resolver needs to learn a
   genuinely content-sensitive read policy over free-form questions it
   has never had to answer before (every prior prose arm used `corpus.py`'s
   own templated "where/who/what" questions) -- 40 epochs over data this
   small is squarely in the overfit-to-a-fixed-bias regime, which is
   consistent with (3)'s exact severed/unsevered tie: the head converged
   to a shortcut that doesn't need memory at all BECAUSE there wasn't
   enough signal to teach it one that does.

**Conclusion: this spike could not separate "the architecture can't do
real-prose QA" from "35 training items can't teach any architecture
real-prose QA."** The pipeline-integration half of the thesis test (can a
real benchmark even be built and scored end to end through the
deterministic parser + existing resolver, with a working memory-severed
control) is answered YES. The comprehension half (can the TRAINED
resolver read memory for free-form questions) needs the full-scale run
section 2 costs out, not a re-run of this sample size.

## 5. Tests

`tests/test_e2e_qa_spike_adapter.py` (new, 2 tests): PASS -- the minimal
adapter keeps a free-form-question episode instead of dropping it, and
leaves the old templated-question branch unchanged.

`pytest -x -q tests -k "prose or episode or corpus or clause"`: 51 tests
passed, 0 failures, before the 10-minute cap cut the run off mid-suite
(no summary line reached) -- this matches the PRE-EXISTING "full suite
does not complete near the stated 10min cap" issue already on record from
`clause-extraction-fix`/`k12-bench-fix` (quantum_parser slowness, not a
regression from this branch's own changes). No failures were observed in
what ran.

## 6. Wall-clock per run (this environment, CPU, single-threaded)

| run | coverage pass | batch build | training (40 epochs) | total |
|---|---|---|---|---|
| FairytaleQA | 440.0s (train 99.4 + val 176.0 + test 164.5) | 14.3s | 17.55 min | 25.14 min |
| MCTest | 34.1s (train 17.9 + val 8.6 + test 7.6) | 12.0s | 8.09 min | 8.88 min |
| focused pytest subset | -- | -- | -- | >10 min (cap, cut off) |

(An early, since-corrected coverage pass that deduped by STORY instead of
by exact context-sentence tuple undercounted true sentence coverage and
caused a slow fallback batch-build path -- see `scripts/e2e_qa_spike.py`'s
`_ctx_key` docstring; the numbers above are from the corrected run.)

## 7. What's committed

- `src/nsm_ct/clause_reactor.py`: the minimal MC-question adapter
  (section 0).
- `scripts/e2e_qa_spike.py`: the full driver (native-split loading,
  passage-capped subsampling, coverage/answerable-ceiling measurement,
  model training with keep-best, the memory-severed control, cheat
  baselines, gate verdict, JSON report) -- reuses `train_prose.py`/
  `eval_prose.py`/`cheat_baselines_k12.py` throughout rather than
  duplicating any of their logic.
- `tests/test_e2e_qa_spike_adapter.py`: the adapter regression (section 0).
- `runs/e2e_qa_spike_ftqa.{json,pt}`, `runs/e2e_qa_spike_mctest.{json,pt}`:
  the reports and trained (keep-best) checkpoints from the runs this
  document reports (small -- well under 30MB).
