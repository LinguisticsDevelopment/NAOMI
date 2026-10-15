# Clause Extraction Fix (B1 + C1 + B2) — branch `clause-extraction-fix`

Implements the three targeted fixes `dev/GOLD_COVERAGE_AUDIT.md` ("Part 3 --
Root cause" / "Proposed fix") specified but deliberately did not implement.
Extraction-only: no change to the parser grammar, USVS, or the encoder. The
clause/role schema is unchanged; only more clauses come out of
`extract_discourse`.

## What changed

All changes are in `src/nsm_ct/clause.py`.

- **B1** (`_secondary_fact_clauses`, `clause.py:604-665`): a bare intransitive
  secondary/coordinated clause (no PP, no OBJECT/INDIRECT_OBJECT) had no
  `val_idx`, and the `if val_idx is not None:` guard had no `else` -- the
  clause was silently skipped. Added the `else` branch (`clause.py:647-658`),
  copying `_primary_discourse`'s own bare-subject fallback
  (`clause.py:883-888`, née `804-808` pre-fix): `Clause(predicate=pred,
  args=[SUBJECT]+_extra_args(...))`.
- **C1** (`_primary_discourse`'s coordinated-VALUE branch, `clause.py:820-833`):
  VP-coordination ("the lion shook his mane and showed his teeth") attaches
  the shared SUBJECT edge to the COORDINATOR node itself, so `pred` (==
  `graph.token(clause_idx)`) resolves to the coordinator's own surface token
  ("and"/"or"/"but"), not a real predicate. Added a structural guard —
  `if clause_idx in by_coord:` — detected BEFORE the existing "coordinated
  VALUE" logic runs (that branch is for a different shape: one real shared
  predicate, several coordinated PP/value objects). When it fires, each
  coordinated element becomes its OWN clause with its OWN predicate
  (`graph.token(el)`), sharing the one real `subj`. Structural, not lexical —
  it can never misfire on a genuine coordinated-value clause, whose predicate
  node is never also a coordinator child.
- **B2** (`_recover_reporting_matrix_clauses`, new function, `clause.py:719-750`,
  wired into `extract_discourse` at the `clauses = clauses +
  _recover_reporting_matrix_clauses(graph)` call): quote-attribution word
  order ("said little jack", "replied peggy", "thought buster") postposes the
  subject after the verb; this grammar has no subject-inversion rule for it,
  so the postposed subject attaches as the verb's OBJECT/INDIRECT_OBJECT
  instead and the verb node gets NO SUBJECT edge at all. Invisible to both
  `_primary_discourse` (needs a SUBJECT edge to resolve `pred`) and
  `_secondary_fact_clauses` (walks existing SUBJECT edges only). The new
  function scans PREDICATE/CLAUSE nodes whose token is in the new
  `REPORTING_VERBS` closed list (`clause.py:151`, `ATTRIBUTION_VERBS` plus
  `thought`/`promised`/`perceived`/`wondered`, per the audit's own proposed
  list) AND that have no SUBJECT edge; the rightmost OBJECT/INDIRECT_OBJECT
  child becomes SUBJECT (the head noun of the postposed subject NP; a
  misattached premodifier, when the parser also splits one off, sorts
  before it and survives as an extra role rather than being lost — see
  "said little jack" in the tests below). Gated on "no SUBJECT edge", so it
  can never duplicate a clause `_primary_discourse`/`_secondary_fact_clauses`
  already built.

All three fixes were designed against the actual parser output (`HypGraph`
edges), not just the audit's prose description — a small debug harness
(`parser._parse_graph(sentence)` + dump nodes/edges) was used to confirm each
bug's exact structural signature on real holdout sentences before writing the
fix, and to confirm each fix's gating condition cannot fire on an unrelated,
correctly-handled shape (verified: "mary is in the kitchen or the office",
"mary and john are in the garden", "mary gave the ball to john in the
garden" all parse identically before/after).

## Known remaining gaps (not in scope for B1/C1/B2)

- A handful of reporting/narration verbs ("shouted", "cried", "thought" in
  one case) appear as fully **orphaned graph nodes** — no SUBJECT edge AND no
  OBJECT/INDIRECT_OBJECT edge AND no edge of any kind connecting them to
  anything (e.g. holdout #9's "the fish thought", #10's "shouted", #38's
  "cries"). This is a parser/grammar attachment failure, not an extraction
  bug — there is no edge in the graph for B1/B2 to walk. Out of scope here
  (would need a parser-grammar fix, which the task brief explicitly
  excludes).
- 3+-way VP-coordination with a degenerate re-parse (holdout #46: "yawned,
  and slowly got to his feet and shook himself" — the 3rd coordinate's
  structure is scrambled enough that `clause_idx in by_coord` never becomes
  true) is not fixed; C1's structural guard only fires on the exact 2-level
  shared-subject-on-coordinator shape it was designed for.
- The infinitival/trailing complement of a VP-coordination's 2nd+ element can
  still be lost when the parser attaches it to the coordinator node rather
  than to the element itself (holdout #41: "began to run" recovers predicate
  `began` with the right SUBJECT, but "run" itself doesn't surface) — a
  separate, pre-existing PP-attachment quirk, not part of C1's described fix.
- The large "other / subordinate embedding" bucket from the audit (relative
  clauses, "which"/"that" complements, multiply-nested subordination — ~15
  of the original 40 cases) is explicitly out of scope per the audit
  ("architectural gap, not a single bug").

## Tests

New file `tests/test_clause_extraction_fix.py`, 7 tests, 2+ per bug, each
using a real sentence quoted verbatim from `runs/holdout_sentences.txt`
(the hand-audited holdout set):

| test | bug | holdout # | asserts |
|---|---|---:|---|
| `test_b1_bare_intransitive_coordinated_clause_recovered` | B1 | #10 | `laughed(SUBJECT=he)` and a `ran(SUBJECT=boy)` clause now present |
| `test_b1_bare_intransitive_matrix_clause_recovered` | B1 | #48 | `thought(SUBJECT=she)` now present |
| `test_c1_vp_coordination_shares_subject_own_predicates` | C1 | #6 | no `and`-predicate clause; `shook`/`showed` each own clause, SUBJECT=lion, correct OBJECTs |
| `test_c1_vp_coordination_fox_looked_and_began` | C1 | #41 | no `and`-predicate clause; `looked`/`began` each own clause, SUBJECT=fox |
| `test_b2_reported_speech_postposed_subject_replied` | B2 | #5 | `replied(SUBJECT=peggy)` now present |
| `test_b2_reported_speech_postposed_subject_said` | B2 | #16 | `said(SUBJECT=jack)` now present |
| `test_b2_reported_speech_postposed_subject_thought` | B2 | #33 | `thought(SUBJECT=buster)` now present |

All 7 pass.

### Full-suite run

The task brief asked for `pytest -x -q tests` (cap 10 min). Empirically, in
this environment the full 1302-test suite does not complete anywhere near 10
minutes — a timed run reached only ~22% of tests after 26 minutes (some
fixtures build a fresh parser/USVS stack per test file), so running it to
completion was infeasible within this task's budget. Instead, every test
file that imports `nsm_ct.clause`, `extract_discourse`/`extract_clauses`, or
any of the helper functions touched here (40 files, ~583 tests incl. the 7
new ones) was run to completion:

- **Before this fix** (stashed, i.e. `origin/gold-coverage-audit`'s
  `clause.py`): same 4 failures below, confirmed identical failure messages.
- **After this fix**: `576 passed, 4 failed` in 1438s (~24 min).

The 4 failures are **pre-existing and unrelated to this change** — confirmed
by re-running the exact same 4 tests against the unmodified (`git stash`'d)
`clause.py` and getting byte-identical failures:

- `tests/test_quantum_adapter.py::test_two_pps_both_survive_extraction` —
  asserts `len(cl.args) == 4` for "mary gave the ball to john in the
  garden ."; the actual (both before and after this fix) clause already
  carries 6 args (2 extra `DESCRIPTION=the` roles from the pre-existing
  "Phase 1a richer-gold" `_extra_args` modifier-walking, M48/M50) — a stale
  test expectation from before that feature landed, not something B1/C1/B2
  touches.
- `tests/test_spanish_freeze.py::test_stream_equivalence_smoke` — a
  `relation_match / relation_total >= 0.95` threshold (gets 164/174 =
  0.943) on an EN/ES parse-equivalence smoke test; no English/Spanish
  pronoun or clause-shape logic in this fix.
- `tests/test_morphology.py::test_english_parsing_byte_identical_regression`
  — asserts a byte-identical golden shape for "mary went to the garden ."
  that's missing a `DESCRIPTION=the` role the current (pre-fix, too) code
  already emits; same stale-golden-snapshot cause as the first failure.
- `tests/test_garden_path.py::test_garden_path_v2_trait_templates_all_verified`
  — `verify_garden_path_trait_templates()` reports 4 templates not
  verified; unrelated to clause extraction (a template-verification helper
  in `curriculum2.py`).

No test was updated for this submission since none of the 4 failures were
caused by, or fixed by, B1/C1/B2 — they are called out here per the task
brief's instruction to investigate every pre-existing failure, not silently
pass over it.

## Before/after gold-coverage measurement

Per the task brief: built one gold sample per branch over (a) the 98
sentences in `runs/holdout_sentences.txt` and (b) a **seeded (seed 0) random
300** sentences from `data/corpus/real_gutenberg_*.txt` (**shrunk from 400**
to fit this task's time budget — same seed/pool both runs, so it's the same
396 combined sentences on both sides), via a small wrapper
(`build_sample_gold.py`, not committed — a throwaway driver over
`scripts/build_encoder_gold_v2.py`'s own `load_corpus`/`_new_parser`/
`process_sentences`, forest mode `top1`) rather than modifying the real
`GOLD_CORPUS_GLOB`-driven overnight pipeline. Before = `git stash`'d
`clause.py` (byte-identical to `origin/gold-coverage-audit`); after = this
branch. Both builds used the identical sentence list (deterministic
`random.Random(0).sample`). `scripts/audit_gold_coverage.py` was then run on
each resulting JSONL.

| metric | before | after | Δ |
|---|---:|---:|---:|
| records built (of 396 attempted) | 278 | 288 | +10 |
| outcomes: ok / cap-hit / no-hypothesis / grounding-fail | 278 / 9 / 81 / 28 | 288 / 9 / 81 / 18 | grounding-fail -10 |
| mean VERB coverage | 54.3% | **67.9%** | **+13.6 pts** |
| % records <50% VERB-covered | 36.8% | **20.3%** | **-16.5 pts** |
| mean NOUN/PROPN coverage | 56.9% | 63.3% | +6.4 pts |
| mean clauses-kept / verb-tokens ratio | 0.708 | 0.918 | +0.210 |
| multi-verb records collapsed to 1 clause | 30.6% (66/216) | **16.4%** (36/220) | **-14.2 pts** |

The `grounding-fail` outcome (every one of the top-k hypotheses yields zero
clauses, so the whole sentence is dropped) falls from 28 to 18 — some
previously fully-empty records now get at least one clause purely from these
3 fixes, which is also why `ok` record count rose (278 -> 288) even though
the sentence list is identical. This lines up with the audit's own rough
extrapolation ("mean VERB coverage ... up by very roughly 15-20 points ...
into the high-60s/low-70s%" — landed at 67.9%, within that predicted range)
and with the gate math: still well above a 30% multi-verb-collapse
threshold, but roughly halving the gap from the pre-fix 30.6%.

By length bin, the gain concentrates in the shorter/mid sentences (bin A:
78.9% -> 89.6%, bin B: 59.1% -> 79.0%, bin C: 61.4% -> 71.1%) with bin D
(26+ tokens) improving less (43.4% -> 55.9%) — consistent with the
"other/subordinate embedding" bucket (which scales with sentence length and
clause depth) being the dominant remaining gap, exactly as the audit
predicted.

## Re-judgment of the 33 original DROPPED_CLAUSE holdout cases

Re-ran the same 98 holdout sentences through both builds and compared the
kept tree's clause-predicate list for each of the 33 sentences the original
hand audit marked `DROPPED_CLAUSE` (`dev/GOLD_COVERAGE_AUDIT.md` Part 2).
"FULL" here means the specific clause/predicate the original audit quoted as
missing is now present (not a re-run of the full 6-way hand-audit taxonomy,
which would need re-reading all 33 trees against the source text from
scratch — the task's own re-judgment ask is narrower: did the NAMED missing
clause come back).

**14 now FULL** (the quoted missing clause's predicate is now present):
#4 (`perceived`), #5 (`replied`), #11 (`replied`), #13 (`asked`), #16
(`said`), #17 (both `is` clauses of the 2-of-3 parallel statements), #30
(`seen`), #31 (`seemed`), #33 (`thought`), #34 (`was`, "there it was"), #36
(`'m`, "'m here"), #37 (`wanted`), #42 (`expected`), #48 (`thought`).

**4 now PARTIAL** (some but not all of the quoted content recovered):
#10 ("he just laughed" -> `laughed(SUBJECT=he)` recovered, and the 2nd `ran`
clause recovered with the right SUBJECT=boy; "shouted" itself remains an
orphaned graph node -- see "Known remaining gaps"); #25 (`have` appears,
a step toward "it might have been different" but not a clean match); #27
(both `and`-predicate WRONG_PREDs are gone -- `ran`/`got`/`got` all now real
predicates -- but the audit's named 4th coordinate, "hid the drum...", is a
3+-way coordination still not recovered); #28 (`had` appears, a step toward
"it had been made" but not a clean match).

**15 still DROPPED** (no change versus the original audit): #1, #3, #8, #9,
#14, #18, #21, #23, #24, #26, #35, #38, #44, #47, #49 -- of these, #18, #35,
#47, #49 failed to parse at all within this harness's resource cap on BOTH
branches (long, heavily-subordinated sentences -- `ParseResourceExceeded`,
unrelated to clause.py), #9 and #38 are the orphaned-reporting-verb-node gap
noted above (same mechanism as #10's "shouted"), and the rest (#1, #3, #8,
#14, #21, #23, #24, #26, #44) are subordinate/relative-clause embedding or
other constructions the audit's own "other" bucket already scoped as
out-of-scope for this fix.

**0 newly wrong**: no case's predicate list got WORSE or introduced a new
incorrect predicate relative to the original audit's "before" state (spot-
checked against the before-build's own predicate lists, not just the
original `encoder_gold_v4b_16k.jsonl` numbers, since the before-build here is
a fresh parse, not a byte-for-byte replay of that file).

14 FULL + 4 PARTIAL = 18/33 (55%) of the originally-dropped clauses are now
at least partially recovered, concentrated almost entirely in B2 (reporting/
cognition matrix clauses: 4,5,11,13,16,33,48 = 7 of the 14 FULLs) and B1
(bare intransitives recovering as a side effect in subordinate/relative
positions: 17,30,31,34,36,37,42 = 7 more). C1's clean wins (#6, #41 in the
dedicated tests above) didn't happen to land inside this specific 33-sentence
subset's *named* missing-clause text, but are directly verified by the two
C1 regression tests.

## What remains

Per the audit's own scoping: unwalked subordinate/relative-clause embedding
(the ~15-case "other" bucket) is the dominant remaining gap, especially for
longer sentences (bin D). Three narrower residual gaps surfaced during this
fix (see "Known remaining gaps" above): fully-orphaned reporting-verb graph
nodes, 3+-way VP-coordination, and a VP-coordination element's own
trailing PP/infinitive attaching to the coordinator instead of the element.
None of these were in the B1/C1/B2 scope; fixing them would need either
parser-grammar changes (out of scope per the task brief) or a 4th extraction
pass.
