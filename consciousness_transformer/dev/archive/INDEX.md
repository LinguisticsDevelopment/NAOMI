# Archive index

Docs moved here because their content is superseded and nothing in code or
tests references them by path (checked by grepping the whole repo before
each move; see `git log --follow` on any file here for its original home
and history). See `MILESTONE_MAP.md` in this directory for the M-number ->
goal -> outcome table, and `GOALS.md` at the repo root for current status.

- **ENCODER_DATA_STRATEGY.md** — 2026-09-06 recommendation on how to grow
  encoder training data without burning compute. Superseded by the scaling-curve
  work that actually ran (`ENCODER_IO_CONTRACT_V2.md`, the arms results in the
  ledger, and the v4b/v5/v6 gold lineage).
- **ENCODER_GOLD_V3_SMALL_STATS.md** — auto-generated candidate-lattice stats
  for the v2/v3-small gold build (959 records, no lemmatization). Superseded by
  later, lemmatized gold builds (v4b).
- **ENCODER_GOLD_V4B_ALL_STATS.md** — stats for an intermediate v4b gold variant
  (986 records). Superseded by `ENCODER_GOLD_V4B_16K_STATS.md`, which stats the
  gold the current (frozen) encoder checkpoint was actually trained on.
- **ENCODER_GOLD_V4B_MARGIN_STATS.md** — stats for the v4b "margin" gold variant
  used in a single seed-pair check. Superseded the same way.
- **ENCODER_GOLD_V4_SMALL_STATS.md** — stats for the pre-POS-aware v4-small gold
  build. Superseded by v4b (POS-aware lemmatization fixed a bug this build had).
- **ENCODER_TRAIN_FULL_STATS.md** — 2026-09-04 stats for the very first full
  encoder train (985-record gold, pre-aliasing-fix checkpoint). Long superseded
  by the v4b/arms-4a checkpoint that is current.
- **EXECUTOR_DESIGN_REVIEW.md** — 2026-08-30 adversarial review of
  `EXECUTOR_DESIGN.md`. Its own header says every finding was accepted and
  folded into `EXECUTOR_DESIGN.md` the same day; kept here only for the
  paper trail.

## Not moved, though their content is also dated

A number of other `dev/` docs are just as clearly superseded (the 2026-09-08
audit, the Aurora sprint plan, the original resolver build plan, the hand-gold
draft, the pre-fix K-12/prose/semantic-mapping reports, the Spanish feasibility
probe, and the first "universal encoder" design draft, among others) but each
is still cited by a filename in at least one source file or test as design
rationale (a docstring or comment pointing back to it, e.g.
`corpus.py:1` citing `dev/AURORA_SPRINT.md`, or `convert_corpus.py`'s
`--taxonomy-out` default pointing at `dev/PROSE_FAILURE_TAXONOMY.md`). Per the
cleanup's own rule (grep before moving; leave in place and list it if code or
tests reference it), those stay in `dev/` rather than here. See the docs-cleanup
task's final report for the full list.
