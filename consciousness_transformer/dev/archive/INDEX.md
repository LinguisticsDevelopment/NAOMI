# Archive index

Superseded docs, moved here (`git mv`, history preserved) because their
content has been overtaken by later work and nothing reads them at runtime.
See `MILESTONE_MAP.md` in this directory for the M-number -> goal -> outcome
table, and `GOALS.md` at the repo root for current status. Comment/docstring
references elsewhere in the repo were updated to the new path when a doc
moved here; nothing dangles (verified by grep).

- **AUDIT_2026-09-08.md** — the first independent audit. Superseded by
  `dev/reports/AUDIT_2026-10-15.md` (audit #2).
- **AURORA_SPRINT.md** — the Aurora-compute sprint plan. The Sept-15 deadline
  it was built around was dropped by the owner on 2026-08-30 (see `GOALS.md`
  decisions log); its reprioritization is folded into `GOALS.md`.
- **DECODER_DESIGN.md** — the first decoder (output-half I/O) design draft.
  Superseded by `dev/design/DECODER_MEMORY_DESIGN.md` ("decoder v2"), itself
  now on hold.
- **ENCODER_DATA_STRATEGY.md**, **ENCODER_GOLD_V3_SMALL_STATS.md**,
  **ENCODER_GOLD_V4B_ALL_STATS.md**, **ENCODER_GOLD_V4B_MARGIN_STATS.md**,
  **ENCODER_GOLD_V4B_SMALL_STATS.md**, **ENCODER_GOLD_V4_SMALL_STATS.md**,
  **ENCODER_TRAIN_FULL_STATS.md** — intermediate gold-build stats and a data
  strategy memo, all superseded by `dev/reports/ENCODER_GOLD_V4B_16K_STATS.md`
  (the gold the current/current-retraining encoder checkpoint actually uses).
- **EXECUTOR_DESIGN_REVIEW.md** — 2026-08-30 adversarial review. Its own
  header says every finding was folded into `dev/design/EXECUTOR_DESIGN.md`
  the same day; kept only for the paper trail.
- **HAND_GOLD_DRAFT.md** — 16 hand-drafted hard-case records, explicitly
  DRAFT/not-trained-on. Superseded by the generated hard-gold templates
  (`HARD_GOLD_GEN_STATS.md`, also archived) which the lead directed instead.
- **HARD_GOLD_GEN_STATS.md** — stats for the template-based hard-gold
  generator's v2-v4 widening passes. The gold it describes is now baked into
  the standard gold-build pipeline; kept here as the historical record of how
  it got there rather than as live reference.
- **INTEGRATION_PLAN.md** — mid-term USVS-integration plan (M30-M32).
  M31/M32 are done; M30 (the WSD correctness gate) is still textually open
  per `dev/design/ROADMAP_LONG_TERM.md` and `GOALS.md` goal 3 (Resolve) —
  **flagged as an uncertain archive call**; consult this doc first if WSD
  work resumes, even though it's filed here.
- **M62_GOLD_VOLUME_PROBE.md** — a one-off corpus-volume probe. Its finding
  is folded into `dev/reports/ENCODER_GOLD_V4B_16K_STATS.md`'s discussion.
- **NEXT_ARC_PLAN.md** — the 2026-08-10 M55-M57/Spanish sequencing plan.
  Executed and superseded by the current state in `GOALS.md`.
- **PARSE_JUDGE.md** — LLM parse-judge design, low priority even at the time;
  `dev/reports/AUDIT_2026-10-15.md` lists it under "low expected information,
  stop." Not deleted in case the self-training loop it assumes ever exists.
- **RESOLVER_BUILD_PLAN.md** — the v1 membrane build plan (M52-M55), its own
  predecessor doc says "phases 1-3 done"; fully executed and superseded.
- **SEMANTIC_MAPPING_PLAN.md** — the grounding-phase (M17-M29) plan.
  Explicitly called "historical" by `dev/design/ROADMAP_LONG_TERM.md` itself.
- **SIGNALS_AUDIT.md** — self-audit of M17-M28 grounding signals; that phase
  is a closed door per `consciousness_transformer/README.md`.
- **SPANISH_GOLD_V2_STATS.md**, **SPANISH_SWAP_FEASIBILITY.md** — the
  feasibility probe and build stats that preceded the actual grammar-swap
  test. Superseded by `dev/reports/SWAP_TEST_RESULTS.md` (the actual result).

## Left in `dev/` (not moved here) despite being dated

`PROSE_FAILURE_TAXONOMY.md` stays directly under `dev/` (not archived) because
`scripts/convert_corpus.py`'s `--taxonomy-out` argument defaults to writing to
that exact path at runtime — moving it would silently change where a script
writes by default. Everything else that's merely *cited* by filename in a
docstring or comment was moved and had those references updated to the new
path.
