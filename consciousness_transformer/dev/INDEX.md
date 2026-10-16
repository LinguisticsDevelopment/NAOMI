# dev/ index

`dev/` holds design docs and research reports, grouped by what they're for.
For current project status, read `GOALS.md` at the repo root first — it's
built from these. One line per doc below.

## design/ — live design-of-record (still governs code)

- `ENCODER_IO_CONTRACT_V2.md` — CANONICAL candidates-first encoder I/O schema.
- `ENCODER_MODEL_SPEC.md` — implementable spec for the candidate-lattice encoder.
- `ENCODER_GRAMMAR_FORMAT_PROPOSAL.md` — CANONICAL grammar-as-retrieval format the encoder gold builders use.
- `EXECUTOR_DESIGN.md` — the learned executor's decision doc (D1/D2 locked).
- `LTM_DESIGN_BRIEF.md` — long-term-memory design (§5 = the locked decisions).
- `DECODER_MEMORY_DESIGN.md` — decoder v2 design; **ON HOLD**, see its banner.
- `MIND_INTERFACE.md` — the perception<->thinking membrane contract (v2 locked).
- `OP_INVENTORY.md` — the mind's full op inventory, fixed vs. learned.
- `OP_LIBRARY_MAP.md` — op-library -> clause-reactor integration map.
- `UNIVERSAL_ENCODER_DESIGN.md` — the design that motivated the current encoder's candidates-first approach.
- `ROADMAP_LONG_TERM.md` — the long-term roadmap (Spanish/Advice/Grammar-as-Memory far-future items, standing house rules).
- `TRACK_C_DESIGN.md` — the chained-op-algebra research spike behind the executor.

## reports/ — current measurement evidence (cited in GOALS.md)

- `AUDIT_2026-10-15.md` — the current independent audit (verdict: continue with changes).
- `GOLD_COVERAGE_AUDIT.md` — 2026-10-15 clause-coverage audit that found the 53.5% verb-coverage gap.
- `CLAUSE_EXTRACTION_FIX.md` — the fix for that gap (54.3% -> 67.9% verb coverage).
- `K12_BENCHMARK_FIX.md` — today's FairytaleQA/MCTest cheat-baseline fix.
- `ENCODER_GOLD_V4B_16K_STATS.md` — stats for the gold the current encoder checkpoint trains on.
- `FAIRYTALEQA_STATS.md` — FairytaleQA corpus characterization.
- `K12_CORPUS_SCOUT.md` — the scout report that found the K-12 reading corpora.
- `SWAP_TEST_RESULTS.md` — the Spanish grammar-swap acceptance test result.
- `CAPACITY_CURVE.md` — memory capacity probe (dim-48 holds ~256 facts @0.95 recall).
- `USVS_GRADED_SCORING.md` — the USVS-graded tree-scoring metric (built, in use).

## archive/ — superseded (see `archive/INDEX.md` for why each one)

22 docs; see `archive/INDEX.md` for the one-line reason each was moved, and
`archive/MILESTONE_MAP.md` for the M-number -> goal -> outcome table.

## Left at this top level

- `HANDOFF.md`, `CURRENT_STATE.md` — stubs; their content was folded into
  `GOALS.md`/`README.md`. Kept as stubs because older scripts/docs may still
  point at these paths.
- `PROSE_FAILURE_TAXONOMY.md` — left here (not archived) because
  `scripts/convert_corpus.py --taxonomy-out` writes here by default at
  runtime.
