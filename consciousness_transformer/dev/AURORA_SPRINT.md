# The Aurora sprint (2026-08-11 → mid-September)

Aurora compute access until ~Sept 15. Objective: reach "SCALE IS THE ONLY
PROBLEM" locally, then spend the cluster on scale. Standing invariant:
the deliverable stays locally-runnable post-training — USVS + grammars +
a sub-MB policy. Nothing in this sprint may add inference-time weight.
User's goal on record: past Tier 2. Honesty machinery (M24, held-out
atoms, forced-wrong floors, cheat baselines) is NON-NEGOTIABLE — it is
what will make scale results believable.

Why this architecture fits an HPC cluster unusually well: our scale work
is embarrassingly parallel — corpus parsing (CPU, no coordination),
thousands of SMALL independent training runs (sweeps/seeds/tournaments),
deterministic knowledge builds (USVS variants, multilingual OMW
mappings). No giant coupled job anywhere.

## Week 1 (now): close v1 + the two corpus prerequisites
- M55 gate (RUNNING) → v1 membrane complete.
- M57 instantiation subsystem (entity instances + attribute facts +
  two-Marys/definite-description curricula) — corpus prerequisite,
  user-designed.
- M58 corpus→episode converter v1 + THE ZERO-SHOT NUMBER on real prose
  (frozen system, self-generated QA) — the single most informative
  measurement this project will produce; it writes week 2's worklist.

## Week 2: fix what zero-shot exposes + first words out
- Parser/extraction round driven by real failure taxonomy; error-
  compounding mitigations (per-sentence confidence, held ambiguity
  instead of wrong certainty).
- First local corpus training (small scale).
- Realization v1 (answers as sentences — reverse grammar, stilted OK).

## Weeks 3-4: minimum viable v2 + the scale harness
- LTM consolidation (needed for multi-passage reading); workspace/emit
  only if the QA format demands it. Dials as fixed hyperparams.
- SCALE HARNESS: episode sharding, deterministic checkpointable builds,
  run orchestration, result aggregation — everything batchable. Dry-run
  the whole pipeline locally at 1% scale.
- Track C rides along as a tournament candidate at scale (sweeps are
  cheap there), NOT a local blocker.

## Week 5 → Aurora
- Corpus at scale (Gutenberg/Simple-Wikipedia-class → millions of
  episodes). Training sweeps: capacity × curriculum × seeds at real
  statistical power. A-vs-C tournament. Multilingual USVS builds (es +
  whatever OMW supports cleanly) → the Spanish Freeze Test at scale.
- Success criterion for the sprint: not "Tier 2 achieved" (5 weeks
  cannot promise that honestly) but "the scaling curves exist" — real
  measured curves of comprehension vs corpus size vs policy size on real
  text, so the Tier-2 question stops being philosophy and becomes
  extrapolation.

## Deprioritized until after (recorded, not forgotten)
Fluent generation, pragmatics, full dial conditioning, Advice Test
machinery, parser round beyond what zero-shot demands.
