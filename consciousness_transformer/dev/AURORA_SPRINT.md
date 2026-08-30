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

## Reprioritization 2026-08-11 (post M-ES1, user decision)
- M57 EXPANDS to "robust memory schema": entity instances + attribute
  edges + resolver-driven WRITE-BACK + PROVENANCE METADATA on every write
  (source, language, timestamp, trust — makes invariant #4 real) +
  morphological signals (number/gender subtypes) flowing parser→membrane→
  memory attributes + inverse queries + capacity curve. This is the
  corpus prerequisite and the priority.
- M55c (garden-path redesign) DEPRIORITIZED: two leak-failures say the
  synthetic task is hard to pose honestly in our controlled world; real
  text supplies natural garden paths later. Design law retained; v1
  membrane recorded as 2.5/3, acceptable for the sprint.

## Reprioritization 2026-08-30 (user decision): the clock is dropped
The Sept-15 Aurora deadline no longer governs sequencing; Aurora may not
happen and that is acceptable. Standing priorities, in order:
1. The system must stop requiring MINIMAL episodes — long, many-entity,
   many-fact discourse with mixed referring devices and many question
   types. Training memory footprint (the order-3 memory is B×d³ per step
   with autograd history; ~5-8GB per arm at 1500 eps) is a direct
   blocker for longer episodes and gets engineered down first.
2. The memory becomes COMPREHENSIVE for the v2 design: provenance wired
   into live reactor writes (invariant #4 real), morphology attributes,
   inverse queries, capacity curves, and episodic LTM + consolidation
   (multi-passage reading needs it) — the full M57 reprioritization list
   plus the LTM organ, not the minimal cut.
3. THEN the prose test (M58 corpus→episode converter + zero-shot number).
Local machine is for unit tests/smokes only; full-scale validation runs
in the cloud routine (never on the user's machine). Ops rules unchanged.
