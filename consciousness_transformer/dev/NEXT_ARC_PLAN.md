# Next arc: C → M55 → the corpus campaign → words out → Spanish
(2026-08-10, post-M54c. Predecessor: RESOLVER_BUILD_PLAN.md, phases 1-3 done.)

## M56 — Track C: the op-based resolver (next up)
Goal at end of step: ONE resolver for every collapse type — no per-task
heads — built from named operations with learned routing.
- Ops v1 (task-agnostic, each individually probe-able): mem-query(entity,
  rel), interact(a,b) (the multiplicative candidate×context op M54c proved
  the generalist lacks), feature-match, prior-mix. Routing: learned weights
  over op outputs with sparsity pressure (train soft, evaluate hard).
- Gate, set by M54c data: match Track A on BOTH standing benchmarks —
  pronouns (task 0.913, binding 1.000/1.000) and senses (task ~0.863,
  binding ~0.680/0.629) — with legible routing (we can state which ops fire
  for which candidate type). Distillation stage allowed if needed (M54c
  showed routing may need guidance); must survive its removal.
- If C fails its gate: A remains the working configuration, C's failure
  mode gets ledgered, and the Advice Test path is re-planned.

## M55 — parse-hypothesis collapse (completes the v1 membrane)
Goal: garden-path sentences resolved by memory coherence — the parser
reports its top-K with margins, the mind picks. Built on C if C passed,
A-style rank head otherwise. Garden-path battery = the gate; margins
must correlate with sentence difficulty.

## M57 — the corpus campaign (the answer to "feed it a large corpus")
Goal at end of step: the system reads REAL prose and measurably
comprehends it — the controlled-English claim goes wild or dies.
1. Corpus→episode converter: parser turns narrative prose (start small
   and concrete: graded readers / simple stories, then Simple English
   Wikipedia) into clause streams; comprehension questions generated
   SELF-SUPERVISED from the extracted facts (hold out a clause, ask it —
   who/where/what-has via the queried-role machinery). No hand labeling.
2. Zero-shot measurement first (current model, frozen), then training at
   whatever scale this box tolerates; compute wall flagged when hit, not
   assumed (bigger runs belong on better hardware).
3. FIRST WORDS OUT land here: answers realized as sentences, not MC —
   single-clause surface realization through the grammar run backwards
   (reverse_parser.py is the stub). Stilted-but-correct is the v1 bar.
Full generation (long-form, self-extending output) is NOT this step —
it's the v2 workspace/emit loop, after.

## v2 organs — workspace, emit gate, LTM, dials
After M57's zero-shot + first training results: the locked MIND_INTERFACE
v2 design gets built (consolidation STM→LTM first, then workspace+emit —
which is when long, abstract, self-continued output becomes real).

## Spanish — two stages, deliberately
- CURRICULUM-SCALE freeze test (cheap, any time after M55; perception-side
  agent work): OMW-es sense mapping + minimal Spanish grammar + translated
  curriculum templates; run the FROZEN reactor. Early, small-stage proof of
  the interlingua claim.
- REAL-TEXT freeze test (the roadmap's decisive version): after M57 shows
  English real-text comprehension at its ceiling — Spanish proves the
  architecture only if there's demonstrated competence to transfer.

## Standing rules unchanged
Agents build (foreground, no commits, smokes only — and smokes verify
MECHANICS, never verdicts: two smoke-inversions on record); main session
verifies and runs gates solo; one training run at a time; every step
ledgered win or lose.
