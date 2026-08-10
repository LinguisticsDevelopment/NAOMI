# Next arc: C → M55 → the corpus campaign → words out → Spanish
(2026-08-10, post-M54c. Predecessor: RESOLVER_BUILD_PLAN.md, phases 1-3 done.)

## M56 — Track C RESEARCH SPIKE (design study — explicitly NOT a buy-in)
User's brake, 2026-08-10: single-step routing over op outputs is just B
with named features. A real instruction-set machine CHAINS — one op's
output feeds the next, and the chain is the program. Chaining is both the
entire point of C and the historical graveyard of this architecture
family (soft chains blur; hard chains don't train). C gets a research
spike BEFORE any implementation commitment:
1. The op algebra, formalized: op type signatures (addresses, vectors,
   scalars, candidate sets), what typed chaining means, minimal
   register/stack model, halting.
2. Graveyard survey: NPI, DNC, neural module networks, TerpreT-era
   lessons — why each failed, what our setting changes (deterministic
   perception, small candidate sets, aux supervision available).
3. The TRACE option: unlike the historical systems, we can WRITE the gold
   programs for our solved tasks (pronoun: query-per-candidate →
   feature-match → prior-mix → select; sense: mem-query(subject) →
   interact(candidate, readout) → select). Trace-supervised training
   sidesteps the credit-assignment cliff; the REAL test is then
   compositional transfer — solve a NEW collapse type by recombining ops
   with few/no new traces. That transfer test, not benchmark parity, is
   what would justify buy-in (it is the Advice Test in miniature).
4. Go/kill gates defined IN the spike, before any prototype exists.
Deliverable: a design doc + verdict recommendation. Track A remains the
working configuration throughout; nothing downstream blocks on C.

## M55 — parse-hypothesis collapse (completes the v1 membrane)
Goal: garden-path sentences resolved by memory coherence — the parser
reports its top-K with margins, the mind picks. Built on Track A machinery (a rank
head); C is research-only until its spike reports. Garden-path battery = the gate; margins
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
