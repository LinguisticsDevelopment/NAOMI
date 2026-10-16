# NAOMI — goals and progress

This is the single source of truth for where the project stands. It replaces
tracking progress by milestone number ("M57", "M62b") — new work is tracked
against the goal it serves, with a status and the evidence behind it. For the
full chronological research log (every measurement, including the honest
negatives), see `consciousness_transformer/RESEARCH_NOTES.md`. For how old
M-numbers map onto these goals, see
`consciousness_transformer/dev/archive/MILESTONE_MAP.md`.

The thesis underneath all seven goals: everything the system holds and
manipulates is a **meaning object** over ~65 NSM primes, never tokens. Weights
hold zero information (gradient descent only touches the cognitive policy —
how to perceive, recall, infer, respond); all content lives in a graph/tensor
memory, written by ordinary graph/tensor writes, never by training. See
`consciousness_transformer/MIND_ARCHITECTURE.md` for the design contract.

## Goal status at a glance

| Goal | Status |
|---|---|
| Read | in progress — encoder unfrozen, retraining on fixed gold |
| Remember | working |
| Resolve | working, with an open WSD gap |
| Answer from memory | in progress — benchmark fixed, e2e spike in flight |
| Reason | in progress |
| Speak | on hold |
| Any language | working at curriculum scale, not yet at real-text scale |

---

## 1. Read — text → USVS-grounded meaning trees

**What it means.** Turning a sentence into a meaning tree over NSM primes:
predicate + role-bound fillers, with every content word's filler grounded to a
point in the Universal Semantic Vector Space (USVS). Two parallel paths: a
deterministic grammar parser (`quantum_parser`, one grammar file per language)
that is the reference implementation, and a learned encoder
(`src/nsm_ct/encoder_model.py`) meant to generalize past the parser's grammar
coverage.

**Status: in progress.** The grounded lexical space (USVS) is settled and not
being revisited. The learned encoder was frozen pending a downstream
measurement; the owner unfroze it on 2026-10-16 to finish out, now that the
gold it trains on has been fixed.

**Proven, with numbers:**
- USVS grounding is done: synonym AUC 0.86, syn>ant 0.756, hypernym 0.72,
  random-pair ~0 (`consciousness_transformer/README.md`, M17-M29 in the
  milestone map).
- The encoder is real but modest: rank-1 edge-F1 0.622 (P 0.750) on the frozen
  checkpoint, vs a POS-heuristic cheat baseline of **0.268** — a genuine +0.354
  margin (`consciousness_transformer/dev/reports/AUDIT_2026-10-15.md`,
  `dev/reports/K12_BENCHMARK_FIX.md`, `scripts/cheat_encoder_baseline.py`). It is
  data-limited (0.49 @788 -> 0.55 @3,000 -> 0.63 @8,000 records), one seed per
  point.
- The gold it trains on was dropping clauses: verb coverage was 53.5%, 40.5%
  of records under 50% covered. The 2026-10-15 clause-extraction fix (three
  targeted bugs in `clause.py`) moved that to **54.3% -> 67.9%** mean verb
  coverage, re-saving 14 of 33 originally dropped-clause holdout sentences as
  fully recovered (`consciousness_transformer/dev/reports/GOLD_COVERAGE_AUDIT.md`,
  `dev/reports/CLAUSE_EXTRACTION_FIX.md`).
- The Spanish grammar-swap acceptance test transfers on one metric with the
  encoder frozen (`dev/reports/SWAP_TEST_RESULTS.md`) — see "Any language" below.

**Open.** Whether the encoder actually helps comprehension downstream (vs.
just the deterministic parser on what it already covers) is still
unmeasured — that is gated on the "Answer from memory" spike below. Code-
switching / multilingual input to the encoder itself doesn't exist (its
hard-coded POS rules and prime set are English-only).

**Next.** `encoder-gold-v6` (in flight): rebuild gold on the fixed clause
extraction, retrain, beat the POS cheat baseline clearly, confirm with a
second seed.

---

## 2. Remember — tensor memory, schema, provenance, long-term memory

**What it means.** Working memory (STM) as an order-3 `entity (x) relation (x)
value` tensor; attribute facts are ordinary writes into that same tensor
(`e_instance (x) attr:<name> (x) value`); a separate long-term memory (LTM)
tensor for what survives consolidation; every write carries provenance
(source, language, timestamp, trust) in an append-only log outside the tensor,
since superposition would destroy an audit trail stored inside it.

**Status: working.** The schema, the instance registry, write-back, LTM, and
consolidation are all built and gated.

**Proven, with numbers:**
- Capacity: dim-48 memory holds ~256 facts at >=0.95 forward recall
  (`dev/reports/CAPACITY_CURVE.md`).
- Entity-instance subsystem (registry + attribute facts + provenance) built
  and resolver-driven write-back proven at full scale.
- The read path (post-collapse read at the resolved address + entity-axis
  inverse read) proven: forced-gold accuracy 0.423 -> 0.856.
- The full schema proven on rich (3-8 entity) episodes, with explicit binding
  shown necessary once episodes reach 8 entities.
- Episodic LTM proven: consolidation carries facts across passages; identity
  linking through the existing resolver contract hits 1.000.
(All of the above: `consciousness_transformer/RESEARCH_NOTES.md`, M57/M57a-c
and M59 in the milestone map; design: `dev/design/LTM_DESIGN_BRIEF.md`.)

**Open.** Pruning/forgetting policy at scale beyond what's been tested;
truth-memory promotion (reusing the tier-generic consolidate op for facts
that graduate from "believed" to "known") is designed but not built.

**Next.** No specific run in flight; this goal is in the "working, revisit
when the next goal needs more of it" state.

---

## 3. Resolve — pronouns, word senses, readings

**What it means.** Binding a pronoun or an ambiguous reading to the right
referent/sense at the point a clause is consumed, via a learned collapse over
a candidate set (never a hand rule).

**Status: working, with an open gap.** Pronoun resolution ("Track A") is
proven at full scale. Memory-context word-sense disambiguation beat the MFS
floor honestly on a rematch. The one piece that is built but **unmeasured**
is the coherence-driven sense resolver's correctness vs. the MFS baseline on
gold data — that gate decides whether it ever sits in the live perception
slot.

**Proven, with numbers:**
- Pronoun binding ("Track A") resolves reference perfectly at full scale;
  generalizing pronoun binding re-confirmed at 1.000 after a memorization scare
  was caught and fixed.
- Memory-context WSD rematch against the M30 MFS floor: won honestly.
- v1 membrane gate: pronouns and senses proven together at full scale (readings
  — garden-path ambiguity resolution — leak again under the combined load;
  deprioritized, see `dev/archive/AURORA_SPRINT.md`'s 2026-08-11 reprioritization).
(`RESEARCH_NOTES.md` M52-M56c in the milestone map; design:
`dev/design/MIND_INTERFACE.md`.)

**Open.** The coherence `IterativeSenseResolver` (`src/nsm_ct/wsd.py`) has no
correctness measurement yet. Garden-path reading collapse was deprioritized in
favor of the memory-schema work and never revisited.

**Next.** No specific run in flight.

---

## 4. Answer from memory — QA on real prose, abstain instead of confabulate

**What it means.** Read real sentences (not synthetic templates), write what
they say into memory, and answer comprehension questions from that memory —
abstaining rather than inventing an answer when memory doesn't contain it.
This is the thesis's central untested risk: the whole architecture is proven
on synthetic curricula, not yet on real text at scale.

**Status: in progress.** The zero-shot number exists and is honest; the first
K-12-style benchmark was found to be solvable by cheating and has just been
fixed; the full end-to-end spike (teacher-built episodes over real K-12
passages, with a memory-severed control) is in flight.

**Proven, with numbers:**
- Zero-shot prose number (frozen system, self-generated QA): overall 0.583 vs
  a 0.250 random floor; wild (fully unseen) text 0.389; the system abstains
  honestly rather than confabulating (`RESEARCH_NOTES.md` M58b/M58 in the
  milestone map).
- **Today's benchmark fix:** the FairytaleQA MC benchmark drew distractors from
  *other stories*, so "appears in this passage" alone solved it — substring-cheat
  accuracy was **0.973**. Switching to same-passage distractors drops that cheat
  to **0.250**, exactly random. MCTest mc160 (native in-story distractors) was
  converted and gates the same way at 0.260 vs a 0.350 gate (never was broken).
  `scripts/cheat_baselines_k12.py` and `scripts/cheat_encoder_baseline.py` are
  now wired into the standard report for every future number on this
  benchmark. (`dev/reports/K12_BENCHMARK_FIX.md`, `dev/reports/AUDIT_2026-10-15.md`.)

**Open.** The end-to-end QA spike (teacher-built episodes over the fixed
FairytaleQA entity items + MCTest, with a severed-memory control) is running
(branch `e2e-qa-spike`) but has not reported a number yet. Downstream
comparison of teacher-built vs. encoder-built episodes hasn't run. Abstention
on mismatched-passage ("not stated") questions hasn't been measured directly
yet — it's designed (pair a question with the wrong story's passage, gate on
abstain-rate >=70% mismatched / <=15% matched) but not run.

**Next.** Land the e2e spike's number: gate is test accuracy >= best cheat
baseline + 10 points AND the memory-severed ablation within 3 points of
random. If it fails, this becomes the project's only work until it doesn't.

---

## 5. Reason — executor / multi-step inference

**What it means.** A small learned executor over typed registers (addresses,
vectors, scalars, candidate sets) that chains ops — one op's output feeding
the next op's input — rather than a single fixed program per clause. D1: hard
(straight-through) selection for memory-key addresses; soft selection only for
distributions/scalars. D2: the GRU's self-loop carries only control signals
(previous op, step, register types, scalar margins), never register data — a
strict split so the controller can't shortcut by reading the answer directly.

**Status: in progress.** The op library and the deterministic executor (the
VM the controller governs) are built; the symbolic floor (forward-chaining +
unification) reproduces gold answers. The learned controller's navigation
policy is proven on synthetic multi-hop and backward-chaining tasks. Whether
chained op-selection generalizes compositionally to a new task via
leave-one-program-family-out (LOFO) is Phase 2's first gate and has run twice
with a mixed verdict.

**Proven, with numbers:**
- Multi-hop reasoning that reads its own output cracks depth: held-out depth-3
  reaches ~0.97 with adequate hops.
- Backward (goal-directed) chaining takes ProofWriter depth-2 from 0.48 to
  ~1.0 (`consciousness_transformer/README.md`).
- LOFO gate #1: inconclusive as instrumented (two full-gap recoveries, four
  gap-less cells, one oracle inversion). LOFO gate #2, after an instrumentation
  repair: **GO** — 4/4 informative cells captured 100% of the oracle-floor gap,
  tracelessly (`RESEARCH_NOTES.md`, "Executor LOFO gate" entries).

**Open.** Learned when-to-stop (adaptive halting) is still maturing. PROVENANCE
and FORCE are deliberately not selectable ops. The Advice Test (acquiring a new
task purely by explanation, zero retraining) is a far-endgame target, not
scoped yet.

**Next.** No specific run in flight; the design is `dev/design/EXECUTOR_DESIGN.md`
(D1/D2 and the full decision set), read-ordered behind `dev/design/TRACK_C_DESIGN.md`,
`dev/design/OP_INVENTORY.md`, and `dev/design/MIND_INTERFACE.md`.

---

## 6. Speak — decoder / surface realization

**Status: ON HOLD (owner decision, 2026-10-16).** The memory-frame decoder
confabulated. The owner's read is that this is because memory recall from real
prose is not mature enough yet — not a decoder-architecture problem — so the
fix is to prove "Answer from memory" first, not to keep iterating the decoder.
**The system must never make up an answer; it abstains instead.** Revisit only
after end-to-end QA from memory is proven.

**What's on record.** The original `function_vocab` realizer memorized 2,047
content words and confabulated (train token-acc 0.79 vs dev 0.16) because the
grounded structure covers only ~24% of surface tokens — exact-surface
reconstruction by copy is structurally impossible. The replacement design
(retrieval-augmented realizer over dialectic memory,
`dev/design/DECODER_MEMORY_DESIGN.md`, now carrying an ON HOLD banner) failed its
coverage gate twice and was never rebuilt against a reduced-realization spec.
First-words-out (single-clause, stilted-but-correct realization through the
grammar run backwards) was scoped as part of the real-prose comprehension
step (`dev/archive/NEXT_ARC_PLAN.md` M57) and hasn't happened yet either.

**Next.** Nothing, by design, until "Answer from memory" reports a real
number.

---

## 7. Any language — grammar swap / Spanish freeze test

**What it means.** The decisive falsification test of the knowledge/policy
separation: map a Spanish lexicon onto USVS, author a Spanish grammar, and
test reading comprehension with the trained reactor **frozen** — zero
retraining. Sense signatures are keyed to WordNet synset IDs, not English
strings, so a correctly-perceived Spanish sentence should be invisible to the
frozen mind; any gap localizes to perception (parser/lexicon/role-map), which
is inspectable by construction.

**Status: working at curriculum scale, not yet at real-text scale.**

**Proven, with numbers:**
- Stage 1 (curriculum-scale freeze test): the interlingua claim holds at the
  stream level (`RESEARCH_NOTES.md`, M-ES1).
- The grammar-swap acceptance test (EN-trained encoder, frozen weights,
  evaluated on Spanish records) transfers on one metric
  (`dev/reports/SWAP_TEST_RESULTS.md`).

**Open.** The decisive, real-text-scale version of the test
(`dev/design/ROADMAP_LONG_TERM.md`, "The Spanish Freeze Test") is sequenced *after*
English real-text comprehension reaches its own ceiling — i.e., it waits on
"Answer from memory" landing a real number first. The far-future
"Grammar-as-Memory" arc (grammar rules as addressable, inducible memory rather
than static files) is unscoped design, not in progress.

**Next.** Nothing in flight; blocked behind "Answer from memory."

---

## Decisions log

Dated owner/lead decisions, by topic (not by M-number). Pull requests and
ledger entries should cite the topic, matching the headings below.

- **2026-08-11 — memory schema (M57).** Attribute facts are ordinary writes
  into the `entity (x) relation (x) value` tensor (`e_instance (x)
  attr:<name> (x) value`); the instance registry is an index only (minted
  atoms, id -> vector), no knowledge lives in it; provenance is a
  membrane-side append-only log, one record per gated write (it cannot live
  in the tensor — superposition destroys the audit trail).
- **2026-08-30 — long-term memory.** Separate LTM tensor with additive reads,
  STM-only writes; identity linking through the existing resolver contract
  (`from_ltm` feature + `link_threshold` dial); consolidation as a named
  substate event (READING -> WIND-DOWN -> CONSOLIDATE) gated by `trust_ltm`
  on provenance; the consolidate op is tier-generic so truth-memory promotion
  can reuse it later. See `dev/design/LTM_DESIGN_BRIEF.md` §5.
- **2026-08-30 — executor.** D1: hard (straight-through) selection for memory
  keys (Addr); soft selection only for Dist/Scalar types. D2: the GRU
  self-loop carries only control signals (prev op, step, register types,
  scalar margins), never register data (the Harvard split). Track A stays the
  tournament baseline; LOFO is Phase 2's first gate; PROVENANCE/FORCE are not
  selectable ops. See `dev/design/EXECUTOR_DESIGN.md` §6.
- **2026-08-30 — the sprint clock is dropped.** The Aurora compute deadline no
  longer governs sequencing (Aurora may not happen; that's acceptable).
  Standing priority order from that point: stop requiring minimal episodes
  (long, many-entity discourse) -> make memory comprehensive for the v2 design
  (provenance, morphology, inverse queries, capacity curves, episodic LTM) ->
  then the real-prose test. See `dev/archive/AURORA_SPRINT.md`.
- **2026-10-16 — encoder unfrozen.** Keep finalizing the encoder (close to
  done): retrain on gold rebuilt with the fixed clause extraction
  (`encoder-gold-v6`, in flight), beat the POS cheat baseline clearly, confirm
  with a second seed.
- **2026-10-16 — decoder stays on hold.** It confabulated; believed cause is
  immature memory recall, not the decoder design. Revisit only after memory
  recall / end-to-end QA from memory is proven. The system must abstain
  rather than make up an answer, always.
- **2026-10-16 — docs reorganized by goal.** Progress is tracked against the
  seven goals in this file, not against M-numbers. New ledger entries are
  titled "`<Goal>: <result>`". Old M-numbers remain findable via
  `consciousness_transformer/dev/archive/MILESTONE_MAP.md`.

## Ops rules

The working agreement (who writes code, agent discipline, training-run rules,
the cheat-baseline and held-out-atom laws) lives in `CLAUDE.md` at the repo
root — read that for how work actually gets done here. This file is *what*
the project is trying to prove and where it stands; `CLAUDE.md` is *how*.
