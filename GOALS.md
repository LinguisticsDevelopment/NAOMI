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
| Read | usable on short simple text (5-8% of links wrong at <=10 words; ~20% beyond); the parser itself is ~15% wrong on children's stories; joining pieces of long literary sentences remains unsolved |
| Remember | live memory is now a layered graph (`src/naomi/memory.py`); earlier results were on the archived flat-tensor path |
| Resolve | working on archived code, with an open WSD gap; to be re-attached |
| Answer from memory | full chain plateaus at ~0.35 ROUGE-L on held-out FairytaleQA over three thinker versions (v2b 0.348, v3 0.345; cheat 0.220, oracle 0.757; `dev/reports/MIND_V3.md`); step-down thinker v4 is worse (0.297, `dev/reports/MIND_V4.md`), short-to-long improves only partly (keeps 45% on 21+ vs 30%) and overshoots in depth. Hand-read: long-sentence failures are policy errors (sentence choice ~1/3), not tree errors; trees hold the answer equally at all lengths (57%). Thinker v5 (composes the answer by copying from all of memory, owner-approved question completion, `dev/reports/MIND_V5.md`): 0.298, beats cheat 0.200, below the old method 0.346; it learned to drop the question frame (references are short phrases), so the learned-unknown test did not run. Looped thinker (owner's intended design: one learned transition over USVS trees + state + pre-stored memory, learned halt, `dev/reports/MIND_LOOP.md`): first version 0.177, below cheat 0.201; overfits 20 stories (0.656); halt collapsed to one step; memory use shallow. Loop v2 (`dev/reports/MIND_LOOP_V2.md`): halt fixed (uses ~3-4 steps) but steps add nothing; counterfactual-memory training did not make it read memory (swapped answer appears 1.8%); 0.180 vs cheat 0.201. Loop v3 (owner's asymmetric grade +1/+0.25/-1, `dev/reports/MIND_LOOP_V3.md`): learns total silence (coverage 0), because its word choice is right only 12.6% of the time. Three loop versions show the thinker cannot learn to find the answer in memory from FairytaleQA alone. Next: owner decision on what teaches retrieval |
| Reason | in progress on archived code; to be re-attached on the minimal loop |
| Speak | decoder un-parked as the encoder's round-trip partner only; must be learned (grammar cannot generate) |
| Any language | working at curriculum scale (archived code), not yet at real-text scale |

**2026-11-14 refactor.** The curriculum-era code was moved to
`consciousness_transformer/archive/` (same relative paths; pre-refactor tree
also at branch `integration-2026-11`, commit 2e18721). Live code is the slim
`consciousness_transformer/src/nsm_ct` reader/grounding core plus the new
`consciousness_transformer/src/naomi` read -> remember -> recall loop. Paths
below that say `archive/...` point at code that no longer runs from the live
tree.

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
gold it trains on has been fixed. The hybrid reader runs end to end in
`src/naomi/reader.py` (2026-11-14).

**Proven, with numbers:**
- USVS grounding is done: synonym AUC 0.86, syn>ant 0.756, hypernym 0.72,
  random-pair ~0 (`consciousness_transformer/archive/README_pre_refactor.md`, M17-M29 in the
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
- **Gold v6 (2026-10-15, built on the clause-extraction fix):** 22,990
  records (76.0% yield of 30,232 attempted) over the full Gutenberg
  population plus a seeded 125-whole-story K-12 FairytaleQA subset, holdout
  excluded at the corpus level. Mean VERB coverage **67.8%** (vs v4b-16k's
  53.5%). `consciousness_transformer/runs/encoder_gold_v6.jsonl.gz`;
  `dev/reports/ENCODER_GOLD_V6_STATS.md`.
- **Encoder v6, two seeds (2026-10-15/2026-11), confirms the retrain clearly
  beats the cheat baseline:** seed 0 rank-1 edge-F1 **0.708** (P 0.739 /
  R 0.680), seed 1 **0.677**, both on the 94-sentence v6 holdout, both vs
  POS-cheat **0.288** (the v6-holdout cheat floor; reproduces the documented
  0.268 v4b-holdout number) and random 0.000. Clause-count recall 0.901
  (seed 0). Weak families on both seeds: `speaker_prime`, `quantity`.
  Reference checkpoint: `consciousness_transformer/runs/arms/v6_all_hard_seed0.pt`
  (seed 0 — the higher of the two and the one to build on).
  `dev/reports/ENCODER_V6_SEED0.md`, `dev/reports/ENCODER_V6_SEED1.md`,
  `runs/ARTIFACTS.md` (how to load it). This closes a real gap found along
  the way: the prior "THE CHECKPOINT" (`v4b_all_hard_0.pt`) had never
  actually been committed (only its log/curve were force-added, the `.pt`
  itself slipped through `.gitignore`) — a best checkpoint was lost. Both
  v6 checkpoints are now force-added.

- **Encoder vs. parser head-to-head on unseen prose (2026-11-14, branch
  `encoder-vs-parser`):** the complement question — does the encoder add
  real coverage beyond the deterministic teacher, or just distill it? —
  measured clean for the first time: 2,000 never-seen sentences (1,500
  FairytaleQA val/test + 500 Gutenberg), teacher-OK 58.25% overall (76.3%
  FairytaleQA, 23.7% on the complement the teacher fails). On that
  complement, hand-judged (no LLM, 80 complement + 80 matched + 40
  calibration): encoder any-correct **63.8%** vs. POS-cheat **37.5%**.
  Hybrid coverage (teacher OR encoder) **84.9%**, +26.6pp over
  teacher-only vs. the cheat-hybrid's +15.7pp — the encoder earns ~1.7x
  the cheat floor's gain, a genuine value-add, not re-derivable from POS
  alone. Shared failure mode: negation ("not"/"n't") dropped by both
  encoder and cheat. OOD generalization cost: edge-F1 0.708 (in-distribution)
  -> 0.592 (OOD, vs. teacher on teacher-OK sentences). `dev/reports/ENCODER_VS_PARSER.md`.

- **Hybrid reader in the minimal loop (2026-11-14,
  `consciousness_transformer/src/naomi/reader.py`):** deterministic parser
  first, encoder v6 as fallback. On 300 MCTest sentences: >=1 clause for
  99.3% (parser 289, encoder 9, none 2), ~0.17 s/sentence, 6.2% of clauses
  negated. Negation is now marked deterministically per clause (the parser
  and the encoder both drop it on real prose, where 22.5% of FairytaleQA
  passage sentences and 10.3% of MCTest ones contain a negation cue).

**Known reader weaknesses (2026-11-14).** Noisy argument roles on real
prose; predicate nominals/adjectives dropped ("he was not a favourite"
loses "favourite"); possessors dropped ("jim 's father"); subjects lost
across coordination/ellipsis.

**Open.** This answers "does the encoder add coverage beyond the parser"
(yes). Whether the extra coverage helps downstream comprehension is not yet
measured through the new loop (in the first end-to-end run the encoder
supplied the support for 2 of 60 answered MCTest items; see "Answer from
memory"). Code-switching / multilingual input to the encoder
itself doesn't exist (its hard-coded POS rules and prime set are
English-only).

**Next.** Keep predicate nominals/adjectives and possessors in the reader;
fix argument-role noise; add simple coreference. The encoder is a learned
parser driven by the grammar file and will probably replace the
deterministic parser (2026-11-14 decision); the reader passes candidate word
senses up rather than committing to one. The question-side fixes under
"Answer from memory" come first.

---

## 2. Remember — tensor memory, schema, provenance, long-term memory

**What it means.** Working memory (STM) as an order-3 `entity (x) relation (x)
value` tensor; attribute facts are ordinary writes into that same tensor
(`e_instance (x) attr:<name> (x) value`); a separate long-term memory (LTM)
tensor for what survives consolidation; every write carries provenance
(source, language, timestamp, trust) in an append-only log outside the tensor,
since superposition would destroy an audit trail stored inside it.

**Status: working on the archived flat-tensor path; live memory changed.**
The results below were all measured on the curriculum-era code now under
`consciousness_transformer/archive/` (the flat order-3 tensor, instance
registry, `ltm.py`, `membrane.py`). The live memory since 2026-11-14 is the
layered graph of meaning trees in `consciousness_transformer/src/naomi/memory.py`
(layers current / short_term / long_term / fact, working names; every clause
also has a TPR tensor handle). It has **not** been re-proven for capacity,
LTM or consolidation. The schema decisions in the log below were made for the archived path.

**Proven, with numbers (archived path):**
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

**Next.** Re-establish capacity and long-term behaviour on the layered graph
only when "Answer from memory" needs it; `sleep`/`forget` exist in
`src/naomi/mind.py` but are untested at scale.

---

## 3. Resolve — pronouns, word senses, readings

**What it means.** Binding a pronoun or an ambiguous reading to the right
referent/sense at the point a clause is consumed, via a learned collapse over
a candidate set (never a hand rule).

**Status: working on archived code, with an open gap; to be re-attached.**
(`membrane.py`, `wsd.py` and `resolver.py` are under `consciousness_transformer/archive/`.
The live reader keeps word senses as uncommitted candidates and memory picks
at store time, against what it already holds.) Pronoun resolution ("Track A") is
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

**Open.** The coherence `IterativeSenseResolver` (`archive/src/nsm_ct/wsd.py`) has no
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

**Status: in progress.** The minimal read -> remember -> recall loop
(`consciousness_transformer/src/naomi`) runs end to end on real prose with
no training, and abstains instead of inventing answers from unrelated text.
It is **not yet ahead of the word-overlap cheat baseline**: it abstains on
~80-85% of items, and ~89% of failures are abstains where the answer content
is in memory (the loss is question-tree-to-stored-tree matching).

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
- **SUPERSEDED — E2E QA spike (teacher-built episodes, branch `e2e-qa-spike`) — landed,
  INCONCLUSIVE, not a falsification.** Found and fixed a real gap first:
  `archive/src/nsm_ct/clause_reactor.py`'s question-step adapter was silently dropping every
  free-form MC question (no NAME/"the NOUN" match) — fixed additively (~20
  lines), 2 new tests. Parse coverage **87-95%** and answer-in-memory
  ceiling **79-94%** both healthy on both datasets (FairytaleQA, MCTest
  mc160) — the pipeline isn't the bottleneck. The gate (model accuracy >=
  best cheat baseline + 10pts AND memory-severed ablation within 3pts of
  random) **FAILS**, but inconclusively: model and memory-severed accuracy
  were **IDENTICAL (0.400 / 0.400, n=25)** on FairytaleQA — a
  head-not-using-memory result confounded by data starvation (the run was
  passage-capped per CLAUDE.md's shrink-and-say-so rule: full-corpus parse
  build projected hours against a ~90min budget, so this ran on 6/3/4
  passages per split). `dev/reports/E2E_QA_SPIKE.md`;
  `runs/ARTIFACTS.md` for the spike's checkpoints/metrics files.
  **Superseded 2026-11-14:** reviewing the spike showed its QA adapter never
  encoded the question for items with no name/"the NOUN" (122/584
  FairytaleQA, 185/400 MCTest) — the earlier "fix" above was incomplete —
  and its memory-severed control also removed the passage from the
  recurrent state, so it could not have tested memory recall at any scale.
  The 0.400/0.400 result is uninformative for that reason, not just data
  starvation. Script now at `consciousness_transformer/archive/scripts/e2e_qa_spike.py`.

- **First end-to-end numbers, minimal loop (2026-11-14, no training; hybrid
  parser+encoder reader; fresh memory per item; 4-way multiple choice):**

  | | MCTest mc160 (400 items) | FairytaleQA entity-MC (222 items, max 3/story) |
  |---|---|---|
  | answered | 15.0% (60) | 20.7% (46) |
  | accuracy when answered | 55.0% (33/60) | 69.6% (32/46) |
  | accuracy, random guess on abstain | 29.5% | 34.2% |
  | gold answer content in memory | 89.2% | 71.6% |
  | wrong-passage abstain | 100% | 98.6% |
  | random baseline | 25.0% | 25.0% |
  | word-overlap cheat | 45.0% | 56.3% |

  MCTest support came from the parser 58 / encoder 2. Caveat: part of the
  wrong-passage abstain comes from the multiple-choice rule (abstain when no
  option overlaps the support), not only from memory refusing. Hand-read
  failure causes (25 per dataset, MCTest/FTQA): how-many/how-much/how-long
  6/4; copular/attribute questions ("who was X's Y", "what color") 5/5;
  question tree or hole wrong 4/7; answer is a phrase/clause or needs
  cross-sentence inference 5/3; wrong support clause 3/4; why 1/0; negation
  0/1; cloze without hole 1/0; option matching/coreference 0/1. Ledger:
  `consciousness_transformer/RESEARCH_NOTES.md`, "Answer from memory: minimal
  read->remember->recall loop..."; artifacts in
  `consciousness_transformer/runs/ARTIFACTS.md`.

- **Second pass, question-side fixes (2026-11-14, still no training):**
  possessors, copular complements (symmetric `be`), quantities, a
  surface-pattern question builder, clause splitting, shallow he/she/it
  coreference, stricter option backing.

  | | MCTest (400) before -> after | FairytaleQA (222) before -> after |
  |---|---|---|
  | answered | 15.0% -> 30.0% | 20.7% -> 31.1% |
  | accuracy when answered | 55.0% -> 70.8% (85/120) | 69.6% -> 87.0% (60/69) |
  | accuracy, random guess on abstain | 29.5% -> 38.8% | 34.2% -> 44.3% |
  | wrong-passage abstain | 100% -> 99.8% | 98.6% -> 99.5% |
  | question-shuffled control: answered / accuracy | 12.2% / 28.6% | 10.8% / 25.0% |
  | word-overlap cheat | 45.0% | 56.3% |

  The shuffled-question control falls to chance, so answers depend on the
  question. Still below the word-overlap cheat overall because it abstains
  on ~70%. **Caveats:** the fixes were developed while looking at failures
  from these same items (120/80-item subsets), so these are development
  numbers, not held-out ones; and questions are now built by hand-written
  surface patterns in `src/naomi/reader.py` rather than by the grammar
  parser or the encoder — scaffolding that the learned reader should
  replace. Remaining failures: why/inference questions, answers that are a
  phrase or clause, appositives ("Reynard the fox"), parser garbage on
  long/quoted sentences, wrong pronoun antecedents.

**Open.** Question-side matching (see failure causes above); reader gaps
(predicate nominals/adjectives, possessors, coreference). The wrong-passage
abstain rate is partly the multiple-choice rule, so a cleaner abstention
measurement (memory refusing on its own) is still open. Only one cheat
baseline per number is run for now (owner direction, 2026-11-14).

**Next.** (1) Get held-out numbers: run the unchanged loop on FairytaleQA
stories it has not been developed on (198 of 277 stories have no question
files in the repo yet — needs a download). (2) Raise the answer rate:
appositives and naming, phrase/clause answers, multi-sentence support.
(3) Move the question handling from hand patterns into the learned reader.
(4) Bring the learned reasoning model back on top of the graph operations.

---

## 5. Reason — executor / multi-step inference

**What it means.** A small learned executor over typed registers (addresses,
vectors, scalars, candidate sets) that chains ops — one op's output feeding
the next op's input — rather than a single fixed program per clause. D1: hard
(straight-through) selection for memory-key addresses; soft selection only for
distributions/scalars. D2: the GRU's self-loop carries only control signals
(previous op, step, register types, scalar margins), never register data — a
strict split so the controller can't shortcut by reading the answer directly.

**Code archived (2026-11-14):** the executor, op library and learned controller are under `consciousness_transformer/archive/` (`src/nsm_ct/executor.py`, `ops.py`, `mind/`); the results below stand as historical and are to be re-attached on top of the minimal loop (reasoning model learning a set of operations over the memory graph, tensors as its view; to be tested, not settled).

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
  ~1.0 (`consciousness_transformer/archive/README_pre_refactor.md`).
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

**Code archived (2026-11-14):** the curriculum-scale freeze test code is under `consciousness_transformer/archive/`; its results stand as historical and are to be re-attached on top of the minimal loop. The parser/grammar-swap pieces (`quantum_parser`, `encoder_model.py`) are live.

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
- **2026-11-14 — `master` is the mainline.** Fast-forwarded to the
  `integration-2026-11` cleanup branch; feature work branches off `master` and the
  director merges it back. The director session has push permission via repo
  `.claude/settings.json` (normal pushes allowed; force-push and branch deletion denied).
- **2026-11-14 — next-step planning handed to a fresh Fable session after
  this cleanup.** This integration (branch `integration-2026-11`: October's
  nine unmerged branches plus `encoder-vs-parser`, encoder v6 landed on two
  seeds, the e2e spike's inconclusive result, the CLAUDE.md/README ops
  rewrite, the branch inventory and root tidy below) was done by a Sonnet
  routine so a fresh Fable session could read the repo cold from one
  branch and decide what's next for "Read" (the downstream teacher-vs-
  encoder-episode comparison the `encoder-vs-parser` result motivates) and
  "Answer from memory" (the full-scale e2e re-run) without first having to
  untangle ~95 branches. See `BRANCHES.md` and
  `consciousness_transformer/runs/ARTIFACTS.md`.

- **2026-11-14 — minimum viable codebase; archive the rest.** Owner
  direction: refactor to a minimum viable codebase, archive the rest, stop
  building on half-working curriculum-era code. Heavy validation/gating is
  deferred while the work is this experimental: one cheat baseline per
  reported number is enough for now. See `consciousness_transformer/archive/README.md`.
- **2026-11-14 — encoder is the future parser; senses are passed up.** The
  encoder is a learned parser driven by the grammar file and will probably
  replace the deterministic parser. Word-sense choice needs feedback from
  memory, so the reader passes candidate senses up rather than committing.
- **2026-11-14 — memory is one multi-layer graph.** Meaning trees in layers
  (fact / long-term / short-term / current — working names, not law), whose
  nodes are also tensors. The reasoning model is expected to learn a set of
  operations over that graph, with tensors as the view it sees (to be
  tested, not settled). Live in `consciousness_transformer/src/naomi/memory.py`.
- **2026-11-14 — what "taught" means; canonical path.** "Taught" = teaching
  it how to think through specific things, teaching new words, having it
  correctly identify facts. Path: finish encoder/decoder, then teach it K-12
  lessons to build the mind up to reasoning over its own data, then hold a
  conversation — deliberately loose, open to discussion.
- **2026-11-14 — decoder stays on hold** (unchanged from 2026-10-16).
- **2026-11-14 — director may push to `master`.**

- **2026-11-14 — no hand patterns; answering is learned; encoder+decoder
  train as a pair.** Hand-written language patterns in code and a rule-based
  answerer are antithetical: the second-pass "question-side fixes" and
  `recall.py` were removed the same day (kept under
  `consciousness_transformer/archive/naomi_scaffold/`; their benchmark
  numbers are a signpost of what trees must contain, not a system result).
  Language rules live only in grammar files. The encoder only parses; the
  consciousness model learns to answer from memory. Encoder training
  signals, in the owner's words: (1) how correct the tree is, as a
  percentage with specific feedback (clause dropped, what is missing);
  (2) paired with comprehension, checking the answer state in USVS;
  (3) plus the decoder, natural-language answer to a natural-language
  question; (4) encoder+decoder round trip, same text in and out. Chosen
  direction: train encoder and decoder together (4) with the middle tree
  validated independently (1), to break the chicken-and-egg between them.
  This lifts the 2026-10-16 decoder hold for the round-trip role only.

- **2026-11-15 — function words live in USVS (design approved).** Meaning
  lives in USVS only; primes are a starting set, special dimensions may be
  added. Function words are a closed class defined per language as data
  (`quantum_parser/lexicons/<lang>.json`) and placed on named dimensions:
  the 26 prime dimensions that were empty by accident (a shared stopword
  list) are filled, and seven are added (KNOWN, UNKNOWN, GOAL, SOURCE, AND,
  MALE, FEMALE). Negation is a graded dimension (0 not negated .. 1 fully).
  Operator dimensions are filled only by closed-class words — content words
  such as "unhappy" get no negation value, and the existing content-word
  placement and antonym design are not touched. Every word is a node in a
  nested tree: the tree gives scope, the dimensions give degree. Numbers
  are treated as names for now (later probably a math language); idioms are
  facts in memory. This supersedes the 2026-09-07 D3/D4 decisions
  (quantity/additive/focus as structural labels; I/YOU outside USVS), the
  preposition-to-role table, and "USVS is settled" for the closed class
  only. Gold and encoder must be rebuilt on the new tree format. Design:
  `consciousness_transformer/dev/design/FUNCTION_WORDS_IN_USVS_PROPOSAL.md`;
  background: `dev/reports/LIT_FUNCTION_WORDS_SEMANTICS.md`,
  `USVS_DESIGN_HISTORY.md`, `USVS_FUNCTION_WORD_AUDIT.md`.

- **2026-11-15 — USVS content-word placement is under review (planning
  only; nothing changed yet).** Diagnosis and alternatives:
  `consciousness_transformer/dev/reports/USVS_CONTENT_DIAGNOSIS.md`,
  `USVS_ALTERNATIVES.md`. Owner rulings so far: degree and negation words
  may act on a word's position on a dimension as well as having their own
  operator dimensions; "minimum" means fewest dimensions overall, judged by
  every dimension earning its place; dimensions may be derived from the
  relation graph and named afterwards (an LLM may label them); finding the
  space may involve training as long as it is transparent, has provenance
  and is reproducible — not opaque random weights; a new version is
  probably built beside the frozen 607-dimension space, replacing it only
  if it wins. Owner idea under test: each dimension stores an importance
  (weight) and a position (value) per word. NSM primes are judged a poor
  basis for content words in a flattened space; they stay for closed-class
  and operator words. Working rule: plan heavily, identify issues and
  alternatives before implementing; use cheaper agents and fewer at once.

- **2026-11-16 — encoder-decoder pair method approved.** Encoder outputs
  each word's head and relation; decoder v1 produces word order from a
  tree; stage B is self-training with the grammar, the parser and the round
  trip as referees; the parser's unconnected pieces are treated as unknown,
  never as targets. `consciousness_transformer/dev/design/ENCODER_DECODER_PAIR_PROPOSAL.md`.

- **2026-11-17 — treebanks as pre-labelled examples.** The stage-B referees
  could not tell right added links from wrong ones, so the missing evidence
  about how clauses connect is taken from human-annotated treebanks,
  converted into the project's tree format by a data table (no words in
  it; the same table runs on Spanish). Owner condition: this is only a
  source of pre-labelled sentences for learning to encode and decode — it
  must not become a way of hand-coding how a particular language works.
  The project's grammar remains the definition of the format and wins
  wherever its parser commits; converted links are used only where the
  parser is silent. Treebank data is fetched by script and not committed.
  `consciousness_transformer/dev/reports/TREEBANK_TEACHER_FEASIBILITY.md`.

- **2026-11-17 — consciousness v1 approved** (`consciousness_transformer/dev/design/CONSCIOUSNESS_V1_PROPOSAL.md`).
  Answers by pointing to a memory node (speaking stays parked); trained on
  the answer only, one step first, then several; MCTest and FairytaleQA
  stand in for K-12 lessons (no particular K-12 source chosen); first
  operations find / follow / compare / answer / abstain. Owner caution:
  do NOT lock the system to basic sentences — it must show generative-
  grammar behaviour (finite means, unbounded sentences by composition).
  So short sentences are where training starts, not a limit: every result
  is reported by sentence length, and generalisation from short to long
  (train on short, test on long) is an explicit test.
- 2026-11-17 — consciousness v2 approved (`dev/design/CONSCIOUSNESS_V2_PROPOSAL.md`): v1's pick-a-node-then-map-to-option test judged broken by the owner. Full chain encoder -> thinker -> decoder, answer scored as TEXT (ROUGE-L + tree score % with reasons); thinker trained toward the reference answer tree, then on the final text score; answer words from story/question only for now; decoder also chooses words from meaning (reported beside order-only); short sentences first as a widening weighting, all lengths always reported.

## Ops rules

The working agreement (who writes code, agent discipline, training-run rules,
the cheat-baseline and held-out-atom laws) lives in `CLAUDE.md` at the repo
root — read that for how work actually gets done here. This file is *what*
the project is trying to prove and where it stands; `CLAUDE.md` is *how*.
