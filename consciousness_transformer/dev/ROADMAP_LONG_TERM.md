# Long-term roadmap (reference — not scoped; scope each stage when it starts)

The order below is dependency order. **The semantic-mapping finish line shipped
2026-08-07 (M29 USVS)** — stages 1–2 are now the active mid-term arc, scoped in
`INTEGRATION_PLAN.md` (M30 WSD gate = NOW, then M31 handles, M32 curriculum).

## 1. WSD: decoupled, gated, slotted (ACTIVE — M30 in INTEGRATION_PLAN.md)

Interface (settled): the **parser stays untrained** (grammar files —
`mind/grammar.py` / `quantum_parser`; same parser, different grammar file per
language, far-future) and emits lemma + POS + syntactic context. The **WSD module
may be trained**: `resolve(lemma, context) → sense id → USVS coordinate`, behind
a facade so the parser never knows which resolver is active.

- Gate 1 — correctness on SemCor vs the **MFS floor**. First pass is
  training-free (USVS-signature similarity) so it measures USVS itself; a
  trained coherence resolver is only worth building if the signatures show
  signal. Winner takes the slot; a loss is recorded and MFS stays.
- Gate 2 — task payoff: an **ambiguity-bearing comprehension curriculum**
  (episodes whose answer flips with the sense) so WSD shows up in answer
  accuracy. This doubles as the seed of the comprehension-question corpus.

## 2. The bridge: USVS fillers in the meaning graph (ACTIVE — M31)

Replace label-based CONCEPT handles: a content word's filler/handle vector in
`meaning_graph.py` / mind/ meaning objects becomes a **fixed deterministic
projection of its USVS signature** (named axes → d-dim filler; frozen matrix, no
learning, axes stay readable). Sense selection = the stage-1 winner. Touch
points: `meaning_graph.GraphNode` handles, `mind/schema.py`,
`clause_reactor.build_clause_batch` value vectors.

Extrinsic gate (house rule — substrate milestones must pay downstream): concept
handle-dereference margins improve vs §0j medians (0.034 @ d256), AND at least one
consumer metric moves (STM read-resolution, or option scoring on the ambiguity
curriculum). If neither moves: record the negative, keep the artifact standalone.

## 3. Composition over grounded primes (the central problem)

Clause meaning = TPR binding over **placed coordinates**, with negation/antonymy as
**signed operations** consulting the artifact's edge store at composition time
(post-grounding roadmap items B+C merged). Held-out gate: composed-clause
coordinates reconstruct entailment direction + negation flip on unseen clauses —
M19.4's "dictionary from geometry" lifted from words to clauses.

## 4. The comprehension corpus shift

Training moves to comprehension questions almost exclusively, over meaning vectors
(`<vector A> subject <vector B> verb`), never node labels — the direct test of the
vectors-increase-comprehension thesis. Starts generated (controlled grammar +
ambiguity + derivation-required); real text waits for stage 5. Compare against the
node-ID baseline: that delta is the headline number for the whole grounding arc.

## 5. The learned parser (membrane upgrade)

If/when controlled English becomes the bottleneck on real text: a trained encoder
mapping sentences to the same parse/meaning objects (the `input_encoder.py` TODO
seam). Peripheral plumbing by design — the reasoner never sees tokens either way.
The deterministic grammar parser remains the reference implementation and the
cross-language path (grammar definition files, one per language; early NAOMI had a
rudimentary version).

## 6. Cross-language (far future)

Same parser + a different grammar file → same meaning objects; the space is
language-independent by construction (NSM primes). OMW/colexification signals
(deferred from the mapping plan) become relevant here as the sense-alignment
bridge.

## Standing house rules

- M24: any propagation metric excludes its scored pairs.
- Extrinsic validation: a substrate milestone counts only when a downstream
  consumer measurably benefits.
- Closed doors (M17–M25): more axes, sense-nodes as primary space, trained
  per-word embeddings, distributional signals in the placement.
- No transformers inside the loop; attention only as a single readable
  select-over-explicit-objects op. Depth lives in the loop, not in layers.

## The Spanish Freeze Test (endgame validation, user-specified 2026-08-09)

The decisive falsification test of the knowledge/policy separation: map a
Spanish lexicon onto USVS, author Spanish grammar, and test reading
comprehension WITH THE TRAINED REACTOR FROZEN — zero retraining.

Why it's decisive: sense signatures are keyed to SYNSET IDs, not English
strings (Open Multilingual Wordnet links Spanish lemmas to the same
Princeton synsets — "gato" → cat.n.01 → the SAME vector). A translated
episode therefore produces a near-identical clause stream at the membrane.
If perception does its job, Spanish is INVISIBLE to the mind. Prediction:
comprehension within noise of English. Any gap localizes to perception
(parser/lexicon/role-map) and is inspectable — the architecture makes the
failure analysis possible, which is itself part of the claim.

Transfers for free: sense layer (synset-keyed), reactor weights, memory
machinery, resolver heads, entity atoms (language-neutral by design).
Needs building (all perception-side, all deterministic):
1. Spanish tag lexicon generated from OMW-es (the M41 recipe verbatim).
2. Spanish grammar rulesets (spanish.json — the parser was designed for
   grammar-file swaps; quantum_parser already carries Spanish tagging
   stubs). The interactive/agent parser-dev loop pattern applies.
3. Spanish role map (clause.py _PREP_RELATION is English: needs es
   equivalents — "en"→PLACE, "a"→PLACE/dative, "de"→SOURCE...) — argues
   for making the role map a per-language data file, like the grammar.
4. Spanish word-coordinate layer: either propagate placement over OMW-es
   relations (M29 recipe) or derive word coords from synset-mean of the
   M46 sense layer (cheaper; senses already transfer).
5. Spanish surface templates for the existing curriculum generators
   (translation, not new logic) + pronoun feature rows (ella/él — gender
   is richer in Spanish, the feature table already has the dims).

Sequencing: after real-text English comprehension is measured (the other
big unopened door). Both are perception-side campaigns; the mind is done
being the bottleneck for either.

## The Advice Test (far endgame, user-specified 2026-08-09)

The vision one level above the Spanish Freeze Test: skills themselves
migrate from weights into memory. If the policy bottoms out as a small
universal executor (attend / bind / write / branch / loop over its own
memory — a Turing-complete instruction set), then a new capability is
acquired by EXPLANATION — imperative/conditional clauses parsed by the
same parser, grounded in the same space, stored as a procedure by the
same consolidation machinery — with ZERO retraining. (Lineage: McCarthy's
Advice Taker, 1959 — proposed at AI's founding, abandoned for lack of a
grounded substrate to take advice into.)

Test shape, when we get there: teach the frozen system a task type it was
never trained on, purely via instruction sentences; measure task success.
Gradient descent demoted to bootstrap (learning the interpreter once).

Design consequences already in the locked v2 spec that this depends on:
trust dials applied to PROCEDURES (teachability implies gullibility —
provenance + testing before promotion), the patience dial as the halting
budget for taught procedures, the workspace loop as the execution engine.
Reframes the A/B experiment's stakes: every specialist head A forces us to
keep is evidence the executor isn't universal yet; every capability that
falls out of B's shared mechanism is a step toward the interpreter that
only needs explaining.

## The Grammar-as-Memory arc (far scope, user-specified 2026-08-30)

Two configurations of ONE machine, to be tested as arms: (a) word-by-word
incremental — ingest and thinking in the same loop (natural discussion,
interruptions = hypothesis revision + the patience dial; the M55
hypothesis membrane is the substrate); (b) two-stage — tree builder runs
to completion, then the mind loop (text). The only difference is whether
ingest and thinking are together or separate: a configuration, not a fork.

Grammar rules move from files into an addressable MEMORY SPACE: rule
application = collapse over a candidate set of applicable rules (rules
match abstract categories, not words — proven by M-ES1's spanish.json
clone). A small learned policy selects rules (the executor pattern at the
perception layer; trace supervision free from the deterministic parser's
own successful parses). Spanglish/code-switching = two rulesets active,
selection switching mid-tree.

Consequence: grammar INDUCTION rides the existing promotion machinery —
propose rule from evidence → store low-trust → corroborate/contradict
against parsing success → promote via the tier-generic op with a trust
dial. Learning a new language = writing rules into memory (the Advice
Test applied to grammar). Gate when we get there: the SPANISH INDUCTION
TEST — induce Spanish rules from text with the mind frozen and compare
against the hand-authored spanish.json as gold (the Freeze Test in
reverse). Sequencing: hold until deterministic parser rounds measurably
plateau (coverage-per-round curve); the learned rule-selection parser
(stage 5) comes first and is a prerequisite.

## The two claims (user-articulated 2026-08-31, sharpened on ledger evidence)

1. SMALL-POLICY COMPREHENSION: bAbI/Memory-Networks-class QA (entities,
   places, coreference, multi-hop, cross-passage) from a sub-MB policy
   (~700KB total, 2.7k-param resolver) — three orders of magnitude below
   that lineage — with measured OUT-OF-DISTRIBUTION transfer to real
   prose (M58b: 0.583 vs 0.250 floor, zero prose in training) and
   adversarial validation that lineage never ran.
2. NO CONFABULATION BY CONSTRUCTION: the system can be wrong but cannot
   hallucinate — every answer is a tensor read of things that were
   written, so errors always have locatable causes (parse, binding,
   interference), each diagnosable. Deterministic AND tracked (the
   executor anchor replays the pipeline bit-for-bit; provenance answers
   "why do you believe X"). Third leg: honest abstention under
   uncertainty (M58b: 46% abstain on prose, error-correlated; rises
   under sabotage). Inference is hard-argmax throughout — the
   determinism claim holds exactly at answer time.
Facts learn on the edge (gated writes at inference, auditable), and the
current numbers are the FLOOR: before prose training, before LTM/truth
population at scale, behind a parser reading 18.6% of wild sentences.
