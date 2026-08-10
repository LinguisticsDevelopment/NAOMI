# Research Notes

Separates the **research** (open, unsolved) from the **engineering** (built,
working) in the NSM Consciousness Transformer, and records the design decisions.

## The research-vs-engineering boundary

**Engineered and working** — the stateful loop and its scaffolding:
- The state-transition step, the heads (transition / action repertoire / control /
  response / trust), the `Psyche` unroll (self-controlled loop by default,
  sequential as a baseline), content-addressed overwrite-not-forget memory across
  both tiers, the loss, metrics, config, tokenizer, the curriculum generator, the
  bAbI loader, and the NSM prime constants. With **no action and no trust labels**,
  the model chooses read/think/respond and absorb/append/skip per item and learns
  *when* to answer and *whom* to trust purely from answer correctness (~90%
  held-out, including the corrupting-distractor, corroboration, and update levels).

**Mocked / placeholder / stubbed** — the actual research lives here:
- What the **consciousness state means** and its real objective (abstract +
  placeholder consistency loss).
- **Semantic composition** onto NSM primes (`MockSemanticMapper`).
- A **consistent parser** (the rule parser is experimental; wrapped optionally).
- **Cross-episode credit assignment** for the APPEND action (uses trust × action
  prob as a proxy).
- **Discrete input-pull control** (the controlled loop is a differentiable
  approximation; true RL control is unbuilt).
- **Textbook ingestion** (`TextbookSource` is a stub — the north star).

## Open problems

### 0. Emergent actions and trust: built, with honest caveats
The action repertoire {ABSORB, APPEND, RESPOND, SKIP} and a **trust** signal are
**not** supervised — all shaped only by the answer loss. The answer is a
RESPOND-probability-weighted aggregate (so "when to respond" emerges); trust
scales how strongly each item is written to memory (so "whom to trust" emerges,
because answering corroboration episodes correctly requires discounting the
contradiction). Open issues:
- **Trust: memory is load-bearing now; the trust *gate* still isn't.** The
  **memory-bottleneck readout** (`answer_logits_mem`: the response head with the
  state zeroed, weighted by `weight_mem_answer`) forces the answer to be
  recoverable from memory alone — and it works: mem-only accuracy tracks full
  accuracy (~98% train) with no cost to the headline number. But a focused
  diagnostic (level-5-only training, bottleneck heavy) shows both the trust-head
  gap *and* the effective write-gate gap (`trust_gap(out, batch, key=
  "write_gates")`) hover near zero while mem-only accuracy is high: the model
  discounts the contradiction in **write-content space** (what it writes for the
  false item simply doesn't match the question's read query), not in gate space.
  So "whom to trust" demonstrably emerges *in memory*, but no scalar gate is
  forced to carry it; the trust head and ABSORB are redundant routes. Making the
  gate the necessary mechanism (constrain write vectors to item content only?) is
  open, as is distinguishing **contradiction from temporal update** ("moved to")
  — both look like a memory conflict; overwrite memory (§0c) treats them the same
  way (latest trusted write wins).
- **APPEND** still only pays off in *future* episodes, so it has no within-episode
  answer gradient; it is gated by trust × its action prob, a sensible proxy but
  not real **cross-episode credit assignment** (RL / a value over future
  retrieval). Symptom: the model still APPENDs novel *questions* as "world facts"
  (LTM overwrite-by-text dedups them, but doesn't stop them).
- **Response timing is under-determined** on easy episodes (one fact answers the
  question); `mean_respond_position` is a neutral diagnostic, not a target.

### 0b. The self-controlled read / think / respond loop (BUILT — default)
`agent.Psyche._forward_controlled` is now the default loop. Each tick a
`control_gate` over `{READ, THINK, RESPOND}` decides whether to advance a read
pointer and ingest the next sentence (its **full token sequence** — mean-pooling
the sentence destroyed the word-level detail the MC head needs and would not
train), reason internally with no new input, or contribute to the answer
(RESPOND-weighted aggregate). So responses are sparse and it can process and
*wait*. Tick-level quantities are attributed back to items, so the loss / metrics
/ consolidation paths are unchanged. ~90% held-out, on par with sequential.
Honest caveats / next rungs:
- It is a **differentiable approximation**: a soft-advancing pointer (`p += P(READ)`)
  with a **READ-biased init** so it ingests input before learning to think/respond.
  Truly discrete **input-pull control** ("decide, then fetch the next sentence") is
  an RL / discrete-control problem — the next rung. A confidence/halting signal
  (and the WSD coherence head) is the natural "I'm ready to answer now" trigger.
- "THINK" is implicit (ticks where the model neither advances much nor responds),
  not a hard no-input mode; sharpening it is open.

### 0c. Overwrite, not forget (BUILT)
Memory **overwrites**, it does not decay. Working memory is content-addressed
(`WorkingMemory.write_content`): a write goes to the best-matching filled slot
(updating that fact in place) or a fresh slot for novel content, so a later
trusted fact about the same subject supersedes the earlier one while unrelated
slots are untouched. Long-term memory overwrites by **fact identity** (matching
entry text) so distinct facts still grow the repo while re-statements update in
place. Open: overwrite keys on the model's write *vector* (working) or exact text
(LTM); **subject-level** update ("mary moved" should overwrite mary's location
regardless of wording) needs structure we don't yet extract, and contradiction vs.
temporal-update still resolve identically.

### 0d. Chained-question answering (trained capability + probe)
Multi-question streams are now **trained**, not just probed. The controlled loop
emits `question_logits`: per question, a RESPOND-weighted readout over the ticks
between reading that question and reading the next (a soft pointer window) — so
several questions are answered in **one unreset run**, differentiably. The
per-question loss (`weight_multi`) trains it; `chained_question_episode` /
`generate_chained_episodes` supply streams (`facts, Q1, Q2(other), Q1'(repeat)`).
`scripts/probe_consistency.py` trains with and without the loss on identical data
and reports held-out consistency (Q1 vs Q1') and per-question accuracy
(`answer_at_positions` remains the hard readout for untrained diagnostics).
Result (held-out, n=64, one unreset run each, 2500 steps): with 96 chained
training episodes per-question accuracy sits at ~0.5; with 384 it reaches
**0.84/0.92/0.92 (Q1/Q2/Q1') with consistency 0.89** — on par with the
single-question ~0.9. Without the per-question loss, the same 384 episodes leave
accuracy at ~0.5 (consistency 0.84, partly "consistently wrong"): the loss, not
just the data, converts chained streams into the capability. Remaining open: a
calibrated "answer now" signal instead of the fixed soft window, and longer
chains / more questions per stream.

### 0e. Thought objects — trees of meaning as the model's working unit
The architecture is being reframed: the unit the model works with is a **thought
object** = a parse tree whose word-leaves each carry a **meaning tree** — the word's
reductive explication as a *tree of NSM primes* (an NSM "molecule"). Meaning is
recursive *structure*, not a flat vector. The thinking model's job is to
**manipulate** thought objects (tree + memory → tree); it does **not** build trees
(text↔tree is the parser's / a reverse parser's job). A 3-part chicken-and-egg —
(1) meaning (word → prime tree), (2) WSD, (3) the tree→tree thinking model —
bootstrapped by alternating training.

**Representation decision: explicit (non-embedded) first.** The I/O unit stays
explicit structure (trees of primes), not a single opaque "meaning vector". Explicit
trees are *verifiable* (lossless round-trip lets us read whether meaning is
preserved); collapsing to opaque dims up front is an uninspectable lossy bottleneck.
Vector-space "thinking" still happens *inside* the model (the continuous state +
memory). A learned tree↔vector codec (the opaque "meaning vector") is a deferred,
optional optimization (Stage E), supervised by the explicit representation if tree
manipulation proves too expensive.

**Stage A (built, default).** Thought-object plumbing, end-to-end, mock meanings:
- `thought.py`: `ThoughtObject` (parse tree + per-leaf `MeaningTree`), an
  `AbstractMeaningResolver`, and `MockMeaningResolver` (word → deterministic small
  prime tree — meaning-free but structured, real before WordNet).
- **Lossless serialization** (`serialization.serialize_thought` ↔
  `deserialize_thought`) round-trips to identity — the "parser maintains meaning
  without loss" requirement and the exact target a future tree-decoder emits.
  `reverse_parser.thought_to_text` is the tree→text seed.
- Fed to the model as **additive, zero-init** per-token streams (role, depth, and a
  bag of meaning-prime ids) on top of the words — so a lossy parse/meaning can only
  help or be ignored, never corrupt the input. No regression (~0.85–0.92 held-out).
- Honest limits: meaning input is a **bag of primes per word** (the full meaning
  *tree* lives on the thought object + serialization, not yet in the model input);
  the rule parser is lexicon-bounded (drops words on open-domain text — a
  parser-quality limit, not a serialization loss); meanings are mock.

**Stage B — real meaning (BUILT, default).** Words now resolve to real meaning
(`meaning.NSMMeaningResolver`): **prime → cited NSM molecule → WordNet-gloss
decomposition → SOMEONE/SOMETHING**. The parser was fixed first (PP/modifier
"pointer" edge-direction + subordinate/relative-clause + "to" disambiguation), so
content words stop dropping. Anti-hallucination is structural: `nsm_molecules.py`
holds 37 *cited* molecules with a `source` invariant and **zero fabricated
explications** (Goddard's explication papers were paywalled, so none were invented);
`wordnet.py`/`WordNetSenseInventory` give real senses/glosses/hyper-hyponym relations
(nltk); gloss decomposition derives only from real definitions; unknowns never
invent a prime. The semantic web also seeds **bootstrap LTM** (`bootstrap_memory.py`,
65 primes + 37 molecules + edges) that APPEND grows. No training regression
(held-out ~0.96 with real meanings). **Honest limits:** meaning is *coarse* — much
collapses to SOMETHING because molecules lack sourced explications and gloss
decomposition bottoms out; the model still ingests it as the additive zero-init
bag-of-primes (full meaning *trees* aren't fed to the model yet); "john"→WordNet's
toilet sense vs "mary"→SOMEONE shows first-sense/name handling is rough.

### 0f. TPR meaning vectors — non-flattening representation (prototype, MEASURED)
The flattening problem: meaning reaches the model as a capped 4-prime bag
(`meaning_prime_ids`), destroying the tree. The chosen remedy is **Tensor Product
Representations** (Smolensky 1990): bind each child (prime/molecule filler vector)
to a structural **role** (position × relation family, role families in orthogonal
subspaces — the "noun-axis × verb-axis" factoring) via outer product and sum;
unbinding inverts it. Molecules stay first-class units (no eager expansion);
vectors are **deterministic and unique** per structure. Verified against the
literature before building (user asked for a double-check):
- Pure TPR is exact but its dimension grows **exponentially with depth**; HRR-class
  compression holds dimension fixed at the cost of noise + cleanup (Plate).
- **TPR-RNN (Schlag & Schmidhuber, NeurIPS 2018)**: a fixed THIRD-ORDER TPR memory
  (entity⊗relation⊗value), trained end-to-end by gradient descent to SOTA on
  bAbI — our task family. "Infinite-depth thinking" = recursion **in time**
  (the loop), not unbounded tensor order. Trainability is TPR's strength: bilinear
  exact ops, clean gradients, no cleanup inside the training path.

**Prototype results** (`tpr.py`, `scripts/probe_tpr.py`, on REAL trees):
- **Matrix form (one exact level, d=128): 1.00 label recovery on every real
  explication tree** (gold kill/broke/sad/happy/children + DeepNSM
  snake/egg/water/dog/house, up to 31 children). This is the representation.
- **Fully contracted single d-vector is measurably lossy** (0.22–1.00 at d=128;
  0.41–1.00 at d=256 on the same trees) — a collapsed vector is NOT good enough as
  the model input; the matrix should be fed (or a learned projection of it).
- **Stacked contractions collapse** (depth-3 ≈ 0.05–0.23) and exact order-growth
  explodes (d=128 depth-3 = 1.1 GB) — both confirm: per-level exact matrices +
  depth-through-the-loop (the TPR-RNN shape) is the architecture.
**Gate verdict: PASSED for the matrix form.** Next integration step: a word's
meaning = its cached `encode_matrix` (d×d) of the explication tree; the thinking
memory (Stage D) takes the TPR-RNN order-3 form. The contracted vector remains
useful only as a cheap similarity key (e.g. LTM addressing), not as the meaning.

### 0g. Clauses as the unit; token-free TPR; entity-keyed cross-clause memory
The model should **not embed tokens**. The only atomic vectors are the ~65 NSM
**primes**; molecules/words/clauses are all *composed* by TPR binding (§0f). The
unit of thought is the **clause** (predicate + arguments — the parser already emits
it), and reasoning = **correlating clauses through shared entity-variables** (the
TPR-RNN `entity⊗relation⊗value` memory). A clause argument is either an **entity**
(a name/referent → a *variable*: a fresh atomic vector, NOT decomposed — NSM's
"someone X") or a **content word** (its explication → TPR). Built + verified
(`clause.py`, `scripts/probe_clause_tpr.py`, prototype, non-neural):
- "mary is in the kitchen" assembles into ONE d×d TPR matrix from `variable(mary)` +
  `TPR(kitchen)` bound to roles — **zero token embeddings** — and decodes back to
  SUBJECT→mary (cos 1.0) + PLACE→kitchen (4/4 primes).
- Cross-clause: "mary is in the kitchen . she went to the office ." → both clauses
  write `(mary, PLACE, ·)` into an order-3 `EntityMemory` ("she"→mary by recency);
  `query(mary, PLACE)` returns **office** (cos 1.0, updated) — two clauses correlated
  purely through the shared variable.
**Honest crux:** general **coreference** (keying entities that aren't explicit
names) is the real hard part — only recency/explicit-name is handled. And explication
**coverage** is now load-bearing: with no token embedding to fall back on, a content
word lacking an explication collapses to a bare prime.
**Decision gate PASSED.** Next = the model rebuild: replace `model.token_embedding`
with clause-TPR input, make Psyche's working memory the order-3 `EntityMemory`
(TPR-RNN), depth via the loop. Tokenization survives only as the parser's text reader.

### 0h. The token-free clause reactor — perception fixed, only REACTION learned (BUILT)
The model rebuild's core is built and trains (`entity_memory.py`, `clause_reactor.py`,
`scripts/train_clause.py`). **Perception is frozen/grounded** — each clause becomes a
deterministic `(entity, relation, value)` triple of TPR/prime vectors (no token
embedding). The **only learned parameters** (a ~160k-param GRU controller + heads)
decide *how to react*: a write **gate** into the order-3 `entity⊗relation⊗value`
memory, a **respond** weight (timing), and a **generated** response meaning-vector
scored *contrastively* against the fixed option meaning-vectors.

**Memory write is now two-gated** (`entity_memory.write(…, gate, overwrite)`):
`delta = gate·value − overwrite·old`. `overwrite≈gate` → the slot becomes `value`
(an UPDATE / recency, levels 3/6); `overwrite≈0` → `value` is *added* (a VOTE — repeated
assertions accumulate so the corroborated majority wins, level 5). The controller also
sees the clause **predicate** filler (`pred:is` vs `pred:moved`), the signal that
distinguishes a vote from an update. Both gates and the predicate are the levers that
let one shared memory serve corroboration *and* contradiction without interference.
Probed on the canonical update ("mary is in kitchen" → "mary moved to office"): the
trained controller fires **overwrite 0.37 on `is`** (vote) vs **0.91 on `moved`**
(replace), and the final memory reads `cos(office)=1.00, cos(kitchen)=−0.14` — the
two-gate works exactly as intended; the residual below is instance-level, not a
mechanism failure.

**Closing the gap (val 0.82 → ~0.88).** The dominant limiter was *not* the policy but a
**perception bug**: gloss decomposition kept only a content-word's root *label*, which
for store-resolved words is the generic wrapper `"EXPLICATION"`, collapsing distinct
words (`kitchen`≡`bathroom`) to identical meaning-vectors — an **oracle ceiling of
0.875** (58/240 episodes had a distractor >0.95 cosine to the gold). Attaching the
content word's actual (depth-bounded) **subtree** instead of its label lifts the oracle
ceiling to **1.000** (0 collisions, every `dim`). With that + the two-gate + predicate +
minibatched training the reactor reaches **train ≈ 0.94 / val ≈ 0.87–0.88** (resp-mass
at the question ≈ 0.90). Per-level val: L1 1.00, L2 0.83, L5 0.94, L6 0.93; the residual
is **L3 0.72 / L4 0.79** (bare-update recency and respond-before-the-trailing-move
timing) — genuine reaction-policy difficulty (parsing of "moved" → triple is verified
clean), not perception. **Not yet done:** the destructive removal of `token_embedding`
+ the token Psyche/dataset — still gated on fully closing the last ~0.08 to the token
model (~0.96), so we don't ship a regression.

**Roadmap (remaining):**
- **Close the gap, then hard-replace.** Improve the reactor (coverage, d, overwrite-
  vs-vote for corroboration) to match the token baseline, then delete `token_embedding`
  and the token path.
- **Coreference + coverage.** Real entity-keying for pronouns/definite NPs; source
  more molecule/word explications so content words ground richly, not to SOMETHING.
- **Stage C — WSD wired.** Wire `wsd.IterativeSenseResolver` into the loop: pick the
  sense (→ meaning tree) from `(state, memory)` context; write the sense-resolved
  meaning, not the surface token, into memory.
- **Stage D — tree→tree thinking model.** Add a decoder (first seq2seq here) that
  emits a thought object from `encoder(input thoughts) + memory`, read out via the
  reverse parser. Pin the exact output mechanism when reached.
- **Stage E (optional) — learned tree↔vector codec.** Only if explicit manipulation
  is too expensive; trained against the lossless explicit representation as oracle.

### 0i. Logical clause structure — coordinators relate clauses; store-as-OR then decide truth (PHASE A numpy gate + PHASE B trainable, MEASURED)
The next capability: **logical structure between clauses** — keep each clause as its
own *lossless* meaning-matrix and let the **coordinator** be what *relates* them, so a
contradiction/disjunction is **stored as an OR** (both kept, distinguishable) and a
later "decide truth" step resolves it, tagging the loser with a **FALSE adjective**
rather than deleting it ("overwrite but don't forget"). Phase A is a numpy gate
(`clause.extract_discourse` / `build_discourse_tpr` / truth-tagging / `DisjunctionBuffer`;
`scripts/probe_discourse_tpr.py`), no training.

**Parser reality (correcting the plan's premise).** The exploration assumed
`quantum_adapter` "preserves the full tree" — it does **not** for coordination. The raw
hypothesis represents `A or B` as two COORDINATION edges pointing *up* from the
coordinated elements (kitchen, office) to the coordinator (`or`); the adapter's
parent→child tree walk only descends, so the coordinated places are unreachable from the
root and **dropped** (which is why `extract_clauses` lost the PP). The fix: read
discourse from a new flat `quantum_adapter.HypGraph` (every node + every typed edge),
exposed via `ParserInputEncoder._parse_graph`. The parser DOES emit what we need:
COORDINATION edges (or/and/but), a `MODIFIER 'not'` node for negation, and SUBORDINATION
for because. `extract_clauses` and `hypothesis_to_tree` are untouched (regression-safe;
185→193 tests green).

**Grounding.** NSM has no AND/OR exponent — disjunction *is* **MAYBE**; coordinators
ground natively (`or→MAYBE`, `not→NOT`, `because→BECAUSE`, `if→IF`). Each disjunct is a
verbatim `clause_tpr` d×d matrix (nothing summed across disjuncts); the coordinator
binds `role(i,COORDINATION)⊗contract(clause[j])` plus the connective atom on a reserved
CONNECTIVE role, so the OR is itself readable. Truth tags use a local 3-atom codebook
{TRUE, FALSE, MAYBE}: `tag(M,v)=M+role(TRUTH)⊗filler(v)`, recovered by unbind+cleanup.

**Gate result (`probe_discourse_tpr.py`, dim 256, PASS ✅).** On the real parse of
"mary is in the kitchen or the office .": 2 distinct clauses + 1 OR link extracted; the
OR link recovers the related clause index (j=1) and the connective reads **MAYBE (cos
1.00)**; both disjuncts store tagged MAYBE; **unresolved query → MAYBE** (first-class
uncertain answer). Ingesting "mary is not in the kitchen ." re-tags kitchen **FALSE** /
office **TRUE**, and **both clauses stay recoverable** (the FALSE disjunct is not
forgotten); **resolved query → OFFICE**.

**Honest caveat (fidelity is cleanup-level, not raw-cosine).** The plan promised "clause
matrices round-trip cos 1.0"; the operative fidelity is **unbind + nearest-neighbour
cleanup** (exactly how `decode_clause` and the buffer already read values), which is
correct for every disjunct at every dim. A *bare* unbind carries cross-talk — the
codec's per-relation **±1 sign diagonals** make different-relation roles non-orthogonal
even on distinct columns — so raw cosine is only a diagnostic (kitchen 0.68 vs runner-up
0.66 at d=256, widening to 0.80 vs 0.79 at d=512; office 0.97). The thin kitchen margin
is the genuine risk to carry into Phase B; mitigations are higher `d` or orthonormal
(sign-free) clause roles. `decide_truth` itself is exact — it compares *stored* (not
unbound) value vectors. Storage is O(#disjuncts·d²) (a buffer of matrices); fine for the
demo, does not scale to a chapter (the contracted keys are the only scalable part; d=1024
already OOMs the d²×d contraction matrix).

**Phase B — wired into the trainable reactor (DONE, the logic is learned emergently).**
The capability now trains end-to-end with **no auxiliary truth loss** — only the
contrastive answer loss, on a curriculum where the logic is *necessary*:
- **Curriculum** (`episode.py`, max_level 6→8). L7 disjunction: half the episodes are
  left UNRESOLVED ("X is in the A or the B." → answer **maybe**, the NSM atom — a
  first-class uncertain answer); half are RESOLVED by a negation ("…not in the A." →
  answer B). L8 negation-removal (assert A, assert B, "not in B" → answer A; recency
  would wrongly say B). The option set always contains `maybe`, A and B, so guessing
  cannot win.
- **Perception** (`clause_reactor.build_clause_batch`) now streams through
  `extract_discourse` on the flat graph: a disjunction emits one step per disjunct
  (carrying the **OR/MAYBE atom** on a new `coord` channel → the controller VOTEs them
  into the same slot, superposed), a negation emits a step carrying the **NOT atom**.
  Plain facts carry a zero coord (identical to the old single-triple path; L1-L6 triples
  verified unchanged, incl. "moved to"). The `maybe` option grounds to `filler_vec(MAYBE)`.
- **Model.** A `coord` channel feeds the GRU; a **`decide_truth`** head
  (`Linear(hidden+d,1)`) outputs a per-step *refutation* strength that makes the value
  write **negative** (`value_gate = write − decide_truth`), so a NOT step *subtracts* a
  previously-voted value — "A or B" superposes both, "not A" cancels A, query → B. No
  separate buffer: the existing order-3 `entity⊗relation⊗value` memory holds the
  disjunction as a superposition and the negate-write resolves it (simpler than a parallel
  buffer and keeps `entity_memory` parameter-free/green; the lossless multi-matrix buffer
  stays the numpy Phase-A artifact). Unresolved → the response generator reads an
  *ambiguous* superposition (+ the coord-primed state) and learns to emit MAYBE.
- **Result** (`train_clause.py`, dim 48, 480 eps, 80 epochs; emergent, answer-only):
  overall **val ≈ 0.86** (levels 1-6 broadly hold). Per-level val: the new logic works —
  **L7-unresolved→MAYBE 1.00, L7-resolved-by-negation 0.86, L8 negation-removal 0.70**;
  L1 0.92, L2 0.89, **L3 0.91** (up from 0.72 — the graph perception + extra heads help),
  L4 1.00, L5 0.80, L6 0.75 (vote/overwrite dip slightly — the negate head mildly competes;
  an honest tradeoff, not a regression in mechanism). A unit overfit confirms the answer
  loss *alone* teaches `decide_truth` to resolve disjunctions (acc → 1.0 on the
  logic-necessary episodes). Tests 193→197 green.

**Honest scope / next.** L8 (0.70) is the weakest — subtracting exactly the named value
without disturbing the slot is delicate; the per-disjunct truth labels are scaffolded
(metrics-only) as a fallback aux loss if the emergent signal needs strengthening.
Conditionals (SUBORDINATION/`if`-then) are deferred (the parser emits the SUBORDINATION
edge cleanly; only the modus-ponens reaction is unbuilt). The order-3 superposition is
lossy for >2 disjuncts (fine for the binary curriculum); the lossless path is the numpy
buffer. Still pending from §0h: the destructive `token_embedding` removal.

### 0j. ClausePsyche — a graph of meaning-objects; collapse/expand IS definition; operators are nodes (MEASURED)

The successor to §0i. Memory is a **graph** (`meaning_graph.py`), not a superposition: a
clause is a distinct node carrying a *lossy vector handle* **and** a *lossless serialized
structure* (`serialize_thought`). The unifying operation is **collapse/expand =
definition**: `collapse` files a structure and returns a handle; `expand` dereferences it
back to the exact structure. Losslessness lives in the **stored structure, never in the
vector** (fixed-dim vectors can't be invertible) — the vector only *addresses*.

- **Collapse/expand (`collapse.py`, probe_collapse).** Exact round-trip **100%** at d=256
  and d=512 (the hard gate). Vector dereference splits by kind: **CONCEPT 7/7 top-1**
  (noise-robust at +5%), thin margins (median **0.034 @256, 0.016 @512** — recorded, not
  assumed); **CLAUSE handles alias** on the shared `SOMEONE` variable label (entity
  identity is a *variable token* in the structure, not in the label-based handle), so
  clauses are addressed by graph edges + the exact path, never by handle similarity. This
  is the honest face of the §0i cross-talk caveat: correctness is the structure, the
  vector is a shortcut.
- **Operators are NODES, not flags (`apply_operator`/`read_operator`, probe_operators).**
  An operator binds its clause-argument on a reserved orthonormal role
  (`role_vec(i, "OP_ARG")`), exactly like `tag_truth`. Deconvolution gate: NOT/MAYBE read
  back at **score ~0.99**, clause argument recovered at **cos ~0.99**, the exact clause is
  reachable via the `OPERATES_ON` edge, and the wrapped clause vector is **untouched**
  (`np.allclose`) — the recoverability a flat `List[SubType]` flag (old `quantum_parser`)
  could never give. The arg is bound *unit-normalized* (contracted handles have tiny norm
  and would otherwise be swamped by the operator-label term).
- **STM/LTM read-time resolution (`clause_psyche_graph.py`, probe_psyche_graph).** Distinct
  clauses **share one referent** (co-reference, not duplication); contradictions are kept
  and resolved at *read* time — FALSE (negated) clauses drop, an affirmed clause wins by
  recency, an unresolved disjunction is MAYBE, a disjunction narrowed to one survivor
  resolves. All symbolic, no training: **L8** (kitchen/office/not-office → kitchen),
  **L3** recency "moved" → office, **L7a** unresolved → MAYBE, **L7b** narrowed → office.
- **Neural ClausePsyche (`clause_psyche.py`, train_clause_psyche).** A sibling of the §0h
  reactor (kept as baseline). Same fixed grounded perception; the GRU hidden is the
  carried **consciousness state**; an op-routing head is present; response **generators**
  emit factored fillers assembled by fixed binds into a **d×d clause matrix** — a generated
  meaning-object, not a 1-of-4 pick. Trained by **Frobenius to the gold clause matrix +
  decode-CE + consistency** (no MC answer head). Result (240 eps, 80 epochs, d=64): frob
  2.87→0.08, **val clause-decode 0.77**, and the hard logical levels are solved —
  **L8 1.00, L7-resolved 1.00, L7-maybe 1.00** (L2 0.83, L6 0.86; L1/L3 weaker on tiny val
  counts). End-to-end probe: the generated matrix unbinds back to the correct place at
  **0.88** on held-out L8. The genuine residual is **dereferencing margin** (thin concept
  margins above) — mitigated by always routing correctness through the exact stored
  structure. Tests 197→221 green; `ClauseReactor`/`train_clause.py` untouched.

**Deferred:** deep recursive collapse / learned handle embeddings; full co-reference (still
recency/explicit-name); conditionals (`if`-then); learned trust/corroboration; text surface
realization (the deliverable is the decoded *meaning object*). LTM is minimal (consolidate
affirmed facts; pruning deferred). Still pending from §0h: the `token_embedding` removal.

### 0k. Emergent reasoning in the ClausePsyche loop — modus ponens / transitivity / abstain (MEASURED; honest negative result on hops)

The north star: reason so the model *believes* its answers (correct ≡ derivable from the
in-context premises), and know *whether* to respond (abstain when it cannot derive). Built on
the §0j ClausePsyche graph; reasoning is meant to **emerge from the neural loop**, not a
symbolic engine (the symbolic `reasoning_oracle` only generates/grades).

- **R1 curriculum + oracle (`reasoning_oracle.py`, `episode.py` L9-L11).** Self-contained
  episodes whose answer is *never stated* — only derivation reaches it: **L9** conditional
  (modus ponens), **L10** transitivity/inheritance (is-a chain), **L11** unanswerable →
  abstain (`idk`). A tiny forward-chainer derives the gold answer + chain + `answerable` flag.
  Gate: the oracle derives L9/L10, flags L11, and the gold answers are **not directly
  retrievable** (retrieval/recency fail — derivation is necessary).
- **R2 perception (`clause_reactor.py`).** Reasoning streams ground from the oracle's
  *structured premises* (the same meaning the sentence carries; the derived answer is never
  input) — a conditional streams antecedent+consequent with the **IF atom on the coord
  channel**, is-a/CAN facts stream with a general relation vocab. This sidesteps fragile
  parsing of "if/then" and "is-a"; L1-L8 stay on the parser path. `ClauseBatch` gains an
  `answerable` flag; `idk` grounds to its own atom.
- **R3 model (`clause_psyche.py`).** After ingesting the stream (facts written to the order-3
  STM), the controller runs **K inference hops**: each queries the STM with a state-derived
  key, updates the consciousness state, and may **write a derived `(entity,relation,value)`
  back into the STM** (a materialized intermediate belief). A consciousness-state **ABSTAIN**
  head gives a first-class "I don't know" — the state's first real job. `hops=0` = the
  single-pass baseline. Loss = Frobenius + decode-CE on **answerable** episodes + an **abstain
  BCE** (`should_abstain = 1 − answerable`) so the model is trained to respond only with what
  it can derive.

**Results (val, L1-L11, answer-only / no chain supervision).** Single-pass **hops=0** (d=64,
140 ep): **val 0.90** — **L10 transitivity 1.00, L8 negation 1.00, L11 abstain 0.83**, but
**L9 modus ponens 0.43**. **hops=3** (d=48, 120 ep): val 0.59, L10 1.00, L9 0.20. End-to-end
probe: L10 answers "bark" with its justification chain, L11 abstains correctly, **L9 wrongly
abstains**.

**Honest negative result — the explicit multi-hop loop does NOT beat the single-pass on these
short chains** (and underperforms at small scale: extra hop params, harder optimization). A
unit 2-hop task confirms it (single-pass and hops both hit 1.0): a GRU + order-3 memory already
*composes* a 2-hop chain in one pass. The loop's theoretical payoff is **depth**, which a
2-hop curriculum never exercises — so emergence-via-looping is *plausible but unproven here*;
deeper-chain curricula are the real test. The reasoning + abstain **mechanism works and emerges
in both** variants.

**The genuine crux is L9.** L9 (answerable) and L11 (unanswerable) are *structurally identical*
conditionals — both show "if in A, can see X"; they differ only in whether the stated place
matches the antecedent. The model conflates them and **over-abstains** (abstain precision ~0.56),
dropping answerable L9s. Modus-ponens *antecedent-gating* is the part that does not emerge from
the answer-only signal; the oracle's `gold_chain` is scaffolded as the aux-supervision fallback
to switch on next. **The consciousness state got its first functional job** (the abstain/whether
gate) — a partial close of §1. Deferred unchanged: deeper chains, real (non-in-context)
knowledge, min-dimension/WSD, the §0h token-path removal.

### 0l. Reason-until-confident — adaptive halting (the consciousness state decides *when* to stop)

Per the user: don't fix the number of thinking steps — **reason until confident it either KNOWS
or CANNOT know**, then answer or abstain. This gives the consciousness state a second job (when
reasoning is done) and makes *when* to answer emergent (it was previously read off the fixed last
hop; *whether* to answer was already emergent via the abstain head).

- **Mechanism (`clause_psyche.py`, `halting=True`).** ACT-style (Graves 2016): each hop the state
  emits a halt probability `sigmoid(halt_head)`; halting mass accumulates and the loop stops when
  confident or at a **max-hops** safety cap; the answer/abstain reads off a **halt-weighted
  settled state** (differentiable, no hard branch). A small **ponder cost** rewards stopping as
  soon as it is sure. "Confident it can't know" = a confident abstain.
- **Result (d=48, max-hops 5, 110 ep, L1-L11).** val **0.67** — *below* the single-pass 0.90 and
  the fixed-hop runs: on these short chains the halting adds optimization burden with no accuracy
  win. L10 transitivity 1.00, L9 modus ponens **0.40** (still the weak spot), L11 abstain 0.50.
  **Think-steps mostly collapsed to ≈2.0** (the known ACT failure mode) **but with a real, faint
  adaptive bump on exactly the if/then conditionals: L9 = 2.4, L11 = 2.3 vs 2.0 everywhere else**
  — it ponders longer precisely where it is uncertain (the conditionals), which is the intended
  behaviour, weakly. e2e: it answers an L9 correctly, abstains correctly on L11, and per-example
  step counts vary (some hit the cap).
- **Honest verdict.** The "reason-until-confident" loop is built and functional, *when*-to-answer
  is now emergent, and the consciousness state owns the halt/abstain decision (further §1
  progress). But on this short curriculum it neither improves accuracy nor yields crisp
  variable-depth (semi-collapsed). It needs **deeper tasks** (where stopping early actually
  matters) and/or the scaffolded **aux chain-supervision** (for L9 modus ponens) to pay off — the
  short 2-hop curriculum simply does not reward adaptive depth. Tests: +3 halting (forward ponder
  bounds, still-learns, stops-before-cap); `halting=False` keeps the fixed/single-pass baselines.

### 0m. Held-out diagnostic — the loop is NEEDED for depth, but does not yet LEARN depth

After §0l (halting collapsed on short chains), we built deep variable-length chains (L12/L13)
and ran a clean held-out diagnostic: **fixed relation vectors, fresh held-out entity vectors,
two distractor chains** (so neither memorization nor "grab the lone answer" can succeed). Two
competing is-a chains end in different abilities; the question targets one chain's root; depth
1 vs 3; train and eval entities disjoint. Result (d=28, hidden=80, ~700 it):

```
single-pass  depth-1 acc=0.60            depth-3 acc=0.01
halting(6)   depth-1 acc=0.65 steps=2.0  depth-3 acc=0.26 steps=2.0   (chance=0.25)
```

**Two findings, both important:**
1. **The loop is genuinely necessary for depth.** Single-pass depth-3 = **0.01** on held-out
   (below chance) — one forward read = one memory lookup, so a 3-hop chain is unreachable in a
   single pass. (Earlier "single-pass solves depth-3" was pure **memorization**; held-out kills
   it.) This validates the whole premise: deep chains are real reasoning, not retrieval.
2. **But the loop does not yet learn depth either.** Halting depth-3 = **0.26 ≈ chance** — the
   hop loop barely beats single-pass and does not learn a *generalizable* multi-hop traversal.
   Steps still collapsed to 2.0. The bottleneck has **moved**: it is no longer the curriculum
   (now shortcut-free) or the halting schedule — it is the **hop mechanism itself**.

**Diagnosis of the hop mechanism.** Each hop derives its query keys `(q_ent, q_rel)` from the
GRU **state** via fresh `Linear`s. To traverse `N0 →IS N1 →IS N2 →IS N3 →CAN ans`, hop k+1 must
query with the **entity just read at hop k** — an arbitrary held-out vector. A from-scratch
`Linear(state)` does not learn to reproduce that read vector, so the chain does not propagate.
This is the classic hard core of emergent multi-hop (bAbI): the loop needs an **iterative
inference module that queries from the previous read**, not a state projection. Fix is
**architectural** (chain the next query off the last read; cf. TPR-RNN inference hops), plus
more capacity/data — not more puzzles or halting tricks. Caveat: the diagnostic is small
(d28/hidden80/700it); numbers are suggestive, not definitive, but the direction is clear.

**Status:** the deep curriculum (L12/L13) is now a *sound* shortcut-free benchmark; the open
problem is making the loop actually traverse it. Halting/abstain remain built and green.

### 0n. "Read its own output" cracks multi-hop reasoning (held-out)

§0m's bottleneck: each hop derived its query entity from the state via a `Linear`, which could
not re-conjure an arbitrary held-out entity vector, so the chain never propagated. The fix
(user's idea): **the value just read becomes the next entity to look up** — the loop reads its
own output. Each hop now queries memory with `focus` (initialised to the question entity,
updated to `mem_read` each step); only the *relation to follow* is decided from the state (a
choice among a few fixed relation atoms — the easy part). Removed `q_ent` and the write-back
heads (`clause_psyche.py`); traversal is now read-only focus-chaining.

**Definitive held-out result** (fixed relations, fresh entities, distractor chains, d=24-32):

```
single-pass (0 hops)            depth-3 held-out acc = 0.00   (one read = one lookup)
focus-chaining + 4 fixed hops   depth-3 held-out acc = 0.97   <- generalises to unseen entities
```

A depth-3 chain (3 is-a links + 1 "can") needs exactly 4 hops; given them, the loop **traverses
the chain and generalises** — the core multi-hop reasoning capability, achieved, and it is the
mechanism (0 vs 4 hops of the *same* model) that makes the difference, not init luck (single-pass
fails depth-3 at ~0.00 across every run). The entity-conjuring problem is gone.

**Remaining bottleneck — halting calibration.** When the model picks its *own* step count
(`halting=True`), it stops at ~1 step (avg 1.1) and fails depth-3: it has not learned to stop at
the *end* of the chain (after following "can"), which varies with depth. Because focus-chaining
puts the answer at the exact hop that follows "can", under/over-stepping misses it — so halting
must learn "stop when I've reached an answer/ability", e.g. direct halt-supervision toward the
chain depth (the scaffolded Step-4 lever) or a "read matches an answer option" stop signal. The
*reasoning* works; the *when-to-stop* is the open piece. Tests: 10 clause_psyche green
(focus-chaining is backward-compatible). Caveat: small scale (d≤32); the relation-to-follow
choice and halting are what remain.

### 0o. Thinking in meaning objects + attention over thoughts (job 2: when to stop)

Two upgrades, per the user: (a) the loop should think in **meaning objects** (the structured
clause it answers with), not the bare value vector §0n fed back; (b) it needs to decide *when to
stop* (job 2 — §0n's halting collapsed to ~1 step).

- **The loop now generates a thought each step and reinputs it** (`clause_psyche.py`): the working
  thought is `(subject, relation, value)`; `subject ← the value just read` (focus-chaining, sourced
  not conjured), `relation ← q_rel(state)` (the choice), `value ← read from STM`.
- **"Stop" = pick which thought is the answer.** A hard ACT halt — even after the gate was given
  the read value — **robustly collapsed to ~1 step** (held-out depth-1/3 ≈ 0.35, ≈ chance). So the
  hard halt was replaced by **soft attention over the model's own per-step thoughts**: `out_head`
  scores each thought ("is this the answer?", seeing its value), a softmax selects the answer
  thought, and the reasoning depth = where the attention lands. Differentiable; cannot collapse.

**Held-out result, model choosing emergently** (fixed relations, fresh entities, distractors):

```
                       depth-1   depth-3
hard ACT halt           0.35      0.36     (~chance; bails at step 1)
attention over thoughts 0.84      0.56     <- emergent selection recovers reasoning
(ceiling, fixed 4 hops)  --       0.97
```

Emergent depth-3 went from chance to **0.56** (ceiling 0.97 with hand-set hops). **Open:** the
selected-thought depth does not yet strongly scale with chain depth (avg ~1.6 steps for both), so
"how deep it thinks" is weak, and depth-3 trails the fixed-hop ceiling. Next lever: supervise the
thought-selection toward "the read that matches a candidate answer" (output when the thought is
answer-shaped) — structural, no oracle needed. The *reasoning + answer-selection* work emergently;
*scaling depth to the chain* is the remaining piece.

### 0p. Produce-answer decision + asymmetric reward (PonderNet), and the cold-start

The §0o attention readout *blended* an answer out of every step — it never actually decided to
*produce* an answer. Per the user, the right design: loop until confident, take a discrete
**produce-answer** step, and **read the answer only at that step**; train it with the asymmetric
reward **(++correct, −−wrong, −abstain)** so a cheap shallow guess (heavily punished) is worse
than "I don't know" (mildly punished) — forcing the model to walk until confident or abstain.

- **Implemented (`clause_psyche.py`):** a PonderNet halting distribution `pi_k = h_k·Π_{j<k}(1-h_j)`
  over produce-at-step, with abstain = `Π(1-h_k)` (the "never produced" mass). The answer is the
  halting-weighted produced thought; `compute_clause_psyche_losses` gains the asymmetric-reward
  branch (`r_correct/r_wrong/r_abstain`) used whenever `halt_dist` is present.
- **The asymmetric reward ALONE collapses** (held-out depth-1/3 ≈ chance, 1 step): a **cold-start**
  — to learn to walk deep, deep steps need gradient; but the halting only sends gradient where it
  already lands; step 1 grabs all the mass and the deep steps get *zero* gradient, so the walk
  never starts. (Fixed-hop works precisely because every step gets full gradient.)
- **The documented fix — a PonderNet exploration prior** (KL of the produce-distribution toward a
  geometric) — nudges it to *sometimes* produce later, giving deep steps gradient. **This escapes
  the collapse:** held-out accuracy climbs instead of flat-lining:

```
                          depth-1        depth-3
asymmetric reward only    ~chance(1 step) ~chance(1 step)   <- cold-start collapse
  + exploration prior     0.19 -> 0.52    0.24 -> 0.44      <- climbing @1500 iters, still rising
(attention blend, 0o)     0.84            0.56              (higher, but blends — does not decide)
```

**Status, honest:** the user's reward design is sound and the *produce-answer decision* now works
(escapes collapse, learns emergently), but it is (a) **undertrained** (still climbing at 1.5k
iters) and (b) currently **below the attention blend's numbers**, and (c) the produced-step does
**not yet scale with depth** (the prior pins it ~2.5). It is the *faithful* architecture (a real
decision, not a blend); maturing it needs more training + prior annealing (strong prior early for
exploration, decayed so the reward can shape per-example depth). Tests: 10 clause_psyche green in
the reward+prior regime (`w_prior`).

### 0q. The `mind/` build (M0-M3): meaning-space substrate + learned controller with teacher op-trace supervision (MEASURED)

A cohesive `nsm_ct/mind/` subpackage realizes `MIND_ARCHITECTURE.md` — one substrate, one controller
family — *importing* the gate-tested primitives rather than re-implementing them. Built and gate-tested
in four milestones:

- **M0 — schema freeze.** `mind/schema.py` pins the one meaning-object contract (node kinds, typed
  edges, the 5 operator-nodes, ~62 primes, the grammatical relations, and the *separate* reasoning-
  relation vocabulary). Meaning-object round-trips via `collapse`/`expand`.
- **M1 — knowledge layer.** `mind/knowledge.py` (`KnowledgeGraph`) stores facts, taxonomy, and
  **variable-bearing Horn rules as graph data**; `mind/persistence.py` gives exact disk save/load
  (closes the in-memory-only gap). `derive()` lifts the oracle's unifier over **graph-resident** rules
  and reproduces the oracle's L9/L10 answers. (Caught + fixed a real bug: `rules()` collapsed rules
  that share a name, e.g. L13's `mp` — each rule now carries a unique id.)
- **M2 — deterministic executor (the VM).** `mind/ops.py` + `mind/executor.py`: the cognitive
  instruction set {PERCEIVE, RECALL, INFER, CONSOLIDATE, SUPERSEDE, RESPOND, HALT} as deterministic
  ops over a live `STM` (reusing the §0j read-resolution) + the M1 LTM. `INFER` = focus-chaining
  realized exactly via `forward_chain` (its `DerivStep` chain *is* the unrolled traversal; the trace
  is faithful by construction). **Gold op-traces solve L7-L11 with abstain on L11; the probe scores
  230/230 (1.00) over generated L9-L13**, incl. deep is-a chains and chained modus ponens.
- **M3 — learned controller + teacher op-trace supervision.** This turns on the `gold_chain`
  aux-supervision §0k/§0n/§0p deferred. The proven focus-chaining controller (`clause_psyche.py`) is
  surfaced as a typed op-emitter (`mind/controller.py`); `mind/teacher.py` produces, per episode, the
  gold op-trace (replay-validated through the M2 executor → oracle answer), the gold relation-to-follow
  path (generic **BFS over the streamed memory edges**), the gold depth, and the answerable flag;
  `mind/controller_losses.py` adds **relation-to-follow CE + halt CE under a soft→discrete anneal**.
  Decided runtime story: the **learned vector loop is the runtime reasoner**; the M2 symbolic executor
  is **teacher + validator only** (never in the runtime answer path).

  **Measured (small held-out, the honest caveat is scale — 30 val episodes):** the controller learns to
  **emit the teacher's op-trace** — held-out relation-to-follow match **0.94** (fixed-hop) / **0.83**
  (halting). Turning on **halting** (the configuration the halt-CE targets) improves exactly the pieces
  flagged as not-emerging-from-answer-only, over the `halting=False` run: **L12 multi-hop 0.33→0.57,
  L11 abstain 0.40→0.86, L9 modus ponens 0.00→0.50, abstain P/R 0.40/0.40 → 0.64/0.69** (L10 stays 1.00).
  **Honest negative:** it does **not** reach the §0n 0.97 multi-hop ceiling here — `halting=False` with
  *fixed* hops > chain depth **overshoots** the answer (focus walks past it), and the halting/reward
  regime is undertrained + a touch unstable at this scale/iters (rel-match dipped mid-run), exactly the
  §0p maturation profile. The **mechanism + supervision are validated** (op-trace match high; the
  targeted levels all move the right way); reaching the ceiling is a **scale + iterations + anneal-
  stability tuning** task (more data, more steps, prior annealing), not an architectural gap.

  Unit gates green: teacher correctness (gold traces replay to oracle answers), BFS relation paths,
  read-encoder recovery, controller forward, relation-to-follow imitation overfit (>0.9). Full CT suite
  **262 passed** (the additive `clause_psyche.py` change does not regress `test_clause_psyche.py`).

- **M4 — the two loops over one substrate, and the engine that matures M3.** `mind/conscious_loop.py`
  (RECALL from LTM → run the M3 controller → answer + faithful op-trace → write-back; with a symbolic
  validator) and `mind/subconscious_loop.py` (consolidate STM→LTM; **offline INFER** = forward-chain over
  LTM and materialize derived facts as *direct* facts; **self_train** = replay + freshly-generated
  episodes train the controller with the M3 teacher loss, accumulating iterations across rounds, anchored
  by the oracle). Firm deterministic gates (5, green): **teach-once with no weight update** (a graph
  write is recalled with the controller's weights byte-identical — the core invariant, *in code*);
  **offline-infer makes a multi-hop chain a direct 1-hop fact**; STM→LTM consolidation; conscious-loop
  answer + faithful trace + validator agreement.

  **Measured — M4 matures M3 (the headline):** the one-shot M3 run topped out ~0.57 (L12) / 0.71
  overall on ~75 train episodes. Running the **subconscious self-train/replay loop**, held-out decode
  climbs monotonically across rounds — **0.41 → 0.49 → 0.46 → 0.54 → 0.80 → 0.88** (op-trace match
  holding ~0.95+), still rising when the run hit the wall-clock cap at round 6/12. So the scale +
  iterations the subconscious loop supplies is exactly what M3 needed: **decode reaches ~0.88 held-out**,
  a large gain over the one-shot number, with no architectural change — just the loop running. (The
  offline-infer→direct-fact benefit is gated separately by unit test; this driver isolates the self-train
  trend and does not consolidate into LTM, hence `LTM facts=0` in its log.) Full CT suite **267 passed**.

- **M6 — the native language membrane (talk + explain itself).** `mind/membrane.py` is a deterministic,
  **owned, LLM-free** `text <-> meaning` membrane over the curriculum's controlled grammar: `render`
  reproduces the exact curriculum sentences (replacing the leaf-concat `reverse_parser` seed), `parse`
  is the template-inverse — over the existing `(subject, relation, value)` meaning objects + truth/
  operator tags. `mind/verbalize.py` does **"respond with thinking"**: it renders the *actual* `DerivStep`
  provenance the M2/M4 trace already carries (pruned to the answer-relevant path), so the "because ..."
  is the real derivation, not a story. Gates (5, green, no training): render-matches-curriculum;
  cycle-consistency both directions (meaning->text->meaning exact; >90% of curriculum text reproduces
  verbatim, the rest — `moved to`, name-`is a` — normalize to the same meaning); faithful verbalization
  (the stated chain == the executor's support chain). `scripts/talk_to_mind.py` demos it end-to-end —
  e.g. "a beagle is a dog / a dog can bark / what can a beagle do?" -> "Because a beagle is a dog and a
  dog can bark, a beagle can bark. A beagle can bark." — and a deep L12 chain verbalized as its actual
  multi-hop path. Open: open-domain parsing (the `quantum_parser` upgrade) and a learned decoder
  cold-started on the renderer's pairs (the renderer is the corpus). Full CT suite **272 passed**.

- **M8 — a real broad dataset (ProofWriter), step 1: the substrate reproduces its gold logic.**
  ProofWriter (Allen AI; RuleTaker successor) is this architecture's task at scale: facts + Horn rules
  + a query -> True/False/Unknown, open-world (Unknown = derive-or-abstain), depth 0-5, negation,
  arbitrary predicates/entities — far broader than the 7-relation toy curriculum. `mind/datasets/
  proofwriter.py` parses its structured `representation` form into **4-tuple literals**
  `(subject, predicate, object, polarity)`; because `reasoning_oracle.forward_chain`/`_unify` are
  arity-generic, the existing engine is **reused unchanged** with polarity as the 4th element (the
  universal variable "someone"/"they" -> the oracle's `?x`). `verify()` labels a query TRUE if the asked
  polarity is in the forward-chain closure, FALSE if the opposite is, else UNKNOWN.

  **Measured (the trust anchor): forward-chain reproduces ProofWriter OWA gold at 0.989** over a
  multi-depth sample — True 1.00, False 0.99, Unknown 0.98, and **0.98-0.99 at every depth 0-5**. So the
  reasoning substrate is **not toy-curriculum-specific**: it matches a real, broad published dataset
  out of the box. (One bug caught: the query carries its own polarity — ProofWriter asks both "X is kind"
  and "X is not kind" — fixing `verify` to honor it took False from 0.07 -> 0.99.) Gates: parser + verify
  unit tests (data-free) + a real-data parity gate (skips if the git-ignored download is absent).
  `scripts/fetch_proofwriter.py` reproduces the data; `scripts/probe_proofwriter.py` reports parity by
  depth. **Step 2 (open): training the learned controller on ProofWriter** needs a *verification* readout
  (options {true,false,idk}, abstain=Unknown) + a polarity/variable-rule perception — a real adaptation
  of the value-generation controller, not a drop-in. Full CT suite green (parity gate included).

- **M8 step 2 — training the learned controller on ProofWriter (verification mode; honest baseline).**
  Verification = 3-way MC: options are the answer atoms `{true, false, idk}` (Unknown is an option,
  not abstain), so the controller's existing contrastive head is reused; `proofwriter.build_pw_batch`
  streams facts + rules (IF-coord, variables as atoms) + the full query triple (polarity folded into
  the value atom `v:+:kind` / `v:-:kind`). Trained (answer-only) on real OWA depths 0-2, minibatched.
  Compute note: the order-3 entity memory is `[b,d,d,d]` (cost grows as d^3); d=64 was ~6.4s/minibatch
  (intractable on CPU), d=32 ~0.8s.

  **Measured (honest negative):** the small CPU-trained controller barely beats the majority-class
  baseline — held-out **~0.50 vs 0.457 majority** (peak 0.515 @ epoch 15), roughly flat across depths
  (d0 ~0.53, d1 ~0.45, d2 ~0.50). The **pipeline trains end-to-end on real broad data**, but the
  *learned* controller does **not** crack ProofWriter at this scale — a stark gap from the **symbolic
  substrate's 0.989** parity. This sharpens the M3 lesson: **answer-only supervision is insufficient**
  (M3 needed the teacher's gold relation/op-trace to reach 0.82-0.90). Next levers, in order:
  (1) **proof-chain teacher supervision** — ProofWriter ships gold proofs; supervising the hops with
  them is the M3-proven fix; (2) **scale** (d/data/epochs; RuleTaker needed a fine-tuned transformer on
  full data); (3) a dedicated verification head instead of the value-generation->cosine bottleneck.
  Gates: parser/verify unit tests + a `build_pw_batch`->controller-forward gate (7 tests).
  `scripts/train_proofwriter.py` (periodic depth-graded eval, checkpointed).

- **M9 — proof-chain teacher supervision on ProofWriter (MEASURED; robust honest negative).**
  Applied the M3-proven lever: our `forward_chain` (0.989 vs gold) is the proof teacher
  (`proofwriter.proof_path`/`value_codebook`/`proof_supervision` backward-trace the chain to the query
  for the ordered **derived-value** sequence). ProofWriter is *attribute rule-chaining* (same entity,
  relation almost always `is`, the VALUE changes furry→kind→smart), so — unlike M3's relation-to-follow
  — the per-hop target is the derived **value**: a `gen_derive` head scored against the value codebook
  (`controller_losses.value_supervision_loss`), plus a `derive_chain` mode that chains the loop on its
  own generated derived value (focus ← `gen_derive(state)`) so the supervision is **causally on the
  answer path**.

  **Measured across three configs (d=32, depths 0-2, 996 train, 30 ep, CPU):** answer-only **0.50** ·
  decoupled-aux-head teacher **0.48** · derive-chain teacher **0.458** (peak 0.505 @ ep20, then
  declines). **All flat at the 0.457 majority baseline.** The first negative looked like a placement
  bug (aux head off the answer path); fixing that (derive-chain) did **not** help — so it is **not** a
  supervision-placement issue but a **ceiling**. Diagnosis: (1) **scale** — this is a ~10^4-param model
  on 996 examples / 30 CPU epochs; RuleTaker/ProofWriter was cracked by **fine-tuning RoBERTa-large
  (3.5×10^8) on ~500k**; (2) **architecture fit** — the order-3 memory + focus-chaining was proven for
  *entity traversal* (robin→bird→animal), but ProofWriter is *Horn-rule application with variable
  binding over many predicates*, a different computation the loop doesn't express at this size, even
  with the derivation on the answer path; (3) supervision is sparse (279/996 derivable) but the
  supervised items don't lift either — pointing at capacity, not density. **Honest standing:** the
  **symbolic substrate generalizes to real broad data (0.989); the learned controller does not yet**,
  and the evidence says the remaining lever is **real scale (GPU, full data, a larger controller)** —
  not another supervision tweak — or accepting the symbolic engine as the runtime reasoner for broad
  combinatorial reasoning. M3's 0.82 stands for *regular, small-vocabulary* worlds; it does not transfer
  to ProofWriter's breadth on a CPU budget. Gates: proof-teacher correctness + value-loss + `derive_chain`
  forward (18 tests). Code lands default-off (`derive_chain=False`), so the §0n curriculum path is
  untouched.

- **M10 — the controller DRIVES the symbolic engine (the decisive correction; MEASURED WIN).**
  User-identified root cause of the M8/M9 ceiling: those made the *vector loop be the reasoner* and tried
  to learn ProofWriter's *logic in the weights* — which the architecture forbids ("no information in
  weights"). The fix: **the controller emits ops; the executor's `INFER` = `forward_chain` (0.989) does
  the derivation; the controller only learns the navigation *policy*** — small, cheap to train a lot
  (knowledge is graph data), transferable, *not a split*.
  - **Step 1a (foundation):** `ops.RESPOND_VERIFY` + `Executor.load_theory`/4-tuple-INFER/`respond_verify`/
    `apply_rule`. Gate: the executor-driven verdict **== the 0.989 symbolic floor exactly** across depths
    0–3 (same engine) — so a controller driving `[INFER, RESPOND_VERIFY]` *inherits* broad-data correctness.
  - **Step 2 (learned navigation = neural-guided proof search):** the controller **selects which rule to
    apply next** (the proven contrastive head over `encode_rule` candidate vectors; `proof_search.ProofSearch`),
    the executor applies that one rule **symbolically**, bounded + goal-directed; **Unknown = step-budget
    exhaustion** (OWA abstain). Teacher = the gold proof as a rule-selection sequence
    (`proofwriter.proof_rule_steps`, reusing `forward_chain`'s chain).
  - **Measured (dim=32, depths 0–2, CPU, 40 ep):** train rule-select acc **0.30→0.97**; **rollout verdict
    accuracy 0.84** (d0=1.00, **d1=1.00**, d2=0.48). **vs the M8/M9 ceiling of 0.50 (= majority) on the SAME
    data** — the architecture, not scale, was the problem. d1=1.00 is perfect one-step navigation; the
    engine derives, only *which move next* is learned.
  - **Honest frontier:** d2 plateaus at 0.48 *despite* 0.97 single-step (teacher-forced) accuracy; the
    overall 0.84 also benefits from Unknown-via-budget being correct, so the *pure* navigation signal is the
    provable depths (d1=1.00, d2 weak). Gates: executor parity + gold-teacher + rollout-terminates +
    apply_rule→proof. `scripts/train_proofwriter.py --navigate`.
  - **Step 2b — DAgger to fix d2 (MEASURED NEGATIVE + isolated root cause).** Hypothesis: d2 is exposure
    bias (trained teacher-forced, rolled out on its own states). Built on-policy DAgger
    (`proofwriter.gold_plan`/`expert_action` — a recovery-capable expert on ANY state;
    `proof_search.collect_dagger`; `train_proofwriter --dagger`). **Result: d2 stayed at *exactly* 0.48**
    across 4 rounds (d1 0.96→1.00; overall 0.827→0.840). DAgger did not move it — so the hypothesis is
    wrong. **Diagnostic (`scripts/diagnose_d2.py`, genuine 2-step provable items): the policy fails at the
    FIRST move — first-rule-correct 9/40 = 0.23** (vs d1=1.00), second-rule|first 0.33, fully-solved 0.05.
    **Ablation:** oversampling the rare multi-step first-moves ×8 barely moved it (0.23→0.30, second→0.00) —
    so it is **not data imbalance** either. **Root cause: the policy learned a myopic 1-hop heuristic**
    ("pick the rule whose consequent = the goal") — perfect at d1, near-chance at d2, whose first move is a
    *non-goal-matching intermediate*. The forward-facts+goal encoding **cannot represent multi-hop
    goal-relevance** (that an intermediate rule is a precondition for a rule that yields the goal). Neither
    exposure bias nor imbalance — it's **representational**. **Next milestone (reframed): goal-directed /
    backward navigation** — a subgoal stack or a goal-relevance signal over rules, so the controller can
    plan more than one hop. DAgger code is kept (sound infra; lifts d1→1.00; the right tool *if* the
    bottleneck were exposure bias). Suite 286.
  - **Step 3 — goal-directed BACKWARD navigation (the fix; MEASURED WIN).** Same `MindController`, same
    contrastive head, same executor — only the search *direction* changes. Forward training fixed the
    `is_q` goal and varied the facts (→ the myopic shortcut); **backward fixes the facts (STM context) and
    varies the `is_q` subgoal**, so each step is "pick the rule whose consequent matches THIS subgoal" —
    the 1-hop skill the model already does perfectly. `proof_search.backward_step` (unify consequent↔
    subgoal → grounded antecedents, via `reasoning_oracle.unify`/`ground`); `BackwardSearch` (subgoal-stack
    rollout: prove query→TRUE, negation→FALSE, budget→Unknown); `proofwriter.backward_examples` (teacher =
    `gold_plan.rule_of[subgoal]`, reusing `build_proofsearch_batch` unchanged — subgoal in the goal slot).
    **Measured (dim=32, depths 0–2, CPU): d2 0.48→1.00 (peak ep20; 0.92 settled), overall verdict acc
    0.827→0.963**, **and in ⅓ the steps** (d2 ≈3.4 vs forward's 6.0, bounded by the proof). The plateau
    forward search *never* moved (0.48 flat through 40 ep + 4 DAgger rounds) breaks immediately because
    backward chaining decomposes depth-2 into the learnable 1-hop decisions. **Honest caveats:** d1 dips
    0.96→0.88 (the two-pass TRUE/FALSE verdict occasionally proves a spurious branch on 1-hop items — a
    backtracking/verification gap, not a planning one); d2 oscillates 0.92–1.00 across late epochs (eval
    n=25/depth). The win is the *direction*, exactly as the user specified: *input → build STM/context → if
    a question, work backwards, same state machine.* `scripts/train_proofwriter.py --backward`; suite 288.

- **M11 — ONE conscious door over a clause feed (self-routing; no pre-segregated question).** The user's
  directive: stop the several entry points (`recall`, `respond(episode)` *handed* an `is_q` pointer +
  multiple-choice `options`, ProofWriter verification only in a train script); instead **feed one stream of
  clauses and let the system decide per-clause how to respond — never told "this is the question," and
  deriving the answer rather than selecting from options.** Key enabler: a question is **already an intrinsic
  meaning-type** (`membrane.parse` → `("query", …)`, parsed from the interrogative form like a "?"), so
  routing on the clause's own type is faithful, not out-of-band instruction. **`ConsciousLoop.consume(feed)`**
  (`mind/conscious_loop.py`) is the single door: `fact`/`disj`→learn into the accumulated working theory,
  `rule`→learn the rule, `("query",s,r,v,pol)`→**`BackwardSearch`** (yes/no → TRUE/FALSE/Unknown),
  `("query",s,r)`→`forward_chain` to derive the value (wh). `proofwriter.feed_from_example` adapts a theory
  to a feed with no NL. **Measured: ProofWriter answered *through the door* at acc 0.922 (d0=1.00 d1=0.88
  d2=0.68)** — the one-door path IS the backward reasoner, not a weaker fallback — and a hand-typed feed
  (teach 3 clauses, then ask) self-routes: *is alice smart? → true (2 steps); is alice green? → Unknown*. No
  `is_q`, no options, controller weights untouched (learning is graph/theory writes). **v1 scope (honest):**
  routing is by intrinsic meaning-type and memory is the **per-feed accumulated 4-tuple theory** (= "build
  the STM/context from the feed"); unifying that with the durable *polarity-aware* `KnowledgeGraph` (it is
  3-tuple+truth today) and making the *route itself learned* (controller emits PERCEIVE vs RESPOND) are the
  next steps. Natural-language English-in remains a separate later layer. `scripts/talk_feed.py`; suite 290.

- **M12 — the routing decision itself is LEARNED (the controller emits the act).** M11 still chose
  absorb-vs-answer with a code `if tag == "query"` switch; the user wanted the consciousness to make that
  call. Key enabler (already present): `ClausePsyche.op_head` emits a per-clause op (`out["op_logits"]`), and
  **`is_q` is not a GRU input** — so an act predicted from the clause encoding is honest, not a leaked flag.
  Because a ProofWriter yes/no question is content-identical to its assertion, the interrogative mood
  (`pred="p:?"`) is the *necessary* marker the router reads (a "?"). **`mind/routing.py`**: encodes each
  clause as a one-step batch (mood in the predicate), gold act = WRITE (declarative→absorb) vs RESPOND
  (interrogative→answer), CE `act_routing_loss` over `op_logits`; `predict_acts` drives **`consume`** (the
  `if tag` switch is gone — only the absorb/answer *decision* is the model's; yes/no-vs-wh stays a content-
  arity choice). `train_proofwriter --backward` trains ONE controller jointly (navigation + routing; disjoint
  heads). **Measured: act accuracy 1.000 on 1,524 *unseen* test clauses** (generalizes — different entities/
  relations); **ProofWriter through the learned door = 0.926 (d0=1.00 d1=0.88 d2=0.70)** — no regression vs
  M11's 0.922; **mood-flip is decisive** — identical content `alice is furry` routes to *absorb* as a
  statement and *answer* as a question. **Honest scope:** the classification is easy (mood is a clean marker)
  — the deliverable is *architectural* (the last hand-wired control switch is gone; the decision is the
  model's) + generalization, not a headline accuracy. Still per-clause, not yet a full workspace dynamic
  (context-sensitive routing, clarification). **NLP English-in (grammar-file encoders/decoders) is next** —
  the parser will emit the mood marker this learned router already consumes. `scripts/talk_feed.py`; suite 292.

- **M13 (stages 3–4) — referentials + light WSD (deterministic).** **Coreference** (`mind/coref.py`):
  resolve pronouns across the conversation by **recency + gender/number agreement + subject salience**
  (Centering-lite, small name→gender lexicon) — `converse` threads a `Coref`, resolving a pronoun subject
  to its antecedent and registering concrete mentions. Measured: with `john` in the office and `mary` in
  the garden, *"what can **she** hold?"* → mary's rule, *"what can **he** hold?"* → john's — agreement beats
  recency. **Prepositions:** in/on/at/inside/into → PLACE (curated frame map). **WSD:** done *structurally* —
  a determiner means the next token is a noun, so the homograph "can" is the modal in predicate position
  ("a robin **can** fly" → CAN) and a noun after a determiner ("hold the **can**" → value `can`); no sense
  model needed. Honest: full open-domain noun WSD (bank/bat) and split-antecedent/bridging coreference stay
  deferred. Parity with `membrane.parse` still exact (0/429). `mind/coref.py`; suite 305.

- **M13 (stages 1–2) — talk to it: owned controlled-language encoder/decoder with subordination.** The
  system reasoned/routed in meaning-space but could only talk through a flat 7-relation regex. Decisions:
  **controlled English first** (owned, deterministic, round-trip-tested; NOT the fallible open-domain
  `quantum_parser`) but **broad subordination**, **deterministic-first** (no learned WSD/coref). Built a
  **recursive-descent parser** `mind/grammar.py` (structural — driven by articles/prepositions/verbs, not a
  vocab list) emitting the *same* tagged meaning objects `consume` already takes, so it drops in unchanged.
  **Full parity:** `grammar.parse == membrane.parse` on all 429 curriculum sentences — a drop-in superset.
  **New (subordination):** conditionals `if X, Y` → a **grounded** rule (named conditional is about that
  entity — does NOT over-generalize to `?p`); **quantified descriptions** "everyone who is in the kitchen
  can see the window" → a `?p` **universal** rule; restrictive relatives "the N that …" likewise; yes/no
  questions carry polarity. The **`ConsciousLoop.converse(lines)`** door: English in → parse → `consume` →
  English out; unparsable input abstains (no silent garbage). **End-to-end gate passes:** *"everyone who is
  in the kitchen can see the window. mary is in the kitchen. what can mary see?"* → **"Mary can see the
  window."** (no controller — wh + forward-chain). **Honest scope / next:** the three hard problems are
  staged — **referentials** (pronouns via recency+agreement+salience; prepositions via a curated
  frame→relation lexicon) and **light WSD** (POS + frame lexicon; context-overlap for residual noun
  polysemy) are stages 3–4, not yet built; relatives drop the head-noun type (single-antecedent rules);
  built-in reasoning schemas (INHERITANCE) have no controlled-English sentence and stay seeded knowledge.
  `mind/grammar.py`, `scripts/talk.py`, `tests/test_mind_grammar.py`; suite 300.

- **M14 — from auto-search to natural back-and-forth (substrate: L2 ask-when-blocked + L4 cross-turn
  memory).** `converse` was stateless — one query → one terse answer or a dead-end "I don't know"; a
  database query, not a dialogue. The brainstorm (a JARVIS/C-3PO register, grounded by construction since
  every utterance is rendered from real derivation) named the **learned drive (L6)** as the destination
  and chose to build the conversational *substrate* first. Built `mind/conversation.py` — a stateful
  `Conversation` session over `ConsciousLoop`: **(L4)** facts/rules persist across turns, a persistent
  `coref.Coref` resolves "she/it" across turns, and a `topic` tracks the last subject; **(L2)** a blocked
  query no longer dead-ends — `reasoning_oracle.find_missing_premise` (one-hop backward unify over the
  rule whose consequent matches the goal) computes the *exact* missing premise, the system **asks it back**
  as an English question (`membrane.render_polar_question` + `verbalize_ask`), remembers the blocked query
  in `pending`, and **resolves** it ("Then yes — …") once a later turn supplies the premise. Deeper chains
  surface one premise per turn — itself natural back-and-forth. **Instrumented for L6:** every turn logs a
  `TurnOutcome` (answered/asked/resolved/abstained/learned + knowledge-gained = pending questions this
  statement unblocked) — the reward/telemetry a future drive optimizes; recorded, not yet consumed by any
  policy. Grounded by construction: it asks only for a premise a rule needs and answers only what it
  derives. **Headline gate = one LONG conversation** (`scripts/stress_converse.py`, `tests/test_converse_
  scale.py`): a growing session to **1500 derivable facts → 750/750 answers correct vs the symbolic
  oracle**, asks/resolves/abstains all correct *along the way*. **Honest negative:** the re-feed-accumulated
  design is O(KB)/turn — per-query latency grows 2.4ms→18ms as the KB grows (reported; the place a learned
  drive + incremental memory would later address). `mind/conversation.py`, `reasoning_oracle.find_missing_
  premise`, `membrane.render_polar_question`, `verbalize.verbalize_ask/_resolution`, `scripts/talk.py`,
  `tests/test_mind_conversation.py` + `tests/test_converse_scale.py`; suite 316.

### 0r. Calibrated initiative — bounded volunteering (L3) + the learned drive (L6), supervised then sequential-RL (MEASURED)
- **M15 (supervised drive).** Added the *volunteer* action — after answering, surface ONE relevant, true,
  unsaid fact selected over the real `forward_chain` closure, hard per-turn budget (`max_volunteer=0` ≡ the
  M14 baseline). The **drive** (`mind/drive.py` `DrivePolicy`: a small MLP over 8 grounded features →
  4 masked actions {ANSWER, VOLUNTEER, ASK, QUIET}) decides *when* to use it, trained by masked CE against a
  synthetic-user teacher (`mind/drive_env.py`) whose gold is a **latent usefulness the policy never sees**
  (`relevant(depth) AND focused(backlog)`). Genuinely learned (not a coded rule) because the policy must
  predict the latent from grounded features. **Gate PASS:** held-out acc 0.815; learned drive beats both
  always-on/yappy and never/Siri on volunteer F1 (0.66 vs 0.50 vs 0.00) and ask F1 (0.50 vs 0.32 vs 0.00) —
  the quantified line between the two. Honest: the latent is deliberately noisy (Bayes-opt < 1.0).
- **M16 (sequential RL).** The single-turn teacher is blind to *consequences*. Added the first **sampled
  policy-gradient** in the repo (REINFORCE + value baseline, `drive.sample_action`/`drive_rl_loss`),
  mirroring the existing `clause_psyche` PonderNet expected-reward idiom (–reward, exploration prior,
  soft→discrete anneal) but adding multi-turn returns. A grounded hidden-goal **user simulator**
  (`mind/drive_rollout.py`) makes the gain genuinely sequential via three symbolic couplings: a **distractor
  stream** (separable only by `focus` — the feature the M15 teacher was trained to *ignore*), a **patience
  budget** (off-path initiative disengages the user → goal missed), and **backlog gating** (overloaded asks
  stall). Reward = symbolic goal-progress (the floor — no judge/reward-model). Warm-started from M15, trained
  on a subconscious-style loop (`mind/drive_subconscious.py`). **Headline gate PASS:** on held-out dialogues
  the RL drive reaches the user's goal **0.97 vs 0.60** for the supervised drive (= always-on; the oracle
  ceiling is 0.97), return 4.3 vs 2.2 — and the **coupling-OFF ablation ties (1.00 = 1.00)**, the proof the
  gain is *sequential*, not a re-tuned single-turn proxy.
- **Three honest findings.** (1) **turns-to-goal is survivorship-confounded** — the supervised drive's lower
  average is over only the easy 60% it solves; the gate is goal-rate + return, not turns. (2) **The RL drive's
  single-turn calibration drops** (0.82→0.75) and that is *correct*, not a regression: it learned to use
  `focus`, useless in the single-turn distribution but essential sequentially. (3) **Warm-start helps as an
  *initialization* but a KL-anchor back to the teacher *hurts*** — the supervised policy is "always-act",
  wrong on distractors, so anchoring fights the learning (`w_anchor` defaults to 0). Robust recipe (goal-rate
  0.975 across seeds 0/1/2): lr 2e-3, init temp 1.0, entropy 0.01, value-weight 0.3, 300 episodes/round —
  gentle enough to avoid the early "push VOLUNTEER down everywhere before `focus` is learned" collapse.
- Files: `mind/drive.py` (value head + warm-start remap + `sample_action`/`drive_rl_loss`, mask-safe entropy/KL),
  `mind/drive_env.py`, `mind/drive_rollout.py`, `mind/drive_subconscious.py`, `scripts/train_drive.py` +
  `scripts/train_drive_rl.py`, `tests/test_mind_drive.py` (7) + `tests/test_mind_drive_rl.py` (9) + 4 L3 tests.
- **Next (documented, unbuilt):** the simulator is a stand-in; a real/learned user model (and the graded judge
  distilled to a reward model) is the frontier — but the sequential-RL machinery + the floor-as-reward now exist.

### 1. What is the consciousness state? (and its real loss)
The state is deliberately **abstract** — a learned vector with no imposed meaning
("figure it out later"). The auxiliary `consciousness_consistency_loss` is a
placeholder (L2 between consecutive states, a "don't thrash" prior); note it
*rises* as the model learns, because encoding facts legitimately moves the state.
Open: define what the state should represent (self-model? working-memory
summary? confidence/awareness?) and an objective that rewards *coherent* state
evolution rather than mere stability. `TODO(consciousness-loss)` in `losses.py`.

### 2. Semantic composition onto NSM primes
Still the central unsolved problem and explicitly out of scope. `MockSemanticMapper`
carries no meaning. NAOMI already has a *designed but unbuilt* geometric encoder
(`quantum_parser`'s `compose_subtree`: parse tree → vector via edge-type
operators). Open: how meaning is represented over the ~65 NSM primes, how
composition (negation/quantifiers/scope) works, and how it is supervised.

**WSD is a tractable first slice of this** (`wsd.py`). Instead of composing
meaning, it picks among a *finite* candidate set (each sense an NSM-prime
signature), disambiguating with a memory/state context. It is **coherence-driven
and self-correcting**: a learned coherence head asks "does this interpretation
make sense given the state?"; if not, the state updates and all senses are
re-evaluated (the chosen sense can flip across hops). No gold sense labels — the
coherence signal drives it. Currently the inventory and the sense→prime
signatures are **mocked** (`MockSenseInventory`); the real inventory is WordNet
(`WordNetSenseInventory` hook, which is itself this same mapping problem). Open
next: real WordNet senses, learning the coherence signal from self-supervised
"surprise"/contradiction rather than synthetic labels, and **wiring WSD into the
Mind loop** so sense-resolved representations (not raw tokens) are what get
written to memory.

### 2b. Multi-hop reasoning and adaptive halting
`agent.Mind` now supports fixed `reasoning_hops` passes over memory at the
question ("reason with states"). Open: make halting **adaptive** (ACT-style) —
stop reasoning when a coherence/confidence signal says the answer is settled,
rather than a fixed hop count. The WSD coherence head is the prototype for that
signal; unifying the two (one coherence mechanism for both WSD re-evaluation and
reasoning-loop halting) is the interesting direction.

### 3. Input encoding: the parser is experimental → "chained transformers"
The rule-based parser is inconsistent, so it is **not** the spine — the default
input encoder is plain tokenization, and `ParserInputEncoder` wraps the parser
optionally (degrading to tokens on any failure). The user's fallback if natural
language proves too messy is **chained transformers**: a learned encoder that
maps a sentence to the input object, implementing the same `AbstractInputEncoder`
interface. Not built; the seam is ready.

### 4. Memory: two tiers built; pruning + consolidation still open
There are now **two tiers**: per-episode `WorkingMemory` (local context) and a
persistent `LongTermMemory` that accumulates a growing repo of entries +
connections across episodes (`lifelong.py` grows it). The state controls
retention via a `consolidate_gate`. Still open:
- **Pruning/forgetting** is a placeholder FIFO cap (`_maybe_prune`); a real
  policy should weigh connection strength, recency, and redundancy.
- The **connection graph** is simplistic (entries from one episode are linked);
  real associative structure (typed relations, co-retrieval, inferred links)
  is unbuilt — this is where the "giant repo of connections" should become a
  genuine knowledge graph (and connects to NAOMI's WordNet/triple work).
- Long-term contents are **non-parametric** (detached vectors); retrieval read
  projections are learned but the store is a database, not weights. Whether/when
  to distill the repo back into parameters is open.
- Working-memory writes are soft slots at a fixed index; content-addressed
  allocation is future work.

### 5. Tree serialization (flat → hierarchical)
`serialize_parse_tree` now includes semantic-role relations but is still a flat
pre-order stream; hierarchy is not recoverable. Open: structural position
encodings / tree-aware or graph encoders. `TODO(tree-encoding)`.

### 6. Consciousness-dimension ablation
`consciousness_dim` is a free knob and the state is opaque. Open: ablate it
(8/48/128/512) against accuracy and state-transition stability; does capacity
help or just drift?

### 7. Coherence checking
Nothing verifies the answer is consistent with the stored facts / state. Open: a
coherence objective or verifier (closely related to problem 1), and
contradiction detection over the `CausalTable` / memory.

### 8. The textbook north star
The goal is to **read a textbook chapter as the context stream and answer its
(often multiple-choice) homework questions**. `TextbookSource` is a stub. Open:
chapter → clean context stream segmentation, and homework-question extraction →
`Episode`. Multiple choice is kept first-class precisely because it gives the
densest training signal ("most training up").

### 9. Reconciling NSM primes with NAOMI's 51 anchor dimensions
NAOMI's existing semantic space uses 51 linguistically-motivated anchors
(nominals/scopes/roles + grammatical + logical); this module uses the ~65 NSM
primes. They are conceptually adjacent but distinct, and nobody has decided which
(or how) is the basis. Open question, not yet addressed.

## Decisions made while building

- **Transformer as a state-transition function, not a text model.** NAOMI's
  existing thesis is explicitly anti-transformer-for-reasoning; here the
  transformer is the *transition operator* inside a loop, and the reasoning is
  carried by the persistent state + external memory.
- **Episodes expand into a per-sentence unroll**, padded to a max context length
  with a step mask; the question is a distinct aligned final step for every
  episode (clean batching).
- **Memory writes are gated by the ABSORB action probability** and threaded
  through the unroll as state (not stored on the module) for clean autograd.
- **Multiple choice kept as first-class** (densest signal) and reuses v1's option
  scoring; open-ended (bAbI) supported via an answer-vocab classifier.
- **Weak action supervision** (statements→ABSORB, question→RESPOND) teaches the
  procedure; it converges to 100% action accuracy quickly.
- **Parser demoted to an optional input encoder** given its inconsistency; the
  default path has zero parser dependency so install/tests stay light.
- **Deterministic data sources**; bAbI degrades gracefully to the curriculum
  generator when the download is blocked.

## NSM prime inventory — caveats
`nsm_primes.py` follows the canonical ~65-prime table (Goddard & Wierzbicka 2014)
to the best of our knowledge. Version-dependent details are flagged
`TODO(canonical-list)` (e.g. whether `DON'T WANT` is a distinct prime, the
possession prime's exact form `(IS) MINE` vs. `HAVE`, the allolex set). These
should be verified against a primary source; nothing was fabricated to fill gaps.

### 0s. The word-meaning-value generator — a derived minimal basis + a deterministic clause==word reduction operator (M17, MEASURED)
Talking about feelings exposed the real frontier: the system *looks meaning up*
in the DeepNSM/gold explication dictionary (the "prior prime stuff") instead of
*deriving* it, and there was **no measure of word-meaning understanding at all**
(only task-answer accuracy). M17 builds a `ground/` subpackage that *generates* a
word's meaning from NSM primes (the axes) + WordNet relations (the points/web),
with DeepNSM demoted to a held-out external check. This directly answers the §9
open question ("which / how is the basis"): the basis is **derived, not assumed.**

**Model.** NSM primes = degrees of freedom (axes); WordNet = the LTM dictionary
(points + relations). A *meaning value* is a coupled (ParseTree object + grounded
coordinate over the basis — never flattened to a bag). The basis is found by
**Minimum Description Length** (minimize #axes *and* decomposition depth jointly),
seeded by NSM-65. The deterministic, learning-free **reduction operator** flattens
a definition-clause to a prime fixpoint and **lexicalizes** a reduced clause back
to the word it defines (exact normal-form match, then coordinate-closeness
fallback over grounded points). Validation = **clause==word self-consistency**,
lookup-free.

**Measured.**
- *Baseline harness (M17.0, depth 3, 31 words):* convergence 0.730, prime_grounding
  0.567 (only ~57% of leaves reach a prime — the depth-2 truncation quantified for
  the first time), DeepNSM-agreement 0.083.
- *Reduction operator (M17.1):* deterministic + idempotent + confluent; exact
  round-trip 31/31; perturbed-clause recovery (drop a definition word) 24/30 via
  coordinate closeness.
- *Basis discovery (M17.2, 619-word relationally-closed vocab):* 15 interpretable
  primitives promoted (act, make, person, give, feeling, state, life … — a
  Longman-style defining vocabulary, the predicted "more than just NSM"); MDL
  monotonically 45669→36770; grounding 0.127→0.250.
- *Understanding evaluation (M17.3; 319 held-out words OUTSIDE DeepNSM, 300 covered
  for the external check), seed NSM-65 → derived basis:* grounding 0.126→**0.255**
  · convergence 0.698→**0.769** · syn>ant discrimination 0.227→**0.455** (n=22) ·
  hypernym containment 0.268→**0.422** (n=23) · clause==word round-trip exact 0.909,
  perturbed 0.387 (n=319) · DeepNSM agreement 0.072→0.059 (n=300). Deriving the
  basis improves *every* understanding metric on words the dictionary never covered.

**Honest negatives / boundaries.**
- syn>ant discrimination improves but stays **below chance (0.455 < 0.5)**: antonyms
  share nearly all gloss structure, so coordinate-cosine rates them as *similar*.
  Coordinates alone cannot separate antonyms — this needs explicit antonym **edges**
  (the relational web), confirming the design's "antonyms differ on minimal axes"
  point. Future work.
- DeepNSM agreement is low and the derived basis slightly *lowers* it (0.072→0.059)
  — which **demonstrates independence**: the generator never used the dictionary,
  and gloss-decomposition ≠ curated explication.
- Perturbed recovery drops at scale (0.933 @31 words → 0.387 @319) as the index
  gains confusable neighbours; exact round-trip is partly by construction.
- Molecules are grounded *points* (not axes) with no prime explication yet, so
  molecule-grounded words have an empty prime coordinate. MDL / feedback-vertex
  selection is heuristic (not a proven minimum). Values are per-*sense* (first-sense
  bootstrap; WSD deferred).

**Reuse (no rebuilds):** `data_structures.ParseTree`, `serialization` (lossless),
`collapse`/`DEFINES`, `tpr` (coordinate handle), `nsm_primes` (seed axes),
`meaning._resolve_uncached`/`_clone_bounded` patterns + WordNet glosses (as
decomposition *material*, never the runtime answer), `wordnet` (+ new
`antonyms`/`synonyms`/`hypernyms`). **Affect (the original M17) becomes M18:**
valence falls out of *generated* values (does the reduced value contain GOOD/BAD),
built on this measured grounding rather than the dictionary.

Gates: 36 new tests (canonical 6 · consistency 10 · reduction 8 · basis 7 ·
evaluation 5); full CT suite green.

### 0t. Scale + make the web do work — 10k corpus, polarity coords, multi-signal basis, graph closeness (M18, MEASURED)
M17 left two gaps: syn>ant discrimination below chance, and everything on a
~600-word vocab. M18 scaled to a ~10k **gloss-vocabulary** corpus (the words
WordNet uses to define other words) and made the WordNet relational web do the
antonym work coordinates can't. All additive under `ground/`.

- **M18.0 scale + caching.** `corpus.gloss_vocabulary(n)` ranks content-word
  frequency across all 117k WordNet glosses (offline, deterministic). `DecompCache`
  caches base decompositions; a promoted axis reproduces
  `naive_decompose(extra_axes=…)` by an in-memory prune+relabel (bit-equivalent),
  so basis scoring needs no WordNet re-calls. Warm 10k ~9s; extra-aware full pass
  ~250ms (vs ~24s uncached); full 10k basis search ~250s. The basis **converges**
  as the corpus grows — promoted axes (act, having, relating, state, person,
  particular, substance, things, number, cause, make …) near-identical at
  1k/5k/10k; grounding 0.119→0.280. A real defining vocabulary, not an artefact.

- **M18.1 polarity coordinates.** Signed NSM pole pairs (GOOD/BAD→±EVAL,
  BIG/SMALL→±SIZE, MUCH_MANY/LITTLE_FEW→±QTY) + a gloss-magnitude axis (hot="high"
  vs cold="low"; recovers the negation words decomposition drops as stopwords) +
  morphological negation (un-/non-/dis-/-less, base-validated) flipping the base's
  poles. Measured (3k corpus, 397 antonym pairs): syn>ant 0.413→0.466 (+0.053).
  Honest: improves but coordinates alone stay **below chance** — antonyms share
  definitional structure, and synonym sense-mismatch (WSD deferred) depresses it.

- **M18.2 multi-signal basis selection.** MDL gates (shortlist = frequent
  un-grounded words), relatedness steers (synonym-cosine + hypernym-containment on
  a train split). **Honest negative:** held-out syn_cos 0.258→0.251,
  hyp_containment 0.500→0.508 — flat/mixed. Basis-axis selection weakly controls
  relatedness; the levers are the coordinate and the edges, not which words are
  atomic.

- **M18.3 graph-aware closeness — the antonym solution.** closeness =
  polarity-coordinate cosine − an antonym penalty from **train-split** antonym
  edges propagated one synonym hop. Evaluated on a held-out synonym-vs-antonym
  discrimination (P(synonym pair closer than antonym pair); 0.5 = chance),
  circularity-free (the held-out pair's own edge is in the test split, never the
  train edges). Measured (3k corpus, 230 held-out antonym pairs):
  **pure coordinate 0.333 · polarity 0.392 · graph-aware 0.638** (λ=1.0). The web
  pushes held-out antonyms from "looks similar" (below chance) to clearly separated
  (above chance) — the result the coordinate work pointed to all along.

Honest boundaries: graph closeness needs train antonym edges near the held-out
pair (antonyms are sparse, ~2.3/word), so coverage caps the gain; values are
per-sense (first-sense bootstrap; WSD deferred); MDL/feedback-vertex selection is
heuristic. Gates: 19 new tests (corpus/cache 5 · polarity 4 · multisignal 5 ·
closeness 5); full CT suite green.

### 0u. The unified meaning space — reverse-engineer the minimum axes from ALL relational signal, place words, rebuild the dictionary (M19, MEASURED)
The reframe (user): M17/M18 were partial views of *one* problem, not separate
systems. Every lexical relationship is signal about the geometry of meaning; use
all of it to reverse-engineer the **minimum interpretable axes**, place words in
that one space, then **rebuild the dictionary** as geometry grounded in it.
**Hard invariant (the user's condition): every axis is NAMED/interpretable — no
arbitrary word2vec dimensions.**

- **M19.0 unified relation store.** `wordnet.py` wrappers for the wider relations
  (`lexname`, `attribute` bidirectional, `similar_to`, `derivational`, `meronym`,
  `verb_group`); `RelationGraph` carries them all plus the two *feature* relations
  that name axes (`lexname`, `attribute`). 3k coverage: lexname 1.00, synonym 0.94,
  derivational 0.89, is_a 0.82, antonym 0.39, similar 0.31, meronym 0.28; ~16k
  in-vocab word-word relational pairs.
- **M19.1 interpretable axis set + dimensionality.** 234 named candidate axes
  (65 primes + 126 attribute dimensions + 43 lexname categories). The word×axis SVD
  (the right matrix — the synonym-affinity eigenspectrum was degenerate): intrinsic
  dim ~46 (90% energy), effective ~17. Meaning is low-dimensional AND interpretable.
- **M19.2 placement by constraint satisfaction.** Anchored coordinate (attribute
  axes signed by gloss magnitude → antonyms at opposite poles on a *shared* axis,
  non-circular) + stable relational relaxation (synonym/similar label propagation;
  spectral radius ≤1, no learned weights). **Closes the M18 seam:** held-out
  syn-vs-ant with PLAIN cosine 0.404→**0.693** (α=0.7), beating M18.3's
  comparison-time penalty (0.64). Antonymy now lives in the position, not a
  correction.
- **M19.3 minimality.** ~**30 named axes reproduce 95%** of the relational fidelity
  (full 0.69; K=5→0.61, K=30→0.66, K=234→0.69). The minimal set is NSM primes +
  lexname categories (SOMETHING, ONE, PART, BODY, lex:noun.act, WHEN, PEOPLE,
  KIND…). The empirical "minimum axes of meaning."
- **M19.4 dictionary reconstruction.** From positions alone, held-out AUC:
  **synonym 0.857, similar 0.734, hypernym 0.722** — the dictionary genuinely
  reconstructs from geometry. antonym-by-distance 0.275 (honest: antonyms are
  near-but-*opposite*, not far; the proper antonym metric is the placement
  syn-vs-ant, 0.69).

Honest boundaries: attribute coverage is 8% of words (35% of adjectives), so
attribute-pole anchoring is partial; antonymy is near-but-opposite (distance
predictors mis-score it); novel-pair surfacing is mostly noise at this fidelity
(the generative payoff is weak); per-first-sense (WSD still deferred). Placement is
deterministic label propagation; axes never rotate/mix, so every word's coordinate
stays readable. Gates: 18 new tests (relations 5 · axes 4 · placement 4 ·
minimality 3 · dictionary 2); full CT suite green.

### 0v. The honest, normalized meaning space (M20, MEASURED)
The user caught real rigor holes in M19.3: "30 axes seems low — are unrelated pairs
overlapping / using different sections of an axis? should we lock axes to [-1,1]?"
Live diagnostics confirmed the critique:
- **Unrelated pairs overlap:** random-pair cosine **0.32** (synonym 0.75) — the placed
  coordinate is dense (137/234 axes active/word) and absence was encoded as 0 (no
  disagreement).
- **Axes incommensurable:** per-axis std attribute 0.014 vs lexname 0.068.
- **"30" was a variance-ranking artifact:** ablation showed lexname not load-bearing as
  a block (drop → no loss) while variance-ranking kept it.

- **M20.0 normalization.** `ground/normalize.py` per-axis transforms (fit on the corpus
  coordinate, not relation labels — not circular). The user's "lock to [-1,1]" instinct
  was right, but the literal min-max **backfires** (random 0.98 — shared min-pole
  domination, the same failure as absence=-1). The fixes: **z-score standardize**
  (random 0.32→**0.007**, commensurable) and **tanh(z-score)** (bounds each axis to a
  readable **[-1,1]** — the user's intent — random 0.32→0.145, best syn>ant
  discrimination raw→tanh **0.673→0.756 held-out**).
  > **[M24 leakage correction]** This was originally reported as **0.82→0.94**, which was
  > *leaked*: the placement propagated over ALL synonym pairs and then scored discrimination
  > on (a subset of) those same pairs. Re-run held-out (test pairs excluded from propagation),
  > raw is 0.673 and tanh **0.756** — tanh is still the best normalization for antonyms, but
  > the magnitude was inflated ~0.18. See §0z.
- **M20.1 honest minimality.** Rank axes by leave-one-out *contribution* on the
  normalized space, not variance. Full discrimination (held-out, tanh) **0.756** *[M24:
  was 0.94, leaked]*. The honest curve **rises then falls**: it climbs to a **PEAK ~0.83
  around K≈80–120**, then the tail axes *add noise* (all 214→0.756) — so the last ~100
  axes hurt. The minimal set is a named **mix** (primes + lexname + a few attributes).
  "30" wasn't a pure artifact, but the real story is the peaked curve.
- **M20.2 re-audit (honest trade-off).** Dictionary reconstruction (held-out AUC):
  | norm | synonym | similar | hypernym | antonym |
  |---|---|---|---|---|
  | raw | 0.857 | 0.734 | 0.722 | 0.275 |
  | standardize | 0.848 | 0.756 | 0.722 | 0.349 |
  | tanh | 0.779 | 0.649 | 0.722 | 0.506 |
  Normalization is **not a pure win**: standardize keeps synonym/similar AND fixes
  overlap (best all-around); tanh maximizes the hard antonym case + bounds axes to
  [-1,1] but costs synonym-vs-random AUC. Hypernym (binary-feature containment) is
  metric-invariant. Default = tanh (honors the bounded-[-1,1] request + best syn>ant);
  standardize is the all-around alternative — surfaced to the user as a choice.

Honest boundaries: antonym-by-distance is the wrong predictor for antonymy
(near-but-opposite) regardless — the proper antonym metric is syn>ant discrimination
(tanh **0.756 held-out** *[M24: was 0.94, leaked]*). Gates: 9 new tests (normalize 5 ·
honest-minimality 3 · dictionary param 1); full CT suite green.

### 0w. M21 — the Null-aware bipolar representation: unrelated words made unrelated (win) + honest negative on per-word antonyms (MEASURED)
The user's brainstorm: `0` conflates "neutral" with "doesn't apply"; pure antonyms
(good/bad) should be ±1 on a clean axis; each axis should carry ONE meaning; and we
should make **unrelated** words unrelated (separation), not just related words close.
Built all three co-designed (representation + metric + contrastive objective).

- `ground/sparse_value.py`: each word is `(value, mask)` — a value only on its
  *applicable* axes (Null elsewhere). Antonym prime-pairs collapse to one signed
  axis (GOOD/BAD→EVAL: good=+1, bad=-1, grass=Null). Metric: distinctiveness-weighted
  (IDF) cosine — `"full"` (norm over full content → unrelated ~0) or `"masked"`
  (shared-axes only → crisp per-pair, hot/cold=-1.0).
- `ground/contrastive.py`: torch optimization of the values (mask/axes fixed, so
  every axis stays interpretable) — synonym→+1, antonym→-1, random→0, anchored to
  the principled signs.

Measured (2–3k gloss corpus):
- **WIN — unrelated separation (the user's main ask):** random-pair similarity
  **0.32 (M19 dense) → 0.08 (sparse Null) → 0.03 (after contrastive)**. dog/justice
  1.0→0, good/grass=0 (words sharing no applicable axes are *exactly* 0). Per-pair
  bipolar is clean (hot/cold=-1.0 masked). Words occupy ~**2.9 applicable axes**
  each (vs 137/234 dense) — "one meaning per axis" much closer to true.
- **HONEST NEGATIVE — per-word antonym discrimination:** held-out syn>ant
  **0.39 (sparse) → ~0.48 (contrastive)** — improves but stays *below chance*, far
  short of the M19.2 relational-propagation **0.69–0.76 held-out** *[M24: this range was
  written 0.69–0.94; the 0.94 was leaked, honest tanh is 0.756]*. The free optimization is also
  fragile on *pure* antonyms (good/bad collapses to 1.0 even anchored), because
  good/bad share their category and the single EVAL contrast is fragile.

Conclusion: the sparse **Null representation** and the **relational propagation** are
**complementary** — the Null model wins on unrelated-separation + interpretability;
the propagation wins on antonym discrimination. Per-word contrastive optimization
does NOT replace propagation for antonyms — recorded honestly, not hidden. The
sparse representation is the better answer to "make unrelated words unrelated" and
"one meaning per axis"; antonym-aware tasks should keep using M19.2 placement. Gate:
5 new sparse tests; full CT suite green.

### 0x. M22 — sense nodes (WSD by construction) + fused Null+propagation (MEASURED)
The user's call: fuse Null+propagation, add sense WSD. Built on **sense (synset)
nodes** so grounding is per-sense and synonymy is matched *by construction*
(co-lemmas of a synset collapse into one node).

- `wordnet.senses()` extended with per-sense `antonyms` (`lemma.antonyms()`) +
  `frequency` (`lemma.count()`).
- `ground/sense_graph.py`: `SenseGraph` — synset nodes, each grounded from ITS OWN
  gloss; sense-specific relations (similar_to near-synonym clusters, per-sense
  antonyms, hypernym, attribute, lexname). `build_sense_sparse` gives the M21 Null
  `(value,mask)` over sense-nodes.
- `ground/fusion.py`: `fused(a,b)` = **threshold-gated propagation** — relatedness
  (IDF mask-overlap) gates comparability, propagation (relax over
  similar_to + co-hyponym) supplies discrimination.

Measured (1.5k gloss words → 3450 sense nodes):
- Unrelated separation holds (random sparse-sim **0.10**); relations are sense-clean.
- Per-representation, similar≈antonym (0.52/0.52, disc 0.44) — antonyms still
  unsolved by the representation alone.
- **FUSION (the win):** propagation over similar+co-hyponym gives sim>ant **0.632**
  (above chance); the **threshold** gate (τ=0.15) cuts random overlap
  **0.198→0.116 while preserving discrimination 0.632** — separation AND antonyms
  *together*, which neither Null-only (0.10 / 0.44) nor propagation-only (0.198 /
  0.63) achieved alone. A *product* gate FAILED (crushed disc to 0.495); the
  threshold gate is the fix.

Honest boundaries: sim>ant 0.63 is above chance but **below the word-graph
propagation ceiling (0.69–0.76 held-out** *[M24: was written 0.69–0.94; the 0.94 was
leaked, honest tanh is 0.756]*) — sense-nodes rely on similar_to/co-hyponym
(sparser than word synonymy), so absolute discrimination is lower and the plan's
"both at once" target was NOT met (recorded, not hidden). Sense-matching gives
correct per-sense grounding and the fusion delivers separation+discrimination
together, but does not lift absolute antonym discrimination beyond the word-graph.
Gate: 4 new sense/fusion tests; full CT suite green.

### 0y. M23 — denser sense-correct close edges + a leakage correction to §0x (MEASURED)
Goal: thicken the sense graph so propagation reaches the word-graph's antonym numbers.
Added three per-synset close-edge sources to `SenseGraph` (`ground/sense_graph.py`):
`derivational` (`lemma.derivationally_related_forms()`, cross-POS), `meronym`
(part/member/substance + holonyms), `gloss_overlap` (senses sharing ≥K gloss content
words — the "use the definitions" edge), plus `close_edges(*types,
exclude_antonyms=True)` which unions types and drops known antonym pairs.

Coverage (1.5k words → 3450 sense nodes): derivational 582 pairs, meronym 161,
gloss_overlap 229 (vs similar 154, antonym 97); `close_edges` excludes antonyms
cleanly (0 leaked).

**The leakage correction (important).** Re-running the M22 fusion held-out —
*removing the test similar pairs from the propagation edges* — exposed that §0x's
**sim>ant 0.632 was inflated by leakage**: the M22 probe/test propagated over the
first 70% of `similar+cohyponym` (which contains *all* 154 similar pairs) and then
evaluated on a subset of those same similar pairs. With test pairs held out, the
honest numbers (threshold gate 0.15):

| close-edge set | edges | held-out sim>ant | random (gated) |
|---|---|---|---|
| {similar+cohyponym} (M22 base) | 1878 | 0.424 | 0.115 |
| **{similar+deriv+mero}** | 827 | **0.497** | 0.118 |
| {+gloss_overlap} | 1006 | 0.496 | 0.120 |

Findings, honest: **(1)** derivational+meronym *is* the best sense-node close set
(0.497 vs cohyponym 0.424) and drops antonym contamination — a real, modest lift;
gloss-overlap adds nothing and slightly raises random, so it is **dropped**. **(2)**
But once leakage is removed, sense-node propagation barely beats chance (~0.50) and is
**nowhere near the word-graph** — because propagation only helps pairs *connected* by
training edges; held-out similar pairs revert to the raw representation (~0.44). The
generalizing win is **not** antonym discrimination — it is the **threshold gate cutting
unrelated overlap (random 0.18→0.12)**, which holds up out-of-sample.
> **[M24 follow-up]** This paragraph originally speculated that "prior word-graph numbers
> (0.69–0.94) are likely similarly leakage-optimistic." The M24 audit settled it and the
> speculation was **half right**: the **0.693** (M19.2, via held-out `evaluate_placement`)
> is honest, but the **0.94** (M20 tanh, via `honest_minimality`/`probe_normalize` which
> placed over all synonym pairs then scored them) *was* leaked → honest held-out **0.756**.
> So the word-graph still genuinely beats sense-nodes on antonyms (0.756 vs ~0.50); M23's
> conclusion holds and is starker. See §0z.

Gate: 5 sense/fusion tests (leaky M22 test replaced with a held-out one); probe [8] prints
the honest sweep; full CT suite green.

### 0z. M24 — full leakage audit across M17–M23 (MEASURED)
The M22→M23 leak prompted a systematic audit: trace every headline metric to the code that
produced it and classify **H** (computed via a held-out `evaluate_*` function) or **A**
(ad-hoc probe/test with a hand-rolled split), then re-run every (A) held-out.

**Provenance table (headline metrics → source → verdict):**

| metric (notes) | source | verdict |
|---|---|---|
| M18.3 graph-aware closeness 0.638 | `closeness` (circularity-free) | **H — clean** |
| M18.2 multisignal held-out syn_cos | `multisignal.py:104–138` | **H — clean** |
| M19.2 placement syn>ant **0.693** | `evaluate_placement` (`placement.py:123–133`) | **H — clean** |
| M19.3 minimality 0.69 + K-curve | `minimality.py:53–68` (train_pairs passed) | **H — clean** |
| M19.4 / M20.2 dictionary AUCs | `evaluate_dictionary` (`dictionary.py:80–111`) | **H — clean** |
| M20.0 normalization random overlap | `probe_normalize` (random pairs) | clean (not a train/test metric) |
| M20.0/M20.1 syn>ant **0.94 (tanh)** | `honest_minimality.py:57` + `probe_normalize.py:49` — `place()` over ALL synonym pairs, scored on them | **A — LEAKED → 0.756 held-out** |
| M21 Null random 0.03 / syn>ant 0.39 | `sparse_value.pair_similarity` (no propagation) | clean (no training) |
| M21(b) contrastive | `test_ground_sparse.py:90` trains on train half, checks only random separation | clean (no leaked number) |
| M22 sim>ant **0.632** | ad-hoc fusion eval (test pairs = propagation edges) | **A — LEAKED → ~0.50** (fixed in M23) |
| M17.3 / M18.1 syn>ant (0.455 / 0.466) | coordinate cosine, no propagation | clean (no training) |

**Findings.** Two genuine leaks existed, both from the same bug — *placement propagated
over the same synonym pairs it then scored*: **M22's 0.632** (found + fixed in M23) and
**M20's 0.94** (found here → honest held-out **0.756**; leak table: raw 0.828→0.673,
standardize 0.767→0.625, tanh 0.938→0.756). Every other headline metric goes through the
held-out `evaluate_*` functions (which pass an explicit `train_pairs` = train half and score
the disjoint test split) or is leak-free by construction (coordinate-cosine / random-pair
metrics train on nothing). **The qualitative conclusions all survive** — tanh is still the
best normalization for antonyms, the minimality curve still peaks then adds a noise tail, and
the word-graph still beats sense-nodes — only three inflated magnitudes were corrected.

**Fixes.** `honest_minimality.py` and `probe_normalize.py` now `place(..., train_pairs=
train_syn+train_sim)` and score held-out; `test_ground_honest_minimality.py` /
`test_ground_normalize.py` updated off the leaked thresholds.

**Rule going forward:** any syn>ant / relatedness metric must either go through an
`evaluate_*` function or explicitly exclude its scored pairs from the propagation edges
(`train_pairs=` / `close = [p for p in edges if p not in test_set]`). A metric that
propagates over the pairs it scores is leaked by construction. Gate: 2 code fixes + 2 test
corrections; full CT suite green.

### M25 — retrain the placement on ALL signals: an honest negative (MEASURED)
The natural response to M23's wall (propagation only helps *connected* pairs): stop
propagating, **train** the position from all relations at once. `ground/joint_place.py`
fits per-word values on the **existing named axes** (init = anchored coordinate) with a
joint loss over every relation on its **train split** — synonym→+1, similar→+τ,
antonym→−1, hypernym(`is_a`)→related, meronym+derivational→mild, random→0, + an anchor
regularizer. Thesis guardrails held throughout: the Null mask is re-imposed every step
(no content on inapplicable axes — non-overlap, minimum values) and axes never rotate
(interpretable). Evaluated held-out (M24 rule): trained on train pairs, scored on the
disjoint test pairs, same split and same tanh space as the propagation baseline.

**Result — a clear, robust negative. Free per-word training loses to propagation on every
held-out metric** (2–3k gloss corpus, tanh):

| method | syn AUC | syn>ant | random |
|---|---|---|---|
| **propagation (`place`)** | **0.72** | **0.73** | **0.18** |
| joint-trained (best of a 4-config sweep) | 0.60 | 0.56 | 0.27 |

And **more training makes it worse** — cranking iters / antonym / negative weights drove
syn AUC 0.60→0.54 (classic overfitting: the free per-word values fit the train pairs while
held-out pairs degrade). Two structural reasons, both fundamental (not tuning):
1. **No generalization.** Propagation is *transductive* — it spreads anchored meaning along
   the graph at inference, so held-out pairs benefit and the coordinate densifies
   (~137/234 axes active). Free per-word fitting moves only the words that appear in train
   pairs; held-out pairs (other words) keep ~anchored values, so they don't improve.
2. **The mask caps density (by design).** The thesis mask forbids adding axes a word
   doesn't already have — exactly the axis-spreading propagation uses to make synonyms
   share support. Training cannot densify past it.
3. **Antonym cap confirmed** (as the plan flagged): masked per-word values can't push words
   that share all applicable axes to opposite (no bipolar axis to flip), so syn>ant sits
   near chance (~0.56) vs propagation's 0.73.

**Takeaway.** For this transductive lexical setting, **deterministic propagation over the
named axes is the right mechanism** — "just train it on all the signals" underperforms it,
and the experiment validates the existing design rather than replacing it. The all-signals
lever that *did* pay off was M23's relation set feeding propagation, not a trained objective.
A generalizing trainer would have to learn a *shared function* of graph context (not free
per-word values), which cannot add un-named axes without breaking the thesis — out of scope.
`joint_place.py` is kept as the documented negative (like the M21b contrastive one). Gate:
3 joint tests (mechanics + held-out split + determinism); probe `[9]` prints the head-to-head;
full CT suite green.

### The next layer — can everything live in the embedding space? (post-grounding roadmap)
The grounding arc (M17–M25) converged on a clear substrate: **a minimal set of non-overlapping,
interpretable named axes over which WORD-senses are *placed* by deterministic propagation** —
honestly evaluated (M24) and shown to resist both reframes tried (sense-nodes M22–23, trained
embeddings M25). So: **can everything be placed as a point in that space?** The measured answer
is **no — and the boundary is informative, not a failure.**

**What the coordinate DOES capture (place it as a point):** *gradient, "how-alike" meaning* —
synonymy (held-out AUC 0.86), hypernymy (0.72), similarity (0.73), and unrelatedness (random
→ ~0, the M21 win). For this, the embedding space is the right and sufficient representation, and
"more axes" is not the lever (M20 showed the tail is noise; M25 showed free training overfits).

**What resists placement (needs structure ON the space, not more axes):**
1. **Opposition is a relation, not a position.** Antonymy caps at ~0.73–0.76 because opposites
   *share* almost all structure — good/bad differ only in sign on a shared axis. Every attempt to
   push it into the coordinate (per-word bipolar M21, sense-nodes M22–23, joint training M25) hit
   the same wall; the only lever that has ever moved it is a **signed relational edge** used at
   scoring/composition time. Antonymy lives on the graph, not in the distance.
2. **Composition is an operation, not a point.** A clause/proposition is not a location in the
   word space — it is a *binding* of grounded primes (negation, quantifiers, scope). This is the
   project's central open problem (§2): you *place the primes*; you *compose the meaning*.
3. **Reasoning is dynamics, not placement.** Inference is transformation of meaning over the Mind
   loop, not a static coordinate.

**So we keep going deeper — by LAYERING structure on the grounded floor, not deepening the floor.**
Concrete next steps, in dependency order:
- **(A) Sense-resolve into the Mind (nearest, highest-leverage).** The M22 `SenseGraph` + the
  existing `wsd.WordNetSenseInventory` hook make sense→prime signatures *real grounded coordinates*
  instead of the current `MockSenseInventory`. Wire sense-resolved grounded vectors (not raw
  tokens) into what the Mind writes to memory (§2's stated "wire WSD into the Mind loop"). This is
  where grounding stops being a standalone lexicon and starts feeding the reasoner.
- **(B) Compose clauses over the grounded primes (the central problem, §2).** Represent a clause
  as a composition of placed word-coordinates (the designed-but-unbuilt `compose_subtree` /
  TPR binding), keeping antonymy/negation as *signed operations* per finding (1). Honest test:
  does a composed-clause coordinate reconstruct the clause's relations (entailment direction,
  negation flip) held-out — the M19.4 "dictionary from geometry" test lifted from words to clauses.
- **(C) Antonymy as a first-class relational layer.** Stop chasing it in position; give the space a
  companion signed-edge store consulted at comparison/composition time (generalizes finding (1)).
- **(D) Grounded consciousness-state (the long arc).** Project the Mind's state onto the
  interpretable named axes so reasoning is inspectable in prime terms, not opaque vectors.

**Closed doors (do not re-open without new information):** more axes (M20 noise tail), sense-nodes
as the *primary* space (M22–23 underperform the word-graph held-out), and trained per-word
embeddings (M25 overfits and loses to propagation). The substrate is settled; the work moves up a
level, from placing words to composing meaning.

### M27 — delete the token stack (the §0h debt, paid)

The destructive removal deferred since §0h is done: the legacy token-fed transformer
stack is deleted, per the user's call ("it's still in version control"). The last
commit containing it is tagged **`token-stack-final`**.

**Deleted:** `model.py` (the last transformer in the package), `agent.py`,
`memory.py`, `long_term_memory.py`, `bootstrap_memory.py`, `dataset.py`,
`lifelong.py`, `losses.py`, `metrics.py`, `config.py` (+ `configs/default.yaml`),
`semantic_mapper.py`, `parser_interface.py` (orphan); scripts `train_phase1`,
`eval`, `lifelong`, `probe_consistency`; their tests. **M26.2's
`GroundedMeaningEncoder` is deleted with the stack** — it wired grounded meaning
into the token model's capped bag-of-primes channel (the wrong integration point;
the grounded signal belongs in the mind/ meaning-graph fillers). The
`GroundedWordNetSenseInventory` (M26.0) and the WSD machinery survive untouched —
they are the real asset.

**Moved, not lost:** `PARSE_LABELS` → `structure.py` (parser plumbing),
`split_episodes` → `episode.py`, `consciousness_consistency_loss` →
`clause_psyche.py` (its only consumer). `input_encoder.py` is stripped to the
parser front-end (`ParserInputEncoder` + trivial fallback) — per §0g, tokenization
survives only as the parser's text reader. README rewritten around the living
mind/ architecture.

**What this buys:** the package now contains zero transformers — the learned parts
are the GRU controller family + the drive MLP, exactly matching the thesis
("weights hold zero information", depth through the loop). The M26 arc's next step
re-lands on the right seam: sense → *placed* M19–M25 coordinate → meaning-graph
filler handles (not the deleted meaning-bag channel).

### M28.0 — the WordNet-only baseline audit (Step A of dev/SEMANTIC_MAPPING_PLAN.md, MEASURED)

Plans now live in `dev/` (`SEMANTIC_MAPPING_PLAN.md` middle-term,
`ROADMAP_LONG_TERM.md` reference) — per the user: tightly scoped immediate steps,
hours not weeks, plans as repo files. Step A establishes the one table every Step B
signal is judged against (`scripts/probe_m28_baseline.py`, 3k gloss corpus, 234
axes, **17s total on CPU** — no compute wall anywhere in sight).

**Post-M27 reproduction — the cleanup broke nothing:** placement anchored
0.404 → placed **0.693** (α=0.7; = the M19.2 record); dictionary AUCs raw
synonym **0.867** / similar 0.748 / hypernym 0.727 / antonym 0.261, tanh
0.769 / 0.642 / 0.727 / 0.512 (= M19.4/M20.2 within split noise); tanh syn>ant
0.775 (probe_normalize; recorded 0.756 — split variance).

**POS-region breakdown (new):** held-out syn>rand is **uniform across regions** —
n-n 0.860, v-v 0.848, a-a 0.883, mixed 0.849 — so the "verb region is
under-signaled" hypothesis is **not supported** at this corpus scale for synonym
separation. What IS thin is **in-bucket antonym coverage** (v-v has 5 test antonym
pairs, a-a 10 — vs n-n 139) and the a-a sample overall (21 test synonym pairs).
Honest caveats: buckets use each word's *dominant* POS (first synset), so
adjective/verb pairs with noun homonyms leak into n-n/mixed; the gloss vocabulary
is noun-heavy by construction.

**Implication for Step B:** VerbNet/FrameNet are *not* triggered by verb-synonym
weakness (their gate condition in the plan). The real improvement targets, in
order: (1) **hypernym AUC 0.727** — flat through every milestone since M19.4;
(2) **antonym edge breadth** outside nouns (feeds both placement and the signed
edge store); (3) similar 0.748. The WordNet-remainder signals (entailment, cause,
pertainyms, domain links, also_see) land next, one at a time, each as a held-out
delta against this table.

### M28.1 — the nine-signal sweep (harness + 3 parallel agents; MEASURED, 2.5 wins / 6.5 negatives)

Built `ground/ablation.py` — one judge for every candidate signal: the M28.0 table
across train-split jitters (mean ± noise band; a signal "moves" a metric only
beyond 2× band), M24 enforced by construction (extra close edges filtered against
every scored test split; expanded antonym pairs scored on the side, never mixed
into the original held-out set). Added `hypernym_cos_auc` (placed-cosine readout
of held-out is_a) since anchored containment is placement-independent. Signals
plug in as `signal_<name>.extras()` modules; `scripts/ablate_signal.py` runs one.
Nine signals were then implemented + ablated by three parallel agents (two
Sonnet, one Haiku on the templated batch), ~20s per ablation.

**Winners (land in the default set via `signal_combined`):**
- **also_see** (close edges): similar_auc **+0.015** solo (5× band), nothing else
  beyond noise.
- **domain** (topic/region/usage as named feature axes): random overlap
  **−0.032** (0.253→0.221 — the M21 unrelated-separation objective) at a small
  real cost (hypernym_cos −0.006).
- **satellite-cluster indirect antonymy** (edge store, placement-inert):
  **2,553 new pairs** (~11× the held-out store; v-v and a-a breadth included);
  the existing space separates the new pairs from synonyms at **0.781** — better
  than the original antonym pairs (0.695), so the expansion agrees with the
  geometry. Precision is good-not-clean (compass-direction artifacts, off-first-
  sense pairs like minor/star) — filter before the M29 artifact.
- **Combined** (all three together): similar **+0.018**, random **−0.029**,
  hypernym_cos −0.008, everything else within noise — the wins compose, no
  interaction surprises.

**Negatives (documented, modules kept, edges NOT in the default set):**
- **genus–differentia gloss parse — the informative one.** Coverage 0.943,
  pointer-agreement 0.246 (properly independent of synset pointers). As a
  *symmetric close edge* it moves its target (hypernym_cos **+0.010**) but costs
  similar **−0.021** (7× band) and random +0.022, even after capping hub genera
  ("act", "person", "state" absorb everything — degree cap 3). The mechanism is
  the lesson: pulling word↔genus together makes **co-hyponyms** look similar.
  Hypernymy, like antonymy, is *directional structure* — it belongs in the
  relational store (and the M29 dictionary), not in closeness. Re-route there.
- **entailment**: regresses similar −0.006 (beyond band) — verb entailment is
  sequence, not sameness.
- **pertainym, cause**: null (within noise).
- **verbgroup** (solo ablation of the M19.0 field): null — and 872/1015 of its
  pairs are M24-dropped as collisions with held-out synonym pairs, i.e. the
  relation is mostly redundant with synonymy.

Suite: +33 tests (ablation 6 · genus 17 — includes validity gates · satellite 12
· batch 24 + relations regression, all green; full count 425+2 skips). The sweep
hit-rate (~30%) is what an honest held-out bar should produce. Next: M29 —
freeze the winning signal set, full-sense placement, publish the artifact + the
English sense→coordinate dictionary (satellite pairs filtered, genus edges as a
directed relation therein).

### M29 — USVS: the Universal Semantic Vector Space artifact (BUILT + MEASURED)

`ground/usvs.py` + `scripts/build_usvs.py` publish the semantic-mapping arc as
one versioned, deterministic, loadable artifact (**72s full build on CPU** — the
anticipated compute wall never materialized). What it combines (the "correct
parts" as measured, nothing else):
- **The placed word core** — 9,946 gloss-vocabulary words placed by deterministic
  propagation over the winning close set (synonym 22,856 + similar 3,988 +
  also_see 1,172 edges) on **607 named axes** (primes + attributes + lexnames +
  the M28.1 domain features, which contribute ~370 axes at 10k vocab).
- **The full sense layer** — all **117,659 WordNet synsets** grounded from their
  own gloss (M22 semantics, bit-equal to `gloss_prime_weights`, via a word-level
  decompose cache: 117k glosses in ~24s), stored sparse (444,965 nonzeros ≈ 3.8
  axes/sense).
- **The relational store** (what the coordinate cannot carry): **11,317 antonym
  edges with provenance tiers** (direct 
  / satellite_head / satellite_satellite; 3,383 in the default query tiers —
  tiering replaces naive filtering, consumers pick their floor) and **1,546
  directed genus edges** (the M28.1 lesson applied: hypernymy as relation, not
  closeness).
- **The dictionary** — `dictionary.jsonl.gz`: every sense with lemmas, gloss, and
  its named-axis signature. Artifact ≈ 20MB, loads in 3s, fingerprinted
  (`72b00a67c2b9daca`); blobs are git-ignored (deterministic rebuild), the pin
  (`usvs_meta.json`: fingerprint + axes + edge store + counts) is committed.

**Scale validation (held-out, harness):** the 10k-corpus baseline shows **no
small-corpus artifact** — synonym AUC 0.868→**0.885**, similar 0.747→**0.803**,
random 0.255 (flat) vs the 3k table; syn>ant eases 0.695→0.676 (883 vs 251 test
antonym pairs — a harder, better-sampled test). The combined winner set holds at
10k: similar +0.011, random −0.022, antonym AUC +0.005 (newly beyond band), the
known hypernym_cos cost −0.009. Production build uses ALL edges (no split);
quality numbers only ever come from the held-out harness — stated in the module
docstring.

**Honest spot-check caveats (recorded, not hidden):** lemma-level antonym tiers
inherit sense conflation (antonyms("hot") includes "bad" via the slang sense —
the WSD layer, not the store, is where that resolves); genus coverage is thin
(1,546 edges — the hub cap that saved similarity costs recall); dense-cosine
anecdotes can sit above the random mean (dog/justice 0.551 vs population 0.233)
— per-pair readings need the Null-mask/tanh lenses, the population statistics
are the contract. Gates: 6 USVS tests (determinism incl. fingerprint, save/load
identity, M22 grounding parity, tier structure, query API). USVS is the
substrate the roadmap's step (A)/(B) integrations consume next: sense →
signature → meaning-graph filler handles.

### M30 — the WSD gate on SemCor (USVS viability test #1; MEASURED — robust honest negative)

The first integration gate (dev/INTEGRATION_PLAN.md): do USVS sense signatures
carry enough signal to disambiguate in context? Tested **training-free** by
design (`scripts/probe_wsd_semcor.py`) so the number measures the artifact, not
a model: context = leave-one-out mean of the sentence's other annotated words'
MFS-sense signatures (gold labels never enter the context); prediction = most
similar candidate signature. Variants: raw cosine, **IDF-weighted axes** (the
M21 distinctiveness fix — ubiquitous primes like SOMETHING otherwise dominate),
and placed-core context.

**Measured (4,000 SemCor sentences = 44,682 instances, 37,353 polysemous, 17s):**

| resolver | all | polysemous | n | v | a | r |
|---|---|---|---|---|---|---|
| **MFS (floor)** | 0.772 | **0.727** | 0.749 | 0.584 | 0.849 | 0.747 |
| usvs (raw) | 0.422 | 0.309 | 0.319 | 0.153 | 0.469 | 0.341 |
| **usvs-idf (best)** | 0.459 | **0.353** | 0.390 | 0.169 | 0.486 | 0.381 |
| usvs-core | 0.446 | 0.337 | 0.369 | 0.152 | 0.476 | 0.383 |
| random | 0.393 | 0.274 | 0.266 | 0.159 | 0.424 | 0.322 |

**Verdict (per the pre-registered decision rule):** signatures carry *real but
weak* contextual signal — usvs-idf beats random by +0.079 everywhere except
verbs (chance: 0.169 vs 0.159) — but sit ~0.37 below the MFS floor. **MFS keeps
the perception slot.** Interpretation, honest in both directions: (1)
training-free Lesk-class resolvers historically land in exactly this band and
also lose to MFS, so this is "context matching without training doesn't clear
the known-hard bar," not "the space is broken" — USVS's validated strengths
(held-out similarity structure, M29) are a different, passing test. (2) The
binding constraint is the long-documented grounding coarseness: signatures
average ~3.8 axes/sense, verbs ground worst (the axis inventory is noun/adj-
heavy — M28.0 showed verb *synonym* structure fine, but per-sense signatures
can't separate verb senses at all). **The fix, if WSD-in-space is pursued, is
grounding depth (sourced explications; deeper gloss decomposition), not
resolver machinery** — and the trained coherence resolver only becomes worth
building after signatures show more signal. Next integration gate: M31 filler
handles (MFS senses), which does not depend on beating MFS here.

**M30b — the fallback-role stratification (user's architecture question).** The
intended design is parser-picks-first (structure: POS, slots — the M13
precedent), USVS only as the fallback for *unsure* words. Two clarifications,
measured: (1) M30 already granted the parser's biggest pick — candidates were
restricted to gold POS — so 0.353 is the *within-POS residual* the fallback
would own. (2) Stratifying by the frequency prior's own confidence (unsure =
no counts, top-2 within 1, or ratio > 0.6 — 11,071 of 37,353 instances):

| resolver | sure (26,282) | unsure (11,071) |
|---|---|---|
| MFS | 0.787 | **0.584** |
| usvs-idf | 0.341 | 0.382 |
| random | 0.243 | 0.348 |

USVS does relatively better exactly where the prior is weak (0.382 unsure vs
0.341 sure — the right *shape* for a fallback) but still loses the unsure slice
to MFS by 0.20, sitting only +0.034 over random there. **Verdict: the
signatures don't yet earn even the tiebreaker slot.** The joint
"every-sense-in-every-slot, maximize full-tree coherence" resolver (the §2
coherence design + the parser hypothesis lattice) remains the right eventual
architecture, but it is untestable on SemCor (the owned grammar doesn't parse
open text — its test arrives with the M32 ambiguity curriculum), and pairwise
slot-coherence cannot recover distinctions the representation doesn't carry —
at ~3.8 axes/sense the bottleneck is the signature, not the search. Perception
order of record: parser structure → MFS; grounding depth is the lever that
re-opens this gate.

**M30c — "is the comparison busted?" diagnostics (MEASURED — no; the task is).**
Three suspects, three verdicts: (1) **Sense vectors are NOT degenerate** —
intra-word top-2 sense signatures average cosine **0.406** (only 5.2% > 0.9),
vs random cross-word sense pairs 0.171: a word's senses are clearly separated
in the space, so a working resolver is *representable*. (2) **The comparison
machinery is fine** — the identical cosine/masking stack produces the held-out
0.885/0.803 similarity AUCs. (3) **Every training-free context construction
fails identically**: prime-signature bag 0.309, +IDF 0.353, placed-core context
0.337, and a v2 sense vector built from the gloss content words' *placed core
coordinates* (the artifact's strongest layer) 0.327 — all in one band, far
under MFS 0.727. Diagnosis: a **bag of co-occurring words in a lexical-
relatedness space is weakly coupled to sense selection** — topical relatedness
(what USVS encodes, validated) is simply different information from what picks
senses (frequency priors + selectional/syntactic constraints — which this
architecture already assigns to MFS + the parser). This matches the field's
history: Lesk-class and embedding-Lesk methods sit in exactly this band without
supervision. **Standing conclusion: stop pointing USVS at context-WSD.** Its
validated strengths (similarity structure, antonym store, unrelated separation)
are what M31/M32 consume; sense-picking belongs to parser structure + MFS, and
the in-architecture WSD test (selectional constraints in parse slots) arrives
with the M32 ambiguity curriculum where the owned grammar actually parses.

**M30d — external yardstick: SimLex-999 (MEASURED — USVS matches corpus-trained
embeddings, untrained and transparent).** SimLex-999 measures *similarity*
(dog/wolf), not association (dog/leash) — the property this architecture needs
and the one distributional models are weakest at. `scripts/probe_simlex.py`:
Spearman ρ vs human judgments — plain cosine **0.333** (999/999 coverage),
**+ the signed antonym-edge store 0.380** (the two-part design visibly earning
its keep: +0.047 from relational correction at comparison time), **placed core
layer + edges 0.417** (n=738). Published baselines: word2vec (billions of
training tokens) ~0.41–0.44, GloVe ~0.37–0.41, count/PMI ~0.30, WordNet
path+IC graph measures ~0.50–0.58, human ceiling ~0.67. So: **USVS's core sits
at word2vec level with zero corpus training, full determinism, and every axis
named** — the confidence answer for "is this better than the alternative FOR
WHAT WE WANT": opaque embeddings fail the architecture's requirements
categorically (weights-as-knowledge, no audit, language-locked) while USVS now
matches them on the one number they'd compete on. Honest caveats: WordNet
path-based graph measures still beat vector cosine on SimLex nouns (~0.55) —
but they are pairwise graph algorithms, not vectors a model can consume or
compose, and the taxonomic signal they exploit is available to us as future
placement signal; USVS coverage/nuance on rare words is thinner than
billion-token models.

### M31 — USVS as filler/handle vectors: BOTH extrinsic gates PASS (MEASURED — the arc pays downstream)

The payoff experiment the whole grounding arc existed for. Two gates, two
Sonnet agents, main session = bridge + review (`src/nsm_ct/usvs_bridge.py`:
deterministic axis-NAME-keyed projection of any USVS coordinate to any d;
structure survives projection — dog~puppy 0.93 vs dog~justice 0.55 @ d256).

**Gate (a) — concept dereference (`scripts/probe_m31_handles.py`), 500-word
index, +5% noise, both dims:**

| handles | d | top-1 | median margin |
|---|---|---|---|
| status quo (label-TPR) | 256 | 0.316 | 0.0004 |
| **USVS** | 256 | **0.962** | **0.1263** |
| status quo | 512 | 0.274 | 0.0004 |
| **USVS** | 512 | **0.948** | **0.1230** |

The honest re-read of §0j this forces: the old "7/7 top-1, margin 0.034" was a
**7-node index** — at a realistic 500 concepts the label handles collapse
(top-1 0.32, margins ~0); they were never going to scale. USVS handles give
~300× the margin and 0.95+ retrieval under noise.

**Gate (b) — the consumer (clause reactor, `train_clause.py
--meaning-source`), identical config/seed both arms (192 train / 48 val eps,
60 epochs — a quick-budget config, so compare the DELTA not the absolutes):**
explication baseline val **0.521** → USVS fillers val **0.812** (+0.29);
per-level: L1 0.43→0.71, L2 0.33→0.83, L6 0.29→1.00, L7-res 0.33→1.00 (L8
0.67→0.50 on n=6 — watch, don't panic). Entity variables stay atomic; unknown
words fall back to explication; default remains "explication" until the
full-budget retrain (below). Tests: 19 green (bridge 6 + reactor-M31 +
clause_reactor regression).

**Verdict: the extrinsic-validation rule is finally satisfied in the strong
direction — the semantic space measurably improves both memory addressing and
task accuracy in a downstream consumer.** Follow-ons, in order: (1) flip
`meaning_source` default to usvs + full-budget retrain of the reactor/psyche
baselines (the §0h/§0i numbers get re-recorded); (2) M32 ambiguity curriculum
(sense-level fillers via `usvs_sense_handle` are ready); (3) mind/-side
adoption (meaning-graph handles at write time). Ops note: both agents stalled
awaiting their own background-task notifications after their runs completed —
results were collected directly from the run outputs; agent checkpoint-resume
needs watching in future fan-outs.

**M31b — full budget confirms; default flipped (MEASURED).** At the §0i budget
(480 eps / 80 epochs / dim 48 / max_level 8, same seed both arms):
explication val **0.698** → USVS val **0.885** (+0.19; per-level: L4 1.00,
L6 0.94, L5 0.80 — USVS wins every level except L8 tie 0.70).
`meaning_source` now **defaults to "usvs"** everywhere (explication stays as
the tested fallback for USVS-unknown words). **Recorded drift, not hidden:**
today's explication rerun (0.698) is below the historical §0i record (~0.86)
on current code — the arms are internally comparable (identical code/seed) and
the USVS arm ≈ matches the old record's level, but the baseline shift is
unexplained (suspects: M26-era episode/meaning changes, torch version) and
stands as an open observation. 14 reactor tests green.

### M32 — the ambiguity curriculum: sense choice becomes a SCORED task (MEASURED, gate PASS +0.75)

The first episodes whose answers depend on which SENSE of a homograph is meant
(`episode.generate_ambiguity_episodes`; families: bank riverbank/institution,
bat animal/club, plant factory/flora, organ body/instrument; 50% of episodes
rigged so MFS is the WRONG reading). Each episode carries gold + MFS sense ids
(live-verified synsets). The no-training probe
(`scripts/probe_m32_ambiguity.py`, 400 episodes, d=256) grounds the homograph
three ways and checks which answer option wins in USVS space:

| grounding | all | sense-flipped half | unflipped half |
|---|---|---|---|
| word-level (sense-blind) | 0.522 | 0.510 | 0.534 |
| MFS sense | 0.530 | **0.247** | 0.796 |
| **gold sense** | **0.895** | **1.000** | 0.796 |

Three facts in one table: the curriculum's trap works (MFS collapses to 0.25
exactly where it should), **USVS sense vectors are sufficient — choose the
right sense and the answer follows (ceiling 1.00)**, and nothing currently does
the choosing. The +0.753 gold−MFS gap on the flipped half is the prize the
future tree-coherence disambiguator (the §2 design: candidate parses × USVS
vectors → coherent assignment) competes for — it finally has a benchmark,
a floor, and a ceiling. 9 tests green; existing levels untouched.

### M33 — mind-line USVS handle adoption (MEASURED through the graph API)

`meaning_graph.py`/`collapse.py` gain an opt-in `handle_fn` hook (default None =
byte-identical old behavior; CLAUSE/REFERENT/OPERATOR nodes untouched; unknown
words fall back). Probe through the real graph API (300 concepts filed via
`collapse`, +5% noise, d=256): label handles top-1 **0.373** / median margin
0.0005 → USVS hook **0.987** / margin **0.159**. The M31 win transfers to the
line the user talks to. 20 tests green (new + collapse/meaning-graph/psyche
regressions).

### M34 — the sense chooser EXISTS: first trained WSD, 3/4 cross-word transfer (MEASURED)

`sense_chooser.py` (24.7k params — a policy, not a knowledge store): scores
each candidate sense's USVS vector against a context vector (mean USVS handle
of the episode's other content words), trained SUPERVISED on M32's ostensive
gold labels (`train_sense_chooser.py`, ~65s CPU).
- **In-distribution: the M32 gap closes 100%** — flipped-half benchmark 1.000
  (= the oracle ceiling; floor was MFS 0.247@d256). Expected — 4 homographs are
  memorizable; the number that matters is:
- **Leave-one-family-out (train 3 families, test the unseen 4th):** bat/plant/
  organ held out → **1.000 flipped benchmark on never-seen homographs** — the
  chooser learned a transferable context→sense-vector matching function, not
  word memorization. **bank held out → 0.000** (= the no-resolution floor):
  one family's context geometry doesn't transfer; honest 3/4.
- Methodological catch (recorded): `usvs_sense_handle`'s projection is lossier
  at small d — at d=64 even the GOLD oracle drops to 0.760 on the flipped half;
  d=128 restores the 1.000 ceiling; all comparisons re-baselined same-d.
9 tests green. Next rungs: more families (the M32 generator scales), context
from parse-tree neighbors instead of bag (the §2 tree-coherence design), and
weaning from gold labels onto answer-only signal.

### M35 — template-generalization audit: NO overfitting; the parser is the surface bottleneck (MEASURED)

The user's hypothesis (reactor gains might be template memorization) tested
head-on. `curriculum2.py`: same facts, two DISJOINT surface template sets, each
candidate template verified through the REAL parser with exact subject/place
binding checks (catches passive false-positives). **12 of ~30 phrasings kept
(100% parse success); 9 dropped** — passives, wh-clefts, unknown verbs all fail
the parser, not the model. Three-arm experiment (usvs fillers, dim 48, same
seed): train-A/val-A **0.850**, train-A/**val-B (unseen templates) 0.850** —
**zero overfit gap**, bit-identical per-level; mixed-train 0.833 (noise).
Architecturally explained: perception grounds every sentence to
(entity, relation, meaning-vector) BEFORE the model sees it, so surface form
never reaches the learned part. Caveats recorded: question phrasing untested;
parser-rejected constructions untested by construction. Deliverables for the
scaling push: the verified 12-template inventory + 61-noun place vocabulary +
the parse-verification helper. 13 tests green.

### M36 — parser robustness round 1 (interactive review loop; 12→18 constructions, MEASURED)

Run as a two-phase dev loop (Sonnet diagnoses/proposes → review → implement):
phase 1 traced all 9 dropped phrasings to exact failure points (empirically,
with a hypothesis tracer — it also corrected the briefing: the prep→PLACE
frame map lives in `clause.py:_PREP_RELATION`, not mind/coref.py, and the
429-sentence membrane parity suite never touches quantum_parser). Findings:
5 were **lexicon gaps** (inside/near missing as prepositions; came/sat/found
unknown irregular verbs → default to NOUN → sentence loses its verb), 1 was a
missing frame entry ("by" — the parse was fine all along), 1 needed a scoped
consumer rule ("entered" — bare-object locatives), 2 need real grammar
architecture. Approved + implemented: tagger entries (inside, near → ADP;
come/came, sit/sat families → VERB), frame entries (by, near → PLACE, with the
passive-agent landmine documented in code), and `_LOCATIVE_TRANSITIVE_VERBS =
{entered, exited, reached}` consulted only when no PP exists ("left" excluded
as ambiguous). **Verified: all 6 targets now parse with correct SUBJECT/PLACE;
both WSD-family collision sentences parse better than before ("sat on the
bank" gains its verb); quantum_parser suite 82/82; affected consumer tests
49/49; template inventory 12→18 (new set "C", A/B frozen for
reproducibility).** Deferred with reasons on record: "is located" (scorer
tie-break picks a SUBJECT-less hypothesis at a 0.740 tie — aux1 blast radius),
wh-clefts (no equative ruleset), passive voice (SubType.PASSIVE defined,
never used by any rule — the biggest missing construction).

### M37 (pending runs) — 31 homograph families + the first scaling curve

Interrupted by a usage-limit kill mid-execution; all code landed and is
tested (31 families live in `episode.py` with sense pre-flight checks;
`curriculum2.scaled` mode + `probe_scaled_training.py` with 23 curriculum
tests green). The re-runs were **killed at the user's request** — the
full-scale runs were overloading the machine — so M37 closes on partials,
salvaged from `runs/m37_*.log`:

- **Sense chooser, 31 families, in-distribution (COMPLETE):** d=128,
  24,705 params, 6,200 train / 1,550 val episodes. sense_acc **1.000** on
  all subsets; flipped-half benchmark_acc **0.817** = the same-d GOLD
  ceiling exactly (same-d MFS floor 0.287) → **100% of the floor→ceiling
  gap closed** in-distribution at 31 families (was 4 families in M34).
  The leave-one-family-out N-rotation (the generalization question — does
  the bank-style transfer failure dissolve with 30 training families?)
  never produced a row before the kill. Re-run solo (`nice -19`, 47 min,
  full table in `runs/m37_chooser.log`): **mean flipped-half benchmark
  0.612 across 31/31 rotations** — but the raw mean undersells it, because
  the same-d ceiling column shows the task isn't even solvable-with-gold
  for every family at d=128:

  - **7 rotations have no headroom** (same-d GOLD ceiling ≈ floor or 0):
    ball, court, jam, racket, yard (ceiling 0.000 — gold sense handles at
    d=128 fail these families outright; yard's floor is 1.000 with ceiling
    0.000, i.e. gold does *worse* than MFS there), plus cell and hood
    (floor = ceiling ≈ 0.51). These are **projection/handle failures, not
    chooser failures** — consistent with Part 2's overall same-d ceiling
    of 0.817 (~18% of flipped items unsolvable even with gold at d=128).
  - **On the 16 rotations with real headroom** (floor 0, ceiling 1): mean
    0.680; **9 transfer perfectly** (bark, bill, iron, nail, palm, spring,
    star, tie + mouse 0.987), pupil 0.884, ring 0.651, **bank 0.355 — the
    M34 bank failure half-dissolves** (was hard 0.000 at 3 training
    families), and 4 stay at zero: date, fan, organ, pool.
  - 5 rotations are trivial (floor = ceiling = 1.000: bass, bat, pitcher,
    plant, staff — MFS already solves them; no signal either way).

  Verdict: more training families genuinely improve zero-shot transfer
  (0/1 → ~10/16 on solvable families), but it's not universal — and the
  d=128 projection ceiling is now the measured bottleneck for ~1/5 of
  families, which points at d=256 handles (canonical M32 ceiling was
  1.000 at d=256) before blaming the chooser further.
- **Scaling grid (1 of 7 cells):** 480 episodes @ dim 48, 80 epochs →
  val 0.750 in 2.6 min (169k params); per-level L1 0.80 / L2 0.92 /
  L3 0.72 / L4 0.89 / **L5 0.59 / L6 0.60** — the recency/overwrite levels
  are where a small model starves first. Data-vs-capacity curve unmeasured.

Rerun recipe when compute allows: both commands are in `runs/` log headers;
budget the grid down (fewer epochs or 3 cells) before retrying locally.

### M38 — parser round 2: passive voice + the SUBJECT tie-break (interactive loop; 18→20, MEASURED)

Same two-phase dev loop as M36; the design phase validated everything in
scratch copies before proposing, and surfaced **three latent pre-existing
bugs** in the grammar machinery: `inf1` and `neg1` had unscoped MODIFIER
patterns (silently consuming "can be"/"be" before aux composition — the same
bug family as the wh-cleft's `verb1`/`rel2` collision, now the round-3
candidate), and `Hypothesis.is_equivalent` ignored node flags (dedup could
silently drop voice marking).

**Landed:** (1) the scorer tie-break — `completeness_key` (has-SUBJECT first,
then core-role count) as a **secondary sort key only**, `.score` untouched;
root cause on record: the structural score counts edges but is blind to which
relations exist, so SUBJECT-less and SUBJECT-bearing parses tie exactly.
Isolated validation: 9/10 corpus sentences byte-identical, only "is located"
changes (gains its SUBJECT). (2) **Passive voice** — the grammar's first:
new `SubType.PAST_PARTICIPLE` (reusing PARTICIPLE broke gerunds — caught in
scratch), one aux1 rule anchored on the participle, found→VERB, plus the two
companion scoping fixes; "mary can be found in the garden" now parses with
SUBJECT=mary through the modal chain. (3) The `is_equivalent` flags fix.
**Verified:** quantum_parser 82/82 at every increment; combined trace diff
13/17 byte-identical with the 2 targets fixed, wh-cleft untouched, one
spurious duplicate merged; consumer battery 37/37; curriculum negation
("is not in") and disjunction re-verified by hand; templates **18→20**
("is located", "can be found" promoted to set C; only the wh-cleft remains
dropped). Honest flags on record: "does not run" (do-support negation)
changed laterally — was nonsensical-complete, now incomplete, neither correct,
nothing asserts on it; "PASSIVE marker reliably wins its dedup tie" scoped
out (SUBJECT-binding is correct either way); agentive-"by" guard ticketed at
the `_PREP_RELATION` landmine (blocked on the adapter carrying node flags +
an AGENT role downstream).

### M39 — parser stress battery: the round-3 map (MEASURED)

`scripts/probe_parser_stress.py`: 34 sentences across 10 categories, run
through the real pipeline (`ParserInputEncoder._parse_graph` →
`extract_discourse`), each case asserted against expected role→token pairs
(not just "a clause came out"). Outcomes: PASS / WRONG (clauses came out,
structure wrong) / NO_CLAUSE (parsed, nothing extracted) / NO_PARSE.

| category   | pass | wrong | no_clause | rate |
|------------|-----:|------:|----------:|------|
| sanity     | 3    | 0     | 0         | 3/3  |
| passive    | 2    | 0     | 2         | 2/4  |
| prep       | 5    | 0     | 3         | 5/8  |
| subord     | 1    | 1     | 1         | 1/3  |
| relative   | 0    | 1     | 1         | 0/2  |
| pronoun    | 2    | 1     | 0         | 2/3  |
| conj       | 1    | 2     | 0         | 1/3  |
| negation   | 2    | 0     | 0         | 2/2  |
| question   | 0    | 0     | 2         | 0/2  |
| open_vocab | 4    | 0     | 0         | 4/4  |
| **TOTAL**  | 20   |       |           | **20/34 = 0.59** |

NO_PARSE never happened — the parser always produces *something*; failures
are downstream structure. Root causes, diagnosed per cluster:

1. **Missing WORD_TAG_DICT entries** (cheapest fix, biggest single bucket):
   `behind`/`beside` (ADP) kill two prep cases; `moved`/`broken` (need
   PAST_PARTICIPLE) kill two passive cases; `thinks`/`knows` absent (VERB).
   Twist: "mary **knows** that john is in the kitchen" PASSES — the unknown
   matrix verb drops out and the embedded clause extracts clean — while
   known-verb "mary **said** that the ball is in the garden" goes WRONG
   (the embedded PP `in the garden` attaches to `said`; the embedded clause
   vanishes). Complement clauses aren't handled; they're accidentally
   survivable only when the matrix verb is unknown.
2. **POS-ambiguity mis-tags**: "the ball came from the **shed**" — `shed`
   tagged VERB, PP never assembles, zero clauses. Scorer/tagger issue, not
   a missing prep (`from` is ADP and maps to SOURCE fine).
3. **No sentence segmentation**: "mary went to the garden . **she found the
   ball .**" parses as ONE hypothesis — `she` becomes OBJECT of `went`,
   sentence 2 never yields a clause. Training feeds sentences one at a time
   so the curriculum never sees this; any real-passage input needs an
   upstream splitter (trivial) or parser-side segmentation.
4. **Coordination is broken at extraction**: "mary **and** john are in the
   garden" → SUBJECT = literally `and` (twice). "X is in A and Y is in B" →
   second clause gets PLACE=`bat` (the conjunct's subject). COORDINATION
   edges exist in the graph; `extract_discourse` doesn't consume them.
5. **Relative clauses invert headedness**: "the ball that mary found is in
   the garden" → single clause pred=`found` subj=`mary` place=`garden`; the
   matrix clause (ball-in-garden) is lost entirely.
6. **Wh-questions**: both NO_CLAUSE (known; wh-cleft was deferred in M38
   with the verb1/rel2 collision as the prerequisite).

Genuinely good news: **open_vocab 4/4** — zeppelin/wombat/gertrude/quokka
sentences all extract perfectly (the unknown-word POS guesser + scorer
handle novel nouns), so the "ranked #1 gap for real text" from M38 is
narrower than feared: it's specific *known-word* gaps and ambiguity, not a
general OOV collapse. Negation 2/2 at clause level (the NEGATIVE flag still
isn't carried through the adapter — polarity is invisible downstream; same
adapter-flags ticket as agentive-"by").

Round-3 priority order this implies: (1) dict batch + past participles —
mechanical, unblocks 4 cases; (2) coordination in `extract_discourse` —
COORDINATION edges are already in the graph; (3) sentence splitter at the
encoder seam; (4) complement/relative clause scoping in the grammar;
(5) wh-questions last (needs the verb1/rel2 collision fix).

### M40 — the chooser on raw USVS axes: projection exonerated, grounding indicted (MEASURED)

Follow-up to M37's verdict, at the user's suggestion that projecting at all
was philosophically wrong for the chooser: `usvs_bridge` gains a raw mode
(`d == n_axes` → the handle IS the unit-normalized 607-axis coordinate,
transparent, zero loss) and `train_sense_chooser.py --d 0` re-ran the whole
protocol on it (116.7k params from input width alone; batch capped at 4096;
2.8 h niced, `runs/m40_chooser_raw.log`).

**M37's "d=256 projection bottleneck" hypothesis is FALSIFIED.** The raw
607-dim gold-sense ceiling on the 31-family flipped half is **0.813** —
statistically the same as d=128's 0.817. The ~19% ceiling deficit was never
projection loss; it is in the USVS signatures themselves. (The canonical
1.000 ceiling was measured on the original 4 families only; at 31 families
the geometry has real holes.) Corollary: the ceiling-0 families — **ball,
court, racket, yard** (0.000 even raw; cell/mole/hood stuck ≈ 0.5) — are
**grounding failures**: the gold sense's own signature cannot rank the MC
options. Same disease class as M30's verdict (grounding depth, not
machinery).

In-distribution (Part 2): flipped 0.813 = raw ceiling exactly, gap again
100% closed — the chooser saturates whatever the substrate supports, at
every d tested.

Leave-one-family-out: **mean 0.662 raw vs 0.612 projected**, with large
family-level swings both ways:

- **bank 0.355 → 1.000** — M34's canonical transfer failure fully
  dissolves in raw space. fan 0→1.000, pool 0→1.000, pupil 0.884→1.000,
  jam 0→0.254 (jam's ceiling also went 0→1.000: solvable raw, weakly
  transferred).
- Regressions: **pitcher 1.000 → 0.000** (floor AND ceiling are 1.000 —
  the chooser picked a third sense, neither MFS nor gold, whose signature
  loses the ranking; a genuine transfer error, invisible at d=128),
  star 1.000→0.740, ring 0.651→0.498.
- Persistent genuine transfer failures: date 0.000, organ 0.090 (both
  have ceiling 1.000 raw — the signal exists; the policy doesn't find it
  zero-shot).

Verdict: raw named axes are the right substrate for the chooser (no loss,
more transparent, modestly better transfer) and the open problems are now
cleanly split in two: (a) **grounding depth** for the ceiling-0 families —
a USVS/dictionary problem; (b) **zero-shot policy transfer** for
date/organ/jam — a chooser/curriculum problem. Neither is a dimensionality
problem; that door closes.

### M41 — the WordNet-backed tag lexicon: hand-listing ends (MEASURED)

The user's call on seeing M39: "we were hand tagging????" — WORD_TAG_DICT
was ~200 hand-listed words while the project already owns WordNet/USVS with
117k senses. `scripts/build_parser_lexicon.py` now generates
`quantum_parser/data/en_lexicon.json.gz` (159,993 entries, 15,090 multi-POS,
0.59 MB gz, fingerprint 80fa20bf6b16c13f, committed): every single-word
WordNet lemma, every POS it can be, frequency-ordered (SemCor counts), plus
generated inflections carrying morphological subtypes — plurals [PLURAL],
3sg [THIRD_PERSON, SINGULAR], -ing [PARTICIPLE], past/-ed
[PAST_PARTICIPLE] (irregulars from WordNet's exc lists; past vs participle
indistinguishable there, both flagged — documented).

Wiring precedence in `pos_tagger.py`: hand dict (now genuinely closed-class:
determiners/pronouns/aux/prepositions — 14 prepositions added: behind,
beside, above, below, beneath, across, along, around, onto, upon, against,
beyond, outside, off) → PROPN capitalization guard → lexicon entry[0] →
old suffix heuristics (lexicon absent = old behavior, never an error).
Multi-POS words expose ALL their tags to the hypothesis lattice via
get_possible_tags — the scorer picks the reading that completes a tree
("every option in every slot", the project's stated design).

Stress battery: **20/34 → 23/34**. prep 5/8 → **8/8** ("came from the shed"
fixed by NOUN/VERB branching, not by a dict entry). passive 2/4 → 3/4
(moved). One honest regression: subord 1/3 → 0/3 — "knows that john…"
previously PASSED by accident (unknown matrix verb dropped out, embedded
clause extracted clean); with "knows" now a real verb it fails structurally
like "said". Complement clauses are confirmed pure grammar work. Suites:
quantum_parser 82 → 91 (9 new lexicon tests), curriculum templates A 6/6,
B 6/6, C 8/8 — zero regression.

Remaining parser map after M41: complement clauses (uniform failure mode),
relative-clause headedness, coordination in extract_discourse, sentence
splitter, bare passives ("the window was broken" — no clause), wh-questions.
All grammar/extraction tier — the dict tier is closed for good.

### M42 — grounding-depth diagnosis: why the dead families are dead (MEASURED)

`scripts/probe_grounding_depth.py` on the 7 families M40 showed dead/stuck
even with gold senses in raw 607-axis space. The mechanism is uniform:
**sense signatures are shallow (2-6 active axes) and the wrong answer wins
through GENERIC axes.** court.n.04 loses to "judge" via SOMETHING=0.449;
hood.n.08 ({lex:noun.artifact, BODY} — two axes total) loses to "car" via
noun.artifact=0.420; yard.n.02 loses via SOMETHING=0.353. SOMETHING appears
in 66,875 of 117,659 sense signatures (57%) — cosine on raw signatures is
substantially a genericity match. (M30 called this: "much collapses to
SOMETHING". Now it's the measured kill mechanism.)

Cheap-fix test — IDF axis weighting (df over all sense signatures,
idf = log(N/(1+df)), elementwise reweight, renormalize): sense-level option
ranking across all 31 families goes **50/62 → 53/62**; cell and racket
fully revive; ball/court/hood/mole/yard do NOT — their signatures simply
lack content axes to reweight. Also noted: the benchmark is associative
(sense → related word, ball.n.01 → "game"), which shallow signatures can't
support even in principle.

Verdict: (a) IDF weighting is a real, cheap metric improvement — candidate
to land as an opt-in in USVS similarity + chooser inputs behind the usual
ablation; (b) the remaining 5 families need **deeper explications** (their
glosses mention head/face/protect, tennis/hit, dance/party — content the
current decomposition drops to SOMETHING). Grounding depth is now the
substrate's #1 open item, with a concrete casualty list to test against.

### M43 — the first scaling curve: data-limited, not capacity-limited (MEASURED, data axis)

`probe_scaled_training.py` (curriculum2 scaled mode: mixed A+B templates,
61-noun pool, 4-8 facts / 2-4 entities per episode), dim 48 / 169k params
fixed, 80 epochs:

| episodes | val_acc | minutes |
|---------:|--------:|--------:|
| 480      | 0.750   | 2.4     |
| 1000     | 0.825   | 6.3     |
| 2000     | 0.885   | 15.9    |

Monotone data scaling at fixed capacity; the (2000, dim=48) cell reproduced
0.885 exactly on a second run (deterministic harness). Per-level at 2000:
L1 0.88 / L2 0.95 / L3 0.93 / L4 0.91 / **L5 0.75** / L6 0.88 — recency
under distractors is the residual weakness, and it is the level that
improved most with data (0.59 → 0.75 from 480 to 2000). The capacity axis
(dim 64/96 at 2000 episodes) was killed twice by host crashes and is
**abandoned on this machine by user decision** — the run overloads WSL. The
data-axis answer stands: the reactor at this size wants more episodes, not
more width. `--axis` flag added to the probe for partial reruns elsewhere.

### M44 — gloss content-word enrichment: honest negative, M24 earns its keep (MEASURED)

Sonnet-agent prototype of the M42 "deeper explications" lever
(`ground/explication.py`, opt-in `enriched_sense_dense(usvs, sid, alpha)`,
default-off identity; `scripts/probe_explication_depth.py`; 5 tests).

Root-cause trace (hood.n.08, "a headdress that protects the head and
face"): `sense_prime_weights` keeps decomposition leaves ONLY if they are
literal NSM primes — `head` decomposes in one hop to the named axis HEAD,
which EXISTS in usvs.axes but is not in the prime whitelist, so it is
**discarded**; `protects`→body survives by coincidence. The 57%
SOMETHING-saturation has the same shape: generic primes win because
decomposition keeps whatever whitelisted prime it stumbles into and throws
away the specific named axes it actually reaches.

The enrichment (blend gloss content words' placed-core coordinates into the
sense vector): **negative**. With the M24 leakage guard ON (answer word
excluded when it appears verbatim in the gloss), enriched+IDF peaks at
51/62 sense rankings vs **53/62 for IDF alone** — two new regressions, zero
casualties revived. With the guard OFF it "wins" (55/62) — but every flip
is literal leakage (court.n.04's gloss contains "game", ball.n.09's
contains "dance", fan/tie likewise): the exact artifact M24 exists to
catch. SemCor subsample: enriched+IDF +0.003 over usvs-idf — noise.

Standing conclusions: (1) the depth lever is NOT gloss-blending at query
time — it is the prime whitelist in `sense_prime_weights`: molecule axes
the decomposition already reaches (HEAD, FACE, ...) should survive into
signatures. That is a build-path change with a casualty-list gate — the
concrete next experiment. (2) M24 flagged a fake win again; the rule stays.

### M45 — parser round 3 (agent pair): stress battery 23/34 → 29/34 (MEASURED)

Two Sonnet agents with disjoint file ownership (both stopped mid-final-
verification by the host-crash cycle; their on-disk work verified and
landed by the main session). Four changes:

1. **Quantum-branching engine fix** (`quantum_parser.py`): when several
   rule matches hit one pass, ANY ambiguous anchor previously caused every
   OTHER anchor's independent transformation to be dropped on each branch.
   Now independent (single-match) anchors apply on every branch and only
   the Cartesian product of genuinely ambiguous anchors' alternatives is
   explored. This is what unlocked complement clauses.
2. **PUNCT tagging** (`pos_tagger.py`): bare punctuation tokens tagged
   PUNCT instead of falling through to default-NOUN — kills the phantom
   NOMINAL "." that could satisfy rules' "NOMINAL after" patterns
   (the PLACE="." artifact).
3. **Tie-break extension** (`scorer.py` completeness_key): now
   (has_subject, subject_count, other_core) — a reading with two complete
   clauses beats one that folded the second subject into an OBJECT.
   Backward compatible (first element unchanged).
4. **Extraction upgrades** (`clause.py`, `input_encoder.py`): sentence
   splitter at the encoder seam (multi-sentence input parsed per sentence,
   HypGraphs merged with offset indices; single-sentence path byte-
   identical), coordinated subjects ("mary and john" → one clause per
   conjunct via COORDINATION edges), and secondary-sentence fact clauses
   (later sentences arrive as extra SUBJECT edges).

Battery: subord 0/3→3/3, pronoun 2/3→3/3, conj 1/3→2/3, relative 0/2→1/2;
total **29/34 (0.85)**. Remaining: bare passive ("the window was broken" —
still no clause), one relative ("the man who came..."), one conj, and
wh-questions (out of scope, verb1/rel2 collision). Gates: quantum_parser
91/91, curriculum templates A/B/C 20/20, clause/encoder test subset 41/41.
TODO: the agents were stopped before writing regression tests for the new
behaviors — battery + suites cover them, but dedicated tests should land
with round 4.

### M46 — senses regrounded IN the space: USVS is the metalanguage (MEASURED: parity, adopted on architectural grounds)

User's call, closing the M44 thread: "USVS is our Semantic Metalanguage, we
don't need a vestigial lossy one." Sense signatures are no longer built by
gloss-to-prime decomposition; `sense_usvs_weights` (ground/usvs.py) grounds
each sense **in the placed space itself** — classical genus + differentia:
unit coordinates of its direct hypernyms' lemmas (w=1.0) blended with its
gloss content words' coordinates (w=0.7), each contributing word discounted
by 1/sqrt(n_senses) and each component scaled by its mean confidence,
top-24-axis sparsified, plus the lex: axis as before. Prime decomposition
survives ONLY as the fallback (1,208 of 117,659 senses = 1.0%).
`build_usvs(sense_grounding="usvs"|"primes")` keeps the legacy path for A/B.
New artifact: fingerprint **e0daef638b640dd5**, nnz 2,784,733 (~24 axes per
sense, was ~4), 75s build.

Design lesson measured on the way: including the synset's OWN lemma
coordinates (v1) cratered same-word discrimination (62-proxy 50→44) — a
word's coordinate is a mixture over its senses, so the headword drags every
sense toward the dominant one. The thing being defined must not appear in
its own definition; genus + differentia only.

Gates — parity across the board, reported straight:
- 62-sense ranking proxy: new 50 plain / 52 idf vs old 50 / 53.
- SemCor subsample (2,056 inst): poly 0.237 vs 0.232 (MFS 0.507). Noise.
- Episode-level flipped-half GOLD ceiling (3,450 eps): **0.810 vs 0.812**.
  ball and racket REVIVE (0→1); pitcher and seal break (1→0/0.48);
  court/yard/cell/hood/mole unchanged. court + yard fail under BOTH
  groundings → remaining suspicion moves to the placed word coordinates /
  association geometry, not sense signatures.
- Chooser in-distribution spot check (raw d=607, new artifact): sense_acc
  1.000, benchmark 0.796 = the new same-d ceiling exactly — the trained
  consumer still saturates whatever the substrate supports.

Adopted as default despite parity: one grounding mechanism instead of two,
senses live on the same axes words do (~24 vs ~4), transparency improves
(ball.n.09 now reads attr:dancing where it read SOMETHING), and the lossy
whitelist path is retired to OOV fallback. The extrinsic-validation house
rule is noted, not waived: this is recorded as an architectural
simplification at measured parity, NOT a performance win. Full leave-one-
family-out chooser revalidation deferred to better hardware (training-scale
runs crash this box).

### M47 — parser round 4 (Sonnet agent): stress battery 29/34 → 32/34 (MEASURED)

All three remaining non-question failures fixed at their diagnosed layer,
plus the regression tests M45 owed. (1) Bare passive ("the window was
broken"): the GRAMMAR was fine — SUBJECT edge present all along; every
branch of `_primary_discourse` demanded a PP or object before emitting a
clause, so subject-only predicates fell through to nothing. Fallback added:
a bare SUBJECT-only clause. (2) Object-gap relatives ("the ball that mary
found is in the garden"): rel1 only knew subject-gap ("the man who came");
one new 3-element rule (NOMINAL + NOMINAL[RELATIVE] + NOMINAL + PREDICATE)
leaves the head noun unconsumed so clause1 builds the matrix clause, and
the M45 secondary-clause walker recovers it. (3) Clausal coordination
("...garden and the bat is in the shed"): noun2 runs before any CLAUSE
exists and can't distinguish "garden and bat" from a real coordinated NP —
a genuine ordering ambiguity. Engine-level lookahead judged too invasive;
extraction instead recovers the orphan (a subject-less PREDICATE with its
own PP, right of a COORDINATION group → rightmost conjunct is its subject),
verified inert on legitimate coordinated-subject/value shapes.

Battery **32/34 (0.94)** — passive 4/4, relative 2/2, conj 3/3; only
wh-questions remain (out of scope, verb1/rel2). Suites: quantum_parser
98/98 (7 new), clause/encoder subset 50/50 (12 new), templates 20/20.
Parser is now ahead of the curriculum's needs; next parser work should be
driven by whatever the scaled corpus actually fails on, not by this
battery (saturated except questions).

### M48 — the real-text failure map: 89% subject-coverage, 2-arg ceiling exposed (MEASURED)

`scripts/probe_realtext.py` (Sonnet agent; deterministic, rerunnable in
~70s): first 400 SemCor sentences of length 5-25 tokens through the REAL
pipeline. **CLAUSE_OK 356/400 = 89.0%** (a clause with a real SUBJECT came
out); NO_CLAUSE 43; NO_PARSE 1 (a recursion-limit crash in the hypothesis
search — engine reliability item, caught and degraded gracefully).

All 44 failures taxonomized by hand from their graphs:
- APPOSITIVE_INTERRUPT 13 — "pete rozelle, the league commissioner,
  pointed out" — a comma/dash/paren span between subject and verb defeats
  SUBJECT attachment (strict adjacency). Grammar rule; most fixable.
- QUANTIFIER_SUBJECT 9 — "both were under...", "each of the four..." —
  this/both/each/all as standalone subjects keep their determiner tag.
- QUOTE_INVERSION 6 — "'' , said long jim ." — postposed subject comes out
  as OBJECT of the reporting verb. New rule, bounded trigger.
- HOMOGRAPH_MISTAG 5 — "meek expressed...", "the tie was..." (partly a
  lowercasing artifact — the probe lowercases, destroying the PROPN
  capitalization signal).
- EXISTENTIAL_THERE 3, FRONTED_ADJUNCT 2, fragments 3 (not bugs),
  GERUND_SUBJECT/CONTRACTED_COPULA 1 each.

**The load-bearing secondary finding: the 2-arg extraction ceiling.** Even
on passing sentences the best clause never exceeds SUBJECT + ONE other
role (88.5% cap at exactly 2 args; 11.5% are SUBJECT-only stumps) —
`_fact_clause`/`_primary_discourse` are curriculum-scoped to the first
PREPOSITION edge, so every additional PP/object/modifier on real prose is
silently dropped. The 89% headline measures "got a subject," not "captured
the sentence." The probe tracks this richness histogram so future rounds
see the ceiling move.

Round-5+ ranked list: (1) appositive/interrupter gap-skip rule,
(2) quote-inversion rule, (3) quantifier-subject tagging, (4) existential
there, (5) fronted adjuncts (byproduct of #1's mechanism), (6) homograph
mis-tags (scorer-level, entangled with lowercasing), (7) recursion-limit
engine guard — plus, orthogonal and arguably first: **lift the 2-arg
extraction ceiling**, since it bounds what every downstream consumer can
see regardless of parse quality.

### M49 — wh-questions land: the stress battery saturates at 34/34 (MEASURED)

Sonnet agent. Root cause for both question shapes was the same: with no
NOMINAL before the copula (wh-fronting / subject-aux inversion),
predicate1's generic transitive rule grabbed the postposed subject as an
OBJECT, and no SUBJECT edge ever formed. Fix: a new `position_constraints`
DSL feature (opt-in, zero effect on existing rules — a rule can require a
matched element to sit at a specific sentence position) + one new ruleset
`question1` ahead of the promotion chain: a wh-rule (sentence-initial
RELATIVE specifier + copula + NOUN → SUBJECT, stamps SubType.QUESTION) and
an inversion rule (sentence-initial copula + NOUN → SUBJECT). Both leave
node types unchanged so aux1/verb1/predicate1 proceed normally. A scoring
subtlety was found and fixed: the consumed wh-word must keep its
SPECIFICATION edge or the correct hypothesis under-scores the OBJECT
misreading; with parity restored, the M38/M45 completeness tie-break picks
the SUBJECT reading.

**Battery 34/34 (1.00)** — question 2/2, all other categories unchanged.
Suites: quantum_parser 105/105 (7 new), question+round3 extraction tests
15/15 (3 new), templates 20/20. The wh-cleft template stays dropped,
now with a full diagnosis: beyond the verb1/rel2 collision, rel2 can't
skip the copula between anchor and RELATIVE, and the template needs a
subject/place swap no rule produces — a dedicated pseudo-cleft rule,
ticketed, not forced. SubType.QUESTION is stamped on the parse node but
not yet carried through the adapter (same node-flags gap as PASSIVE —
one adapter change would unblock both).

The hand battery is DONE as a driver (34/34). Round 6+ is owned by the
M48 real-text list: extraction 2-arg ceiling, appositive gap-skip, quote
inversion, quantifier subjects.

### M50 — the 2-arg extraction ceiling falls + adapter flags (MEASURED)

Sonnet agent, consciousness_transformer side. `_extra_args` in clause.py
now walks EVERY argument edge of a predicate (all PPs — including those
quantum_parser's noun3 attaches to the object NP, confirmed empirically —
plus OBJECT/INDIRECT_OBJECT), appended after the primary role so existing
consumers are unaffected; scoped to the clause's own nodes so merged
multi-sentence graphs can't cross-leak. Real-text richness histogram:
was 88.5% capped at exactly 2 args → now 3-7 args on ~66% of clauses,
SUBJECT-only stumps 11.5% → 4.8%. CLAUSE_OK held exactly; battery 34/34;
zero existing-test expectation changes needed.

Adapter node-flags passthrough (one change, three tickets): ParseNode and
HypGraph carry SubType names (backward-compatible defaults). Lands (a) the
M38-ticketed agentive-"by" guard — "the ball was found by mary" now yields
AGENT=mary, while "mary is by the garden" keeps PLACE (both tested);
(b) Clause.is_question from the M49 QUESTION stamp; (c) PASSIVE visible
downstream for whatever needs it next. 18 new adapter tests.

### M51 — real-text grammar trio: coverage 89.0% → 94.5% (MEASURED)

Sonnet agent, quantum_parser side. Mechanism discovered first: punctuation
is tagged NIL, no rule ever consumes NIL, and the matcher's adjacency
search only skips CONSUMED nodes — so a comma is a permanent wall for
every strict-adjacency rule. Fix pattern: punctuation SubTypes (COMMA,
DASH, PAREN_*, QUOTE_*) + rules that explicitly consume interrupter spans.

- APPOSITIVE_INTERRUPT 13→4: appositive1 ruleset (comma/dash/paren ×
  nominal/adverb content). Residuals are gerund-content appositives, an
  interior list, and two HOMOGRAPH_MISTAG artifacts (other bucket).
- QUANTIFIER_SUBJECT 9→0: first attempt (PRON branch in AMBIGUOUS_WORDS)
  worked in isolation but died to order-sensitive max_hypotheses pruning
  on real sentences — measured, reverted. Final: SubType.QUANTIFIER +
  quant1 (DESCRIPTOR/SPECIFIER+QUANTIFIER→NOUN) placed after noun1 (real
  determiner reading keeps first claim; guards tested) and before noun3
  (partitive "each of the four shots" attaches for free).
- QUOTE_INVERSION 6→4 fixed, 2 residual: quote1 emits SUBJECT for the
  postposed speaker, jumping straight to CLAUSE so clause1 can't double-
  fire. The 2 residuals parse CORRECTLY as single units but the M45
  sentence splitter treats mid-quote !/? as sentence boundaries and
  severs the reporting clause — input_encoder follow-up ticketed.
- EXISTENTIAL_THERE mechanism built (exist1) and unit-tested; none of the
  3 corpus instances flip (leading quotes / compound-noun gap / modal
  copula) — honest zero, kept for its regression-free correctness.

Real text: **378/400 = 94.5%** (the 22-sentence delta exactly matches the
class fixes — no other bucket moved). quantum_parser suite 105 → 126.
Battery 34/34, templates 20/20. Residual real-text map: homograph
mis-tags 5, appositive leftovers 4, splitter-severed quotes 2, misc
singletons — plus the compound-noun ("league commissioner") merge gap
that surfaced twice as a blocking prerequisite.

### M52 — the model consumes full argument structure (MEASURED, resolver plan phase 1)

Sonnet agent + main-session gate run. Batch plumbing (`_context_steps` /
`build_clause_batch` only — the ClauseReactor model class is UNCHANGED, per
the design constraint): a transfer clause (give/hand/pass/take) unrolls
into one step per role sharing the transferred OBJECT as the entity;
questions carry their QUERIED ROLE as the relation vector ("who has"→
RECIPIENT, "where"→PLACE default). Old episodes produce byte-identical
batches (exact-equality regression test). New curriculum: TransferCurriculum-
Generator (levels transfer_place / transfer_who), all 4 templates parser-
verified with EVERY role exact. Design finds recorded: (1) the dative
landmine — "gave the ball TO john" mislabels recipient as PLACE
(_PREP_RELATION maps "to" unconditionally), so the curriculum uses the
double-object construction; a dative fix is resolver-era work. (2) "who
has X" excludes TAKE verbs — taker-possession needs AGENT-vs-RECIPIENT
resolution, deferred to M53+ by design.

Gate (probe_m52_transfer.py, 1500 eps, dim 48, 60 epochs, two arms):
control old-only **0.843**; mixed old-subset **0.827** (within noise — no
cost); transfer subset **0.853** (place 0.88, WHO-HAS 0.82 vs chance 0.25
— queried-role routing works). Suites: 51 reactor/curriculum/transfer
tests green; templates 20/20 + transfer 4/4. Phase 1 of
RESOLVER_BUILD_PLAN complete; phase 2 (membrane types + pronoun A/B) next.

### M53a — the membrane exists: candidate sets + anti-recency pronoun data (MEASURED)

Sonnet agent, resolver plan phase 2 first half (no resolver, no model
changes — placeholder gold binding proves the pipeline). New
`src/nsm_ct/membrane.py`: Candidate/CandidateSet (the generic v1 shape
senses and parse hypotheses will reuse), EntityCandidateSet (candidates +
priors + mention feature vector + gold index + provenance),
entity_registry, pronoun_entity_candidate_set. Feature vectors (dim 6):
one real USVS axis (lex:noun.person — carries signal: woman 1.0, mary
0.30, ball 0.02) + 5 hand-specified closed-class dims (PERSON, GENDER_F/M,
NONPERSON, PLURAL) per the design doc's escape hatch — checked first:
attr:gender/sex exist in the 607 but are noise (~1e-5), and pronouns have
no word_coord at all. ClauseBatch carries cand_* tensors (None-default,
byte-identical batches for pronoun-free episodes — regression-tested).

PronounCurriculumGenerator: "mary went to the garden . john went to the
kitchen . she found the ball ." → "where is the ball ?" — the answer
genuinely requires the binding. Anti-recency is a COUNTER, not RNG: ≥50%
of episodes have the correct antecedent NOT most recent. **Nearest-entity
baseline: 0.500 overall, 0.000 on the anti-recency half** — recency is
structurally worthless by design. All templates parser-verified (she/he/
it/they all parse; generator uses she/he, it/they deferred). Smoke with
gold binding: pronoun level 1.000, old 0.842 — pipeline sound.

Landmine found (pre-existing, unfixed, ticketed): "fred" is absent from
the WordNet lexicon and WORD_TAG_DICT, so the -ed suffix heuristic tags it
VERB and its context sentences silently drop (mask=1 not 2) in the
EXISTING curriculum whenever fred is sampled. Excluded from the pronoun
name pool; parser-side fix (names batch in closed-class dict or heuristic
reorder) queued for the next parser round.

75 tests green; templates 20/20 + transfer 4/4 + pronoun 5/5 unchanged.
Next: M53b — Track A coref head vs Track B shared scorer on this data.

### M53 — the first collapse: Track A resolves reference PERFECTLY; Track B binds but interferes (MEASURED)

M53b (Sonnet agent): resolver contract in `src/nsm_ct/resolver.py` — both
tracks take (candidate entities, features, priors, mask, per-candidate
memory readouts, controller state) → [B,C] logits. Track A CorefHead:
specialist, NO controller state (MLP over [cand; mem_read; mention_feat;
prior]). Track B SharedScorer: the design doc's literal
score(candidate, mem_read, state). ClauseReactor gains an OPTIONAL
resolver (default None = byte-identical, proven by independent
re-derivation + all prior tests unmodified); collapse happens BEFORE the
write (soft mixture in training, argmax at eval), margins exposed.
Deliberate architectural difference: any A-vs-B gap is attributable to
the state input specifically.

Gate run (1500 eps, 1/2 old + 1/4 transfer + 1/4 pronoun, dim 48, 60
epochs, three arms; nearest-entity baseline 0.499 overall / 0.000
anti-recency, n=188):

| arm | task | pronoun task | binding | binding anti-recency | params |
|---|---|---|---|---|---|
| gold ceiling | 0.917 | 1.000 | — | — | 0 |
| **Track A** | **0.913** | **1.000** | **1.000** | **1.000** (44/44) | 2,521 |
| Track B | 0.807 | 0.550 | 0.963 | 0.932 | 2,841 |

**Track A matches the gold ceiling with PERFECT binding — including every
anti-recency case, where recency scores 0.000 by construction.** The
membrane design works end to end: reference resolution as memory-keyed
collapse, learned from task reward + aux binding loss, no recency
heuristic anywhere. Track B's failure is the informative kind: binding is
nearly perfect (0.932) yet task accuracy craters (0.807, pronoun-level
0.550) — consulting the controller state entangles resolving with
answering (the state also drives write gates; the smoke-scale B>A signal
inverted at real scale). Round 1: the specialist wins on functionality
AND parameter count. Per the plan, B is not eliminated — the same shared
scorer contests M54 (senses), where its generality argument actually
starts; but "reference resolution needs the running thought-state" is now
measured FALSE at this scale.

### M54 — sense collapse joins the membrane: pipeline works, but the curriculum doesn't FORCE binding (MEASURED, honest partial)

Build (Sonnet agent): SenseCandidateSet through the same membrane (separate
batch fields from entity candidates — entities are memory ADDRESSES, senses
are VALUES; unifying would muddy both). Silent-bug find: M32's ambiguity
episodes NEVER flowed through the reactor before — the batch builder
dropped every one (question shape had no recognizable entity); all prior
chooser results were standalone. Fixed via _ambiguity_steps (rel:SENSE
address + same-clause context token). Track A adds SenseHead (M34 chooser
shape, memory-context, 4,673 params @ d48); Track B is ONE literal shared
instance across pronouns and senses (params counted once, gradients
accumulate — tested). 93 tests; both byte-identity regressions hold.
~22% of ambiguity episodes still drop on the ticketed fred/"hammered"
parser landmines (now a real data cost — fix promoted).

Gate (1500 eps, 40/20/20/20 mix, dim 48, 80 epochs, four arms):

| arm | task | pronoun bind (anti-rec) | sense bind overall / flipped |
|---|---|---|---|
| gold ceiling | 0.917 | — | — |
| **MFS floor** | **0.910** | — | — |
| Track A | 0.860 | **1.000 (1.000)** | 0.370 / 0.407 |
| Track B (shared) | 0.800 | 0.849 (0.840) | 0.304 / 0.222 |

Findings, straight: (1) **The data doesn't make binding matter: MFS-floor
task 0.910 ≈ gold 0.917.** The reactor-form ambiguity episodes leak — the
same-clause context token gets written to memory, so the model answers
from association without needing the sense. Exactly the lesson M53a's
anti-recency design existed to prevent, now measured on the sense side:
capability curricula must make the capability NECESSARY. (2) Sense binding
itself is weak: A's 0.370 overall is BELOW always-pick-MFS (~0.41), though
0.407 on the flipped half clears the by-construction 0.000 floor — with no
task pressure (see 1), the aux loss alone doesn't teach it. (3) Pronoun
resolution is ROBUST: A keeps 1.000/1.000 with the sense head coexisting.
(4) A > B again on both capabilities; B's shared-instance argument remains
unproven, not disproven.

Next: **M54b — a binding-critical ambiguity curriculum** (the answer must
depend on the SENSE through memory, not on a co-written context word;
mirror the anti-recency discipline: an association-only baseline must sit
at floor). Then re-run these arms unchanged.

### M54b — memory-context WSD works: the M30 rematch, won honestly (MEASURED)

The binding-critical curriculum (Sonnet agent; agent caught and reverted
its own leak mid-build — an address-swap variant where a near-zero write
gate preserved the answer-revealing cue, measured as zero gold/MFS gap,
rebuilt with a genuine entity-keyed em.query at collapse time). Data
honesty gates: association-only baseline 0.487-0.516 across seeds
INCLUDING a fresh main-session seed (chance by construction — both
senses' cue words present, attached to different entities); 47 eligible
family/sense pairs after an exhaustive leak audit (pool family + 13
anchors excluded, fred/bill names excluded).

Gate run (1500 eps, 50% old + 50% sense-binding, dim 48, 80 epochs):

| arm | task total | sense-kind task | sense binding overall / flipped |
|---|---|---|---|
| gold ceiling | 0.953 | **1.000** | — |
| MFS floor | 0.700 | **0.483** | — |
| **Track A** | 0.863 | **0.823** | **0.680 / 0.629** |
| Track B (shared) | 0.727 | 0.544 | 0.374 / 0.429 |

The exam is now real: gold-vs-MFS gap on the sense kind is 52 points (was
0.7 in M54's leaky curriculum). On it, **Track A closes 66% of the
floor→ceiling gap: the mind disambiguates a homograph by consulting
entity-keyed memory** — "what do I know about mary?" → river → riverbank —
where association is chance and MFS actively misleads. This is the first
positive WSD-inside-the-mind result: M30's bag-context loss is avenged by
memory context, exactly as the membrane design predicted. Not saturated
(pronouns hit 1.000; senses are harder — 0.629 flipped binding leaves
real headroom for curriculum scale/head tuning), but the mechanism is
PROVEN.

A/B verdict after three fair rounds: **A three, B zero.** With genuine
task pressure B manages 0.544 sense-kind task (barely above the 0.483
floor). The specialist-heads architecture is the working configuration;
the universal-executor bet (see roadmap: Advice Test) is not falsified —
B stays in the codebase and contests M55 — but at this scale, with these
capacities, "methods fall out of one mechanism" has not materialized.
Standing result, recorded straight. Full-vocabulary WSD (SemCor-class)
remains open; the proven mechanism is the credible attack on it.

### M54c — the B diagnosis: pronouns were CAPACITY, senses want an INTERACTION PRIOR (MEASURED)

User's challenge to the 3-0 scoreboard: B fought at ~40% of A's parameters
(2,841 vs 7,194 combined) — "B loses" vs "B is starved" were confounded.
Diagnostic arms (Sonnet agent build, 90 tests incl. byte-exact regression
of default SharedScorer; seven solo runs at full scale):

m53 mix (pronoun-critical; reference A = 0.913 task, 1.000/1.000 binding):
| arm | task | binding overall/anti-recency |
|---|---|---|
| B original (2,841) | 0.807 | 0.963 / 0.932 |
| **B-wide (7,248)** | **0.913** | **1.000 / 1.000** |
| B-nostate-wide | 0.910 | 1.000 / 1.000 |
| B-distilled | 0.917 | 1.000 / 1.000 |

**Pronouns: it was model size, full stop.** Every capacity-matched B
variant is IDENTICAL to A — perfect binding, task at ceiling. The M53
"state entanglement" interpretation is retired: with adequate width the
state input costs nothing. The generalist was never worse at reference;
it was half-sized.

m54b mix (sense-critical; reference A = 0.863 task, 0.680/0.629 binding;
MFS floor task 0.700, gold 0.953):
| arm | task | sense binding overall/flipped |
|---|---|---|
| B original | 0.727 | 0.374 / 0.429 |
| B-wide | 0.737 | 0.531 / 0.229 |
| B-nostate | 0.733 | 0.449 / 0.329 |
| B-nostate-wide | 0.727 | 0.517 / 0.186 |
| **B-distilled** | **0.787** | **0.667 / 0.486** |

**Senses: size no, state-removal no, distillation most-of-the-way.**
(Also: the smoke-scale B-nostate-wide star, 0.789 binding, collapsed to
0.517/0.186 at scale — the SECOND smoke-inversion this arc; smoke tables
are for mechanics only, never verdicts.) Distillation nearly reaching A
says B can roughly REPRESENT the sense solution but cannot FIND it —
optimization trouble layered on a genuine architectural gap: A's
SenseHead inherits M34's multiplicative candidate×context interaction
feature; plain-MLP B must invent multiplication from scratch and doesn't.

Synthesis → the Track C mandate: shared mechanisms scale fine where the
task is matching-by-lookup (pronouns), and fail where they lack an
INTERACTION operator the specialist got as a prior. So give the operator
explicitly: Track C (named ops — query, compare/interact, feature-match,
prior-mix — with learned routing) hard-codes the ABILITY, not the cases
(user's framing). C's design bar, set by this data: match A on BOTH
capabilities with legible routing, no per-task heads. A/B final score:
A 3, B 1 (pronoun round overturned on capacity appeal), senses to neither
— to C's motivation.

### M56 — Track C design spike: verdict "more research"; and a hole blown in M53's halo (DESIGN + one measured finding)

`dev/TRACK_C_DESIGN.md` (629 lines, Sonnet research agent, no code
touched). Op algebra formalized (7 ops explain the whole A/B record:
mem_query, feature_match, prior, interact, score, select, emit; register
model; K_max=6 halting; gold programs written out for both solved tasks).
Graveyard survey reframed per the user's correction: the failures were
mostly ENVIRONMENTAL (no traces, learned representations under the ops,
noisy perception, huge search spaces) — all absent here; the mechanical
residue (gradient-through-composition) is small at our chain lengths and
trace-defused. Training design: trace-supervised → trace-weaned →
compositional transfer with M55's parse-hypothesis collapse as the
zero-new-ops transfer target. Go/kill gates with numbers, ~10k param
budget, ~1-milestone build cost. Verdict: MORE RESEARCH — two hours-scale
checks before any prototype.

**The measured finding (PROVISIONAL until the ablation runs): M53's
perfect pronoun binding is likely a closed-set memorization artifact.**
The gold pronoun program needs a per-candidate feature to match against
the mention's feature — and the membrane never built one: only a
mention-level feature exists, broadcast identically to every candidate.
Checked empirically: entity atoms carry ZERO gender geometry
(cos(mary,john)=0.089, cos(mary,sandra)=-0.043 — same-gender names no
closer). So CorefHead's 1.000/1.000 is a 6-name lookup table learned in
RESOLVER WEIGHTS — name→gender knowledge silently migrated into weights,
violating the "knowledge in structures" invariant, and would not transfer
to a 7th name. The membrane rule needs an enforcement clause: feature
knowledge must live in membrane feature vectors, not be recoverable-only
from weight memorization.

Next (the spike's two pre-prototype checks): (1) held-out-name ablation
to confirm the memorization finding, then add the per-candidate feature
register (NAME_GENDER exists in membrane.py — the data was there, the
per-candidate plumbing wasn't) and re-gate CorefHead on held-out names;
(2) write M55's gold program in the notation prospectively to confirm the
transfer target is expressible before Stage-1 code exists.

### M56b — memorization CONFIRMED, membrane fixed, generalization restored (MEASURED)

The held-out-name ablation (600 pronoun episodes, two seed/name-pair
runs):

| head | train names | HELD-OUT names |
|---|---|---|
| old CorefHead (mention feature only) | 1.000 | **0.000** |
| fixed (per-candidate features) | 1.000 | **0.875** |

(Second pair mary/daniel: old 0.410 held-out, fixed 0.870.) M56's
hypothesis is confirmed exactly: M53's celebrated 1.000 was a six-name
lookup table in resolver weights — on unseen names the old head scores
WORSE than chance. With per-candidate feature vectors plumbed through the
membrane (the gender/person data always existed in membrane.NAME_GENDER;
only the per-candidate wiring was missing), binding generalizes to names
never seen in training at 0.87+. The "knowledge in structures" invariant
now has its enforcement precedent: WHEN A CAPABILITY LOOKS PERFECT, TEST
IT ON HELD-OUT ATOMS — closed-set curricula hide weight-memorization.
SharedScorer untouched (proven by test); byte-identity preserved; 186
tests green.

Also: §1.10 appended to TRACK_C_DESIGN — M55's gold program nearly
type-checks; it needs one register-model extension (a per-candidate Addr
slot: WHICH (entity, relation) to query is itself part of what each parse
hypothesis asserts). Third C-prerequisite, prospectively predicted by the
design's own Risk #1 — the formalism is earning its keep before a line of
executor code exists.

Standing corrections: M53's headline number stands only for the closed
set; the honest generalizing number is 0.875 held-out with the fixed
membrane. Real-scale re-gate of the pronoun benchmark with per-candidate
features + held-out names queued as the next solo run.
