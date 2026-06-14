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
