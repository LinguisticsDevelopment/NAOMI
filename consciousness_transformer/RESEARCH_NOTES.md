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
