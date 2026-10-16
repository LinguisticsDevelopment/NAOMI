# Research Notes

Separates the **research** (open, unsolved) from the **engineering** (built,
working) in the NSM Consciousness Transformer, and records the design decisions.

> This is the one canonical ledger (the repo-root `RESEARCH_NOTES.md` is a
> pointer here). Entries before 2026-10-16 use M-numbers; see
> `dev/archive/MILESTONE_MAP.md` for what each M-number served and its
> outcome. New entries are titled by goal (see `GOALS.md`), not M-number.

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

### M28.0 — the WordNet-only baseline audit (Step A of dev/archive/SEMANTIC_MAPPING_PLAN.md, MEASURED)

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

The first integration gate (dev/archive/INTEGRATION_PLAN.md): do USVS sense signatures
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

`dev/design/TRACK_C_DESIGN.md` (629 lines, Sonnet research agent, no code
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

### M56c — the re-gate: generalizing pronoun binding restored to 1.000 (MEASURED, closes the M56 arc)

Full scale (3000 train episodes, dim 48, 80 epochs, held-out names
sandra/bill, n=750 held-out evaluations):

| head | train names | HELD-OUT names |
|---|---|---|
| old (mention-feature only) | 1.000 | 0.353 |
| **fixed (per-candidate features)** | 1.000 | **1.000** |

The M53 capability is not merely repaired — with the knowledge moved to
its rightful place (per-candidate feature vectors at the membrane), the
resolver binds pronouns PERFECTLY on names it never trained on. The
honest headline for reference resolution is now: 1.000, held-out,
feature-driven, with the name→gender knowledge in an inspectable table
and zero of it in weights. The M56 arc (design spike → memorization
exposé → membrane fix → re-gate) is closed; it cost four days-scale
steps and upgraded both a flagship result and the design discipline
(held-out-atom testing now standard).

### M55a — parse-hypothesis membrane + the honest ambiguity census (MEASURED plumbing, one headline)

Sonnet agent. The C-prerequisite per-candidate Addr register is BUILT
(Candidate.query_entity/query_relation, query_candidates_per_addr —
§1.10 satisfied); parser top-K exposure (opt-in, default-off);
HypothesisCandidateSet; third _collapse branch + hyp_resolver slot; all
three candidate kinds coexist in one batch without cross-contamination;
byte-identity twice over; 277 tests green. Meta-key collision
("homograph" reused by two branches, silently mis-routing) found and
fixed during build.

**The headline — the ambiguity census**: of 1,171 sentences across 33
shapes, ONLY the modal/main-verb homograph collision ("{name} can
{homograph} .") yields exact score ties; every other structurally
distinct reading pair has a CONFIDENT margin (0.08-0.37). This grammar's
garden-path capacity is narrow: one collision family, 13 homographs.
Broad garden-pathing awaits richer grammar or real text — recorded
straight, not padded with synthetic ambiguity.

Baselines at floor by construction: parser-top-1 exactly 0.500 (its
tie-break always picks OBJECT; gold alternates), association 0.42-0.51.
Placeholder-bound smoke 1.000.

**Flagged for M55b (the agent's own honest caveat, load-bearing): the
gold reading is currently an internal counter, NOT inferable from the
context facts.** Plumbing-adequate, but a resolver trained on this could
not exceed 0.5 legitimately — M55b's FIRST job is the M54b move: make the
context facts determine the correct reading (binding-critical redesign),
prove it with a gold-vs-forced-wrong gap, THEN train the collapse.

### M55 — v1 membrane gate: pronouns and senses PROVEN at full scale; readings leak AGAIN (MEASURED, honest partial)

Three arms, m55 mix (old/pronoun/sense/garden-path 40/20/20/20), 1500
eps, dim 48, 80 epochs. Per-kind:

| arm | garden_path | old | pronoun | sense |
|---|---|---|---|---|
| gold-binding | 1.000 | 0.895 | 1.000 | 1.000 |
| **wrong-binding** | **1.000** | 0.911 | 1.000 | 1.000 |
| Track A | 0.456 | 0.911 | 1.000 (bind 1.000/1.000) | 0.698 (bind 0.623/0.536) |

**The wrong-binding arm scoring 1.000 on garden paths is the verdict:
forcing the WRONG reading costs nothing at scale — the trait fact that
determines the gold reading is itself model-visible in memory and
bijectively predicts the answer, so the model shortcuts trait→answer and
never needs the binding.** The small-scale gap probe (1.000 vs 0.341) was
the THIRD smoke-scale result to invert at full scale. Track A's weak
reading numbers (0.561 binding, 0.456 task) are therefore moot — there
was no legitimate task pressure to learn from.

What IS proven at full scale, all three heads coexisting: pronoun
binding 1.000/1.000 (again, now alongside two other active heads) and
sense binding at M54b levels. v1 membrane scorecard: 2.5 of 3.

New curriculum-design rule (the third leak teaches it): **the
gold-determinant must not itself be answer-predictive** — it must be
information that selects the READING while the answer still flows only
through the bound choice. M54b satisfied this accidentally (the cue word
WAS the answer, so binding wrong emitted the wrong vector); M55b's trait
violated it. M55c redesign queued with a mandatory full-scale
wrong-binding arm BEFORE any resolver training (smoke gaps no longer
accepted as gates for curriculum validity).

### M-ES1 — Spanish Freeze Test, stage 1: the interlingua claim holds at the stream level (MEASURED)

Perception-side only, zero retraining. OMW-es sense mapping (coverage:
places 60/61, transfer objects 6/6, ambiguity senses 60/62), es_lexicon
via the M41 recipe (--lang spa), spanish.json rebuilt as a full clone of
english.json's rulesets (rules match abstract categories, not words —
tested, not assumed; the pre-existing partial spanish.json was replaced),
SpanishMeaningResolver (OMW translation → unmodified English parent).

**Stream equivalence, 240 parallel EN/ES episode pairs: entities 240/240
IDENTICAL, relations 400/400 IDENTICAL, values cosine median 0.984 (mean
0.842, exact 42.5%).** All 93 value leaks trace to FIVE words whose
OMW-es sense ORDERING differs from English MFS (ball→bootlicking.s.01
first is an OMW data-quality artifact) — a mapping-layer fix (align by
synset, not Spanish-MFS order), not an architecture problem. To the
frozen mind, entities and relations are provably language-invariant
already; values are one resolver fix from it.

Blockers reported straight: clause.py's _PRONOUNS/is_entity don't know
ella/él (pronoun episodes excluded from the gate); Spanish question
copula needs language-gated tag ambiguity; _PREP_RELATION extended at
the encoder seam (disjoint keys, English byte-identical). 22 new tests;
quantum_parser 134/134; English suites green.

Verdict: the deepest architectural claim — one metalanguage, languages
as interchangeable perception front-ends — survives its first measured
contact. Week-5 Aurora multilingual plans are GO-flavored pending the
synset-alignment fix + a true frozen-model comparison once checkpointing
lands.

### M57a — entity-instance subsystem built: registry + attribute facts + provenance (BUILT, Sonnet agent, director-verified)

The v2 addendum's instantiation design, first half, as pure deterministic
membrane-side engineering (no training): `src/nsm_ct/instances.py` + 16
tests. Schema decision on record (CLAUDE.md): knowledge in the tensor,
enumeration in the registry, audit in the log.

- InstanceRegistry mints FRESH seeded-RNG atoms (not codec-hashed from
  the name — that hash IS the defect being fixed): mary#1 ≠ mary#2,
  deterministic per seed, near-orthogonality regression-gated.
- Attribute facts are ordinary entity_memory writes: instance ⊗
  attr:<name> (codec filler, same as rel:) ⊗ value. Roundtrip argmax
  100% with 5 instances × 3 attributes superposed in one dim-48 tensor.
- candidates_for = enumerate-and-score by attribute match — "mary" →
  both Marys (2-candidate set), "the doctor" → kind=doctor instances —
  with a thin to_candidate_set adapter onto the EXISTING membrane
  CandidateSet/resolver contract (membrane.py stays torch-free per its
  own constraint). inverse_query ("who is a doctor?") is the same
  machinery unthresholded, margins clean in tests.
- ProvenanceLog: append-only frozen-dataclass records (source, language,
  timestamp, trust, step) per gated write — invariant #4's auditable
  record lives HERE; the superposed tensor cannot carry it. Timestamps
  caller-supplied, never wall-clock.

Suite: 16/16 new, full 798 passed / 2 env-gated skips (19m20s), director
re-ran new file + resolver/tpr/membrane subset independently. Next:
M57b resolver-driven write-back ("she is tall" writes to MARY's node;
gate = no-gold inference test) — the registry threading note is in the
agent's report: clause_reactor grounds candidate keys via codec fillers
today; instance ids ground via registry.lookup instead.

### M57b — resolver-driven write-back: PROVEN at full scale, after one full-scale leak caught and fixed (MEASURED)

The mechanism: `cand_addr_mask` on ClauseBatch — on flagged pronoun-
subject steps the entity-collapse weights redirect the write ADDRESS
(Σ w·candidate atoms; soft train / hard eval, same _collapse_weights),
value left as stated. None everywhere = byte-identical (regression-
tested). "She is tall" now physically writes tall to MARY's node.

THE LEAK (v1 curriculum, 4th smoke/full-scale inversion on record):
cheat arm (no candidates, no resolver) hit 1.000 at full scale vs 0.25
smoke floor — the question ALWAYS asked about the pronoun's referent, so
the pronoun sentence's attribute was ALWAYS the answer; the GRU carried
it to the question step, no binding needed. Same design-law violation
as M55b in a new costume: the answer-bearing statement was identifiable
by surface form. Also: wrong-binding-via-corrupted-aux-gold is toothless
— task pressure overrides the aux loss; forcing must hit the COLLAPSE.

The fix (curriculum v2 + forcing machinery): every entity gets a named
attribute first, the pronoun statement OVERWRITES the referent's slot
(reactor's own learned overwrite gate — no special-casing), questions
target either entity uniformly, stale attr kept in the options as the
signature wrong answer; `cand_forced_index` teacher-forces the collapse
one-hot (generic: addr and value rows; -1/None = byte-identity).

Full scale (dim 48, 1500 eps, 60 epochs, Track A, aux on true gold),
writeback kind / referent-targeted / other-targeted:
| arm | task | ref-targeted | other-targeted |
|---|---|---|---|
| normal | 1.000 | 1.000 | 1.000 |
| forced-gold | 1.000 | 1.000 | 1.000 |
| forced-wrong | 0.000 | 0.000 | 0.000 |
| cheat | 0.728 | 0.644 | 0.860 |
| no-gold eval | 1.000 | 1.000 | 1.000 |

The 1.000→0.000 forced-wrong crater (with kind-old intact at 0.882) is
the proof the answer flows through where the write lands and nothing
else. Binding accuracy 1.000 incl. anti-recency (n=77). NO-GOLD GATE
PASSED: zero gold anywhere at eval, trained resolver alone. Cheat at
0.644 on binding-critical questions is an honest M54c-style capacity
note: the GRU partially binds in hidden state at single-episode scale —
explicit machinery adds +0.36 where binding matters; necessity at scale
rests on cross-passage memory, where hidden state cannot survive.

Full suite 822 passed / 2 env-gated skips. Logs: runs/m57b_v2_*.log.
Next: M57c — instance atoms (M57a registry) replace var: name atoms in
the candidate sets, two-Marys + definite-description curricula,
provenance wired into reactor writes, morphology signals, capacity curve.

### M57c battery #1 — write-back replicates in the mix; instance episodes fail EVEN UNDER FORCED-GOLD: the read path is the gap (MEASURED, honest partial)

First cloud-routine battery (Anthropic cloud, 4 cores/15GB; logs on
branch m57c-battery-logs). Instance mix: 1500 eps (500 old / 500
writeback / 500 instance, inverse_frac 0.3), dim 48, 60 epochs, Track A.

| kind / arm | normal | forced-gold | forced-wrong | cheat | no-gold |
|---|---|---|---|---|---|
| writeback | 1.000 | 1.000 | 0.000 | 0.652 | 1.000 |
| instance | 0.356 | 0.423 | 0.279 | 0.317 | 0.356 |
| old | 0.865 | 0.865 | 0.865 | 0.875 | 0.865 |
| inverse_query | 0.138 | 0.207 | 0.138 | 0.172 | 0.138 |

(1) M57b's proof REPLICATES inside the mixed batch, cell for cell —
the director's per-(row,step) evidence-relation fallback held.
(2) Instance episodes (chance 0.25) fail in EVERY arm including
forced-gold 0.423 — the decision-table cell that says "mechanism, not
curriculum, not resolver": handing over the correct referent for free
does not solve the task. Cause = the seam the M57c build disclosed:
QUESTION steps that refer by description/pronoun/ambiguous name still
read memory at the PRE-collapse address (mem_read computed before
_collapse; write-side redirect only), so "what is the doctor like?"
never reads the doctor's node and must answer from GRU recall. Resolver
instance-binding at chance (0.438) is downstream: with the read path
broken, correct binding earns no reward. Per-device subsets confirm:
unique-name other-targeted questions do best (ambiguous_name/other
0.722), description/pronoun referent-targeted worst (0.231/0.500 at
n=13/6). (3) inverse_query BELOW chance: no entity-axis read exists —
"who is tall?" needs the tensor unbound along the entity axis
(einsum bijk,bj,bk->bi), then contrastive over instance atoms.

Verdict: M57c NOT proven; mechanism incomplete, not leaky. Fix is
implementation, not redesign: M57c.2 = post-collapse mem_read recompute
on redirected question steps + entity-axis inverse read. Ops: each arm
5-8GB RSS (B×d³ memory tensor with autograd across T) — 4 parallel arms
OOM'd on 15GB; footprint engineering is now priority 1 per the
2026-08-30 reprioritization (longer episodes are impossible otherwise).

### M57c battery #2 — read path PROVEN (forced-gold 0.423→0.856); resolver does not learn instance binding — it lacks an evidence·target interaction (MEASURED, honest partial)

Cloud routine, minibatched (--batch-size 64, 2 arms concurrent, OMP 2):
33 min/arm, 2.5GB peak (was 180 min, 5-8GB). Same mix as battery #1.

| kind / arm | normal | forced-gold | forced-wrong | cheat | no-gold |
|---|---|---|---|---|---|
| instance | 0.567 | 0.856 | 0.442 | 0.577 | 0.567 |
| inverse_query | 0.621 | 0.724 | 0.483 | 0.690 | 0.621 |
| writeback | 1.000 | 1.000 | 0.217 | 1.000 | 1.000 |
| old | 0.913 | 0.913 | 0.913 | 0.894 | 0.913 |
Forced-gold per device (ref-targeted): ambiguous name 1.000, pronoun
1.000, definite description 0.692; inverse 0.724.

(1) M57c.2's post-collapse read is PROVEN: with the correct referent
handed over, the instance world is solved (0.423→0.856; names/pronouns at
ceiling); the entity-axis read lifts inverse queries from 0.138 to 0.72.
(2) The resolver does NOT learn instance binding (binding/instance 0.486,
normal≈cheat): "normal low, forced-gold high, forced-wrong low" = honest
curriculum + working mechanism + resolver not learning. Cause (by
inspection of _collapse): the resolver sees each candidate's attr:kind /
attr:gender READOUT but never the referring expression's TARGET vector
("doctor") to compare against — M54c's lesson repeated: senses needed an
INTERACTION PRIOR; instance binding needs a per-candidate
cos(evidence_readout, target) feature. Pronouns bound in M53 only
because gender rode in the explicit feature register. Next: M57c.3 =
cand_evidence_target + per-candidate interaction feature into the
resolver (M56b register path), aux on gold_index.
(3) Cheat = 1.000 on 2-entity writeback now (was 0.652): better
optimization lets the GRU bind internally in SHORT episodes. Forced-wrong
still craters (0.217) so the explicit path is used when present;
NECESSITY must be shown where state-carry breaks — rich episodes (8
entities, T=50), the next battery's job. Definite descriptions at 0.692
even under forced-gold: the question-side re-read still competes with
stale GRU recall; watch in battery #3.

### M57 battery #3 — the schema PROVEN on rich episodes; explicit binding shown NECESSARY at 8 entities (MEASURED, M57 closes)

Cloud routine; instance mix + rich_frac 0.3 (350 writeback / 350 instance
/ 450 rich, 3-8 entities, 1-4 overwriting referring statements), dim 48,
60 epochs, batch 64, Track A with M57c.3's interaction feature. ~100
min/arm, peak 6.4GB (T≈50). Logs: branch m57b3-battery-logs.

| kind / arm | normal | forced-gold | forced-wrong | cheat | no-gold | evprior |
|---|---|---|---|---|---|---|
| rich | 0.816 | 0.816 | 0.551 | 0.531 | 0.816 | 0.816 |
| rich @8 entities | 0.938 | 0.938 | 0.812 | 0.500 | 0.938 | 0.750 |
| instance | 0.716 | 0.910 | 0.403 | 0.522 | 0.716 | 0.657 |
| writeback | 1.000 | 1.000 | 0.368 | 1.000 | 1.000 | 1.000 |
| rich inverse | 0.677 | 0.645 | 0.548 | 0.581 | 0.677 | 0.710 |
Binding accuracy (normal): rich 0.954 (n=261), instance 0.819 (was 0.486
in battery #2), writeback 1.000.

(1) The resolver LEARNS instance binding with the evidence·target
interaction: rich binding 0.954; normal == forced-gold on rich episodes
(the gold no longer adds anything). M57c.3's diagnosis confirmed at scale.
(2) NECESSITY of explicit binding, finally measured: the GRU state-carry
shortcut that solves 2-entity writeback (cheat 1.000) BREAKS on rich
discourse — cheat 0.531 vs 0.816 overall, 0.500 vs 0.938 at 8 entities.
Longer discourse ⇒ the explicit memory mechanism carries more of the
task. The architecture's central bet is now a measured curve.
(3) No-gold gate passed on the rich world (0.816 == normal).
Gaps recorded: recency-only cases (ambiguous name / same-gender pronoun,
n=4-5) stay low — resolver has no recency feature (add one); definite-
description questions about OVERWRITTEN slots cap ~0.64 in rich even
under forced-gold — a read-side limit (post-collapse re-read competes
with stale recall), not binding; inverse queries mid-60s (route the
entity readout directly to the answer); --evidence-prior dial adds
nothing (stays off); footprint at T≈50 is 6.4GB — not finished.

VERDICT: M57 robust memory schema PROVEN. Instances, attribute facts,
write-back, resolved reads, inverse reads, provenance, morphology,
capacity curve, rich episodes — all in, all measured. Next: the op
inventory doc (explicit ops + learned parameters, per the lead), then
LTM per the locked decisions, then M58 prose.

### M58a — corpus→episode converter + the prose failure taxonomy: real text parses 5.8% (MEASURED; perception is the bottleneck)

src/nsm_ct/corpus.py + scripts/convert_corpus.py: sentence splitting, the
SAME parser path _context_steps uses, a 6-code failure taxonomy, and
self-generated QA by holding out a clause and asking it through the M52
queried-role table (PLACE/RECIPIENT/AGENT). Two corpora through one
pipeline: 30 authored graded-reader paragraphs (synthetic prose) and 22
real paragraphs (Burgess 1920, Gutenberg).

| corpus | sentences | parsed ok | episodes |
|---|---|---|---|
| synthetic | 120 | 55.0% | 29 |
| real | 86 | 5.8% | 4 |

Real-text failures: unknown-word 54.7% (hyphenated compounds, closed-
class gaps e.g. "until"), multiple-parses-unresolved 23.3% (scorer
saturation on long sentences: 4 hypotheses, margin 0.000), no-parse 8.1%
(quotation fragments), pronoun-unresolvable 4.7%, unsupported 3.5%.
Bug surfaced: _PREP_RELATION maps "to"→PLACE unconditionally, so the
prepositional dative ("gave the ball to john") mis-roles the recipient;
curricula only ever used the double-object form.

Verdict: the roadmap's prediction holds — on wild text the gap localizes
to PERCEPTION and is inspectable. A zero-shot comprehension number at
5.8% coverage would be meaningless; dev/PROSE_FAILURE_TAXONOMY.md is
the worklist: (1) lexicon coverage (compounds, closed-class table), (2)
parse-tie breaking (hypothesis membrane + margins), (3) the dative role
map, (4) quotation/fragment handling. Then M58b: zero-shot on a SAVED
checkpoint (next battery adds --save) over converted episodes.

### M58c — parser round #1: real-text parse rate 5.8% → 18.6% strict / 52.3% usable (MEASURED)

Five additive perception-side fixes, all regression-gated (curriculum
parsing proven unchanged; quantum_parser 134/134): hyphen-compound tail
tagging + WordNet de-inflected fallback + NAME/NUM tiers + closed-class
additions (unknown-word on real text 54.7%→2.3%); parse TIES now emitted
as HypothesisCandidateSet with margins under a new "parsed-ambiguous"
code instead of failing (the M55a membrane's first real-text use); the
prepositional-dative fix ("gave the ball to john" → RECIPIENT when the
verb is a transfer verb — three pre-existing tests had enshrined the bug
and were corrected); quotation stripping + fragment-skipped; a passage-
level pronoun registry. A scorer-level tie-break was tried, caught by a
regression test, REVERTED, and re-implemented as corpus-local reranking
— the gate machinery working as designed.

| corpus | ok before | ok after | ok+ambiguous | episodes |
|---|---|---|---|---|
| synthetic | 55.0% | 63.3% | 71.7% | 29→30 |
| real | 5.8% | 18.6% | 52.3% | 4→18 |

Remaining real-text buckets: pronoun-unresolvable 24.4% (needs real
coreference — the mind's own resolver, once prose episodes carry
candidate sets), unsupported-construction 11.6% (attribution-wrapped
narration), no-relation-extracted 5.8% (needs a plain-OBJECT question
template). Next round: route prose pronouns through the membrane
resolver instead of treating them as parser failures — the architecture
says this is the MIND's job, and the machinery exists.

### M59 — episodic LTM PROVEN: consolidation carries facts across passages; identity linking 1.000 through the EXISTING resolver (MEASURED, M59 closes)

Cloud battery #4, document curriculum (2 passages, 600 docs, dim 48, 40
epochs, ~23 min/arm, 1.2GB — the B=1 DocumentRunner is fine at this
scale). Logs: branch m59-battery-logs.

| arm | overall | type i (LTM-only) | type ii (mention's fact) | type iii (original intact) | link |
|---|---|---|---|---|---|
| normal | 0.933 | 0.913 | 0.946 | 0.944 | 1.000 |
| forced-gold | 0.933 | 0.913 | 0.946 | 0.944 | 1.000 |
| forced-wrong | 0.492 | 0.913 | 0.268 | 0.111 | (forced) |
| cheat | 0.517 | 0.391 | 0.732 | 0.167 | n/a |
| no-ltm | 0.633 | 0.326 | 1.000 | 0.278 | 1.000 |
| no-gold | 0.933 | 0.913 | 0.944 | 0.944 | n/a |

(1) LTM contribution isolated: type i 0.913 vs the 0.326 no-ltm floor —
cross-passage facts flow ONLY through consolidation. (2) Identity
linking learns to 1.000 via the existing resolver contract + from_ltm
feature (invariant #5 held; no new head); no-gold gate passed. (3) The
forced-wrong crater is SELECTIVE — the fingerprint of a real mechanism:
untouched facts survive (type i 0.913), identity-dependent answers
collapse (ii 0.268, iii 0.111). Wrong linking corrupts exactly what
wrong linking should. (4) Cheat 0.517: the GRU cannot fake cross-passage
identity — LTM + registry live outside its state; explicit linking is
structurally necessary, no caveat. Footnotes: forced-gold == normal
(resolver already perfect); trust_ltm ran at default — the dial sweep
(consolidation selectivity under trust gating) is built+unit-tested but
NOT yet battery-proven; deferred with the LTM→Truth tier.

The v2 memory organ chain is now: STM (proven) → episodic LTM (proven)
→ truth memory (designed, tier-generic promote tested for genericity).
Next: Track A confirmation battery with recency/cleanup at scale +
--save (the frozen checkpoint M58b's zero-shot needs), executor Phase 2.

### M60 — Track A confirmation: recency + abstain PROVEN at scale; frozen checkpoint shipped; a direct-route anomaly recorded (MEASURED)

Rich mix + --cleanup, 4 arms, ~3h/arm (full-thread rich episodes). Logs +
checkpoint: branch m60-battery-logs (m60_checkpoint.pt, 700KB — the
sub-MB policy invariant holds with margin).

| kind/arm | normal | forced-wrong | cheat | no-gold |
|---|---|---|---|---|
| overall | 0.887 | 0.550 | 0.710 | 0.887 |
| instance | 0.836 | 0.403 | 0.478 | 0.836 |
| rich | 0.837 | 0.561 | 0.541 | 0.837 |
| ambiguous_name ref | 1.000 | 0.500 | 0.000 | 1.000 |

(1) RECENCY closes the M57#3 gap at full scale: ambiguous-name
referent-targeted 0.250 → 1.000; binding rich 1.000, instance 0.914.
(2) ABSTAIN (caution dial) works at scale: 9.3% abstain, 0.926
when-confident vs 0.887 overall — and rises to 17% under forced-wrong,
i.e. the margin knows when the model is being sabotaged.
(3) ANOMALY, recorded not resolved: inverse_direct_acc = 0.000 (0/45,
systematic) in resolver-trained arms vs 0.822 in cheat vs 1.000 at
smoke. The learned head still answers inverse queries (~0.6-0.7); only
the raw entity-axis readout anti-correlates with the correct option
under the full model. The smoke finding did NOT replicate → the inverse
path is NOT switched to the direct route; diagnosis queued (suspects:
the negate/overwrite gates reshaping stored values; write-address soft/
hard train-eval mismatch on inverse-relevant slots).
Next: M58b zero-shot prose on the frozen checkpoint.

### Executor LOFO gate #1 — INCONCLUSIVE AS INSTRUMENTED: two full-gap recoveries, four gap-less cells, one oracle inversion (MEASURED, no GO/KILL declared)

7 cloud runs (500 eps/family, dim 48, 60 epochs, ~90 min each). Logs:
branch executor-lofo-logs. Per held-out family (learned/oracle/floor):
writeback 0.960/0.960/0.460 (100% of a 0.50 gap recovered traceless —
the 12-step program, composed from task loss + straight-through alone);
recall_link 0.440/0.440/0.250 (100% of 0.19); plain_fact 0.270/=/=,
defdesc 0.470/0.470/0.460, inverse 0.510/0.510/0.470 (no gap →
uninformative); pronoun 1.000/0.170/0.960 (ORACLE BELOW FLOOR —
invalid cell, red flag). The GO/KILL criteria are unevaluable with 4/6
cells gap-less; declaring either would be dishonest.

Instrument defects identified: (1) entry-op selector majority-collapses
even WITH traces (baseline defdesc/inverse op_acc 0.000; plain_fact is
87% of steps — trace loss needs family/class balancing); (2) the shared
resolver was built use_cand_feature=False (Phase 2's documented
sidestep), flattening exactly the families whose oracle edge needs the
interaction feature; (3) the pronoun gold program or forced-trace eval
is broken at scale (oracle 0.170 < floor 0.960) — diagnose before
re-running. Clean positives: write_violations 0 across ~40k clauses;
wall-clock ratio ~1.0x; soft-vs-argmax delta 0.000 everywhere (D1
hard-for-keys costs nothing). Next: LOFO instrument repair, then gate
re-run.

### M58b — THE ZERO-SHOT PROSE NUMBER: 0.583 overall vs 0.250 floor; wild text 0.389; the system abstains honestly (MEASURED — the sprint's central measurement exists)

Frozen M60 checkpoint (700KB, zero prose in training, zero gold at eval)
over 48 converter episodes (30 synthetic-prose / 18 real Gutenberg),
current M58c parser, 0 build failures. scripts/eval_prose.py.

overall 0.583 (floor 0.250, majority 0.042); synthetic 0.700, real
0.389; AGENT 0.577 (n=26), PLACE 0.500 (n=16), RECIPIENT 0.833 (n=6);
abstain_rate 0.458 with 0.615-when-confident — on out-of-distribution
prose the caution dial fires 5x more often than on curricula (0.093)
and correctly.

What bounds the number (both precisely diagnosed, both perception-side):
(1) extraction noise survives into episode CONTENT — pronouns/verbs
mis-extracted as held-out entities ("where is the himself ?") make some
episodes test memory-of-nonsense; the converter needs an episode-quality
filter. (2) A real grounding BUG in bare PLACE clauses:
_context_steps writes entity = codec.filler_vec("var:"+subj)
unconditionally while the question reads via _ent_vec (var: only for
the 6 curriculum names) — measured cosine(write,read) = -0.28 for
non-curriculum subjects: those questions are structurally unanswerable.
AGENT/RECIPIENT don't share the bug (both sides use _ent_vec).

Verdict: the controlled-English claim survives first contact with real
text — comprehension transfers zero-shot at 2.3x floor (1.6x on wild
text) THROUGH a perception layer that parses only 18.6% of wild
sentences strictly, with honest abstention on the rest. The number will
move with perception, exactly as the architecture predicts. M58d: fix
the PLACE grounding bug + episode-quality filter, re-measure.

### M58d — grounding fix + quality filter: the eval got MORE honest, the frozen number moved sideways (MEASURED; retraining is the lever)

Fix 1 (one line): bare-PLACE writes now ground the entity via _ent_vec —
the same function the question reads through — cosine(write,read) = 1.0
by construction everywhere (was -0.28 for non-curriculum subjects).
Curriculum batches proven BYTE-IDENTICAL (torch.equal over ~100 episodes
across four generators vs the pre-fix builder). Fix 2: episode-quality
filter (pronoun/reflexive/verb-primary/closed-class held-outs rejected;
27 of 75 candidates) with a new taxonomy code.

Zero-shot re-measure, same frozen checkpoint: overall 0.583→0.558
(n 48→43); synthetic 0.700→0.741; real 0.389→0.250 (n=16 — noise
territory); PLACE 0.500→0.615; acc-when-confident 0.615→0.731.
Reading: the filter removed junk the model was rightly abstaining on
(abstain 0.458→0.395, confident-accuracy up), and the corrected PLACE
grounding is now structurally right but OUT-OF-DISTRIBUTION for a
checkpoint whose PLACE circuit only ever saw var: atoms in training.
The fix pays at the NEXT training run, not on a frozen model — recorded
as the honest residual, not a regression. Residual noted: the distractor
pool is unfiltered (noise words as wrong options) — future item.
Next: corpus scale-up (statistical power) → retrain on corrected
grounding → re-measure zero-shot on held-out documents.

### LOFO instrument repair — balance works, pronoun inversion explained, and a control-signal information bottleneck found (BUILT + diagnosed)

Repairs: (1) inverse-frequency family weights in the trace loss —
pronoun/writeback/recall op-selection 0.000-0.440 → ~1.000 at smoke;
(2) use_cand_feature=True restored (gate #1 ran with the interaction
feature accidentally OFF), which also exposed and fixed a padding crash
in Executor.run(); (3) the pronoun "oracle < floor" inversion was NOT a
gold-program bug: M53a's placeholder value IS the gold answer, so the
floor scores high for free, and gate #1's crippled resolver sank the
oracle — with the feature restored, oracle ≥ floor. programs.py
untouched. 28 tests green.

FINDING (needed a director ruling): definite_desc_read and inverse_query
share byte-identical control-signal traces until the final step — the
D2 Harvard split (control-only selector inputs) cannot discriminate them
(op_acc pinned at 0.000 regardless of balance/scale; an information
bottleneck, not a training defect). RULING (director, D2-consistent):
the control signal may carry STRUCTURAL batch flags (is_q, inverse_mask,
cand_addr_mask, evidence-target/from_ltm presence) — perception-side
facts about clause SHAPE, the same facts the deterministic dispatcher
reads; the split forbids register CONTENTS, not structure. Extension
queued; gate #2 fires after it.

### Executor LOFO gate #2 — GO: 4/4 informative cells capture 100% of the oracle-floor gap, traceless (MEASURED)

Repaired instrument (balanced trace loss, interaction feature, structural
flags), 7 cloud runs, ~100 min each, EXIT:0 all. Logs: executor-lofo2-logs.
Per held-out family (learned/oracle/floor): writeback 1.000/1.000/0.510;
defdesc 0.800/0.800/0.490; inverse 0.720/0.720/0.450; recall_link
0.520/0.520/0.280 — the executor with ZERO traces for the held-out
family reaches ORACLE level in every informative cell. plain_fact
self-reports uninformative (old_l1_6 scores ~0.24-0.31 in EVERY arm
including oracle under the shared mixed config — a recorded config gap
to fix, not a composition datum); pronoun remains invalid by curriculum
construction (placeholder=gold ⇒ floor ≈ 1.0; the inversion persists at
scale). Design criterion (≥0.5 of gap on ≥4/6) formally met → GO.
Deeper finding: held-out op_acc = 0.000 while task = oracle — the
selector does NOT copy unseen gold traces; it composes FUNCTIONALLY
EQUIVALENT alternative programs (and the floor arm proves they are not
the plain-fact program). Baseline: all six families' op_acc ≥ 0.935
with traces; soft-vs-argmax delta 0.000 everywhere (D1 hard costs
nothing); kill-criteria ratio ≈ 1.0x. PHASE 3 (variable-length programs,
learned halting) is now a LEAD DECISION — not built pending sign-off.

### M58 — THE BIG-N ZERO-SHOT PROSE NUMBER (MEASURED; the honest floor)

Frozen M60 checkpoint (700KB, trained ONLY on synthetic curricula, zero
prose, git a976936), evaluated on real-prose episodes from an 8-file
corpus (5 Gutenberg books: Alice/Burgess×2/Bryant/Edgeworth + 3 synthetic
prose sets), converter+parser at commit b67c8b6. Build: 130/170 episodes
built (40 build-failures = sentences the parser resource-caps dropped
honestly, not crashes). Wilson 95% CIs.

| metric | acc | 95% CI | n=43 prior |
|---|---|---|---|
| overall | 0.523 | 0.438–0.607 | 0.558 |
| synthetic prose | 0.667 | 0.562–0.757 | 0.741 |
| real (wild Gutenberg) | 0.233 | 0.132–0.377 | 0.250 |
| AGENT | 0.475 | 0.353–0.600 | — |
| OBJECT (round-2, new) | 0.697 | 0.527–0.826 | — |
| PLACE | 0.441 | 0.289–0.605 | — |
(random floor 0.25.)

Reading: overall 0.523 is ABOVE chance with real error bars (CI floor
0.438) — the first statistically-backed zero-shot claim. Synthetic prose
(0.667, CI excludes chance) is solid. WILD text (0.233, CI 0.13–0.38) is
NOT distinguishable from chance at this n — honest headline: transfer to
CONTROLLED prose proven, transfer to WILD prose NOT YET demonstrated,
and bounded by perception (40 dropped + capped/degraded parses), not
shown to be comprehension-bound. Conservative floor 3x over: pre-
grounding-fix checkpoint, pre-prose-training, parseable subset only.

Bug found + fixed en route (real, valuable): quantum_parser built the
FULL POS-tag Cartesian product before truncating (56.6M tuples on one
61-word Alice sentence) → OOM in the converter AND a 6-hour hang in the
eval (uncapped _parse at clause_reactor's batch-build path). Fix:
itertools.islice (lazy, byte-identical output) + opt-in
max_ruleset_hypotheses/max_parse_seconds caps via per-call
config_override (default None → every existing caller byte-identical;
curriculum signature gates + quantum_parser 134/134 green). SEAM: same
root bug hit at two call sites → hoist caps into parser defaults later.

NEXT (the point of the floor): prose TRAINING (scripts/train_prose.py),
warm-start from m60, held-out documents, --frozen-eval comparator == this
0.523 baseline. "Does reading real text help it read real text" becomes
a measured delta.

### Prose-training v1 — GATE 1 caught a real hang; training BLOCKED, M58 unaffected (2026-09-02)

Dispatched the first prose-training run with a full-suite sanity gate
FIRST. Gate worked as designed: 1137 tests pass, but
test_corpus.py::test_converted_episodes_build_batch_and_forward HANGS
indefinitely (non-terminating at 7m30s, two independent confirmations) —
so training never ran (correctly). Stack trace:
parse_passage(corpus.py:919) -> _parse_one_sentence(:819) ->
_parse_one_sentence_uncapped(:846) -> _parse_topk_one(input_encoder:375)
-> parse -> _apply_all(quantum_parser:164) -> apply_rule(:272) ->
copy(data_structures:147).

Root cause (the memfix cap has a GAP): (a) the batch-build re-parse path
still routes through an explicitly UNCAPPED variant
(_parse_one_sentence_uncapped) — no config_override with the caps; (b)
even on the capped path the ruleset check fires only AFTER a full
_apply_all returns, so one runaway _apply_all (deep-copying a large
hypothesis inside itertools.product(*ambiguous_groups)) never hits it.
This path is what build_clause_batch's prose steps AND train_prose use,
so prose TRAINING is blocked until capped. The M58 zero-shot number is
UNAFFECTED (eval_prose.py's own build path was capped and completed).

Also found: (1) omw-2.0 missing from the cloud setup recipe (only
wordnet+omw-1.4 downloaded) — 4 spanish_freeze tests fail until
`nltk.download('omw-2.0')`; add it to every cloud dispatch. (2)
test_ground_normalize's discrimination test takes ~419s (uncached
wordnet senses() per word x1200) — slow, not hung; a caching target.

FIX (M58f, dispatched): cap the batch-build re-parse path (route
parse_passage through the capped config_override) AND move the
quantum_parser cap check inside _apply_all so a single runaway call is
bounded; gate = the hanging test terminates with capped drops,
quantum_parser 134/134 + curriculum signature gates green.

### M58f verification — FIX INCOMPLETE: hang persists, cause (a) never addressed (2026-09-02)

Verification-only run: Check 1 (the hanging test alone, 900s, no
contention) did NOT terminate — killed at real 15m0.045s, faulthandler
SIGABRT trace still stuck in parse_passage(corpus.py:919) ->
_parse_one_sentence(corpus.py:829). The M58f branch (f575909) fixed
cause (b) — the quantum_parser per-match deadline inside _apply_all —
and added a train_prose crash-guard, but its diff touched ONLY
quantum_parser + train_prose; corpus.py was never changed. Cause (a) from
the dispatch (the batch-build re-parse path routes through the UNCAPPED
_parse_one_sentence/_parse_one_sentence_uncapped, never passing the cap
config_override) is UNFIXED — which is exactly why test_corpus.py (via
parse_passage, not train_prose) still hangs. NOT merged. Pattern to
watch: each agent fixes one layer and misses another (memfix -> quantum
cap gap; M58f -> quantum fixed but corpus.py caller missed). Pending
Check 3 (frozen-eval): if train_prose's own path completes, the test
hang is isolated cleanup and training can proceed; if it also hangs, the
corpus.py cap is a hard blocker. Decide after Check 3.

Check 3 UPDATE (decisive): frozen-eval Gate-2 COMPLETED cleanly (9m8s),
so PROSE TRAINING IS NOT BLOCKED — the hang is isolated to the ONE test
that calls parse_passage DIRECTLY (test_corpus.py:195) through the
uncapped _parse_one_sentence; production paths (eval_prose/train_prose)
use the per-episode build_one guard that catches ParseResourceExceeded
and drops the sentence (5/34 dropped here, fine). Frozen-eval numbers
(m58f branch, n=29): overall 0.552, synthetic 0.765, real 0.250,
curriculum RETENTION 0.900 (no-forgetting baseline). Confirms M58's
0.523 holds. No third layer appeared — the remaining fix is EXACTLY the
diagnosed cause (a): route corpus.py's _parse_one_sentence batch-build
re-parse through the capped config_override. Durable seam: caps are
scattered across call sites (converter capped, eval guarded, but
parse_passage's direct helper uncapped) -> hoist the cap into the parser
entry so no caller can forget it. Fix M58f2 dispatched; NOT merging the
partial m58f branch until the complete thing verifies (hang test
terminates + quantum 134 + frozen-eval).

### M58f2/f3 — the hang's TRUE root cause: quadratic GC, not parser caps (RESOLVED, merged)

Supersedes the earlier "uncapped parse_passage path" guesses. Found via
py-spy on the live process + instrumented repros: test_corpus hung
because (A) an UNCONDITIONAL gc.collect() ran in a `finally` on EVERY
sentence — the process heap holds the ~1M-object USVS/WordNet tables, so
a full collect costs ~0.3s regardless of garbage, × 1838 sentences ≈
550s of pure waste (measured: gc total dropped from ~550s to ~20s once
gated); (B) the two fallbacks (_quoted/_attribution) each got a fresh
CORPUS_MAX_PARSE_SECONDS budget instead of sharing one deadline. The
per-sentence PARSER caps already worked (sentences parse 2-10s or cap at
10s) — it was never a parser-cap gap. Fix (m58f3, merged): gate the
gc.collect to only the except-ParseResourceExceeded branch + a shared
_remaining() deadline across primary parse and both fallbacks; PLUS made
test_converted_episodes_build_batch_and_forward fast (small fixture; the
full-corpus version kept as @pytest.mark.slow) so it stops taking ~19min
and the suite gate is practical. Carried the m58f quantum_parser
per-match deadline + train_prose per-episode build guard along. Prose
training was never actually blocked (frozen-eval ran clean throughout:
0.552 / retention 0.900). META-lesson for the ledger: this cloud
environment's lost background-notification sleeps cost far more wall-
clock than any compute — long jobs must stay foreground/chunked, and a
sturdier compute path is worth it before the next big campaign.

### Prose training BLOCKED again — 3rd uncapped parse call site → hoist the cap (2026-09-02)

The GC fix (161d236) is confirmed good (test_corpus 11/11 fast, frozen-eval
clean). But the full train_prose run HUNG in batch-build (py-spy: unbounded
deepcopy in quantum_parser apply_rule <- _parse_graph_one). Root cause:
scripts/eval_prose.py's per-episode probe (build_one/partition_buildable,
used by train_prose to build the 128-doc TRAIN split) calls
build_clause_batch WITHOUT the max_hypotheses/max_seconds caps, so it
re-parses a pathological train-set sentence with NO cap. Frozen-eval
didn't hit it (that sentence is in the train split, not the 42-doc eval
split). This is the THIRD distinct uncapped parse call site (converter →
corpus.py; batch-build → clause_reactor; now eval_prose probe). The
PATTERN is the real bug: quantum_parser.parse() enforces no default cap,
so every caller must remember to pass one and keeps forgetting.
DECISION (director): stop patching call sites; HOIST a generous default
wall-clock cap into quantum_parser's ParserConfig (max_parse_seconds
default None -> e.g. 30.0). This bounds EVERY caller at once, cannot bind
on curriculum (ms) or normal prose (2-10s), and turns the hang into a
catchable ParseResourceExceeded that build_one's existing guard drops.
Fix dispatched (M58g); prose training resumes after.

### M58g RESOLVED — default parse cap hoisted; train_prose hang gone (2026-09-02)

Fix landed (8662172): quantum_parser/src/parser/data_structures.py:249
ParserConfig.max_parse_seconds default None -> 30.0. Every parse() call is
now bounded even when a caller forgets an explicit override; override still
available via config_override. eval_prose.py build_one already had a broad
`except Exception` so it drops the now-raised ParseResourceExceeded and
continues — no caller edit needed. VERIFY: tests/ suite 77 passed (no
regression from the cap); quantum_parser 63f/60p/11e are PRE-EXISTING
(identical with fix stashed out, all DSL grammar-load errors, unrelated).
PROOF: train_prose smoke (2 epochs) completed in 4.24 min instead of
hanging — 10s batch cap fired, per-episode probe dropped 5/42 culprits,
training proceeded and saved prose_smoke.pt. This closes the parser-hang
saga (3 uncapped call sites -> one default bound at the parser). Full
20-epoch prose-training run (the M58 payoff) re-fired on top of this.

### M61 — the prose-training payoff: does reading real text help it read real text (2026-09-02)

First full prose-training run on the cap-fixed pipeline (ba743ef).
train_prose.py, 20/20 epochs, held-out-DOCUMENT split (42 docs fully held
out), curriculum-frac 0.5, warm-started from frozen M60. Branch
prose-training-run1 (14cc65e, checkpoint prose_v1.pt). peak_rss 9.1GB;
build dropped 5/42 prose-eval + 22/256 training-mix over-long episodes
(the 30s default cap doing its job; no hang).

BEFORE (frozen M60, held-out docs, n=38):
  overall 0.579 | AGENT 0.500 OBJECT 0.750 PLACE 0.600 |
  synthetic 0.826 (19/23)  real 0.200 (3/15) |
  abstain 0.395  acc_when_confident 0.696 (n=23)
  curriculum retention 0.900 (instance .848 old .853 writeback 1.000)
AFTER (20 epochs, held-out docs, n=37):
  overall 0.622 | AGENT 0.550 OBJECT 0.750 PLACE 0.667 |
  synthetic 0.783 (18/23)  real 0.357 (5/14) |
  abstain 0.351  acc_when_confident 0.792 (n=24)
  curriculum retention 0.885 (instance .788 old .868 writeback 1.000)
DELTA: overall +0.043; real +0.157; synthetic -0.043;
  acc_when_confident +0.096; abstain -0.044; retention -0.015.

HONEST READ: direction is positive on real text (the point) and there is
NO catastrophic forgetting (retention held ~0.89). BUT magnitude is within
noise at this eval size: n~37-38, Wilson 95% on 0.622 is ~+/-0.15, so
+0.043 overall is NOT significant on its own; real +0.157 is just +2/~14
correct. synthetic dipped slightly (expected — capacity reallocated toward
real text). CAVEAT: BEFORE n=38 vs AFTER n=37 (wall-clock cap is
non-deterministic, one extra episode dropped on the AFTER pass) — eval
sets near-identical, not byte-identical. VERDICT: encouraging first signal,
not yet a claim. To confirm needs more held-out documents (bigger corpus)
and/or more epochs; a clean re-run with a fixed pre-parsed episode set
(so BEFORE/AFTER share an identical eval set) would remove the n-mismatch.

### M62/M62b — teacher gold-volume probe: encoder-distill is GO (2026-09-02)

M62 (fetch McGuffey from Gutenberg) BLOCKED: cloud routine egress denies
the open web (only PyPI/npm/GitHub-raw allowlisted). No text fabricated.
For the properly-graded K-12 corpora (McGuffey/RACE/ARC) we will need to
grant egress or MOUNT the corpus (like data/corpus/ already holds cached
Gutenberg). Non-blocking for now.

M62b (in-repo real corpus): measured how much clean GROUNDED-TREE gold the
deterministic parser manufactures from real prose = the teacher-signal
feasibility question for the learned encoder (dev/UNIVERSAL_ENCODER_DESIGN
open-q #2). 500 sentences, 125 per length-bin, from 1475 unique deduped
sentences across 5 real Gutenberg children's-lit files (alice, bryant,
burgess, busterbear, edgeworth). Gold = discourse graph with >=1 clause
carrying a real SUBJECT via extract_discourse, default 30s ParserConfig cap.

  bin A (<=8 tok):  63/125 = 50.4%   cap-hit 0%    med 0.005s p90 0.024s
  bin B (9-15):    113/125 = 90.4%   cap-hit 0%    med 0.078s p90 0.135s
  bin C (16-25):   116/125 = 92.8%   cap-hit 0%    med 0.226s p90 0.554s
  bin D (26+):     114/125 = 91.2%   cap-hit 7.2%  med 1.181s p90 18.29s
  overall:         406/500 = 81.2%   cap-hit 1.8%

Failure modes (94/500): grounding-fail 84 (almost all bin-A bare
interjections/fragments with NO subject to ground -- "oh !", "oh dear !";
a measurement artifact of the gold def, not a parse failure), cap-hit 9
(all bin D, long sentences, ruleset 'predicate1'), too-long 1 (127-tok
> 100-word hard cap).

READ (honest): yield is LOWEST in the SHORTEST bin -- counterintuitive, but
because bin A is dominated by contentless interjections, not because short
sentences are hard. For actual PROPOSITIONAL sentences (B/C/D), yield is a
flat ~90-93% and does NOT degrade with length within the cap. So teacher
gold on real propositional prose is ABUNDANT (~90%) at every length.
VERDICT: parser does NOT need hardening before distillation -- the gold
engine works. M63 (encoder distill, Stage i) can start. Caveat: gold YIELD
RATE is proven high; gold VOLUME (~1300 trees from this in-repo corpus) is
bounded only by source-text ACCESS (egress/mount), not by the parser --
the deterministic parser is an unlimited engine given more source text.
Next real decision belongs to lead: red-pen the encoder design doc + decide
whether to grant egress / mount more public-domain text to scale gold
volume before M63.

### M63.1 / M63.1c — encoder gold landed (teacher bulk + hand-authored hard cases) (2026-09-03/04)

M63.1 (branch encoder-gold-v1, commit e5717f2): deterministic-teacher gold
corpus over all in-repo real text. 1259 gold records / 1475 attempted =
85.4% yield; 4.4MB jsonl. Failures: grounding-fail 180, cap-hit 28,
too-long 6, other 2. Distributions: median 2 clauses/tree, 2 role-slots/
clause, and MEDIAN 5 candidate senses/content token (p90 16) -- confirms the
retrieval-conditioning input is real+bounded. Frozen I/O contract:
dev/ENCODER_IO_CONTRACT.md. This is the bulk distillation data (kept on the
branch, not merged to mainline, like m60-battery-logs).

M63.1c (branch encoder-handgold-v1, commit 5317d13, opus): 33 hand-authored
gold records for the 3 families the teacher structurally can't emit
(imperative synth-subject, interjection appraisal, elision-with-context) +
dev/ENCODER_IO_CONTRACT_ADDENDUM.md (the context_ref construct) +
dev/design/ENCODER_GRAMMAR_FORMAT_PROPOSAL.md (encoder-reference grammar format, 6
rules). Key design result: point-to-context = "select a node from the
retrieved MEMORY-candidate set", exactly symmetric to grounding = "select a
sense from the retrieved SENSE-candidate set" -> ONE retrieval-select action
(EMIT_CONTEXT_REF), on-architecture not a bolt-on.

TWO FLAGS for lead (both real):
1. VALENCE COVERAGE HOLE: "valence from USVS not hard-coded" works only for
   interjections WITH a WordNet/USVS sense (~4/15: shit/wow/damn/hell). Pure
   interjections (alas/ugh/ouch/oh/ah/oops/phew/hurray) have NO synset -> no
   sense to ground to. So the decision presupposes a small USVS interjection
   pseudo-sense table (eval-axis coords) that doesn't exist yet. Agent marked
   these stance_lexicon:"needed" (stores "unresolved", never a hard-coded
   polarity). Director take: bounded task, add interjection senses INTO USVS
   (data, not code -> honors the invariant), do before pure-interjection
   valence is needed.
2. SUPERVISION/ALIGNMENT (the load-bearing one): a gold context_ref is an
   index into serialized prior_context; at run time the antecedent is a
   memory node under a runtime memory_handle. How is the gold pointer a
   supervised loss over a live retrieval candidate set? Director take:
   RESOLVED in principle -- treat serialized prior_context AS the training-
   time memory-candidate set; supervise EMIT_CONTEXT_REF as pointer-selection
   over it, IDENTICAL to sense-selection; make candidate featurization mirror
   runtime retrieval. Tractable; folds into the M63.2 encoder-model spec,
   not a blocker. Awaits lead sign-off.

NEXT: lead red-pens ENCODER_GRAMMAR_FORMAT_PROPOSAL + the two calls above;
then M63.2 (encoder model) -- RESERVED for lead go.

### M63.1c cross-validation — 2nd independent draft SHARPENS flag #1 (2026-09-04)

The local Opus agent (dispatched pre-offline, survived, finished ~10h) wrote
a SECOND independent hand-gold draft (33 records) -> branch encoder-handgold-v2.
Both drafts independently converged: APPRAISAL is the odd family out. The v2
draft sharpens flag #1 into the real architectural point:
- For interjections it's NOT just "pure ones lack a WordNet sense." Even
  interjections WITH a sense fail: "shit" retrieves sense_candidates=
  [shit.n.01 (feces)], but the appraisal target is annoyance.n.01 -- which is
  NOT in the token's candidate set. So the emotion/reaction sense is NEVER
  the surface token's retrieved WordNet candidate, for ANY interjection.
- Consequence: appraisals BREAK the frozen retrieval-conditioning assumption
  ("the correct sense is one of the present token's candidates"). Every other
  family fits the one mechanism (grounding = select a retrieved candidate
  sense of a present token); appraisals do NOT. They require a SEPARATE
  retrieval index: interjection-span -> candidate REACTION senses, which is
  not lemma-reachable and likely needs a small hand-curated reaction
  inventory. (v2 also adds a useful `ref.source:"self"` = point to another
  clause in the same record, beyond context/memory.)
- Also: POS enum has no INTJ; interjections tag as NOUN/ADV, so the appraisal
  signal must ride on a clause_type discriminator, not POS.
REVISED flag #1 for lead: the decision is not "add ~11 interjection senses"
but "appraisals need their own reaction-sense inventory + retrieval index."
This is THE open question for the appraisal family before it can train like
the rest. Imperatives + elision were comfortably human-writable; appraisals
are the snag.

### Appraisal grounding DECISION (lead, 2026-09-04) — derive connotation, no table

Lead REJECTS an interjection->emotion table (off-philosophy). Instead:
- An interjection grounds to its LITERAL USVS sense ("shit!" -> shit.n.01),
  which IS in the token's retrieved sense candidates -> the "appraisals need
  a separate retrieval index" problem DISSOLVES; same grounding mechanism as
  every word. No table, no second index.
- The appraisal is a LEARNED operation on grounded structure: (1)
  connotation-evaluation = read the sense's valence off the GOOD<->BAD eval
  axis already anchored by the NSM GOOD/BAD primes in USVS (valence DERIVED
  from structure, never imported/hard-coded); (2) projection = the
  interjection act projects that valence onto the target (standalone / or
  attached to the reacted-to proposition). Both ops GENERALIZE beyond
  interjections (epithets, sarcasm: "you snake!") -> connotation for the
  whole vocabulary, not 15 words.
- PURE interjections (alas/ugh/oh) have no literal sense: ADD them to USVS
  via GLOSS-grounding (Wiktionary-style gloss through the existing
  ground/usvs.py gloss->prime->coordinate pipeline). Deterministic lexical
  data, not a trained model. Then they ground like any sense.
- NRC-VAD / Warriner: static human-rated tables (not trained models, so they
  pass the no-probabilistic bar) BUT imported valence is less on-philosophy
  than derived; SKIP as primary, keep only as optional validation of derived
  valence. Not expected to be needed.
- HONEST COST: connotation-eval must be LEARNED from USVS structure; works
  where GOOD/BAD anchors give signal, research-risk on subtle/neutral senses;
  testable on held-out words. The right kind of thing to learn, not hand.
CONSEQUENCE: the two hand-draft addenda (encoder-handgold-v1/v2) assumed a
reaction-sense index / emotion inventory -> both need REVISION to
"literal-or-gloss sense + learned connotation-eval + projection". context_ref
(elision) + imperative synth-subject are UNAFFECTED and stand.

### CORE BOUNDARY DECISION (lead, 2026-09-04) — encoder emits candidates, comprehension disambiguates

The encoder does NOT disambiguate. It emits the CANDIDATE LATTICE: all real
sense candidates per node + the parser's top-k POSSIBLE TREES (incl.
structural/attachment ambiguity) + unresolved link slots (pronouns, elided
args). It commits to NOTHING. The COMPREHENSION model (GRU over memory)
resolves ALL of it -- WSD, pronoun/coref linking, elision-fill, attachment --
as ONE primitive: "select the right candidate from memory-conditioned
candidates." Chosen over "encoder does WSD" to avoid PARSER BLEED (encoder
staying a pure transducer) AND because it makes encoder gold deterministic +
label-free.

Consequences:
- MFS/WSD gold problem (#5) DISSOLVED: encoder gold = candidate set (all
  senses via senses_of + parser top-k trees), no correct-sense label needed.
  No sense_chooser wiring for the encoder. WSD is comprehension, trained by
  K-12.
- context_ref is no longer elision-specific: it IS the general
  unresolved-slot->select-from-candidates primitive; pronoun/sense/elision
  are instances.
- The 1259 teacher-gold trees are the WRONG FORM (committed MFS single tree).
  REGENERATE as candidate-lattice gold (top-k structures + per-slot sense
  candidates, no pick). Teacher supports it natively (max_hypotheses=20,
  senses_of returns all). Encoder eval = did it produce the right CANDIDATE
  SET (recall), not the right pick.
- Encoder emits a parse FOREST (top-k) not one resolved tree -- structural
  ambiguity is comprehension's to resolve, same as sense ambiguity.

SEQUENCING (lead): corpus/K-12 test + comprehension training are POST
encoder/decoder build (the "comprehension retrain"). So corpus-sourcing
(egress/mount) and K-12 curriculum DROP OUT of the pre-build checklist.
Revised pre-build (encoder/decoder) checklist: (1) candidate-lattice gold
regenerate; (2) canonical candidates-first schema (reconcile v1/v2, strip
appraisal-node over-build, keep context_ref generalized + synth-subject +
ref.source:self); (3) encoder-reference grammar format finalize; (4) decoder
design; (5) pure-interjection USVS gloss-senses. Build starts only when all
five are ironed out.

### OVERNIGHT encoder/decoder build — running morning report (2026-09-04)

Autonomous pipeline (philosophy locked: encoder = candidate-lattice
transducer, comprehension disambiguates). Status as it progresses:
- STEP 1 DONE + MERGED (2714a2f): canonical candidates-first I/O contract
  dev/design/ENCODER_IO_CONTRACT_V2.md. Verified: lattice replaces committed tree
  (all senses/node + top-k forest + unresolved slots); ONE unified
  `grounding` construct (types sense/reference/elision share candidates+
  retrieval envelope; source lexicon/self/context/memory); imperative synth
  "you" kept; appraisal STRIPPED (interjection -> literal/gloss sense +
  utterance_kind:"interjection"; connotation is comprehension-side); eval =
  candidate-SET recall (sense/structure/slot recall). Design doc §11 added.
- STEP 2 (gold regen -> candidate lattice) + STEP 4 (decoder design): FIRED.
- STEP 4 DONE + MERGED (07b1bbe): dev/archive/DECODER_DESIGN.md. Phase-1 rule-grounded
  short-answer realizer (sense_id->lemma via sense_lemmas, entity book,
  grammar-forward word order); abstention ("I don't know") first-class;
  no-confab ABLATION gate (sever memory->decoder => output collapses to
  abstention, never invents). Reuses mind/membrane.py render path. Phase-2
  learned realizer deferred.
- STEP 2 (gold->lattice) still RUNNING (~30min in). STEP 3 (grammar finalize)
  FIRED.
- STEP 2 DONE (encoder-gold-v2 branch, 5878fea): teacher gold as CANDIDATE
  LATTICE. 985 records (of 1475; no-hypothesis 372, grounding-fail 107,
  cap-hit 11 -- yield lower than v1's 1259 due to stricter lattice validity,
  ACCEPTABLE). REAL FOREST: median 3 trees/sentence (p90 6, max 8), only
  157/985 single-tree. Median 5 sense-candidates/node (p90 18); 4770 pronoun
  reference slots. Shape verified: lattice.trees[] with per-node grounding
  {type:sense, candidates:[...]} (NO committed pick), utterance_kind,
  reference slots. Data kept on branch. (Minor: some corpus header lines leak
  as junk records -- cleanup deferred, non-blocking.)
- STEP 3 DONE + MERGED (a081a80): dev/design/ENCODER_GRAMMAR_FORMAT_PROPOSAL.md
  canonical, candidates-first. Rules LICENSE + emit candidate/unresolved slots,
  never disambiguate; action_map to encoder actions; interjection SIMPLIFIED to
  ground-only (appraisal/FEEL/reaction-index DROPPED); utterance_kind has no
  "appraisal"; PP-attachment emitted as forest branches; multilingual pro-drop
  as one surface-absent+context rule.
- STEP 5 (encoder-model SPEC) + STEP 7 (decoder impl) FIRED.
- STEP 7 DONE + MERGED (8c367f0): src/nsm_ct/decoder.py + tests/test_decoder.py.
  Phase-1 rule-grounded realizer GREEN: 16 tests pass (12.6s). Realizes
  who->"Mary." / where->"The garden." / "Mary is in the garden." / yes-no->
  "Yes." / abstain->"I don't know." NO-CONFAB ABLATION PASSES all 6 cases
  (sever memory->decoder => "I don't know.", zero content-word leak).
  Reuses membrane.py RELATION_TEMPLATES/render_fact; new realize() API +
  sense_id->lemma + entity book + abstention fallback. DECODER READY.
- STEP 5 DONE + MERGED (64488c2): dev/design/ENCODER_MODEL_SPEC.md. Sound:
  never-argmax (candidate-pick not representable), sub-MB controller
  (~0.33M params), CPU-trainable smoke <=10min, loss = candidate-SET
  emission (copy gold set), eval = candidate-set recall, concrete impl
  deliverable list. STEP 6 (encoder impl) FIRED.

### MORNING REPORT (2026-09-04) — encoder + decoder I/O layer BUILT overnight

Autonomous overnight build of the candidates-first I/O layer. Philosophy
(locked): encoder = pure token->candidate-lattice transducer; comprehension
resolves everything (WSD/coref/elision/attachment) downstream via
select-from-memory-candidates.

(a) READY (all merged to mainline):
- ENCODER (src/nsm_ct/encoder_model.py, 199,026 params = ~0.80MB fp32 =>
  SUB-MB confirmed; retrieval-conditioned grammar-constrained transition
  parser). Smoke-trained (150 rec, 2 ep, ~4.6min CPU). Candidate-set recall
  on held-out (n=40) vs random-legal baseline: SENSE 93.8% vs 2.7% (34x);
  SLOT 62.7% vs 0.0%; STRUCTURE 0.0% (both -- see limits). NEVER ARGMAXES by
  construction: no head scores candidate-vs-candidate; sense candidates
  copied verbatim from retrieval (unit-tested + 100% exact-copy spot-check).
  Train loss 61.9->42.5 over 2 ep (learning cleanly). 5 unit tests pass.
- DECODER (src/nsm_ct/decoder.py): phase-1 rule-grounded short-answer
  realizer. 16 tests green. Realizes who/where/attribute/yes-no; ABSTENTION
  ("I don't know.") first-class; NO-CONFAB ABLATION passes (sever
  memory->decoder => abstention, zero content leak).

(b) FILE MAP: src/nsm_ct/encoder_model.py, src/nsm_ct/decoder.py,
scripts/{train_encoder,eval_encoder}.py, tests/{test_encoder_model,
test_decoder}.py, configs/encoder_smoke.yaml; dev/{UNIVERSAL_ENCODER_DESIGN,
ENCODER_IO_CONTRACT_V2, ENCODER_GRAMMAR_FORMAT_PROPOSAL, ENCODER_MODEL_SPEC,
DECODER_DESIGN}.md. Data: candidate-lattice gold (985 rec) on branch
encoder-gold-v2; smoke checkpoint on branch encoder-model.

(c) STUBBED / smoke-only (NOT production):
- Encoder trained on a 150-record SMOKE subset only; full train (all 985 +
  more epochs) is the obvious follow-up. STRUCTURE RECALL 0% is the real
  limitation -- exact top-k-tree match isn't learned at smoke scale (sense +
  slot emission are; structure needs full training and/or a looser match).
- Stage-ii Spanish + Stage-iii code-switch: not started.
- Pure-interjection USVS gloss-senses: deferred (USVS-fingerprint blast
  radius). Teacher-gold correctness: not audited (yield measured, not
  correctness). Hand-gold sense_ids: unverified. Some corpus-header junk
  records leak into the gold. MFS-dissolution decision means these matter
  less for the encoder (it emits candidate sets) but matter for gold hygiene.

(d) NEEDS THE LEAD (next phase): the COMPREHENSION MODEL -- the mind that
consumes the encoder's candidate lattice, RESOLVES it (WSD/coref/elision/
attachment) via select-from-memory-candidates over grounded tensor memory,
reasons, and answers; trained by the K-12 read-then-answer ladder; decoder
realizes its answers. This is where connotation ("how does Bob feel?") and
all comprehension live. Open questions: (1) context_ref supervision =
selection-over-retrieved-candidates (settled in principle); (2) corpus /
egress for real graded K-12 text (cloud routines can't reach the open web --
need egress grant or mounted corpus); (3) full encoder train + structure-
recall fix before relying on the lattice. I/O layer is ready to plug the
comprehension model into.

### Spanish grammar-SWAP-test feasibility: READY (2026-09-04)

Probe (branch spanish-swap-feasibility, dev/archive/SPANISH_SWAP_FEASIBILITY.md).
The swap test (train encoder EN, then feed Spanish grammar+senses+text to the
SAME weights) is a DATA swap, no encoder change -- confirmed feasible.
- CHECK 1 (USVS resolves Spanish): NO out-of-box (English lemmas only), but
  the fix is small+ADDITIVE: union lemmas(lang="spa") from OMW (already in
  the downloaded corpus, mcr/wn-data-spa.tab) into the lemma->sense_id index.
  Verified standalone: 88,942 Spanish lemmas indexed in ~4.3s; 8/9 test words
  resolve to the SAME grounded synsets as English (perro->dog.n.01,
  gato->cat.n.01, nino->child.n.01). NO re-grounding, FINGERPRINT UNCHANGED
  (only the lemma lookup extends, not sense coords). Narrower gap: conjugated
  Spanish verbs (esta/corre/come) miss OMW base lemmas -> predicate
  sense-grounding falls back to entity (contract-valid); lemmatize-before-
  lookup is the optional fix.
- CHECK 2 (quantum_parser tags+parses Spanish): YES, already built on branch
  -- spanish.json grammar, tag_spanish_sentence + SPANISH_WORD_TAG_DICT,
  es_lexicon.json.gz, ParserInputEncoder(lang="es"), meaning_es.py,
  curriculum2 TEMPLATES_ES, probe_spanish_freeze.py. All 3 test sentences
  parse to correct grounded role-labeled trees when correctly accented
  (diacritics matter: "esta"/DET vs "está"/VERB).
- Orthogonal note: the V2 gold BUILDER (build_encoder_gold_v2.py) lives on
  branch encoder-gold-v2, not mainline -- adapt it for lang="es".
PLAN for the swap test (all data/setup, NO encoder redesign): (a) extend USVS
lemma index with Spanish OMW (additive, fingerprint-safe); (b) adapt
build_encoder_gold_v2.py for lang="es" over Spanish sentences -> Spanish
candidate-lattice test set; (c) run encoder_full.pt on it -> candidate-set
recall on never-trained Spanish vs random = THE acceptance test. Honest
caveat: Spanish predicate-verb sense-grounding limited by the base-lemma gap;
noun/entity/structure recall is the clean signal.

### Encoder full EN train + Spanish test-set ready; STRUCTURE gap found (2026-09-04)

FULL ENGLISH TRAIN (branch encoder-train-full, encoder_full.pt): 985 recs,
80/10/10 split, d_model=128, 15 ep, loss 40.5->6.0 monotonic. Params 342,834
(~1.37MB; spec's own "~1.3MB" figure -- "sub-MB" was loose; under the 2MB
ceiling). Held-out TEST candidate-set recall:
  SENSE 0.931 (vs 0.041 random), SLOT 0.978 (vs 0.000 random), consistent
  train/dev/test (no overfit gap) -- grounding + slot emission GENUINELY
  LEARNED. STRUCTURE recall 0.000 on ALL splits incl train.
STRUCTURE DIAGNOSIS (real, not metric artifact): the 7-action transition
system has NO STOP/terminal action -- legal_action_types returns
["OPEN_CLAUSE"] even after the buffer is consumed, so the policy re-opens
clauses forever (6-50 per tree vs gold 1-4); teacher forcing never teaches
"stop" (oracle just ends). Fairer metrics: clause-boundary F1 0.16-0.22,
clause-count match ~0-2%. => ARCHITECTURAL: needs a STOP action added to the
inventory (+ taught). Agent correctly did NOT change architecture. This is
the REMAINING LEVER to finish the encoder's structure -- a targeted bug fix,
not a redesign.
Also flagged: train_encoder.py non-smoke default hash_buckets=32768 blows the
budget (~5MB); overridden to 4096 here -> 1.37MB. Fix the default.

SPANISH TEST SET (branch spanish-gold-v2, runs/spanish_gold_v2.jsonl): 208
records, V2-shape valid. USVS Spanish lemma resolution added ADDITIVELY
(wordnet.py spanish_lemmas + build_usvs union) -- fingerprint PROVABLY
unchanged (e0daef638b640dd5 before==after; fingerprint hashes axes+meta not
lemmas) + English senses_of unchanged; perro->dog.n.01 etc. now resolve in
the production build. Grounding 71% entity / 29% sense; conjugated predicate
verbs 100% entity-fallback (known base-lemma gap). Forest median 1 tree
(templates simpler than EN children's-lit).
NEXT: swap test (EN-trained encoder on Spanish) fired -- sense/slot transfer
is the clean cross-lingual proof; structure 0% in BOTH langs (same STOP bug)
so it doesn't confound. Then the STOP-action fix to finish structure.

### GRAMMAR-SWAP ACCEPTANCE TEST result (2026-09-04) — cross-lingual grounding transfers, on 1 metric

Ran the ENGLISH-only-trained encoder (encoder_full.pt, frozen) on 208
never-seen SPANISH lattice records; Spanish senses+grammar swapped into the
memory input; same USVS (fp e0daef638b640dd5 intact, perro->dog.n.01), same
metric, no retrain. Branch encoder-swap-test (45f88ff), dev/reports/SWAP_TEST_RESULTS.md.

RESULT (Spanish, n=208):
- SENSE recall (candidate-set / grounding-SITE recall): 1.000 (308/308) vs
  random-legal ~0.07-0.10 => ~10-13x. At/above English's 0.931.
- SLOT recall: N/A -- the Spanish templates (curriculum TEMPLATES_ES) contain
  ZERO reference/elision sites, so slot-transfer could NOT be measured.
- STRUCTURE recall: 0.000, identical to English -- the same missing-STOP-action
  bug in BOTH languages (not a cross-lingual failure).

VERDICT (honest): grounding transfers cross-lingually by DATA SWAP ALONE on
the metric this test could measure. Same frozen EN weights + Spanish
grammar/lexicon as memory -> the encoder correctly decides WHERE Spanish
tokens need sense-grounding (the structural grammar->grounding policy), ~12x
random. The encoder is applying a learned apply-grammar-to-ground policy, not
memorizing English. That's real evidence for the universal-encoder thesis.
CAVEATS (do not oversell): (1) ONE metric -- slot transfer untested (no
Spanish slot sites); (2) sense recall = grounding-SITE-type correctness
(which per the candidates-first design IS the encoder's job -- it emits the
candidate set, comprehension selects -- but it is coarser than "picks the
right sense"); (3) Spanish sites skew noun/place (verbs entity-fallback), not
the same distribution as English; (4) structure still 0% both langs.

ENCODER 'FINISHED' STATUS: grounding + slot emission strong (EN 0.93/0.98);
cross-lingual grounding-site transfer shown (1.0 vs 0.08 Spanish). REMAINING
to finish: (a) STOP-action fix for structure (the one real bug, both langs);
(b optional) a Spanish gold set WITH slot/elision sites for a fuller swap
test; (c minor) Spanish verb base-lemma gap. Decoder already done. NOT
starting comprehension model until (a) lands (structure) per lead.

### DECODER PLAN UPDATE (lead, 2026-09-05) — trained decoder, round-trip test

Supersedes "Phase-1 deterministic realizer only". The decoder should be a
TRAINED (learned) realizer, same philosophy as the encoder. Objective AND
acceptance test = ROUND-TRIP RECONSTRUCTION (autoencoder):
  text -> ENCODER -> grounded structure -> DECODER -> text ,  output == input.
- SELF-SUPERVISED: the source sentence IS the target; no separate decoder
  gold needed. Train the decoder to reconstruct the source from its grounded
  structure (use teacher-gold committed structures as the structure->text
  training pairs; the surface sentence is the label).
- TEST: encode a sentence, immediately decode, compare to input -- easy,
  self-checking, tests encoder+decoder TOGETHER as faithful inverses.
  Report reconstruction accuracy (exact-match rate + token-level F1; 100%
  exact not required -- articles/inflection may vary).
- Round-trip needs a committed reading to decode from (the encoder emits a
  lattice); for reconstruction use the top tree + surface tokens/lemmas
  (sense DISAMBIGUATION is not needed to regenerate surface -- that's
  comprehension's job). So the round-trip tests the STRUCTURE+realization
  faithfulness, not sense selection.
- KEEP the no-confabulation ablation (sever structure -> output collapses to
  abstention/empty, never invents) -- reconstruction is the SAFE form of a
  learned decoder (target is the known input, structure is the bottleneck),
  but the ablation stays the guarantee.
- STILL grammar-SWAPPABLE for language: realize via the swapped grammar so
  English-structure -> Spanish grammar -> Spanish out (and Spanish in -> out).
DIRECTIVE to overnight loop: PHASE 3 decoder step = build+TRAIN the learned
reconstruction decoder + round-trip eval (not just make Phase-1 swappable).

### MORNING REPORT (2026-09-05) — encoder STOP fix MERGED; next step is COMPUTE-bound

DONE + MERGED tonight (from branch encoder-stop-fix ebde6c4; code files only,
not its stale RESEARCH_NOTES):
- ENCODER STOP-ACTION FIX (src/nsm_ct/encoder_model.py + tests + train_encoder.py):
  8-action transition system with a terminal STOP; linearize_tree emits STOP;
  legal_action_types(open_clause,i,T,has_clause) -- STOP legal only when i>=T,
  no open clause, AND has_clause (>=1 clause closed) -- the has_clause gate
  fixes a degenerate empty-tree early-exit an undertrained model would take;
  beam_decode terminates on STOP; train_encoder hash_buckets default
  32768->4096 (sub-MB). 8 encoder unit tests green (oracle round-trips w/
  STOP, mask never excludes gold, decode terminates via STOP, sense-copy
  invariant). Mechanically verified: emitted trees now TERMINATE (bounded
  clause counts, not runaway); smoke structure recall NON-ZERO (small,
  undertrained -- per the run's own report).

HONEST LIMITATION -- structure-recall PAYOFF is COMPUTE-BOUND: CPU cannot
converge the model overnight. A 150-rec/1-epoch smoke is ~7min; an
undertrained model rarely emits STOP so beam-decode eval is slow; a real
985-rec multi-epoch train is infeasible on this CPU box. The fix is correct;
demonstrating a strong structure-recall number needs real compute. (The
STOP-fix session burned ~3h grinding CPU smokes chasing this and had to be
wound down.)

ALREADY BANKED (prior runs, on mainline/branches): encoder grounds well
(held-out sense 0.931 / slot 0.978 vs random 0.041/0.000); cross-lingual
grounding transfer PROVEN (EN-trained -> never-seen Spanish sense-site recall
1.000 vs ~0.08 random; fingerprint-safe OMW-spa lemma extension); decoder
Phase-1 rule-grounded realizer done (16 tests, no-confab ablation passes).

NOT DONE (compute-bound and/or gated): full converged encoder retrain +
structure recall >0; the TRAINED reconstruction decoder + round-trip test;
the comprehension-model spec (gated on encoder+decoder actually working);
Spanish-grammar robustify (not run). (Note: a redundant capture routine
encoder-stop-fix2 was also fired mid-wind-down before the original push was
noticed; ignore/dedupe -- the original ebde6c4 is canonical.)

>>> DECISION FOR THE LEAD (hyper-critical -- blocks everything next): provision
GPU / paid compute for real training, OR deliberately shrink to a tiny
curriculum the CPU CAN converge. Everything downstream (structure recall,
trained decoder, comprehension) waits on this. Say which and the next session
proceeds.

### Colab training vehicle prepped (2026-09-05)

Encoder training is CPU-ONLY (encoder_model.py has no device handling -> GPU
would error; confirmed). Colab's value = UNINTERRUPTED long runtime (the
harness interruptions are what killed the overnight CPU runs), not GPU speed.
MERGED to mainline:
- scripts/colab_train_encoder.py + colab/Encoder_Train.ipynb (branch
  colab-encoder-notebook ee68a8f): self-contained, smoke-tested end-to-end;
  trains encoder (default 984 recs x 50 ep, ~50-60min CPU) + English
  candidate-set recall + Spanish swap eval. Smoke even at tiny scale showed
  Spanish sense 0.727 vs 0.036 random (transfer holding).
- TRAINED RECONSTRUCTION DECODER (branch decoder-trained-code 3487c21):
  src/nsm_ct/decoder_trained.py (DecoderTrainedModel, 183,826 params ~0.735MB
  sub-MB; GRU over structure nodes -> surface tokens; copy-from-structure
  no-confab) + scripts/train_decoder.py (self-supervised reconstruction) +
  tests/test_decoder_trained.py (10 green incl no-confab ablation; tiny-train
  sanity exit 0). Trains structure->text; round-trip test = text->encoder->
  decoder->text (exact-match + token-F1).
NEXT: unified colab notebook (encoder + decoder in one run) being built +
smoke-tested; then lead runs it on Colab.

### Unified Colab notebook READY (2026-09-05) — lead runs it

scripts/colab_train_all.py + colab/Train_Encoder_And_Decoder.ipynb (branch
colab-full-notebook 2e6a13e, merged to mainline). One run trains ENCODER
(candidate-lattice, STOP-fixed) + learned RECONSTRUCTION DECODER, then reports:
EN candidate-set recall (sense/slot/STRUCTURE vs random), Spanish grammar-SWAP
recall, decoder reconstruction (exact-match + token-F1), the autoencoder
ROUND-TRIP (text->encoder.beam_decode->top tree->decoder.realize->text), and a
no-confab spot check. Smoke-tested end-to-end on cloud CPU (exit 0, full
RESULTS printed). Reuses all existing train/eval machinery verbatim +
predicted_tree_to_structure bridge. CPU-only (encoder has no GPU path) ->
Colab's value is uninterrupted runtime; ~2h total est. Checkpoints don't
auto-return: lead downloads them or pastes the RESULTS block back to ledger
the real numbers. Notebook URL: colab.research.google.com/github/
LinguisticsDevelopment/NAOMI/blob/colab-full-notebook/consciousness_transformer/
colab/Train_Encoder_And_Decoder.ipynb

### COLAB RUN 1 results + lead ideas (2026-09-05)

First full Colab run (checkpoints on branch colab-trained-ckpts):
- ENCODER (50ep, 788 train): EN test sense 0.928 / slot 0.957 / STRUCTURE
  0.000 (vs random 0.041/0.000/0.000). Spanish grammar-swap sense 1.000 vs
  0.032 (ZERO ES training) -- cross-lingual transfer proven at scale.
- DECODER (80ep): reconstruction from GOLD tree exact 0.000 / token-F1 0.360;
  ROUND-TRIP (enc->dec) exact 0.000 / token-F1 0.323. No-confab 10/10 abstain.
DIAGNOSIS (from numbers): grounding + cross-lingual + no-confab WORK; structure
exact-match 0 and decoder reconstruction weak. Round-trip (0.32) ~= decoder-
from-gold (0.36) => the DECODER is the round-trip bottleneck, not the encoder's
imperfect trees.
LEAD IDEAS (2026-09-05):
1. Decoder: more training + MEMORY-side realization signal -- store each saved
   clause-tree's PARENT RAWTEXT in the memory structures so the realizer learns
   surface/"dialectic" patterns (meaning->how-it-was-said). Retrieval-augmented
   realization; keep it from degenerating into memorizing training sentences.
2. structure recall 0 while sense/slot 0.93/0.96 seems inconsistent -> but it's
   PER-SITE (0.93/0.96) vs WHOLE-TREE-EXACT (0). Lead's key point: there ISN'T
   a single "right" tree -- the encoder emits a candidate FOREST (candidates-
   first); exact-match-to-one-gold-tree is likely the WRONG metric. Right
   metric = gold-tree-in-forest SET recall + soft clause-boundary/attachment F1.
   Don't add a loss that collapses to one tree (breaks candidates-first).
NEXT: diagnosis routine loads the checkpoints -> soft structure metrics + set
recall + example trees + decoder error breakdown, to tell close-vs-broken.

### DIAGNOSIS: encoder structure-0.000 is REAL; 93/96 sense/slot was a MIRAGE (2026-09-06)

Ran scripts/diagnose_colab_ckpts.py on the Colab run-1 checkpoints (98-rec
held-out split reproduced exactly; branch encoder-diag-logs:runs/diag_output.txt).
FINDINGS (decisive):
- [C] edges-off: 98.5% of gold clauses are 4+ edges from the nearest emitted
  clause; 0 exact, 0 one-off. NOT metric strictness -- structure genuinely broken.
- Emitted trees are OVER-ATTACHMENT SOUP: single clause w/ 20-50 role edges,
  same token under SUBJECT+OBJECT+PLACE, predicates = punctuation/<null>/function
  words. [D] predicate token_index agreement = 17%.
- WHY 93/96 looked good: sense/slot are RECALL-ONLY ("token i grounded as sense
  anywhere in forest?"). A dump-everything forest trivially covers every token ->
  ~1.0 recall regardless of correctness. Node-recall WITH relation label = 0.657;
  clause-exact = 0.000. Model beats random (0.04) by DUMPING, not by grounding.
  We skipped CLAUDE.md's cheat-baseline-at-floor law for the encoder.
- ROOT CAUSE (hypothesis): STOP/REDUCE actions never learned to fire (rare in
  oracle -> under-weighted by plain CE) + severe undertraining (50 ep CPU).
- DECODER: reconstruction FROM GOLD tree content=0.386 function=0.404 (both bad,
  equal) + repetition collapse ("let let let", "three three three"). Can't copy
  from a perfect structure yet => memory/retrieval-frame idea PREMATURE.
REVISED PLAN (supersedes "encoder works, move to decoder+memory"):
  1. metric fix: add PRECISION + dump-everything cheat baseline (re-score
     existing ckpt, no retrain) -- non-negotiable gate.
  2. loss/decode fix: make it commit+stop (up-weight terminal/REDUCE / penalize
     over-attach).
  3. THEN long Colab run (fixing 1-2 first; longer training on broken objective
     just overfits dumping).
  4. decoder: more training/capacity vs gold until copy-from-structure works;
     THEN the memory frame.
STATUS: "encoder built+trained (0.93/0.96)" claim RETRACTED -- that was recall
inflated by over-generation. Encoder structure competence is the real blocker.

### ENCODER-FIX-V2 built + merged (2026-09-06, mainline 060550d)

The mirage is now QUANTIFIED (scripts/rescore_encoder.py, runs/rescore_output.txt,
n=98 EN test), model vs baselines:
  MODEL: sense 0.935 slot 0.966 | edge_precision 0.100 edge_recall 0.644 overgen 8.12x
  DUMP : sense 1.000 slot 0.885 | edge_precision 0.016 edge_recall 0.196 overgen 9.77x
  RANDOM: sense 0.031            | edge_precision 0.010 edge_recall 0.010 overgen 0.06x
=> model overgen (8.12) sits right beside the dump cheat baseline (9.77); edge
precision 0.10 (90% of emitted edges spurious). The old 0.93/0.96 was recall
inflated by ~8x over-generation, NOT structure. BUT model edge_recall 0.644 >>
dump 0.196 and random 0.010 => it DID learn which edges are plausible; it just
never learned to stop. Fixable shape.
FIXES SHIPPED (7 commits, 41 tests green, smoke exit 0):
  - edge_precision/edge_recall/overgen_ratio (best-tree EVAL view; forest never
    collapsed -> candidates-first preserved) + policy="dump" cheat baseline.
  - class-weighted CE up-weighting STOP/CLOSE_CLAUSE via --terminal-weight
    (default 4.0; lib default 1.0 = backward-compatible) -- the commit/stop fix.
  - confidence-gated decode --commit-margin (commit top-1 unless 2nd within
    margin) = the lead's "give a set only when unsure".
  - colab_train_all.py + notebook run-2 defaults: --enc-epochs 300, --dec-epochs
    200, --dec-d-model 96, dump baseline in report, git-push cell -> branch
    colab-trained-ckpts-v2. Notebook clones encoder-fix-v2 (kept alive).
GATE for run-2 SUCCESS: model edge_precision must rise WELL above dump's 0.016
and overgen_ratio must fall toward 1.0 (currently 8.12). Recall alone proves
nothing now.
NEXT: lead runs colab/Train_Encoder_And_Decoder.ipynb (run-2). Then evaluate
precision/overgen. If terminal-weight+epochs insufficient, escalate to mask
tightening or GPU-path (encoder is CPU-only; 300 ep may be slow on Colab).

### RUN-2 RESULT + ROOT CAUSE CONFIRMED: it's the MASK, not loss/capacity (2026-09-06)

Colab run-2 (100/100 ep, terminal-weight 4.0, dec d_model 96) FINAL RESULTS:
  encoder model: edge_precision 0.111 (run-1 0.100) overgen 7.50 (run-1 8.12)
                 -> FLAT. terminal-weight loss fix did essentially nothing.
  decoder recon-from-gold token_f1 0.359 (run-1 0.360) -> FLAT despite 2x
                 d_model + more epochs. Decoder ceiling is NOT capacity.
  no-confab 10/10 abstain; Spanish sense 1.000 (transfer holds).
Both "train more/bigger" levers FALSIFIED. ROOT CAUSE found at line level
(src/nsm_ct/encoder_model.py):
  - _apply_action (~L826): GROUND/EMIT_UNRESOLVED_SLOT only advance the buffer
    pointer when state.i < T; at i>=T they produce token_index=None nodes and
    do NOT advance.
  - legal_action_types keeps GROUND legal at i>=T (docstring: for rare
    duplicate-token gold). => once the buffer is consumed, the policy emits
    UNBOUNDED GROUND -> <null>#None phantom nodes (the spam that dominated every
    emitted tree; gold ~6 edges, model best-tree ~45, ~35 are null phantoms).
  - This is why terminal-weight failed: CLOSE can't win when the junk action is
    legal & free every step.
FIX (gold-safe): make GROUND illegal at i>=T. The oracle (linearize_tree) NEVER
GROUNDs content at i>=T (every content node consumes a fresh advancing token),
so this can't mask a gold action -- guarded by the full-corpus oracle-legality
test. Structurally caps tree size ~ token count -> overgen toward 1.0.
IN FLIGHT: no-retrain confirmation routine (trig_018pGHLKSHdcea1eJqH8eJcJ, branch
encoder-mask-fix) re-scores the run-2 ckpt under strict_ground mask + proves
oracle-legality + breaks down null-node spam by gtype (GROUND vs EMIT_SYNTH
prime vs EMIT_UNRESOLVED reference/elision -- the latter two aren't killed by
strict_ground and may need a bound too). If overgen drops toward 1.0, bake the
mask fix in + retrain (training mask must match decode mask).

### CORRECTION: strict_ground fix FAILED + was NOT gold-safe (2026-09-06)

The proposed "GROUND illegal at i>=T" fix (prev entry) is WRONG on both counts
(re-score on run-2 ckpt, branch encoder-mask-fix, runs/mask_fix_output.txt):
  - NO EFFECT: model overgen 8.27 (loose) -> 8.20 (strict), edge_precision
    0.101 -> 0.101. Did nothing.
  - NOT GOLD-SAFE: oracle-legality check FAILED -- 933 gold GROUND actions
    (0.96% of 97530 steps) genuinely land at i>=T via the duplicate-token-index
    collision (e.g. rec6: token_index=59, i=59, T=58). The docstring was right;
    my "oracle never grounds at i>=T" claim was wrong. This fix would mask gold
    actions -> +inf loss. DEAD.
  - Null nodes are only 16.3% of emitted (prime/EMIT_SYNTH_SLOT dominates them
    at 82% of nulls, but nulls overall are minor). Even zero nulls -> overgen
    ~6.9. So the 8x over-generation is REAL-TOKEN over-emission, mechanism NOT
    understood. ~120 emitted nodes/tree vs a handful gold; math doesn't close
    with a monotonic-advance-only GROUND, so the transition dynamics aren't what
    the code reading suggested.
TWO fix hypotheses now falsified (loss-weight; strict_ground). STANCE: stop
theorizing -> AUDIT emitted trees first (trig_01H3Vczy1FvXYj5fAHvH8pHR, branch
encoder-audit-logs): action histogram (does it ever STOP or hit max_clauses=20
cap?), tree shape vs gold, token-index duplication, 8 full example trees. Design
fix #3 from THAT data, not theory.

### ROOT CAUSE FOUND: beam_decode ALIASING BUG (not the model at all) (2026-09-06)

Decode audit (branch encoder-audit-logs:runs/audit_output.txt) revealed the real
mechanism -- a Python object-aliasing bug in beam_decode, NOT model/mask/loss:
  - Winning beams take SHORT, correct action sequences (100% terminate via real
    STOP; e.g. "she answered herself" -> OPEN,EMIT_UNRESOLVED,GROUND,GROUND,
    CLOSE,SHIFT,STOP = a clean parse) BUT emit HUGE trees (that 7-action beam
    produced a 32-node tree). Mathematically impossible without pollution.
  - THE BUG (encoder_model.py ~L739): on beam fork, child BeamState uses
    `cur_clause=b.cur_clause` -- the SAME open-clause dict + roles LIST shared
    across all sibling beams. Every sibling's GROUND/EMIT appends into ONE shared
    roles list. Identity check: sharing factor 2.65, largest shared roles list
    316 nodes. Winning tree = union of many beams' roles.
  - This ALONE produced: overgen ~8x, edge_precision 0.10, structure_recall 0,
    inflated sense recall, token dup up to 16x, 95.6% within-clause collisions.
  - Training is TEACHER-FORCED (single trajectory, no branching) -> NEVER
    affected. That's why the loss curve was healthy and real.
CONSEQUENCE: ALL prior decode metrics are ARTIFACTS. The "encoder over-generates /
is broken" verdict (run-1 AND run-2) is WRONG -- it was beam pollution. The three
"failed fixes" (terminal-weight loss, strict_ground mask, decoder capacity) all
did nothing because none touched the actual bug. The encoder's true quality is
UNMEASURED and may be good.
FIX (decode-only, no retrain): deepcopy cur_clause per fork. Branch
encoder-aliasfix (trig_01CVABFD6npLqmCGb84GnWk4): fix + regression test (tree
node count <= winning-beam emit-action count) + re-score run-2 ckpt + round-trip.
Awaiting: does overgen collapse toward 1.0 and edge_precision/structure jump?
LESSON: validate the eval harness before concluding the model is broken (CLAUDE.md
"perfect-looking results get held-out tests" cuts both ways -- so do terrible ones).

### RESOLVED: aliasing fix -> encoder WORKS. Decoder is the real bottleneck (2026-09-06)

Re-score of the SAME run-2 checkpoint under FIXED beam_decode (deepcopy cur_clause
per fork; mainline 2004059; branch encoder-rescore-logs:runs/rescore_fixed_output.txt):
  ENCODER held-out test (n=98)   CORRUPTED -> FIXED:
    edge_precision   0.101 -> 0.700
    overgen_ratio    8.27  -> 1.082   (near-perfect edge count)
    structure_recall 0.000 -> 0.212   (whole-tree EXACT)
    edge_recall      0.648 -> 0.695
    sense_recall     0.921 -> 0.787   (honest now; 0.92 was over-gen inflation)
    slot_recall      0.963 -> 0.795
  vs DUMP (prec 0.016, overgen 9.77) -> model now crushes the cheat baseline.
  SPANISH grammar-swap (zero ES training, n=208): edge_precision 0.756, overgen
    0.93, structure_recall 0.139, sense 1.000. Cross-lingual grounding transfer
    REAL + strong under clean metric.
VERDICT: the encoder was WORKING the whole time. The "over-generation / structure
broken" verdict (run-1 AND run-2) was 100% the beam_decode aliasing artifact.
Three fix hypotheses (terminal-weight loss, strict_ground mask, decoder capacity)
all no-op'd because none was the bug. The lead's original instinct ("0% can't be
right at 93/96") was correct -- it was a harness bug. ~0.70 precision/recall from
a 343K-param sub-MB parser + zero-shot cross-lingual = the I/O encoder thesis
holds.
REMAINING: the DECODER. Round-trip token_f1 0.35, output garbled even fed the
FIXED (clean) top tree; decoder-from-GOLD-tree 0.36 was never aliasing-affected,
so this limitation is REAL. This is where more decoder training + the lead's
memory-frame idea (store parent rawtext, retrieval-augmented realization with a
copy-from-structure gate) actually apply -- now measurable on clean structure.
NEXT: (1) encoder is essentially done for now; (2) focus shifts to decoder --
retrain/redesign the realizer against clean structure; consider the memory frame.

### TRAINING-PATH AUDITS: encoder CLEAN, decoder has a real bug (2026-09-06)

ENCODER (branch encoder-lossaudit-logs): training/loss path SOUND. Teacher-forced
next-action accuracy 0.888 overall (STOP 1.000, CLOSE_CLAUSE 0.816); every CE
term correct; mask leak-free; metric arithmetic hand-verified (PASS). 0.888 TF
vs 0.70 fixed-decode edge-F1 = normal exposure bias. No loss bug. Encoder DONE.

DECODER (branch decoder-lossaudit-logs): the lead's "logic bug in training"
instinct CONFIRMED, deeper than "too binary":
  - Teacher-forced token acc TRAIN 0.787 vs DEV 0.160 (0.63 gap) = MEMORIZATION.
  - ROOT: build_function_vocab (decoder_trained.py:113) mines EVERY surface token
    not covered by a structure node -> 2047 entries FULL OF CONTENT WORDS
    (bumped, shillings, pope, crown, hungry, inhabited...). The decoder generates
    content from a memorized list instead of copying from structure -> memorizes
    (train) + confabulates (dev: "gingerbread boy"). This is a NO-CONFAB HOLE:
    the function head can emit content absent from the structure.
  - DEEPER CAUSE (the real finding): of 24473 target tokens only 5953 (24%) are
    copyable from structure; 76% are "gaps". The grounded structure is a
    PRED-ARG SKELETON covering ~1/4 of surface words. Exact-surface reconstruction
    by copy is impossible -> function-vocab was papering over it by memorizing.
  - "too binary" exact-position CE is SECONDARY: lemmatized token-F1 (0.352) ~=
    raw (0.345), so output is genuinely wrong content, not synonym/order harshness.
  - Minor: function_vocab built over train+dev combined (vocab leak).
DESIGN FORK (LEAD DECISION, architecture-level): (A) richer structure (ground
more content; more gold) / (B) memory-frame retrieval for the 76% gap (lead's
idea; copy-gated) / (C) honest partial realizer: curate function_vocab to a true
closed class (~100 fn words), decoder realizes only copyable + fn words, stops
confab, measures the HONEST ceiling of what current structure can reconstruct.
RECOMMENDATION: C first (cheap, no new data, fixes confab hole, its ceiling number
decides B vs A). Awaiting lead's pick before building.

### DECODER-V2 STEP 1: the decoder isn't the bottleneck -- the STRUCTURE is (2026-09-06)

Node->span viability report (branch decoder-memory-step1:runs/decoder_memory_step1.txt,
test split n=98, 2400 tokens):
  - Grounded-structure coverage 23.1% (555/2400). Gap 76.9%: 61.4% function,
    38.6% CONTENT -> ~30% of ALL surface tokens are content the structure dropped.
  - Derivation (ii) connectives are CONTENT, near-unique per instance (e.g.
    (OBJECT,SUBJECT) key: 17 occ / 17 distinct strings). NOT a learnable closed
    set. Derivation (i) node-blocks: mean len 4.3, 29% bleed rate.
  => the memory-frame-with-connectives as specced does NOT cleanly work, because
     the "glue" is dropped content, not function words.
  - QUALITY problem too (not just sparsity): e.g. "isabella and caroline ...
    asserted their belief ..." grounds only subordinate-clause fragments
    (integrity/it/was/instance/power) and MISSES the matrix S/V/O. Teacher gold
    grounds the wrong/minor constituents sometimes.
SYNTHESIS: the decoder is NOT the bottleneck. The grounded structure (teacher
gold trees) is a sparse (23%), sometimes-mis-grounded pred-arg skeleton. Both the
round-trip-exact goal AND the memory decoder are capped by structure
coverage+quality. No decoder can realize meaning the structure never captured.
LEAD FORK (architecture): (A) richer/correct structure (more nodes, right
constituents -> better teacher/gold; the expensive lever) vs (reduced-realization)
accept structure as a lossy meaning-core, realize only captured nodes with
function-word glue, DROP round-trip-exact as the goal (cheap, on-thesis, but
capped by quality). Reduced-realization makes connectives clean BY CONSTRUCTION
(drop gap-content, keep gap-function) but does NOT fix the wrong-constituent
quality issue. Either way a GOLD-QUALITY AUDIT is needed -> dispatched.
Connects to dev/archive/ENCODER_DATA_STRATEGY.md: the lever is tree RICHNESS/CORRECTNESS,
not sentence count. HELD for lead decision; encoder-data research done (6564173).

### GOLD-QUALITY AUDIT (director, hand + objective over all 985) (2026-09-06)

Hand-judged 24 rendered gold trees, THEN quantified objectively over all 985 top
trees (correcting an eyeball overstatement -- the objective numbers are milder):
  ROLE nodes (n=4323): 37% function-POS, BUT 33pts are PRONOUNS (legit args);
    only ~4% genuinely-suspect (DET 29, ADP 79, CCONJ 51, AUX 19). Role grounding
    is mostly fine. (NOUN 1733, PRON 1422, VERB 415, ADJ 244, PROPN 233...)
  PREDICATE nodes (n=1931): VERB 63.1% (correct), AUX 29.9% (much = legit copula
    "he IS old"; some = wrong modal/aux e.g. "will look"->PRED:will), CCONJ 4.6%
    (wrong). So predicate quality decent-with-a-real-minority-wrong.
CORRECTION: my 24-example read suggested ~20% "badly mis-grounded"; the objective
numbers say per-node grounding is DECENT (~4% suspect roles, ~5% clearly-wrong
predicates). I overstated. Lesson (again): quantify before concluding.
REFINED CONCLUSION: the bottleneck is SPARSITY, not per-node wrongness. Trees
capture ~23% of tokens (drop modifiers/adjuncts/conjuncts, sometimes a whole
matrix/subordinate clause). What IS grounded is mostly right. Therefore:
  - REDUCED-REALIZATION (decoder realizes the mostly-correct captured core with
    function-word glue) is MORE viable than the step-1 connective analysis implied
    -- that analysis forced connectives to carry dropped content only because it
    targeted FULL-surface reconstruction. Target the captured core instead ->
    connectives shrink to function glue.
  - Encoder "more data" still not the lever: more thin sentences don't fix
    sparsity; RICHER trees (more nodes/sentence) do. = fork A, OR reduced-real.
STATUS: autonomous work assigned by lead COMPLETE (encoder-data strategy +
decoder viability + gold audit). Genuinely blocked on a lead architecture call:
FORK for decoder/structure = (A) richer trees vs (reduced-realization + accept
thin trees). Both downstream steps need this decision. Holding.

### LEAD DECISIONS (2026-09-06)
1. DECODER/structure = RICHER STRUCTURE. Keep verbatim round-trip as the goal;
   fix the root sparsity by making gold trees capture more of the surface
   (modifiers/adjuncts/determiners/conjuncts/subordinate clauses), not just the
   ~23% pred-arg core. Reduced-realization REJECTED.
2. ENCODER = MEASURE SCALING CURVE. Train at n=200/400/788 (fixed epochs),
   report edge-F1 + teacher-forced-acc vs n, to decide if more data helps before
   spending. Needs a Colab run from the lead.
PLAN (richer structure): cost hinges on WHERE content is dropped --
build_encoder_gold_v2.py:build_clause_dict copies cl.predicate + cl.args from
nsm_ct.clause.extract_discourse(graph). If the parser GRAPH still holds the
dropped modifiers/adjuncts (=> richer EXTRACTION, moderate) vs the parser never
emits them (=> parser work, expensive). Investigation dispatched (local agent) to
trace parse->clause->gold and return recoverability + cost per content category.
Then: enrich extraction -> regenerate richer gold -> retrain encoder+decoder.
NEXT: (a) await pipeline-trace verdict -> design richer-extraction; (b) build the
scaling-curve Colab notebook for the lead.

### RICHER-STRUCTURE COST: it's EXTRACTION, not parser work (2026-09-06)

Pipeline trace (local investigation): the quantum_parser emits a FULL typed
dependency graph -- every content word present with a typed edge (adjectives=
DESCRIPTION, adverbs=SPECIFICATION, determiners=DESCRIPTION[DET], adjunct PPs=
MODIFICATION, complements=SUBJECT_COMPLEMENT, subord clauses=SUBORDINATION,
coordination=COORDINATION). The ~77% surface loss is ENTIRELY at extraction:
src/nsm_ct/clause.py extract_discourse whitelists only SUBJECT/OBJECT/
INDIRECT_OBJECT/PREPOSITION/MODIFICATION(PP)/COORDINATION and ignores the rest
(_extra_args L465-494, _relation_for L417-424, _clause_from_node L213-231).
=> NO parser work needed. Richer structure = richer extraction + schema widening.
COST: Phase1 modifiers (adj/adv/complement) CHEAP + additive (new flat role
entries -> new relation labels; encoder action space/oracle/mask UNCHANGED, role
vocab auto-extends, metric _gold_sites is label-agnostic). Phase2 determiners
(cheap, route to structural feature -- no WordNet sense). Phase3 subordinate
clauses (moderate -- recurse extract_discourse into SUBORDINATION subgraph +
discourse link). Schema nesting (modifier->head pointer) deferred; flat roles
first (fine for realization = copy tokens in order).
PLAN: Phase1a = additive DESCRIPTION/SPECIFICATION/SUBJECT_COMPLEMENT extraction
-> regenerate gold SAMPLE -> measure coverage lift (23% -> ?) as a GATE before
retrain. Dispatched. Proceeding autonomously (extraction branch, per lead).

### RICHER-STRUCTURE PHASE 1a: modifiers done, honest partial (2026-09-06)

Branch richer-gold-phase1 (clause.py additive change; runs/richer_phase1_report.txt).
Coverage 22.9% -> 30.1% (+7.2pts, +31% rel). New roles: DESCRIPTION 324,
SPECIFICATION 57, COMPLEMENT 25; newly-covered POS 51% DET / 28% ADJ / 14% ADV.
Extraction verified CORRECT (10/10 examples attach to right clause; DESCRIPTION
child=modifier, SPECIFICATION parent=modifier -- opposite directions, both
handled; SUBJECT_COMPLEMENT dead in grammar -> COMPLEMENT fires instead; both
walked). No parser work.
CAVEATS (honest): (1) +7pts is the SMALL slice -- modifiers were easy; the big
coverage is SUBORDINATE CLAUSES (phase 3) + coordination/multi-verb. 30% not yet
enough for verbatim round-trip. (2) Richer roles RIPPLE: 6 downstream tests fail
(corpus.py _prose_steps role-count heuristics `len(other_roles)==1`, sense-binding
curriculum "one content word" invariant) -- all correctly diagnosed as consumers
assuming the OLD narrow role set, NOT extraction bugs. Must be fixed before merge
(phase 1b). So richer-structure = extraction(cheap) + downstream coupling(moderate)
+ multi-phase, NOT one-shot.
GATE: does adding subordinate clauses reach reconstruction-viable coverage AND
make connectives clean (the step-1 blocker: connectives were content because
content was dropped)? Phase 3 dispatched (subordination extraction + re-measure
coverage + connective function/content %). If it plateaus low or connectives
stay content-laden, verbatim round-trip may be unreachable via extraction ->
escalate to lead. Phase 1 NOT merged (breaks 6 tests); phase 3 builds on branch.

### RICHER-STRUCTURE HITS A WALL: verbatim round-trip not reachable via extraction (2026-09-06)

Phase 3 (subordination) result (branch richer-gold-phase3, runs/richer_phase3_report.txt,
198-sentence sample):
  Coverage: orig 22.7% -> phase1(modifiers) 29.9% -> phase3(+subordination) 30.2%
    (subordination adds only +0.27pp -- SUBORDINATION rare, 17/200 parses, and
     most complement clauses were already caught unlinked by _secondary_fact_clauses).
  CONNECTIVE GATE (the decider): content fraction of the connective gap
    orig 39.4% -> phase1 38.9% -> phase3 38.9% (step-1 baseline 38.6%). UNCHANGED.
    Connectives still near-UNIQUE per (rel,rel) key (novel content per sentence).
  Regression: same 6 downstream failures as phase1, no new (subordination clean).
VERDICT: richer-structure-via-EXTRACTION PLATEAUS at ~30% coverage / ~39%-content
connectives. Verbatim round-trip is NOT reachable this way. Deep reason: for
verbatim reconstruction EVERY content token must be a copyable NODE, but the
connective content is ~39% and NEAR-UNIQUE (novel) -> can't be templated as
connectives, must be nodes -> needs ~full coverage -> extraction can't cheaply
get there (remaining content is a long tail: coordination, serial verbs,
embedded material, parser gaps -- steeply diminishing returns).
KEEP: phase1 modifier extraction (+7pp, richer grounding: adj/adv/complement) is
a GENUINE encoder-gold improvement regardless of the decoder decision -- worth
merging after fixing the 6 downstream role-count consumers.
REOPENED FORK (lead's call, NEW evidence): the lead chose richer-structure +
verbatim round-trip over reduced-realization; that path now shows an evidentiary
wall. Options: (a) REDUCED-REALIZATION (now evidence-favored; phase1 richer trees
make the captured core bigger -> fuller faithful paraphrase; on-thesis = structure
is abstracted meaning); (b) grind full-construction extraction (coordination/
serial-verbs/... diminishing returns, long, parser gaps); (c) reconsider whether
verbatim round-trip is the right GOAL at all (thesis implies lossy abstraction).
RECOMMENDATION: pivot to reduced-realization (or reframe the goal); keep phase1
gold for the encoder. ESCALATING to lead (reverses their decision w/ new data).

### ENCODER PUSH ready (scheduled sampling + scaling curve) (2026-09-06)
Branch encoder-schedsamp: scheduled sampling added to teacher_force_loss
(--sched-samp-max default 0.25, anneal; DEFAULT p_ss=0 = unchanged, 16 tests
green); scripts/scaling_curve.py (train 200/400/788, report edge-F1 + TF-acc);
colab/Encoder_Push.ipynb (Drive-safe overnight, 100 epochs). Smoke exit 0.
Targets the measured 0.70-decode vs 0.888-TF exposure-bias gap (the encoder =
weakest link, per lead) with NO new data. LEAD to run the notebook.
Read: edge-F1 rising w/ n => data-limited; edge-F1 > 0.70 at n=788 => sched-samp
recovered exposure bias.
NEXT (director, autonomous-safe): (a) structure-match decoder evaluator (lead's
cycle-consistency metric); (b) land phase-1 richer-gold modifier win (fix 6
downstream role-count consumers + merge).

### STRUCTURE-MATCH evaluator WORKS (cycle-consistency metric) (2026-09-07)
Branch decoder-structmatch: scripts/eval_structure_match.py. encode->decode->
re-encode->edge-F1(tree_A,tree_B). The re-encode-from-raw-text path works
end-to-end (parse text' -> record -> beam_decode). Baseline (OLD buggy decoder,
19 held-out): per-example edge-F1 mean ~0.28.
KEY: SANITY CONTROL (GOLD text -> re-parse -> beam_decode, decoder BYPASSED)
scores only ~0.5-0.8, NOT 1.0 -> the metric's CEILING is ENCODER-LIMITED
(encoder 0.70 + parser noise make tree_A != tree_B even for perfect input). So
the structure-match TRAINING SIGNAL for the decoder is capped by encoder quality
-> improving the encoder (schedsamp push, running now) raises the decoder's
ceiling too. Confirms "encoder is the weakest link" from another angle.
=> metric is fair (gold-text high, garbage 0) and ready to be the decoder's
training reward once the memory-decoder is rebuilt. Verbatim token-F1 (0.32)
retired in favor of this.

### ENCODER SCALING CURVE result: DATA-LIMITED (2026-09-07)
Branch encoder-schedsamp-results (scaling_log.txt), 100 epochs/size, schedsamp p_ss=0.25:
  n_train  edge_P  edge_R  edge_F1  overgen  tf_acc
    200    0.618   0.520   0.565    0.830    0.814
    400    0.651   0.511   0.573    0.814    0.826
    788    0.713   0.654   0.682    0.952    0.877
VERDICT 1 (DATA-LIMITED = YES): tf_acc rises monotonically 0.814->0.826->0.877
(ceiling still climbing), edge-F1 jumps 0.573->0.682 at 788, no flattening. MORE
DATA is the encoder lever. (Lead's "encoder is weakest link" + "measure scaling
curve" both validated.)
VERDICT 2 (scheduled sampling = ~NEUTRAL, my idea didn't pan out): vs old no-
schedsamp n788 (edge_P 0.70/F1 ~0.70), this run 0.713/0.682 -- flat on F1 (recall
dropped 0.695->0.654), overgen improved 1.08->0.95 (better calibration). The
0.88-tf vs 0.68-F1 gap is NOT recoverable exposure bias -- it's error COMPOUNDING
over the derivation (per-step 0.88 -> sequence ~0.68), and per-step acc is itself
data-limited. Keep schedsamp low/off (mild overgen benefit only).
=> ENCODER FIX = WAY MORE GRAMMATICAL ENGLISH GOLD (lead: "just need way more
examples of working english"). Plan: (1) expand public-domain children's-lit
corpus (fetch_corpus.py --allow-download; Burgess/Potter/Grimm/Andersen/Jacobs/
McGuffey/Aesop) from ~985 -> several thousand sentences; (2) rebuild gold with the
NOW-RICHER extraction (mainline) -> encoder_gold_v3; (3) retrain encoder on it
(bigger n -> higher edge-F1). Corpus-expansion routine dispatched (branch
corpus-expand + colab/Gold_Expand.ipynb). English-only; Spanish stays held-out.

### CORPUS EXPANDED 11x: 1,475 -> 16,411 unique English sentences (2026-09-07)
Branch corpus-expand: fetch_corpus.py extended (nltk gutenberg budgets raised +
verified GITenberg direct-downloads). Sources: Burgess 856, Alice 1220, Bryant
2245, Edgeworth 3515, Grimm 3003, Andersen 2897, Jacobs English Fairy Tales 2817,
McGuffey 410 (+synthetics excluded from CORPUS_GLOB). All public-domain simple
children's-lit English (same genre); English-only (Spanish still held-out).
Smoke gold build (30 new sentences, richer clause.py): 22 records, modifiers
(DESCRIPTION/SPECIFICATION/COMPLEMENT) present. colab/Gold_Expand.ipynb built
(Drive-safe, fsync-per-record, per-sentence cap) -> encoder_gold_v3.jsonl.
build_encoder_gold_v2.py: env-var out override + parse caps + fsync durability.
CAVEAT (key): the CPU-only encoder CANNOT train on all 16K in feasible time
(n=788 x100ep = 3.9h; 16K x100ep ~= 80h). PLAN: build gold (partial ok), TRAIN on
a ~3-5K SUBSET at ~30 epochs (feasible ~4-7h) -> big edge-F1 jump per the
data-limited scaling curve. Exploiting the FULL 16K is the trigger for a
GPU-capable encoder (deferred engineering item, needs lead nod).
NEXT: lead runs Gold_Expand.ipynb -> encoder_gold_v3; then train encoder on a
subset (prep that notebook). corpus-expand NOT yet merged to mainline (hold until
richer gold validated to help).

### GOLD STRATEGY: two-part gold + forest must reflect REAL ambiguity (lead, 2026-09-07)
Lead re-flagged two things about gold composition:
1. TEACHER-BULK vs HAND-AUTHORED (the encoder's whole point): the 16K corpus
   expansion only scales what the DETERMINISTIC teacher can parse (declaratives).
   The structures the LEARNED encoder exists for -- imperatives (synth SUBJECT=you),
   interjections (ground token to literal USVS sense + stance kind; connotation is
   COMPREHENSION-side off GOOD<->BAD, NOT encoder gold), elision/fragments
   (context_ref antecedent-by-content), synthesized args, code-switch -- appear in
   ZERO teacher gold and MUST be hand-fabricated. Encoder trains on teacher-bulk
   UNION hand-authored hard cases (bulk=competence, hand=thesis). Gate: schema must
   be HUMAN-WRITABLE.
2. FOREST WIDTH: gold stores lattice.trees = parser top-k, but MOST sentences have
   ONE valid parse. Storing top-k pollutes gold with INVALID low-confidence parses
   and trains the encoder to over-emit candidate trees (a distinct over-generation
   source from the aliasing bug -- "too big trees for no reason"). FIX: margin-based
   pruning -- keep top-1 unless top-2 within a score margin (genuine ambiguity);
   most sentences -> 1 tree. The candidate lattice should reflect REAL ambiguity,
   not parser hypothesis count.
DISPATCHED (parallel): (A) gold-qa -- per-source parse-quality of new fairy-tale
sources (yield/node-quality vs Burgess/Alice control) + forest-width distribution
+ margin-pruning test (branch gold-qa). (B) hand-gold-draft (OPUS) -- schema-
writability gate + existing-hand-gold check + ~16 draft hard-case candidates for
lead review (branch hand-gold-draft, dev/archive/HAND_GOLD_DRAFT.md). Both analysis/draft
only; nothing merged. Gate the 16K gold build on QA (source-filter + prune) first.

### GOLD QA + HAND-GOLD DRAFT results (2026-09-07)
QA (branch gold-qa, runs/gold_qa_report.txt):
- FOREST TOO WIDE (lead confirmed): current gold mean 3.46 trees/sentence, 61%
  have 3+. Multi-tree variants are SPURIOUS -- trivial modifier-attachment wobble
  (PLACE vs SPECIFICATION vs DESCRIPTION) or dropped clauses, NOT genuine
  ambiguity; 50% of multi-tree are exact score-ties that are still spurious.
  REC: prune gold to TOP-1 (1 tree/sentence) by default; keep token sense-
  candidate SETS (candidates-first preserved where real); genuine structural
  ambiguity = hand-curated, not parser top-k.
- SOURCE QUALITY: new fairy-tale sources noisier -- yield Grimm/Jacobs 50%,
  Andersen/McGuffey 58-62% vs Burgess control 82%; suspect-role 26-28% vs 18%.
  But parse-FAILURES = lost gold not noise; OK trees only modestly noisier; top-1
  pruning cleans further. REC: keep all sources.
HAND-GOLD (branch hand-gold-draft, dev/archive/HAND_GOLD_DRAFT.md, OPUS):
- SCHEMA IS HUMAN-WRITABLE (yes). Human writes ~4-10 lines (surface, clause
  split, utterance_kind, predicate, role labels, filler = W()/PRIME()/CTX()/
  MEM()); scripts/hand_gold.py fills tokens/pos(real tagger)/candidates/indices/
  ctx-handles. Gate validated: 16 drafts pass linearize+legality+round-trip+
  finite train-loss; 5 injected corruptions caught.
- 16 draft hard cases (imperatives/interjections/elision/synth). LEAD DECISIONS
  RAISED: D2 interjection grounding (entity-now vs wait for USVS gloss-senses);
  D3 'me'/'I' (no I prime -> <UNK_PRIME>, add prime or memory-ref); D4 missing
  role labels QUANTITY/ADDITIVE/FOCUS (add to vocab or accept loss); D6 stranded
  aux tense/polarity (elision slot name its carrier?).
AWAITING LEAD: (1) forest rule (rec top-1); (2) source keep-all (rec yes);
(3) review HAND_GOLD_DRAFT.md D2/D3/D4/D6. THEN apply source-filter+prune to
build_encoder_gold_v2 and run the 16K gold build. Nothing merged/trained yet.

### DECISIONS LOCKED + HANDOFF (2026-09-07) -> see dev/CURRENT_STATE.md
Lead decisions this session (full detail in dev/CURRENT_STATE.md "DECISIONS LOCKED"):
  1. Forest -> TOP-1 (spurious multi-trees; keep 2+ only on wide score gap).
  2. Keep all corpus sources (failures = lost gold not noise).
  3. D3: add prime I (me->I); YOU exists.
  4. D4: add role labels QUANTITY/ADDITIVE/FOCUS (structural only, NOT USVS coords).
  5. D2: interjections ground to a REAL sense (content->WordNet sense; pure ugh/
     alas -> gloss-ground into USVS); utterance_kind=interjection; no appraisal node.
  6. D6: elision slots retain surface carrier (keep 'did' token_index for tense/polarity).
Lead flagged we're getting lost -> wrote dev/CURRENT_STATE.md as the single
resume doc (component status + locked decisions + branch map + next-action order).
Context refresh planned; this + RESEARCH_NOTES + memory carry all state.
NEXT: implement decisions 1-6 (one routine) -> lead red-pens HAND_GOLD_DRAFT.md ->
run Gold_Expand.ipynb -> train encoder on ~3-5K subset -> build memory-decoder.

### INDEPENDENT AUDIT (Opus, fresh eyes, 2026-09-08) — CONTINUE WITH CHANGES
Full report: dev/archive/AUDIT_2026-09-08.md. Lead asked "are we pouring energy into a
bad direction?" Verdict: direction defensible, but the plan spends the week on
the LEAST uncertain half and bundles three uncontrolled changes (top-1 prune +
richer extraction + 11x corpus) into one retrain. Load-bearing findings:
1. The encoder's ONLY value over the teacher parser is on sentences the teacher
   CANNOT parse (inference needs no parser: build_features ~L201) -- and that has
   NEVER been measured. Cheap. -> DISPATCHED (trig: encoder-complement-probe).
2. edge_P 0.70 / structure 0.21 are BEST-OF-8-TREES oracle numbers
   (_best_tree_overlap ~L912 picks the max-overlap tree; overgen ignores forest
   width). Rank-1 committed-tree metrics never reported. -> DISPATCHED (same
   routine). Gate: rank-1 edge-F1 >= 0.55.
3. Data-limited verdict rests on ONE point (n=788) at fixed epochs (4x the
   gradient steps of n=200), one seed; 200->400 was flat (+0.008). tf_acc trend
   mildly supports it. -> FOLDED INTO the v3 training plan: arms v2@788 vs
   v3@788 vs v3@3000 at FIXED gradient steps, 2 seeds -- answers "more" and
   "cleaner" separately instead of a seed replication on old gold.
4. Memory-frame decoder: its OWN build-order gate ("coverage near-total or
   revisit") failed twice (23.1% -> 30.2%, connectives 39% content) and the
   design was never revised, yet it is still next-action 5. -> HOLD; respec
   toward reduced-realization (lead had already reopened that fork). Keep
   confab-rate as the hard gate; structure-match is diagnostic until its
   gold-text control clears 0.9.
5. 16 hand-authored records are ~0.2% of a 8-10K bulk; they cannot TRAIN four
   constructions. Use them as a held-out TEST set per family. -> recommend to
   lead (red-pen still needed; they become the exam, not the lesson).
6. "Universal / grammar-as-input" is not implemented: compute_fired_rules is 7
   hard-coded POS heuristics; Spanish swap swapped tagger + test-gold generator,
   not a grammar channel. Spanish gold = 6x6x6 template grid, verbs -> entity;
   "cross-lingual transfer at scale" is over-sold. -> ledger correction; no
   further Spanish claims until a real grammar channel exists.
7. GPU encoder is NOT the bottleneck (per-record training loop, Python-bound,
   343K params); 16K x 30 epochs ~ 24h CPU, arms parallelise. -> DROP the GPU
   engineering item. Aurora clock already dropped 08-30; AURORA_SPRINT.md is
   superseded for sequencing.
8. Biggest untouched risk: no comprehension-model spec; nobody has shown a small
   GRU over tensor memory can RESOLVE a candidate lattice at all. A 2-day toy
   spike over existing gold lattices would say more about the thesis than the
   16K build. -> LEAD DECISION (architecture).
Director actions: D1-D6 routine dispatched anyway (cheap, uncontroversial,
trig_01AAtchs2heh9TSMCfGEYXXv -> branch decisions-d1-d6); measurement routine
dispatched (rank-1 + complement probe -> branch encoder-complement-probe);
Gold_Expand.ipynb stays GATED on those two results.

### D1-D6 IMPLEMENTED + MERGED (Sonnet routine, director-reviewed, 2026-09-08)
Branch decisions-d1-d6 (9 commits, on corpus-expand ∪ hand-gold-draft) merged to
mainline. What changed:
- D1 build_encoder_gold_v2: DEFAULT forest = top-1. `--forest {top1,margin,all}`,
  `--forest-margin` (0.02, the QA report's own number); margin mode keeps a 2nd
  tree only if within margin AND not an exact tie AND structurally distinct on
  the core skeleton (modifier-attachment wobble ignored). Env vars mirror the
  flags so Gold_Expand.ipynb needs no edit. Smoke (50 mixed sentences): top1 ->
  26/26 buildable records have exactly 1 tree, mean nodes/record 32.6 -> 10.0
  (3.2x smaller targets); margin=0.02 -> {1:22, 2:3, 3:1}.
- D2 ground/usvs.py: PURE_INTERJECTION_GLOSSES (20 words) minted as
  `interj.<w>.01` senses via the existing gloss->coordinate pipeline, appended
  AFTER the fingerprint is computed (fingerprint byte-identical e0daef638b640dd5).
  Content interjections already resolve through senses_of; no change needed.
- D3 encoder_model.PRIMES = [YOU, I, <UNK_PRIME>]; clause.FIRST_PERSON_SINGULAR
  {i, me, myself} routed to prime I by both gold builders; linearize_tree's prime
  branch now consumes a real surface token (was hard-coded token_index=None) with
  matching teacher_force_loss/replay buffer fixes. Old checkpoints fail LOUDLY
  (strict load_state_dict size mismatch) -- run-2 ckpt is now incompatible with
  mainline code; the rank-1 re-score routine runs on pre-merge mainline, fine.
- D4 clause.MODIFIER_RELATIONS += QUANTITY/ADDITIVE/FOCUS (structural only; the
  existing consumers clause_reactor/corpus/curriculum2 filter on that set).
- D6 clause-level optional `predicate_token_index` carries the stranded aux
  ("did") for elision/reference predicates; oracle round-trips it. Contract v2
  doc updated.
- tests/test_decisions_d1_d6.py: 21 gates, green. 16 hand-gold drafts rebuilt
  (ugh -> interj.ugh.01; me too -> prime I; the dog did -> carrier @2) and pass
  linearize+legality+round-trip. Full fast suite: 4 failures, ALL pre-existing
  on the merge base with byte-identical assertions (garden_path trait templates,
  morphology byte-identical regression, quantum_adapter two_pps, spanish_freeze)
  -- not introduced here; carried as known-red, to triage separately.
Also lands on mainline by this merge: the 16,411-sentence corpus + Gold_Expand.ipynb
(corpus-expand) and hand_gold.py + HAND_GOLD_DRAFT.md (hand-gold-draft), per the
lead's locked "keep all sources" decision.

### AUDIT GATES MEASURED — BOTH FAIL: rank-1 F1 0.534, complement 18% (2026-09-08)
Branch encoder-complement-probe (merged; runs/rescore_rank1.txt, runs/
complement_summary.txt, runs/complement_probe.txt = all 77 rendered cases for
the lead's hand-judgment). Run-2 checkpoint, n=98 test split, reproduction of
the old best-of-8 numbers exact (0.700 / 1.082 / 0.212).
GATE A (rank-1 committed tree, threshold F1 >= 0.55): FAIL.
  | view              | edge_P | edge_R | edge_F1 | structure_exact |
  | best-of-8 oracle  | 0.700  | 0.695  | 0.698   | 0.212 |
  | rank-1 (= k=1)    | 0.559  | 0.512  | 0.534   | 0.062 |
  Forest width: mean 7.97 of max 8 (min 6) -- the model NEVER stops emitting
  alternatives; forest-pooled overgen 1.157 (mean per-record 3.36). The 0.70
  headline was an oracle over 8 hypotheses; committed, the encoder gets the
  whole tree right 6% of the time. Direct cause is visible: v2 gold averaged
  3.46 trees/sentence, so the encoder was TRAINED to emit forests. D1 (top-1
  gold, merged today) targets exactly this -- untested until a retrain.
GATE B (teacher-complement usable-rate, >= 50% pass / < 25% distillation): FAIL
  at 18.2% (14/77) -- BUT the composition matters:
  - bin A grounding-fail fragments ("oh dear !", "more !"): 3/46 = 7%. J1 needs a
    VERB/AUX predicate, which verbless fragments cannot satisfy, AND no gold of
    any kind contains these constructions. This bucket measures "never trained
    on it", not "cannot generalize".
  - cap-hit LONG sentences (parser exceeded 30 s): 5/9 = 56% usable, median
    3.4 s vs the parser's 30 s cap (8.9x). n=9 -- suggestive, not a claim.
  - grounding-fail bins B/C: 3/12 = 25%, 3/9 = 33%.
  - CONTROL: encoder rank-1 on easy teacher-parseable text 66% usable vs the
    teacher's own trees 74% (judge ceiling). J1 26% / J2 55% / J3 96% on the
    complement (phantoms are NOT the failure; predicate choice is).
HONEST READING: on this checkpoint the learned encoder is parser distillation
with (i) a real but tiny-n edge on sentences the parser times out on and (ii)
zero capability on the fragment/interjection/imperative family, for the
mundane reason that it has never seen one. The hand-gold plan (16 records) is
~0.2% of the bulk and cannot change (ii) (audit finding 2).
DIRECTOR DECISION (within remit, cheap, decisive -- runs before any 16K build):
  Build v3 gold (top-1, richer, D1-D6) for ONLY the original 1,475 sentences on
  the cloud box (~15 min; no Colab needed), then run the fixed-step arms
  v2@788 vs v3@788 (2 seeds, same held-out sentences, ~2 h each). The ONE
  number: does rank-1 edge-F1 clear 0.55 and forest width drop toward 1 when
  the targets are single trees? YES -> encoder salvageable, proceed to v3@3000
  and the 16K build. NO -> the learned encoder is not earning its place; the
  parser becomes the encoder and the week moves to the comprehension spike.
LEAD DECISIONS REQUESTED (architecture): (1) if the retrain fails the gate,
accept the hybrid (deterministic parser as encoder on parseable text, learned
encoder only as the cap-hit fallback) and move effort to comprehension? (2) the
fragment family: fund hand-gold at the scale that could teach it (hundreds per
construction, human time) or drop it from the encoder's remit and let
comprehension handle fragments via candidates? (3) the comprehension spike
(audit finding 8) -- go?
CAVEAT on GATE B (director, after reading runs/complement_probe.txt): the auto-
judge is LENIENT. J1 accepts any VERB/AUX token as predicate ("may" in "he may be
a bully"), J2 any NOUN/PRON subject, and neither checks role sanity ("was" as
PLACE, "'s" as a second SUBJECT) or sense POS (the rendered first candidate for
verbs is often a noun sense: spread.n.01, saturday.n.01 -- candidates-first
display, not a pick, but it shows the candidate ORDER is noun-first). So 18.2%
is an UPPER bound on human-usable output; the 56% cap-hit figure likewise. Lead:
hand-judge the 77 rendered cases before weighting the cap-hit bucket at all.

### TRAINING-ARMS TOOLING MERGED (Sonnet routine, 2026-09-08)
Branch encoder-train-arms: src/nsm_ct/encoder_train_util.py (shared loop with
--max-steps exact optimizer-step budget, seeded stratified subsets, holdout by
sentence text, evaluate_full = best-of-k + rank-1 + forest width), flags on
train_encoder.py and colab_train_encoder.py (--max-steps, --n-train/--subset-seed,
--holdout-file/--eval-gold/--eval-gold-alt, --skip-spanish), scripts/make_holdout.py
(runs/holdout_sentences.txt = the exact seeded n=98 test split of run-2, asserted
by tests/test_encoder_holdout.py), scripts/run_encoder_arms.sh (v2_788/v3_788/
v3_3000 x seeds 0,1; skips v3 arms if gold absent; summary.tsv), colab/
Encoder_Arms.ipynb. 41 tests green; unchanged-path smoke byte-identical.
Smoke throughput 0.505 s/step at smoke config (d_model 64).

### GOLD v3-SMALL BUILT (top-1, D1-D6) on v2's exact 1,475-sentence population (2026-09-08)
Branch encoder-gold-v3-small (jsonl stays there, 8 MB; builder flags + stats +
pre-expand corpus copies merged to mainline). FINDING en route: corpus-expand
MODIFIED the original five corpus files in place, so `data/corpus_v3_small/`
now holds the pre-expand text (rev 9d4aead) -- use those for any v2-comparable
build. Results: 959 records (v2: 985), ALL exactly 1 tree; nodes/record 8.9
(v2 22.3 summed over ~3.5 trees); prime I 161; interj.* 1; QUANTITY/ADDITIVE/
FOCUS 0 (constructions absent from classic prose -- hand-gold territory).
Coverage: 959/985 v2 sentences; the 26 missing are all long sentences that hit
the 30 s parse cap under a contended CPU (v3 cap-hit 37 vs v2 11; the other
outcome buckets are identical). Holdout evaluation against v3 targets may thus
be on <98 sentences; the runner reports the count. Builder gained
`--corpus-files` / `--corpus-glob` / GOLD_CORPUS_GLOB.
ARMS DISPATCHED: v2_788 vs v3_788, seeds 0 and 1, STEPS=8000 (= run-2's ~8.2K
optimizer steps: 788 records x ~3.3 derivations/record / batch 32 x 100 epochs),
holdout = run-2's exact test split, scored on v2 AND v3 targets, rank-1 + forest
width. Two routines (one per seed), branches encoder-arms-seed0/-seed1.

### LEAD DECISIONS (2026-09-08, after the gate results)
1. The LEARNED encoder stays the DEFAULT regardless of the v2/v3 arm outcome
   (robustness; Spanglish/code-switch is only reachable with a learned
   transducer). The hybrid "parser as encoder" option is REJECTED; the arms
   now decide HOW to train it, not WHETHER.
2. Hand-gold is FUNDED as GENERATED data: templates per hard construction
   family with typed slots (VERB/NOUN/ADJ/PROPN/...) filled from USVS/WordNet
   pools, seeded, gated through hand_gold.py's validator. Hundreds per family,
   with filler-held-out AND template-held-out test splits.
3. No red-pen pass on the 16 drafts; training/eval failures are the review.
Director: dispatched the generator routine (branch hard-gold-gen). The
comprehension spike (audit finding 8) still awaiting the lead's answer.

### LEAD DIRECTION (2026-09-08): self-training loop + revisable comprehension
- Once the encoder works, it SELF-TRAINS: its good parses on edge cases become
  gold for the next round (bootstrapping the hard-construction families beyond
  the generated templates).
- Comprehension-side resolving is "sort of done" (M53-M57 membrane); with
  consistent encoder lattices carrying full context it gets easier. Required
  property: the system must be able to detect that it misunderstood a sentence
  and RE-EVALUATE when later context arrives -> the candidate lattice (or its
  provenance) must stay revisable in memory, not be collapsed and discarded.
- The audit's "comprehension spike" is therefore reframed: not a new build, but
  an evaluation of the EXISTING resolver on encoder-emitted lattices once the
  v3 encoder exists, plus the re-evaluation-on-new-context property as a
  measured capability.
Director note on self-training: the filter that admits a self-parse to gold
must not be the model's own confidence (today's gate results show best-of-8 vs
rank-1 confidence is uncalibrated). Candidates already in the repo: structure-
match round-trip (encode->decode->re-encode agreement), the no-phantom gate
(J3, 96% even on failures), teacher agreement where the teacher parses, and
comprehension-side consistency (a parse that contradicts memory is suspect).
DISPATCHED (2026-09-08 ~08:53Z, lead-approved): (a) K-12 corpus SCOUT (branch
k12-scout: open-license graded readers WITH comprehension questions, licenses,
samples, teacher parse yield, fetch script); (b) LLM PARSE JUDGE (branch
llm-parse-judge: tree flattener + Claude Haiku 4.5 good/bad verdict via the
Anthropic SDK with Batches mode, mock backend for tests, calibration on gold /
corrupted-gold / encoder-complement trees). Judge = the self-training admission
filter; low priority, cheap (~$1 per 1K trees at Haiku rates, half in batch).

### HARD-GOLD GENERATOR MERGED (Sonnet routine, 2026-09-08)
scripts/gen_hard_gold.py + src/nsm_ct/hard_gold_templates.py: 8 families x
66 typed-slot templates (imperative, pure/content interjection, elision,
additive/focus, quantity, synth-subject, speaker prime), pools from USVS +
WordNet (N 60, ADJ 60, VT 40, VI 30, INTERJ_PURE 20, INTERJ_CONTENT 40, PROPN 6
= the curriculum name universe, PRON 5, QUANT 4). --per-family 100 seed 0 ->
971 records, 0 gate failures (same gate as the 16 drafts), splits train 687 /
test_filler 142 / test_template 142 (2 whole templates per family held out).
Smoke train: loss falls 27.4 -> 16.5 in 40 steps; sense_recall 0.95 vs 0.05
random. load_gold now accepts comma-separated files (teacher gold + hard gold).
On mainline: the three split files (2.1 MB); hard_gold_all.jsonl dropped (dup).
KNOWN GAPS (follow-up dispatched): (1) hand_gold.ground_W / senses_of look up
the RAW surface, so regular plurals (dogs, boys) and regular pasts (walked)
return no senses and were excluded from pools -- N_pl is irregular-only (30),
VI_past only 10. Fix = WordNet morphy lemmatization in the grounding lookup
(check the teacher-gold path grounds plurals the same way). (2) No a/an
agreement ("a optic"). (3) Fillers are semantically random ("the poultry
planted the physician") -- fine for structure, but the judge/encoder should
not be trained to accept them as MEANINGFUL; keep them for structure only.

### K-12 CORPUS SCOUT MERGED (Sonnet routine, 2026-09-08) -> dev/reports/K12_CORPUS_SCOUT.md
19 sources evaluated (license, grades, questions?, format, fetch, parse yield).
TOP-3: (1) FairytaleQA -- Apache 2.0, 278 stories / 10,580 expert-written
free-text QA pairs with an explicit/implicit tag (explicit = answer in the
passage; implicit = inference -- a stepping stone toward "not stated"); teacher
parse yield 94.0% (vs Gutenberg baseline 81.2%). (2) MCTest -- MSR research
terms, 1,160 grade-school stories x 4 MC questions; 88.0% yield. (3) African
Storybook -- CC-BY leveled readers, NO questions; 94.0% yield. No source ships
"not answerable" items; "not stated" must be synthesized (distractor questions
from other passages) or come later. CLEAR corpus = per-excerpt readability
score, useful as the curriculum DIFFICULTY ORDER, not as training text.
CAVEAT: the cloud sandbox's egress blocks huggingface, gutenberg.org, CMU,
StoryWeaver, ASP, GDL, Bloom, Core Knowledge, EngageNY, Wikipedia -- only
GitHub-mirrored sources could be fetched; RACE/CBT/CoQA/StoryWeaver need a
fetch from an unrestricted network (scripts/fetch_k12.py has the recipes).
Samples under data/k12_samples/. NEXT: FairytaleQA -> Episode converter
(dispatched) so the comprehension phase has real K-12 QA ready when the
encoder is.

### HARD-GOLD v2 MERGED: lemmatized grounding + a/an agreement (2026-09-08)
USVS.senses_of_surface(word): raw surface first, then WordNet morphy over
noun/verb/adj; groundings now carry a `lemma` field (contract v2 updated).
Pools: N_pl 30 -> 60 (regular plurals now ground), VI_past 10 -> 30, VT_past
39 -> 40. Regenerated at --per-family 150: 1,368 records, 0 gate failures
(train 979 / test_filler 185 / test_template 204, 3.3 MB). a/an agreement fixed.
FINDING (material): the TEACHER gold path did NOT lemmatize either --
build_encoder_gold_v2.ground_word called raw senses_of, so inflected content
words in v2/v3 gold (past-tense verbs, plural nouns) ground as bare `entity`
whenever the surface form is not itself a WordNet lemma. build_encoder_gold_v2
now uses the lemmatized lookup, but v2 and v3-small gold were NOT rebuilt (the
arms are mid-run on them; both share the defect, so the v2-vs-v3 comparison
stays fair). NEXT gold build (v4) inherits the fix automatically; expect
materially more sense-grounded nodes. See dev/archive/HARD_GOLD_GEN_STATS.md v2 notes.

### LLM PARSE JUDGE MERGED (Sonnet routine, 2026-09-08) -> dev/archive/PARSE_JUDGE.md
src/nsm_ct/tree_render.py (render_tree: sentence + one `[kind] PREDICATE=w
(gloss) | ROLE=w (gloss)` line per clause, <= 600 chars; the complement probe
now uses it), src/nsm_ct/parse_judge.py (Judge interface; HaikuJudge =
claude-haiku-4-5 via messages.parse + Batches API keyed by custom_id, retry/
backoff, fail-fast on 4xx; MockJudge = J1/J2/J3 + first-sense/POS agreement;
hard J3 pre-filter before any API call), scripts/judge_parses.py (--filter good
= the self-training admission gate), scripts/calibrate_judge.py. 32 tests green.
MOCK calibration: gold 84% good; corrupted 84% detected (100% on predicate-
swap and phantom-role corruptions; SUBJECT/OBJECT swaps are invisible to
structure by design -- the Haiku judge exists for those); encoder complement
16.9% good. Cost estimate ~$0.79 per 1K trees standard, ~$0.40 batched (from
rendering length, not measured). Haiku calibration NOT run: no API key in the
sandbox; command in dev/archive/PARSE_JUDGE.md for the lead.
SIDE-FINDING (consistent with the lemmatization gap): in v2 gold, top-tree
PREDICATE groundings are entity 1,113 vs sense 820 -- most past-tense verbs
were never sense-grounded. v4-small (dispatched, lemmatized) should invert that.

### GOLD v4-SMALL BUILT (lemmatized) + a POS-blind lemmatizer bug found (2026-09-08)
Branch encoder-gold-v4-small (NOT merged; superseded by v4b in flight). Same
1,475-sentence population, 60 s parse cap: 986 records (v3: 959; all 26 v3
cap-hits recovered), all 1 tree, holdout coverage 98/98, v2 coverage 985/985.
THE number: sense-grounded nodes 4,306 -> 5,860 (+1,554), entity-grounded
2,835 -> 1,655; 1,311 groundings exist only because of lemmatization (took ->
take, depended -> depend, beds -> bed). BUG in the samples: morphy is called
without POS -> `was` -> lemma `wa` -> washington.n.02; `can` (AUX) -> can.n.01;
`to` (ADP) -> tho.n.01; VERB `engraved` -> engraved.s.01 (adjective first).
Fix dispatched (v4b): POS-aware morphy from the tagger tag, no morphy on
function words, POS-matched senses ordered first (candidates kept). v4b is
the gold for the lemmatized training arm; v4 is a data point only.

### GOLD v4b-SMALL: POS-AWARE LEMMATIZATION MERGED (2026-09-08)
senses_of_surface(word, pos_hint): function tags (AUX/DET/ADP/PART/PRON/CCONJ/
SCONJ/PUNCT/NUM/SYM) never reach morphy; content tags restrict morphy to the
mapped WordNet POS and POS-priority-sort the candidates (nothing dropped);
wrong-POS raw hits get the POS-matched lemma's senses merged in first. Both
gold builders thread the tagger POS through. v4 -> v4b on the same 1,475
sentences: records 986 -> 987; sense nodes 5,860 -> 5,411 (function words no
longer sense-grounded); FIRST-candidate POS matches the tagger 2,440 -> 4,784;
function-word sense groundings 994 -> 535 (the rest are raw-surface hits, e.g.
`can`, left as candidates by design); holdout 98/98. Samples: took->take.v.01,
began->begin (verb), was (AUX)->entity, engraved->verb sense first. Hard gold
regenerated with the same helper (splits updated). 47 tests green. Gold v4b
(branch encoder-gold-v4b-small, jsonl stays there) is the input for the
lemmatized training arm, to run after the v2/v3 arms finish.

### ARMS seed 1: v2_788 vs v3_788 at 8,000 steps -- TOP-1 GOLD AT EQUAL STEPS LOSES (2026-09-08)
Branch encoder-arms-seed1 (runs/arms/*.log, summary.tsv, .pt). Same 98 held-out
sentences, same code (mainline w/ D1-D6), beam 6 / k 6, subset_seed 1.
| arm (seed 1)      | targets | best-of-6 edge_P/R | rank-1 edge_F1 | struct-exact | width |
| v2_788 (forest)   | v2      | 0.713 / 0.657      | **0.557**      | 0.066        | 5.99  |
| v2_788 (forest)   | v3      | 0.712 / 0.489      | 0.446          | 0.032        | 5.99  |
| v3_788 (top-1)    | v2      | 0.442 / 0.546      | **0.373**      | 0.017        | 5.96  |
| v3_788 (top-1)    | v3      | 0.540 / 0.537      | 0.421          | 0.021        | 5.95  |
| run-2 reference   | v2      | 0.700 / 0.695 (k8) | 0.534          | 0.062        | 7.97  |
READ: (1) the tooling REPRODUCES run-2 (v2_788: rank-1 0.557 vs 0.534, best-of
0.713 vs 0.700) -- the comparison is trustworthy, and v2-gold training passes the
rank-1 >= 0.55 gate, barely. (2) Top-1 gold at the SAME optimizer budget is
clearly WORSE (-0.18 rank-1 on v2 targets, -0.03 even on its own v3 targets).
Two non-exclusive causes: (a) OVERFIT -- v3 has 1 derivation/record so 8,000
steps = 325 epochs (v2: 101); (b) the forest's "spurious" alternative trees act
as data AUGMENTATION (attachment variety) that single-tree gold removes. (3)
Forest width ~6 for both = beam width; decode always returns k -> width is NOT
a gold property, dropped as a gate. (4) Structure-exact is ~0.02-0.07 everywhere:
whole-tree correctness is the real ceiling problem, unchanged by any gold.
NEXT (tooling in flight, branch encoder-train-arms-v2): keep-best-on-dev
checkpointing + learning curves separate (a) from (b). Arms: v2_788 (forest),
v4b_788 (top-1, lemmatized, POS-aware), v4b_788_hard (+generated hard gold,
scored per family on held-out templates). If v4b-keep-best still loses to
forest gold, the answer is D1's `margin` mode (keep genuinely distinct
alternatives) rather than strict top-1 -- a data point for the lead, not a
relitigation: forest variety appears to be training signal.
Seed 0 still running; FairytaleQA finish and arms-v2 tooling in flight.
SEED 0 REPLICATES (branch encoder-arms-seed0): v2_788 rank-1 F1 0.557 (seed 1:
0.557 -- identical to 3 decimals), v3_788 0.410 on v2 targets / 0.488 on v3
targets (seed 1: 0.373 / 0.421). Two seeds, same direction, gap 0.15-0.18.
VERDICT: strict top-1 gold at equal budget is worse than forest gold. Next arm
set adds v4b gold in D1's `margin` mode and in `all` (forest) mode so top-1 vs
margin vs forest is compared at EQUAL (lemmatized, POS-aware) quality, with
keep-best checkpointing. Gold builds dispatched (branch encoder-gold-v4b-variants).
ARMS COMPLETE (both seeds merged; runs/arms/summary.tsv has all four rows):
| arm | seed | v2-target rank-1 F1 | v3-target rank-1 F1 | best-of-6 edge_P |
| v2_788 forest | 0 | 0.556 | 0.468 | 0.691 |
| v2_788 forest | 1 | 0.557 | 0.446 | 0.713 |
| v3_788 top-1  | 0 | 0.410 | 0.488 | 0.450 |
| v3_788 top-1  | 1 | 0.373 | 0.420 | 0.442 |
Note: on the top-1 (v3) targets the two training golds are roughly tied
(0.45-0.49 vs 0.42-0.49); the forest-trained model is strictly better on
forest targets and never worse elsewhere. Precision is the big loser under
top-1 training (0.69-0.71 -> 0.44-0.45): the top-1-trained model over-emits.

### TRAIN-ARMS v2 TOOLING MERGED: dev eval + keep-best + curves (2026-09-08)
train_encoder.py: --eval-every N (k=1 rank-1 F1 on the DEV holdout, logged to
<out>.curve.tsv), --keep-best (best-dev checkpoint saved as <out>, last as
<out>.last.pt, best_step in config), --patience, --extra-eval (comma list of
extra gold files scored per `family`), epochs_equivalent in the summary.
run_encoder_arms.sh: EVAL_EVERY=250 KEEP_BEST=1 defaults, ARMS filter, new arms
v4b_788 and v4b_788_hard (v4b + generated hard gold, scored on the held-out
hard-gold splits per family). tests/test_encoder_curve.py. Final eval reported
for BOTH the kept-best and the last checkpoint.

### GOLD v4b VARIANTS BUILT (2026-09-08) -> branch encoder-gold-v4b-variants
Same 1,475 sentences, lemmatized POS-aware grounding, three forest policies:
| policy | records | trees/sent | 1 / 2 / 3+ tree share | derivations | nodes/record (all trees) |
| top-1  | 987 | 1.000 | 100 / 0 / 0     | 987   | 9.15  |
| margin | 986 | 1.398 | 79 / 10 / 11    | 1,378 | 14.27 |
| all    | 986 | 3.563 | 12.5 / 21.7 / 65.8 | 3,513 | 32.88 |
tree[0] identical across policies (0 mismatches); holdout 98/98 for all.
Margin's kept 2nd trees: 41.5% differ in clause count, 58.5% in a core role
assignment -- i.e. genuine structural alternatives, not modifier wobble.
ARMS-2 DISPATCHED (keep-best on dev, 8,000 max steps, same 98 holdout):
 R-a: v4b_788 (top-1) + v4b_margin_788, seed 0      -> branch encoder-arms2-a
 R-b: v4b_all_788 (forest) + v4b_788_hard, seed 0   -> branch encoder-arms2-b
 R-c: v2_788 (forest control, keep-best) + v4b_margin_788 seed 1 -> encoder-arms2-c
Question: at equal gold quality, does variety (margin/all) beat top-1, and does
keep-best rescue top-1? Plus the first per-family numbers on the hard-gold
held-out templates (v4b_788_hard).

### FAIRYTALEQA CONVERTED + CHARACTERIZED (2026-09-08) -> dev/reports/FAIRYTALEQA_STATS.md
Branch k12-fairytaleqa merged: fetch_k12 --source fairytaleqa (277/278 stories,
4,082 sections, 10,556 QA rows, Apache 2.0, raw data gitignored; manifest +
10-story sample committed), convert_fairytaleqa.py (--parse none|sample|all),
build_fairytaleqa_stats.py, eval_prose.py --answer-type entity hook, tests.
Episodes committed: runs/fairytaleqa_episodes.jsonl (1,886 episodes: train
1,561 / val 186 / test 139; attribute mix action 31%, causal 27%, character
13%, feeling 9%, outcome 7%, prediction 7%, setting 7%; explicit 77%).
NOTE: 1,886 episodes from 10,556 QA rows is NOT yet explained -- the converter
emits one episode per (section, question); the drop accounting is checked in
the run log below / next routine. Treat the stats as a characterization of the
converted subset until confirmed.
PARSE YIELD on FairytaleQA sentences (2,000-sentence sample, teacher parser):
strict full-tree 19.8%, usable 58.8%, grounding-fail 40.7% (pronoun-
unresolvable 14%, unknown-word 10%, unsupported-construction 8%), cap-hit 0.5%.
Only 2.3% of episodes have EVERY passage sentence usable. The scout's 94% was
a looser criterion; by the M62 standard, children's prose is NOT easy for the
teacher -- perception is the bottleneck here too, which is the encoder's job.
ANSWER TYPES (the number for the comprehension plan): (a) entity/NP verbatim
in passage 19.5% (367 episodes, all MC-scoreable with the current QA head);
(b) verb phrase 5.6%; (c) free-text sentence 14.7%; (d) not a substring of
the passage at all 60.2%. => ~80% of FairytaleQA needs the realization side
and/or inference beyond a single memory read. The K-12 phase should start on
the 367 entity items + MCTest (4-way MC, distractors provided) while the
decoder/realization question is reopened.

### ARMS-2 COMPLETE (keep-best on dev, 8K max steps, same 98 holdout) -- the arms-1 verdict was a TARGET-SHAPE artifact (2026-09-08)
Kept-best rank-1 edge-F1, two target sets (v2 = old forest-era trees; v4b = current
richer, lemmatized, POS-aware top-1 trees):
| arm (gold)                | seed | on v2 targets | on v4b targets | best step (epochs) |
| v2_788  forest, old gold  | 0    | **0.583** (P .73) | n/a (arms-1: 0.469/0.446 on v3) | 750 (9.5) |
| v4b_788 top-1             | 0    | 0.391 | **0.491** (P .58) | 1,250 (~50) |
| v4b_margin_788            | 0    | 0.384 | 0.445 | 6,750 |
| v4b_margin_788            | 1    | 0.381 | 0.471 | 1,750 (51) |
| v4b_all_788 forest        | 0    | 0.421 | 0.470 (P .62) | 4,250 (48) |
| v4b_788 + hard gold (979) | 0    | 0.312 | 0.404 | 1,500 (61) |
READ: (1) Each model scores best on the target shape it was trained on. The
arms-1 "forest beats top-1" gap (0.557 vs 0.37-0.41) was v2-shaped targets
rewarding the v2-trained model; on the CURRENT gold's targets, top-1 (0.49) >=
forest (0.47) ~ margin (0.45-0.47) ~ old v2 model (0.45-0.47). VARIETY DOES NOT
HELP at equal gold quality. D1 (top-1) stands. (2) Keep-best matters: every arm
peaks early (v2 at 750 steps = 9.5 epochs; v4b top-1 at 1,250); arms-1's fixed
8,000 steps overtrained everything; keep-best lifted the v2 control 0.557 ->
0.583. (3) The richer targets are HARDER: ~0.49 committed edge-F1 vs 0.58 on the
sparser old trees; whole-tree exact stays ~5%. The gate for "encoder works" must
be stated on the current gold's targets: rank-1 0.49 today. (4) Hard gold mixed
1.2:1 with real gold HURTS in-domain (0.49 -> 0.40): the mix ratio / curriculum
needs tuning (e.g. 1:4, or hard gold as a second phase).
HARD-GOLD GENERALIZATION (first real numbers; v4b_788_hard, rank-1 F1 per family):
| family              | unseen fillers | unseen templates |
| pure_interjection   | 1.000 | 1.000 |
| content_interjection| 0.992 | 0.979 |
| imperative          | 0.956 | 0.272 |
| additive_focus      | 0.750 | 0.233 |
| elision             | 0.708 | 0.711 |
| quantity            | 0.633 | 0.667 |
| synth_subject       | 0.667 | 0.667 |
| speaker_prime       | 0.482 | 0.662 |
Interjections and elision GENERALIZE to unseen templates; imperatives and
additive/focus are learned per-template (0.96 -> 0.27, 0.75 -> 0.23): those two
families need more template variety (the generator has 8-9 each; widen).
NEXT: the lever is DATA VOLUME at v4b quality. 16K-corpus v4b gold build
dispatched to the cloud box (branch encoder-gold-v4b-16k); then arms v4b_3000 /
v4b_8000 with keep-best on the same 98 holdout = the scaling curve the audit
asked for, done properly. Separately: hard-gold mix-ratio arm (1:4) and more
imperative/additive templates.

### HARD-GOLD v3 MERGED: imperative 9 -> 29 templates, additive/focus 8 -> 22 (2026-09-08)
Motivated by arms-2's per-family result (imperatives/additive memorized per
template). 86 templates total; 4 templates per family held out for the two
widened families (imperative held-out: imp_neg_vt_obj, imp_vocative,
imp_vt_obj_adv, imp_vt_obj_place_on; additive: add_and_a_n_too, add_pron_also,
add_pron_too, add_propn_also). 1,448 gated records, 0 failures (train 1,038 /
test_filler 200 / test_template 210). New runs/hard_gold_train_small.jsonl =
200-record family-stratified sample for a 1:4 mixing arm. Next arm set (after
the 16K gold): v4b_3000, v4b_8000, and v4b_788 + hard_small (1:4) with per-
family extra-eval on the wider held-out templates.

### LEAD DIRECTIVE (2026-09-08): score trees in USVS space, not binary
Loss and evaluation must compare the FLATTENED output tree to the gold tree in
USVS space (graded similarity of node groundings, with role/attachment
weighting), not binary edge match. Today both are binary: teacher-forced CE over
discrete actions; edge-F1 on exact (role, token) pairs. A near-miss sense or a
one-head-off modifier scores 0, same as a phantom, and yields no gradient.
Dispatched (Opus design + build): usvs_tree_similarity metric; re-score every
arms-1/2 checkpoint on it alongside edge-F1; soft-target loss option for the
heads that make graded choices; one smoke arm. FairytaleQA is confirmed as
COMPREHENSION data only (parser sees it only at run time / via judge-admitted
self-training later).

### GOLD v4b-16K BUILT (2026-09-08) -> branch encoder-gold-v4b-16k (68 MB jsonl stays there)
Full corpus-expand population, top-1, lemmatized, POS-aware, 30 s cap, 4 workers
(builder gained --workers; 0.20 s/sentence/worker; ~56 min): 16,411 sentences ->
10,225 records (62.3%); no-hypothesis 28.9%, grounding-fail 8.2%, cap-hit 0.6%.
Per-source yield: Burgess 80%, Bryant 75%, McGuffey 68%, Andersen 66%, Jacobs
59%, Grimm 58%, Edgeworth 57%, Alice 48%. Nodes 94,151: sense 60.9%, entity
23.2%, reference 14.2%, prime 1.7%. Holdout test 98/98 covered. (Dev holdout
file is regenerated by make_holdout.py at arm setup; not committed.)
ARMS-3 DISPATCHED: (a) SCALING v4b_3000 + v4b_8000 (seeded subsets of the 16K
gold, holdout excluded, keep-best, 8K max steps) -> encoder-arms3-a; (b) MIX
v4b_788 + hard_gold_train_small (1:4) with per-family eval on the v3 held-out
templates -> encoder-arms3-b. Both report edge-F1; the graded USVS metric
lands separately (branch usvs-graded-scoring) and will re-score these.

### USVS-GRADED SCORING BUILT + ALL ARMS RE-SCORED (Opus routine, 2026-09-08) -> branch usvs-graded-scoring
src/nsm_ct/usvs_graded.py: node vector = candidate-set mean in USVS + a
reserved block for entity/reference/elision/prime (type + surface identity);
pairwise score = role_weight x max(0, cosine); role-confusion matrix in one
place; predicate 2x / core 1x / modifier 0.5x; exact Hungarian alignment;
graded P/R/F + clause-structure term; forest handling. Sanity: identity 1.000;
corrupted 0.393; RANDOM OTHER SENTENCE 0.285 = the ambient USVS floor (sense
signatures are non-negative) -- every graded number must be read against 0.285,
not 0. Re-score of all 9 checkpoints (rank-1, 98 holdout):
| arm            | v2 edge-F1 | v2 graded-F | v4b edge-F1 | v4b graded-F |
| v2_788 kb      | 0.583 | 0.623 | 0.465 | 0.556 |
| v4b_788 kb     | 0.391 | 0.459 | 0.491 | 0.581 |
| v4b_margin s0  | 0.384 | 0.483 | 0.445 | 0.564 |
| v4b_margin s1  | 0.381 | 0.468 | 0.471 | 0.562 |
| v4b_all kb     | 0.420 | 0.487 | 0.470 | 0.557 |
| v3_788 s0      | 0.410 | 0.492 | 0.488 | 0.544 |
| v4b_788_hard   | 0.312 | 0.390 | 0.404 | 0.498 |
Ranking on v4b targets unchanged at the top (v4b top-1 leads on both metrics);
minor reorderings below. The graded scale is compressed (floor 0.285, so
0.58 graded ~ 0.41 above floor) -- the arms' "wrong" edges are mostly near-
misses in USVS space, not nonsense, which is what the lead expected.
SOFT LOSS (--loss usvs-soft): role head soft over the confusion matrix (T=0.25),
gtype head soft over the contract's equivalences, source head soft
(context ~ memory); action/terminal/kind/prime heads stay hard. FINDING: the
encoder NEVER scores sense candidates (candidates-first) and has NO node
embedding, so a sense-space gradient has no head to attach to under the
current architecture. The one hook is a new Linear(controller_hidden, d_axes)
projection trained with cosine loss against the gold node's USVS vector --
new architecture, small, opt-in. Dispatched (branch usvs-aux-head) as the
direct implementation of the lead's "loss in USVS space" directive.

### ARMS-3(b) interim: widened templates FIX imperative/additive generalization (2026-09-08)
v4b_788 + FULL v3 hard gold (1,038 records, 1.3:1), keep-best (step 2,500):
in-domain rank-1 F1 0.464 on v4b targets (arms-2 mix with v2 hard gold: 0.404;
no-mix reference 0.491) / 0.411 on v2 targets. Per-family rank-1 F1 on the v3
held-out splits (unseen fillers / UNSEEN TEMPLATES, 4 templates held out for
the widened families):
| family               | fillers | templates | arms-2 templates |
| imperative           | 0.915 | **0.781** | 0.272 |
| additive_focus       | 0.742 | **0.729** | 0.233 |
| pure_interjection    | 1.000 | 1.000 | 1.000 |
| content_interjection | 1.000 | 0.978 | 0.979 |
| elision              | 0.694 | 0.708 | 0.711 |
| synth_subject        | 0.667 | 0.667 | 0.667 |
| quantity             | 0.667 | (n=4) | 0.667 |
| speaker_prime        | 0.322 | 0.429 | 0.662 |
READ: template variety is what generalization needs -- 29/22 templates moved
unseen-template scores from ~0.25 to ~0.75. speaker_prime regressed (0.66 ->
0.43; it was not widened and its n is small) -- next generator pass should widen
it too. The 1:4 mix arm (200 hard records) is still running; it decides whether
the in-domain cost (0.491 -> 0.464) can be recovered.

### ARMS-3(b) COMPLETE: hard-gold mix ratio -- use the FULL hard gold (2026-09-08)
| arm (v4b_788 + hard gold, v3 templates) | in-domain rank-1 (v4b tgt) | imperative unseen-tmpl | additive unseen-tmpl | interjections |
| no hard gold (reference)                | 0.491 | -- | -- | -- |
| + 200 hard (1:4)                        | 0.473 | 0.605 | 0.680 | 0.88-0.97 |
| + 1,038 hard (1.3:1)                    | 0.464 | 0.781 | 0.729 | 0.98-1.00 |
READ: the in-domain cost of mixing is 0.02-0.03 either way (seed-noise scale);
the 1:4 sample gives up ~0.1-0.2 on the hard families for it. DECISION: train
with the FULL hard gold from now on. At 16K-scale real gold (8,000 records)
the full hard set is a natural ~1:8, so the next scaling arm should be
v4b_8000 + hard_gold_train (v4 templates once merged). speaker_prime stays
weak (0.30-0.43) -> generator v4 widening (in flight).

### HARD-GOLD v4 MERGED: all eight families widened (2026-09-08)
Templates 100 -> 149 (elision 8->20, quantity 8->20, speaker_prime 9->26,
synth_subject 8->22; imperative 29 and additive 22 from v3); 4 held out per
widened family. 1,531 gated records, 0 failures (train 1,083 / test_filler 217
/ test_template 231); hard_gold_train_small regenerated (200). Schema cannot
express: `let 's {VT} ...` (first-person-plural synth subject), `{PROPN} as
well .` (two-token additive particle), `it is for me .` (copular + speaker) --
noted for a later contract extension. Next: the v4b_8000 + full hard gold arm
with per-family eval on these wider held-out templates.

### THE SCALING CURVE (ARMS-3a, seed 0, keep-best, same 98 holdout) -- DATA-LIMITED CONFIRMED, PROPERLY (2026-09-09)
| train sentences (v4b gold) | rank-1 F1 v4b targets | P | struct-exact | rank-1 F1 v2 targets | best step (epochs) |
| 788    | 0.491 | 0.585 | 0.05 | 0.391 | 1,250 (50) |
| 3,000  | 0.549 | 0.667 | 0.19 | 0.480 | 1,750 (19) |
| 8,000  | **0.632** | **0.713** | **0.21** | 0.534 | 6,500 (26) |
Monotone and still climbing at 8,000 (dev rank-1 0.53 at the best step; last-
step test ~equal to best -> not yet saturated). Whole-tree exact 5% -> 21%.
This is the audit's scaling gate passed on the CURRENT targets with keep-best:
the committed encoder clears 0.55 at 3,000 sentences and 0.63 at 8,000 -- the
learned encoder is earning its place, and the lever is exactly what the lead
said: way more grammatical English gold. Next: 10,225 (all of the 16K gold) +
full v4 hard gold; then the next corpus expansion (the K-12 readers parse at
~60-94% and are the natural next source).

### USVS AUX HEAD: honest negative at 788 (2026-09-09) -> branch usvs-aux-head (merged)
Opt-in Linear(controller -> USVS axes) + cosine loss on emitted nodes, W=0.5,
with --loss usvs-soft. The head LEARNS the space (mean cosine to gold node
vectors 0.80 on the holdout) but the committed tree got WORSE: rank-1 0.438 vs
0.491 (v4b targets), 0.340 vs 0.391 (v2). One seed, one weight, 788 sentences.
READ: at this data scale the auxiliary objective competes with the structural
one. Not dead -- re-test at 8,000 sentences with W in {0.1, 0.25}, and note the
graded metric column when the routine's final report lands. The soft-target
loss alone (roles/gtype/source) was not isolated in this pair; that arm is
cheap and should run with the retest.

### USVS-space loss retest at 8,000 (ARMS-4b, seed 0, keep-best) -- BOTH ARMS DONE, ROBUST HONEST NEGATIVE (2026-09-09) -> branch encoder-arms4-b
Isolating the soft-target loss from the aux head at scale, against the
v4b_8000 default-loss reference (rank-1 edge-F1 0.632 on v4b targets, best
step 6,500). Two arms, `--loss usvs-soft`, same 8,000-sentence v4b_16k pool
/ 98-sentence holdout as the scaling curve above; both eval on v4b targets
(primary) and v2 (alt), --metric graded on both.

| arm | best_step | edge-F1 v4b (best / last) | graded-F v4b (best / last) | edge-F1 v2 (best / last) | graded-F v2 (best / last) | head-cosine v4b/v2 |
|---|---|---|---|---|---|---|
| default (reference) | 6,500 | 0.6326 / 0.6277 | n/a (no graded eval recorded; checkpoint not retained) | 0.5342 / 0.5331 | n/a | n/a |
| soft-only (`LOSS=usvs-soft`) | 7,000 | 0.5787 / 0.5647 | 0.6540 / 0.6480 | 0.4755 / 0.4825 | 0.5706 / 0.5764 | n/a (aux off) |
| soft + aux W=0.1 (`LOSS=usvs-soft AUX=0.1`) | 5,000 | 0.5768 / 0.5506 | 0.6525 / 0.6289 | 0.4880 / 0.4738 | 0.5749 / 0.5651 | 0.7709 / 0.6511 (best); 0.7679 / 0.6498 (last) |

Neither USVS-space loss variant beats the default at 8,000. The soft-target
loss ALONE (no aux head) still trails by -0.053 edge-F1 on v4b (0.579 vs
0.632) and -0.059 on v2 -- so the 788-sentence "honest negative" was never
about the aux-head pairing specifically; the soft role/gtype/source targets
themselves cost structural F1, at both scales tested. Lightening the aux
head from W=0.5 (788-sentence arm) to W=0.1 does not close the gap either:
soft+aux0.1 sits within noise of soft-only (0.577 vs 0.579 v4b, 0.488 vs
0.476 v2 -- the W=0.1 aux head is mildly LESS harmful on v2 and mildly more
on v4b, not a clear win either way) despite the head learning the USVS space
just as well as at 788 sentences (cosine 0.77 v4b / 0.65 v2, vs 0.80 at
788). READ: this is now a robust negative across two data scales and two aux
weights, not a scale-starved one -- the soft CE targets on
role/grounding-type/source compete with the structural training signal
regardless of how much data or how light the aux term is. Not recommended
for the default training recipe.

### USVS-SPACE LOSS: NEGATIVE AT 8,000 TOO -- metric stays, loss dropped (ARMS-4b, 2026-09-09)
Same 8,000-sentence pool, keep-best, seed 0, v4b targets:
| loss                          | rank-1 edge-F1 | best step |
| default (hard CE)             | **0.632** | 6,500 |
| soft-target only              | 0.579 | 7,000 |
| soft-target + aux head W=0.1  | 0.577 | 5,000 |
(at 788: default 0.491; soft + aux W=0.5 0.438.) The graded loss variants
trail the default by ~0.05 at both scales and in every combination tried.
DECISION (director): keep the USVS-graded METRIC as the evaluation (it is
the lead's directive for scoring, and it is more informative than edge-F1);
train with the hard targets. The aux head stays in the code, off by default.
Left open: a graded loss might pay once the encoder scores sense candidates
itself (it does not today -- candidates-first), i.e. it is a comprehension-
side idea more than an encoder one.

### MAIN LINE AT FULL SCALE: v4b ALL 10,225 + full v4 hard gold (ARMS-4a, 2026-09-09) -- THE CHECKPOINT
Keep-best at step 8,000 of 12,000 (dev 0.550; ~34 epochs total), 98 holdout:
| targets | rank-1 edge-F1 | P | R | struct-exact |
| v4b (current) | **0.622** | **0.750** | 0.716 | 0.214 |
| v2 (old)      | 0.551 | 0.594 | 0.718 | 0.112 |
vs v4b_8000 (no hard gold): 0.632 / P 0.713 / struct 0.214 -- equal within
noise, precision +0.04. Adding the 1,083 hard records at ~1:9 cost nothing.
Hard families (v4 held-out; unseen fillers / unseen templates): pure interj
1.00/1.00, content interj 0.99/0.83, imperative 0.83/0.66, additive 0.75/0.75,
elision 0.68/0.68, quantity 0.67/0.54, synth_subject 0.65/0.53, speaker_prime
0.36/0.41. At 1:9 the hard families sit ~0.1 below the 1.3:1 mix on unseen
templates; speaker_prime remains the weak family even after v4 widening.
CHECKPOINT: branch encoder-arms4-a, runs/arms/v4b_all_hard_0.pt (kept-best).
This is the encoder for the comprehension phase. Scaling is not saturated
(10K -> ? ); next gold = the K-12 readers (v5). Dispatched: v5 gold build over
FairytaleQA sentences (branch encoder-gold-v5-k12); and an Opus design +
prototype of ENCODER-DRIVEN episode building (learned encoder's lattice ->
Episode, replacing the parser in corpus.py) measured on the 1,923 entity-
answer FairytaleQA items (branch encoder-episodes).

### INDEPENDENT AUDIT #2 (Opus, fresh eyes, 2026-10-15) -- CONTINUE WITH CHANGES: freeze encoder, front-load comprehension
Full report: consciousness_transformer/dev/reports/AUDIT_2026-10-15.md. Five weeks idle
since 09-09; the two routines dispatched then (encoder-gold-v5-k12,
encoder-episodes) never pushed and the trigger list was empty -- lost.
Load-bearing findings:
1. 0.622 rank-1 edge-F1 is honestly measured: no train/holdout leakage (0 of
   10,225 records overlap the 98 at Jaccard>=0.8), keep-best picks on dev not
   test, rank-1 = forest[0] sorted by logprob, modifier roles counted.
2. BUT no cheat baseline (CLAUDE.md law). A ~30-line POS heuristic scores
   edge-F1 0.40 on the same holdout -> real margin ~+0.22. Metric lenient:
   sense choice never scored, edges pooled across clauses.
3. Teacher gold is PARTIAL: ~46% of VERB tokens appear in their gold tree; 39%
   of records cover <50% of verbs. The student learns to drop clauses too.
4. The planned FairytaleQA MC benchmark is BROKEN: distractors come from OTHER
   stories while the answer is verbatim in the passage; "pick the option in
   the passage" scores ~97% (553/584 + 31 ties, 0 misses).
5. Previous audit's finding 8 (comprehension spike) still IGNORED; finding 1
   (complement) measured once, never on the new checkpoint; hybrid rejected
   before any downstream number existed.
Recommended order (info/hour): (1) fix benchmark + cheat baselines; (2) e2e
QA spike on teacher-built episodes w/ memory-severed control; (3) encoder vs
teacher vs hybrid downstream; (4) gold clause-coverage audit; (5) abstention.
DIRECTOR (2026-10-15): dispatched (1) -> branch k12-bench-fix, (4) ->
gold-coverage-audit, and re-dispatched v5 gold -> encoder-gold-v5-k12 (cheap,
still needed as parsed FairytaleQA for episodes). (2) follows (1). Held for
LEAD: freeze encoder scaling; permit the hybrid comparison (3) as a
MEASUREMENT (learned encoder stays default per 09-08 decision until a number
says otherwise).

### GOLD COVERAGE AUDIT (2026-10-15) -> branch gold-coverage-audit
Full report: consciousness_transformer/dev/reports/GOLD_COVERAGE_AUDIT.md. Reproduces
+ sharpens audit #2 finding 3. All 10,225 records, torch-free
(scripts/audit_gold_coverage.py): mean VERB coverage 53.5%, 40.5% of records
<50% VERB-covered; 31.9% of 8,009 multi-verb records collapse to ONE clause.
Worst source: edgeworth/grimms/andersen (~49-51%); worst length bin: 26+tok
(42.6%). Hand audit, first 50 holdout sentences (fixed split, no regen
needed): 33/50 (66%) DROPPED_CLAUSE, 7 WRONG_PRED, 1 WRONG_ATTACH, 1
DROPPED_ARG, 8 FULL. GATE (>30% -> fix extraction first): FAILS, 2x over.
ROOT CAUSE, empirically checked (ran all top-8 parser hyps through
extract_discourse, not just hyp[0], on 4 samples): parser (A) + D1 top-1
pruning (C) RULED OUT -- every hypothesis agrees. Cause is (B) extraction,
3 bugs in src/nsm_ct/clause.py: B1 bare-intransitive 2nd/coordinated clause
has no val_idx fallback (_secondary_fact_clauses:630, cf. working fallback
at 804-808); C1 VP-coordination ("did A and did B") misroutes through the
coordinated-VALUE branch, predicate="and" (clause.py:744-755); B2 no path
exists for reported-speech/cognition matrix clauses ("said X", "X thought")
-- their SUBJECT edge is never subs[0]. B1+C1+B2 = primary cause for ~25/40
cases; rest is unwalked subordinate/relative embedding (architectural, not
1 bug). Fix proposed (NOT implemented, per scope): 3 changes, ~45 lines
total, file:line in the doc. Est. gain (rough, n=50 extrapolation): mean
VERB coverage 53.5% -> high-60s/low-70s%; large residual gap remains. NEXT:
lead decides on the 3-part fix before resuming encoder scaling (already
frozen per audit #2).

### CLAUSE EXTRACTION FIX (B1+C1+B2) -> branch clause-extraction-fix (2026-10-15)

Implemented the gold-coverage audit's proposed 3-part fix (clause.py,
extraction-only, parser/USVS/encoder untouched): B1 gave
`_secondary_fact_clauses` a bare-subject fallback (mirrors
`_primary_discourse`'s own); C1 added a structural guard
(`clause_idx in by_coord`) so VP-coordination builds one clause per
coordinated VERB instead of misrouting into coordinated-VALUE with the
coordinator token as a fake shared predicate; B2 added
`_recover_reporting_matrix_clauses` to recover postposed-subject
attribution ("said X"/"thought X", no SUBJECT edge at all) via a closed
REPORTING_VERBS list. 7 new regression tests, all pass; the
clause.py-dependent test subset (40 files/~583 tests -- the full
1302-test suite does not complete near the stated 10min cap in this env):
576 passed / 4 failed, all 4 confirmed PRE-EXISTING (byte-identical on
unmodified clause.py) and unrelated. Before/after gold sample (98 holdout
+ seeded 300 random corpus sentences, shrunk from 400): mean VERB
coverage 54.3% -> 67.9% (+13.6pt), <50%-verb-covered 36.8% -> 20.3%,
multi-verb-collapsed-to-1-clause 30.6% -> 16.4%. Re-judged the 33
original DROPPED_CLAUSE cases: 14 FULL, 4 PARTIAL, 15 still dropped
(parse-cap sentences + the out-of-scope subordinate-embedding bucket +
a few orphaned verb nodes with no edges at all). 0 newly wrong. Full
writeup: dev/reports/CLAUSE_EXTRACTION_FIX.md.

### K-12 BENCHMARK FIXED + CHEAT BASELINES (2026-10-15) -> branch k12-bench-fix
Full report: consciousness_transformer/dev/reports/K12_BENCHMARK_FIX.md. Fixed
convert_fairytaleqa.py's distractor policy: --distractors same-passage
(default) draws distractors from the episode's OWN passage instead of other
stories, so passage-presence no longer identifies the answer. Gate (substring
cheat <= random+0.10) PASSED first try, same 584/584 entity episodes surviving
(>=2 distractors) as before -- no iteration needed. Cross-story substring cheat
0.973 -> same-passage 0.250 (== random exactly). Added
scripts/cheat_baselines_k12.py (random/majority/substring/word-overlap) and
ran it on cross-story, same-passage, and newly-converted MCTest mc160
train+dev (400 eps, native in-story distractors, gate PASSES at 0.260 vs
0.350 -- was never broken). Added scripts/cheat_encoder_baseline.py: the
audit's POS heuristic, scored through the real score_record/aggregate_recall
path -- measures edge-F1 0.268 (P0.246/R0.295) on the 98-sentence v2 holdout
vs encoder-gold-v4b-16k gold (fetched fresh, 10,225 records), vs the
encoder's 0.622 rank-1 -- margin +0.354, notably BELOW the audit's hand
estimate (0.40, never run against code); now wired into train_encoder.py's
final report and eval_encoder.py by default. 14 new tests (distractor
policy, cheat baselines, heuristic baseline), all passing; one pre-existing,
unrelated eval_prose.py test failure found (not touched this branch). Full
1,315-test suite does not finish in a 10-min cap (pre-existing quantum_parser
slowness, not from this change).
