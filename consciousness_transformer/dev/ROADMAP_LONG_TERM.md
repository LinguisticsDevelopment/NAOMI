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
