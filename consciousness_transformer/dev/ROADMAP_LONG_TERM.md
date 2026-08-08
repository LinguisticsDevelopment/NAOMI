# Long-term roadmap (reference — not scoped; scope each stage when it starts)

The order below is dependency order. Nothing here starts until the semantic-mapping
finish line (`SEMANTIC_MAPPING_PLAN.md`) ships its artifact.

## 1. The bridge: grounded fillers in the meaning graph

Replace label-based CONCEPT handles: a content word's filler/handle vector in
`meaning_graph.py` / mind/ meaning objects becomes a **fixed deterministic
projection of its placed coordinate** (named axes → d-dim filler; frozen matrix, no
learning, axes stay readable). Sense selection starts as MFS; the resolver slot
belongs to stage 2. Touch points: `meaning_graph.GraphNode` handles,
`mind/schema.py`, `clause_reactor.build_clause_batch` value vectors.

Extrinsic gate (house rule — substrate milestones must pay downstream): concept
handle-dereference margins improve vs §0j medians (0.034 @ d256), AND at least one
consumer metric moves (STM read-resolution, or option scoring on the ambiguity
curriculum). If neither moves: record the negative, keep the artifact standalone.

## 2. WSD: decoupled, gated, slotted

Interface (settled): the **parser stays untrained** (grammar files —
`mind/grammar.py` / `quantum_parser`; same parser, different grammar file per
language, far-future) and emits lemma + POS + syntactic context. The **WSD module
may be trained**: `resolve(lemma, context) → sense id → placed coordinate`, behind
a facade so the parser never knows which resolver is active.

- Gate 1 — correctness: coherence resolver (`wsd.IterativeSenseResolver` +
  `GroundedWordNetSenseInventory`) vs **MFS** on SemCor (nltk). MFS is the floor
  and is famously hard to beat; winner takes the slot, a loss is recorded and MFS
  stays.
- Gate 2 — task payoff: an **ambiguity-bearing comprehension curriculum**
  (episodes whose answer flips with the sense) so WSD shows up in answer accuracy.
  This doubles as the seed of the comprehension-question training corpus.

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
