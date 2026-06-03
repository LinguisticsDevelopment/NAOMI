# Research Notes

This file separates the **research** (genuinely open, unsolved) from the
**engineering** (built, working, boring on purpose) in the NSM Consciousness
Transformer scaffold, and records the decisions made while building it.

## The research-vs-engineering boundary

**Engineered and working** (don't expect novelty here — these exist to make the
research tractable):

- The transformer block, the two output heads, the training/eval loops, the
  loss *combination* machinery, the tokenizer, the config system, the toy data
  generator, and the NSM prime constants.
- The abstract interfaces (`AbstractParser`, `AbstractSemanticMapper`,
  `AbstractMemory`) and the `FeatureBuilder` that composes them.

**Mocked / placeholder** (the actual research lives behind these seams):

- Semantic composition onto NSM primes (`MockSemanticMapper`).
- The NAOMI parser (`MockNaomiParser`).
- Long-term / episodic memory (`MockMemoryStore`).
- The consciousness consistency loss (`consciousness_consistency_loss`).

## Open problems

### 1. Semantic composition onto NSM primes
**This is the central unsolved problem and we deliberately did not attempt it.**
`MockSemanticMapper` lights up pseudo-random primes by hashing tokens; it carries
no meaning. The real task: map a parse tree to a structured representation over
the ~65 NSM primes (and NSM "molecules"/semantic templates) such that the
representation actually paraphrases the sentence's meaning. Questions:
- What is the target structure — a bag of prime activations, a prime-typed
  graph, a sequence of NSM explications?
- How is compositionality handled (negation, quantifiers, embedding,
  scope)? NSM has its own grammar of primes; the mapper must respect it.
- How is this supervised? There is no large corpus of gold NSM explications.

### 2. Tree serialization (flat → hierarchical)
`serialize_parse_tree` emits a lossy flat pre-order token stream with `[NODE]`
markers. The model cannot recover hierarchy from it. Open work:
- Structural position encodings (e.g. path-from-root embeddings), recursive /
  tree-LSTM-style encoders, or a graph transformer over the parse.
- Whether to feed *syntax* at all once the semantic mapper exists, or to feed
  the NSM semantic structure directly.

### 3. Consciousness dimension — meaning and ablation
`consciousness_dim` is a free hyperparameter and the state vector is currently
**opaque**: nothing forces its dimensions to mean anything. Open work:
- Ablate `consciousness_dim` (e.g. 8 / 32 / 128 / 512) against task performance
  and state-transition stability. Does more capacity help, or just drift?
- Should the state be *grounded* (e.g. tied to NSM-prime activations, mental
  predicates like THINK/KNOW/WANT/FEEL) rather than free?
- Does the state carry information across turns, or collapse?

### 4. Memory pruning
`MockMemoryStore` returns one fixed pseudo-random vector per query. A real
episodic memory will grow without bound. Open work:
- Retrieval (embedding search) and, crucially, **pruning/forgetting** policy:
  what to keep, merge, or discard, and on what schedule.
- How retrieved memory should interact with the consciousness state (additive
  injection, as now, is almost certainly too weak).

### 5. Coherence checking
There is currently no mechanism that verifies the model's response is
*consistent* with its consciousness state, the retrieved memory, and the
evidence in the passage. Open work:
- A coherence objective or verifier (this is what the consciousness consistency
  loss should eventually become — see below).
- Contradiction detection over the `CausalTable` / NSM representation.

## Placeholder: the consciousness consistency loss
Currently `consistency_loss = MSE(next_state, current_state)` — an *inertial*
"don't drift" prior. This is a stand-in, not a hypothesis. The real auxiliary
objective should reward state evolution that is *coherent* with the evidence,
the memory, and the produced answer, and should probably allow (even require)
large, meaningful state changes when the input warrants them. Marked
`TODO(consciousness-loss)` in `losses.py`.

## Decisions made while building (and why)

- **Self-contained subfolder + package `nsm_ct`.** Kept entirely inside
  `consciousness_transformer/` with its own `pyproject.toml` so it installs and
  tests independently of the Go parsers and `quantum_parser` elsewhere in NAOMI.
- **Plain YAML config, not Hydra.** Smaller dependency surface; typed dataclasses
  give most of the ergonomics. `TODO(config)` notes when to switch.
- **PyTorch, CPU-friendly, tiny defaults.** The model is intentionally small so
  `pytest` and `train_phase1.py` run in seconds on CPU.
- **Causal-LM framing for multiple choice.** Each example expands into 4
  causal-LM rows (one per option); answer prediction reuses the LM head by
  scoring option likelihoods. This keeps exactly two heads while supporting the
  three-term loss, and avoids a separate, throwaway classification head.
- **Consciousness & memory injected at reserved token slots.** Adding their
  projections onto `[CONSC]`/`[MEM]` embeddings keeps the sequence all-integer
  and the label/shift bookkeeping simple. A richer fusion is future work.
- **Deterministic mocks.** The mock parser/mapper/memory are seeded/hash-based so
  tests and runs are reproducible.

## NSM prime inventory — caveats
The inventory in `nsm_primes.py` follows the canonical ~65-prime table
(Goddard & Wierzbicka 2014) to the best of our knowledge. Version-dependent
details are flagged with `TODO(canonical-list)` in that module — notably whether
`DON'T WANT` is a distinct prime, the possession prime's exact form
(`(IS) MINE` vs. `HAVE`), and the full set of allolexes. These should be
verified against a primary source before any linguistic claim is made; we did
**not** fabricate entries to fill gaps.
