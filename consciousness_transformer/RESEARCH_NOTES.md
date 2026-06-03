# Research Notes

Separates the **research** (open, unsolved) from the **engineering** (built,
working) in the NSM Consciousness Transformer, and records the design decisions.

## The research-vs-engineering boundary

**Engineered and working** — the stateful loop and its scaffolding:
- The state-transition step, the three heads (transition / action / response),
  the `Mind` unroll, read/write `WorkingMemory`, the loss combination, metrics,
  config, tokenizer, the curriculum generator, the bAbI loader, and the NSM
  prime constants. The loop demonstrably learns to absorb facts, retrieve the
  relevant one (including recency), and answer (~95% held-out on the curriculum).

**Mocked / placeholder / stubbed** — the actual research lives here:
- What the **consciousness state means** and its real objective (currently
  abstract + a placeholder consistency loss).
- **Semantic composition** onto NSM primes (`MockSemanticMapper`).
- A **consistent parser** (the rule parser is experimental; wrapped optionally).
- **Long-term memory** (only per-episode working memory exists).
- **Textbook ingestion** (`TextbookSource` is a stub — the north star).

## Open problems

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

### 4. Memory: long-term persistence and pruning
Memory is **per-episode working memory** only — it resets each episode and never
forgets within one. Open: episodic/long-term memory that persists across
episodes and a turn, plus a **pruning/forgetting** policy (what to keep, merge,
discard). The current write is a soft slot at a fixed index; a real system needs
content-addressed allocation.

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
