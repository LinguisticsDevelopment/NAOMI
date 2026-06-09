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
