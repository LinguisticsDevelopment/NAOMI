# MIND — a meaning-space substrate with a learned consciousness riding on it

> **Project outline.** This document is the agreed architecture and build roadmap for the next stage
> of NAOMI's consciousness transformer: one system you can **talk to**, that **remembers**,
> **reasons** (derive-or-abstain), and that you can **teach**. It is the design contract the build in
> `src/nsm_ct/mind/` follows. For the open research log and measured results behind it, see
> `RESEARCH_NOTES.md`.

## The one-paragraph thesis
Everything the system holds and manipulates is a **meaning object** in a **meaning vector space** —
never tokens. A **learned "consciousness"** (a state machine over that space, *not* a next-token
transformer) governs how a stimulus is perceived, recalled, reasoned over, consolidated, and answered.
The substrate (the meaning graph + its operators) is deterministic and inspectable; the consciousness
is the only learned part, and it learns **how to think**, never **what is true**.

## Two cleanly separated kinds of learning (the spine)
- **Weights hold zero information.** Gradient descent touches *only* the cognitive policy — how to
  perceive, recall, infer, react, respond. **No facts, rules, or taxonomy are ever baked into the
  weights.** Freeze the weights and wipe the graph and the system knows *how to think* and *nothing
  at all*.
- **All content lives in the LTM graph.** Facts, rules, and taxonomy are graph data, **written quickly
  during interaction** and **read back via retrieval**. "Teach it once, it knows forever" is a **graph
  write, not a weight update** — instant, non-destructive, inspectable. Knowledge never migrates into
  weights; consolidation is graph→graph (STM→LTM), never graph→weights.

Consequences: the system **does not learn language** (NL I/O is peripheral plumbing); **teaching has no
shallow/deep distinction** (always a graph write); **variable binding is not a training problem** (a
rule is graph data with a variable slot, bound by deterministic unification — only *which rule to
retrieve when* is learned).

## The two-part spine
```
   LEARNED CONSCIOUSNESS  — a learned state machine
   ops: PERCEIVE · RECALL · INFER · CONSOLIDATE · SUPERSEDE/FORGET · RESPOND · HALT
   reads a graph embedding, emits (operation, operands); a deterministic executor applies it
   carries a persistent STATE = what to look at + how to react (holds no knowledge)
                                   │ all ops read/write meaning objects in
                                   ▼
   MEANING-SPACE SUBSTRATE (the medium)
   meaning objects (predicate + role-bound fillers, operators, referents) embedded in a
   continuous meaning vector space; STM (working set) + LTM (durable store) both hold these
```
- **Storage vs. read-view are separate layers.** The canonical store is a **symbolic typed meaning
  graph** (lossless, decodable). The controller never holds meaning as a differentiable blob — it
  reads a **learned graph embedding** as its state view. Differentiability is needed only for the
  *read* and the *policy*, never for the *representation of meaning*.
- **The controller governs; it does not transform meaning.** Meaning is acted upon by the substrate's
  operators under the controller's direction. The reasoning trace *is* the sequence of
  `(operation, operands)` chosen — inspectable by construction, and **faithful** (each derived object
  carries provenance).

## The substrate already exists
`src/nsm_ct/meaning_graph.py` is a built, gate-tested dual-coded graph store: each `GraphNode` carries
a lossy vector `handle` **and** a lossless serialized `structure` ("the vector never has to be
invertible because the exact truth is the stored structure"). `NodeKind {CONCEPT, REFERENT, CLAUSE,
OPERATOR}`; typed edges `{SLOT, ABOUT, COREF, OPERATES_ON, SUPERSEDES, DEFINES}`; operator-nodes
`{NOT, MAYBE, AND, OR, IF}` via `apply_operator`/`read_operator`. The committed vocabulary: ~62 NSM
**primes** (`nsm_primes.py`), ~17 typed **relations** (`structure.py`), 5 **meaning-operators**. TPR
(`tpr.py`) is the encoding; STM read-resolution (`clause_psyche_graph.py`) handles
recency/negation/disjunction.

## The four goals are behaviors of one system
- **REASON** — transform meaning-state to derive an answer or **abstain**, with a recoverable trace.
- **REMEMBER** — meaning objects persisted at two horizons (working / durable) with consolidation.
- **TEACH** — a lesson is a meaning object **written into the LTM graph**, read back via retrieval; a
  graph write, never a weight update.
- **TALK** — a learned, **owned** encode/decode between text and meaning-space (no LLM at the edges).

## Two loops over one substrate
- **Conscious loop** (stimulus-driven, reactive): encode → recall → reason → decode response → write
  back to STM.
- **Subconscious loop** (background): consolidate STM→LTM, forget/supersede, **infer offline**, and
  **self-train/replay**. Consolidation and learning are the *same* loop.

## Training
A **graded LLM judge** provides dense shaping reward over a **symbolic correctness floor**
(`reasoning_oracle.py`); the judge is distilled into an owned reward model so the end-state is LLM-free.
**Abstain is supervised by the floor, not the judge** (or the system learns to bluff). Control style =
**soft selection annealing to discrete**: train with a temperature high→one-hot so gradients flow early
and the runtime trace is a clean discrete op-by-op sequence.

## Reasoning curriculum (grounded in the oracle's gold rules)
Property inheritance over is-a; is-a transitivity; modus ponens / variable-binding conditionals;
defeasible/recency resolution; negation & disjunction (→ `MAYBE`); multi-hop; derive-or-abstain. New
reasoning kinds are added by adding rules to LTM, not by retraining.

## Build roadmap (gate-tested milestones)
New code lands in **`src/nsm_ct/mind/`**, importing the reused primitives — *not* rewriting them. Every
milestone ends in a measurable gate; the 321-test suite stays green throughout.

| Milestone | Goal | Gate |
| --- | --- | --- |
| **M0** | This doc + freeze the meaning-object schema (`mind/schema.py`) + scaffold `mind/` | doc committed; schema imports clean; meaning-object round-trips via collapse/expand |
| **M1** | Knowledge layer: LTM as one durable, **disk-persistent** meaning graph; taxonomy + **variable-bearing rules in the graph**; lift unification into a live executor | `derive()` over graph-resident rules reproduces the oracle's L9/L10 answers; graph persists + reloads identically |
| **M2** | The deterministic executor (the VM the controller governs) — `INFER` = focus-chaining (§0n) + unification | hand-written gold op-traces solve L8–L11 and abstain on L11 |
| **M3** | Read-encoder + learned controller (retargets the proven `clause_psyche` focus-chaining/PonderNet machinery to emit ops); soft-anneal imitation of gold traces | held-out op-trace match + depth-3 multi-hop matches the §0n ~0.97 ceiling |
| **M4** | The conscious + subconscious loops over one STM/LTM | end-to-end episode + **teach-by-graph-write, retrieve-later with no weight update** |
| **M5** | Training stack: graded LLM judge over the symbolic floor, distilled reward model; abstain supervised by floor | beats floor-only without reward-hacking; abstain calibrated |
| **M6** | Native language membrane (encoder reuses `quantum_parser`+`meaning.py`; new decoder), parallelizable from M1 | cycle-consistency + faithful verbalized trace |
| **M7** | Textbook north star (`TextbookSource`) | chapter → episodes; homework answered |

Dependency order is forced **M0→M1→M2→M3→M4→M5**, with **M6 parallelizable from M1** (needs only the
frozen schema).

## What this explicitly is NOT
- **Not a pure transformer** — a learned consciousness over meaning-space.
- **Not token-substrate** — nothing "really" runs on tokens underneath.
- **Not three legacy stacks wired together** — one substrate, one controller family, two loops; existing
  builds are reused as primitives, not stapled.
- **Not symbolic-truth-at-runtime** — symbolic reasoning is the teacher/validator; the learned
  consciousness is the runtime reasoner.
