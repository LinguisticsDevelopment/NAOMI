# Decoder v2 — retrieval-augmented realizer over dialectic memory

Status: DESIGN (lead-approved direction 2026-09-06). Supersedes the
`function_vocab`-based realizer in `src/nsm_ct/decoder_trained.py`, which the
2026-09-06 audit found broken: `build_function_vocab` mined 2047 *content*
words from uncovered surface tokens, so the decoder memorized content (train
token-acc 0.79 vs dev 0.16) and confabulated (a no-confab hole). Root cause:
the grounded structure covers only ~24% of surface tokens (a predicate-argument
skeleton), so exact-surface reconstruction by copy is impossible and the vocab
was papering the 76% gap by memorization.

## Thesis
Realization routes through **memory**, not baked decoder weights. Each grounded
node remembers *how meanings like it have been said before* (the surface block
it realized as, in context). To realize a new structure, retrieve those
per-node exemplars and stitch them with learned connectives. "Helper words"
fall out of the retrieved/stored context — never a memorized content vocab.
This is the memory-read invariant made literal for the output half.

## Memory unit (derived from EXISTING gold — no new annotation)
For every training record + its gold top tree, for each node `n` store a
`RealizationExemplar`:
- `sense`: node grounding (sense id + USVS vector) — the retrieval key.
- `relation`: node's role label (SUBJECT / OBJECT / PREDICATE / ...).
- `span`: the surface token-block `n` realized as (see derivation below).
- `parent_relation`, and `left_relation`/`right_relation` of its neighbors —
  connective context.
- `source_text`: the full original sentence (provenance; also lets the
  comprehension side read register/connotation later).

### Span derivation (deterministic, from tokens + node token_index)
Nodes carry a single `token_index`. Walk the sentence left→right:
- each node "owns" its own token index (its minimal content span);
- the run of **uncovered** tokens between node A (index i) and the next node B
  (index j) is a **connective** unit keyed on `(A.relation, B.relation)`;
- leading/trailing uncovered runs are connectives keyed on `(BOS, first.rel)` /
  `(last.rel, EOS)`.
This assigns **every** surface token to either a node-span or a relation-keyed
connective → target coverage ~100% (vs 24% copy-only). VALIDATION STEP 1 below
measures the real coverage before any model is built.

## Realization (inference)
Given a committed structure's nodes in surface order:
1. For each node, retrieve the k nearest stored exemplars by
   `cosine(sense_vector) + relation match` → candidate spans.
2. A small learned selector picks/blends among the k candidate spans for that
   node (conditioned on the node + its neighbors).
3. Between adjacent nodes, emit the learned connective for their
   `(rel_a, rel_b)` key (retrieved/aggregated over memory, not free-generated).
4. Concatenate in node order.

## No-confabulation gate (hard, ablation-tested)
- Content tokens emitted MUST come from a retrieved span of a node **present in
  the current structure**. Never import a neighbor's content that isn't a node.
- Connectives are a closed, learned set keyed on relation pairs — not open
  content generation.
- Sever the structure (no nodes) → nothing retrievable → `realize()` returns
  `[]`. Same ablation gate as v1; must stay green.
- Retrieval at TEST time excludes the test sentence itself (no parroting).

## Training (self-supervised, no new data)
Source sentence is the target. Learn the per-node span selector + the connective
model to reconstruct the surface from (structure + retrieved memory). Loss on
the SELECTION among retrieved candidates + connective choice — NOT exact-position
open-vocab CE (that was the "too binary" secondary issue). Round-trip test:
text → encoder.beam_decode → structure → realize → text.

## Metrics
- token-F1 + exact_match (as now), reported raw and lemma.
- **coverage**: % of gold surface tokens attributable to a node-span or a
  connective (the viability number).
- **confab rate**: % of emitted content tokens NOT traceable to a present
  node's retrieved span (must be ~0).
- held-out generalization: train vs dev token-acc gap (v1's 0.63 gap is the
  bug-signature to beat).

## Build order
1. **Span-alignment data layer + coverage report** (NO training) — derive
   exemplars from `encoder_gold_v2.jsonl`, report surface coverage %,
   connective-key distribution, and 10 worked examples. GATE: if coverage isn't
   near-total, revisit derivation before building the model.
2. Retrieval index (sense+relation) + span selector + connective model.
3. Training loop + round-trip eval + confab-rate + no-confab ablation.
4. Fold into the Colab run-2 notebook (replacing the v1 decoder).
