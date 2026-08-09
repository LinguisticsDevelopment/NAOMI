# The mind membrane — variables in/out and the interaction contract

Design doc for the perception↔thinking interface (2026-08-09, post-M51).
Decision recorded here: **Tracks A and B are built side by side behind ONE
shared membrane contract** — A (distinct resolution modules) for
functionality, B (one shared collapse mechanism) to see whether A's methods
fall out of a single learned operation. The contract below is what makes
that a fair fight: identical inputs, identical gates, drop-in swap.

## 0. What the mind loop ACTUALLY is today (measured from clause_reactor.py)

Per clause step t, batchwise:

```
mem_read = query(memory, e_t, r_t)                    # read BEFORE write, keyed by current clause
state    = GRU([e_t, r_t, v_t, p_t, c_t, mem_read], state)
gate     = σ(write_gate(state)) · statement_mask
owr      = σ(overwrite_gate(state)) · gate            # replace vs accumulate (vote)
neg      = σ(decide_truth(state, v_t)) · statement    # learned refutation (subtract)
memory   = write(memory, e_t, r_t, v_t, gate − neg, overwrite=owr)
respond_t, response_vec_t = heads(state, mem_read)
answer   = cosine(Σ softmax(respond)·response_vec, option_vecs)
```

Two corrections to the intuitive picture:

1. **Memory writes are LOCAL and gated, not global.** Each step touches one
   (entity, relation) address in the order-3 entity⊗relation⊗value store.
   "Updating all the memory states" is not what happens — and that's a
   feature (transparency: the store is offline-queryable) but also a limit.
2. **There is a feedback loop, but it's controller↔memory only.** The GRU
   state is recurrent and the memory read feeds it — that IS attention
   (content-based addressing), the transformer analogy is correct. But
   (a) the read address is dictated by the CURRENT clause's (e, r) — the
   controller cannot compose its own queries (no free recall), and
   (b) **nothing flows back to perception.** Perception is strictly
   upstream today. The "blur" is precisely adding that missing edge.

## 1. Variables IN (per clause step) — v0 today → v1 target

| variable | today (v0) | v1 target |
|---|---|---|
| entity e | one atomic handle; fresh atom per name; pronouns = unresolvable distinct atoms | **candidate set** [(entity_atom, prior)] + a FEATURE vector (USVS person/number axes) so matching is possible; singletons for resolved names |
| relation r | one role handle (primary role only — M50's extra args are DROPPED at batch build) | **full arg set** [(role, value), ...] per clause |
| value v | one content handle (sense = MFS or gold via meta) | **candidate set** [(sense_handle, MFS-rank prior)] for homographs; singleton otherwise |
| predicate p | handle | unchanged + flags (PASSIVE, ...) |
| coord c | NOT marker etc. | unchanged |
| structure | one parse, pre-chosen by structural score | **top-K hypotheses** [(clause-stream, parser score)] when the structural margin is low; K=1 when confident |
| question | is_question step flag | + **queried role** (where→PLACE, who→AGENT/SUBJECT) |
| options | K option meaning-vectors | unchanged |

Design law (the membrane rule): **perception never guesses.** Anything it
cannot resolve deterministically crosses as a candidate set with structural
priors. Perception stays untrained and honest about uncertainty; every
context-dependent choice belongs to the trained side.

## 2. Variables OUT — v0 today → v1 target

| variable | today (v0) | v1 target |
|---|---|---|
| memory ops | write/overwrite/negate gates (local) | unchanged mechanism, applied to the RESOLVED binding |
| answer | response vector → cosine vs options | unchanged |
| respond timing | softmax over steps | unchanged |
| **resolutions** | — (nothing returns to perception) | per candidate set: chosen binding + margin. Written back into the discourse record (pronoun→referent, homograph→sense, lattice→parse). This is the new edge. |
| **confidence** | — | collapse margin per decision; low margin = "this sentence was hard" (garden-path signature; later: triggers reanalysis) |

## 3. The interaction loop (v1)

1. Parser/grounder emit clause-with-candidates (+ priors, + features).
2. Mind reads memory — v1 allows LEARNED addressing: query composed from
   candidate features + controller state, not only the literal (e, r).
3. **Collapse**: candidates → binding, before the write.
   - **Track A (functionality)**: distinct heads — coref head (entity
     candidates vs STM readout), sense head (the M34 chooser with its
     context source swapped from bag-of-words to memory readout), parse
     re-rank head. Each independently gateable and measurable.
   - **Track B (emergence)**: ONE shared scorer
     `score(candidate_vec, mem_read, state) → logit`, softmaxed within
     each candidate set, applied uniformly to entities/senses/hypotheses.
     Same I/O as A. The question B answers: do A's specialized behaviors
     fall out of one mechanism given the same curriculum?
4. Resolved clause is written (existing gated machinery, unchanged).
5. Resolution + margin returned across the membrane to the discourse record.

## 4. Shared gates (identical for A and B — this is the experiment)

- Pronoun curriculum: episodes whose answers REQUIRE she/he/it→antecedent
  binding (no recency shortcut in the data design: nearest-entity baseline
  must sit near floor).
- Ambiguity curriculum: the M32/M40 protocol, chooser-style, with memory
  context replacing bag context.
- Garden-path battery: low-margin parses whose correct reading requires a
  memory fact; both tracks scored on final answer + chosen parse.
- Multi-arg episodes (give/take: AGENT/OBJECT/RECIPIENT/PLACE) — the M52
  robustness prerequisite, trained BEFORE any resolver work so memory
  content is rich enough to resolve against.
- Regression: existing curriculum L1–6 unchanged-or-better.

Scoring the fight: functionality first (does it work), then parameter
count, then transfer (leave-one-family/leave-one-shape-out). If B matches A
within noise at comparable size, B wins on architectural grounds (fewer
seams). If B lags on a specific capability, that capability is evidence of
a genuinely distinct mechanism — which is a research result, not a failure.

## 5. Sequencing

- M52: multi-arg curriculum + reactor consumes full arg sets (v1 rows
  "relation"/"structure" K=1). No resolver yet.
- M53: membrane types (candidate sets, features, priors) + pronoun
  curriculum + Track A coref head + Track B scorer on the SAME data.
- M54: sense collapse joins (chooser context swap) — A vs B continues.
- M55: parse-hypothesis collapse + garden-path battery; reanalysis loop
  explicitly deferred until collapse-margin data says it's needed.

Out of scope for v1 (recorded so they're chosen, not forgotten): soft/
superposed writes (Model C), reanalysis, LTM candidates for pronouns
(STM-only first), learned addressing beyond the resolver's needs.
