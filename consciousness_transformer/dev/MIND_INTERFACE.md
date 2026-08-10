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

---

# v2 — THE LOCKED DESIGN (draft for sign-off, 2026-08-09)

Everything below supersedes nothing above — v1 (candidate sets + resolver)
is step 2 of this loop. This section adds the full state schema, the dials,
and the ambiguity-safety contract, in plain language. No proceeding path is
recommended here; the design gets signed off first.

## Glossary (each term defined once, plainly)

- **USVS** — the meaning space. Every word sense is a point whose
  dimensions have names. This is the metalanguage the whole mind runs on:
  it trains on structured thoughts, not on statistics of word co-occurrence.
- **Handle** — a meaning's vector address in that space.
- **Clause** — one unit of thought: who did what, to what, where.
- **STM (short-term memory)** — what the mind currently holds about the
  ongoing situation.
- **Episodic LTM (long-term memory)** — remembered past episodes/stories;
  the stuff opinions are informed by.
- **Truth memory** — what the mind believes about the world in general.
- **Controller state** — the small learned network's running
  train-of-thought vector, carried step to step.
- **Workspace** — the single slot holding the thought currently under
  consideration; filled from outside (heard sentence) or from inside
  (recall, hypothesis, rewrite of the previous thought).
- **Candidate set** — when perception isn't sure (pronoun, homograph,
  hard parse), it hands over ALL the options plus how plausible each looks
  structurally. Perception never guesses.
- **Collapse / resolver** — the learned step that picks one option, using
  memory as context.
- **Margin** — how decisively it picked. Low margin = "that was hard."
- **Emit gate** — the decision of whether the current thought stays
  internal (keep thinking) or becomes actual output (speak).
- **Consolidation** — copying what mattered from STM into LTM, and — with
  much more caution — promoting repeatedly-confirmed facts into truth
  memory.
- **Membrane** — the boundary between deterministic perception and learned
  thinking. Everything that crosses it is listed in the v1 tables.

## The dials (explicit, named, tunable scalars — never buried in weights)

| dial | what it controls | mechanism it feeds |
|---|---|---|
| **trust_ltm** | how easily an episode's content is kept long-term | consolidation threshold, STM → episodic LTM |
| **trust_truth** | how much independent corroboration a fact needs before it becomes "believed about the world" | promotion threshold, STM/LTM → truth memory (strictly higher bar than trust_ltm) |
| **caution** | minimum collapse margin to hard-bind; below it the ambiguity is HELD, not guessed | resolver binding threshold — this is the ambiguity-safety dial |
| **yap_emit** | how readily inner content is spoken at all | emit-gate bias |
| **yap_continue** | after emitting, how often the loop re-enters its own output to extend/elaborate it | workspace re-entry bias |
| **patience** | thinking budget: how many silent inner loops per input before forced emit-or-move-on | loop budget (v1: fixed hyperparameter; later: a trained cost-of-thinking trade-off) |

Design law for dials: they are runtime inputs to gates (like temperature),
so behavior is tunable, per-configuration, and testable without retraining.
First implementation treats them as fixed hyperparameters; conditioning the
policy on them (so one model serves many dial settings) is a later,
separately-gated step.

## Ambiguity-by-design: what "unclear input is OK" means, stage by stage

Uncertainty is never destroyed and never forced into a guess — it is
represented, at every stage, in one of four sanctioned forms: (1) multiple
candidates with priors, (2) an OPEN binding in the discourse record,
(3) a low margin on a decision, (4) an abstention answer.

| unclear thing | designed behavior |
|---|---|
| unparseable fragment | crosses as a minimal "fragment" workspace item (grounded words, no structure claim, floor confidence); never a fake parse, never a crash |
| low-margin parse | top-K hypotheses cross with scores (v1 contract); resolver picks or holds |
| unknown word | USVS fallback chain (explication → placeholder atom that KEEPS its surface form so it can be bound later); "learn the new word" is future work but the placeholder is the designed slot for it |
| unresolvable pronoun | binding stays OPEN below the caution threshold; answers about it use the existing abstain atoms (idk / MAYBE — already in the codebase, already trained against) |
| contradiction with memory | never silently overwritten: the existing vote/overwrite/negate gates + the trust dials decide; low trust keeps both votes alive |
| garden path (all parses low-margin) | hold ambiguous; reanalysis (re-read triggered by low margin) is the named, deferred mechanism |

## Invariants (the design laws — locked)

1. Knowledge lives in inspectable structures (space, graphs, memory
   tensors); **weights hold policy only**.
2. **Perception is deterministic and never guesses** — uncertainty crosses
   the membrane as candidate sets with structural priors.
3. **The mind may abstain**; uncertainty is represented, never silently
   dropped (the four sanctioned forms above).
4. **Every memory write is gated, local, and auditable** — what was
   written, when, from what evidence, at what trust setting.
5. **One resolver contract**; Track A (distinct heads) and Track B (one
   shared scorer) are swappable implementations behind it.
6. **Dials are explicit named scalars**, chosen — never emergent settings
   discovered after the fact.
7. **Inner and outer content are the same vector**; emission is a gate,
   not a separate pathway. (Thought = speech withheld.)
8. **Compute per input is budgeted** (patience); halting is a designed
   trade-off, not an accident of the architecture.
9. The whole loop operates on the metalanguage: structured clauses of
   grounded meanings. Ambiguity is explicit at the boundary — the
   robustness bet of this project over token-statistical systems, now a
   stated invariant rather than an aspiration.

## What exists / what's new (so the build size is honest)

| piece | status |
|---|---|
| STM (entity⊗relation⊗value, gated writes) | exists, per-episode |
| controller recurrence | exists (GRU), now explicit in schema |
| abstain atoms (idk/MAYBE) | exist, trained against |
| vote/overwrite/negate write policy | exists (the trust dials will parameterize its thresholds) |
| candidate sets + resolver (v1, M52–55) | designed, not built |
| episodic LTM + consolidation | new |
| truth-memory promotion | new (seed exists: L5 corroborate/contradict) |
| workspace + emit gate + re-entry | new (respond-timing head is the degenerate ancestor) |
| dials as runtime inputs | new (thresholds exist implicitly today) |

---

# v2 addendum — the entity-instance subsystem (user-specified, 2026-08-10)

Standing defect it fixes: today identity IS the name string (entity atoms
are minted deterministically from the name — var:mary is the same vector
everywhere), so two Marys collapse into one memory entity and "another
person also named mary" is UNREPRESENTABLE. Names and referents are fused;
M56b proved feature-knowledge must not hide in strings or weights — this
is the same rule applied to identity.

Design (DRT discourse-referents shape):
1. First mention MINTS a fresh instance atom (arbitrary vector — correct
   for individuals; it's a variable, not a meaning). mary₁ ≠ mary₂.
2. All properties become ATTRIBUTE FACTS in the entity⊗relation⊗value
   store: name(e₁,"mary"), kind(e₁,person), gender(e₁,F@prior),
   occupation(e₁,doctor)... — inspectable, source-tagged, updatable by
   discourse (name-based gender priors get overridden by observed
   bindings; the caution dial holds low-margin cases open).
3. EVERY referring expression = candidate generation over instances by
   attribute match, resolved by the EXISTING membrane resolver:
   - "mary" → instances with name=mary (two Marys = a real candidate set)
   - "she"  → person instances with compatible features
   - "the doctor" → instances with kind=doctor  ← definite descriptions,
     the dominant referring device in real prose, unified under the same
     collapse machinery. Pronouns stop being special; they're the
     referring expression with the least attribute evidence.
4. Cross-episode identity ("is this mary THAT mary?") becomes an explicit
   consolidation-time decision (LTM linking), not a string collision.

Build timing: with the corpus campaign (real prose has multiple Johns and
definite descriptions immediately) / as the front half of v2 LTM work.
Registry + minting + attribute writes are deterministic membrane-side
work; only the collapse policy (already built, already held-out-validated)
is learned.
